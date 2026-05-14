"""
run_grain_size_distribution.py
==============================
Reproduces Mattiuz thesis Figs 17-18: effect of grain size distribution on
K_eff for an interconnected porous polycrystal.

Methodology (from old_files/Test porosità/vol_distribution_IGB_calc.py)
----------------------------------------------------------------------
- 200 equal-radius (lagR=1) RSA seeds thrown in the RVE.
- A Gaussian weighting function on the *distance from the RVE centre* assigns
  desired volumes to each seed:
      weight(centre) = exp(-0.5 * (||centre - L/2|| - mean)^2 / sigma^2)
  with mean = L/2 = 5.
- merope.algo_fit_volumes_3D iteratively adjusts the Laguerre weights so the
  tessellation cells match the desired volumes.
- Standard delta=1 GB layer + Boolean spherical pores at phi=0.20 are added.
- Two cases:
    sigma = 0.5 -> polydisperse (peaked Gaussian, edges get tiny volumes)
    sigma = 3.0 -> monodisperse (broad Gaussian, near-equal volumes)

Outputs
-------
Results_GrainSizeDistribution/
    summary.csv                         sigma, real_p, Kxx, Kyy, Kzz, Kmean, K_loeb
    sigma_0p5/                          per-case directory
        structure.vtk, Coeffs.txt, thermalCoeff_amitex.txt
        grain_volumes.csv               desired volume per seed (% of total)
    sigma_3p0/                          (idem)
    volume_histograms.png               thesis Fig 17 analogue
    keff_comparison.png                 thesis Fig 18 analogue

Run
---
    cd ~/Merope
    source Env_Merope.sh
    export PYTHONPATH=$PYTHONPATH:./project_root
    python project_root/experiments/run_grain_size_distribution.py
    python project_root/experiments/run_grain_size_distribution.py --recover
    python project_root/experiments/run_grain_size_distribution.py --plot-only
"""
import sys
import argparse
from math import sqrt
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import merope
import sac_de_billes

from core.solver import ThermalSolver
from core.utils import ProjectManager

# --- Configuration (matches old_files/Test porosità/vol_distribution_IGB_calc.py) ---
L_DIM = [10.0, 10.0, 10.0]
TOTAL_VOLUME = float(np.prod(L_DIM))
N_VOX = 150
SEED = 0

INCL_R = 0.25
LAG_R = 1.0
NB_SPHERES = 200
DELTA = 1.0
INCL_PHI = 0.20
MEAN = 5.0          # = L/2

K_THERMAL = [1.0, 1.0, 1e-3]
ALPHA_LOEB = 1.37   # paper-canonical recalibrated value

SIGMA_VALUES = [0.5, 3.0]

OUTPUT_DIR = Path("Results_GrainSizeDistribution")
N_CPUS = 8


def weighting_function(center, mean: float, sigma: float) -> float:
    distance = sqrt(sum((c - 0.5 * dim) ** 2 for c, dim in zip(center, L_DIM)))
    return float(np.exp(-0.5 * (distance - mean) ** 2 / (sigma ** 2)))


def build_structure(sigma: float):
    """Return (structure, desired_volumes_pct) for a given sigma."""
    incl_phase = 2
    delta_phase = 3
    grains_phase = 0

    # 1. Spherical pore inclusions (Boolean)
    sph_pores = merope.SphereInclusions_3D()
    sph_pores.setLength(L_DIM)
    sph_pores.fromHisto(
        SEED, sac_de_billes.TypeAlgo.BOOL, 0.0,
        [[INCL_R, INCL_PHI]], [incl_phase],
    )
    multi_pores = merope.MultiInclusions_3D()
    multi_pores.setInclusions(sph_pores)

    # 2. Equal-radius RSA seeds, then volume-fitted Laguerre tessellation
    tab_radii = [LAG_R] * NB_SPHERES
    tab_phases = [1] * NB_SPHERES
    seeds = sac_de_billes.throwSpheres_3D(
        sac_de_billes.TypeAlgo.RSA,
        sac_de_billes.NameShape.Tore,
        L_DIM, SEED, tab_radii, tab_phases, 0.0,
    )

    desired = [weighting_function(s.center, MEAN, sigma) for s in seeds]
    total = sum(desired)
    desired_volumes = [w / total * TOTAL_VOLUME for w in desired]

    algo = merope.algo_fit_volumes_3D(L_DIM, seeds, desired_volumes)
    algo.proceed(1e-6 * TOTAL_VOLUME, 3000, False)
    fitted_seeds = algo.getCenterTessels()

    poly = merope.LaguerreTess_3D(L_DIM, fitted_seeds)
    multi_grains = merope.MultiInclusions_3D()
    multi_grains.setInclusions(poly)

    ids = multi_grains.getAllIdentifiers()
    multi_grains.addLayer(ids, delta_phase, DELTA)
    multi_grains.changePhase(ids, [1 for _ in ids])

    mapping = {incl_phase: grains_phase, delta_phase: grains_phase}
    structure = merope.Structure_3D(multi_pores, multi_grains, mapping)

    desired_pct = [100.0 * v / TOTAL_VOLUME for v in desired_volumes]
    return structure, desired_pct


def run_case(sigma: float, no_solver: bool = False) -> dict:
    pm = ProjectManager()
    solver = ThermalSolver(n_cpus=N_CPUS)

    sigma_tag = f"sigma_{str(sigma).replace('.', 'p')}"
    case_dir = OUTPUT_DIR / sigma_tag
    case_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== sigma={sigma} ({sigma_tag}) ===")
    structure, desired_pct = build_structure(sigma)

    pd.DataFrame({"grain_volume_pct": desired_pct}).to_csv(
        case_dir / "grain_volumes.csv", index=False,
    )

    grid_params = merope.vox.create_grid_parameters_N_L_3D([N_VOX] * 3, L_DIM)
    grid = merope.vox.GridRepresentation_3D(structure, grid_params, merope.vox.VoxelRule.Average)

    fractions = {0: 0.0, 2: 0.0}
    try:
        analyzer = merope.vox.GridAnalyzer_3D()
        fractions = analyzer.compute_percentages(grid)
    except Exception as e:
        print(f"  [warn] analyzer skipped: {e}")

    grid.apply_homogRule(merope.HomogenizationRule.Voigt, K_THERMAL)
    printer = merope.vox.vtk_printer_3D()

    with pm.cd(str(case_dir)):
        printer.printVTK_segmented(grid, "structure.vtk", "Coeffs.txt", nameValue="MaterialId")
        if no_solver:
            res = {"Kxx": 0.0, "Kyy": 0.0, "Kzz": 0.0, "Kmean": 0.0}
        else:
            res = solver.solve()

    real_p = float(fractions.get(2, 0.0))
    k_loeb = 1.0 - ALPHA_LOEB * real_p

    print(f"  [DONE] sigma={sigma} | RealP={real_p:.4f} | "
          f"Kxx={res['Kxx']:.4f} Kyy={res['Kyy']:.4f} Kzz={res['Kzz']:.4f} "
          f"Kmean={res['Kmean']:.4f} | K_Loeb={k_loeb:.4f}")

    return {
        "sigma": float(sigma),
        "RealP": real_p,
        **res,
        "K_loeb": k_loeb,
    }


def run_sweep(recover: bool = False, no_solver: bool = False) -> pd.DataFrame:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / "summary.csv"

    existing = None
    if recover and csv_path.exists():
        existing = pd.read_csv(csv_path)
        existing = existing[existing["Kmean"] > 0.0].reset_index(drop=True)
        print(f"[RECOVER] Kept {len(existing)} rows with non-zero Kmean")

    rows = [] if existing is None else existing.to_dict("records")

    def is_done(sigma):
        if existing is None:
            return False
        return (np.abs(existing["sigma"] - sigma) < 1e-6).any()

    todo = [s for s in SIGMA_VALUES if not is_done(s)]
    print(f"=== Grain-size sweep: {len(todo)} cases to run ===")

    for sigma in todo:
        rows.append(run_case(sigma, no_solver=no_solver))
        df = pd.DataFrame(rows).sort_values("sigma").reset_index(drop=True)
        df.to_csv(csv_path, index=False)

    df = pd.DataFrame(rows).sort_values("sigma").reset_index(drop=True)
    df.to_csv(csv_path, index=False)
    print(f"\n[SAVE] {csv_path} ({len(df)} rows)")
    return df


def plot_volume_histograms(output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = {0.5: "steelblue", 3.0: "darkorange"}
    for sigma in SIGMA_VALUES:
        sigma_tag = f"sigma_{str(sigma).replace('.', 'p')}"
        path = output_dir / sigma_tag / "grain_volumes.csv"
        if not path.exists():
            print(f"  [skip] {path} not found")
            continue
        df = pd.read_csv(path)
        ax.hist(df["grain_volume_pct"], bins=20, alpha=0.6,
                color=colors.get(sigma, "grey"),
                edgecolor="black", label=fr"$\sigma$ = {sigma}")
    ax.set_xlabel("Grain volume (% of RVE)")
    ax.set_ylabel("Frequency")
    # ax.set_title("Grain volume distribution")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    out = output_dir / "volume_histograms.png"
    fig.savefig(out, dpi=300)
    print(f"[PLOT] {out}")


def plot_keff_comparison(df: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    x = np.arange(len(df))
    width = 0.35
    ax.bar(x - width / 2, df["Kmean"], width, label=r"$K_\mathrm{mean}$ (FFT)", color="steelblue")
    ax.bar(x + width / 2, df["K_loeb"], width, label=r"$K_\mathrm{Loeb}$ baseline", color="lightgray")
    ax.set_xticks(x)
    ax.set_xticklabels([fr"$\sigma$ = {s}" for s in df["sigma"]])
    ax.set_ylabel(r"$K_\mathrm{eff}$ (W/m·K)")
    # ax.set_title("K_eff vs grain-volume distribution width")
    ax.grid(True, linestyle="--", alpha=0.4, axis="y")
    ax.legend()
    fig.tight_layout()
    out = output_dir / "keff_comparison.png"
    fig.savefig(out, dpi=300)
    print(f"[PLOT] {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--recover", action="store_true", help="Skip cases already in summary.csv with non-zero Kmean")
    ap.add_argument("--no-solver", action="store_true", help="Geometry only (debug)")
    ap.add_argument("--plot-only", action="store_true", help="Regenerate plots from existing CSV")
    args = ap.parse_args()

    if args.plot_only:
        df = pd.read_csv(OUTPUT_DIR / "summary.csv")
    else:
        df = run_sweep(recover=args.recover, no_solver=args.no_solver)

    plot_volume_histograms(OUTPUT_DIR)
    plot_keff_comparison(df, OUTPUT_DIR)
