"""
run_anisotropy.py
=================
Reproduces Mattiuz thesis Fig 16: directional thermal conductivity vs grain
aspect ratio for an anisotropic Laguerre polycrystal with grain-boundary-clipped
spherical pores.

Methodology (from old_files/Test porosità/aniso_delta_calc.py)
--------------------------------------------------------------
- Laguerre tessellation seeded with RSA spheres of radius lagR.
- Volume-preserving anisotropy applied to the polycrystal:
      aspect_ratio = [1.0, gamma, 1.0/gamma]
  so the determinant is 1 and the RVE volume is conserved.
- Spherical pores (Boolean, overlap allowed) of radius inclR with
  target volume fraction inclPhi laid on phase 2.
- A grain-boundary layer of thickness delta is added as phase 3.
- Final overlay maps {pores -> grains, boundary -> grains}, which clips
  the pores to the GB band only.
- Resulting structure: phase 0 = solid (matrix + boundary), phase 2 = pores.

Sweep: 20 gamma values in [1, 0.1] x 2 porosity levels (phi=0.1, 0.2)
       = 40 cases total.

Outputs:
  Results_Anisotropy/anisotropy.csv
      columns: AspectRatio, TargetPhi, RealP, Kxx, Kyy, Kzz, Kmean
  Results_Anisotropy/AR_<gamma>_Phi_<phi>/    per-case directories
      structure.vtk, Coeffs.txt, thermalCoeff_amitex.txt

Run
---
    cd ~/Merope
    source Env_Merope.sh
    export PYTHONPATH=$PYTHONPATH:./project_root
    python project_root/experiments/run_anisotropy.py            # full sweep
    python project_root/experiments/run_anisotropy.py --recover  # skip done
    python project_root/experiments/run_anisotropy.py --plot-only
"""
import sys
import argparse
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

# --- Configuration (matches old_files/Test porosità/aniso_delta_calc.py) ---
L_DIM = [10.0, 10.0, 10.0]
N_VOX = 100
SEED = 0

INCL_R = 0.3        # pore radius
LAG_R = 3.0         # Laguerre grain size
LAG_PHI = 1.0       # fill RVE with grains before clipping
DELTA = 1.0         # GB layer thickness

K_THERMAL = [1.0, 1.0, 1e-3]   # phase 0 = matrix, phase 1 = matrix (temp), phase 2 = pore

AR_VALUES = np.linspace(1.0, 0.1, 20)
PHI_VALUES = [0.10, 0.20]

OUTPUT_DIR = Path("Results_Anisotropy")
N_CPUS = 8


def build_structure(gamma: float, phi: float):
    """Inline construction (geometry.py helpers don't expose AR + clipped-pore overlay)."""
    incl_phase = 2
    delta_phase = 3
    grains_phase = 0

    # 1. Spherical pore inclusions
    sph_pores = merope.SphereInclusions_3D()
    sph_pores.setLength(L_DIM)
    sph_pores.fromHisto(
        SEED, sac_de_billes.TypeAlgo.BOOL, 0.0,
        [[INCL_R, phi]], [incl_phase],
    )
    multi_pores = merope.MultiInclusions_3D()
    multi_pores.setInclusions(sph_pores)

    # 2. Laguerre seeds (RSA), then anisotropic tessellation
    sph_grains = merope.SphereInclusions_3D()
    sph_grains.setLength(L_DIM)
    sph_grains.fromHisto(
        SEED, sac_de_billes.TypeAlgo.RSA, 0.0,
        [[LAG_R, LAG_PHI]], [1],
    )
    poly = merope.LaguerreTess_3D(L_DIM, sph_grains.getSpheres())
    poly.setAspRatio([1.0, float(gamma), 1.0 / float(gamma)])

    multi_grains = merope.MultiInclusions_3D()
    multi_grains.setInclusions(poly)

    ids = multi_grains.getAllIdentifiers()
    multi_grains.addLayer(ids, delta_phase, DELTA)
    multi_grains.changePhase(ids, [1 for _ in ids])

    # 3. Single overlay: clip pores to GB band; both pore (2) and boundary (3) -> grains (0)
    mapping = {incl_phase: grains_phase, delta_phase: grains_phase}
    return merope.Structure_3D(multi_pores, multi_grains, mapping)


def run_case(gamma: float, phi: float, no_solver: bool = False) -> dict:
    pm = ProjectManager()
    solver = ThermalSolver(n_cpus=N_CPUS)

    case_dir = OUTPUT_DIR / f"AR_{gamma:.3f}_Phi_{phi:.2f}"
    case_dir.mkdir(parents=True, exist_ok=True)

    structure = build_structure(gamma, phi)

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
    print(f"  [DONE] gamma={gamma:.3f}, phi={phi:.2f} | RealP={real_p:.4f} | "
          f"Kxx={res['Kxx']:.4f} Kyy={res['Kyy']:.4f} Kzz={res['Kzz']:.4f} Kmean={res['Kmean']:.4f}")

    return {
        "AspectRatio": float(gamma),
        "TargetPhi": float(phi),
        "RealP": real_p,
        **res,
    }


def run_sweep(recover: bool = False, no_solver: bool = False) -> pd.DataFrame:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / "anisotropy.csv"

    existing = None
    if recover and csv_path.exists():
        existing = pd.read_csv(csv_path)
        print(f"[RECOVER] Loaded {len(existing)} existing rows from {csv_path}")

    if existing is not None:
        ax_cols = ["Kxx", "Kyy", "Kzz"]
        existing = existing[existing[ax_cols].min(axis=1) > 0.01].reset_index(drop=True)
        print(f"[RECOVER] Kept {len(existing)} rows where all axes converged (skipping full and single-axis failures)")
    rows = [] if existing is None else existing.to_dict("records")

    def is_done(gamma, phi):
        if existing is None:
            return False
        match = existing[
            (np.abs(existing["AspectRatio"] - gamma) < 1e-6) &
            (np.abs(existing["TargetPhi"] - phi) < 1e-6)
        ]
        return len(match) > 0

    todo = [(g, p) for p in PHI_VALUES for g in AR_VALUES if not is_done(g, p)]
    print(f"=== Anisotropy sweep: {len(todo)} cases to run "
          f"({len(AR_VALUES)} AR x {len(PHI_VALUES)} phi - already done) ===\n")

    for gamma, phi in todo:
        print(f"--- gamma={gamma:.3f}, phi={phi:.2f} ---")
        rows.append(run_case(gamma, phi, no_solver=no_solver))
        df = pd.DataFrame(rows).sort_values(by=["TargetPhi", "AspectRatio"]).reset_index(drop=True)
        df.to_csv(csv_path, index=False)

    df = pd.DataFrame(rows).sort_values(by=["TargetPhi", "AspectRatio"]).reset_index(drop=True)
    df.to_csv(csv_path, index=False)
    print(f"\n[SAVE] {csv_path} ({len(df)} rows)")
    return df


def plot_anisotropy(df: pd.DataFrame, output_dir: Path) -> None:
    """Thesis Fig 16 analogue: relative |K_mean - K_yy|/K_yy vs gamma."""
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = {0.10: "steelblue", 0.20: "darkorange"}

    for phi, group in df.groupby("TargetPhi"):
        g = group.sort_values("AspectRatio")
        rel = np.abs(g["Kmean"].values - g["Kyy"].values) / g["Kyy"].values
        ax.plot(g["AspectRatio"], 100.0 * rel, "o-",
                color=colors.get(phi, "black"), label=fr"$\phi$ = {phi:.2f}")

    ax.set_xlabel(r"Aspect ratio $\gamma$")
    ax.set_ylabel(r"$|K_\mathrm{mean} - K_{yy}| / K_{yy}$ (%)")
    ax.set_title("Directional anisotropy vs grain aspect ratio")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()

    img_path = output_dir / "anisotropy.png"
    fig.savefig(img_path, dpi=300)
    print(f"[PLOT] {img_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--recover", action="store_true", help="Skip cases already in CSV")
    ap.add_argument("--no-solver", action="store_true", help="Geometry only (debug)")
    ap.add_argument("--plot-only", action="store_true", help="Regenerate plot from existing CSV")
    args = ap.parse_args()

    if args.plot_only:
        df = pd.read_csv(OUTPUT_DIR / "anisotropy.csv")
    else:
        df = run_sweep(recover=args.recover, no_solver=args.no_solver)

    plot_anisotropy(df, OUTPUT_DIR)
