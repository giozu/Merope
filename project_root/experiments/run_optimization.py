"""
experiments/run_optimization.py
================================
Bayesian Optimization of microstructure parameters using Mérope + Amitex.

The script calibrates generation parameters so that the simulated structure
matches an experimental reference image (SEM / tomography slice) as closely
as possible, measured by a combined KS + Chi² morphological score.

Usage (from project_root/):
---------------------------
# Calibrate a DISTRIBUTED porosity (spherical pores in a homogeneous matrix)
python3 experiments/run_optimization.py --mode distributed

# Calibrate an INTERCONNECTED porosity (inter-granular pores with grain boundary)
python3 experiments/run_optimization.py --mode interconnected

# Change number of Bayesian optimization calls (default: 20)
python3 experiments/run_optimization.py --mode distributed --n-calls 30

Optional flags:
  --exp-image PATH   Override the default experimental image path.
  --n3d   INT        Voxel resolution (default: 120 for distributed, 150 for interconnected).
  --n-cpus INT       CPU cores for Amitex (default: 4).
  --seed  INT        Mérope / sac-de-billes RNG seed (default: 0).
  --n-slices INT     Number of 2D slices extracted from the 3D volume (default: 99).
  --run-amitex       Also run Amitex FFT after finding the best geometry and report K_eff.

Output (written to Results_Optimization_<mode>/):
  summary.txt                – best parameters, scores and (optionally) K_eff.
  area_distribution.png      – pore-area histogram: best slice vs experiment.
  convergence.png            – optimization score vs call number.
  best_slice.png             – best matching 2D slice from the optimized structure.

How to run:
    # Porosità DISTRIBUITA
    python3 experiments/run_optimization.py --mode distributed --n-calls 20
    python3 experiments/run_optimization.py --mode distributed --n-calls 20 --n3d 150

    # Porosità INTERCONNESSA
    python3 experiments/run_optimization.py --mode interconnected --n-calls 20

    # + Amitex alla fine per calcolare K_eff
    python3 experiments/run_optimization.py --mode distributed --n-calls 20 --run-amitex

"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Path setup: make project_root importable regardless of CWD
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args

import merope

from core.geometry import MicrostructureBuilder
from core.solver import ThermalSolver
from core.utils import ProjectManager
from core.statistics import evaluate_slices, plot_area_distribution


# ---------------------------------------------------------------------------
# Experimental image paths (defaults)
# ---------------------------------------------------------------------------
_EXP_BASE = Path("/home/giovanni/Merope/Optimization_3D_structure/EXP IMG")
_EXP_IMAGES = {
    "distributed":    str(_EXP_BASE / "exp_distrib_1.png"),
    "interconnected": str(_EXP_BASE / "exp_interconnect_1.png"),
}

# Physical porosity from experiments
_TARGET_POROSITY = {
    "distributed":    0.23,
    "interconnected": 0.21,
}


# ---------------------------------------------------------------------------
# Thermal conductivities (convention: index = phase id)
# distributed:    phase 0=matrix, phase 2=pores  → K = [K_m, K_m, K_p]
# interconnected: phase 0=matrix, phase 2=pores  → K = [K_m, K_m, K_p]
# ---------------------------------------------------------------------------
K_MATRIX = 1.0
K_GAS    = 1e-3
K_THERMAL = [K_MATRIX, K_MATRIX, K_GAS]


# ---------------------------------------------------------------------------
# Helper: generate structure + evaluate score
# ---------------------------------------------------------------------------

def _build_and_score_distributed(
    params: Dict[str, float],
    fixed: Dict[str, Any],
    exp_image: str,
) -> float:
    """Generate a distributed-porosity structure and return the average score.

    Parameters
    ----------
    params  : dict with ``mean_radius`` (log) and ``std_radius``.
    fixed   : dict with all fixed parameters.
    exp_image: path to the experimental reference image.

    Returns
    -------
    float – average morphological score ∈ [0, 1].
    """
    builder: MicrostructureBuilder = fixed["builder"]
    pm: ProjectManager = fixed["pm"]
    target_porosity: float = fixed["target_porosity"]
    num_radii: int = fixed["num_radii"]
    n_slices: int = fixed["n_slices"]
    grid_size: int = fixed["grid_size"]

    mean_r = float(params["mean_radius"])    # already in log-space
    std_r  = float(params["std_radius"])

    # Sample log-normal radii
    rng = np.random.default_rng(fixed["seed"])
    sampled = np.abs(rng.lognormal(mean_r, std_r, num_radii))
    sampled = sampled[(sampled >= 0.005) & (sampled <= 3.0)]
    if sampled.size == 0:
        return 0.0

    phi_each = target_porosity / sampled.size
    radii_phi_list = [[float(r), float(phi_each)] for r in sampled]

    try:
        multi = builder.generate_spheres(radii_phi_list, phase_id=2)
        struct = merope.Structure_3D(multi)

        # Build grid and extract 3D array
        grid_params = merope.vox.create_grid_parameters_N_L_3D(
            [builder.n3D] * 3, builder.L
        )
        grid = merope.vox.GridRepresentation_3D(
            struct, grid_params, merope.vox.VoxelRule.Average
        )

        # Measure porosity before applying homogRule
        try:
            analyzer = merope.vox.GridAnalyzer_3D()
            fracs = analyzer.compute_percentages(grid)
            phi_real = float(fracs.get(2, 0.0))
        except Exception:
            phi_real = target_porosity  # fallback

        err = abs(phi_real - target_porosity) / max(target_porosity, 1e-6)
        por_fitness = max(0.0, 1.0 - err)

        grid.apply_homogRule(merope.HomogenizationRule.Voigt, list(K_THERMAL))
        conv = merope.vox.NumpyConverter_3D()
        array3d = conv.compute_RealField(grid).reshape(
            (builder.n3D,) * 3, order='C'
        )

        # Also write VTK for Amitex (in work_dir)
        with pm.cd(fixed["work_dir"]):
            printer = merope.vox.vtk_printer_3D()
            printer.printVTK_segmented(grid, "structure.vtk", "Coeffs.txt", nameValue="MaterialId")

        slices_dir = os.path.join(fixed["work_dir"], "tmp_slices")
        eval_res = evaluate_slices(
            array3d, exp_image,
            n_slices=n_slices, grid_size=grid_size, temp_dir=slices_dir
        )
        avg_score = eval_res["average_score"]

        w_data = fixed["w_data"]
        w_por  = fixed["w_por"]
        total  = w_data * avg_score + w_por * por_fitness
        print(
            f"   → phi_real={phi_real:.3f}  por_fit={por_fitness:.3f}  "
            f"img_score={avg_score:.3f}  total={total:.3f}"
        )
        return float(total)

    except Exception as exc:
        import traceback
        print(f"   [ERROR distributed] {exc}")
        traceback.print_exc()
        return 0.0


def _build_and_score_interconnected(
    params: Dict[str, float],
    fixed: Dict[str, Any],
    exp_image: str,
) -> float:
    """Generate an interconnected-porosity structure and return the average score.

    Parameters
    ----------
    params  : dict with ``delta`` and ``inter_phi``.
    fixed   : dict with all fixed parameters.
    """
    builder: MicrostructureBuilder = fixed["builder"]
    pm: ProjectManager = fixed["pm"]
    target_porosity: float = fixed["target_porosity"]
    n_slices: int = fixed["n_slices"]
    grid_size: int = fixed["grid_size"]

    delta     = float(params["delta"])
    inter_phi = float(params["inter_phi"])

    try:
        struct = builder.generate_interconnected_structure(
            inter_radius=fixed["inter_radius"],
            inter_phi=inter_phi,
            intra_radius=fixed["intra_radius"],
            intra_phi=fixed["intra_phi"],
            grain_radius=fixed["grain_radius"],
            grain_phi=fixed["grain_phi"],
            delta=delta,
        )

        # Build grid and extract 3D array
        grid_params = merope.vox.create_grid_parameters_N_L_3D(
            [builder.n3D] * 3, builder.L
        )
        grid = merope.vox.GridRepresentation_3D(
            struct, grid_params, merope.vox.VoxelRule.Average
        )

        # Measure porosity before applying homogRule
        try:
            analyzer = merope.vox.GridAnalyzer_3D()
            fracs = analyzer.compute_percentages(grid)
            phi_real = float(fracs.get(2, 0.0))
        except Exception:
            phi_real = target_porosity  # fallback

        err = abs(phi_real - target_porosity) / max(target_porosity, 1e-6)
        por_fitness = max(0.0, 1.0 - err)

        grid.apply_homogRule(merope.HomogenizationRule.Voigt, list(K_THERMAL))
        conv = merope.vox.NumpyConverter_3D()
        array3d = conv.compute_RealField(grid).reshape(
            (builder.n3D,) * 3, order='C'
        )

        # Also write VTK for Amitex (in work_dir)
        with pm.cd(fixed["work_dir"]):
            printer = merope.vox.vtk_printer_3D()
            printer.printVTK_segmented(grid, "structure.vtk", "Coeffs.txt", nameValue="MaterialId")

        slices_dir = os.path.join(fixed["work_dir"], "tmp_slices")
        eval_res = evaluate_slices(
            array3d, exp_image,
            n_slices=n_slices, grid_size=grid_size, temp_dir=slices_dir
        )
        avg_score = eval_res["average_score"]

        w_data = fixed["w_data"]
        w_por  = fixed["w_por"]
        total  = w_data * avg_score + w_por * por_fitness
        print(
            f"   → phi_real={phi_real:.3f}  por_fit={por_fitness:.3f}  "
            f"img_score={avg_score:.3f}  total={total:.3f}"
        )
        return float(total)

    except Exception as exc:
        print(f"   [ERROR] {exc}")
        return 0.0


# ---------------------------------------------------------------------------
# Mode-specific configuration
# ---------------------------------------------------------------------------

def _make_space_distributed() -> List:
    return [
        Real(np.log(0.05), np.log(2.0), name="mean_radius"),
        Real(0.10, 0.60,                name="std_radius"),   # min 0.10: avoids trivial monodisperse solution
    ]


def _make_space_interconnected() -> List:
    return [
        Real(1e-4, 0.05, name="delta"),
        Real(0.01, 0.40, name="inter_phi"),
    ]


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bayesian optimization of Mérope microstructure parameters."
    )
    parser.add_argument(
        "--mode", choices=["distributed", "interconnected"], required=True,
        help="Type of porosity to calibrate."
    )
    parser.add_argument(
        "--exp-image", default=None,
        help="Path to the experimental reference image (overrides default)."
    )
    parser.add_argument("--n-calls", type=int, default=20,
        help="Number of Bayesian optimization calls (default: 20).")
    parser.add_argument("--n3d", type=int, default=None,
        help="Voxel resolution per axis (default: 120 distributed / 150 interconnected).")
    parser.add_argument("--n-cpus", type=int, default=4,
        help="CPU cores for Amitex (default: 4).")
    parser.add_argument("--seed", type=int, default=0,
        help="RNG seed for Mérope (default: 0).")
    parser.add_argument("--n-slices", type=int, default=99,
        help="Number of 2D slices used for image comparison (default: 99).")
    parser.add_argument("--run-amitex", action="store_true",
        help="Run Amitex on the best geometry and report K_eff.")
    args = parser.parse_args()

    mode = args.mode
    exp_image = args.exp_image or _EXP_IMAGES[mode]
    target_porosity = _TARGET_POROSITY[mode]
    n3d = args.n3d or (120 if mode == "distributed" else 150)

    # ── Output directory ─────────────────────────────────────────────────
    output_dir = Path(f"Results_Optimization_{mode.capitalize()}")
    work_dir   = str(output_dir / "work")

    pm = ProjectManager()
    pm.cleanup_folder(str(output_dir))
    Path(work_dir).mkdir(parents=True, exist_ok=True)

    # ── Mérope builder ───────────────────────────────────────────────────
    builder = MicrostructureBuilder(L=[10.0, 10.0, 10.0], n3D=n3d, seed=args.seed)

    # ── Fixed parameters shared by both modes ────────────────────────────
    fixed: Dict[str, Any] = {
        "builder":          builder,
        "pm":               pm,
        "target_porosity":  target_porosity,
        "n_slices":         args.n_slices,
        "grid_size":        20,
        "seed":             args.seed,
        "work_dir":         work_dir,
        "w_data":           0.3,    # weight for image similarity score
        "w_por":            0.7,    # weight for porosity fitness
    }

    if mode == "distributed":
        fixed["num_radii"] = 6     # how many log-normal sample radii per evaluation
        space = _make_space_distributed()

        @use_named_args(space)
        def objective(**params):
            print(f"\n[Call] mean_radius={params['mean_radius']:.4f}  "
                  f"std_radius={params['std_radius']:.4f}")
            score = _build_and_score_distributed(params, fixed, exp_image)
            return -score          # gp_minimize minimises → negate

    else:  # interconnected
        fixed.update({
            "inter_radius": 0.03,
            "intra_radius": 0.10,
            "intra_phi":    0.0,     # no intra-granular porosity per default
            "grain_radius": 1.0,
            "grain_phi":    1.0,
        })
        space = _make_space_interconnected()

        @use_named_args(space)
        def objective(**params):
            print(f"\n[Call] delta={params['delta']:.5f}  "
                  f"inter_phi={params['inter_phi']:.4f}")
            score = _build_and_score_interconnected(params, fixed, exp_image)
            return -score

    # ── Bayesian Optimization ────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Bayesian Optimization — mode: {mode.upper()}")
    print(f"  Experimental image : {exp_image}")
    print(f"  Target porosity    : {target_porosity*100:.1f}%")
    print(f"  Voxel resolution   : {n3d}³")
    print(f"  Calls              : {args.n_calls}")
    print(f"{'='*60}\n")

    res = gp_minimize(
        objective,
        space,
        n_calls=args.n_calls,
        n_initial_points=min(max(3, args.n_calls // 4), args.n_calls - 1),
        random_state=args.seed,
        verbose=False,
    )

    best_score  = -res.fun
    best_params = {dim.name: val for dim, val in zip(space, res.x)}

    print(f"\n{'='*60}")
    print(f"  BEST PARAMETERS FOUND")
    print(f"  Mode: {mode}")
    for k, v in best_params.items():
        print(f"    {k:20s}: {v:.6f}")
    print(f"  Best combined score : {best_score:.4f}")
    print(f"{'='*60}")

    # ── Convergence plot ─────────────────────────────────────────────────
    scores_over_calls = [-y for y in res.func_vals]
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(scores_over_calls) + 1), scores_over_calls,
             "b-o", markersize=5)
    plt.axhline(best_score, color="red", linestyle="--",
                linewidth=1.2, label=f"Best = {best_score:.4f}")
    plt.xlabel("Optimization call")
    plt.ylabel("Combined score")
    plt.title(f"Bayesian Optimization convergence — {mode}")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    conv_path = str(output_dir / "convergence.png")
    plt.savefig(conv_path, dpi=200)
    plt.close()
    print(f"\nConvergence plot saved → {conv_path}")

    # ── Re-generate best structure for final visual comparison ───────────
    print("\nRe-generating best structure for final slice comparison...")
    best_eval: Optional[dict] = None
    best_array3d: Optional[np.ndarray] = None

    try:
        if mode == "distributed":
            rng = np.random.default_rng(args.seed)
            sampled = np.abs(
                rng.lognormal(best_params["mean_radius"], best_params["std_radius"],
                              fixed["num_radii"])
            )
            sampled = sampled[(sampled >= 0.005) & (sampled <= 3.0)]
            phi_each = target_porosity / sampled.size
            radii_phi_list = [[float(r), float(phi_each)] for r in sampled]
            multi = builder.generate_spheres(radii_phi_list, phase_id=2)
            struct = merope.Structure_3D(multi)
        else:
            struct = builder.generate_interconnected_structure(
                inter_radius=fixed["inter_radius"],
                inter_phi=best_params["inter_phi"],
                intra_radius=fixed["intra_radius"],
                intra_phi=fixed["intra_phi"],
                grain_radius=fixed["grain_radius"],
                grain_phi=fixed["grain_phi"],
                delta=best_params["delta"],
            )

        best_vtk_dir = str(output_dir / "best_geometry")
        with pm.cd(best_vtk_dir):
            fractions = builder.voxellate(struct, K_THERMAL)
            phi_real = fractions.get(2, 0.0)

            grid_params = merope.vox.create_grid_parameters_N_L_3D(
                [builder.n3D] * 3, builder.L
            )
            grid = merope.vox.GridRepresentation_3D(
                struct, grid_params, merope.vox.VoxelRule.Average
            )
            grid.apply_homogRule(
                merope.HomogenizationRule.Voigt, list(K_THERMAL)
            )
            conv_np = merope.vox.NumpyConverter_3D()
            best_array3d = conv_np.compute_RealField(grid).reshape(
                (builder.n3D,) * 3, order='C'
            )

            # ── Optional Amitex run ─────────────────────────────────────
            k_eff_result: Dict[str, float] = {}
            if args.run_amitex:
                solver = ThermalSolver(n_cpus=args.n_cpus)
                k_eff_result = solver.solve()
                print(f"  K_eff (Amitex) → Kmean = {k_eff_result.get('Kmean', 0):.5f}")

        # ── Slice evaluation on the best geometry ───────────────────────
        final_slices_dir = str(output_dir / "final_slices")
        best_eval = evaluate_slices(
            best_array3d, exp_image,
            n_slices=args.n_slices,
            grid_size=fixed["grid_size"],
            temp_dir=final_slices_dir,
        )

        # Save the best matching slice next to the output
        if best_eval["best"] is not None:
            best_slice_name, best_slice_scores = best_eval["best"]
            src = os.path.join(final_slices_dir, best_slice_name)
            dst = str(output_dir / "best_slice.png")
            shutil.copy2(src, dst)
            print(
                f"  Best slice saved → {dst}\n"
                f"  KS p={best_slice_scores['ks']:.4f}  "
                f"Chi² p={best_slice_scores['chi']:.4f}  "
                f"score={best_slice_scores['score']:.4f}"
            )

        # ── Pore area distribution plot ─────────────────────────────────
        if best_eval["best"] is not None:
            best_slice_path = str(output_dir / "best_slice.png")
            dist_path = str(output_dir / "area_distribution.png")
            plot_area_distribution(best_slice_path, exp_image, dist_path)
            print(f"  Area distribution plot → {dist_path}")

    except Exception as exc:
        print(f"  [WARNING] Final evaluation failed: {exc}")

    # ── Summary file ─────────────────────────────────────────────────────
    summary_path = output_dir / "summary.txt"
    with open(summary_path, "w") as fp:
        fp.write(f"Bayesian Optimization Summary — {mode}\n")
        fp.write("=" * 50 + "\n")
        fp.write(f"Mode               : {mode}\n")
        fp.write(f"Experimental image : {exp_image}\n")
        fp.write(f"Target porosity    : {target_porosity*100:.1f}%\n")
        fp.write(f"Voxel resolution   : {n3d}^3\n")
        fp.write(f"Optimization calls : {args.n_calls}\n\n")
        fp.write("Best parameters:\n")
        for k, v in best_params.items():
            fp.write(f"  {k:20s}: {v:.6f}\n")
        fp.write(f"\nBest combined score: {best_score:.4f}\n")
        if best_eval is not None:
            fp.write(f"Image avg score    : {best_eval['average_score']:.4f}\n")
            fp.write(f"Real porosity      : {phi_real*100:.2f}%\n")
        if args.run_amitex and k_eff_result:
            fp.write("\nAmitex K_eff result:\n")
            for k, v in k_eff_result.items():
                fp.write(f"  {k}: {v:.6f}\n")

    print(f"\nSummary written → {summary_path}")
    print("\nDone.\n")


if __name__ == "__main__":
    main()
