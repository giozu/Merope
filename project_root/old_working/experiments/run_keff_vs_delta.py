"""
run_keff_vs_delta.py
====================
Reproduces the slide logic: "K_eff vs delta for fixed generation parameter p".
Interconnected porosity is modeled by clipping spherical pores (intra_phi) 
within the grain boundary layer of thickness delta.

If delta -> 0, the pores are constrained to narrow grain boundary cracks -> low K_eff.
If delta -> large, the pores expand into full isolated spheres -> high K_eff.
"""

import sys
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import concurrent.futures
from scipy.optimize import curve_fit

# Ensure project_root/ is on sys.path so `core` is importable
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from core.geometry import MicrostructureBuilder
from core.solver import ThermalSolver
from core.utils import ProjectManager

# --- Configuration ---
L_DIM = [20.0, 20.0, 20.0]    # RVE size (physical units)
N_VOX = 120                    # Voxel resolution

# Phase convention from generate_polycrystal:
#   phase 1  = grain interior (matrix)   → K_matrix
#   phase 2  = grain-boundary film δ     → K_gas  (the porous/crack phase)
K_THERMAL = [1.0, 1.0, 1e-3]   # index 0 unused, 1=matrix, 2=pore

P_TARGETS    = [0.1, 0.2, 0.3]
DELTA_VALUES = np.linspace(0.05, 1.0, 15)
OUTPUT_DIR   = Path("Results_Keff_vs_Delta")

# Grain surface-area-to-volume factor (S_v) for a random Laguerre tessellation.
# phi_boundary ≈ S_v * delta.  For RSA packing with grain_radius R in a box of
# side L, S_v ≈ 3/R (spherical packing approximation).
# To hit phi = p we need:  grain_radius ≈ 3 * delta / p
# But grain_radius must stay in [R_MIN, R_MAX] for a sensible tessellation.
R_MIN = 0.5   # never go below this (too many tiny grains → slow)
R_MAX = 8.0   # never above this (too few grains → RVE not representative)

def _grain_radius_for_phi(p: float, delta: float) -> float:
    """Return grain radius so that GB layer phi ≈ p for given delta.

    Uses the approximate relation  phi ≈ 3 * delta / grain_radius  (Kelvin cell).
    """
    r = 3.0 * delta / p
    return float(np.clip(r, R_MIN, R_MAX))

def worker(task_args):
    p, delta, no_solver = task_args

    # Scale grain size so boundary-layer volume fraction ≈ p
    grain_radius = _grain_radius_for_phi(p, delta)

    builder = MicrostructureBuilder(L=L_DIM, n3D=N_VOX, seed=42)
    solver  = ThermalSolver(n_cpus=2)

    case_dir = OUTPUT_DIR.resolve() / f"P_{p:.2f}_Delta_{delta:.3f}"
    case_dir.mkdir(parents=True, exist_ok=True)

    vtk_path     = case_dir / "structure.vtk"
    coeffs_path  = case_dir / "Coeffs.txt"
    results_file = case_dir / "thermalCoeff_amitex.txt"

    # Physical model (from slide):
    #   generate_polycrystal → phase 1=grain interior (K_matrix), phase 2=GB film (K_gas)
    #   Small δ → thin connected film  → percolating insulator → K_eff LOW
    #   Large δ → thick film → grain interior shrinks → pores become isolated → K_eff RISES
    struct = builder.generate_polycrystal(grain_radius=grain_radius, delta=delta)

    fractions = builder.voxellate(struct, K_THERMAL,
                                  vtk_path=vtk_path,
                                  coeffs_path=coeffs_path)
    p_real = fractions.get(2, 0.0)   # only phase 2 is porous here

    if no_solver:
        res = {"Kmean": 0.0}
    else:
        res = solver.solve(vtk_file=vtk_path, results_file=results_file)

    k_eff = res["Kmean"]
    print(f" [DONE] P={p:.2f} | delta={delta:.3f} | R_grain={grain_radius:.2f}"
          f" | Real phi={p_real:.4f} -> K_eff={k_eff:.4f}")

    return {
        "Target_P":    p,
        "Delta":       delta,
        "Grain_R":     grain_radius,
        "Real_P":      p_real,
        "K_eff":       k_eff,
    }


def run_sweeps(no_solver=False):
    pm = ProjectManager()
    pm.cleanup_folder(str(OUTPUT_DIR))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    tasks = [(p, delta, no_solver) for p in P_TARGETS for delta in DELTA_VALUES]
    print(f"=== K_eff vs Delta Sweep (Parallel, {len(tasks)} tasks, max_workers=4) ===")

    # ProcessPoolExecutor scales extremely well across heavy independent solvers like Amitex
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
        rows = list(executor.map(worker, tasks))

    df = pd.DataFrame(rows)
    df = df.sort_values(by=["Target_P", "Delta"]).reset_index(drop=True)
    df.to_csv(OUTPUT_DIR / "keff_vs_delta.csv", index=False)
    return df

def _sat_exp(delta, a, b, c):
    """Saturating exponential: K = a - b * exp(-c * delta)."""
    return a - b * np.exp(-c * delta)

def plot_slide(df, output_dir):
    fig, ax = plt.subplots(figsize=(8, 6))

    colors   = {0.1: "steelblue", 0.2: "darkorange", 0.3: "forestgreen"}
    p_labels = {0.1: "0.1",       0.2: "0.2",        0.3: "0.3"}

    delta_fine = np.linspace(df["Delta"].min() * 0.5, df["Delta"].max() * 1.05, 400)

    for p, group in df.groupby("Target_P"):
        col = colors.get(p, "black")
        x   = group["Delta"].values
        y   = group["K_eff"].values

        # Scatter: Data points
        ax.scatter(x, y, s=40, color=col, alpha=0.85, zorder=3,
                   label=f"Data p={p_labels.get(p, p)}")

        # Fit: K = a - b * exp(-c * delta)
        try:
            # Sensible initial guess: a~max, b~max-min, c~5/range
            a0 = max(y) * 1.01
            b0 = a0 - min(y)
            c0 = 5.0 / (x.max() - x.min() + 1e-9)
            popt, _ = curve_fit(_sat_exp, x, y,
                                 p0=[a0, b0, c0],
                                 bounds=([0, 0, 0], [1.5, 1.5, 200]),
                                 maxfev=8000)
            ax.plot(delta_fine, _sat_exp(delta_fine, *popt),
                    linestyle="--", color=col, linewidth=1.6,
                    label=f"Fit p={p_labels.get(p, p)}")
        except Exception as e:
            print(f"  [fit warning p={p}] {e}")

    ax.set_xlabel(r"delta", fontsize=12)
    ax.set_ylabel(r"k [W/mK]", fontsize=12)
    ax.set_title(r"$K_\mathrm{eff}$ vs $\delta$ — crack-to-sphere transition", fontsize=13)
    ax.grid(True, linestyle="--", alpha=0.4, color="grey")
    ax.set_xlim(0, df["Delta"].max() * 1.05)
    ax.set_ylim(max(0.0, df["K_eff"].min() - 0.05), 1.05)
    ax.legend(fontsize=9, loc="lower right", framealpha=0.9)

    fig.tight_layout()
    img_path = output_dir / "Slide_Keff_vs_Delta.png"
    fig.savefig(img_path, dpi=300)
    print(f"\nSaved plot to {img_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--no-solver", action="store_true")
    args = parser.parse_args()
    
    if args.plot_only:
        df = pd.read_csv(OUTPUT_DIR / "keff_vs_delta.csv")
    else:
        df = run_sweeps(args.no_solver)
    
    plot_slide(df, OUTPUT_DIR)
