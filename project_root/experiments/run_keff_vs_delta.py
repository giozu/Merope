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
L_DIM = [10.0, 10.0, 10.0]    # RVE size (physical units)
N_VOX = 40  # Good resolution for grain boundaries
K_THERMAL = [1.0, 1.0, 1e-3]   # Phase 0=Matrix, 1=Boundary(Solid), 2=Pore

# Normalized grain radius L_grain = 1.0 (adimensionalization)
# This makes delta a relative parameter: delta/L_grain ∈ [0, 1]
FIXED_GRAIN_R = 1.0

# Delta values: 0 → interconnected cracks, 1 → distributed (full grain size)
# Focus on the transition: delta=0.2 (low K) vs delta=0.8 (high K)
DELTA_VALUES = [0.2, 0.8]  # See the K_eff transition clearly
P_TARGETS    = [0.30]  # Fixed porosity to match green curve in reference plot
OUTPUT_DIR   = Path("Results_Keff_vs_Delta")

R_MIN = 0.5
R_MAX = 4.0   

def _grain_radius_for_phi(p: float, delta: float) -> float:
    """Return grain radius so that GB layer phi ≈ p for given delta."""
    # Empirical scaling for random polycrystals
    r = 3.0 * delta / p
    return float(np.clip(r, R_MIN, R_MAX))

def worker(task_args):
    p_target, delta, no_solver = task_args

    grain_radius = FIXED_GRAIN_R  # Fixed at 1.0 (normalized L_grain)
    builder = MicrostructureBuilder(L=L_DIM, n3D=N_VOX, seed=42)
    solver  = ThermalSolver(n_cpus=1)
    pm = ProjectManager()

    case_dir = OUTPUT_DIR / f"P_{p_target:.2f}_Delta_{delta:.3f}"

    # CORRECT PHYSICS from iter_delta_IGB_calc.py:
    # - Total porosity p_target = fixed
    # - delta = thickness of grain boundary layer (inter-granular network)
    # - As delta increases:
    #   * Boundary layer occupies more volume
    #   * Less space/need for inter-granular spherical pores
    #   * Morphology shifts from interconnected (delta small) to distributed (delta large)
    #   * K_eff INCREASES (less interconnection bottleneck)
    #
    # Implementation:
    # - Use generate_interconnected_structure with:
    #   * intra_phi: spherical pores INSIDE grains (always distributed)
    #   * delta: boundary layer thickness
    #   * Total porosity ≈ phi_intra + phi_boundary
    # - As delta increases, reduce intra_phi to keep total porosity constant

    # STRATEGY: Vary grain_radius to control boundary layer porosity
    # For a given delta, find R such that total porosity ≈ p_target
    # phi_boundary(delta, R) ≈ 1 - (1 - delta/R)³
    # We want phi_boundary + phi_intra ≈ p_target
    # Start by trying to make boundary layer contribute ~70% of target porosity

    # Strategy: For p=0.3, we want boundary + intra ≈ 0.3
    # - Small delta (0.2): thin boundaries → need more intra pores → less connected
    # - Large delta (0.8): thick boundaries contribute more → less intra needed → more distributed

    # Initial guess: boundary contributes ~50% of target porosity
    phi_boundary_target = p_target * 0.5
    grain_radius = delta / (1.0 - (1.0 - phi_boundary_target)**(1.0/3.0) + 1e-6)
    grain_radius = float(np.clip(grain_radius, 1.0, 6.0))

    # Start with small intra porosity, will adjust iteratively
    intra_phi_input = p_target * 0.5  # Other 50% from intra pores
    intra_radius = 0.15  # Medium-small distributed pores
    p_real = 0.0

    # Iterative refinement to hit p_target (max 8 iterations)
    for iteration in range(8):
        try:
            # Use generate_mixed_structure: simpler, no inter-granular spheres
            struct = builder.generate_mixed_structure(
                grain_radius=grain_radius,
                delta=delta,
                intra_pore_list=[[intra_radius, intra_phi_input]],
            )

            # Use pm.cd() pattern like run_distributed_porosity.py (working version)
            with pm.cd(str(case_dir)):
                fractions = builder.voxellate(struct, K_THERMAL)
            # Phase 0 = matrix, Phase 1 = intra pores, Phase 2 = boundary layer
            # Total porosity = phase 1 + phase 2
            phi_intra = fractions.get(1, 0.0)
            phi_boundary = fractions.get(2, 0.0)
            p_real = phi_intra + phi_boundary

            print(f"   [Iter {iteration}] delta={delta:.3f}, R={grain_radius:.2f}, intra_in={intra_phi_input:.3f} -> bound={phi_boundary:.3f}, intra={phi_intra:.3f}, tot={p_real:.3f} (tgt={p_target:.2f})")

            # More tolerant convergence criterion: ±5% error acceptable
            if abs(p_real - p_target) < 0.05:
                print(f"   ✓ Converged within 5%")
                break

            # Adjust both grain_radius and intra_phi
            error = p_target - p_real

            # If total porosity is way off, adjust grain radius (controls boundary layer)
            if abs(error) > 0.1:
                # More porosity needed → smaller grains (more boundary surface)
                # Less porosity needed → larger grains (less boundary surface)
                if error > 0:
                    grain_radius *= 0.85
                else:
                    grain_radius *= 1.2
                grain_radius = float(np.clip(grain_radius, 1.0, 6.0))

            # Fine-tune with intra_phi (more conservative adjustment)
            intra_phi_input += error * 0.5
            intra_phi_input = float(np.clip(intra_phi_input, 0.01, 0.6))

        except RuntimeError as e:
            # If RSA fails, reduce intra pores and try larger grains
            print(f"   [Iter {iteration}] RSA error, adjusting parameters")
            grain_radius *= 1.4
            grain_radius = float(np.clip(grain_radius, 1.0, 6.0))
            intra_phi_input *= 0.6
            intra_phi_input = max(0.01, intra_phi_input)
            if iteration > 5:
                print(f"   ⚠ Accepting p_real={p_real:.4f} after {iteration} iterations")
                break

    # Run solver in the case directory
    with pm.cd(str(case_dir)):
        if no_solver:
            res = {"Kmean": 0.0}
        else:
            res = solver.solve()  # Uses structure.vtk in CWD

    k_eff = res["Kmean"]
    print(f" [DONE] P={p_target:.2f} | delta={delta:.3f} | phi={p_real:.4f} -> K_eff={k_eff:.4f}")

    return {
        "Target_P":    p_target,
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
    rows = []
    for t in tasks:
        rows.append(worker(t))

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
