"""
run_keff_vs_delta.py
====================
Reproduces the slide logic: "K_eff vs delta at fixed total porosity".
Interconnected porosity is modeled using the parameter delta (thickness
of the porous region at grain boundaries).
To keep the total porosity p constant while increasing delta, the
grain_radius must be increased. This changes the morphology from a dense
network of thin cracks (delta -> 0, low K_eff) to a sparse network of
thick pores (delta -> L_grain, high K_eff).
"""

import sys
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Ensure project_root/ is on sys.path so `core` is importable
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from core.geometry import MicrostructureBuilder
from core.solver import ThermalSolver
from core.utils import ProjectManager
import merope

# --- Configuration ---
L_DIM = [10.0, 10.0, 10.0]
N_VOX = 120
K_THERMAL = [0.0, 1.0, 1e-3]  # [Ignored, Matrix, Pores]

P_TARGETS = [0.1, 0.2, 0.3]
DELTA_VALUES = np.linspace(0.1, 1.0, 15)
OUTPUT_DIR = Path("Results_Keff_vs_Delta")

def find_grain_radius(builder, target_p, delta, tol=0.005, max_iter=15):
    """
    Binary search for the grain_radius that gives the target porosity `target_p`
    for a given `delta`.
    """
    low, high = 0.5, 20.0  # Search bounds for grain radius
    best_r = 5.0
    analyzer = merope.vox.GridAnalyzer_3D()
    
    for _ in range(max_iter):
        mid = (low + high) / 2.0
        try:
            struct = builder.generate_polycrystal(grain_radius=mid, delta=delta)
            
            # Fast voxelization just for porosity check (no Amitex solver)
            grid_repr = merope.vox.GridRepresentation_3D(struct, builder.grid_params, merope.vox.VoxelRule.Average)
            frac = analyzer.compute_percentages(grid_repr)
            p_real = frac.get(2, 0.0)
            
            error = p_real - target_p
            best_r = mid
            
            if abs(error) < tol:
                break
                
            if error > 0:
                # Too much porosity -> cracks are too dense -> grains are too small -> increase radius
                low = mid
            else:
                high = mid
        except Exception:
            # If radius is too large/small and RSA packing fails, adjust bounds
            high = mid
            
    return best_r

def run_sweeps(no_solver=False):
    pm = ProjectManager()
    builder = MicrostructureBuilder(L=L_DIM, n3D=N_VOX, seed=42)
    solver = ThermalSolver(n_cpus=4)
    pm.cleanup_folder(str(OUTPUT_DIR))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    rows = []
    
    print("=== K_eff vs Delta Sweep (Fixed Porosity) ===")
    
    for target_p in P_TARGETS:
        print(f"\n--- Target Porosity: {target_p*100:.1f}% ---")
        
        for delta in DELTA_VALUES:
            print(f"  > Tuning grain_radius for Delta = {delta:.3f} ...", end="", flush=True)
            r_grain = find_grain_radius(builder, target_p, delta)
            
            case_dir = OUTPUT_DIR / f"P_{target_p:.2f}_Delta_{delta:.3f}"
            with pm.cd(str(case_dir)):
                # Generate final structure with tuned radius
                struct = builder.generate_polycrystal(grain_radius=r_grain, delta=delta)
                fractions = builder.voxellate(struct, K_THERMAL)
                p_real = fractions.get(2, 0.0)
                
                print(f" tuned R={r_grain:.2f} -> Validated p={p_real:.4f}")
                
                if no_solver:
                    res = {"Kmean": 0.0}
                else:
                    res = solver.solve()
                    
                k_eff = res["Kmean"]
                print(f"    -> K_eff = {k_eff:.4f}")
                
                rows.append({
                    "Target_P": target_p,
                    "Delta": delta,
                    "Grain_Radius": r_grain,
                    "Real_P": p_real,
                    "K_eff": k_eff
                })

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_DIR / "keff_vs_delta.csv", index=False)
    return df

def plot_slide(df, output_dir):
    fig, ax = plt.subplots(figsize=(8, 6))
    
    colors = {0.1: "blue", 0.2: "darkorange", 0.3: "green"}
    
    for p, group in df.groupby("Target_P"):
        ax.plot(group["Delta"], group["K_eff"], 'o', markersize=5, 
                color=colors.get(p, "black"), label=f"p = {p}")
        ax.plot(group["Delta"], group["K_eff"], '-', alpha=0.3, color=colors.get(p, "black"))

    ax.set_xlabel(r"$\delta$ (Thickness of porous region at GB)", fontsize=12)
    ax.set_ylabel(r"$K_\mathrm{eff}$ [W/m·K]", fontsize=12)
    ax.set_title("Effect of $\delta$ on Connectivity at Fixed Porosity", fontsize=14)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    ax.set_xlim(0, 1.2)
    ax.set_ylim(-0.05, 1.0)
    
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
