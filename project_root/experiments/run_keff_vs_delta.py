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

# Ensure project_root/ is on sys.path so `core` is importable
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from core.geometry import MicrostructureBuilder
from core.solver import ThermalSolver
from core.utils import ProjectManager

# --- Configuration ---
L_DIM = [20.0, 20.0, 20.0]  # RVE size
N_VOX = 120                 # Voxel resolution
K_THERMAL = [1.0, 1.0, 1e-3]  # Matrix (0,1) and Pores (2)

P_TARGETS = [0.1, 0.2, 0.3]
# Sweep over delta
DELTA_VALUES = np.linspace(0.05, 1.0, 15)
OUTPUT_DIR = Path("Results_Keff_vs_Delta")

def run_sweeps(no_solver=False):
    pm = ProjectManager()
    builder = MicrostructureBuilder(L=L_DIM, n3D=N_VOX, seed=42)
    solver = ThermalSolver(n_cpus=4)
    pm.cleanup_folder(str(OUTPUT_DIR))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    rows = []
    
    print("=== K_eff vs Delta Sweep (Transition Crack to Spheres) ===")
    
    for p in P_TARGETS:
        print(f"\n--- Target P (intra_phi generation) = {p} ---")
        
        for delta in DELTA_VALUES:
            case_dir = OUTPUT_DIR / f"P_{p:.2f}_Delta_{delta:.3f}"
            
            with pm.cd(str(case_dir)):
                # Use generate_interconnected_structure:
                # inter_phi = 0.0
                # intra_phi = p
                # grain_radius = 2.0
                struct = builder.generate_interconnected_structure(
                    inter_radius=0.5, inter_phi=0.0,
                    intra_radius=0.5, intra_phi=p,
                    grain_radius=3.0, grain_phi=1.0,
                    delta=delta
                )
                
                fractions = builder.voxellate(struct, K_THERMAL)
                p_real = fractions.get(2, 0.0)
                
                print(f"  > Delta = {delta:.3f} | Real Porosity = {p_real:.4f}", end="", flush=True)
                
                if no_solver:
                    res = {"Kmean": 0.0}
                else:
                    res = solver.solve()
                    
                k_eff = res["Kmean"]
                print(f"    -> K_eff = {k_eff:.4f}")
                
                rows.append({
                    "Target_P": p,
                    "Delta": delta,
                    "Real_P": p_real,
                    "K_eff": k_eff
                })

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_DIR / "keff_vs_delta.csv", index=False)
    return df

def plot_slide(df, output_dir):
    fig, ax = plt.subplots(figsize=(8, 6))
    
    colors = {0.1: "steelblue", 0.2: "darkorange", 0.3: "forestgreen"}
    
    for p, group in df.groupby("Target_P"):
        ax.plot(group["Delta"], group["K_eff"], 'o', markersize=5, 
                color=colors.get(p, "black"), label=f"p = {p}")

    ax.set_xlabel(r"$\delta$ (Thickness of porous region at GB)", fontsize=12)
    ax.set_ylabel(r"$K_\mathrm{eff}$ [W/m·K]", fontsize=12)
    ax.set_title("Effect of $\delta$ on Connectivity (Crack to Sphere Transition)", fontsize=14)
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
