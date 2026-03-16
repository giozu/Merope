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

# Import Merope directly for manual structure building
import merope
import sac_de_billes

# --- Configuration ---
L_DIM = [10.0, 10.0, 10.0]    # RVE size (physical units)
N_VOX = 100  # Higher resolution for better accuracy (100^3 = 1M voxels)
K_THERMAL = [1.0, 1.0, 1e-3]   # Phase 0=Matrix, 1=Boundary(Solid), 2=Pore

# Normalized grain radius L_grain = 1.0 (adimensionalization)
# This makes delta a relative parameter: delta/L_grain ∈ [0, 1]
FIXED_GRAIN_R = 1.0

# Delta values: scan from 0.1 to 0.9 to capture full crack-to-sphere transition
DELTA_VALUES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
# Porosity targets: three curves like in the reference plot
P_TARGETS    = [0.1, 0.2, 0.3]
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

    builder = MicrostructureBuilder(L=L_DIM, n3D=N_VOX, seed=42)
    solver  = ThermalSolver(n_cpus=1)
    pm = ProjectManager()

    case_dir = OUTPUT_DIR / f"P_{p_target:.2f}_Delta_{delta:.3f}"

    # Phase strategy from iter_delta_IGB_calc.py:
    # - Create Laguerre grains with boundary layer
    # - Create spherical pore inclusions
    # - Overlay pores on grains: pores "win" where they exist
    # - Final phases: 0=grains+boundaries (solid), 2=pores (low K)
    # K_THERMAL = [K_solid, K_solid, K_pore]

    incl_phase = 2      # Pores
    delta_phase = 3     # Grain boundary layer (temporary)
    grains_phase = 0    # Grain cores (solid)

    # Domain size
    domain_size = L_DIM

    # Microstructure parameters
    lagR = 1.0          # Laguerre grain size
    lagPhi = 1.0        # Fill entire RVE with grains
    inclR = 0.15        # Pore radius
    inclPhi_input = p_target * 0.8  # Initial guess for pore volume fraction

    p_real = 0.0

    # Iterative refinement to hit p_target
    for iteration in range(8):
        try:
            current_seed = 42 + iteration

            # 1. Create spherical pore inclusions (phase 2)
            sphIncl_pores = merope.SphereInclusions_3D()
            sphIncl_pores.setLength(domain_size)
            sphIncl_pores.fromHisto(
                current_seed,
                sac_de_billes.TypeAlgo.BOOL,
                0.0,
                [[inclR, inclPhi_input]],
                [incl_phase]  # Phase 2 = pores
            )
            multiInclusions_pores = merope.MultiInclusions_3D()
            multiInclusions_pores.setInclusions(sphIncl_pores)

            # 2. Create Laguerre tessellation for grains
            sphIncl_grains = merope.SphereInclusions_3D()
            sphIncl_grains.setLength(domain_size)
            sphIncl_grains.fromHisto(
                current_seed,
                sac_de_billes.TypeAlgo.RSA,
                0.0,
                [[lagR, lagPhi]],
                [1]  # Temporary phase for Laguerre seeds
            )
            polyCrystal = merope.LaguerreTess_3D(domain_size, sphIncl_grains.getSpheres())

            multiInclusions_grains = merope.MultiInclusions_3D()
            multiInclusions_grains.setInclusions(polyCrystal)

            # Add grain boundary layer (phase 3) and set grain cores to phase 1
            ids = multiInclusions_grains.getAllIdentifiers()
            multiInclusions_grains.addLayer(ids, delta_phase, delta)  # Boundaries = phase 3
            multiInclusions_grains.changePhase(ids, [1 for _ in ids])  # Cores = phase 1

            # 3. Combine structures with remapping
            # Following iter_delta_IGB_calc.py line 124-127:
            # Remap both pores (2) and boundaries (3) to grains (0)
            # But pores overlay on top, so where pores exist, they stay as pores
            dictionnaire = {incl_phase: grains_phase, delta_phase: grains_phase}
            structure = merope.Structure_3D(
                multiInclusions_pores,
                multiInclusions_grains,
                dictionnaire
            )

            # 4. Voxellate
            with pm.cd(str(case_dir)):
                fractions = builder.voxellate(structure, K_THERMAL)

            # Phase 0 = grains+boundaries (solid), Phase 2 = pores
            phi_solid = fractions.get(0, 0.0)
            phi_pores = fractions.get(2, 0.0)
            p_real = phi_pores

            print(f"   [Iter {iteration}] delta={delta:.3f}, inclPhi_in={inclPhi_input:.3f} -> solid={phi_solid:.3f}, pores={phi_pores:.3f} (tgt={p_target:.2f})")

            # More tolerant convergence criterion: ±5% error acceptable
            if abs(p_real - p_target) < 0.05:
                print(f"   ✓ Converged within 5%")
                break

            # Adjust pore volume fraction to hit target porosity
            error = p_target - p_real
            inclPhi_input += error * 0.8  # Aggressive adjustment
            inclPhi_input = float(np.clip(inclPhi_input, 0.01, 0.7))

        except RuntimeError as e:
            # If RSA/BOOL fails, reduce pore fraction
            print(f"   [Iter {iteration}] Sphere packing error: {e}")
            inclPhi_input *= 0.7
            inclPhi_input = max(0.01, inclPhi_input)
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
        "Grain_R":     lagR,
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
