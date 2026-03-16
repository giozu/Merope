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

# --- Configuration (matching iter_delta_IGB_calc.py) ---
L_DIM = [10.0, 10.0, 10.0]    # RVE size (physical units)
N_VOX = 100  # Resolution (100^3 = 1M voxels)
K_THERMAL = [1.0, 1.0, 1e-3]   # Phase 0=Solid, 1=Solid, 2=Pore

# Parameters from iter_delta_IGB_calc.py
INCL_R = 0.3      # Pore radius (10x bigger than before!)
LAG_R = 3.0       # Laguerre grain size (3x bigger than before!)
LAG_PHI = 1.0     # Fill entire RVE

# Delta range from iter_delta_IGB_calc.py: 0.39 to 3.0
DELTA_VALUES = np.linspace(0.39, 3.0, 21)  # 21 points like reference
# Porosity targets: three curves
P_TARGETS = [0.1, 0.2, 0.3]
OUTPUT_DIR = Path("Results_Keff_vs_Delta")

R_MIN = 0.5
R_MAX = 4.0   

def _grain_radius_for_phi(p: float, delta: float) -> float:
    """Return grain radius so that GB layer phi ≈ p for given delta."""
    # Empirical scaling for random polycrystals
    r = 3.0 * delta / p
    return float(np.clip(r, R_MIN, R_MAX))

def worker(task_args):
    """Worker function following EXACT pattern from iter_delta_IGB_calc.py"""
    p_target, delta, no_solver = task_args

    builder = MicrostructureBuilder(L=L_DIM, n3D=N_VOX, seed=42)
    solver = ThermalSolver(n_cpus=1)
    pm = ProjectManager()

    case_dir = OUTPUT_DIR / f"P_{p_target:.2f}_Delta_{delta:.3f}"

    # Phase IDs (same as iter_delta_IGB_calc.py lines 82-84)
    incl_phase = 2      # Pores
    delta_phase = 3     # Grain boundary layer (temporary)
    grains_phase = 0    # Grains (final solid phase)

    # Domain size
    domain_size = L_DIM
    lagRphi = [LAG_R, LAG_PHI]

    # Initial guess for inclPhi using pre-calculated array as starting point
    a = [0.421, 0.3, 0.24, 0.201, 0.175, 0.157, 0.144, 0.134, 0.127, 0.121,
         0.117, 0.1125, 0.11, 0.108, 0.105, 0.103, 0.101, 0.1, 0.099, 0.099, 0.099]
    inclPhi_array = [p_target * 10 * x for x in a]
    delta_idx = np.argmin(np.abs(DELTA_VALUES - delta))
    inclPhi = inclPhi_array[delta_idx]

    p_real = 0.0

    # Iterative convergence loop (max 10 iterations)
    for iteration in range(10):
        try:
            seed = 42 + iteration  # Change seed if packing fails

            # 1. Create spherical pore inclusions (phase 2)
            sphIncl_pores = merope.SphereInclusions_3D()
            sphIncl_pores.setLength(domain_size)
            sphIncl_pores.fromHisto(
                seed,
                sac_de_billes.TypeAlgo.BOOL,
                0.0,
                [[INCL_R, inclPhi]],
                [incl_phase]
            )
            multiInclusions_pores = merope.MultiInclusions_3D()
            multiInclusions_pores.setInclusions(sphIncl_pores)

            # 2. Create Laguerre tessellation for grains
            sphIncl_grains = merope.SphereInclusions_3D()
            sphIncl_grains.setLength(domain_size)
            sphIncl_grains.fromHisto(
                seed,
                sac_de_billes.TypeAlgo.RSA,
                0.0,
                [lagRphi],
                [1]  # Temporary phase
            )
            polyCrystal = merope.LaguerreTess_3D(domain_size, sphIncl_grains.getSpheres())

            multiInclusions_grains = merope.MultiInclusions_3D()
            multiInclusions_grains.setInclusions(polyCrystal)

            # Add grain boundary layer (phase 3) and set cores to phase 1
            ids = multiInclusions_grains.getAllIdentifiers()
            multiInclusions_grains.addLayer(ids, delta_phase, delta)
            multiInclusions_grains.changePhase(ids, [1 for _ in ids])

            # 3. SINGLE OVERLAY: pores on grains
            dictionnaire = {incl_phase: grains_phase, delta_phase: grains_phase}
            structure = merope.Structure_3D(
                multiInclusions_pores,
                multiInclusions_grains,
                dictionnaire
            )

            # 4. Voxellate
            with pm.cd(str(case_dir)):
                fractions = builder.voxellate(structure, K_THERMAL)

            # Extract phase fractions
            phi_solid = fractions.get(0, 0.0)
            phi_pores = fractions.get(2, 0.0)
            p_real = phi_pores

            print(f"   [Iter {iteration}] delta={delta:.2f}, inclPhi={inclPhi:.3f} -> pores={p_real:.4f} (target={p_target:.2f})")

            # Check convergence (within 2% of target)
            if abs(p_real - p_target) < 0.02:
                print(f"   ✓ Converged!")
                break

            # Adjust inclPhi for next iteration
            error = p_target - p_real
            inclPhi += error * 0.7  # Moderate adjustment factor
            inclPhi = float(np.clip(inclPhi, 0.01, 0.9))

        except RuntimeError as e:
            print(f"   [Iter {iteration}] Packing failed: {e}")
            inclPhi *= 0.8  # Reduce if packing fails
            inclPhi = max(0.01, inclPhi)
            if iteration > 6:
                print(f"   ⚠ Accepting p_real={p_real:.4f} after {iteration} iterations")
                break

    # Run solver
    with pm.cd(str(case_dir)):
        if no_solver:
            res = {"Kmean": 0.0}
        else:
            res = solver.solve()

    k_eff = res["Kmean"]
    print(f" [DONE] P_target={p_target:.2f} | delta={delta:.2f} | P_real={p_real:.4f} -> K_eff={k_eff:.4f}")

    return {
        "Target_P": p_target,
        "Delta": delta,
        "Grain_R": LAG_R,
        "Real_P": p_real,
        "K_eff": k_eff,
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
