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
L_DIM = [10.0, 10.0, 10.0]     # RVE size (physical units)
N_VOX = 250
K_THERMAL = [1.0, 1.0, 1e-3]   # Phase 0=Solid, 1=Solid, 2=Pore

# Parameters
INCL_R = 0.3      # Pore radius
LAG_R = 1.0       # Laguerre grain size
LAG_PHI = 1.0     # Fill entire RVE

# Delta range
DELTA_VALUES = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8]

# Porosity targets
P_TARGETS = [0.1, 0.2, 0.3]
OUTPUT_DIR = Path("Results_Keff_vs_Delta")

# N_CPUS
N_CPUS = 12

def worker(task_args):
    p_target, delta, no_solver = task_args

    builder = MicrostructureBuilder(L=L_DIM, n3D=N_VOX, seed=42)
    solver = ThermalSolver(n_cpus=N_CPUS)
    pm = ProjectManager()

    case_dir = OUTPUT_DIR / f"P_{p_target:.2f}_Delta_{delta:.3f}"

    # Phase IDs
    incl_phase = 2      # Pores
    delta_phase = 3     # Grain boundary layer (temporary)
    grains_phase = 0    # Grains (final solid phase)

    # Domain size
    domain_size = L_DIM
    lagRphi = [LAG_R, LAG_PHI]

    # Initial guess for inclPhi (adaptive formula based on delta)
    # For small delta (crack-like), need higher inclPhi to reach target porosity
    # For large delta (sphere-like), need lower inclPhi
    # Empirical formula: inclPhi ~ p_target * (base + decay/delta)
    base = 1.0
    decay = 3.0
    inclPhi = p_target * (base + decay / max(delta, 0.2))
    inclPhi = float(np.clip(inclPhi, 0.01, 0.85))  # Clamp to valid range immediately

    p_real = 0.0

    # Iterative convergence loop (max 20 iterations, tighter tolerance)
    MAX_ITER = 20
    TOLERANCE = 0.01  # ±1% (tighter than before)

    for iteration in range(MAX_ITER):
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

            # Add grain-boundary layer (phase 3) and set cores to phase 1
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
            phi_pores = fractions.get(2, 0.0)
            p_real = phi_pores

            print(f"   [Iter {iteration}] delta={delta:.2f}, inclPhi={inclPhi:.3f} -> pores={p_real:.4f} (target={p_target:.2f})")

            # Check convergence (using TOLERANCE = ±1%)
            if abs(p_real - p_target) < TOLERANCE:
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
            if iteration > 15:  # Increased from 6 to allow more attempts
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

def extract_results_from_folders():
    """
    Extract results from existing case folders and rebuild CSV.
    Scans all P_X.XX_Delta_X.XXX folders and reads their results.
    """
    print(f"[EXTRACT] Scanning {OUTPUT_DIR} for existing case folders...")

    if not OUTPUT_DIR.exists():
        print(f"[EXTRACT] Output directory does not exist: {OUTPUT_DIR}")
        return None

    rows = []
    case_dirs = sorted(OUTPUT_DIR.glob("P_*_Delta_*"))

    for case_dir in case_dirs:
        try:
            # Parse folder name: P_0.10_Delta_0.390
            folder_name = case_dir.name
            parts = folder_name.split('_')
            p_target = float(parts[1])
            delta = float(parts[3])

            # Read K_eff from thermalCoeff_amitex.txt
            coeff_file = case_dir / "thermalCoeff_amitex.txt"
            if not coeff_file.exists():
                print(f"  [SKIP] {folder_name}: missing thermalCoeff_amitex.txt")
                continue

            with open(coeff_file, 'r') as f:
                lines = f.readlines()
                # Format: K_xx K_xy K_xz (first line)
                k_values = lines[0].strip().split()
                k_eff = float(k_values[0])  # K_xx

            # Read Real_P from Coeffs.txt phase fractions
            coeffs_file = case_dir / "Coeffs.txt"
            p_real = p_target  # Default
            if coeffs_file.exists():
                with open(coeffs_file, 'r') as f:
                    for line in f:
                        if 'Phase 2' in line or 'phase 2' in line:
                            # Try to extract porosity from phase fraction line
                            parts = line.split()
                            for i, part in enumerate(parts):
                                try:
                                    val = float(part)
                                    if 0.0 < val < 1.0:  # Likely a fraction
                                        p_real = val
                                        break
                                except:
                                    continue

            rows.append({
                "Target_P": p_target,
                "Delta": delta,
                "Grain_R": LAG_R,
                "Real_P": p_real,
                "K_eff": k_eff,
            })
            print(f"  ✓ {folder_name}: K_eff={k_eff:.4f}, P_real={p_real:.4f}")

        except Exception as e:
            print(f"  [ERROR] {case_dir.name}: {e}")
            continue

    if not rows:
        print("[EXTRACT] No valid results found")
        return None

    df = pd.DataFrame(rows)
    df = df.sort_values(by=["Target_P", "Delta"]).reset_index(drop=True)

    csv_path = OUTPUT_DIR / "keff_vs_delta.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n[EXTRACT] ✓ Extracted {len(rows)} cases")
    print(f"[EXTRACT] ✓ Saved to {csv_path}")

    return df

def solve_existing_cases():
    """Run Amitex solver on all existing case folders that have structure.vtk."""
    print(f"[AMITEX] Scanning {OUTPUT_DIR} for existing geometries...")

    if not OUTPUT_DIR.exists():
        print(f"[AMITEX] Output directory does not exist: {OUTPUT_DIR}")
        return None

    solver = ThermalSolver(n_cpus=N_CPUS)
    pm = ProjectManager()
    rows = []

    case_dirs = sorted(OUTPUT_DIR.glob("P_*_Delta_*"))
    for case_dir in case_dirs:
        vtk_file = case_dir / "structure.vtk"
        if not vtk_file.exists():
            print(f"  [SKIP] {case_dir.name}: no structure.vtk")
            continue

        # Parse folder name: P_0.10_Delta_0.390
        parts = case_dir.name.split('_')
        p_target = float(parts[1])
        delta = float(parts[3])

        print(f"  Solving {case_dir.name}...")
        with pm.cd(str(case_dir)):
            res = solver.solve()

        k_eff = res["Kmean"]

        # Read real porosity from Coeffs.txt if available
        p_real = p_target
        coeffs_file = case_dir / "Coeffs.txt"
        if coeffs_file.exists():
            with open(coeffs_file, 'r') as f:
                for line in f:
                    if 'Phase 2' in line or 'phase 2' in line:
                        for part in line.split():
                            try:
                                val = float(part)
                                if 0.0 < val < 1.0:
                                    p_real = val
                                    break
                            except ValueError:
                                continue

        print(f"  [DONE] {case_dir.name}: K_eff={k_eff:.4f}")
        rows.append({
            "Target_P": p_target,
            "Delta": delta,
            "Grain_R": LAG_R,
            "Real_P": p_real,
            "K_eff": k_eff,
        })

    if not rows:
        print("[SOLVE-ONLY] No valid cases found")
        return None

    df = pd.DataFrame(rows)
    df = df.sort_values(by=["Target_P", "Delta"]).reset_index(drop=True)
    csv_path = OUTPUT_DIR / "keff_vs_delta.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n[SOLVE-ONLY] Solved {len(rows)} cases, saved to {csv_path}")
    return df


def run_sweeps(no_solver=False, recover=False):
    pm = ProjectManager()
    csv_path = OUTPUT_DIR / "keff_vs_delta.csv"

    # Only cleanup if NOT recovering
    if not recover:
        pm.cleanup_folder(str(OUTPUT_DIR))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load existing results if recovering
    existing_df = None
    if recover and (OUTPUT_DIR / "keff_vs_delta.csv").exists():
        existing_df = pd.read_csv(OUTPUT_DIR / "keff_vs_delta.csv")
        print(f"[RECOVER] Found {len(existing_df)} existing cases in CSV")
    else:
        print(f"[DEBUG] NOT loading CSV: recover={recover}, exists={csv_path.exists()}")

    # Generate all tasks
    all_tasks = [(p, delta, no_solver) for p in P_TARGETS for delta in DELTA_VALUES]

    # Filter out completed cases if recovering (use tolerance for float comparison)
    if recover and existing_df is not None:
        def is_completed(p, delta):
            """Check if (p, delta) exists in existing_df with tolerance."""
            match = existing_df[
                (np.abs(existing_df["Target_P"] - p) < 1e-6) &
                (np.abs(existing_df["Delta"] - delta) < 1e-6)
            ]
            return len(match) > 0

        tasks = [t for t in all_tasks if not is_completed(t[0], t[1])]
        print(f"[RECOVER] Skipping {len(all_tasks) - len(tasks)} completed cases")
        print(f"[RECOVER] Running {len(tasks)} new cases")
    else:
        tasks = all_tasks

    print(f"=== K_eff vs Delta Sweep ({len(tasks)} tasks) ===")

    # Run only missing tasks
    rows = []
    for t in tasks:
        rows.append(worker(t))

    # Combine with existing data if recovering
    if recover and existing_df is not None:
        new_df = pd.DataFrame(rows)
        df = pd.concat([existing_df, new_df], ignore_index=True)
        print(f"[RECOVER] Combined {len(existing_df)} old + {len(new_df)} new = {len(df)} total cases")
    else:
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
    parser.add_argument("--plot-only", action="store_true", help="Only regenerate plots from existing CSV")
    parser.add_argument("--no-solver", action="store_true", help="Skip Amitex solver (geometry only)")
    parser.add_argument("--recover", action="store_true", help="Resume from existing results (skip completed cases)")
    parser.add_argument("--extract", action="store_true", help="Extract results from existing folders and rebuild CSV")
    parser.add_argument("--amitex", action="store_true", help="Run Amitex solver on existing geometry (skip generation)")
    args = parser.parse_args()

    if args.amitex:
        df = solve_existing_cases()
        if df is None:
            print("[ERROR] No cases solved")
            sys.exit(1)
    elif args.extract:
        # Extract mode: scan folders and rebuild CSV
        df = extract_results_from_folders()
        if df is None:
            print("[ERROR] Failed to extract results")
            sys.exit(1)
    elif args.plot_only:
        # Plot-only mode: read existing CSV
        df = pd.read_csv(OUTPUT_DIR / "keff_vs_delta.csv")
    else:
        # Normal/recover mode: run simulations
        df = run_sweeps(no_solver=args.no_solver, recover=args.recover)

    plot_slide(df, OUTPUT_DIR)
