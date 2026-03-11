import sys
from pathlib import Path
import argparse
import os

# Ensure project_root/ is on sys.path so `core` is importable
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import concurrent.futures

from core.geometry import MicrostructureBuilder
from core.solver import ThermalSolver
from core.utils import ProjectManager

# --- CONFIGURAZIONE ---
L_DIM = [10.0, 10.0, 10.0]
N_VOX = 120 # Number of voxels per side

# SPHERE_R_VALUES = [0.2, 0.4, 0.5, 0.6, 0.8] # Radii of spherical pores to test
SPHERE_R_VALUES = [0.4, 0.6]

# Range of porosity to test (5% - 25%)
PHI_VALUES = [0.05, 0.20]

# Thermal properties (convention: 0 = matrix, 2 = pores)
K_MAT = 1.0
K_PORE = 0.001
K_THERMAL = [K_MAT, K_MAT, K_PORE]


def maxwell_eucken(phi, k_m: float = 1.0, k_p: float = 0.001):
    """Theoretical Maxwell–Eucken model for isolated spherical pores."""
    beta = (k_p - k_m) / (k_p + 2.0 * k_m)
    return k_m * (1.0 + 2.0 * beta * phi) / (1.0 - beta * phi)


def worker(task_args):
    r_sphere, phi_target, output_dir, no_solver = task_args
    pm = ProjectManager()
    builder = MicrostructureBuilder(L=L_DIM, n3D=N_VOX, seed=42)
    solver = ThermalSolver(n_cpus=2) # Keep per-simulation CPU to 2

    import merope
    # 1. Generate structure: homogeneous matrix + distributed spherical pores
    multi = builder.generate_spheres([[r_sphere, phi_target]], phase_id=2)
    struct = merope.Structure_3D(multi)

    # Geometric/resolution parameters for this case
    L_RVE = float(builder.L[0])  # cubic RVE
    n_vox = float(builder.n3D)
    L_voxel = L_RVE / n_vox
    ratio_LR = L_RVE / float(r_sphere)          # representativity: L_RVE / R_pore
    ratio_Rlvox = float(r_sphere) / L_voxel     # resolution: R_pore / L_voxel

    # 2. Setup case folder (absolute path)
    sub_dir = output_dir.resolve() / f"R_{r_sphere}" / f"Phi_{phi_target}"
    sub_dir.mkdir(parents=True, exist_ok=True)
    abs_sub_dir = str(sub_dir)

    try:
        # 3. Voxelization (passing absolute paths for outputs)
        fractions = builder.voxellate(
            struct, 
            K_THERMAL, 
            vtk_path=os.path.join(abs_sub_dir, "structure.vtk"),
            coeffs_path=os.path.join(abs_sub_dir, "Coeffs.txt")
        )

        # 4. AMITEX Solver (internally handles chdir to the vtk file directory)
        if no_solver:
            res = {"Kmean": 0.0}
        else:
            # solver.solve() expects to find the .vtk in the work_dir
            # ThermalSolver.solve handles the chdir safely
            res = solver.solve(vtk_path=os.path.join(abs_sub_dir, "structure.vtk"))

        # 5. Data collection
        phi_real = fractions.get(2, 0.0)
        k_eff = res["Kmean"]
        k_theory = maxwell_eucken(phi_real, K_MAT, K_PORE)
        error_perc = abs(k_eff - k_theory) / k_theory * 100.0 if k_theory > 0 else 0.0

        warning = ""
        if ratio_Rlvox < 5.0:
            warning = f" [WARN: R/l_vox={ratio_Rlvox:.1f}<5]"

        print(
            f" [DONE] R={r_sphere} | Target_Phi={phi_target:.4f} | Real_Phi={phi_real:.4f} | "
            f"K_Sim={k_eff:.4f} | K_Max={k_theory:.4f} | Err={error_perc:.2f}%{warning}"
        )

        return {
            "Phi_Requested": phi_target,
            "Phi_Real": phi_real,
            "K_Simulation": k_eff,
            "K_Maxwell": k_theory,
            "Ratio_LR": ratio_LR,
            "Ratio_Rlvox": ratio_Rlvox,
            "R_pore": float(r_sphere),
        }

    except Exception as e:
        print(f"Error during case R={r_sphere} Phi={phi_target}: {e}")
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Run distributed porosity simulations or plot existing results.")
    parser.add_argument("--plot-only", action="store_true", help="Plot validation results without running simulations.")
    parser.add_argument("--no-solver", action="store_true", help="Skip Amitex solver (generate geometry only).")
    args = parser.parse_args()

    output_dir = Path("Results_Distributed_Validation")
    csv_path = output_dir / "validation_results.csv"

    # --- PLOTTING ONLY MODE ---
    if args.plot_only:
        if not csv_path.exists():
            print(f"Results file not found: {csv_path}")
            return
        print(f"Loading results from: {csv_path}")
        df = pd.read_csv(csv_path)

    # --- SIMULATION MODE (standard) ---
    else:
        pm = ProjectManager()
        pm.cleanup_folder(str(output_dir))
        output_dir.mkdir(parents=True, exist_ok=True)

        print("=== VALIDATION OF DISTRIBUTED POROSITY (SPHERES ONLY, PARALLEL) ===")
        tasks = [(r, p, output_dir, args.no_solver) for r in SPHERE_R_VALUES for p in PHI_VALUES]

        # ProcessPoolExecutor scales well across heavy independent solvers
        with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
            raw_results = list(executor.map(worker, tasks))

        results_list = [r for r in raw_results if r is not None]

        # --- SAVE RESULTS ---
        if not results_list:
            print("No results.")
            return

        df = pd.DataFrame(results_list)
        df = df.sort_values(by=["R_pore", "Phi_Requested"]).reset_index(drop=True)
        df.to_csv(csv_path, index=False)
        print(f"\nResults table saved to: {csv_path}")
    
    # --- PLOT 1: K_eff vs Phi_Real ---
    plt.figure(figsize=(8, 6))

    # 1. Theoretical line (Maxwell) over the entire range
    x_theory = np.linspace(0.0, df["Phi_Real"].max() * 1.1, 50)
    y_theory = maxwell_eucken(x_theory, K_MAT, K_PORE)
    plt.plot(x_theory, y_theory, "k-", linewidth=1.5, label="Maxwell (Teorico)")

    # 2. Simulated points
    plt.plot(
        df["Phi_Real"],
        df["K_Simulation"],
        "bo",
        markersize=8,
        label="Mérope + Amitex (Simulazione)",
    )

    # Formatting
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.xlabel("Porosità (Frazione di Volume)")
    plt.ylabel("Conduttività Termica Efficace (K_eff)")
    plt.title("Validazione: Porosità Distribuita vs Maxwell–Eucken")
    plt.legend()

    # Save plot
    img_path = output_dir / "validation_plot.png"
    plt.savefig(img_path, dpi=300)
    plt.close()
    print(f"Plot saved to: {img_path}")

    # --- PLOT 2: K_eff vs L_RVE / R_pore ---
    plt.figure(figsize=(8, 6))
    plt.plot(
        df["Ratio_LR"],
        df["K_Simulation"],
        "bo",
        markersize=8,
        label="Mérope + Amitex (Simulazione)",
    )
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.xlabel("Rapporto L_RVE / R_pore")
    plt.ylabel("Conduttività Termica Efficace (K_eff)")
    plt.title("K_eff vs Rappresentatività (L_RVE / R_pore)")
    plt.legend()
    img_path = output_dir / "K_eff_vs_L_RVE_over_R_pore.png"
    plt.savefig(img_path, dpi=300)
    plt.close()
    print(f"Plot saved to: {img_path}")

    # --- PLOT 3: K_eff vs R_pore / L_voxel ---
    plt.figure(figsize=(8, 6))
    plt.plot(
        df["Ratio_Rlvox"],
        df["K_Simulation"],
        "ro",
        markersize=8,
        label="Mérope + Amitex (Simulazione)",
    )
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.xlabel("Rapporto R_pore / L_voxel")
    plt.ylabel("Conduttività Termica Efficace (K_eff)")
    plt.title("K_eff vs Risoluzione del Poro (R_pore / L_voxel)")
    plt.legend()
    img_path = output_dir / "K_eff_vs_R_pore_over_L_voxel.png"
    plt.savefig(img_path, dpi=300)
    plt.close()
    print(f"Plot saved to: {img_path}")

    # --- PLOT 4: K_eff vs R_pore ---
    plt.figure(figsize=(8, 6))
    # Different colors for different porosities
    scatter = plt.scatter(
        df["R_pore"],
        df["K_Simulation"],
        c=df["Phi_Requested"],
        s=100,
        cmap="viridis",
        alpha=0.7,
        edgecolors="black",
        linewidths=1,
    )
    # Colorbar to show the porosity
    cbar = plt.colorbar(scatter)
    cbar.set_label("Target Porosity (φ)", rotation=270, labelpad=20)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.xlabel("Pore Radius (R_pore)")
    plt.ylabel("Effective Thermal Conductivity (K_eff)")
    plt.title("K_eff vs Pore Radius")
    img_path = output_dir / "K_eff_vs_R_pore.png"
    plt.savefig(img_path, dpi=300)
    plt.close()
    print(f"Plot saved to: {img_path}")

if __name__ == "__main__":
    main()