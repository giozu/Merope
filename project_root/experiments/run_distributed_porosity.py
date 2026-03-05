import sys
from pathlib import Path
import argparse

# Ensure project_root/ is on sys.path so `core` is importable
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run distributed porosity simulations or plot existing results.")
    parser.add_argument("--plot-only", action="store_true", help="Plot validation results without running simulations.")
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
        builder = MicrostructureBuilder(L=L_DIM, n3D=N_VOX, seed=42)
        solver = ThermalSolver(n_cpus=4)

        pm.cleanup_folder(str(output_dir))

        results_list = []

        print("=== VALIDATION OF DISTRIBUTED POROSITY (SPHERES ONLY) ===")

        for r_sphere in SPHERE_R_VALUES:
            print(f"\n# === Simulating Sphere Radius: {r_sphere} ===")

            for phi_target in PHI_VALUES:
                print(f"\n--- Simulating Sphere Porosity: {phi_target * 100:.1f}% (R={r_sphere}) ---")

                # 1. Generate structure: homogeneous matrix + distributed spherical pores
                multi = builder.generate_spheres([[r_sphere, phi_target]], phase_id=2)
                # Phase 0 = matrix, phase 2 = pores; consistent with K_THERMAL
                import merope

                struct = merope.Structure_3D(multi)

                # Geometric/resolution parameters for this case
                L_RVE = float(builder.L[0])  # cubic RVE
                n_vox = float(builder.n3D)
                L_voxel = L_RVE / n_vox
                ratio_LR = L_RVE / float(r_sphere)          # representativity: L_RVE / R_pore
                ratio_Rlvox = float(r_sphere) / L_voxel     # resolution: R_pore / L_voxel

                # 2. Setup case folder and change directory with ProjectManager
                # Structure: Results/R_0.5/Phi_0.10
                sub_dir = output_dir / f"R_{r_sphere}" / f"Phi_{phi_target}"
                with pm.cd(str(sub_dir)):
                    try:
                        # 3. Voxelization (writes structure.vtk + Coeffs.txt in the case folder)
                        fractions = builder.voxellate(struct, K_THERMAL)

                        # 4. AMITEX Solver (uses structure.vtk in the CWD)
                        res = solver.solve()

                        # 5. Data collection
                        phi_real = fractions.get(2, 0.0)
                        k_eff = res["Kmean"]
                        k_theory = maxwell_eucken(phi_real, K_MAT, K_PORE)
                        error_perc = abs(k_eff - k_theory) / k_theory * 100.0

                        # Quality control on pore resolution
                        if ratio_Rlvox < 5.0:
                            print(
                                f"   [WARNING] R_pore/L_voxel = {ratio_Rlvox:.2f} < 5. "
                                "Pore shape is under-resolved."
                            )

                        # Compact final print for each iteration
                        print(
                            "   Target: {t:.4f} | Real: {pr:.4f} | "
                            "K_Sim: {ks:.4f} | K_Max: {km:.4f} | R/l_vox: {rlv:.2f}".format(
                                t=phi_target,
                                pr=phi_real,
                                ks=k_eff,
                                km=k_theory,
                                rlv=ratio_Rlvox,
                            )
                        )
                        print(f"   -> Error vs Analytical model (Maxwell): {error_perc:.2f}%")

                        results_list.append(
                            {
                                "Phi_Requested": phi_target,
                                "Phi_Real": phi_real,
                                "K_Simulation": k_eff,
                                "K_Maxwell": k_theory,
                                "Ratio_LR": ratio_LR,
                                "Ratio_Rlvox": ratio_Rlvox,
                                "R_pore": float(r_sphere),
                            }
                        )

                    except Exception as e:
                        print(f"Error during case Phi={phi_target}: {e}")

        # --- SAVE RESULTS ---
        if not results_list:
            print("No results.")
            return

        output_dir.mkdir(parents=True, exist_ok=True)

        df = pd.DataFrame(results_list)
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