from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from core.geometry import MicrostructureBuilder
from core.solver import ThermalSolver
from core.utils import ProjectManager

# --- CONFIGURAZIONE ---
L_DIM = [10.0, 10.0, 10.0]
N_VOX = 120            # Risoluzione (compromesso accuratezza/costo)
SPHERE_R = 0.5         # Raggio pori sferici

# Range di porosità da testare (5% - 25%)
PHI_VALUES = [0.05, 0.10, 0.15, 0.20, 0.25]

# Proprietà termiche (convenzione: 0 = matrice, 2 = pori)
K_MAT = 1.0
K_PORE = 0.001
K_THERMAL = [K_MAT, K_MAT, K_PORE]


def maxwell_eucken(phi, k_m: float = 1.0, k_p: float = 0.001):
    """Modello teorico di Maxwell–Eucken per pori sferici isolati."""
    beta = (k_p - k_m) / (k_p + 2.0 * k_m)
    return k_m * (1.0 + 2.0 * beta * phi) / (1.0 - beta * phi)


def main() -> None:
    pm = ProjectManager()
    builder = MicrostructureBuilder(L=L_DIM, n3D=N_VOX, seed=42)
    solver = ThermalSolver(n_cpus=4)

    output_dir = Path("Results_Distributed_Validation")
    pm.cleanup_folder(str(output_dir))

    results_list = []

    print("=== VALIDAZIONE POROSITÀ DISTRIBUITA (SOLO SFERE) ===")

    for phi_target in PHI_VALUES:
        print(f"\n--- Simulating Sphere Porosity: {phi_target * 100:.1f}% ---")

        # 1. Genera struttura: matrice omogenea + sfere porose distribuite
        multi = builder.generate_spheres([[SPHERE_R, phi_target]], phase_id=2)
        # Fase 0 = matrice, fase 2 = pori; coerente con K_THERMAL
        import merope  # import locale per evitare dipendenze inutili lato modulo

        struct = merope.Structure_3D(multi)

        # Parametri geometrici/di risoluzione per questo caso
        L_RVE = float(builder.L[0])  # assumiamo RVE cubico
        n_vox = float(builder.n3D)
        L_voxel = L_RVE / n_vox
        ratio_LR = L_RVE / float(SPHERE_R)          # rappresentatività: L_RVE / R_pore
        ratio_Rlvox = float(SPHERE_R) / L_voxel     # risoluzione: R_pore / L_voxel

        # 2. Setup cartella caso e cambio directory con ProjectManager
        sub_dir = output_dir / f"Phi_{phi_target}"
        with pm.cd(str(sub_dir)):
            try:
                # 3. Voxelizzazione (scrive structure.vtk + Coeffs.txt nella cartella caso)
                fractions = builder.voxellate(struct, K_THERMAL)

                # 4. Solver AMITEX (usa structure.vtk nel CWD)
                res = solver.solve()

                # 5. Raccolta dati
                phi_real = fractions.get(2, 0.0)
                k_eff = res["Kmean"]
                k_theory = maxwell_eucken(phi_real, K_MAT, K_PORE)
                error_perc = abs(k_eff - k_theory) / k_theory * 100.0

                # Controllo di qualità sulla risoluzione dei pori
                if ratio_Rlvox < 5.0:
                    print(
                        f"   [WARNING] R_pore/L_voxel = {ratio_Rlvox:.2f} < 5. "
                        "La forma del poro è sotto-risolta."
                    )

                # Stampa finale compatta per ogni iterazione
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
                print(f"   -> Errore rispetto a Maxwell: {error_perc:.2f}%")

                results_list.append(
                    {
                        "Phi_Requested": phi_target,
                        "Phi_Real": phi_real,
                        "K_Simulation": k_eff,
                        "K_Maxwell": k_theory,
                        "Ratio_LR": ratio_LR,
                        "Ratio_Rlvox": ratio_Rlvox,
                        "R_pore": float(SPHERE_R),
                    }
                )

            except Exception as e:  # pragma: no cover - difensivo
                print(f"Errore durante il caso Phi={phi_target}: {e}")

    # --- PLOT RISULTATI ---
    if not results_list:
        print("Nessun risultato.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(results_list)
    csv_path = output_dir / "validation_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nTabella risultati salvata in: {csv_path}")
    
    # --- PLOT 1: K_eff vs Phi_Real ---
    plt.figure(figsize=(8, 6))

    # 1. Linea teorica (Maxwell) su tutto il range
    x_theory = np.linspace(0.0, df["Phi_Real"].max() * 1.1, 50)
    y_theory = maxwell_eucken(x_theory, K_MAT, K_PORE)
    plt.plot(x_theory, y_theory, "k-", linewidth=1.5, label="Maxwell (Teorico)")

    # 2. Punti simulati
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

    # Salva grafico
    img_path = output_dir / "validation_plot.png"
    plt.savefig(img_path, dpi=300)
    plt.close()
    print(f"Grafico salvato in: {img_path}")

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
    print(f"Grafico salvato in: {img_path}")

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
    print(f"Grafico salvato in: {img_path}")

    # --- PLOT 4: K_eff vs R_pore (per diverse porosità) ---
    plt.figure(figsize=(8, 6))
    # Colori diversi per diverse porosità
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
    # Aggiungi colorbar per mostrare la porosità
    cbar = plt.colorbar(scatter)
    cbar.set_label("Porosità Target (φ)", rotation=270, labelpad=20)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.xlabel("Raggio del Poro (R_pore)")
    plt.ylabel("Conduttività Termica Efficace (K_eff)")
    plt.title(f"K_eff vs Raggio del Poro (R_pore = {SPHERE_R:.2f} costante)")
    img_path = output_dir / "K_eff_vs_R_pore.png"
    plt.savefig(img_path, dpi=300)
    plt.close()
    print(f"Grafico salvato in: {img_path}")

if __name__ == "__main__":
    main()