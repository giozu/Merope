import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

# Ensure project_root/ is on sys.path so `core` is importable
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from core.geometry import MicrostructureBuilder
from core.solver import ThermalSolver
from core.utils import ProjectManager

# --- CONFIGURAZIONE ---
L_DIM = [10, 10, 10]
N_VOX = 120            # Risoluzione (120 è un buon compromesso velocità/dettaglio)
GRAIN_R = 2.5          # Raggio medio grani
DELTA_GB = 0.2         # Spessore bordi grano (Rete interconnessa fissa)
SPHERE_R = 0.5         # Raggio pori sferici

# Valori di porosità sferica da testare (dal 5 c% al 20%)
PHI_INTRA_VALUES = [0.05, 0.10, 0.15, 0.20]

# Proprietà Termiche: [Ignorato, Matrice, Poro]
K_MAT = 1.0
K_PORE = 0.001
K_THERMAL = [0.0, K_MAT, K_PORE]

def maxwell_eucken(phi, k_m=1.0, k_p=0.0):
    """Calcola la K teorica per pori sferici isolati (Maxwell)."""
    # Beta factor per sfere isolanti
    beta = (k_p - k_m) / (k_p + 2*k_m)
    return k_m * (1 + 2*beta*phi) / (1 - beta*phi)

def main():
    # Setup
    pm = ProjectManager()
    builder = MicrostructureBuilder(L=L_DIM, n3D=N_VOX, seed=42)
    solver = ThermalSolver(n_cpus=4)
    
    # Cartella Output
    output_dir = "Results_Thesis_Graph"
    pm.cleanup_folder(output_dir)
    
    results_list = []

    print("=== AVVIO GENERAZIONE GRAFICO TESI ===")
    print(f"Risoluzione: {N_VOX}^3 | Delta Bordi: {DELTA_GB}")

    # --- CICLO DI SIMULAZIONE ---
    for phi_target in PHI_INTRA_VALUES:
        print(f"\n--- Simulating Intra-granular Phi: {phi_target*100:.1f}% ---")
        
        # 1. Genera Struttura Mista
        # Nota: Se phi_target è 0, passiamo lista vuota per evitare errori
        pores = [[SPHERE_R, phi_target]] if phi_target > 0 else []
        
        struct = builder.generate_mixed_structure(
            grain_radius=GRAIN_R, 
            delta=DELTA_GB, 
            intra_pore_list=pores
        )
        
        # 2. Voxelizzazione
        # 2. Voxelization
        # Setup case folder (absolute path)
        case_dir = Path(output_dir).resolve() / f"Phi_{phi_target}"
        case_dir.mkdir(parents=True, exist_ok=True)
        abs_case_dir = str(case_dir)
        
        try:
            # 3. Voxelization (passing absolute paths for outputs)
            fractions = builder.voxellate(
                struct, 
                K_THERMAL, 
                vtk_path=case_dir / "structure.vtk",
                coeffs_path=case_dir / "Coeffs.txt"
            )
            
            # 4. Solver AMITEX (handles chdir internally)
            res = solver.solve(vtk_path=os.path.join(abs_case_dir, "structure.vtk"))
            
            # 5. Raccogli Dati
            phi_real_tot = fractions.get(2, 0.0) # Porosità totale (Sfere + Bordi)
            k_eff = res['Kmean']
            
            print(f"   -> Phi Totale: {phi_real_tot:.4f} | K Effettivo: {k_eff:.4f}")
            
            results_list.append({
                "Target_Intra": phi_target,
                "Phi_Total": phi_real_tot,
                "K_Simulation": k_eff,
                "K_Maxwell": maxwell_eucken(phi_real_tot, K_MAT, K_PORE) # Teorico a pari porosità
            })
            
        except Exception as e:
            print(f"Errore durante la simulazione: {e}")

    # --- SALVATAGGIO E PLOT ---
    df = pd.DataFrame(results_list)
    csv_path = f"{output_dir}/results_table.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nDati salvati in: {csv_path}")

    # Generazione Grafico
    plt.figure(figsize=(8, 6))
    
    # 1. Curva Teorica (Maxwell - Solo Sfere)
    # Creiamo una linea continua per il riferimento
    x_theory = np.linspace(0, df["Phi_Total"].max() * 1.1, 50)
    y_theory = maxwell_eucken(x_theory, K_MAT, K_PORE)
    plt.plot(x_theory, y_theory, 'k--', label="Maxwell (Solo Sfere - Teorico)")

    # 2. I tuoi punti (Simulazione Mista)
    plt.plot(df["Phi_Total"], df["K_Simulation"], 'ro-', markersize=8, linewidth=2, label="Mérope (Mista: Sfere + Bordi)")

    # Decorazioni
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.xlabel("Porosità Totale (Frazione di Volume)")
    plt.ylabel("Conduttività Termica Efficace (K_eff / K_mat)")
    plt.title(f"Effetto della Porosità Interconnessa (Delta={DELTA_GB})")
    plt.legend()
    
    # Salva
    plot_path = f"{output_dir}/Thesis_Graph_K_vs_Phi.png"
    plt.savefig(plot_path, dpi=300)
    print(f"Grafico salvato in: {plot_path}")
    print("=== FINITO ===")

if __name__ == "__main__":
    main()