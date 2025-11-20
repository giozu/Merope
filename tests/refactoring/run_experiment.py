import numpy as np
import os
from merope_engine import MeropeSim

# --- CONFIGURAZIONE GEOMETRICA ---
L_SIDE = [20, 20, 20]  # Come nel tuo script Delta Scan
N_VOXEL = 200          # Risoluzione aumentata come richiesto
LAG_R = 3.0            # Dimensione grani
LAG_PHI = 1            # 100% grani

# Raggi pori
INCL_R_INTER = 0.5     # Raggio pori sui bordi
INCL_R_INTRA = 0.2     # Raggio pori dentro i grani

# --- PARAMETRI DI STUDIO ---
SEEDS = [0] # Puoi metterne di più per statistica
TARGET_POROSITIES = [0.1, 0.2, 0.3]
DELTA_RATIOS = np.linspace(0.1, 0.9, 10) # Da 10% a 90% del raggio grano

# --- INIZIALIZZAZIONE ---
sim = MeropeSim(work_dir="DeltaScan_Final_Run")
output_file = "delta_scan_results_modular.txt"

# Header compatibile col tuo plotter
with open(output_file, "w") as f:
    f.write("p_target\tp_por_meas\tp_delta\tp_intra\tdelta_ratio\tdelta_phys\tKxx\tKyy\tKzz\tKmean\n")

print(f"--- CONFIGURAZIONE ---")
print(f"RVE: {L_SIDE}, Voxel: {N_VOXEL}")
print(f"Pori Inter R: {INCL_R_INTER}, Pori Intra R: {INCL_R_INTRA}")
print(f"Casi totali: {len(TARGET_POROSITIES) * len(DELTA_RATIOS) * len(SEEDS)}")

# --- LOOP PRINCIPALE ---
for p_tot in TARGET_POROSITIES:
    for ratio in DELTA_RATIOS:
        
        # 1. Calcolo Parametri Fisici (LA TUA LOGICA)
        delta_phys = ratio * LAG_R
        
        # Ripartizione porosità: 
        # Basso ratio -> Delta piccolo -> Pori concentrati sul bordo (p_delta alto)
        # Alto ratio -> Delta grande -> Pori spostati dentro (p_intra alto)?
        # NOTA: Nel tuo script era: p_delta = p_tot * (1 - ratio). Mantengo quella.
        p_delta = p_tot * (1.0 - ratio)
        p_intra = p_tot - p_delta
        
        for seed in SEEDS:
            print(f"\n>>> P_tot={p_tot:.2f}, Ratio={ratio:.2f} (Pd={p_delta:.3f}, Pi={p_intra:.3f})")
            
            try:
                # Chiamata all'Engine (metodo nuovo)
                real_por, k_mean, raw_matrix = sim.run_case_delta_scan(
                    n3D=N_VOXEL,
                    L=L_SIDE,
                    seed=seed,
                    lagR=LAG_R,
                    lagPhi=LAG_PHI,
                    inclR=INCL_R_INTER,
                    p_delta=p_delta,
                    incl2R=INCL_R_INTRA,
                    p_intra=p_intra,
                    delta_phys=delta_phys
                )
                
                # Estrazione Tensore
                if raw_matrix:
                    k_xx, k_yy, k_zz = raw_matrix[0][0], raw_matrix[1][1], raw_matrix[2][2]
                else:
                    k_xx, k_yy, k_zz = 0, 0, 0
                
                # Salvataggio
                with open(output_file, "a") as f:
                    f.write(f"{p_tot:.4f}\t{real_por:.4f}\t{p_delta:.4f}\t{p_intra:.4f}\t"
                            f"{ratio:.4f}\t{delta_phys:.4f}\t"
                            f"{k_xx:.6f}\t{k_yy:.6f}\t{k_zz:.6f}\t{k_mean:.6f}\n")
                            
            except Exception as e:
                print(f"❌ Errore critico: {e}")

print("\nSimulazione completata. Usa 'plot_delta_scan.py' (aggiornando il nome file) per graficare.")