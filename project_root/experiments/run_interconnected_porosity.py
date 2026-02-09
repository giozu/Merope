from __future__ import annotations

from typing import Dict

from core.geometry import MicrostructureBuilder
from core.solver import ThermalSolver
from core.utils import ProjectManager


# ----------------------------------------------------------------------
# CONFIGURAZIONE FISICA E NUMERICA
# ----------------------------------------------------------------------

# RVE e voxelizzazione (coerente con inco_intra_inter_polycrystal.py)
L_DIM = [10.0, 10.0, 10.0]
N_VOX = 200
SEED = 0

# Parametri geometrici legacy (inter + intra + grains + delta)
INTER_R = 0.03   # raggio pori inter-granulari
INTER_PHI_LIST = [0.40]  # frazioni volumetriche da esplorare (inter)

INTRA_R = 0.10   # raggio pori intra-granulari
INTRA_PHI_LIST = [0.19]  # frazioni volumetriche da esplorare (intra)

GRAIN_R = 1.0    # raggio per i semi Laguerre
GRAIN_PHI = 1.0  # packing per i semi (1.0 = RVE pieno di grani)

DELTA_LIST = [0.003]  # spessori del bordo di grano da testare

# Proprietà termiche (convenzione legacy: 0 = matrice, 2 = pori)
K_MATRIX = 1.0
K_GAS = 1e-3
K_THERMAL = [K_MATRIX, K_MATRIX, K_GAS]


def main() -> None:
    """Esegue un piccolo sweep di casi inter+intra usando l'API OOP."""
    pm = ProjectManager()
    builder = MicrostructureBuilder(L=L_DIM, n3D=N_VOX, seed=SEED)
    solver = ThermalSolver(n_cpus=4)

    base_folder = "Results_Interconnected"
    pm.cleanup_folder(base_folder)

    for inter_phi in INTER_PHI_LIST:
        for intra_phi in INTRA_PHI_LIST:
            for delta in DELTA_LIST:
                case_dir = (
                    f"{base_folder}/"
                    f"p_inter_{inter_phi:.3f}_"
                    f"p_intra_{intra_phi:.3f}_"
                    f"delta_{delta:.4f}"
                ).replace(".", "_")

                print(
                    f"\n=== Case: inter={inter_phi:.3f}, "
                    f"intra={intra_phi:.3f}, delta={delta:.4f} ==="
                )

                with pm.cd(case_dir):
                    # 1. Costruisci la microstruttura completa (inter+intra)
                    struct = builder.generate_interconnected_structure(
                        inter_radius=INTER_R,
                        inter_phi=inter_phi,
                        intra_radius=INTRA_R,
                        intra_phi=intra_phi,
                        grain_radius=GRAIN_R,
                        grain_phi=GRAIN_PHI,
                        delta=delta,
                    )

                    # 2. Voxelizza (scrive structure.vtk + Coeffs.txt nella cartella del caso)
                    fractions: Dict[int, float] = builder.voxellate(struct, K_THERMAL)

                    # 3. Risolvi con Amitex (thermalCoeff_amitex.txt nella stessa cartella)
                    results = solver.solve()

                    # 4. Logga i risultati in un summary.txt a livello superiore
                    phi_pore = fractions.get(2, 0.0)  # porosità totale (fase 2)
                    log_data = {
                        "inter_phi_target": inter_phi,
                        "intra_phi_target": intra_phi,
                        "delta_phys": delta,
                        "phi_pore_measured": phi_pore,
                        **results,
                    }
                    pm.log_results("../summary.txt", log_data, header=list(log_data.keys()))

                    print(
                        f"   -> phi_pore = {phi_pore:.4f}, "
                        f"Kmean = {results['Kmean']:.4f}"
                    )


if __name__ == "__main__":
    main()

