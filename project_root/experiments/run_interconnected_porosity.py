import os
import sys
import argparse
from pathlib import Path
from typing import Dict

# Ensure project_root/ is on sys.path so `core` is importable
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from core.geometry import MicrostructureBuilder
from core.solver import ThermalSolver
from core.utils import ProjectManager
import numpy as np

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-solver", action="store_true", help="Skip Amitex solver")
    args = parser.parse_args()

    pm = ProjectManager()
    builder = MicrostructureBuilder(L=L_DIM, n3D=N_VOX, seed=SEED)
    solver = ThermalSolver(n_cpus=4)

    output_dir = Path("Results_Interconnected")
    pm.cleanup_folder(str(output_dir))

    for inter_phi in INTER_PHI_LIST:
        for intra_phi in INTRA_PHI_LIST:
            for delta in DELTA_LIST:
                case_name = (
                    f"p_inter_{inter_phi:.3f}_"
                    f"p_intra_{intra_phi:.3f}_"
                    f"delta_{delta:.4f}"
                ).replace(".", "_")
                
                case_dir = output_dir.resolve() / case_name
                case_dir.mkdir(parents=True, exist_ok=True)
                abs_case_dir = str(case_dir)

                print(f"\n=== Case: {case_name} ===")

                try:
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

                    # 2. Voxelizza (Apply K, passing absolute paths)
                    fractions: Dict[int, float] = builder.voxellate(
                        struct, 
                        K_THERMAL, 
                        vtk_path=case_dir / "structure.vtk",
                        coeffs_path=case_dir / "Coeffs.txt"
                    )

                    # 3. Solver Amitex (handles chdir internally)
                    if args.no_solver:
                        print(f"Skipping solver for {case_name}")
                        results = {"Kxx": 0, "Kyy": 0, "Kzz": 0, "Kmean": 0}
                    else:
                        results = solver.solve(vtk_path=os.path.join(abs_case_dir, "structure.vtk"))

                    # 4. Logga i risultati in un summary.txt a livello superiore
                    phi_pore = fractions.get(2, 0.0)  # porosità totale (fase 2)
                    log_data = {
                        "inter_phi_target": inter_phi,
                        "intra_phi_target": intra_phi,
                        "delta_phys": delta,
                        "phi_pore_measured": phi_pore,
                        **results,
                    }
                    pm.log_results(str(output_dir / "summary.txt"), log_data, header=list(log_data.keys()))

                    print(
                        f"   -> phi_pore = {phi_pore:.4f}, "
                        f"Kmean = {results['Kmean']:.4f}"
                    )
                except Exception as e:
                    print(f"Error during {case_name}: {e}")


if __name__ == "__main__":
    main()
