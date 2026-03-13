import os
import sys
import argparse
from pathlib import Path

# Ensure project_root/ is on sys.path so `core` is importable
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from core.geometry import MicrostructureBuilder
from core.solver import ThermalSolver
from core.utils import ProjectManager

# Config
L_DIM = [10, 10, 10]
N_VOX = 150

# Index 0: ignored
# Index 1: Matrix (K=1.0)
# Index 2: Pores (K=0.001)
K_THERMAL = [0.0, 1.0, 0.001]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-solver", action="store_true", help="Skip Amitex solver")
    args = parser.parse_args()

    pm = ProjectManager()
    builder = MicrostructureBuilder(L=L_DIM, n3D=N_VOX, seed=42)
    solver = ThermalSolver(n_cpus=4)
    
    output_dir = Path("Results_Mixed")
    pm.cleanup_folder(str(output_dir))
    
    # PARAMETERS
    DELTA_GB = 0.2 
    S_RADIUS = 0.1

    for phi_intra in [0.15]:
        print(f"Starting mixed porosity study: phi_intra={phi_intra}")
        
        # Setup case folder (absolute path)
        case_name = f"Intra_{phi_intra}"
        case_dir = output_dir.resolve() / case_name
        case_dir.mkdir(parents=True, exist_ok=True)
        abs_case_dir = str(case_dir)

        try:
            # 1. Generate
            struct = builder.generate_mixed_structure(
                grain_radius=2.5, 
                delta=DELTA_GB, 
                intra_pore_list=[[S_RADIUS, phi_intra]]
            )
            
            # 2. Voxelize (passing absolute paths)
            fractions = builder.voxellate(
                struct, 
                K_THERMAL, 
                vtk_path=case_dir / "structure.vtk",
                coeffs_path=case_dir / "Coeffs.txt"
            )
            porosity = fractions.get(2, 0.0) # Pore is Phase 2
            
            # 3. Solve (handles chdir internally)
            if args.no_solver:
                print(f"Skipping solver for Phi_intra={phi_intra}")
                results = {"Kxx": 0, "Kyy": 0, "Kzz": 0, "Kmean": 0}
            else:
                results = solver.solve(vtk_file=os.path.join(abs_case_dir, "structure.vtk"))
        
            # 4. Log
            log_data = {
                "Phi_Intra": phi_intra, 
                "Total_Porosity": porosity, 
                **results
            }
            pm.log_results(str(output_dir / "summary.txt"), log_data, header=list(log_data.keys()))
            
            print(f"Target Phi: {phi_intra} | Result Porosity: {porosity:.4f} | K: {results['Kmean']:.4f}")
        except Exception as e:
            print(f"Error during Phi_intra={phi_intra}: {e}")

if __name__ == "__main__":
    main()