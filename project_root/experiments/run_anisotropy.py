import os
import sys
from pathlib import Path

# Ensure project_root/ is on sys.path so `core` is importable
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from core.geometry import MicrostructureBuilder
from core.solver import ThermalSolver
from core.utils import ProjectManager
import numpy as np

# Config
L_DIM = [10, 10, 10]
N_VOX = 100
K_THERMAL = [0.0, 1.0, 1e-3]
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-solver", action="store_true", help="Skip Amitex solver")
    args = parser.parse_args()

    pm = ProjectManager()
    builder = MicrostructureBuilder(L=L_DIM, n3D=N_VOX, seed=42)
    solver = ThermalSolver(n_cpus=4)
    
    output_dir = Path("Results_Anisotropy")
    pm.cleanup_folder(str(output_dir))
    
    # Study aspect ratio from 1.0 (spherical) to 0.2 (flat)
    # Loop over 5 steps for a decent sweep
    for ar in np.linspace(1.0, 0.2, 5):
        print(f"Starting aspect ratio: {ar:.2f}")
        
        # Setup case folder (absolute path)
        case_dir = output_dir.resolve() / f"AR_{ar:.2f}"
        case_dir.mkdir(parents=True, exist_ok=True)
        abs_case_dir = str(case_dir)

        try:
            # 1. Generate geometry
            # Inside the builder, Phase 1 is assigned to the boundary layer
            struct = builder.generate_polycrystal(grain_radius=3.0, delta=0.05, aspect_ratio=[1.0, float(ar), 1.0/float(ar)])
            
            # 2. Voxellate (Apply K, passing absolute paths)
            fractions = builder.voxellate(
                struct, 
                K_THERMAL, 
                vtk_path=case_dir / "structure.vtk",
                coeffs_path=case_dir / "Coeffs.txt"
            )
            porosity = fractions.get(2, 0.0) # Pore is phase 2

            # 3. Run Amitex Solver (internally handles chdir safely)
            if args.no_solver:
                print(f"Skipping solver for AR={ar:.2f}")
                results = {"Kxx": 0.0, "Kyy": 0.0, "Kzz": 0.0, "Kmean": 0.0}
            else:
                results = solver.solve(vtk_path=os.path.join(abs_case_dir, "structure.vtk"))
            
            # 4. Log results
            log_data = {
                "AspectRatio": ar, 
                "Porosity": porosity, 
                **results
            }
            pm.log_results(str(output_dir / "summary.txt"), log_data, header=list(log_data.keys()))
            print(f"Finished Aspect Ratio: {ar:.2f} | Porosity: {log_data['Porosity']:.4f}")
        except Exception as e:
            print(f"Error during AR={ar:.2f}: {e}")

if __name__ == "__main__":
    main()