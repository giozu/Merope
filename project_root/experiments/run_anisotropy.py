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

def main():
    pm = ProjectManager()
    builder = MicrostructureBuilder(L=L_DIM, n3D=N_VOX, seed=42)
    solver = ThermalSolver(n_cpus=4)
    
    pm.cleanup_folder("Results_Anisotropy")
    
    # Study aspect ratio from 1.0 (spherical) to 0.2 (flat)
    for ar in np.linspace(1.0, 1.0, 1):
        print("Starting aspect ratio: ", ar)
        with pm.cd(f"Results_Anisotropy/AR_{ar:.2f}"):
            # 1. Generate geometry
            # Inside the builder, Phase 1 is assigned to the boundary layer
            struct = builder.generate_polycrystal(grain_radius=3.0, delta=0.05, aspect_ratio=[1.0, ar, 1/ar])
            
            # 2. Voxellate (Apply K values)
            fractions = builder.voxellate(struct, K_THERMAL)
            porosity = fractions.get(2, 0.0) # Il poro è fase 2

            # 3. Run Amitex Solver
            results = solver.solve()
            
            # 4. Log results (Phase 1 is the pore fraction)
            log_data = {
                "AspectRatio": ar, 
                "Porosity": porosity, 
                **results
            }
            pm.log_results("../summary.txt", log_data, header=list(log_data.keys()))
            print(f"Finished Aspect Ratio: {ar:.2f} | Porosity: {log_data['Porosity']:.4f}")

if __name__ == "__main__":
    main()