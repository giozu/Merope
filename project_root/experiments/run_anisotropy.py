from core.geometry import MicrostructureBuilder
from core.solver import ThermalSolver
from core.utils import ProjectManager
import numpy as np
import merope

# Config
L = [10, 10, 10]
N3D = 100
K_VALUES = [1.0, 1.0, 1e-3] # Matrix, Matrix, Gas

def main():
    pm = ProjectManager()
    builder = MicrostructureBuilder(L, N3D)
    solver = ThermalSolver(n_cpus=4)
    
    pm.cleanup_folder("Results_Anisotropy")
    
    for ar in np.linspace(1.0, 0.2, 5):
        with pm.cd(f"Results_Anisotropy/AR_{ar:.2f}"):
            # 1. Build anisotropic grains
            grains = builder.generate_polycrystal(grain_radius=3.0, delta=0.5, aspect_ratio=[1.0, ar, 1/ar])
            
            # 2. Combine and Voxellate (Phase 3 is the delta/pore)
            struct = merope.Structure_3D(grains, {3: 0}) # Map Phase 3 to Pore index
            fractions = builder.voxellate(struct, K_VALUES)
            
            # 3. Solve
            results = solver.solve()
            
            # 4. Log
            log_data = {"AspectRatio": ar, "Porosity": fractions[2], **results}
            pm.log_results("../summary.txt", log_data, header=list(log_data.keys()))

if __name__ == "__main__":
    main()