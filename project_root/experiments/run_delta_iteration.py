from core.geometry import MicrostructureBuilder
from core.solver import ThermalSolver
from core.utils import ProjectManager
import numpy as np
import merope

L = [10, 10, 10]
N3D = 100

def main():
    pm = ProjectManager()
    builder = MicrostructureBuilder(L, N3D)
    solver = ThermalSolver(n_cpus=4)
    
    pm.cleanup_folder("Results_Delta_Sweep")
    
    for d in np.linspace(0.01, 1.0, 10):
        with pm.cd(f"Results_Delta_Sweep/Delta_{d:.3f}"):
            # Pure Interconnected porosity sweep
            inter = builder.generate_polycrystal(grain_radius=3.0, delta=d)
            struct = merope.Structure_3D(inter, {3: 0})
            
            fractions = builder.voxellate(struct, [1.0, 1.0, 1e-3])
            results = solver.solve()
            
            log_data = {"Delta": d, "Porosity": fractions[2], **results}
            pm.log_results("../summary.txt", log_data, header=list(log_data.keys()))

if __name__ == "__main__":
    main()