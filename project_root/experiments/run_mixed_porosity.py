from core.geometry import MicrostructureBuilder
from core.solver import ThermalSolver
from core.utils import ProjectManager
import merope

L = [10, 10, 10]
N3D = 150
K_VALUES = [1.0, 1.0, 1e-3]

def main():
    pm = ProjectManager()
    builder = MicrostructureBuilder(L, N3D, seed=42)
    solver = ThermalSolver(n_cpus=6)
    
    pm.cleanup_folder("Results_Mixed")
    
    # Parametric variation of Intra vs Inter porosity
    for phi_intra in [0.05, 0.1, 0.15]:
        with pm.cd(f"Results_Mixed/Intra_{phi_intra}"):
            # Generate two structures
            inter = builder.generate_polycrystal(grain_radius=2.0, delta=0.2)
            intra = builder.generate_spheres([[0.3, phi_intra]])
            
            # Combine: Spheres (Phase 2) and Layers (Phase 3) become Material Index 2 (Gas)
            # Matrix is Material Index 0
            mapping = { 2: 0, 3: 0 } 
            struct = merope.Structure_3D(intra, inter, mapping)
            
            fractions = builder.voxellate(struct, K_VALUES)
            results = solver.solve()
            
            log_data = {"Phi_Intra_Target": phi_intra, "Total_Porosity": fractions[2], **results}
            pm.log_results("../summary.txt", log_data, header=list(log_data.keys()))

if __name__ == "__main__":
    main()