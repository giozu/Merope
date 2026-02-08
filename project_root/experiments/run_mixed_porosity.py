from core.geometry import MicrostructureBuilder
from core.solver import ThermalSolver
from core.utils import ProjectManager

# Config
L_DIM = [10, 10, 10]
N_VOX = 120

# Index 0: ignored
# Index 1: Matrix (K=1.0)
# Index 2: Pores (K=0.001)
K_THERMAL = [0.0, 1.0, 0.001] 

def main():
    pm = ProjectManager()
    builder = MicrostructureBuilder(L=L_DIM, n3D=N_VOX, seed=42)
    solver = ThermalSolver(n_cpus=4)
    
    pm.cleanup_folder("Results_Mixed")
    
    # PARAMETERS
    # Increase delta so it's visible at N=120 (delta 0.2 is approx 2.5 voxels thick)
    DELTA_GB = 0.2 
    S_RADIUS = 0.5 # Larger spheres are easier to see

    for phi_intra in [0.05, 0.10, 0.15]:
        folder = f"Results_Mixed/Intra_{phi_intra}"
        with pm.cd(folder):
            # Generate
            struct = builder.generate_mixed_structure(
                grain_radius=2.5, 
                delta=DELTA_GB, 
                intra_pore_list=[[S_RADIUS, phi_intra]]
            )
            
            # Voxelize
            fractions = builder.voxellate(struct, K_THERMAL)
            
            # Solve
            results = solver.solve()
            
            # Log (Pore is Phase 2)
            porosity = fractions.get(2, 0.0)

            log_data = {
                "Phi_Intra": phi_intra, 
                "Total_Porosity": fractions.get(2, 0.0), 
                **results
            }

            pm.log_results("../summary.txt", log_data, header=list(log_data.keys()))
            
            print(f"Target Phi: {phi_intra} | Result Porosity: {porosity:.4f} | K: {results['Kmean']:.4f}")

if __name__ == "__main__":
    main()