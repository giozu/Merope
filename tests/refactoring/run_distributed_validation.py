import numpy as np
import pandas as pd
import merope
import sac_de_billes
from merope_engine import MeropeSim

class DistributedSim(MeropeSim):
    """
    Estensione di MeropeSim specifica per la porosità distribuita.
    Ignora la logica complessa dei bordi di grano (IGB) per focalizzarsi
    su pori sferici isolati (Intragranulari) come da teoria di Loeb.
    """
    
    def _generate_distributed_structure(self, L, seed, porosity, r_pore, r_grain):
        # 1. Genera la Matrice (Grani Policristallini) - Phase 1 (Matrice)
        # Usiamo RSA per i semi dei grani
        sph_grains = merope.SphereInclusions_3D()
        sph_grains.setLength(L)
        sph_grains.fromHisto(seed, sac_de_billes.TypeAlgo.RSA, 0., [[r_grain, 1.0]], [self.PHASE_GRAINS_INIT])
        
        polyCrystal = merope.LaguerreTess_3D(L, sph_grains.getSpheres())
        multiInc_Matrix = merope.MultiInclusions_3D()
        multiInc_Matrix.setInclusions(polyCrystal)
        # Imposta tutti i grani a Fase 1 (Matrice Semplice)
        multiInc_Matrix.changePhase(multiInc_Matrix.getAllIdentifiers(), [1 for _ in multiInc_Matrix.getAllIdentifiers()])
        
        struct_Matrix = merope.Structure_3D(multiInc_Matrix)

        # 2. Genera la Porosità Distribuita (Sfere) - Phase 2 (Gas)
        # Usiamo RSA per evitare sovrapposizioni eccessive, tipico della "distributed"
        sph_pores = merope.SphereInclusions_3D()
        sph_pores.setLength(L)
        # Generiamo sfere di Fase 2 con la porosità target
        sph_pores.fromHisto(seed, sac_de_billes.TypeAlgo.RSA, 0., [[r_pore, porosity]], [self.PHASE_INCL])
        
        multiInc_Pores = merope.MultiInclusions_3D()
        multiInc_Pores.setInclusions(sph_pores)
        struct_Pores = merope.Structure_3D(multiInc_Pores)

        # 3. Combina: I pori "bucano" la matrice.
        # Non usiamo dizionari complessi: Phase 2 (Poro) sovrascrive Phase 1 (Matrice).
        # K vector è [Matrix, Matrix, Gas], quindi Phase 2 = Gas. Perfetto.
        structure = merope.Structure_3D(struct_Matrix, struct_Pores)
        
        return structure

    def run_distributed_case(self, n3D, L, seed, porosity, r_pore, r_grain):
        # Setup cartelle
        case_name = f"Distr_P{porosity:.3f}_Seed{seed}"
        case_dir = self._prepare_case_dir(case_name) # Helper method o copia logica manuale
        
        # Generazione
        structure = self._generate_distributed_structure(L, seed, porosity, r_pore, r_grain)
        
        # Voxellizzazione e Calcolo (riutilizziamo la logica base ma adattata)
        vtk_path = case_dir + "/structure.vtk"
        coeff_path = case_dir + "/Coeffs.txt"
        
        gridParams = merope.vox.create_grid_parameters_N_L_3D([n3D]*3, L)
        grid = merope.vox.GridRepresentation_3D(structure, gridParams, merope.vox.VoxelRule.Average)
        
        # Calcola porosità reale
        analyzer = merope.vox.GridAnalyzer_3D()
        phases = analyzer.compute_percentages(grid)
        real_p = phases[2] if 2 in phases else 0.0
        
        # Applica K e Stampa
        grid.apply_homogRule(merope.HomogenizationRule.Voigt, self.K_VECTOR)
        printer = merope.vox.vtk_printer_3D()
        printer.printVTK_segmented(grid, vtk_path, coeff_path, nameValue="MaterialId")
        
        # Esegui Amitex
        k_eff = self._run_amitex(case_dir, "structure.vtk")
        return real_p, k_eff

    # Helper interni per non duplicare codice se non accessibili
    def _prepare_case_dir(self, name):
        import os
        path = os.path.join(self.work_dir, name)
        os.makedirs(path, exist_ok=True)
        return path
    
    def _run_amitex(self, case_dir, vtk_name):
        import os
        import interface_amitex_fftp.amitex_wrapper as amitex
        import interface_amitex_fftp.post_processing as amitex_out
        cwd = os.getcwd()
        try:
            os.chdir(case_dir)
            amitex.computeThermalCoeff(vtk_name, 2)
            res = amitex_out.printThermalCoeff(".")
            return (res[0][0] + res[1][1] + res[2][2]) / 3.0
        except:
            return 0.0
        finally:
            os.chdir(cwd)

# --- PARAMETRI DI VALIDAZIONE ---
L_SIDE = [10, 10, 10]
N_VOXEL = 100
GRAIN_R = 3.0
PORE_R = 0.3 # Raggio pori (influenza trascurabile secondo la tesi [cite: 67])

# Range di porosità dal grafico della tesi (0.0 a 0.3)
porosities = np.linspace(0.01, 0.3, 8) 
ALPHA_LOEB = 1.37 # Valore ottimizzato dalla tesi 

sim = DistributedSim(work_dir="Validation_Distributed_Loeb")
results = []

print(f"Validazione Distributed Porosity vs Loeb (Alpha={ALPHA_LOEB})...")
print("Target_P\tReal_P\tK_Sim\tK_Loeb\tError%")

for p_target in porosities:
    try:
        real_p, k_sim = sim.run_distributed_case(
            n3D=N_VOXEL,
            L=L_SIDE,
            seed=0,
            porosity=p_target,
            r_pore=PORE_R,
            r_grain=GRAIN_R
        )
        
        # Calcolo Teorico Loeb
        k_loeb = 1.0 * (1 - ALPHA_LOEB * real_p)
        
        # Errore relativo
        err = 100 * abs(k_sim - k_loeb) / k_loeb
        
        results.append({"P": real_p, "K_Sim": k_sim, "K_Loeb": k_loeb, "Err": err})
        print(f"{p_target:.3f}\t\t{real_p:.3f}\t{k_sim:.4f}\t{k_loeb:.4f}\t{err:.2f}%")
        
    except Exception as e:
        print(f"Errore su p={p_target}: {e}")

# Salvataggio
pd.DataFrame(results).to_csv("distributed_validation.csv", sep="\t", index=False)
print("\nFinito. Controlla se l'errore è basso (<2% come nella tesi).")