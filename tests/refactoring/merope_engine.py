import os
import shutil
import time
import numpy as np
import merope
import sac_de_billes
import interface_amitex_fftp.amitex_wrapper as amitex
import interface_amitex_fftp.post_processing as amitex_out

class MeropeSim:
    """
    Classe motore per simulazioni Merope + Amitex.
    Gestisce la generazione della microstruttura e l'esecuzione sicura.
    """
    def __init__(self, work_dir="Results"):
        self.work_dir = os.path.abspath(work_dir)
        if os.path.exists(self.work_dir):
            shutil.rmtree(self.work_dir)
        os.makedirs(self.work_dir)
        
        # Parametri Materiale
        self.K_MATRIX = 1.0
        self.K_GAS = 1e-03
        # Ordine fasi standard per Amitex: [Fase 0, Fase 1, Fase 2]
        # Nel tuo scan finale: Fase 0 (non esiste), Fase 1 (Matrice), Fase 2 (Poro)
        # Merope mappa K in base all'ID, quindi dobbiamo essere coerenti.
        # Se la matrice è fase 1 e il poro è fase 2:
        self.K_MAP = {1: self.K_MATRIX, 2: self.K_GAS} 
        
    def _generate_structure_DeltaScan(self, L, seed, lagR, lagPhi, inclR, p_delta, incl2R, p_intra, delta_phys):
        """
        Logica avanzata 'Delta Scan' (dai tuoi script recenti).
        Gestisce porosità intergranulare (p_delta) e intragranulare (p_intra)
        con la logica del doppio merge per pulire le sovrapposizioni.
        """
        # 1. Pori Intragranulari (Fase 2 temporanea)
        sph_intra = merope.SphereInclusions_3D()
        sph_intra.setLength(L)
        if p_intra > 0:
            sph_intra.fromHisto(seed, sac_de_billes.TypeAlgo.BOOL, 0., [[incl2R, p_intra]], [2])
        multi_intra = merope.MultiInclusions_3D()
        multi_intra.setInclusions(sph_intra)
        struct_intra = merope.Structure_3D(multi_intra)

        # 2. Pori Intergranulari (Fase 2 temporanea)
        sph_gb = merope.SphereInclusions_3D()
        sph_gb.setLength(L)
        if p_delta > 0:
            sph_gb.fromHisto(seed, sac_de_billes.TypeAlgo.RSA, 0., [[inclR, p_delta]], [2])
        multi_gb = merope.MultiInclusions_3D()
        multi_gb.setInclusions(sph_gb)

        # 3. Grani Laguerre (Fase 1) + Layer Delta (Fase 3)
        sph_lag = merope.SphereInclusions_3D()
        sph_lag.setLength(L)
        sph_lag.fromHisto(seed, sac_de_billes.TypeAlgo.RSA, 0., [[lagR, lagPhi]], [1])

        poly = merope.LaguerreTess_3D(L, sph_lag.getSpheres())
        multi_poly = merope.MultiInclusions_3D()
        multi_poly.setInclusions(poly)
        
        # Aggiungi layer (fase 3) e resetta grani a fase 1
        multi_poly.addLayer(multi_poly.getAllIdentifiers(), 3, delta_phys)
        multi_poly.changePhase(multi_poly.getAllIdentifiers(), [1 for _ in multi_poly.getAllIdentifiers()])

        # 4. MERGE STRATEGICO "ALLA TESISTA"
        # Step A: Unisci Pori Inter (2) e Grani+Layer. 
        # Mappa Fase 2 (Pori Inter) -> 0 e Fase 3 (Layer) -> 0. 
        # Risultato: Fase 1 (Grani) e Fase 0 (Spazio vuoto sui bordi + Pori inter)
        struct_inter_poly = merope.Structure_3D(multi_gb, multi_poly, {2: 0, 3: 0})

        # Step B: Unisci con Pori Intra.
        # Mappa Fase 0 (il vuoto creato prima) -> Fase 2 (Poro Finale).
        # I pori intra sono già Fase 2.
        final_struct = merope.Structure_3D(struct_inter_poly, struct_intra, {0: 2})
        
        # Risultato finale atteso: Fase 1 = Matrice, Fase 2 = Poro Totale
        return final_struct

    def run_case_delta_scan(self, n3D, L, seed, lagR, lagPhi, inclR, p_delta, incl2R, p_intra, delta_phys):
        """
        Esegue il caso specifico del Delta Scan.
        """
        case_name = f"DScan_Pd{p_delta:.2f}_Pi{p_intra:.2f}_D{delta_phys:.2f}_S{seed}"
        case_dir = os.path.join(self.work_dir, case_name)
        os.makedirs(case_dir, exist_ok=True)

        # --- GENERAZIONE ---
        structure = self._generate_structure_DeltaScan(L, seed, lagR, lagPhi, inclR, p_delta, incl2R, p_intra, delta_phys)

        # --- VOXEL ---
        gridParams = merope.vox.create_grid_parameters_N_L_3D([n3D]*3, L)
        grid = merope.vox.GridRepresentation_3D(structure, gridParams, merope.vox.VoxelRule.Average)
        
        # Analisi fasi
        analyzer = merope.vox.GridAnalyzer_3D()
        phases_fracts = analyzer.compute_percentages(grid)
        
        # Porosità reale = Frazione Fase 2
        real_porosity = phases_fracts.get(2, 0.0)

        # Applicazione K (Usiamo il dizionario K_MAP per sicurezza)
        # Fase 1 -> K_MATRIX, Fase 2 -> K_GAS
        # HomogRule Voigt richiede un vettore ordinato per indice fase [K_fase0, K_fase1, K_fase2...]
        # Poiché fase 0 non c'è, usiamo una lista dummy sicura
        k_vector_input = [0.0, self.K_MATRIX, self.K_GAS] 
        
        grid.apply_homogRule(merope.HomogenizationRule.Voigt, k_vector_input)
        
        # Stampa VTK
        printer = merope.vox.vtk_printer_3D()
        full_vtk = os.path.join(case_dir, "Zone.vtk")
        full_coeff = os.path.join(case_dir, "Coeffs.txt")
        printer.printVTK_segmented(grid, full_vtk, full_coeff, nameValue="MaterialId")

        # --- AMITEX ---
        cwd = os.getcwd()
        results = None
        try:
            os.chdir(case_dir)
            amitex.computeThermalCoeff("Zone.vtk", 2) # 2 processori
            results = amitex_out.printThermalCoeff(".")
        except Exception as e:
            print(f"Errore Amitex: {e}")
        finally:
            os.chdir(cwd)

        k_mean = 0.0
        if results:
            k_mean = np.mean([results[i][i] for i in range(3)])
            
        return real_porosity, k_mean, results