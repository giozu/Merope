# merope_engine.py

import os
import shutil
import numpy as np

import sac_de_billes
import merope
import interface_amitex_fftp.amitex_wrapper as amitex
import interface_amitex_fftp.post_processing as amitex_out


class MeropeEngine:
    """
    Motore per simulazioni Merope + AMITEX.

    Convenzioni di fase (coerenti con i tuoi script recenti):
      - phase 0 = matrice solida
      - phase 2 = poro (totale, dopo merge)
      - phase 3 = layer GB (solo temporaneo, poi assorbito in 0→2)

    Le conduttività sono passate come:
      K = [Kmatrix, Kmatrix, Kgases]
    dove:
      - fase 0 → Kmatrix
      - fase 1 → Kmatrix (non usata ma tenuta per sicurezza)
      - fase 2 → Kgases
    """

    def __init__(self,
                 L,
                 n3D,
                 lagR,
                 lagPhi,
                 inclR_inter,
                 inclR_intra,
                 k_matrix=1.0,
                 k_gas=1e-3,
                 work_dir="DeltaScan_Results",
                 voxel_rule=merope.vox.VoxelRule.Average,
                 homog_rule=merope.HomogenizationRule.Voigt,
                 nproc=6):

        self.L = list(L)
        self.n3D = int(n3D)
        self.lagR = float(lagR)
        self.lagPhi = float(lagPhi)
        self.inclR_inter = float(inclR_inter)
        self.inclR_intra = float(inclR_intra)

        self.voxel_rule = voxel_rule
        self.homog_rule = homog_rule

        self.nproc = int(nproc)

        self.Kmatrix = float(k_matrix)
        self.Kgas = float(k_gas)
        # phase 0 = matrix, phase 1 dummy, phase 2 = pore
        self.K = [self.Kmatrix, self.Kmatrix, self.Kgas]

        # convenzione fasi
        self.phase_matrix = 0
        self.phase_pore = 2
        self.phase_layer = 3

        # cartella di lavoro
        self.work_dir = os.path.abspath(work_dir)
        if os.path.exists(self.work_dir):
            shutil.rmtree(self.work_dir)
        os.makedirs(self.work_dir)

    # ------------------------------------------------------------------
    # helpers directory / AMITEX
    # ------------------------------------------------------------------

    def _case_dir(self, *parts):
        path = os.path.join(self.work_dir, *parts)
        os.makedirs(path, exist_ok=True)
        return path

    def _run_amitex_in_dir(self, case_dir, vtk_name="Zone.vtk"):
        cwd = os.getcwd()
        try:
            os.chdir(case_dir)
            amitex.computeThermalCoeff(vtk_name, self.nproc)
            Kmat = amitex_out.printThermalCoeff(".")
        finally:
            os.chdir(cwd)

        kvals = [Kmat[i][i] for i in range(3)]

        # sanity check
        if not np.all(np.isfinite(kvals)):
            raise RuntimeError(f"Non-finite K from AMITEX in {case_dir}: {kvals}")

        return kvals

    def _voxelize_and_assign(self, structure):
        """
        Voxelizza una struttura Merope, calcola frazioni di fase
        e applica la regola di omogeneizzazione.
        """
        
        grid_params = merope.vox.create_grid_parameters_N_L_3D(
            [self.n3D, self.n3D, self.n3D],
            self.L
        )

        grid = merope.vox.GridRepresentation_3D(
            structure, grid_params, self.voxel_rule
        )

        analyzer = merope.vox.GridAnalyzer_3D()
        phase_fracs = analyzer.compute_percentages(grid)

        # debug opzionale
        print("\n[DEBUG] phase fractions:")
        for pid, frac in phase_fracs.items():
            print(f"  phase {pid}: {frac:.6f}")
        print("--------------")

        grid.apply_homogRule(self.homog_rule, self.K)
        return grid, phase_fracs

    def _check_geometry_scales(self):
        """
        Verifica i criteri geometrici:
          1) L_RVE / R_pore > 10
          2) R_pore / L_voxel > 2
        """
        L_RVE = float(self.L[0])   # assumiamo dominio cubico
        R_pore = float(self.inclR_intra)
        L_voxel = L_RVE / float(self.n3D)

        crit1 = L_RVE / R_pore
        crit2 = R_pore / L_voxel

        if crit1 < 10.0:
            raise ValueError(
                f"Geometric criterion failed: L_RVE/R_pore = {crit1:.2f} <= 10. "
                f"L_RVE={L_RVE}, R_pore={R_pore}"
            )

        if crit2 < 2.0:
            raise ValueError(
                f"Geometric criterion failed: R_pore/L_voxel = {crit2:.2f} <= 2. "
                f"R_pore={R_pore}, L_voxel={L_voxel}"
            )

        print(f"GEOMETRY")
        print(f"[CHECK] R_pore = {R_pore:.2f}")
        print(f"[CHECK] L_RVE = {L_RVE:.2f}")
        print(f"[CHECK] L_voxel = {L_voxel:.2f}")
        print(f"[CHECK] L_RVE/R_pore = {crit1:.2f}  (OK)")
        print(f"[CHECK] R_pore/L_voxel = {crit2:.2f}  (OK)")

    def suggest_n3D(self, R_pore):
        """
        Suggerisce il n3D minimo per soddisfare:
            1) L_RVE / R_pore > 10
            2) R_pore / L_voxel > 2
        con L_voxel = L_RVE / n3D
        
        Ritorna: n3D_min (int)
        """
        L_RVE = float(self.L[0])

        # criterio 1: L_RVE / R_pore > 10  → R_pore < L_RVE / 10
        if R_pore >= L_RVE / 10:
            raise ValueError(
                f"Impossibile: R_pore={R_pore} troppo grande rispetto a L_RVE={L_RVE} "
                f"(serve R_pore < {L_RVE/10})."
            )

        # criterio 2: n3D > 2 * L_RVE / R_pore
        n3D_min = int(np.ceil(2 * L_RVE / R_pore))

        return n3D_min


    # ------------------------------------------------------------------
    # 1) Distributed porosity (nessun grain boundary film)
    # ------------------------------------------------------------------

    def build_distributed_structure(self, seed, p_target):
        """
        Microstruttura: matrice uniforme (phase 0) + pori distribuiti (phase 2).

        - Si usa UNA famiglia di sfere di raggio self.inclR_intra.
        - Algoritmo BOOL per evitare problemi RSA.
        """
        sph = merope.SphereInclusions_3D()
        sph.setLength(self.L)

        if p_target > 0.0:
            sph.fromHisto(
                int(seed),
                sac_de_billes.TypeAlgo.BOOL,
                0.0,
                [[self.inclR_intra, float(p_target)]],
                [self.phase_pore]   # phase 2
            )

        multi = merope.MultiInclusions_3D()
        multi.setInclusions(sph)
        struct = merope.Structure_3D(multi)

        return struct

    # ------------------------------------------------------------------
    # 2) Interconnected porosity (Laguerre + film δ + merge)
    # ------------------------------------------------------------------
    def build_interconnected_structure(self,
                                    seed,
                                    p_delta,
                                    p_intra,
                                    delta_phys):

        L = self.L

        phase_grains   = self.phase_matrix    # 0
        phase_intra    = 1
        phase_inter    = 2
        phase_boundary = 3

        # -------------------------
        # 1) inter-granular pores
        # -------------------------
        sph_inter = merope.SphereInclusions_3D()
        sph_inter.setLength(L)
        if p_delta > 0:
            sph_inter.fromHisto(
                int(seed),
                sac_de_billes.TypeAlgo.BOOL,
                0.0,
                [[self.inclR_inter, p_delta]],
                [phase_inter],
            )
        inter_pores = merope.MultiInclusions_3D()
        inter_pores.setInclusions(sph_inter)

        # -------------------------
        # 2) intra-granular pores
        # -------------------------
        sph_intra = merope.SphereInclusions_3D()
        sph_intra.setLength(L)
        if p_intra > 0:
            sph_intra.fromHisto(
                int(seed),
                sac_de_billes.TypeAlgo.BOOL,
                0.0,
                [[self.inclR_intra, p_intra]],
                [phase_intra],
            )
        intra_pores = merope.MultiInclusions_3D()
        intra_pores.setInclusions(sph_intra)

        # -------------------------
        # 3) Laguerre grains + boundary layer
        # -------------------------
        sph_lag = merope.SphereInclusions_3D()
        sph_lag.setLength(L)
        sph_lag.fromHisto(
            int(seed),
            sac_de_billes.TypeAlgo.RSA,
            0.0,
            [[self.lagR, self.lagPhi]],
            [1],  # temporary IDs for seed
        )

        tess = merope.LaguerreTess_3D(L, sph_lag.getSpheres())
        grains = merope.MultiInclusions_3D()
        grains.setInclusions(tess)

        ids = grains.getAllIdentifiers()
        grains.addLayer(ids, phase_boundary, delta_phys)
        grains.changePhase(ids, [1 for _ in ids])  # temp

        # -------------------------
        # 4) merge inter-pores + grains
        # -------------------------
        map_inter_boundary_to_grains = {
            phase_inter: phase_grains,
            phase_boundary: phase_grains
        }

        struct_inter_on_grains = merope.Structure_3D(
            inter_pores,
            grains,
            map_inter_boundary_to_grains
        )

        # -------------------------
        # 5) overlay intra-pores, force (0 → 2)
        # -------------------------
        final_overlay_remap = {phase_grains: 2}

        final_struct = merope.Structure_3D(
            struct_inter_on_grains,
            intra_pores,
            final_overlay_remap
        )

        return final_struct


    # ------------------------------------------------------------------
    # 3) Public API: run distributed / interconnected cases
    # ------------------------------------------------------------------

    def run_distributed_case(self, seed, p_target,safe_geometry=True):
        """
        Costruisce, voxelizza, lancia AMITEX e ritorna dizionario risultati
        per il caso 'distributed porosity'.
        """
        if safe_geometry:
            # ---------------------------------------------------------
            # 0. clamp L_RVE se è troppo grande rispetto al raggio dei pori
            # ---------------------------------------------------------
            R_pore = float(self.inclR_intra)
            L_RVE = float(self.L[0])

            ratio = L_RVE / R_pore

            if ratio > 20.0:
                L_new = 20.0 * R_pore
                print(f"[INFO] L_RVE too large (ratio={ratio:.2f}), reducing to {L_new}")
                self.L = [L_new, L_new, L_new]
                L_RVE = L_new
            else:
                print(f"[INFO] Using L_RVE = {L_RVE} (ratio={ratio:.2f})")

            # ---------------------------------------------------------
            # 1. calcolo n3D minimo con L_RVE aggiornato
            # ---------------------------------------------------------
            n3D_min = int(np.ceil(self.suggest_n3D(self.inclR_intra)))

            # ---------------------------------------------------------
            # 2. aggiorno n3D PRIMA di ogni altra operazione
            # ---------------------------------------------------------
            if n3D_min > self.n3D:
                print(f"[INFO] n3D too small ({self.n3D}), increasing to {n3D_min}")
                self.n3D = n3D_min
            else:
                self.n3D = n3D_min
                print(f"[INFO] Using n3D = {self.n3D}")

            # 3. ora posso fare il check sulla geometria
            self._check_geometry_scales()

        # 4. directory caso
        case_dir = self._case_dir(
            "distributed",
            f"p{p_target:.3f}".replace(".", "_"),
            f"seed_{seed}"
        )

        # 5. microstruttura
        structure = self.build_distributed_structure(seed, p_target)

        # 6. voxelizza
        grid, phase_fracs = self._voxelize_and_assign(structure)

        p_por_meas = phase_fracs.get(self.phase_pore, 0.0)

        # esporta VTK + Coeffs
        printer = merope.vox.vtk_printer_3D()
        vtk_path = os.path.join(case_dir, "Zone.vtk")
        coeff_path = os.path.join(case_dir, "Coeffs.txt")
        printer.printVTK_segmented(
            grid, vtk_path, coeff_path, nameValue="MaterialId"
        )

        # lancia AMITEX
        kvals = self._run_amitex_in_dir(case_dir, vtk_name="Zone.vtk")
        kmean = float(np.mean(kvals))

        # geometric metrics
        L_RVE = float(self.L[0])
        R_pore = float(self.inclR_intra)
        L_voxel = L_RVE / float(self.n3D)

        crit1 = L_RVE / R_pore       # L_RVE / R_pore
        crit2 = R_pore / L_voxel     # R_pore / L_voxel

        return {
            "case_type": "distributed",
            "seed": seed,
            "p_target": p_target,
            "p_por_meas": p_por_meas,
            "p_delta": 0.0,
            "p_intra": p_por_meas,
            "delta_ratio": 0.0,
            "delta_phys": 0.0,
            "Kxx": kvals[0],
            "Kyy": kvals[1],
            "Kzz": kvals[2],
            "Kmean": kmean,
            "crit1_LRVE_over_Rpore": crit1,
            "crit2_Rpore_over_Lvoxel": crit2,
        }

    def run_interconnected_case(self,
                                seed,
                                p_delta,
                                p_intra,
                                delta_phys):

        p_target = p_delta + p_intra

        case_dir = self._case_dir(
            f"p{p_target:.3f}".replace(".", "_"),
            f"delta_{delta_phys:.4f}".replace(".", "_"),
            f"seed_{seed}"
        )

        structure = self.build_interconnected_structure(
            seed, p_delta, p_intra, delta_phys
        )
        grid, phases = self._voxelize_and_assign(structure)

        p_meas = phases.get(self.phase_pore, 0.0)

        printer = merope.vox.vtk_printer_3D()
        printer.printVTK_segmented(
            grid,
            os.path.join(case_dir, "Zone.vtk"),
            os.path.join(case_dir, "Coeffs.txt"),
            nameValue="MaterialId"
        )

        kvals = self._run_amitex_in_dir(case_dir)
        kmean = float(np.mean(kvals))

        return dict(
            case_type="interconnected",
            seed=seed,
            p_target=p_target,
            p_meas=p_meas,
            p_delta=p_delta,
            p_intra=p_intra,
            delta_phys=delta_phys,
            Kxx=kvals[0],
            Kyy=kvals[1],
            Kzz=kvals[2],
            Kmean=kmean
        )