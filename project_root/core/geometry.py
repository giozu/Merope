import merope
import sac_de_billes
import numpy as np

class MicrostructureBuilder:
    def __init__(self, L, n3D, seed=0):
        self.L = L
        self.n3D = n3D
        self.seed = seed
        self.grid_params = merope.vox.create_grid_parameters_N_L_3D([n3D]*3, L)


    def generate_polycrystal(self, grain_radius, delta=0.0, aspect_ratio=[1.0, 1.0, 1.0]):
        """Creates a Laguerre tessellation with optional grain boundary layers."""
        sph = merope.SphereInclusions_3D()
        sph.setLength(self.L)
        # RSA used to seed the tessellation
        sph.fromHisto(self.seed, sac_de_billes.TypeAlgo.RSA, 0., [[grain_radius, 1.0]], [1])
        
        poly = merope.LaguerreTess_3D(self.L, sph.getSpheres())
        poly.setAspRatio(aspect_ratio)
        
        multi = merope.MultiInclusions_3D()
        multi.setInclusions(poly)
        
        ids = multi.getAllIdentifiers()
        if delta > 0:
            multi.addLayer(ids, 2, delta) # Usa la Fase 2, non la 3!

        multi.changePhase(ids, [1 for _ in ids]) # Fase 1 = Matrice

        return merope.Structure_3D(multi) # Restituisci Structure_3D, non MultiInclusions

    def generate_spheres(self, radii_phi_list, phase_id=2):
        """Creates distributed spherical inclusions."""
        sph = merope.SphereInclusions_3D()
        sph.setLength(self.L)
        sph.fromHisto(self.seed, sac_de_billes.TypeAlgo.BOOL, 0., radii_phi_list, [phase_id]*len(radii_phi_list))
        
        multi = merope.MultiInclusions_3D()
        multi.setInclusions(sph)
        return multi

    def generate_mixed_structure(self, grain_radius, delta, intra_pore_list):
        # 1. Polycrystal
        sph_grains = merope.SphereInclusions_3D()
        sph_grains.setLength(self.L)
        sph_grains.fromHisto(self.seed, sac_de_billes.TypeAlgo.RSA, 0., [[grain_radius, 1.0]], [1])
        poly = merope.LaguerreTess_3D(self.L, sph_grains.getSpheres())
        
        m_grains = merope.MultiInclusions_3D()
        m_grains.setInclusions(poly)
        grain_ids = m_grains.getAllIdentifiers()
        
        m_grains.changePhase(grain_ids, [1 for _ in grain_ids]) # Matrix = 1
        if delta > 0:
            m_grains.addLayer(grain_ids, 2, delta) # Cracks = 2
        
        # 2. Spheres
        m_spheres = merope.MultiInclusions_3D()
        sph_obj = merope.SphereInclusions_3D()
        sph_obj.setLength(self.L)
        sph_obj.fromHisto(self.seed, sac_de_billes.TypeAlgo.BOOL, 0., intra_pore_list, [2])
        m_spheres.setInclusions(sph_obj)

        # 3. Combine
        return merope.Structure_3D(m_spheres, m_grains, {1: 1, 2: 2})
        
    def voxellate(self, structure, K_values, vtk_name="structure.vtk"):
        # Create Grid Representation
        grid_repr = merope.vox.GridRepresentation_3D(structure, self.grid_params, merope.vox.VoxelRule.Average)
        
        # Apply Thermal Coeffs (Crucial: bakes the phases into the grid)
        grid_repr.apply_homogRule(merope.HomogenizationRule.Voigt, K_values)
        
        # SAFE ANALYZER LOGIC
        fractions = {1: 0.0, 2: 0.0}
        try:
            analyzer = merope.vox.GridAnalyzer_3D()
            # Try computing from structure + params directly (often more stable)
            fractions = analyzer.compute_percentages(grid_repr)
            print(f"Voxellation complete. Detected Phases: {fractions}")
        except Exception as e:
            print(f"Analyzer skipped due to: {e}")

        # Export for Amitex
        printer = merope.vox.vtk_printer_3D()
        printer.printVTK_segmented(grid_repr, vtk_name, "Coeffs.txt", nameValue="MaterialId")
        
        return fractions