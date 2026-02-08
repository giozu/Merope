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
            multi.addLayer(ids, 3, delta) # Phase 3 = Interconnected Porosity
        
        multi.changePhase(ids, [1 for _ in ids]) # Phase 1 = Matrix
        return multi

    def generate_spheres(self, radii_phi_list, phase_id=2):
        """Creates distributed spherical inclusions."""
        sph = merope.SphereInclusions_3D()
        sph.setLength(self.L)
        sph.fromHisto(self.seed, sac_de_billes.TypeAlgo.BOOL, 0., radii_phi_list, [phase_id]*len(radii_phi_list))
        
        multi = merope.MultiInclusions_3D()
        multi.setInclusions(sph)
        return multi

    def voxellate(self, structure, K_values, vtk_name="structure.vtk", coeffs_name="Coeffs.txt"):
        """Converts analytical structure to voxels and applies thermal properties."""
        grid = merope.vox.GridRepresentation_3D(structure, self.grid_params, merope.vox.VoxelRule.Average)
        grid.apply_homogRule(merope.HomogenizationRule.Voigt, K_values)
        
        printer = merope.vox.vtk_printer_3D()
        printer.printVTK_segmented(grid, vtk_name, coeffs_name, nameValue="MaterialId")
        
        analyzer = merope.vox.GridAnalyzer_3D()
        fractions = analyzer.compute_percentages(grid)
        return fractions