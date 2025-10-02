"""
utils_microstructure.py

Helper functions for building voxelized structures, running AMITEX,
and post-processing results.
"""

from math import sqrt
import sac_de_billes
import merope
import interface_amitex_fftp.amitex_wrapper as amitex
import interface_amitex_fftp.post_processing as amitex_out

# ---------------------------------------------------------------------------
# FUNCTIONS
# ---------------------------------------------------------------------------

def build_voxelized_structure(domain_size, seed, radius, porosity, conductivities, voxellation, homog_rule=merope.HomogenizationRule.Voigt):
    """Generate a voxelized microstructure with spherical inclusions"""

    # Step 1. Spherical inclusions
    sph = merope.SphereInclusions_3D()
    sph.setLength(domain_size)
    sph.fromHisto(seed, sac_de_billes.TypeAlgo.RSA, 0., [[radius, porosity]], [1])

    multi = merope.MultiInclusions_3D()
    multi.setInclusions(sph)

    # Step 2. Create Structure
    structure = merope.Structure_3D(multi)

    # Step 3. Create grid parameters
    grid_params = merope.vox.create_grid_parameters_N_L_3D(voxellation, domain_size)

    # Step 4. Build grid
    grid = merope.vox.GridRepresentation_3D(structure, grid_params, merope.vox.VoxelRule.Average)

    # Step 5. Analyze phase fractions
    analyzer = merope.vox.GridAnalyzer_3D()
    phase_fractions = analyzer.compute_percentages(grid)
    porosity_calc = phase_fractions[1] # (porosity is phase 1 here)

    # print(dir(merope.HomogenizationRule))
    # ['Largest', 'Reuss', 'Smallest', 'Voigt', ...]
    grid.apply_homogRule(homog_rule, conductivities)

    # Step 7. Export VTK + coeffs
    printer = merope.vox.vtk_printer_3D()
    printer.printVTK_segmented(grid, "Zone.vtk", "Coeffs.txt", nameValue="MaterialId")

    return porosity_calc


def run_amitex(filename = "Zone.vtk"):
    """Run Amitex-FFTP solver to compute effective thermal conductivity."""
    num_procs = 2
    amitex.computeThermalCoeff(filename, num_procs)
    return amitex_out.printThermalCoeff(".")


def read_conductivity_matrix(file_path="thermalCoeff_amitex.txt"):
    """Read conductivity matrix from Amitex output"""
    with open(file_path, "r") as f:
        return [list(map(float, line.split())) for line in f]


def process_matrix(matrix):
    """Extract mean conductivity and anisotropy error"""
    diag = [matrix[i][i] for i in range(3)]
    offdiag = [matrix[i][j] for i in range(3) for j in range(3) if j != i]
    k_mean = sum(diag) / 3
    abs_error = sqrt(sum(v * v for v in offdiag))
    return k_mean, abs_error, diag

