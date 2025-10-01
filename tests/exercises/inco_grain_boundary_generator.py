"""
grain_boundary_generator.py

This script generates a polycrystal microstructure with:
- spherical inclusions (pores) to control porosity,
- a Laguerre tessellation to represent grains,
- an additional grain boundary layer (delta).

The structure is voxelized with Merope and exported to VTK/coeff files
for further analysis or homogenization with Amitex.

Objectives:
- Study the effect of porosity, grain size, and boundary layer thickness
  on effective thermal conductivity.
- Provide a workflow for generating grain + boundary microstructures.

Notes:
- Phase 1 is the porous phase.
- Homogenization rule is Voigt, which requires voxel_rule = Average.
"""

import os
import time
import shutil
import sac_de_billes
import merope
import archi_merope as arch
import interface_amitex_fftp.amitex_wrapper as amitex
import interface_amitex_fftp.post_processing as amitex_out


# ---------------------------------------------------------------------------
# INPUT PARAMETERS
# ---------------------------------------------------------------------------

# Geometry and voxellation
domain_size = [10, 10, 10]    # RVE dimensions
num_voxels = 50               # voxel grid resolution → 
seed = 0                      # random seed for reproducibility

voxel_rule = merope.vox.VoxelRule.Average
homog_rule = merope.HomogenizationRule.Voigt  # requires voxel_rule = Average

# Output
results_folder = "grain_boundary_generator"
vtk_filename = "Zone.vtk"
coeff_filename = "Coeffs.txt"

# Microstructure parameters
inclusion_radius = 0.1     # pore radius
inclusion_fraction = 0.2   # pore volume fraction
grain_size = 4             # average grain radius (Laguerre tessellation)
grain_fill_fraction = 1    # must be 1 → fully fill RVE with grains
boundary_thickness = 0.001 # grain boundary layer thickness

# Phase definitions
phase_pores = 2
phase_boundary = 3
phase_grains = 0

# Material properties
k_matrix = 1.0   # thermal conductivity of matrix
k_gas = 1e-3     # thermal conductivity of pores
conductivities = [k_matrix, k_matrix, k_gas]


# ---------------------------------------------------------------------------
# UTILITIES
# ---------------------------------------------------------------------------

def ensure_folder(path: str):
    """Create a folder fresh (remove it if already exists)."""
    if os.path.exists(path):
        print(f"Folder {path} already exists → removing it.")
        shutil.rmtree(path)
    os.mkdir(path)


def change_dir(path: str):
    """Switch working directory with small delay."""
    time.sleep(0.1)
    os.chdir(path)
    print(f">>> Entered {path}")
    time.sleep(0.1)


# ---------------------------------------------------------------------------
# MICROSTRUCTURE GENERATION
# ---------------------------------------------------------------------------

def build_grain_boundary_structure(
    num_voxels, domain_size, seed,
    incl_radius, incl_fraction,
    grain_size, grain_fill_fraction,
    boundary_thickness,
    phase_pores, phase_grains, phase_boundary,
    voxel_rule, homog_rule, conductivities,
    vtk_filename, coeff_filename
):
    """
    Build polycrystal structure with pores and grain boundary, voxelize and export.
    """

    # Step 1. Create spherical inclusions (pores)
    sph_incl = merope.SphereInclusions_3D()
    sph_incl.setLength(domain_size)
    sph_incl.fromHisto(seed, sac_de_billes.TypeAlgo.BOOL, 0.,
                       [[incl_radius, incl_fraction]], [phase_pores])
    pores = merope.MultiInclusions_3D()
    pores.setInclusions(sph_incl)

    # Step 2. Generate Laguerre tessellation (grains)
    lag_sph = merope.SphereInclusions_3D()
    lag_sph.setLength(domain_size)
    lag_sph.fromHisto(seed, sac_de_billes.TypeAlgo.RSA, 0.,
                      [[grain_size, grain_fill_fraction]], [1])
    polycrystal = merope.LaguerreTess_3D(domain_size, lag_sph.getSpheres())

    grains = merope.MultiInclusions_3D()
    grains.setInclusions(polycrystal)

    # Step 3. Add grain boundary layer
    grains.addLayer(grains.getAllIdentifiers(), phase_boundary, boundary_thickness)
    grains.changePhase(grains.getAllIdentifiers(),
                       [1 for _ in grains.getAllIdentifiers()])

    # Step 4. Merge pores and grains into final structure
    phase_map = {phase_pores: phase_grains, phase_boundary: phase_grains}
    structure = merope.Structure_3D(pores, grains, phase_map)

    # Step 5. Voxelization
    grid_params = merope.vox.create_grid_parameters_N_L_3D(
        [num_voxels, num_voxels, num_voxels], domain_size
    )
    grid = merope.vox.GridRepresentation_3D(structure, grid_params, voxel_rule)

    analyzer = merope.vox.GridAnalyzer_3D()
    phase_fractions = analyzer.compute_percentages(grid)
    analyzer.print_percentages(grid)

    # Apply homogenization
    grid.apply_homogRule(homog_rule, conductivities)

    # Export vtk and coefficients
    printer = merope.vox.vtk_printer_3D()
    printer.printVTK_segmented(grid, vtk_filename, coeff_filename, nameValue="MaterialId")

    return phase_fractions


# ---------------------------------------------------------------------------
# RUN AMITEX (OPTIONAL)
# ---------------------------------------------------------------------------

def run_amitex(vtk_filename):
    """Run Amitex FFT solver and return homogenized conductivity matrix."""
    num_procs = 2
    amitex.computeThermalCoeff(vtk_filename, num_procs)
    return amitex_out.printThermalCoeff(".")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    # Prepare results folder
    ensure_folder(results_folder)
    change_dir(results_folder)

    # Create subfolder for this resolution
    ensure_folder(str(num_voxels))
    change_dir(str(num_voxels))

    # Build structure
    fractions = build_grain_boundary_structure(
        num_voxels, domain_size, seed,
        inclusion_radius, inclusion_fraction,
        grain_size, grain_fill_fraction,
        boundary_thickness,
        phase_pores, phase_grains, phase_boundary,
        voxel_rule, homog_rule, conductivities,
        vtk_filename, coeff_filename
    )

    print("Voxelization completed.")
    print("Phase fractions:", fractions)

    # Optionally run Amitex
    matrix = run_amitex(vtk_filename)
    print("Homogenized conductivity matrix:\n", matrix)

if __name__ == "__main__":
    main()

# >>> Entered grain_boundary_generator
# >>> Entered 300
# Volume fraction : 0.223141
# ----------------------------
# ----------------------------
# AlgoRSA::proceed()
# Fully packed
# Nb of spheres : 1
# Volume fraction : 0.268083
# ----------------------------
# phase 0 : 99.987925%
# phase 2 : 0.012075%
# Voxelization completed.