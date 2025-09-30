###############################################################
# SCRIPT TO COMPUTE:
#   - Effective thermal conductivity (K)
#   - Porosity
#   - Effect of grain aspect ratio and boundary layer thickness
#
# NOTE: Phase 1 is the porous phase
###############################################################

import os
import sac_de_billes
import merope
import shutil
import numpy as np

import archi_merope as arch
import interface_amitex_fftp.amitex_wrapper as amitex
import interface_amitex_fftp.post_processing as amitex_out

use_amitex = True  # set to True if you want to run Amitex

# ---------------------------------------------------------------------------
# VOXEL & CELL INPUTS
# ---------------------------------------------------------------------------

domain_size = [10, 10, 10]     # RVE dimensions
RVE_size = domain_size[1]      # reference RVE size
num_voxels = 100               # voxel resolution → grid = 100³
random_seed = 0                # base random seed
num_seeds = 1                  # number of seeds for statistical variability

voxel_rule = merope.vox.VoxelRule.Average
homog_rule = merope.HomogenizationRule.Voigt  # requires voxel_rule = Average


# ---------------------------------------------------------------------------
# FOLDERS AND OUTPUT FILES
# ---------------------------------------------------------------------------

results_folder = "Result"  
results_path = os.path.join(os.getcwd(), results_folder)

raw_output_file = "Porosity_conduct_results.txt"
vtk_filename = "crack_structure.vtk"
coeff_filename = "Coeffs.txt"


# ---------------------------------------------------------------------------
# MICROSTRUCTURE PARAMETERS
# ---------------------------------------------------------------------------

inclusion_radius = 0.3     # radius of spherical inclusions (pores)
grain_size = 3             # mean grain size parameter
grain_fill_fraction = 1    # must be 1 to fully fill RVE with grains

# Aspect ratio variation (anisotropy of grains)
aspect_ratios_y = np.linspace(1, 0.1, 2)   # y-axis aspect ratios

# porosity fractions to investigate
inclusion_fractions = np.linspace(0.1, 0.2, 2)

boundary_thickness = 1     # boundary layer thickness
boundary_to_grain_ratio = boundary_thickness / grain_size

# Pack grain parameters for Merope
grain_params = [grain_size, grain_fill_fraction]


# ---------------------------------------------------------------------------
# PHASE DEFINITIONS
# ---------------------------------------------------------------------------

phase_pores = 2
phase_boundary = 3
phase_grains = 0

# thermal conductivities
k_matrix = 1.0
k_gas = 1e-3
conductivities = [k_matrix, k_matrix, k_gas]


# ---------------------------------------------------------------------------
# FUNCTION: Build and voxelize structure
# ---------------------------------------------------------------------------

def build_voxelized_structure(
    num_voxels, domain_size, seed, inclusion_radius, inclusion_fraction,
    grain_params, aspect_ratio_y, phase_pores, phase_grains,
    phase_boundary, boundary_thickness, voxel_rule, conductivities,
    vtk_filename, coeff_filename
):
    """
    Build a polycrystal structure with inclusions and boundary layer,
    voxelize it, and return porosity fraction.
    """

    # Step 1. Add spherical inclusions (pores)
    sphere_inclusions = merope.SphereInclusions_3D()
    sphere_inclusions.setLength(domain_size)
    sphere_inclusions.fromHisto(seed, sac_de_billes.TypeAlgo.BOOL, 0., [[inclusion_radius, inclusion_fraction]], [phase_pores])
    multi_inclusions_pores = merope.MultiInclusions_3D()
    multi_inclusions_pores.setInclusions(sphere_inclusions)

    # Step 2. Build Laguerre tessellation (grains)
    sphere_for_laguerre = merope.SphereInclusions_3D()
    sphere_for_laguerre.setLength(domain_size)
    sphere_for_laguerre.fromHisto(seed, sac_de_billes.TypeAlgo.RSA, 0., [grain_params], [1])

    polycrystal = merope.LaguerreTess_3D(domain_size, sphere_for_laguerre.getSpheres())

    # Apply anisotropy (aspect ratio along y-axis)
    aspect_ratio_x = 1
    aspect_ratio_z = 1 / (aspect_ratio_x * aspect_ratio_y)
    polycrystal.setAspRatio([aspect_ratio_x, aspect_ratio_y, aspect_ratio_z])

    multi_inclusions_grains = merope.MultiInclusions_3D()
    multi_inclusions_grains.setInclusions(polycrystal)

    # Add boundary layer (delta)
    multi_inclusions_grains.addLayer(multi_inclusions_grains.getAllIdentifiers(), phase_boundary, boundary_thickness)
    multi_inclusions_grains.changePhase(
        multi_inclusions_grains.getAllIdentifiers(), [1 for _ in multi_inclusions_grains.getAllIdentifiers()]
    )

    # Step 3. Combine structures
    phase_map = {phase_pores: phase_grains, phase_boundary: phase_grains}
    structure = merope.Structure_3D(multi_inclusions_pores, multi_inclusions_grains, phase_map)

    # Step 4. Voxelization
    grid_params = merope.vox.create_grid_parameters_N_L_3D([num_voxels, num_voxels, num_voxels], domain_size)
    grid = merope.vox.GridRepresentation_3D(structure, grid_params, voxel_rule)

    analyzer = merope.vox.GridAnalyzer_3D()
    phase_fractions = analyzer.compute_percentages(grid)
    analyzer.print_percentages(grid)

    # Apply homogenization
    grid.apply_homogRule(homog_rule, conductivities)

    # Export vtk
    printer = merope.vox.vtk_printer_3D()
    printer.printVTK_segmented(grid, vtk_filename, coeff_filename, nameValue="MaterialId")

    # porosity fraction = fraction of pores
    porosity = phase_fractions[phase_pores]
    return porosity


# ---------------------------------------------------------------------------
# FUNCTION: Run Amitex
# ---------------------------------------------------------------------------

def run_amitex():
    """Run Amitex solver and return homogenized conductivity matrix"""
    num_procs = 6
    amitex.computeThermalCoeff(vtk_filename, num_procs)
    return amitex_out.printThermalCoeff(".")


# ---------------------------------------------------------------------------
# UTILITIES
# ---------------------------------------------------------------------------

def write_values_to_file(file_output, values, porosity, aspect_ratio, seed):
    """Write conductivity values (xx, yy, zz, mean) to file"""
    mean_val = sum(values) / len(values)
    with open(file_output, "a") as f:
        f.write(
            f"Porosity_{porosity:.2f}\tAspectRatioY_{aspect_ratio:.2f}\tSeed_{seed}\t"
            f"{values[0]:.4f}\t{values[1]:.4f}\t{values[2]:.4f}\t{mean_val:.4f}\n"
        )


def read_matrix_from_file(file_path):
    """Read conductivity matrix from Amitex output file"""
    with open(file_path, "r") as f:
        return [list(map(float, line.split())) for line in f.readlines()]


def extract_diagonal_values(matrix):
    """Extract diagonal values (Kxx, Kyy, Kzz)"""
    return [matrix[i][i] for i in range(3)]


def update_aggregated_file(file_aggregated, porosity, aspect_ratio, seed, values,
                           inclusion_radius, grain_size, RVE_size, num_voxels, k_matrix, k_gas):
    """Append results and metadata to aggregated output file"""
    mean_val = sum(values) / len(values)

    if not os.path.exists(file_aggregated):
        # create file and write header
        with open(file_aggregated, "w") as f:
            f.write("Input Parameters:\n")
            f.write(f"Boundary thickness: {boundary_thickness}\n")
            f.write(f"RVE size: {RVE_size}\n")
            f.write(f"Inclusion radius: {inclusion_radius}\n")
            f.write(f"Mean grain size: {grain_size}\n")
            f.write(f"Voxel grid: {num_voxels}\n")
            f.write(f"Kmatrix: {k_matrix}\n")
            f.write(f"Kgases: {k_gas}\n")
            f.write("Porosity\tAspectRatioY\tSeed\tK_xx\tK_yy\tK_zz\tK_mean\n")

    with open(file_aggregated, "a") as f:
        f.write(
            f"{porosity:.4f}\t{aspect_ratio:.2f}\t{seed}\t"
            f"{values[0]:.4f}\t{values[1]:.4f}\t{values[2]:.4f}\t{mean_val:.4f}\n"
        )

# ---------------------------------------------------------------------------
# MAIN LOOP
# ---------------------------------------------------------------------------

def main():
    # reset raw output file
    if os.path.exists(raw_output_file):
        open(raw_output_file, "w").close()

    # remove/create main results folder
    if os.path.exists(results_folder):
        shutil.rmtree(results_folder)
    os.mkdir(results_folder)
    os.chdir(results_folder)

    aggregated_file = os.path.join(os.getcwd(), "aggregated_results.txt")

    # loop over porosity, aspect ratio, seeds
    for inclusion_fraction in inclusion_fractions:
        for aspect_ratio_y in aspect_ratios_y:
            for seed in range(num_seeds):
                seed_folder = f"Seed_{seed}"
                os.makedirs(seed_folder, exist_ok=True)
                os.chdir(seed_folder)

                # build structure and compute porosity
                porosity = build_voxelized_structure(
                    num_voxels, domain_size, seed, inclusion_radius, inclusion_fraction,
                    grain_params, aspect_ratio_y, phase_pores, phase_grains,
                    phase_boundary, boundary_thickness, voxel_rule,
                    conductivities, vtk_filename, coeff_filename
                )

                if use_amitex:
                    run_amitex()
                    matrix = read_matrix_from_file("thermalCoeff_amitex.txt")
                    diag_values = extract_diagonal_values(matrix)

                    write_values_to_file(raw_output_file, diag_values, porosity, aspect_ratio_y, seed)
                    update_aggregated_file(
                        aggregated_file, porosity, aspect_ratio_y, seed, diag_values,
                        inclusion_radius, grain_size, RVE_size, num_voxels, k_matrix, k_gas
                    )

                os.chdir("..")  # back to Result/

    os.chdir("..")  # back to base dir

if __name__ == "__main__":
    main()
