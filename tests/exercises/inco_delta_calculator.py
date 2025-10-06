import os
import sac_de_billes
import merope
import shutil

import interface_amitex_fftp.amitex_wrapper as amitex
import interface_amitex_fftp.post_processing as amitex_out

use_amitex = False  # set to True if you want to run Amitex

# ---------------------------------------------------------------------------
# VOXEL & CELL INPUTS
# ---------------------------------------------------------------------------

a = 10
ratio = 25
R_pore = 1.0

L_RVE = ratio * R_pore

domain_size = [L_RVE, L_RVE, L_RVE]     # RVE dimensions

num_voxels = int(a * L_RVE / R_pore)

num_seeds = 1                           # number of seeds for statistical variability

voxel_rule = merope.vox.VoxelRule.Average
homog_rule = merope.HomogenizationRule.Voigt  # requires voxel_rule = Average

grain_size = L_RVE / 2.0     # mean grain size parameter
grain_size = 4.0
grain_fill_fraction = 1     # must be 1 to fully fill RVE with grains
grain_params = [grain_size, grain_fill_fraction]

# Aspect ratio variation (anisotropy of grains)
aspect_ratio_y_values = [1.0] # 1.0, isotropic grains

porosity_values = [0.2]

L_voxel = L_RVE / num_voxels
boundary_thickness = 2 * L_voxel

print(f"num_voxels = {num_voxels}")
print(f"L_voxel = {L_voxel}")
print(f"boundary_thickness = {boundary_thickness}")

boundary_thickness = 0.001
boundary_to_grain_ratio = boundary_thickness / grain_size


# ---------------------------------------------------------------------------
# PHASE DEFINITIONS
# ---------------------------------------------------------------------------

phase_pores = 2
phase_boundary = 3
phase_grains = 0

k_matrix = 1.0
k_gb = 1.0
k_gas = 1e-3

conductivities = [k_matrix, k_gb, k_gas]

# ---------------------------------------------------------------------------
# FOLDERS AND OUTPUT FILES
# ---------------------------------------------------------------------------

results_folder = "inco_delta_calculator"  
results_path = os.path.join(os.getcwd(), results_folder)

raw_output_file = "Porosity_conduct_results.txt"
vtk_filename = "Zone.vtk"
coeff_filename = "Coeffs.txt"

# ---------------------------------------------------------------------------
# FUNCTION: Build and voxelize structure
# ---------------------------------------------------------------------------

def structure_spherical_inclusions(
    num_voxels, domain_size, seed, R_pore, inclusion_fraction,
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
    sphere_inclusions.fromHisto(
        seed, 
        sac_de_billes.TypeAlgo.BOOL, 0., 
        [[R_pore, inclusion_fraction]], 
        [phase_pores]
    )

    pores = merope.MultiInclusions_3D()
    pores.setInclusions(sphere_inclusions)

    # Step 2. Build Laguerre tessellation (grains)
    sphere_for_laguerre = merope.SphereInclusions_3D()
    sphere_for_laguerre.setLength(domain_size)
    sphere_for_laguerre.fromHisto(seed, sac_de_billes.TypeAlgo.RSA, 0., [grain_params], [1])

    polycrystal = merope.LaguerreTess_3D(domain_size, sphere_for_laguerre.getSpheres())

    # Apply anisotropy (aspect ratio along y-axis)
    polycrystal.setAspRatio([1.0, aspect_ratio_y, 1.0/aspect_ratio_y])

    grains = merope.MultiInclusions_3D()
    grains.setInclusions(polycrystal)

    # Add boundary layer (delta)
    grains.addLayer(grains.getAllIdentifiers(), phase_boundary, boundary_thickness)
    grains.changePhase(grains.getAllIdentifiers(),
                       [1 for _ in grains.getAllIdentifiers()])

    # Step 3. Merge pores and grains into final structure
    phase_map = {
        phase_pores: phase_grains,
        phase_boundary: phase_grains
    }
    
    structure = merope.Structure_3D(
        pores, 
        grains, 
        phase_map
    )

    # Step 4. Voxelization
    grid_params = merope.vox.create_grid_parameters_N_L_3D([num_voxels, num_voxels, num_voxels], domain_size)
    grid = merope.vox.GridRepresentation_3D(structure, grid_params, voxel_rule)

    # warn if GB under-resolved
    vox = domain_size[0] / num_voxels
    if boundary_thickness < vox:
        print(f"[WARN] GB thickness δ={boundary_thickness} < voxel size {vox:.3g} → GB may be under-resolved.")

    analyzer = merope.vox.GridAnalyzer_3D()
    phase_fractions = analyzer.compute_percentages(grid)

    analyzer.print_percentages(grid)

    # Apply homogenization
    grid.apply_homogRule(homog_rule, conductivities)

    # Export vtk
    printer = merope.vox.vtk_printer_3D()
    printer.printVTK_segmented(grid, vtk_filename, coeff_filename, nameValue="MaterialId")
    
    return phase_fractions


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
                           R_pore, grain_size, L_RVE, num_voxels, k_matrix, k_gas):
    """Append results and metadata to aggregated output file"""
    mean_val = sum(values) / len(values)

    if not os.path.exists(file_aggregated):
        # create file and write header
        with open(file_aggregated, "w") as f:
            f.write("Input Parameters:\n")
            f.write(f"Boundary thickness: {boundary_thickness}\n")
            f.write(f"RVE size: {L_RVE}\n")
            f.write(f"Inclusion radius: {R_pore}\n")
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
    for porosity in porosity_values:
        for aspect_ratio_y in aspect_ratio_y_values:
            for seed in range(num_seeds):
                seed_folder = f"Seed_{seed}"
                os.makedirs(seed_folder, exist_ok=True)
                os.chdir(seed_folder)

                # build structure and compute porosity
                porosity_calc = structure_spherical_inclusions(
                    num_voxels, domain_size, seed, R_pore, porosity,
                    grain_params, aspect_ratio_y, phase_pores, phase_grains,
                    phase_boundary, boundary_thickness, voxel_rule,
                    conductivities, vtk_filename, coeff_filename
                )

                if use_amitex:
                    run_amitex()
                    matrix = read_matrix_from_file("thermalCoeff_amitex.txt")
                    diag_values = extract_diagonal_values(matrix)

                    write_values_to_file(raw_output_file, diag_values, porosity_calc, aspect_ratio_y, seed)
                    update_aggregated_file(
                        aggregated_file, porosity_calc, aspect_ratio_y, seed, diag_values,
                        R_pore, grain_size, L_RVE, num_voxels, k_matrix, k_gas
                    )

                os.chdir("..")  # back to Result/

    os.chdir("..")  # back to base dir

if __name__ == "__main__":
    main()
