"""
spherical_inclusions.py

This script builds a porous microstructure with spherical inclusions in a solid
matrix, voxelizes it with Merope, and runs Amitex to compute the effective
thermal conductivity tensor. Results are post-processed to extract the mean
conductivity and off-diagonal error, providing a baseline reference case.
"""

###############################################################
# SCRIPT TO COMPUTE:
#   - Effective thermal conductivity (K)
#   - Porosity
#   - For one phase: spherical inclusions only
#
# This is a minimal version: single porosity, single voxel grid.
###############################################################

import os
import time
from math import sqrt
import matplotlib.pyplot as plt

import sac_de_billes
import merope
import interface_amitex_fftp.amitex_wrapper as amitex
import interface_amitex_fftp.post_processing as amitex_out


# ---------------------------------------------------------------------------
# INPUT PARAMETERS
# ---------------------------------------------------------------------------

# RVE and inclusions
porosity = 0.175
R_pore = 1.0 # so that the ratio L_RVE / R_pore is just L_RVE

resolution = 2 # target_resolution --> L_1voxel = R_pore / 4

seed = 0

L_RVE = [10, 15, 20, 25, 30, 35, 40, 45]  # edge lengths of the RVE

# materials
k_matrix = 1.0
k_gas = 1e-3
conductivities = [k_matrix, k_gas]

# results folder
results_folder = "spherical_inclusions"

if not os.path.exists(results_folder):
    os.makedirs(results_folder)

# ---------------------------------------------------------------------------
# FUNCTIONS
# ---------------------------------------------------------------------------
def build_voxelized_structure(domain_size, seed, radius, porosity, conductivities, voxellation):
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

    # Step 5. Analyze phase fractions (porosity is phase 1 here)
    analyzer = merope.vox.GridAnalyzer_3D()
    phase_fractions = analyzer.compute_percentages(grid)
    porosity_calc = phase_fractions[1]

    # Step 6. Apply homogenization
    grid.apply_homogRule(merope.HomogenizationRule.Voigt, conductivities)

    # Step 7. Export VTK + coeffs
    printer = merope.vox.vtk_printer_3D()
    printer.printVTK_segmented(grid, "Zone.vtk", "Coeffs.txt", nameValue="MaterialId")

    return porosity_calc

    
def run_amitex():
    """Run Amitex on Zone.vtk"""
    num_procs = 2
    amitex.computeThermalCoeff("Zone.vtk", num_procs)
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
    error = sqrt(sum(v * v for v in offdiag))
    return k_mean, error, diag


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    results = []
    for L in L_RVE:
        print(f"\n=== Running for L = {L} ===")

        num_voxels = int(resolution * L / R_pore)
        voxellation = [num_voxels]*3

        L_voxel = L / num_voxels
        a_param = R_pore / L_voxel   # should be ≈ resolution

        if abs(a_param - resolution) > 1e-6:
            print(f"Warning: adjusted a={a_param:.3f} instead of {resolution}")

        # create folder for each L
        case_folder = os.path.join(results_folder, f"L_{L}")
        if not os.path.exists(case_folder):
            os.mkdir(case_folder)

        # build files inside case_folder without chdir
        cwd_backup = os.getcwd()
        os.chdir(case_folder)

        porosity_calc = build_voxelized_structure(
            [L, L, L], seed, R_pore, porosity, conductivities, voxellation
        )

        matrix = read_conductivity_matrix() if run_amitex() else None

        if matrix:
            # conductivity
            k_mean, error, diag = process_matrix(matrix)

            results.append((L, L_voxel, a_param, k_mean, error, porosity_calc))

            with open("results.txt", "w") as f:
                f.write(f"L_RVE: {L}\n")
                f.write(f"N_voxel: {num_voxels}\n")
                f.write(f"L_1voxel: {L_voxel:.6f}\n")
                f.write(f"a (target): {resolution:.3f}\n")
                f.write(f"a (realized): {a_param:.3f}\n")
                f.write(f"Target porosity: {porosity:.3f}\n")
                f.write(f"Calculated porosity: {porosity_calc:.3f}\n")
                f.write(f"Kxx: {diag[0]:.6f}, Kyy: {diag[1]:.6f}, Kzz: {diag[2]:.6f}\n")
                f.write(f"K_mean: {k_mean:.6f}, Error: {error:.6e}\n")

            print(f"Results for L={L} stored (target φ={porosity:.3f}, calc φ={porosity_calc:.3f})")

        os.chdir(cwd_backup)

    # plot results
    L_vals   = [r[0] for r in results]
    a_vals   = [r[2] for r in results]
    k_vals   = [r[3] for r in results]
    err_vals = [r[4] for r in results]
    phi_vals = [r[5] for r in results]

    plt.figure(figsize=(8,6))
    plt.plot(L_vals, k_vals, "o-", label="Mean conductivity")
    plt.xlabel("L_{RVE} / R")
    plt.ylabel("Effective conductivity K")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_folder, "convergence_L.png"))
    plt.show()

    plt.figure(figsize=(8,6))
    plt.plot(L_vals, err_vals, "s--", color="red", label="Off-diagonal error")
    plt.xlabel("RVE size L")
    plt.ylabel("Error (anisotropy measure)")
    plt.yscale("log")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_folder, "anisotropy_error.png"))
    plt.show()

    plt.figure(figsize=(8,6))
    plt.plot(L_vals, phi_vals, "d-", color="green", label="Calculated porosity")
    plt.axhline(porosity, color="black", linestyle="--", label="Target porosity")
    plt.xlabel("RVE size L")
    plt.ylabel("Porosity")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_folder, "porosity_vs_L.png"))
    plt.show()

    plt.figure(figsize=(8,6))
    plt.plot(a_vals, [r[3] for r in results], "o-", label="Mean conductivity")
    plt.xlabel("Resolution parameter a = R_pore / L_voxel")
    plt.ylabel("Effective conductivity K")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_folder, "convergence_a.png"))
    plt.show()

if __name__ == "__main__":
    main()