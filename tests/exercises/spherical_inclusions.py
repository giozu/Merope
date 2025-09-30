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

import sac_de_billes
import merope
import interface_amitex_fftp.amitex_wrapper as amitex
import interface_amitex_fftp.post_processing as amitex_out


# ---------------------------------------------------------------------------
# INPUT PARAMETERS
# ---------------------------------------------------------------------------

# RVE and inclusions
porosity = 0.175
inclusion_radius = 1.0
seed = 0
domain_size = [10, 10, 10]   # cubic RVE

# materials
k_matrix = 1.0
k_gas = 1e-9
conductivities = [k_matrix, k_gas]

# voxel resolution
num_voxels = 100

# results folder
results_folder = "spherical_inclusions"


# ---------------------------------------------------------------------------
# FUNCTIONS
# ---------------------------------------------------------------------------

def build_voxelized_structure(domain_size, seed, radius, porosity, conductivities, voxellation):
    """Generate a voxelized microstructure with spherical inclusions"""
    start = time.time()

    # spherical inclusions
    sph = merope.SphereInclusions_3D()
    sph.setLength(domain_size)
    sph.fromHisto(seed, sac_de_billes.TypeAlgo.RSA, 0., [[radius, porosity]], [1])

    multi = merope.MultiInclusions_3D()
    multi.setInclusions(sph)

    grid = merope.Voxellation_3D(multi)
    grid.setPureCoeffs(conductivities)
    grid.setHomogRule(merope.HomogenizationRule.Voigt)
    grid.setVoxelRule(merope.vox.VoxelRule.Average)
    grid.proceed(voxellation)

    # outputs
    grid.printFile("Zone.vtk", "Coeffs.txt")

    print(f"Structure built in {time.time()-start:.2f} s")


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
    # prepare results folder
    if os.path.exists(results_folder):
        print(f"Folder {results_folder} already exists, aborting to avoid overwrite.")
        return
    os.mkdir(results_folder)
    os.chdir(results_folder)

    # create subfolder for voxel grid
    case_folder = str(num_voxels)
    os.mkdir(case_folder)
    os.chdir(case_folder)

    # build structure
    voxellation = [num_voxels, num_voxels, num_voxels]
    build_voxelized_structure(domain_size, seed, inclusion_radius, porosity, conductivities, voxellation)

    # run amitex
    matrix = read_conductivity_matrix() if run_amitex() else None

    if matrix:
        k_mean, error, diag = process_matrix(matrix)
        with open("results.txt", "w") as f:
            f.write(f"Porosity: {porosity:.3f}\n")
            f.write(f"Kxx: {diag[0]:.6f}, Kyy: {diag[1]:.6f}, Kzz: {diag[2]:.6f}\n")
            f.write(f"K_mean: {k_mean:.6f}, Error: {error:.6e}\n")

        print("Results written to results.txt")

    os.chdir("../..")


if __name__ == "__main__":
    main()

# Porosity: 0.175
# Kxx: 0.759073, Kyy: 0.760271, Kzz: 0.758544
# K_mean: 0.759296, Error: 1.582661e-03
