"""
map_L_RVE_resolution.py

2D convergence study: effective thermal conductivity as a function of
RVE size (L_RVE) and resolution parameter (a = R_pore / L_voxel).
"""

import os
from math import sqrt
import matplotlib.pyplot as plt
import numpy as np

import sac_de_billes
import merope
import interface_amitex_fftp.amitex_wrapper as amitex
import interface_amitex_fftp.post_processing as amitex_out

# ---------------------------------------------------------------------------
# INPUT PARAMETERS
# ---------------------------------------------------------------------------

# Geometry
R_pore = 1.0
porosity = 0.10

L_RVE_list = [10, 15, 20, 25, 30, 35]
a_list     = [1, 2, 3, 4, 5, 6]

num_vox_max = 100

# Materials
k_matrix = 1.0
k_gas = 1e-3
conductivities = [k_matrix, k_gas]

# Seed
seed = 0

# Results folder

results_folder = "map_L_RVE_resolution_p" + str(porosity).replace(".", "_")
os.makedirs(results_folder, exist_ok=True)


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
    porosity_calc = analyzer.compute_percentages(grid)[1]

    grid.apply_homogRule(merope.HomogenizationRule.Voigt, conductivities)

    printer = merope.vox.vtk_printer_3D()
    printer.printVTK_segmented(grid, "Zone.vtk", "Coeffs.txt", nameValue="MaterialId")

    return porosity_calc


def run_amitex():
    num_procs = 2
    amitex.computeThermalCoeff("Zone.vtk", num_procs)
    return amitex_out.printThermalCoeff(".")


def read_conductivity_matrix(file_path="thermalCoeff_amitex.txt"):
    with open(file_path, "r") as f:
        return [list(map(float, line.split())) for line in f]


def process_matrix(matrix):
    diag = [matrix[i][i] for i in range(3)]
    offdiag = [matrix[i][j] for i in range(3) for j in range(3) if j != i]
    k_mean = sum(diag) / 3
    abs_error = sqrt(sum(v * v for v in offdiag))
    return k_mean, abs_error, diag


import csv

# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    # make sure convergence_map/ exists
    os.makedirs(results_folder, exist_ok=True)

    results = []
    for L in L_RVE_list:
        for a in a_list:
            print(f"\n=== Running L={L}, a={a} ===")

            num_voxels = int(a * L / R_pore)
            voxellation = [num_voxels]*3
            L_voxel = L / num_voxels

            if num_voxels > num_vox_max:
                print(f"!!!  Skipping case L={L}, a={a} (N_vox={num_voxels} > {num_vox_max})")
                results.append((L, a, num_voxels, L_voxel, np.nan, np.nan, np.nan))
                continue

            case_folder = os.path.join(results_folder, f"L_{L}_a_{a}")
            os.makedirs(case_folder, exist_ok=True)

            cwd_backup = os.getcwd()
            os.chdir(case_folder)

            porosity_calc = build_voxelized_structure(
                [L, L, L], seed, R_pore, porosity, conductivities, voxellation
            )

            try:
                matrix = read_conductivity_matrix() if run_amitex() else None
            except FileNotFoundError:
                print(f"!!! Amitex output missing for L={L}, a={a}")
                matrix = None

            if matrix:
                k_mean, error, diag = process_matrix(matrix)
                results.append((L, a, num_voxels, L_voxel, k_mean, error, porosity_calc))

                with open("results.txt", "w") as f:
                    f.write(f"L_RVE: {L}\n")
                    f.write(f"N_voxel: {num_voxels}\n")
                    f.write(f"L_voxel: {L_voxel:.6f}\n")
                    f.write(f"a: {a}\n")
                    f.write(f"Target porosity: {porosity:.3f}\n")
                    f.write(f"Calculated porosity: {porosity_calc:.3f}\n")
                    f.write(f"K_mean: {k_mean:.6f}, Error: {error:.6e}\n")

            os.chdir(cwd_backup)

    # -------------------------------
    # Save results (CSV + NumPy npz)
    # -------------------------------
    csv_path = os.path.join(results_folder, "summary_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["L_RVE", "a", "N_voxel", "L_voxel", "K_mean", "Error", "Porosity_calc"])
        writer.writerows(results)
    print(f"Saved CSV results to {csv_path}")

    npz_path = os.path.join(results_folder, "summary_results.npz")

    np.savez(npz_path,
            results=np.array(results, dtype=float),
            L_vals=np.array(sorted(set(r[0] for r in results)), dtype=float),
            a_vals=np.array(sorted(set(r[1] for r in results)), dtype=float))

    print(f"Saved NumPy results to {npz_path}")

if __name__ == "__main__":
    main()
