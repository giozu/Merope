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
porosity = 0.175

L_RVE_list = [10, 15, 20, 25, 30]
a_list     = [2, 3, 4, 5, 6]

num_vox_max = 100

# Materials
k_matrix = 1.0
k_gas = 1e-3
conductivities = [k_matrix, k_gas]

# Seed
seed = 0

# Results folder
results_folder = "map_L_RVE_resolution"
os.makedirs(results_folder, exist_ok=True)


# ---------------------------------------------------------------------------
# FUNCTIONS
# ---------------------------------------------------------------------------

def build_voxelized_structure(domain_size, seed, radius, porosity, conductivities, voxellation):
    sph = merope.SphereInclusions_3D()
    sph.setLength(domain_size)
    sph.fromHisto(seed, sac_de_billes.TypeAlgo.RSA, 0., [[radius, porosity]], [1])

    multi = merope.MultiInclusions_3D()
    multi.setInclusions(sph)

    structure = merope.Structure_3D(multi)
    grid_params = merope.vox.create_grid_parameters_N_L_3D(voxellation, domain_size)
    grid = merope.vox.GridRepresentation_3D(structure, grid_params, merope.vox.VoxelRule.Average)

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
    error = sqrt(sum(v * v for v in offdiag))
    return k_mean, error, diag


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
                print(f"!!!  Skipping case L={L}, a={a} (N_vox={num_voxels} > 100)")
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

    # Convert results to arrays for plotting
    L_vals   = sorted(set(r[0] for r in results))
    a_vals   = sorted(set(r[1] for r in results))
    
    K_map = np.full((len(L_vals), len(a_vals)), np.nan)
    for r in results:
        i = L_vals.index(r[0])
        j = a_vals.index(r[1])
        K_map[i, j] = r[4]

    # Heatmap of K_mean
    plt.figure(figsize=(8,6))
    # plt.imshow(K_map, origin="lower", aspect="auto",
    #            extent=[min(a_vals), max(a_vals), min(L_vals), max(L_vals)],
    #            cmap="viridis")
    plt.imshow(np.ma.masked_invalid(K_map), origin="lower", aspect="auto",
           extent=[min(a_vals), max(a_vals), min(L_vals), max(L_vals)],
           cmap="viridis")
    plt.colorbar(label="K_mean")
    plt.xlabel("Resolution parameter a")
    plt.ylabel("RVE size L")
    plt.title("Effective conductivity map")
    plt.tight_layout()
    plt.savefig(os.path.join(results_folder, "K_map.png"))
    plt.show()


if __name__ == "__main__":
    main()
