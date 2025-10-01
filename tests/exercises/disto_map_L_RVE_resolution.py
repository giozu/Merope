"""
map_L_RVE_resolution.py

2D convergence study: effective thermal conductivity as a function of
RVE size (L_RVE) and resolution parameter (a = R_pore / L_voxel).
"""

import os
import numpy as np
import csv

from utils_microstructure import (
    build_voxelized_structure,
    run_amitex,
    read_conductivity_matrix,
    process_matrix
)

# ---------------------------------------------------------------------------
# INPUT PARAMETERS
# ---------------------------------------------------------------------------

# Geometry
R_pore = 1.0
L_RVE_list = [10, 15, 20, 25, 30, 35]
a_list     = [1, 2, 3, 4, 5, 6]

num_vox_max = 100

# Materials
k_matrix = 1.0
k_gas = 1e-3
conductivities = [k_matrix, k_gas]

# Seed
seed = 0

porosity = 0.20

# Results folder
results_folder = "map_L_RVE_resolution_p" + str(porosity).replace(".", "_")
os.makedirs(results_folder, exist_ok=True)

# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
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
                # conductivity
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

        # results.append((L, a, num_voxels, L_voxel, k_mean, error, porosity_calc))
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
