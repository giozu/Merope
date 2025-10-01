"""
Script to compute homogenized thermal conductivity as a function of porosity.
Mérope builds the microstructure and voxelization, AMITEX computes the
homogenized conductivity, and results are stored in structured output files.
"""

import os
import time

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
R_pore = 1.0               # pore radius
L_RVE = 25                 # cubic RVE dimensions (ratio L_RVE / R_pore = 25 is recommended)
a = 4                      # resolution parameter (4 is recommended)

num_voxels = int(a * L_RVE / R_pore)
L_voxel = L_RVE / num_voxels

seed = 0
porosity = 0.20

# Materials
k_matrix = 1.0
k_gas = 1e-3
conductivities = [k_matrix, k_gas]

voxellation = [num_voxels]*3

# Results folder
results_folder = "thermal_conductivity_calculator" + str(porosity).replace(".", "_")
os.makedirs(results_folder, exist_ok=True)

# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    os.makedirs(results_folder, exist_ok=True)
    results = []

    cwd_backup = os.getcwd()
    os.chdir(results_folder)

    print(f"\n=== Running L={L_RVE}, a={a} ===")
    
    import merope
    homog_rules = {
        "Largest":   merope.HomogenizationRule.Largest,
        "Reuss":     merope.HomogenizationRule.Reuss,
        "Smallest":  merope.HomogenizationRule.Smallest,
        "Voigt":     merope.HomogenizationRule.Voigt,
    }

    for rule_name, rule in homog_rules.items():
        print(f"--- Homogenization rule: {rule_name} ---")

        start_time = time.time()
        porosity_calc = build_voxelized_structure(
            [L_RVE, L_RVE, L_RVE],
            seed, R_pore, porosity, conductivities,
            voxellation, homog_rule=rule
        )

        try:
            matrix = read_conductivity_matrix() if run_amitex() else None
        except FileNotFoundError:
            print(f"!!! Amitex output missing for {rule_name}")
            matrix = None

        if matrix:
            k_mean, error, diag = process_matrix(matrix)
            elapsed = time.time() - start_time

            results.append((rule_name, L_RVE, a, num_voxels, L_voxel,
                            k_mean, error, porosity_calc, elapsed))

            out_file = f"results_{rule_name}.txt"
            with open(out_file, "w") as f:
                f.write(f"Homog rule: {rule_name}\n")
                f.write(f"L_RVE: {L_RVE}\n")
                f.write(f"N_voxel: {num_voxels}\n")
                f.write(f"L_voxel: {L_voxel:.6f}\n")
                f.write(f"a: {a}\n")
                f.write(f"Target porosity: {porosity:.3f}\n")
                f.write(f"Calculated porosity: {porosity_calc:.3f}\n")
                f.write(f"K_mean: {k_mean:.6f}\n")
                f.write(f"Error: {error:.6e}\n")
                f.write(f"Elapsed time: {elapsed:.2f} s\n")

    os.chdir(cwd_backup)

if __name__ == "__main__":
    main()
