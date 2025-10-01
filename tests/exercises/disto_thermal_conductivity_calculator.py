import os
import time
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

# resolution = R_pore / L_voxel --> N_voxel
# a_values = [0.01, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1, 2, 3, 4, 5, 6]   # different resolutions to test
a_values = [4]

R_pore_values = [1.0, 2.0, 3.0]   # pore radii to test
ratio = 25 # L_RVE / R_pore = 25

seed = 0

porosity_values = [0.1, 0.2, 0.3]  # porosity to test 

# Materials
k_matrix = 1.0
k_gas = 1e-3
conductivities = [k_matrix, k_gas]

# Results folder
# results_folder = "thermal_conductivity_calculator" + str(porosity).replace(".", "_")
# results_folder = "thermal_conductivity_radius" + str(porosity).replace(".", "_")
results_folder = "thermal_conductivity_porosity"

os.makedirs(results_folder, exist_ok=True)

# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    cwd_backup = os.getcwd()
    os.chdir(results_folder)

    import merope
    homog_rules = {
        # "Largest":   merope.HomogenizationRule.Largest,
        # "Reuss":     merope.HomogenizationRule.Reuss,
        # "Smallest":  merope.HomogenizationRule.Smallest,
        "Voigt":     merope.HomogenizationRule.Voigt,
    }

    results = []

    for a in a_values:
        for porosity in porosity_values:
            for R_pore in R_pore_values:
                L_RVE = ratio * R_pore

                num_voxels = int(a * L_RVE / R_pore)

                if num_voxels <= 0:
                    print(f"!!! Skipping a={a}: resolution too low (num_voxels={num_voxels})")
                    continue

                L_voxel = L_RVE / num_voxels
                voxellation = [num_voxels] * 3

                print(f"\n=== Running porosity={porosity}, R_pore={R_pore}, a={a}, N_voxel={num_voxels} ===")


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
                                            k_mean, error, porosity_calc, R_pore, porosity, elapsed))

                        out_file = f"results_{rule_name}_a{a}_por{porosity}_Rpore{R_pore}.txt"

                        with open(out_file, "w") as f:
                            f.write(f"Homog rule: {rule_name}\n")
                            f.write(f"L_RVE: {L_RVE}\n")
                            f.write(f"N_voxel: {num_voxels}\n")
                            f.write(f"L_voxel: {L_voxel:.6f}\n")
                            f.write(f"a: {a}\n")
                            f.write(f"Target porosity: {porosity:.3f}\n")
                            f.write(f"Calculated porosity: {porosity_calc:.3f}\n")
                            f.write(f"Radius of the pores: {R_pore:.3f}\n")
                            f.write(f"K_mean: {k_mean:.6f}\n")
                            f.write(f"Error: {error:.6e}\n")
                            f.write(f"Elapsed time: {elapsed:.2f} s\n")

    # Write a global CSV for plotting
    with open("summary.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Rule", "L_RVE", "a", "N_voxel", "L_voxel",
                         "K_mean", "Error", "Porosity_calc", "R_pore", "Porosity", "Elapsed_s"])
        writer.writerows(results)

    # -----------------------------------------------------------------------
    # PLOT
    # -----------------------------------------------------------------------
    import pandas as pd
    import matplotlib.pyplot as plt

    data = pd.DataFrame(results, columns=["Rule", "L_RVE", "a", "N_voxel", "L_voxel",
                                          "K_mean", "Error", "Porosity_calc", "R_pore", "Porosity", "Elapsed_s"])

    plt.figure(figsize=(7,5))
    markers = {0.1: "o", 0.2: "s", 0.3: "D"}
    colors = {0.1: "green", 0.2: "orange", 0.3: "blue"}

    for por in porosity_values:
        subset = data[(data["Porosity"] == por) & (data["Rule"] == "Voigt")]
        plt.plot(
            subset["R_pore"], subset["K_mean"],
            marker=markers.get(por, "x"),
            linestyle="--",
            color=colors.get(por, "black"),
            label=f"Porosity {por}"
        )

    plt.xlim(left=0)
    plt.xlabel("Radius")
    plt.ylabel(r"$k_{\mathrm{eff}}$")
    plt.title("Inclusions radius vs effective conductivity")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("k_eff_vs_radius.png", dpi=300)
    plt.show()

    os.chdir(cwd_backup)

if __name__ == "__main__":
    main()
