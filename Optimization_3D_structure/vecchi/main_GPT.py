def ThermalAmitex():
    number_of_processors = 6  # per calcoli paralleli
    voxellation_of_zones = vtkname
    amitex.computeThermalCoeff(voxellation_of_zones, number_of_processors)
    homogenized_matrix = amitex_out.printThermalCoeff(".")
 
import logging # DECIDERE SE IMPLEMENTARE   
import os
import shutil
import numpy as np
from PIL import Image
from scipy.optimize import minimize
from send2trash import send2trash

import merope
from statistical_test_func import evaluate_simulation, compare_images, plot_area_distribution
from MOX_structure_generator import Crack_structure_Voxellation  # assuming this is the correct import
#from Optimization_func import optimize_parameters
import interface_amitex_fftp.amitex_wrapper as amitex
import interface_amitex_fftp.post_processing as amitex_out


##############################################
# PATH AND INITIAL PARAMETERS CONFIGURATIONS #
##############################################

# Results folder
folder_name = 'Result'
base_path = "/home/giovanni/Merope/Optimization_3D_structure"
folder_path = os.path.join(base_path, folder_name)
if os.path.exists(folder_path):
    send2trash(folder_path)
os.mkdir(folder_path)


# Fixed parameters
fixed_params = {

    # File output for Mérope
    "vtkname" : "double_layer_structure.vtk",
    "fileCoeff" : "Coeffs.txt",
    # 3D structure params
    "L" : [10, 10, 10],
    "n3D" : 250,
    "seed" : 0,
    "voxel_rule" : merope.vox.VoxelRule.Average,
    "homogRule" : merope.HomogenizationRule.Voigt,
    "grid_size" : 20,
    #structure params
    "delta0": 0.003,
    "delta1": 1,
    "intraInclRphi": [0.008, 0.0025],
    "incl_phase": 2,
    "delta_phase": 3,
    "grains_phase": 0,
    "lagR": 2.5,
    "lagPhi": 1,
    "inclRphi": [0.05, 0.4],
    "target_porosity": 0.21,
    "num_radius": 10,
    "Kmatrix": 1,
    "Kgases": 1e-03,
    "w_data": 0.2,
    "w_porosity": 0.8,
    # Optimization constants
    "N_SLICES" : 100,
    "TARGET_FITNESS" : 0.8
}

# Variable parameters
variable_params = {
    "mean_radius": np.log(0.4),
    "std_radius": 0.12
}

# Experimental images
experimental_images = [
    os.path.join(base_path, "EXP IMG", "exp_interconnect_1.png")
]



##############################################
# OPTIMIZATION FUNCTION (INTERNAL)           #
##############################################

def optimize_parameters(fixed_params, init_var_params, n_iter=1):
    best_var_params = init_var_params.copy()
    best_score = -np.inf

    # Ranges realistici (log radius): log(0.3 mm) ≈ -1.204, log(0.7 mm) ≈ -0.357
    mean_radius_range = (np.log(0.3), np.log(0.7))  # valori logaritmici!
    std_radius_range = (0.15, 0.4)  # deviazione standard lineare

    for it in range(n_iter):
        cand_params = {
            "mean_radius": np.random.uniform(*mean_radius_range),
            "std_radius": np.random.uniform(*std_radius_range)
        }

        # Evita valori non validi
        if cand_params["std_radius"] <= 0:
            print(f"❌ Skipping invalid std_radius: {cand_params['std_radius']}")
            continue

        print(f"Iteration {it+1}: Trying parameters {cand_params}")

        score = evaluate_simulation(cand_params, fixed_params, experimental_images)

        print(f" --> Combined fitness score: {score:.3f}")

        # Aggiorna il migliore se necessario
        if score > best_score:
            best_score = score
            best_var_params = cand_params.copy()

        # Stop anticipato se score accettabile
        if best_score >= fixed_params.get("TARGET_FITNESS", 0.99):
            print("✅ Target fitness reached!")
            break

    return best_var_params, best_score


##############################################
# MAIN: OPTIMIZATION AND OUTPUT              #
##############################################

if os.path.exists(folder_path):
    send2trash(folder_path)
    
os.mkdir(folder_name)
os.chdir(folder_name)
os.mkdir(str(fixed_params["n3D"]))
os.chdir(str(fixed_params["n3D"]))
final_folder = os.path.join(folder_path, str(fixed_params["n3D"]))

print("Starting params optimization...")
best_params, best_fitness = optimize_parameters(fixed_params, variable_params, n_iter=1)
print("\nOptimization ended!")
print("Best params found:")
print(best_params)
print(f"Mean fitness score: {best_fitness:.3f}")


slices_folder = "2D_microstructure_sections"
slices_path = os.path.join(final_folder, slices_folder)
if os.path.exists(slices_path):
    shutil.rmtree(slices_path)
os.mkdir(slices_path)
os.chdir(slices_path)

results = []
best_result, worst_result = None, None
best_score_val, worst_score_val, total_score = -np.inf, np.inf, 0

avg_score = total_score / fixed_params["N_SLICES"]
os.chdir("..")

print("\n🧪 RESULTS MULTI-SLICE:")
print("---------------------------------------")
if best_result:
    print(f"📈 BEST slice: {best_result[0]}\n  KS p={best_result[1][0]:.3f}, Chi² p={best_result[1][1]:.3f}")
if worst_result:
    print(f"📉 WORST slice: {worst_result[0]}\n  KS p={worst_result[1][0]:.3f}, Chi² p={worst_result[1][1]:.3f}")
print(f"📊 AVERAGE combined score: {avg_score:.3f}")
print(f"il sos è", best_result)

if best_result:
    best_image_path = best_result[0]
    output_plot_path = os.path.join(final_folder, "area_distribution_best_vs_exp.png")
    plot_area_distribution(best_image_path, experimental_images[0], output_plot_path)
    print(f"\nPlot saved in: {output_plot_path}")
else:
    print("\n⚠️ Nessuna slice migliore trovata. Nessun plot salvato.")

# Uncomment the following line if you want to compute thermal conductivity of the structure
# ThermalAmitex(array3d)  # Example placeholder, must define or import ThermalAmitex




