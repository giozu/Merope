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
from scipy.optimize import differential_evolution
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
    "target_porosity": 0.3,
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
    os.path.join(base_path, "exp_img", "exp_interconnect_1.png")
]



##############################################
# OPTIMIZATION FUNCTION (INTERNAL)           #
##############################################

def optimize_parameters(fixed_params, init_var_params, experimental_images, n_iter=10):
    """
    Ricerca casuale dei parametri migliori per massimizzare la fitness.
    """
    best_var_params = init_var_params.copy()
    best_score = -np.inf

    mean_radius_range = (np.log(0.3), np.log(0.7))
    std_radius_range = (0.15, 0.4)

    for it in range(n_iter):
        cand_params = {
            "mean_radius": np.random.uniform(*mean_radius_range),
            "std_radius": np.random.uniform(*std_radius_range),
            "delta1": fixed_params["delta1"]  # se fissa
        }
        print(f"\n🔍 Iterazione {it+1} | Parametri: {cand_params}")

        score = evaluate_simulation(cand_params, fixed_params, experimental_images)
        print(f"✅ Score ottenuto: {score['average_score']:.4f}")


        if score["average_score"] > best_score:
            best_score = score["average_score"]
            best_params = cand_params
            print(f"✅ Score ottenuto: {score['average_score']:.4f}")


        if best_score >= fixed_params["TARGET_FITNESS"]:
            print("🎉 Obiettivo raggiunto!")
            break

    return best_var_params, best_score
'''    
bounds = [(2.0, 6.0), (0.5, 3.0), (0.2, 1.5)]
def objective_function(params):
    candidate_params = {
        "mean_radius": params[0],
        "std_radius": params[1],
        "delta1": params[2]
    }
    score = evaluate_simulation(candidate_params, fixed_params, experimental_images)["average_score"]
    return -score
'''
##############################################
# MAIN: OPTIMIZATION AND OUTPUT              #
##############################################

##############################################
# MAIN: OPTIMIZATION AND OUTPUT              #
##############################################

# Creazione cartelle e cambio directory
os.mkdir(folder_name)
os.chdir(folder_name)
os.mkdir(str(fixed_params["n3D"]))
os.chdir(str(fixed_params["n3D"]))
final_folder = os.path.join(folder_path, str(fixed_params["n3D"]))

# Ottimizzazione parametri
print("Starting params optimization...")
best_params, best_fitness = optimize_parameters(fixed_params, variable_params, experimental_images, n_iter=4)

print("\nOptimization ended!")
print("Best params found:")
print(best_params)
print(f"Mean fitness score: {best_fitness:.3f}")

# Valutazione slice con i parametri ottimali
results = evaluate_simulation(best_params, fixed_params, experimental_images)

best_result = results["best"]    # es: ("slice_12.png", {"ks": 0.87, "chi": 0.91})
worst_result = results["worst"]
avg_score = results["average_score"]

# Stampa risultati statistici
print("\n🧪 RISULTATI STATISTICI MULTI-SLICE:")
print("---------------------------------------")
print(f"📈 BEST slice: {best_result[0]}\n  KS p={best_result[1]['ks']:.3f}, Chi² p={best_result[1]['chi']:.3f}")
print(f"📉 WORST slice: {worst_result[0]}\n  KS p={worst_result[1]['ks']:.3f}, Chi² p={worst_result[1]['chi']:.3f}")
print(f"📊 AVERAGE combined score: {avg_score:.3f}")

# Plot confronto distribuzione aree per la slice migliore
best_image_path = os.path.join(
    folder_path, str(fixed_params["n3D"]), "tmp_slices", best_result[0]
)
output_plot_path = os.path.join(final_folder, "area_distribution_best_vs_exp.png")
plot_area_distribution(best_image_path, experimental_images[0], output_plot_path)
print(f"\n📊 Plot saved in: {output_plot_path}")



# Uncomment the following line if you want to compute thermal conductivity of the structure
# ThermalAmitex(array3d)  # Example placeholder, must define or import ThermalAmitex


