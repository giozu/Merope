# ======================================================================================
# PROJECT: Master's Thesis in Nuclear Engineering
# AUTHOR : [Alessio Mattiuz]
# CREATED: 2024/25
# FILE PURPOSE: 
# Main optimization script to calibrate microstructure parameters via Bayesian Optimization.
# Executes:
#   - Parameter search using `skopt` (Gaussian Process minimization)
#   - Structure generation, slicing, voxel analysis
#   - Evaluation against experimental images using statistical scores
#   - Plotting and saving of results
# ======================================================================================
# GENERAL DESCRIPTION:
# This script is part of a computational framework developed for the Master's thesis.
# The project involves:
#   - Generation of realistic 3D porous microstructures using Mérope.
#   - Morphological and statistical analysis of 2D images.
#   - Parameter optimization to match real experimental microstructures.
#
# Software and libraries used:
#   - Mérope (3D structure generation and voxelization)
#   - AMITEX_FFTP (FFT calculation)
#   - Python (control, analysis, visualization, statistical tests)
# ======================================================================================


def ThermalAmitex():
    number_of_processors = 6  # per calcoli paralleli
    voxellation_of_zones = vtkname
    amitex.computeThermalCoeff(voxellation_of_zones, number_of_processors)
    homogenized_matrix = amitex_out.printThermalCoeff(".")
    
import os
import numpy as np

from scipy.optimize import differential_evolution
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args
from send2trash import send2trash

import merope
from statistical_test_func import evaluate_simulation, compare_images, plot_area_distribution

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
    "intraInclRphi": [0.01, 0.012],
    "incl_phase": 2,
    "delta_phase": 3,
    "grains_phase": 0,
    "lagR": 2.5,
    "lagPhi": 1,
    "inclRphi": [0.1, 0.4],
    "target_porosity": 0.287,
    "num_radius": 4,
    "Kmatrix": 1,
    "Kgases": 1e-03,
    "w_data": 0.2,
    "w_porosity": 0.8,
    # Optimization constants
    "N_SLICES" : 99, # better use multiple of 3 because it is divided by the 3 directions
    "TARGET_FITNESS" : 0.8,
    #"mean_radius_range" : (np.log(0.4), np.log(0.6)),
    #"std_radius_range" : (0.1, 0.2)
}

# Variable parameters

space = [
    Real(np.log(0.1), np.log(2), name='mean_radius'),
    Real(0.01, 0.4, name='std_radius')
]


# Experimental images
experimental_images = [os.path.join(base_path, "exp_img", "exp_interconnect_1.png")]

#experimental_images = [os.path.join(base_path, "exp_img", "exp_distrib_1.png")]

x0 = [[np.log(0.7), 0.03]]  # ad esempio
y0 = [-evaluate_simulation({"mean_radius": np.log(0.7), "std_radius": 0.03, "delta1": fixed_params["delta1"]}, fixed_params, experimental_images)["average_score"]]


##############################################
# OPTIMIZATION FUNCTION (INTERNAL)           #
##############################################

@use_named_args(space)
def objective(**params):
    candidate_params = {
        "mean_radius": params['mean_radius'],
        "std_radius": params['std_radius'],
        "delta1": fixed_params["delta1"]
    }
    score = evaluate_simulation(candidate_params, fixed_params, experimental_images)["average_score"]
    return -score
    


def optimize_parameters(fixed_params, init_var_params, experimental_images, n_iter=10):

    best_var_params = init_var_params.copy()
    best_score = -np.inf

    mean_radius_range = fixed_params["mean_radius_range"]
    std_radius_range = fixed_params["std_radius_range"]

    for it in range(n_iter):
        cand_params = {
            "mean_radius": np.random.uniform(*mean_radius_range),
            "std_radius": np.random.uniform(*std_radius_range),
            "delta1": fixed_params["delta1"]  # se fissa
        }
        print(f"\n Iteration {it+1} | Parameters: {cand_params}")

        score = evaluate_simulation(cand_params, fixed_params, experimental_images)
        print(f"Score obtained: {score['average_score']:.4f}")


        if score["average_score"] > best_score:
            best_score = score["average_score"]
            best_params = cand_params
            print(f"Score obtained: {score['average_score']:.4f}")


        if best_score >= fixed_params["TARGET_FITNESS"]:
            print("Objective reached!")
            break

    return best_params, best_score

##############################################
# MAIN: OPTIMIZATION AND OUTPUT              #
##############################################

os.mkdir(folder_name)
os.chdir(folder_name)
os.mkdir(str(fixed_params["n3D"]))
os.chdir(str(fixed_params["n3D"]))
final_folder = os.path.join(folder_path, str(fixed_params["n3D"]))


print("Starting Bayesian Optimization...")
res = gp_minimize(objective, space, n_calls=10, x0=x0, y0=y0)

best_params = {
    "mean_radius": res.x[0],
    "std_radius": res.x[1],
    "delta1": fixed_params["delta1"]
}
best_fitness = -res.fun  # score era stato negato

print("Best parameters found with BO:")
print(best_params)
print(f"Best fitness score: {best_fitness:.4f}")


results = evaluate_simulation(best_params, fixed_params, experimental_images)

best_result = results["best"]    # es: ("slice_12.png", {"ks": 0.87, "chi": 0.91})
worst_result = results["worst"]
avg_score = results["average_score"]


print("\n STATISTICAL RESULTS MULTI-SLICE:")
print("---------------------------------------")
print(f"BEST slice: {best_result[0]}\n  KS p={best_result[1]['ks']:.3f}, Chi² p={best_result[1]['chi']:.3f}")
print(f"WORST slice: {worst_result[0]}\n  KS p={worst_result[1]['ks']:.3f}, Chi² p={worst_result[1]['chi']:.3f}")
print(f"AVERAGE combined score: {avg_score:.3f}")


best_image_path = os.path.join(
    folder_path, str(fixed_params["n3D"]), "tmp_slices", best_result[0]
)
output_plot_path = os.path.join(final_folder, "area_distribution_best_vs_exp.png")
plot_area_distribution(best_image_path, experimental_images[0], output_plot_path)
print(f"\n Plot saved in: {output_plot_path}")



# Uncomment the following line if you want to compute thermal conductivity of the structure
#ThermalAmitex(array3d)  


