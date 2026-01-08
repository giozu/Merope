def ThermalAmitex():
    number_of_processors = 6  # per calcoli paralleli
    voxellation_of_zones = vtkname
    amitex.computeThermalCoeff(voxellation_of_zones, number_of_processors)
    homogenized_matrix = amitex_out.printThermalCoeff(".")
    
import shutil
import os
import send2trash
from Optimization_func import optimize_parameters, save_slice_images

##############################################
# PATH AND INITIAL PARAMETERS CONFIGURATIONS #
##############################################

# Results folder
folder_name = 'Result'
base_path = "/home/alessio/Thesis_Merope/Merope/Optimization_3D_structure"
folder_path = os.path.join(base_path, folder_name)
if os.path.exists(folder_path):
    send2trash(folder_path)
os.mkdir(folder_path)

# 3D structure parameters
L = [10, 10, 10]
n3D = 250
seed = 0
voxel_rule = merope.vox.VoxelRule.Average
homogRule = merope.HomogenizationRule.Voigt

grid_size = 20 # determinates the width of the subsections for the calculation of CHI^2

# File output for Mérope
vtkname = "double_layer_structure.vtk"
fileCoeff = "Coeffs.txt"

# Fixed parameters
fixed_params = {
    "delta0": 0.003,  # small intergranular layer
    "delta1": 1,       # second layer width
    "intraInclRphi": [0.008, 0.0025],  # intra granular porosity
    # Altri parametri fissi se necessari (es. incl_phase, delta_phase, grains_phase)
    "incl_phase": 2,
    "delta_phase": 3,
    "grains_phase": 0,
    "lagR": 2.5,               # grains size
    "lagPhi": 1,
    "inclRphi": [0.05, 0.4],     # small intergranular layer filling
    "target_porosity": 0.21,
    "num_radius": 10,          # number of porosity sizes
    # Material properties (K, ...)
    "Kmatrix": 1,
    "Kgases": 1e-03,
    # weighting factors
    "w_data": 0.8,          # peso della somiglianza immagine (KS+χ²)
    "w_porosity": 0.2      # peso della corrispondenza di porosità
}



# Variable parameters
variable_params = {
    "mean_radius": np.log(0.4),  # iniziale: log(media)
    "std_radius": 0.12,          # iniziale: deviazione standard
}

# Testing parameters
N_SLICES = 100 # number of slices obtained from the structure and analyzed
TARGET_FITNESS = 0.8  

# Experimental imagines paths
experimental_images = [
    os.path.join(base_path, "EXP IMG", "exp_interconnect_1.png")
    # can add further exp img:
    # os.path.join(base_path, "EXP IMG", "exp_interconnect_2.png")
]

##############################################
# MAIN: OPTIMIZATION AND OUTPUT
##############################################

print("Starting params optimization...")
best_params, best_fitness = optimize_parameters(fixed_params, variable_params, n_iter=10)
print("\nOptimization ended!")
print("Best params found:")
print(best_params)
print(f"Mean fitness score: {best_fitness:.3f}")

# Esegui una simulazione finale con i migliori parametri trovati e conserva le slice
final_folder = os.path.join(folder_path, str(n3D))
if not os.path.exists(final_folder):
    os.mkdir(final_folder)
os.chdir(final_folder)

# Cartella per le slice 2D
slices_folder = "2D_microstructure_sections"
if os.path.exists(slices_folder):
    shutil.rmtree(slices_folder)
os.mkdir(slices_folder)
os.chdir(slices_folder)

# Genera la struttura finale con i parametri ottimizzati
array3d = Crack_structure_Voxellation(n3D, L, seed, fixed_params, best_params)

# Analizza le slice per determinare best, worst e media
results = []
best_result, worst_result = None, None
best_score_val, worst_score_val, total_score = -np.inf, np.inf, 0

for i in np.linspace(0, n3D - 1, N_SLICES, dtype=int):
    slice_img = (array3d[:, :, i] * 255 / array3d.max()).astype(np.uint8)
    img_path = f"slice_{i}.png"
    Image.fromarray(slice_img).save(img_path)
    # Confronta con la prima immagine sperimentale (oppure potresti scegliere in base a una logica)
    result = compare_images(experimental_images[0], img_path, grid_size=20)
    score = result[0] + result[1]  # Combined metric
    total_score += score
    if score > best_score_val:
        best_score_val = score
        best_result = (img_path, result)
    if score < worst_score_val:
        worst_score_val = score
        worst_result = (img_path, result)
    results.append((img_path, result, score))

avg_score = total_score / N_SLICES
os.chdir("..")  # Torna alla cartella finale

print("\n🧪 RESULTS MULTI-SLICE:")
print("---------------------------------------")
if best_result is not None:
    print(f"📈 BEST slice: {best_result[0]}\n  KS p={best_result[1][0]:.3f}, Chi² p={best_result[1][1]:.3f}")
if worst_result is not None:
    print(f"📉 WORST slice: {worst_result[0]}\n  KS p={worst_result[1][0]:.3f}, Chi² p={worst_result[1][1]:.3f}")
print(f"📊 AVERAGE combined score: {avg_score:.3f}")

# Salva un plot della distribuzione delle aree per la BEST slice vs immagine sperimentale
best_image_path = os.path.join(final_folder, slices_folder, best_result[0])
output_plot_path = os.path.join(final_folder, "area_distribution_best_vs_exp.png")
plot_area_distribution(best_image_path, experimental_images[0], output_plot_path)

print(f"\nPlot saved in: {output_plot_path}")

# Remove comment to calculate thermal conductivity of best structure
# ThermalAmitex()

