# ======================================================================================
# PROJECT: Master's Thesis in Nuclear Engineering
# AUTHOR : [Alessio Mattiuz]
# CREATED: 2024/25
# FILE PURPOSE: 
# Functions to analyze and compare morphological features of real vs simulated structures.
# Includes:
#   - Kolmogorov–Smirnov test (pore size distribution)
#   - Chi-square test (spatial density in grid)
#   - 2D autocorrelation and Minkowski functionals
#   - Combined fitness score for optimization and model validation
# ======================================================================================



import os
import shutil
import numpy as np
from PIL import Image
from scipy.stats import ks_2samp, chisquare
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
import matplotlib.pyplot as plt

from MOX_structure_generator import Crack_structure_Voxellation


##############################
# STATISTICAL TESTS FUNCTIONS
##############################

def extract_pore_sizes(image_path):
    area_threshold = 30 # if small -> noise increases, if big -> poor statistics 
    img = Image.open(image_path).convert("L")
    img_array = np.array(img)
    thresh = threshold_otsu(img_array)
    binary = img_array < thresh
    labeled = label(binary)
    props = regionprops(labeled)
    sizes = [np.sqrt(p.area / np.pi) * 2 for p in props if p.area >= area_threshold]
    return np.array(sizes)

def count_pores_in_grid(image_path, grid_size):
    img = Image.open(image_path).convert("L")
    img_array = np.array(img)
    thresh = threshold_otsu(img_array)
    binary = img_array < thresh
    labeled = label(binary)
    props = regionprops(labeled)

    height, width = img_array.shape
    cell_h = height // grid_size
    cell_w = width // grid_size
    counts = np.zeros((grid_size, grid_size), dtype=int)

    for p in props:
        y, x = p.centroid
        i = int(y) // cell_h
        j = int(x) // cell_w
        if i < grid_size and j < grid_size:
            counts[i, j] += 1
    return counts.flatten()
    
    
###########################
# IMG COMPARISON FUNCTION
###########################

def compare_images(path_real, path_generated, grid_size):
    # Estrazione dimensioni pori
    sizes_real = extract_pore_sizes(path_real)
    sizes_gen = extract_pore_sizes(path_generated)

    ks_stat, ks_p = ks_2samp(sizes_real, sizes_gen)
    # Test KS: maggiore p-value indica somiglianza
    ks_score = ks_p

    # Conteggio dei pori in griglia
    counts_real = count_pores_in_grid(path_real, grid_size)
    counts_gen = count_pores_in_grid(path_generated, grid_size)

    # Normalizza counts
    norm_counts_gen = counts_gen * (counts_real.sum() / counts_gen.sum())
    mask = norm_counts_gen > 0
    filtered_real = counts_real[mask]
    filtered_gen = norm_counts_gen[mask]
    filtered_gen *= filtered_real.sum() / filtered_gen.sum()

    chi_stat, chi_p = chisquare(f_obs=filtered_real, f_exp=filtered_gen)
    chi_score = chi_p

    return ks_score, chi_score

def plot_area_distribution(image_path_1, image_path_2, output_path="area_distribution.png"):
    img1 = Image.open(image_path_1).convert("L")
    img_array1 = np.array(img1)
    thresh1 = threshold_otsu(img_array1)
    binary1 = img_array1 < thresh1
    labeled1 = label(binary1)
    props1 = regionprops(labeled1)
    areas1 = [p.area for p in props1]
    
    img2 = Image.open(image_path_2).convert("L")
    img_array2 = np.array(img2)
    thresh2 = threshold_otsu(img_array2)
    binary2 = img_array2 < thresh2
    labeled2 = label(binary2)
    props2 = regionprops(labeled2)
    areas2 = [p.area for p in props2]

    plt.figure(figsize=(8, 6))
    plt.hist(areas1, bins=100, alpha=0.5, label='simulated_slice', log=True)
    plt.hist(areas2, bins=100, alpha=0.5, label='experimental_img', log=True)
    plt.legend()
    plt.xlabel("Areas [pixel]")
    plt.axvline(x=30, color='red', linestyle='--', linewidth=2, label='Threshold')
    plt.ylabel('Counts')
    plt.title('Porosity Size Distribution')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

##############################
# FITTING EVALUATION FUNCTION
##############################   

def evaluate_simulation(var_params, fixed_params, experimental_images, temp_dir="tmp_slices"):

    n3D = fixed_params["n3D"]
    N_SLICES = fixed_params["N_SLICES"]
    grid_size = fixed_params["grid_size"]

    # Pulizia directory temporanea
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir, exist_ok=True)

    # Generazione struttura
    array3d, porosity = Crack_structure_Voxellation(fixed_params, var_params)

    # Controllo porosità
    if porosity < 0.01 or porosity > 0.4:
        print(f"Porosity out of accetable range: {porosity:.3f}")
        return {
            "fitness": 0.0,
            "best": None,
            "worst": None,
            "average_score": 0.0
        }

    # Fitness porosity
    target = fixed_params["target_porosity"]
    err = abs(porosity - target) / target
    por_fitness = max(0.0, 1.0 - err)

    # Slice saving
    slice_names = []
    axes = ['z', 'y', 'x']
    array_shape = array3d.shape
    slices_per_axis = N_SLICES // 3

    for axis in axes:
        axis_idx = {'x': 0, 'y': 1, 'z': 2}[axis]
        n = array_shape[axis_idx]
        indices = np.linspace(0, n - 1, slices_per_axis, dtype=int)

        for count, i in enumerate(indices):
            if axis == 'z':
                slice_img = array3d[:, :, i]
            elif axis == 'y':
                slice_img = array3d[:, i, :]
            elif axis == 'x':
                slice_img = array3d[i, :, :]

            slice_img = (slice_img * 255 / array3d.max()).astype(np.uint8)
            name = f"slice_{axis}_{count}.png"
            Image.fromarray(slice_img).save(os.path.join(temp_dir, name))
            slice_names.append(name)


    # Confrontation with exp images
    exp_img = experimental_images[0]
    slice_results = []

    for name in slice_names:
        path = os.path.join(temp_dir, name)
        ks, chi = compare_images(exp_img, path, grid_size)
        score = (ks + chi) / 2
        slice_results.append((name, {"ks": ks, "chi": chi, "score": score}))

    # Best/worst slice extraction
    sorted_slices = sorted(slice_results, key=lambda x: x[1]["score"], reverse=True)
    best = sorted_slices[0]
    worst = sorted_slices[-1]
    avg_score = np.mean([x[1]["score"] for x in slice_results])

    # Combination
    w_data = fixed_params["w_data"]
    w_por = fixed_params["w_porosity"]
    total_fitness = w_data * avg_score + w_por * por_fitness

    return {
        "fitness": total_fitness,
        "best": best,     # (nome file, dict con p-valori e score)
        "worst": worst,
        "average_score": avg_score
    }


