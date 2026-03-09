"""
core/statistics.py
==================
Statistical comparison functions for morphological validation of simulated
microstructures against experimental images (SEM, tomography slices, etc.).

Functions
---------
extract_pore_sizes(image_path, area_threshold)
    Extract equivalent diameters of pore regions from a greyscale image.

count_pores_in_grid(image_path, grid_size)
    Count pores per cell in an N×N spatial grid.

compare_images(path_real, path_simulated, grid_size)
    Compute KS + Chi² similarity scores between two images.

evaluate_slices(array3d, experimental_image_path, n_slices, grid_size)
    Slice a 3-D array, compare each slice to the experimental image,
    and return best / worst / average score.

plot_area_distribution(image_path_1, image_path_2, output_path)
    Save a histogram of pore areas for two images side by side.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
from scipy.stats import chisquare, ks_2samp
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
import matplotlib
matplotlib.use("Agg")         # non-interactive backend – no display needed
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Low-level feature extractors
# ---------------------------------------------------------------------------

def extract_pore_sizes(image_path: str, area_threshold: int = 30) -> np.ndarray:
    """Return equivalent diameters of pore regions detected in *image_path*.

    Parameters
    ----------
    image_path:
        Path to a greyscale (or RGB converted to L) image.
    area_threshold:
        Minimum region area in pixels.  Smaller regions are treated as noise.
        Lower → more noise; higher → fewer statistics.

    Returns
    -------
    np.ndarray of shape (N,) with equivalent diameters ``2 * sqrt(A/π)``.
    """
    img = Image.open(image_path).convert("L")
    arr = np.array(img)
    thresh = threshold_otsu(arr)
    binary = arr < thresh                     # pores are dark
    labeled = label(binary)
    props = regionprops(labeled)
    sizes = [
        np.sqrt(p.area / np.pi) * 2
        for p in props
        if p.area >= area_threshold
    ]
    return np.array(sizes)


def count_pores_in_grid(image_path: str, grid_size: int = 20) -> np.ndarray:
    """Count pore centroids per cell in a *grid_size* × *grid_size* spatial grid.

    Returns
    -------
    np.ndarray of shape (grid_size²,) – flattened count map.
    """
    img = Image.open(image_path).convert("L")
    arr = np.array(img)
    thresh = threshold_otsu(arr)
    binary = arr < thresh
    labeled = label(binary)
    props = regionprops(labeled)

    h, w = arr.shape
    cell_h = h // grid_size
    cell_w = w // grid_size
    counts = np.zeros((grid_size, grid_size), dtype=int)

    for p in props:
        y, x = p.centroid
        i = int(y) // cell_h
        j = int(x) // cell_w
        if i < grid_size and j < grid_size:
            counts[i, j] += 1

    return counts.flatten()


# ---------------------------------------------------------------------------
# Image-level comparison
# ---------------------------------------------------------------------------

def compare_images(
    path_real: str,
    path_simulated: str,
    grid_size: int = 20,
) -> Tuple[float, float]:
    """Compute KS and Chi² similarity scores between two images.

    The KS test compares the pore-size distributions; the Chi² test compares
    the spatial pore-density distributions on a grid.

    Returns
    -------
    (ks_score, chi_score) : both are p-values ∈ [0, 1].
        Higher p-values → more similar distributions.
    """
    sizes_real = extract_pore_sizes(path_real)
    sizes_sim  = extract_pore_sizes(path_simulated)

    if sizes_real.size == 0 or sizes_sim.size == 0:
        return 0.0, 0.0

    _, ks_p = ks_2samp(sizes_real, sizes_sim)

    counts_real = count_pores_in_grid(path_real, grid_size)
    counts_sim  = count_pores_in_grid(path_simulated, grid_size)

    # Normalise simulated counts to the same total as real counts
    total_real = counts_real.sum()
    total_sim  = counts_sim.sum()
    if total_sim == 0:
        return ks_p, 0.0

    norm_sim = counts_sim * (total_real / total_sim)
    mask = norm_sim > 0
    f_obs = counts_real[mask].astype(float)
    f_exp = norm_sim[mask]
    f_exp *= f_obs.sum() / f_exp.sum()   # re-normalise to same sum

    if f_obs.size < 2:
        return ks_p, 0.0

    _, chi_p = chisquare(f_obs=f_obs, f_exp=f_exp)
    return float(ks_p), float(chi_p)


# ---------------------------------------------------------------------------
# Volume → slices → scoring pipeline
# ---------------------------------------------------------------------------

def evaluate_slices(
    array3d: np.ndarray,
    experimental_image_path: str,
    n_slices: int = 99,
    grid_size: int = 20,
    temp_dir: str = "tmp_slices",
) -> Dict:
    """Slice *array3d* along all three axes, compare each slice to the
    experimental image, and return a summary dict.

    Parameters
    ----------
    array3d:
        3-D numpy array (n, n, n) with voxel conductivity values.
    experimental_image_path:
        Path to the experimental reference image (greyscale PNG/TIFF).
    n_slices:
        Total number of 2-D slices to extract (split equally over x, y, z).
        Should be divisible by 3.
    grid_size:
        Grid resolution used for the Chi² spatial test.
    temp_dir:
        Temporary directory to store the 2-D PNG slices.

    Returns
    -------
    dict with keys:
        ``average_score`` – mean (ks+chi)/2 over all slices,
        ``best``          – (filename, {ks, chi, score}) of the best slice,
        ``worst``         – (filename, {ks, chi, score}) of the worst slice,
        ``slices``        – full list of (filename, scores) tuples.
    """
    # ── temp directory management ──────────────────────────────────────
    temp_dir = str(temp_dir)
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir, exist_ok=True)

    # ── slicing ────────────────────────────────────────────────────────
    n = array3d.shape[0]
    slices_per_axis = max(1, n_slices // 3)
    axes_cfg = [
        ("x", lambda i: array3d[i, :, :]),
        ("y", lambda i: array3d[:, i, :]),
        ("z", lambda i: array3d[:, :, i]),
    ]

    vmax = array3d.max()
    if vmax == 0:
        vmax = 1.0

    slice_names: List[str] = []
    for axis_name, slicer in axes_cfg:
        indices = np.linspace(0, n - 1, slices_per_axis, dtype=int)
        for count, idx in enumerate(indices):
            sl = slicer(int(idx))
            sl_uint8 = (sl * 255 / vmax).astype(np.uint8)
            name = f"slice_{axis_name}_{count:04d}.png"
            img = Image.fromarray(sl_uint8)
            img = img.resize((img.width * 4, img.height * 4), Image.Resampling.NEAREST)
            img.save(os.path.join(temp_dir, name))
            slice_names.append(name)

    # ── comparison ─────────────────────────────────────────────────────
    results: List[Tuple[str, Dict]] = []
    for name in slice_names:
        path = os.path.join(temp_dir, name)
        ks, chi = compare_images(experimental_image_path, path, grid_size)
        score = (ks + chi) / 2.0
        results.append((name, {"ks": ks, "chi": chi, "score": score}))

    if not results:
        return {"average_score": 0.0, "best": None, "worst": None, "slices": []}

    sorted_r = sorted(results, key=lambda x: x[1]["score"], reverse=True)
    avg = float(np.mean([r[1]["score"] for r in results]))

    return {
        "average_score": avg,
        "best":  sorted_r[0],
        "worst": sorted_r[-1],
        "slices": results,
    }


# ---------------------------------------------------------------------------
# Plotting utility
# ---------------------------------------------------------------------------

def plot_area_distribution(
    sim_image_path: str,
    exp_image_path: str,
    output_path: str = "area_distribution.png",
    area_threshold: int = 30,
) -> None:
    """Save a log-scale histogram comparing pore areas of two images.

    Parameters
    ----------
    sim_image_path: path to the simulated (best) slice.
    exp_image_path: path to the experimental reference image.
    output_path:    where to save the PNG.
    area_threshold: minimum area (pixels) used for filtering noise.
    """
    def _areas(path: str) -> List[float]:
        img = Image.open(path).convert("L")
        arr = np.array(img)
        thresh = threshold_otsu(arr)
        binary = arr < thresh
        lbl = label(binary)
        return [p.area for p in regionprops(lbl) if p.area >= area_threshold]

    areas_sim = _areas(sim_image_path)
    areas_exp = _areas(exp_image_path)

    plt.figure(figsize=(8, 6))
    plt.hist(areas_sim, bins=80, alpha=0.55, log=True, label="Simulated slice")
    plt.hist(areas_exp, bins=80, alpha=0.55, log=True, label="Experimental")
    plt.axvline(x=area_threshold, color="red", linestyle="--",
                linewidth=1.5, label=f"Threshold ({area_threshold} px²)")
    plt.xlabel("Pore area [pixel²]")
    plt.ylabel("Counts (log)")
    plt.title("Pore area distribution: simulation vs experiment")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
