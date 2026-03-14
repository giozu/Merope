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

def _normalize(arr: np.ndarray) -> np.ndarray:
    """Min-max rescale *arr* to [0, 255] uint8.

    Ensures that images with different background brightness (e.g. synthetic
    white-background slices vs. grey-background SEM images) are treated
    consistently before Otsu thresholding.
    """
    lo, hi = float(arr.min()), float(arr.max())
    if hi == lo:
        return np.zeros_like(arr, dtype=np.uint8)
    return ((arr.astype(np.float32) - lo) / (hi - lo) * 255).astype(np.uint8)


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
    arr = _normalize(np.array(img))           # contrast-normalize before Otsu
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
    arr = _normalize(np.array(img))           # contrast-normalize before Otsu
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
    if f_obs.sum() == 0 or f_exp.sum() == 0 or f_obs.size < 2:
        return ks_p, 0.0

    f_exp *= f_obs.sum() / f_exp.sum()   # re-normalise to same sum

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
    exp_um_per_px: float = 1.0,
    sim_um_per_px: float = 1.0,
    sim_upscale_factor: int = 4,
) -> None:
    """Save a log-scale histogram comparing pore areas of two images.

    Parameters
    ----------
    sim_image_path      : path to the simulated (best) slice.
    exp_image_path      : path to the experimental reference image.
    output_path         : where to save the PNG.
    area_threshold      : minimum area in *experimental* pixels for noise filtering.
    exp_um_per_px       : physical scale of the experimental image [µm / pixel].
    sim_um_per_px       : physical scale of the simulated slice    [µm / pixel].
    sim_upscale_factor  : upscaling factor used when saving the simulated slice
                          (default 4, matching evaluate_slices). Simulated pixel
                          areas are divided by factor² to bring them to voxel-space,
                          making them directly comparable to experimental pixel areas.
    """
    def _areas(path: str, thr: int) -> List[float]:
        img = Image.open(path).convert("L")
        arr = _normalize(np.array(img))
        thresh = threshold_otsu(arr)
        binary = arr < thresh
        lbl = label(binary)
        return [p.area for p in regionprops(lbl) if p.area >= thr]

    # Simulated slices are saved at sim_upscale_factor× → areas are factor² too large
    sim_thr_px = area_threshold * (sim_upscale_factor ** 2)
    areas_sim_raw = np.array(_areas(sim_image_path, sim_thr_px), dtype=float)
    areas_exp_px  = np.array(_areas(exp_image_path, area_threshold), dtype=float)

    # Correct simulated areas back to voxel-space
    areas_sim_px = areas_sim_raw / (sim_upscale_factor ** 2)

    # Convert to physical area [µm²]
    areas_sim_um2 = areas_sim_px * sim_um_per_px ** 2
    areas_exp_um2 = areas_exp_px * exp_um_per_px ** 2
    thr_um2 = area_threshold * exp_um_per_px ** 2

    use_physical = (exp_um_per_px != 1.0 or sim_um_per_px != 1.0)
    xlabel = "Pore area [µm²]" if use_physical else "Pore area [pixel²]"

    all_vals = np.concatenate([areas_sim_um2, areas_exp_um2])
    bins = np.linspace(0, np.percentile(all_vals, 99.5), 81) if all_vals.size else 80

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.hist(areas_sim_um2, bins=bins, alpha=0.55, log=True, label="Simulated slice",
            color="steelblue")
    ax.hist(areas_exp_um2, bins=bins, alpha=0.55, log=True, label="Experimental",
            color="darkorange")
    ax.axvline(x=thr_um2, color="red", linestyle="--",
               linewidth=1.5,
               label=f"Threshold ({thr_um2:.0f} µm²)" if use_physical
                     else f"Threshold ({area_threshold} px²)")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Counts (log)")
    ax.set_title("Pore area distribution: simulation vs experiment")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)

    if use_physical:
        ax2 = ax.twiny()
        ax2.set_xlim(np.array(ax.get_xlim()) / exp_um_per_px ** 2)
        ax2.set_xlabel("Pore area [pixel² — exp. scale]", fontsize=9, color="gray")
        ax2.tick_params(axis="x", labelsize=8, colors="gray")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

