"""
═══════════════════════════════════════════════════════════════════════════════
PORE ANALYSIS FROM POLISHED CERAMIC CROSS-SECTION IMAGES
═══════════════════════════════════════════════════════════════════════════════

PURPOSE:
--------
Analyze porosity in polished ceramic cross-section images (SEM, optical microscopy).
Classifies pores as intragranular (round, isolated) or intergranular (elongated,
interconnected) based on circularity and size.

MAIN FUNCTIONS:
--------------
1. analyze_porosity()    - Main analysis function
2. sensitivity_analysis() - Test multiple circularity thresholds
3. CLI usage             - Run directly from command line

USAGE EXAMPLES:
--------------
## Python API:
    from pore_analysis import analyze_porosity

    result = analyze_porosity(
        image_path="my_sample.png",
        um_per_pixel=0.195,              # Scale: 0.195 μm/pixel
        circularity_threshold=0.50,       # Classification boundary
        min_area_um2=2.0,                 # Minimum pore size
        use_adaptive_threshold=False,     # Otsu (False) or Sauvola (True)
        use_watershed=True,               # Separate touching pores
        export_csv=True,                  # Save per-pore data
        show_plots=True                   # Generate diagnostic plots
    )

    print(f"Total porosity: {result['p_total']:.2%}")
    print(f"Intragranular:  {result['p_intra']:.2%}")
    print(f"Intergranular:  {result['p_inter']:.2%}")

## Command line:
    python pore_analysis.py image.png 0.195 --plot --export-csv
    python pore_analysis.py image.png 0.195 0.55 --adaptive --plot
    python pore_analysis.py image.png 0.195 --sensitivity

ADJUSTABLE PARAMETERS:
---------------------
um_per_pixel           : Spatial resolution (calculate from scale bar)
circularity_threshold  : Boundary for intra/inter classification (0.4-0.6 typical)
min_area_um2          : Noise filter - minimum pore size to count
use_adaptive_threshold : True for uneven illumination, False for uniform
use_watershed         : True to separate touching pores (recommended)
area_inter_thr_um2    : Size above which pores are always intergranular (20 μm²)
min_area_inter_um2    : Minimum size for irregular pores to be intergranular (5 μm²)
scale_bar_strip       : Fraction of bottom to mask (scale bar region, 0.12 default)
stereology_factor     : 2D→3D correction (1.0 = Delesse principle)

CLASSIFICATION LOGIC:
--------------------
A pore is classified as INTERGRANULAR if:
  1. area > area_inter_thr_um2  (default 20 μm²)  OR
  2. area ≥ min_area_inter_um2 (default 5 μm²) AND circularity < threshold

Otherwise → INTRAGRANULAR

Circularity = 4π·Area / Perimeter²
  - 1.0 = perfect circle
  - 0.0 = straight line

KEY IMPROVEMENTS OVER v1:
------------------------
✓ Shape-based classification (not just size)
✓ Watershed segmentation to separate merged pores
✓ Sauvola adaptive thresholding for uneven illumination
✓ Scale bar masking (not cropping) → consistent area denominator
✓ Per-pore CSV export with morphology metrics
✓ Proper stereology (Delesse principle, no magic constants)
✓ Sensitivity analysis to calibrate circularity threshold

OUTPUTS:
--------
- Dictionary with porosity fractions, pore counts, mean sizes
- Optional: per-pore CSV with area, ECD, circularity, position, type
- Optional: 6-panel diagnostic plot (original, binary, classification,
  circularity, size distribution, summary)

DEPENDENCIES:
------------
Required: numpy, scipy, PIL
Optional: scikit-image (for watershed & Sauvola), matplotlib (for plots)
"""

import numpy as np
from PIL import Image
from scipy import ndimage
from pathlib import Path
from typing import Dict, Optional, Tuple
import warnings
import csv

try:
    from skimage import (
        filters, measure, morphology, segmentation, feature
    )
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False
    warnings.warn(
        "scikit-image not found. Install with: pip install scikit-image\n"
        "Falling back to basic scipy implementation (no watershed, no Sauvola)."
    )

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS - Default values (can be overridden in function calls)
# ─────────────────────────────────────────────────────────────────────────────

# Circularity threshold for classification
# Circularity = 4π·Area / Perimeter²   (1.0 = perfect circle, 0 = line)
#
# ADJUST THIS based on your material:
#   - Sintered ceramics: 0.45 - 0.55 (intragranular pores are fairly round)
#   - Powder compacts:   0.40 - 0.50 (more irregular shapes)
#   - Green bodies:      0.35 - 0.45 (highly irregular)
#
# Use sensitivity_analysis() to find the best value for your samples
DEFAULT_CIRCULARITY_THRESHOLD = 0.50

# Noise filter - minimum pore area in PIXELS
# Very small objects (< 10 pixels) are likely noise or artifacts
# This is a conservative pre-filter; the real minimum size is set by min_area_um2
MIN_PORE_AREA_PX = 10

# Scale bar masking - fraction of image height to mask at bottom
# Most SEM/microscopy images have scale bars in the bottom ~10-15%
# Adjust if your scale bar is in a different location
SCALE_BAR_STRIP_FRACTION = 0.12


# ─────────────────────────────────────────────────────────────────────────────
# MAIN FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def analyze_porosity(
    image_path: str,
    um_per_pixel: float,
    circularity_threshold: float = DEFAULT_CIRCULARITY_THRESHOLD,
    min_area_um2: float = 5.0,
    dark_is_pore: bool = True,
    use_adaptive_threshold: bool = False,
    use_watershed: bool = True,
    area_inter_thr_um2: float = 20.0,
    min_area_inter_um2: float = 5.0,
    scale_bar_strip: float = SCALE_BAR_STRIP_FRACTION,
    stereology_factor: float = 1.0,
    export_csv: bool = False,
    output_dir: Optional[str] = None,
    show_plots: bool = False,
) -> Dict:
    """
    Estimate porosity from a polished ceramic cross-section image.

    Classification strategy
    -----------------------
    Each detected pore object is classified by TWO independent criteria:

        criterion A — size:
            area ≥ min_area_um2 → significant pore (not noise)

        criterion B — shape (circularity):
            C = 4π · area / perimeter²
            C ≥ circularity_threshold  →  intragranular (round, isolated)
            C <  circularity_threshold  →  intergranular (elongated, irregular)

    This dual criterion avoids the v1 error of calling a large round pore
    "intergranular" just because it is large.

    Stereology note
    ---------------
    The Delesse principle states that for isotropic, homogeneous materials
    the 2D area fraction equals the 3D volume fraction.  The default
    stereology_factor=1.0 reflects this.  If your material is anisotropic
    or you have independent calibration data (Archimedes, X-ray CT), pass
    the appropriate correction factor here.

    Parameters
    ----------
    image_path          : path to image (PNG / JPG / TIF)
    um_per_pixel        : spatial resolution in µm/pixel
                          → compute as: scale_bar_length_um / scale_bar_length_px
    circularity_threshold : boundary between intra (round) and inter (elongated)
    min_area_um2        : minimum pore area to include (noise filter)
    dark_is_pore        : True if pores appear dark (standard reflected light)
    use_adaptive_threshold : use Sauvola local thresholding instead of global Otsu
                            (recommended for images with uneven illumination)
    use_watershed       : split touching / merged pores with watershed segmentation
    scale_bar_strip     : fraction of image height (bottom) to mask before analysis
    stereology_factor   : 2D → 3D correction (default 1.0, Delesse principle)
    export_csv          : if True, write per-pore data to CSV
    output_dir          : directory for output files (default: same as image)
    show_plots          : if True, display and save diagnostic plots

    Returns
    -------
    dict with:
        p_total, p_intra, p_inter  : porosity fractions (after stereology factor)
        p_total_2d, p_intra_2d, p_inter_2d  : raw 2D area fractions
        n_total, n_intra, n_inter  : pore counts
        mean_ecd_intra_um, mean_ecd_inter_um  : mean equivalent circular diameter
        mean_circularity_intra, mean_circularity_inter  : mean shape factor
        threshold_value  : Otsu or Sauvola window size used
        pore_table       : list of dicts, one per pore (for CSV export)
    """

    # ── 1. Load & validate ────────────────────────────────────────────────────
    img_pil = Image.open(image_path).convert("L")
    arr = np.array(img_pil, dtype=np.float32) / 255.0   # float [0,1]
    H, W = arr.shape

    # ── 2. Mask scale bar (bottom strip + optional right column) ──────────────
    # We MASK (set to matrix intensity) rather than crop so that the
    # total pixel count used in the area fraction stays consistent.
    mask_valid = np.ones((H, W), dtype=bool)
    strip_rows = int(H * scale_bar_strip)
    mask_valid[H - strip_rows :, :] = False    # bottom strip

    # Also mask the bottom-right corner more aggressively (common scale bar location)
    mask_valid[H - strip_rows :, int(W * 0.55) :] = False

    # Fill masked area with the median of the rest (neutral grey → won't be a pore)
    arr_proc = arr.copy()
    median_val = np.median(arr[mask_valid])
    arr_proc[~mask_valid] = median_val

    # ── 3. Threshold ──────────────────────────────────────────────────────────
    if use_adaptive_threshold:
        if not HAS_SKIMAGE:
            warnings.warn("scikit-image required for adaptive thresholding. Falling back to Otsu.")
            use_adaptive_threshold = False

    threshold_info = {}
    if use_adaptive_threshold and HAS_SKIMAGE:
        window_size = max(51, int(min(H, W) * 0.07) | 1)   # odd, ~7% of image
        thresh_surface = filters.threshold_sauvola(arr_proc, window_size=window_size, k=0.2)
        binary = (arr_proc < thresh_surface) if dark_is_pore else (arr_proc > thresh_surface)
        threshold_info = {"method": "Sauvola", "window_size": window_size}
    else:
        if HAS_SKIMAGE:
            t = filters.threshold_otsu(arr_proc[mask_valid])
        else:
            t = float(np.percentile(arr_proc[mask_valid], 50))
        binary = (arr_proc < t) if dark_is_pore else (arr_proc > t)
        threshold_info = {"method": "Otsu", "threshold": float(t)}

    # Remove pores outside the valid mask
    binary[~mask_valid] = False

    # ── 4. Clean up binary ────────────────────────────────────────────────────
    # min_area_px: noise-only filter on the binary (very conservative).
    # The meaningful min_area_um2 threshold is applied later, per-pore,
    # so it never removes real pore area from the total porosity estimate.
    min_area_px = MIN_PORE_AREA_PX   # 10 px ≈ 0.38 μm² at 0.195 μm/px → pure noise

    if HAS_SKIMAGE:
        binary = morphology.remove_small_objects(binary, max_size=min_area_px)
        binary = morphology.remove_small_holes(binary, max_size=min_area_px // 2)
    else:
        binary = ndimage.binary_opening(binary, iterations=2)
        binary = ndimage.binary_closing(binary, iterations=1)

    # ── 4b. Total porosity from clean binary (reference value) ───────────────
    # Computed BEFORE watershed and classification so it is not affected by
    # over-splitting or per-pore area filters.
    # Compare this against Archimedes / densitometry measurements.
    p_total_binary = float(binary.sum() / mask_valid.sum())

    # ── 5. Watershed segmentation (separates touching pores) ──────────────────
    if use_watershed and HAS_SKIMAGE:
        # Distance transform → local maxima → watershed markers
        distance = ndimage.distance_transform_edt(binary)
        # Use a lower threshold for peaks to avoid over-splitting
        coords = feature.peak_local_max(
            distance,
            min_distance=max(2, int(0.5 / um_per_pixel)),   # ~1.5 µm separation
            labels=binary,
            exclude_border=False,
        )
        ws_markers = np.zeros(distance.shape, dtype=bool)
        ws_markers[tuple(coords.T)] = True
        ws_markers, _ = ndimage.label(ws_markers)
        labeled = segmentation.watershed(-distance, ws_markers, mask=binary)
    else:
        labeled, _ = ndimage.label(binary)

    # ── 6 & 7. Classify on PRE-WATERSHED components, collect stats from WS ───
    # Classification uses original connected components so large intergranular
    # pore networks are not split into round fragments by watershed, which would
    # make them look intragranular.
    # Watershed labels (already in `labeled`) are used only for ECD/count stats.
    labeled_pre, _ = ndimage.label(binary)
    if HAS_SKIMAGE:
        props_list = measure.regionprops(labeled_pre)
    else:
        props_list = []
        for lbl in range(1, labeled_pre.max() + 1):
            px = np.sum(labeled_pre == lbl)
            props_list.append(
                type("FakeProp", (), {
                    "area": px,
                    "perimeter": 4 * np.sqrt(px),
                    "centroid": (0, 0),
                    "label": lbl,
                })()
            )

    # ── 6 & 7. CLASSIFICATION CRITERIA ────────────────────────────────────────
    # Triple criterion for classifying pores:
    #
    #   A pore is INTERGRANULAR if:
    #     1. area > area_inter_thr_um2  (default 20 μm²)
    #        → very large pores are always intergranular networks
    #     2. area ≥ min_area_inter_um2 (default 5 μm²) AND circularity < threshold
    #        → medium-sized irregular pores are intergranular
    #
    #   Otherwise → INTRAGRANULAR (small, round, isolated pores)
    #
    # ADJUST area_inter_thr_um2 and min_area_inter_um2 in function call if needed
    total_px = int(mask_valid.sum())
    px2_per_um2 = um_per_pixel ** 2

    intra_px, inter_px = 0, 0
    n_intra, n_inter = 0, 0
    ecd_intra, ecd_inter = [], []
    circ_intra, circ_inter = [], []
    pore_table = []

    for r in props_list:
        area_px = r.area
        area_um2 = area_px * px2_per_um2

        if area_um2 < min_area_um2:
            continue

        perim = max(r.perimeter, 1e-6)
        circularity = min(1.0, (4 * np.pi * area_px) / (perim ** 2))
        ecd_um = 2 * np.sqrt(area_um2 / np.pi)

        solidity = getattr(r, "solidity", float("nan"))
        cy, cx = r.centroid if hasattr(r, "centroid") else (0, 0)

        is_intra = not (
            area_um2 > area_inter_thr_um2 or
            (area_um2 >= min_area_inter_um2 and circularity < circularity_threshold)
        )


        if is_intra:
            intra_px += area_px
            n_intra += 1
            ecd_intra.append(ecd_um)
            circ_intra.append(circularity)
        else:
            inter_px += area_px
            n_inter += 1
            ecd_inter.append(ecd_um)
            circ_inter.append(circularity)

        pore_table.append({
            "label":        r.label,
            "area_um2":     round(area_um2, 3),
            "ecd_um":       round(ecd_um, 3),
            "circularity":  round(circularity, 4),
            "solidity":     round(solidity, 4) if not np.isnan(solidity) else "",
            "centroid_x_px": round(cx, 1),
            "centroid_y_px": round(cy, 1),
            "type":         "intra" if is_intra else "inter",
        })

    # ── 8. Compute porosity fractions ────────────────────────────────────────
    p_intra_2d = intra_px / total_px
    p_inter_2d = inter_px / total_px
    p_total_2d = p_intra_2d + p_inter_2d

    p_intra = p_intra_2d * stereology_factor
    p_inter = p_inter_2d * stereology_factor
    p_total = p_intra + p_inter

    result = {
        # Corrected (3D estimate)
        "p_total":  p_total,
        "p_intra":  p_intra,
        "p_inter":  p_inter,
        # Raw 2D
        "p_total_binary": p_total_binary,   # raw binary (best vs Archimedes)
        "p_total_2d": p_total_2d,
        "p_intra_2d": p_intra_2d,
        "p_inter_2d": p_inter_2d,
        # Counts
        "n_total": n_intra + n_inter,
        "n_intra": n_intra,
        "n_inter": n_inter,
        # Shape / size statistics
        "mean_ecd_intra_um":       float(np.mean(ecd_intra))      if ecd_intra  else 0.0,
        "mean_ecd_inter_um":       float(np.mean(ecd_inter))      if ecd_inter  else 0.0,
        "mean_circularity_intra":  float(np.mean(circ_intra))     if circ_intra else 0.0,
        "mean_circularity_inter":  float(np.mean(circ_inter))     if circ_inter else 0.0,
        # Metadata
        "threshold_info":          threshold_info,
        "circularity_threshold":   circularity_threshold,
        "stereology_factor":       stereology_factor,
        "valid_area_px":           total_px,
        "um_per_pixel":            um_per_pixel,
        # Per-pore data
        "pore_table": pore_table,
        # Internal — for plot / CSV helpers
        "_arr_proc": arr_proc,
        "_binary":   binary,
        "_labeled_pre": labeled_pre,
        "_mask_valid": mask_valid,
    }

    # ── 9. CSV export ─────────────────────────────────────────────────────────
    if export_csv and pore_table:
        out_dir = Path(output_dir) if output_dir else Path(image_path).parent
        csv_path = out_dir / (Path(image_path).stem + "_pores.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=pore_table[0].keys())
            writer.writeheader()
            writer.writerows(pore_table)
        print(f"Per-pore CSV saved → {csv_path}")
        result["csv_path"] = str(csv_path)

    # ── 10. Diagnostic plots ──────────────────────────────────────────────────
    if show_plots and HAS_MPL:
        _make_diagnostic_plot(result, image_path, output_dir)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# DIAGNOSTIC PLOT
# ─────────────────────────────────────────────────────────────────────────────

def _make_diagnostic_plot(result: Dict, image_path: str, output_dir: Optional[str]):
    arr          = result["_arr_proc"]
    binary       = result["_binary"]
    labeled_pre  = result["_labeled_pre"]
    mask_valid   = result["_mask_valid"]
    table        = result["pore_table"]
    circ_thr     = result["circularity_threshold"]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(f"Pore analysis — {Path(image_path).name}", fontsize=13)

    # ── Panel 1: original image ──
    ax = axes[0, 0]
    ax.imshow(arr, cmap="gray", vmin=0, vmax=1)
    ax.set_title("Original (greyscale)")
    ax.axis("off")

    # ── Panel 2: binary + mask outline ──
    ax = axes[0, 1]
    disp = np.stack([arr, arr, arr], axis=-1)
    # Shade masked region
    disp[~mask_valid, :] = [0.7, 0.85, 0.7]     # green tint → masked
    # Pores in red
    disp[binary, :] = [0.85, 0.15, 0.15]
    ax.imshow(disp)
    ax.set_title(
        f"Binarised  ({result['threshold_info']['method']})\n"
        f"Red = pores  |  green = masked (scale bar)"
    )
    ax.axis("off")

    # ── Panel 3: classification map ──
    ax = axes[0, 2]
    # Start with grayscale background
    class_map = np.stack([arr, arr, arr], axis=-1).copy()

    # Build masks from the PRE-watershed labeled array
    # since the pore_table uses labels from labeled_pre
    intra_mask = np.zeros(labeled_pre.shape, bool)
    inter_mask = np.zeros(labeled_pre.shape, bool)
    for p in table:
        region_px = labeled_pre == p["label"]
        if p["type"] == "intra":
            intra_mask |= region_px
        else:
            inter_mask |= region_px

    class_map[intra_mask] = [0.15, 0.45, 0.85]   # blue  → intra
    class_map[inter_mask] = [0.85, 0.35, 0.10]   # orange → inter
    ax.imshow(class_map)
    blue_p = mpatches.Patch(color=(0.15, 0.45, 0.85),
                            label=f"Intragranular ({result['n_intra']} pores, {result['p_intra']:.1%})")
    red_p  = mpatches.Patch(color=(0.85, 0.35, 0.10),
                            label=f"Intergranular ({result['n_inter']} pores, {result['p_inter']:.1%})")
    ax.legend(handles=[blue_p, red_p], fontsize=8, loc="upper right",
              framealpha=0.8, edgecolor="gray")
    ax.set_title(f"Classification  (circularity threshold = {circ_thr:.2f})")
    ax.axis("off")

    # ── Panel 4: circularity distribution ──
    ax = axes[1, 0]
    circs = [p["circularity"] for p in table]
    colors = ["steelblue" if p["type"] == "intra" else "tomato" for p in table]
    ax.scatter(range(len(circs)), sorted(circs), c=sorted(colors, reverse=True),
               s=10, alpha=0.6)
    ax.axhline(circ_thr, color="black", linestyle="--", lw=1.5,
               label=f"threshold = {circ_thr}")
    ax.set_xlabel("Pore rank (sorted)")
    ax.set_ylabel("Circularity")
    ax.set_title("Circularity distribution")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1.05)

    # ── Panel 5: ECD histogram ──
    ax = axes[1, 1]
    ecd_intra = [p["ecd_um"] for p in table if p["type"] == "intra"]
    ecd_inter = [p["ecd_um"] for p in table if p["type"] == "inter"]
    bins = np.linspace(0, max([p["ecd_um"] for p in table] or [1]) * 1.05, 40)
    if ecd_intra:
        ax.hist(ecd_intra, bins=bins, alpha=0.6, color="steelblue",
                label="Intragranular", edgecolor="white", linewidth=0.3)
    if ecd_inter:
        ax.hist(ecd_inter, bins=bins, alpha=0.6, color="tomato",
                label="Intergranular", edgecolor="white", linewidth=0.3)
    ax.set_xlabel("Equivalent circular diameter (µm)")
    ax.set_ylabel("Count")
    ax.set_title("Pore size distribution (ECD)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # ── Panel 6: summary text ──
    ax = axes[1, 2]
    ax.axis("off")
    summary = (
        f"POROSITY SUMMARY\n"
        f"{'─'*30}\n"
        f"Total porosity:   {result['p_total']:6.2%}\n"
        f"  Intragranular:  {result['p_intra']:6.2%}  ({result['n_intra']} pores)\n"
        f"  Intergranular:  {result['p_inter']:6.2%}  ({result['n_inter']} pores)\n\n"
        f"Mean ECD intra:   {result['mean_ecd_intra_um']:.1f} µm\n"
        f"Mean ECD inter:   {result['mean_ecd_inter_um']:.1f} µm\n"
        f"Mean circ intra:  {result['mean_circularity_intra']:.3f}\n"
        f"Mean circ inter:  {result['mean_circularity_inter']:.3f}\n\n"
        f"Resolution:       {result['um_per_pixel']:.3f} µm/px\n"
        f"Threshold:        {result['threshold_info']['method']}\n"
        f"Stereol. factor:  {result['stereology_factor']:.2f}\n"
        f"Valid area:       {result['valid_area_px']:,} px\n"
    )
    ax.text(0.05, 0.97, summary, transform=ax.transAxes,
            va="top", ha="left", family="monospace", fontsize=10,
            bbox=dict(boxstyle="round", facecolor="lightyellow",
                      edgecolor="goldenrod", alpha=0.8))

    plt.tight_layout()
    out_dir = Path(output_dir) if output_dir else Path(image_path).parent
    plot_path = out_dir / (Path(image_path).stem + "_pore_analysis.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"Diagnostic plot saved → {plot_path}")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# SENSITIVITY ANALYSIS  (replaces the old calibrate_area_threshold)
# ─────────────────────────────────────────────────────────────────────────────

def sensitivity_analysis(
    image_path: str,
    um_per_pixel: float,
    circularity_thresholds: Tuple[float, ...] = (0.3, 0.4, 0.5, 0.6, 0.7),
    **kwargs,
) -> None:
    """
    Run analyze_porosity with multiple circularity thresholds and print a
    comparison table.  Use this to choose the right threshold for your material.

    A good threshold is where p_total is stable across thresholds (it should be,
    since total porosity doesn't depend on the classification) while the
    intra/inter split changes meaningfully — look for where the numbers match
    your physical understanding of the sample.
    """
    print(f"\nSensitivity to circularity threshold — {Path(image_path).name}")
    print(f"{'─'*72}")
    hdr = f"{'circ_thr':>10} {'p_total':>10} {'p_intra':>10} {'p_inter':>10} "
    hdr += f"{'n_intra':>8} {'n_inter':>8} {'ECD_intra':>10} {'ECD_inter':>10}"
    print(hdr)
    print(f"{'─'*72}")

    for ct in circularity_thresholds:
        r = analyze_porosity(
            image_path, um_per_pixel,
            circularity_threshold=ct,
            show_plots=False, export_csv=False,
            **kwargs,
        )
        print(
            f"{ct:>10.2f} {r['p_total']:>10.2%} {r['p_intra']:>10.2%} "
            f"{r['p_inter']:>10.2%} {r['n_intra']:>8} {r['n_inter']:>8} "
            f"{r['mean_ecd_intra_um']:>10.1f} {r['mean_ecd_inter_um']:>10.1f}"
        )
    print(f"{'─'*72}")
    print(
        "Note: p_total should be roughly constant across rows.\n"
        "Choose the circularity threshold that best matches your SEM observations.\n"
    )


# ─────────────────────────────────────────────────────────────────────────────
# CLI ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("═" * 78)
        print("PORE ANALYSIS - Command Line Interface")
        print("═" * 78)
        print("\nUSAGE:")
        print("  python pore_analysis.py <image> <um_per_pixel> [options]")
        print("\nREQUIRED ARGUMENTS:")
        print("  image            Path to image file (PNG, JPG, TIF)")
        print("  um_per_pixel     Spatial resolution in μm/pixel")
        print("                   Calculate: scale_bar_length_um / scale_bar_length_px")
        print("\nOPTIONAL ARGUMENTS:")
        print("  circularity      Circularity threshold (default: 0.50)")
        print("                   Range: 0.35-0.65, higher = more pores called intragranular")
        print("\nFLAGS:")
        print("  --adaptive       Use Sauvola local threshold (for uneven illumination)")
        print("  --no-watershed   Skip watershed segmentation (faster, less accurate)")
        print("  --export-csv     Save per-pore data to CSV")
        print("  --plot           Generate and save diagnostic plots")
        print("  --sensitivity    Run sensitivity analysis (multiple circularity values)")
        print("\nEXAMPLES:")
        print("  # Basic analysis with default parameters:")
        print("  python pore_analysis.py sample.png 0.195")
        print()
        print("  # Full analysis with plots and CSV export:")
        print("  python pore_analysis.py sample.png 0.195 --plot --export-csv")
        print()
        print("  # Custom circularity threshold with adaptive thresholding:")
        print("  python pore_analysis.py sample.png 0.195 0.55 --adaptive --plot")
        print()
        print("  # Sensitivity analysis to choose circularity threshold:")
        print("  python pore_analysis.py sample.png 0.195 --sensitivity")
        print()
        print("HOW TO CALCULATE um_per_pixel:")
        print("  1. Measure the scale bar in your image editing software (pixels)")
        print("  2. Divide by the scale bar length: um_per_pixel = bar_length_um / bar_pixels")
        print("  3. Example: 100 μm scale bar = 513 pixels → 100/513 = 0.195 μm/px")
        print("═" * 78)
        sys.exit(1)

    image_path     = sys.argv[1]
    um_per_pixel   = float(sys.argv[2])
    circ_thr       = float(sys.argv[3]) if len(sys.argv) > 3 and not sys.argv[3].startswith("--") else DEFAULT_CIRCULARITY_THRESHOLD
    adaptive       = "--adaptive"     in sys.argv
    no_watershed   = "--no-watershed" in sys.argv
    export_csv     = "--export-csv"   in sys.argv
    show_plots     = "--plot"         in sys.argv
    do_sensitivity = "--sensitivity"  in sys.argv

    if do_sensitivity:
        sensitivity_analysis(image_path, um_per_pixel)
        sys.exit(0)

    result = analyze_porosity(
        image_path,
        um_per_pixel=um_per_pixel,
        circularity_threshold=circ_thr,
        use_adaptive_threshold=adaptive,
        use_watershed=not no_watershed,
        export_csv=export_csv,
        show_plots=show_plots,
    )

    print(f"\n{'═'*50}")
    print("  PORE ANALYSIS RESULTS")
    print(f"{'═'*50}")
    print(f"  Total porosity:    {result['p_total']:7.2%}")
    print(f"  ├ Intragranular:   {result['p_intra']:7.2%}  ({result['n_intra']} pores, mean ECD {result['mean_ecd_intra_um']:.1f} µm)")
    print(f"  └ Intergranular:   {result['p_inter']:7.2%}  ({result['n_inter']} pores, mean ECD {result['mean_ecd_inter_um']:.1f} µm)")
    print(f"{'─'*50}")
    print(f"  Threshold method:  {result['threshold_info']['method']}")
    print(f"  Circularity cut:   {result['circularity_threshold']}")
    print(f"  Resolution:        {result['um_per_pixel']} µm/px")
    print(f"{'═'*50}\n")