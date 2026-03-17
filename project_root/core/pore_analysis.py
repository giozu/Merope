"""
Automatic pore analysis from experimental images.

This module provides tools to estimate:
- Total porosity
- Boundary-connected porosity (large interconnected pores)
- Intra-granular porosity (small isolated pores)
"""

import numpy as np
from PIL import Image
from scipy import ndimage
from typing import Dict, Tuple
import matplotlib.pyplot as plt

try:
    from skimage import filters, measure, morphology
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False
    print("Warning: scikit-image not found. Using fallback implementations.")


def estimate_porosity_from_image(
    image_path: str,
    area_threshold_um2: float = 50.0,
    um_per_pixel: float = 1.0,
    dark_is_pore: bool = True,
    show_debug: bool = False,
    crop_bottom_right: float = 0.15,
    stereology_correction: float = 0.85,
) -> Dict[str, float]:
    """
    Automatically estimate boundary and intra-granular porosity from SEM/optical image.

    Strategy:
    1. Convert to grayscale and binarize (Otsu threshold)
    2. Identify connected components (pore objects)
    3. Classify pores by size:
       - Large pores (> threshold) → boundary/interconnected
       - Small pores (< threshold) → intra-granular
    4. Compute area fractions

    Parameters
    ----------
    image_path : str
        Path to the experimental image (PNG, JPG, etc.)
    area_threshold_um2 : float
        Threshold area in μm² to separate large vs small pores.
        Typical: 50-100 μm² for boundary pores, <50 μm² for intra-pores.
    um_per_pixel : float
        Spatial resolution (μm/pixel). If unknown, use 1.0 and adjust threshold.
    dark_is_pore : bool
        If True, dark regions are pores. If False, bright regions are pores.
    show_debug : bool
        If True, display debug plots showing segmentation.
    crop_bottom_right : float
        Fraction of image to crop from bottom-right corner to exclude scale bars.
        E.g., 0.15 removes bottom-right 15% (useful for scale bars).
    stereology_correction : float
        Correction factor for 2D→3D conversion (default 0.85).
        2D images systematically overestimate porosity.
        Typical range: 0.80-0.90 for ceramics.
        Calibrate against Archimedes or 3D CT measurements.

    Returns
    -------
    dict
        {
            "p_total": total porosity (fraction),
            "p_boundary": boundary/interconnected porosity (fraction),
            "p_intra": intra-granular porosity (fraction),
            "n_boundary_pores": count of large pores,
            "n_intra_pores": count of small pores,
            "threshold_intensity": Otsu threshold value used,
        }

    Example
    -------
    >>> result = estimate_porosity_from_image(
    ...     "connected_79.png",
    ...     area_threshold_um2=80.0,
    ...     um_per_pixel=100.0 / 512,  # 100 μm scale bar, 512 pixels wide
    ...     show_debug=True
    ... )
    >>> print(f"Total porosity: {result['p_total']:.1%}")
    >>> print(f"Boundary: {result['p_boundary']:.1%}, Intra: {result['p_intra']:.1%}")
    """

    # 1. Load image and convert to grayscale
    img = Image.open(image_path).convert("L")
    img_array = np.array(img, dtype=float) / 255.0  # Normalize to [0, 1]

    # 1b. Crop bottom-right corner to exclude scale bar
    if crop_bottom_right > 0:
        h, w = img_array.shape
        crop_h = int(h * (1 - crop_bottom_right))
        crop_w = int(w * (1 - crop_bottom_right))
        img_array = img_array[:crop_h, :crop_w]
        print(f"Cropped to {crop_h}×{crop_w} (removed bottom-right {crop_bottom_right:.0%})")

    # 2. Binarize using Otsu threshold
    if HAS_SKIMAGE:
        threshold = filters.threshold_otsu(img_array)
    else:
        # Fallback: simple percentile-based threshold
        threshold = np.percentile(img_array, 50)

    if dark_is_pore:
        binary = img_array < threshold  # Dark = pore (True)
    else:
        binary = img_array > threshold  # Bright = pore (True)

    # 3. Clean up binary image (remove noise)
    if HAS_SKIMAGE:
        binary_cleaned = morphology.remove_small_objects(binary, min_size=5)
        binary_cleaned = morphology.remove_small_holes(binary_cleaned, area_threshold=10)
    else:
        # Fallback: simple morphological operations
        binary_cleaned = ndimage.binary_opening(binary, iterations=1)
        binary_cleaned = ndimage.binary_closing(binary_cleaned, iterations=1)

    # 4. Label connected components
    labeled, num_features = ndimage.label(binary_cleaned)

    if HAS_SKIMAGE:
        props = measure.regionprops(labeled)
    else:
        # Fallback: manual region properties
        props = []
        for label_id in range(1, num_features + 1):
            mask = (labeled == label_id)
            area = np.sum(mask)
            props.append(type('Region', (), {'area': area, 'label': label_id})())

    # 5. Compute pixel area threshold
    area_threshold_px = area_threshold_um2 / (um_per_pixel ** 2)

    # 6. Classify pores by size
    total_pixels = img_array.size
    boundary_pixels = 0
    intra_pixels = 0
    n_boundary = 0
    n_intra = 0

    for region in props:
        area_px = region.area
        if area_px >= area_threshold_px:
            # Large pore → boundary/interconnected
            boundary_pixels += area_px
            n_boundary += 1
        else:
            # Small pore → intra-granular
            intra_pixels += area_px
            n_intra += 1

    # 7. Compute porosity fractions (2D area fractions)
    p_boundary_2d = boundary_pixels / total_pixels
    p_intra_2d = intra_pixels / total_pixels
    p_total_2d = p_boundary_2d + p_intra_2d

    # 8. Apply stereological correction (2D → 3D)
    # Empirical factor to account for 2D overestimation
    # Typical range: 0.80-0.90 for ceramics
    # Can be calibrated against Archimedes measurements
    stereology_factor = stereology_correction

    p_boundary = p_boundary_2d * stereology_factor
    p_intra = p_intra_2d * stereology_factor
    p_total = p_total_2d * stereology_factor

    result = {
        "p_total": float(p_total),
        "p_boundary": float(p_boundary),
        "p_intra": float(p_intra),
        "p_total_2d": float(p_total_2d),  # Raw 2D measurement
        "stereology_factor": float(stereology_factor),
        "n_boundary_pores": n_boundary,
        "n_intra_pores": n_intra,
        "threshold_intensity": float(threshold),
        "area_threshold_um2": area_threshold_um2,
        "area_threshold_px": float(area_threshold_px),
    }

    # 8. Debug visualization
    if show_debug:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # Original
        axes[0, 0].imshow(img_array, cmap='gray')
        axes[0, 0].set_title("Original Image")
        axes[0, 0].axis('off')

        # Binary
        axes[0, 1].imshow(binary_cleaned, cmap='gray')
        axes[0, 1].set_title(f"Binary (Otsu={threshold:.2f})")
        axes[0, 1].axis('off')

        # Labeled
        axes[0, 2].imshow(labeled, cmap='nipy_spectral')
        axes[0, 2].set_title(f"Labeled ({num_features} objects)")
        axes[0, 2].axis('off')

        # Boundary pores only
        boundary_mask = np.zeros_like(labeled, dtype=bool)
        for region in props:
            if region.area >= area_threshold_px:
                boundary_mask[labeled == region.label] = True
        axes[1, 0].imshow(boundary_mask, cmap='Reds')
        axes[1, 0].set_title(f"Boundary pores: {p_boundary:.1%} ({n_boundary})")
        axes[1, 0].axis('off')

        # Intra pores only
        intra_mask = np.zeros_like(labeled, dtype=bool)
        for region in props:
            if region.area < area_threshold_px:
                intra_mask[labeled == region.label] = True
        axes[1, 1].imshow(intra_mask, cmap='Blues')
        axes[1, 1].set_title(f"Intra pores: {p_intra:.1%} ({n_intra})")
        axes[1, 1].axis('off')

        # Histogram of pore areas
        areas = [r.area * (um_per_pixel ** 2) for r in props]
        axes[1, 2].hist(areas, bins=50, edgecolor='black', alpha=0.7)
        axes[1, 2].axvline(area_threshold_um2, color='red', linestyle='--',
                          label=f'Threshold={area_threshold_um2} μm²')
        axes[1, 2].set_xlabel('Pore area (μm²)')
        axes[1, 2].set_ylabel('Count')
        axes[1, 2].set_title('Pore size distribution')
        axes[1, 2].set_yscale('log')
        axes[1, 2].legend()
        axes[1, 2].grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(image_path.replace('.png', '_pore_analysis.png'), dpi=150)
        print(f"Debug plot saved to: {image_path.replace('.png', '_pore_analysis.png')}")
        plt.close()

    return result


def calibrate_area_threshold(
    image_path: str,
    um_per_pixel: float = 1.0,
    dark_is_pore: bool = True,
) -> float:
    """
    Interactive calibration to find optimal area threshold.

    Tries multiple thresholds and displays results to help user choose.

    Parameters
    ----------
    image_path : str
        Path to experimental image
    um_per_pixel : float
        Spatial resolution
    dark_is_pore : bool
        Pore color convention

    Returns
    -------
    float
        Suggested area threshold in μm²
    """
    # Try a range of thresholds
    thresholds = [10, 25, 50, 100, 200, 500]

    print(f"\nCalibrating area threshold for: {image_path}")
    print("=" * 70)
    print(f"{'Threshold (μm²)':<20} {'p_total':<12} {'p_boundary':<12} {'p_intra':<12} {'#boundary':<10} {'#intra':<10}")
    print("-" * 70)

    results = []
    for thresh in thresholds:
        res = estimate_porosity_from_image(
            image_path,
            area_threshold_um2=thresh,
            um_per_pixel=um_per_pixel,
            dark_is_pore=dark_is_pore,
            show_debug=False,
        )
        results.append(res)
        print(f"{thresh:<20.1f} {res['p_total']:<12.3f} {res['p_boundary']:<12.3f} "
              f"{res['p_intra']:<12.3f} {res['n_boundary_pores']:<10} {res['n_intra_pores']:<10}")

    print("=" * 70)
    print("\nLook for the threshold where p_boundary stabilizes and p_intra captures small pores.")
    print("Typical range: 50-100 μm² for boundary pores in polycrystals.\n")

    # Find threshold where derivative of p_boundary is smallest (stable point)
    derivs = [abs(results[i+1]['p_boundary'] - results[i]['p_boundary'])
              for i in range(len(results)-1)]
    best_idx = np.argmin(derivs)
    suggested = thresholds[best_idx]

    print(f"Suggested threshold: {suggested} μm² (most stable p_boundary)")
    return float(suggested)


if __name__ == "__main__":
    # Example usage
    import sys

    if len(sys.argv) < 2:
        print("Usage: python pore_analysis.py <image_path> [um_per_pixel] [area_threshold]")
        print("\nExample:")
        print("  python pore_analysis.py connected_79.png 0.195 80")
        sys.exit(1)

    image_path = sys.argv[1]
    um_per_pixel = float(sys.argv[2]) if len(sys.argv) > 2 else 0.195  # 100 μm / 512 px
    area_threshold = float(sys.argv[3]) if len(sys.argv) > 3 else None

    if area_threshold is None:
        # Run calibration
        area_threshold = calibrate_area_threshold(image_path, um_per_pixel)
        print(f"\nRunning analysis with threshold = {area_threshold} μm²...")

    # Run full analysis with debug plots
    result = estimate_porosity_from_image(
        image_path,
        area_threshold_um2=area_threshold,
        um_per_pixel=um_per_pixel,
        show_debug=True,
    )

    print(f"\n{'='*60}")
    print(f"POROSITY ANALYSIS RESULTS")
    print(f"{'='*60}")
    print(f"Total porosity:     {result['p_total']:6.1%}")
    print(f"  Boundary pores:   {result['p_boundary']:6.1%}  ({result['n_boundary_pores']} objects)")
    print(f"  Intra pores:      {result['p_intra']:6.1%}  ({result['n_intra_pores']} objects)")
    print(f"{'='*60}\n")
