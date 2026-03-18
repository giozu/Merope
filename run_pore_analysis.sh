#!/bin/bash
set -e

################################################################################
# PORE ANALYSIS BATCH SCRIPT
################################################################################
# This script runs pore analysis on cross-section images.
# It analyzes porosity and classifies pores as intragranular or intergranular
# based on size and circularity.
#
# USAGE:
#   bash run_pore_analysis.sh
#
# OUTPUTS:
#   - Per-image diagnostic plots (*_pore_analysis.png)
#   - Per-image pore data CSV (*_pores.csv)
#   - Summary CSV (pore_analysis_results.csv)
################################################################################

# Activate environment
# source Env_Merope.sh
# conda activate merope

echo "========================================================================"
echo "PORE ANALYSIS - Experimental Images"
echo "========================================================================"
echo ""

# ══════════════════════════════════════════════════════════════════════════════
# ADJUSTABLE PARAMETERS - Modify these to customize the analysis
# ══════════════════════════════════════════════════════════════════════════════

# ── Image input ───────────────────────────────────────────────────────────────
IMAGE_DIR="Optimization_3D_structure/exp_img"
# Directory containing the input images (.png, .jpg, .tif)

# ── Scale / Resolution ────────────────────────────────────────────────────────
UM_PER_PIXEL=0.195
# Spatial resolution in micrometers per pixel
# HOW TO CALCULATE: measure scale bar in pixels, divide by scale bar length in μm
# Example: if 100 μm scale bar = 513 pixels → 100/513 = 0.195 μm/pixel

# ── Classification threshold ──────────────────────────────────────────────────
CIRCULARITY_THR=0.50
# Circularity threshold to separate intragranular from intergranular pores
# Circularity = 4π·Area/Perimeter²  (1.0 = perfect circle, 0.0 = line)
#
# INTERPRETATION:
#   - Pores with circularity ≥ 0.50 → intragranular (round, isolated)
#   - Pores with circularity < 0.50 → intergranular (elongated, interconnected)
#
# RECOMMENDED VALUES:
#   - Sintered ceramics: 0.45 - 0.55
#   - Powder compacts: 0.40 - 0.50
#   - Use sensitivity_analysis() to calibrate for your material

# ── Noise filtering ───────────────────────────────────────────────────────────
MIN_AREA_UM2=0.5
# Minimum pore area in μm² to include in analysis (noise filter)
# Pores smaller than this are excluded from statistics
# TYPICAL VALUES: 1.0 - 5.0 μm²

# ── Thresholding method ───────────────────────────────────────────────────────
ADAPTIVE=false
# Thresholding algorithm:
#   false → Otsu global threshold (works for most images)
#   true  → Sauvola local adaptive threshold (for uneven illumination)
# USE ADAPTIVE IF: your image has brightness gradients or uneven lighting

# ── Pore separation ───────────────────────────────────────────────────────────
NO_WATERSHED=false
# Watershed segmentation to separate touching pores:
#   false → use watershed (more accurate, separates merged pores)
#   true  → skip watershed (faster, but touching pores count as one)
# RECOMMENDATION: keep false for accurate pore counting

# ── Output files ──────────────────────────────────────────────────────────────
OUTPUT_CSV="pore_analysis_results.csv"

# ── Python script location ────────────────────────────────────────────────────
PORE_SCRIPT="$(pwd)/project_root/core/pore_analysis.py"

# ── Sanity check ──────────────────────────────────────────────────────────────
if [ ! -f "${PORE_SCRIPT}" ]; then
    echo "ERROR: pore_analysis.py not found at:"
    echo "       ${PORE_SCRIPT}"
    echo ""
    echo "Copy the file there, or update the PORE_SCRIPT variable above."
    exit 1
fi

echo "Parameters:"
echo "  Image directory:      ${IMAGE_DIR}"
echo "  Scale:                ${UM_PER_PIXEL} μm/pixel"
echo "  Circularity cut:      ${CIRCULARITY_THR}  (< threshold → intergranular)"
echo "  Min pore area:        ${MIN_AREA_UM2} μm²"
echo "  Adaptive threshold:   ${ADAPTIVE}"
echo "  Watershed:            $([ "$NO_WATERSHED" = true ] && echo disabled || echo enabled)"
echo "  Output CSV:           ${OUTPUT_CSV}"
echo "  Script:               ${PORE_SCRIPT}"
echo ""

# ── CSV header ────────────────────────────────────────────────────────────────
echo "image_name,morphology,p_total_binary,p_total_2d,p_total,p_intra,p_inter,n_intra,n_inter,mean_ecd_intra_um,mean_ecd_inter_um,mean_circ_intra,mean_circ_inter,threshold_method,circularity_thr,um_per_pixel" \
  > "${OUTPUT_CSV}"

# ── Helper ────────────────────────────────────────────────────────────────────
run_analysis() {
    local IMAGE_FILE="$1"
    local MORPHOLOGY_LABEL="$2"

    # Resolve strings BEFORE the heredoc (paths with spaces are safe this way)
    local IMG_BASENAME
    IMG_BASENAME="$(basename "${IMAGE_FILE}")"
    local IMG_STEM="${IMG_BASENAME%.png}"

    # Convert bash booleans → Python booleans
    local PY_ADAPTIVE PY_WATERSHED
    [ "$ADAPTIVE"     = true ] && PY_ADAPTIVE="True"  || PY_ADAPTIVE="False"
    [ "$NO_WATERSHED" = true ] && PY_WATERSHED="False" || PY_WATERSHED="True"

    echo "========================================================================"
    echo "  ${MORPHOLOGY_LABEL^^} MORPHOLOGY — ${IMG_BASENAME}"
    echo "========================================================================"

    # Pass all values as environment variables so the heredoc can be fully
    # quoted (<< 'PYTHON_EOF') — quoted heredoc = zero bash expansion inside.
    IMAGE_FILE="$IMAGE_FILE"           \
    IMG_BASENAME="$IMG_BASENAME"       \
    IMG_STEM="$IMG_STEM"               \
    MORPHOLOGY_LABEL="$MORPHOLOGY_LABEL" \
    PY_ADAPTIVE="$PY_ADAPTIVE"         \
    PY_WATERSHED="$PY_WATERSHED"       \
    python3 << 'PYTHON_EOF'
import os, sys, csv, importlib.util

# ── Load pore_analysis.py by file path (no sys.path manipulation needed) ──
_script = os.environ["PORE_SCRIPT"]
_spec   = importlib.util.spec_from_file_location("pore_analysis", _script)
_mod    = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
analyze_porosity = _mod.analyze_porosity

# ── Read parameters from environment variables ─────────────────────────────────
image_file        = os.environ["IMAGE_FILE"]
img_basename      = os.environ["IMG_BASENAME"]
img_stem          = os.environ["IMG_STEM"]
morphology_label  = os.environ["MORPHOLOGY_LABEL"]
um_per_pixel      = float(os.environ["UM_PER_PIXEL"])
circularity_thr   = float(os.environ["CIRCULARITY_THR"])
min_area_um2      = float(os.environ["MIN_AREA_UM2"])
output_csv        = os.environ["OUTPUT_CSV"]
use_adaptive      = os.environ["PY_ADAPTIVE"] == "True"
use_watershed     = os.environ["PY_WATERSHED"] == "True"

result = analyze_porosity(
    image_file,
    um_per_pixel=um_per_pixel,
    circularity_threshold=circularity_thr,
    min_area_um2=min_area_um2,  # Using MIN_AREA_UM2 from shell parameters
    dark_is_pore=True,
    use_adaptive_threshold=use_adaptive,
    use_watershed=use_watershed,
    area_inter_thr_um2=20.0,
    min_area_inter_um2=5.0,
    scale_bar_strip=0.12,
    stereology_factor=1.0,
    export_csv=True,
    show_plots=True,
)

print(f'\n{"="*60}')
print(f'POROSITY RESULTS — {img_basename}')
print(f'{"="*60}')
print(f'  Total (2D raw):    {result["p_total_2d"]:7.2%}')
print(f'  Total (corrected): {result["p_total"]:7.2%}')
print(f'  ├ Intragranular:   {result["p_intra"]:7.2%}  '
      f'({result["n_intra"]} pores, '
      f'ECD {result["mean_ecd_intra_um"]:.1f} μm, '
      f'circ {result["mean_circularity_intra"]:.3f})')
print(f'  └ Intergranular:   {result["p_inter"]:7.2%}  '
      f'({result["n_inter"]} pores, '
      f'ECD {result["mean_ecd_inter_um"]:.1f} μm, '
      f'circ {result["mean_circularity_inter"]:.3f})')
print(f'{"="*60}')
print(f'  Threshold method:  {result["threshold_info"]["method"]}')
print(f'  Per-pore CSV:      {img_stem}_pores.csv')
print()

row = [
    img_basename,
    morphology_label,
    f'{result["p_total_binary"]:.6f}',
    f'{result["p_total_2d"]:.6f}',
    f'{result["p_total"]:.6f}',
    f'{result["p_intra"]:.6f}',
    f'{result["p_inter"]:.6f}',
    result["n_intra"],
    result["n_inter"],
    f'{result["mean_ecd_intra_um"]:.3f}',
    f'{result["mean_ecd_inter_um"]:.3f}',
    f'{result["mean_circularity_intra"]:.4f}',
    f'{result["mean_circularity_inter"]:.4f}',
    result["threshold_info"]["method"],
    circularity_thr,
    um_per_pixel,
]
with open(output_csv, 'a', newline='') as f:
    csv.writer(f).writerow(row)
PYTHON_EOF

    echo ""
    echo "------------------------------------------------------------------------"
    echo ""
}

# ══════════════════════════════════════════════════════════════════════════════
# IMAGE ANALYSIS - Add or modify image files here
# ══════════════════════════════════════════════════════════════════════════════
# Export variables needed inside the quoted heredoc
export PORE_SCRIPT UM_PER_PIXEL CIRCULARITY_THR MIN_AREA_UM2 OUTPUT_CSV

# Run analysis on each image file
# SYNTAX: run_analysis "<path/to/image.png>" "<morphology_label>"
#
# The morphology_label is a text descriptor for the CSV (e.g., "interconnected", "distributed")
# Add more images by copying the line below and changing the path and label

run_analysis "${IMAGE_DIR}/connected_79.png"   "interconnected"
run_analysis "${IMAGE_DIR}/distributed_77.png" "distributed"

# EXAMPLES: uncomment and modify to analyze additional images
# run_analysis "${IMAGE_DIR}/sample_A.png"   "dense"
# run_analysis "${IMAGE_DIR}/sample_B.png"   "porous"
# run_analysis "path/to/another_image.png"   "custom_label"

# ── Done ──────────────────────────────────────────────────────────────────────
echo "========================================================================"
echo "ANALYSIS COMPLETE"
echo "========================================================================"
echo ""
echo "Output files:"
echo "  Summary CSV:   ${OUTPUT_CSV}"
echo "  Per-pore CSVs: ${IMAGE_DIR}/connected_79_pores.csv"
echo "                 ${IMAGE_DIR}/distributed_77_pores.csv"
echo "  Plots:         ${IMAGE_DIR}/connected_79_pore_analysis.png"
echo "                 ${IMAGE_DIR}/distributed_77_pore_analysis.png"
echo ""
echo "CSV preview:"
echo "------------------------------------------------------------------------"
column -t -s',' "${OUTPUT_CSV}"
echo "------------------------------------------------------------------------"