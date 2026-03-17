#!/bin/bash
set -e  # Exit on error

# Activate environment (commented - depends on your system)
# source Env_Merope.sh
# conda activate merope

echo "========================================================================"
echo "PORE ANALYSIS - Experimental Images"
echo "========================================================================"
echo ""

# Parameters
IMAGE_DIR="Optimization_3D_structure/EXP IMG"
UM_PER_PIXEL=0.195  # 100 μm scale bar / 512 px image width
AREA_THRESHOLD=80.0  # μm² threshold for boundary vs intra pores
OUTPUT_CSV="pore_analysis_results.csv"

echo "Parameters:"
echo "  Image directory:      ${IMAGE_DIR}"
echo "  Scale:                ${UM_PER_PIXEL} μm/pixel"
echo "  Area threshold:       ${AREA_THRESHOLD} μm²"
echo "  Output CSV:           ${OUTPUT_CSV}"
echo ""

# Create CSV header
echo "image_name,morphology,p_total,p_boundary,p_intra,n_boundary_pores,n_intra_pores,threshold_um2,um_per_pixel" > ${OUTPUT_CSV}

# ========================================================================
# 1. Analyze INTERCONNECTED morphology (connected_79.png)
# ========================================================================
echo "========================================================================"
echo "1. INTERCONNECTED MORPHOLOGY - connected_79.png"
echo "========================================================================"

python3 << PYTHON_EOF
import sys
sys.path.insert(0, 'project_root')
from core.pore_analysis import estimate_porosity_from_image

result = estimate_porosity_from_image(
    '${IMAGE_DIR}/connected_79.png',
    area_threshold_um2=${AREA_THRESHOLD},
    um_per_pixel=${UM_PER_PIXEL},
    crop_bottom_right=0.15,
    stereology_correction=0.85,
    show_debug=True,
)

print(f'\n{"="*60}')
print(f'POROSITY ANALYSIS - connected_79.png')
print(f'{"="*60}')
print(f'Total porosity:     {result["p_total"]:6.1%}')
print(f'  Boundary pores:   {result["p_boundary"]:6.1%}  ({result["n_boundary_pores"]} objects)')
print(f'  Intra pores:      {result["p_intra"]:6.1%}  ({result["n_intra_pores"]} objects)')
print(f'{"="*60}')

# Append to CSV
with open('${OUTPUT_CSV}', 'a') as f:
    f.write(f'connected_79.png,interconnected,{result["p_total"]:.4f},{result["p_boundary"]:.4f},{result["p_intra"]:.4f},{result["n_boundary_pores"]},{result["n_intra_pores"]},${AREA_THRESHOLD},${UM_PER_PIXEL}\n')
PYTHON_EOF

echo ""
echo "------------------------------------------------------------------------"
echo ""

# ========================================================================
# 2. Analyze DISTRIBUTED morphology (distributed_77.png)
# ========================================================================
echo "========================================================================"
echo "2. DISTRIBUTED MORPHOLOGY - distributed_77.png"
echo "========================================================================"

python3 << PYTHON_EOF
import sys
sys.path.insert(0, 'project_root')
from core.pore_analysis import estimate_porosity_from_image

result = estimate_porosity_from_image(
    '${IMAGE_DIR}/distributed_77.png',
    area_threshold_um2=${AREA_THRESHOLD},
    um_per_pixel=${UM_PER_PIXEL},
    crop_bottom_right=0.15,
    stereology_correction=0.85,
    show_debug=True,
)

print(f'\n{"="*60}')
print(f'POROSITY ANALYSIS - distributed_77.png')
print(f'{"="*60}')
print(f'Total porosity:     {result["p_total"]:6.1%}')
print(f'  Boundary pores:   {result["p_boundary"]:6.1%}  ({result["n_boundary_pores"]} objects)')
print(f'  Intra pores:      {result["p_intra"]:6.1%}  ({result["n_intra_pores"]} objects)')
print(f'{"="*60}')

# Append to CSV
with open('${OUTPUT_CSV}', 'a') as f:
    f.write(f'distributed_77.png,distributed,{result["p_total"]:.4f},{result["p_boundary"]:.4f},{result["p_intra"]:.4f},{result["n_boundary_pores"]},{result["n_intra_pores"]},${AREA_THRESHOLD},${UM_PER_PIXEL}\n')
PYTHON_EOF

echo ""
echo "========================================================================"
echo "ANALYSIS COMPLETE"
echo "========================================================================"
echo ""
echo "Summary:"
echo "  - connected_79.png:    Interconnected pores (crack-like)"
echo "  - distributed_77.png:  Distributed pores (spherical)"
echo ""
echo "✓ Results saved to: ${OUTPUT_CSV}"
echo "✓ Debug plots saved in: ${IMAGE_DIR}/"
echo ""

# Display CSV content
echo "CSV Preview:"
echo "------------------------------------------------------------------------"
cat ${OUTPUT_CSV}
echo "------------------------------------------------------------------------"
echo ""
