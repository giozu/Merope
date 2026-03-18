#!/bin/bash
set -e  # Exit on error

# Source Merope environment (commented - depends on your system)
# conda activate merope
# source Env_Merope.sh

# Read porosity values from CSV
CSV_FILE="pore_analysis_results.csv"

if [ ! -f "$CSV_FILE" ]; then
    echo "ERROR: $CSV_FILE not found!"
    echo "Please run ./run_pore_analysis.sh first to generate the CSV."
    exit 1
fi

# Extract values for interconnected morphology (line 2, connected_79.png)
P_TOTAL=$(awk -F',' 'NR==2 {print $3}' "$CSV_FILE")
P_BOUNDARY=$(awk -F',' 'NR==2 {print $4}' "$CSV_FILE")
P_INTRA=$(awk -F',' 'NR==2 {print $5}' "$CSV_FILE")

echo "========================================================================"
echo "INTERCONNECTED OPTIMIZATION - Using values from CSV"
echo "========================================================================"
echo "CSV file:          $CSV_FILE"
echo "Total porosity:    ${P_TOTAL} ($(awk "BEGIN {printf \"%.1f\", $P_TOTAL * 100}")%)"
echo "Boundary porosity: ${P_BOUNDARY} ($(awk "BEGIN {printf \"%.1f\", $P_BOUNDARY * 100}")%)"
echo "Intra porosity:    ${P_INTRA} ($(awk "BEGIN {printf \"%.1f\", $P_INTRA * 100}")%)"
echo ""
echo "Using boundary porosity (${P_BOUNDARY}) as target for optimization"
echo "========================================================================"
echo ""

# Run interconnected optimization with Amitex
# Pass all porosity values from CSV
python project_root/experiments/run_optimization.py \
  --mode interconnected \
  --exp-image "Optimization_3D_structure/exp_img/connected_79.png" \
  --target-porosity ${P_BOUNDARY} \
  --p-total ${P_TOTAL} \
  --p-boundary ${P_BOUNDARY} \
  --p-intra ${P_INTRA} \
  --n-calls 50 \
  --n3d 120 \
  --run-amitex

echo ""
echo "========================================================================"
echo "OPTIMIZATION COMPLETE"
echo "========================================================================"
echo "Results saved to: Results_Optimization_Interconnected/"
echo ""
