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

# Extract values for distributed morphology (line 3, distributed_77.png)
P_TOTAL=$(awk -F',' 'NR==3 {print $3}' "$CSV_FILE")
P_INTRA=$(awk -F',' 'NR==3 {print $5}' "$CSV_FILE")

echo "========================================================================"
echo "DISTRIBUTED OPTIMIZATION - Using values from CSV"
echo "========================================================================"
echo "CSV file:          $CSV_FILE"
echo "Total porosity:    ${P_TOTAL} ($(awk "BEGIN {printf \"%.1f\", $P_TOTAL * 100}")%)"
echo "Intra porosity:    ${P_INTRA} ($(awk "BEGIN {printf \"%.1f\", $P_INTRA * 100}")%)"
echo "Boundary porosity: 0.0% (distributed = isolated spherical pores)"
echo ""
echo "Using total porosity (${P_TOTAL}) as target for optimization"
echo "========================================================================"
echo ""

# Run distributed optimization with Amitex
# Only pass total porosity (distributed mode has no boundary pores)
python project_root/experiments/run_optimization.py \
  --mode distributed \
  --exp-image "Optimization_3D_structure/exp_img/distributed_77.png" \
  --target-porosity ${P_TOTAL} \
  --n-calls 50 \
  --n3d 120 \
  --run-amitex

echo ""
echo "========================================================================"
echo "OPTIMIZATION COMPLETE"
echo "========================================================================"
echo "Results saved to: Results_Optimization_Distributed/"
echo ""
