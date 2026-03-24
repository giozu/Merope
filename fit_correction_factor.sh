#!/bin/bash
set -e  # Exit on error

# Activate Merope conda environment
# source Env_Merope.sh
# conda activate merope

# Extract results
# python3 project_root/experiments/run_keff_vs_delta.py --extract

# Fit correction factor (output to Merope/Results_Sigmoidal_Fit)
python3 project_root/experiments/fit_correction_factor.py \
    --csv Results_Keff_vs_Delta/keff_vs_delta.csv \
    --output-dir Results_Sigmoidal_Fit

# Compare optimization results
python3 project_root/experiments/compare_optimization_results.py