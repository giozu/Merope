#!/bin/bash
set -e  # Exit on error

# Activate Merope conda environment
# source Env_Merope.sh
# conda activate merope

# Extract results
# python3 project_root/experiments/run_keff_vs_delta.py --extract

# Change directory
cd project_root

# Fit correction factor
python experiments/fit_correction_factor.py

# Compare optimization results
cd ..
python3 project_root/experiments/compare_optimization_results.py