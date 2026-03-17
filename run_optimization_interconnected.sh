#!/bin/bash
set -e  # Exit on error

# Source Merope environment
# conda activate merope
# source Env_Merope.sh

# Run interconnected optimization with Amitex
python project_root/experiments/run_optimization.py \
  --mode interconnected \
  --exp-image "Optimization_3D_structure/EXP IMG/connected_79.png" \
  --n-calls 50 \
  --n3d 150 \
  --run-amitex

# Predict K_eff from optimization results
python project_root/experiments/predict_keff_from_optimization.py Results_Optimization_Interconnected
