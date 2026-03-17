#!/bin/bash
set -e  # Exit on error

# Activate Merope conda environment
# source Env_Merope.sh
# conda activate merope

# Run distributed optimization with Amitex
python project_root/experiments/run_optimization.py \
  --mode distributed \
  --exp-image "Optimization_3D_structure/EXP IMG/distributed_77.png" \
  --n-calls 50 \
  --n3d 150 \
  --run-amitex

# Predict K_eff from optimization results
python project_root/experiments/predict_keff_from_optimization.py Results_Optimization_Distributed
