#!/bin/bash
set -e  # Exit on error

# Activate Merope conda environment
# source Env_Merope.sh
# conda activate merope

# Run extended delta simulation with recovery
python project_root/experiments/run_keff_vs_delta.py --recover
