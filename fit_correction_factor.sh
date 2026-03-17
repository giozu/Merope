#!/bin/bash
set -e  # Exit on error

# Activate Merope conda environment
# source Env_Merope.sh
# conda activate merope

# Change directory
cd project_root

# Fit correction factor
python experiments/fit_correction_factor.py
