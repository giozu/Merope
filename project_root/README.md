# Merope Thermal Simulation Suite
Professional Research Library for FFT-based Microstructural Analysis.

## Project Structure
- **/lib/**: Core reusable logic (Solver Engine, Calibration logic).
- **/experiments/**: Individual research studies (Convergence, Science Sweeps).
- **/results/**: (Generated) Output CSVs and VTK visualizations.

## Getting Started
1. **Requirements**: 
   - Python 3.8+
   - Merope (Microstructure Generation)
   - AMITEX_FFTP (Mechanical/Thermal Solver)
   - NumPy, SciPy, Matplotlib

2. **Installation**:
   Add the `lib/` folder to your `PYTHONPATH` to allow experiment scripts to import the engine.
   ```bash
   export PYTHONPATH=$PYTHONPATH:$(pwd)/lib
   ```

3. **Workflow Order**:
   - Step 1: Run `lib/calibration.py` to identify the voxel factor `C_LAYER`.
   - Step 2: Run `experiments/run_convergence.py` to find stable domain size.
   - Step 3: Run `experiments/run_science_sweeps.py` for final data.

## Physics Note
All variables named `K` refer strictly to **Effective Thermal Conductivity**.
The solver uses FFT homogenization to solve the steady-state heat equation across heterogeneous domains.
