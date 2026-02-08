# Merope Simulation Suite
**Research Library for FFT-based Microstructural Analysis.**

This suite provides a modular framework for generating complex porous microstructures and calculating their effective thermal conductivity using FFT-based homogenization solvers.

## Project Structure

```text
project_root/
├── core/                  # Core Library Logic
│   ├── geometry.py        # Merope wrappers: Laguerre tessellation & Sphere generation
│   ├── solver.py          # Amitex FFT solver interface & result parsing
│   └── utils.py           # Filesystem management & result logging
├── experiments/           # Research Study Scripts
│   ├── run_anisotropy.py  # Study on directional K vs grain aspect ratio
│   ├── run_delta.py       # Study on grain boundary thickness (interconnected pores)
│   └── run_mixed.py       # Study on dual-porosity (intra + inter-granular)
└── README.md              # Documentation
```

## Getting Started

### 1. Requirements
*   **Merope**: Microstructure generation engine (C++ with Python wrappers).
*   **Sac-de-billes**: Sphere throwing algorithms for RVE seeding.
*   **Amitex_FFTP**: FFT-based solver for periodic homogenization.
*   **Python Dependencies**: `numpy`, `send2trash`.

### 2. Installation
Ensure your Merope environment is active. No standard installation is required, but you must add the project root to your `PYTHONPATH` so the scripts can locate the `core` module:

```bash
cd ~/Merope/project_root
export PYTHONPATH=$PYTHONPATH:.
```

### 3. Workflow Order
The library follows a strictly decoupled workflow to ensure reproducibility:
1.  **Define**: Parameters (L, n3D, grain radius, seed) are set in the experiment script.
2.  **Build**: `core.geometry` generates analytical geometry and converts it to a voxelized grid.
3.  **Solve**: `core.solver` calls Amitex to compute the $3 \times 3$ thermal conductivity matrix.
4.  **Aggregate**: Results are parsed and saved into a `summary.txt` file in the result directory.

To run an experiment (e.g., Anisotropy):
```bash
python3 experiments/run_anisotropy.py
```

## Physics Note

### Homogenization Strategy
The suite calculates the **Effective Thermal Conductivity ($K_{eff}$)** of heterogeneous Representative Volume Elements (RVEs). 
*   **Phase 0 (Solid Matrix)**: Typically defined with $K = 1.0$.
*   **Phase 1 (Pores/Gas)**: Defined with a high-contrast ratio (e.g., $K = 10^{-3}$).

### Voxellation & Numerical Accuracy
The library uses the **VoxelRule.Average** combined with the **Voigt Homogenization Rule**. This ensures that voxels intersected by a grain boundary or pore edge are assigned a weighted average thermal conductivity. This approach significantly reduces the "staircase effect" in numerical results and allows for more accurate representations of thin features (like grain boundary cracks) even at moderate mesh resolutions.

### Microstructural Models
*   **Inter-granular Porosity**: Created by adding layers (thickness $\delta$) to Laguerre tessellation cells.
*   **Intra-granular Porosity**: Created via Boolean/RSA sphere distribution within the solid phase.
*   **Anisotropy**: Controlled via the Aspect Ratio parameter, elongating the grain geometry along specific axes while maintaining periodic boundary conditions.