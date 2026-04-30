# 02 — Code, pipeline, and how to run things

This file is a working manual: where the code lives, what each piece does, and the order in which it actually runs.

## 1. Layout of `~/Merope/`

```
~/Merope/
├── modules/, BUILD-DIR/, INSTALL-DIR/, doc/, Installation/, Licence*    # Mérope library itself — third-party, DO NOT EDIT
├── studies/                                                             # Mérope-internal benchmarks (also third-party)
├── Env_Merope.sh                                                        # Sources Mérope's compiler environment
├── Readme.md, Install.md                                                # Mérope's own docs
├── project_root/                                                        # ★ CANONICAL CODE FOR THIS PAPER ★
│   ├── core/                                                            # Library logic
│   │   ├── geometry.py                                                  # Mérope wrapper: structures, voxelization
│   │   ├── solver.py                                                    # AMITEX-FFTP wrapper
│   │   ├── statistics.py                                                # KS / χ² for slice-vs-image
│   │   ├── pore_analysis.py                                             # Image segmentation + classification
│   │   └── utils.py                                                     # ProjectManager (cd, log, send2trash)
│   ├── experiments/                                                     # Research scripts (one per study)
│   └── README.md                                                        # ⚠ Out of date — lists 3 core files / 3 experiments; reality is 8/12
├── *.sh                                                                 # Shell wrappers that call experiments/ scripts
├── Results_Keff_vs_Delta/, Results_Keff_vs_Porosity/,                   # ★ REAL RESULTS — DO NOT DELETE ★
│   Results_Optimization_*/, Results_Sigmoidal_Fit/                      #   (paper figures sourced from these)
├── *.png, *.csv (top-level)                                             # Mirrors of paper Images/ — also results
├── Optimization_3D_structure/                                           # Pre-project_root prototype; superseded but kept
├── PORE_ANALYSIS_QUICKSTART.md, README_PORE_ANALYSIS.md,                # Up-to-date how-tos for pore_analysis pipeline
│   WORKFLOW_COMPLETE.md
└── old_files/File originali/                                            # Mattiuz-era scripts + thesis. Mark for deletion AFTER mining.
```

**Top-level rule**: `project_root/` is the source of truth. Anything outside it is either the upstream Mérope library, results, or legacy. Don't restructure without a reason.

⚠ Two "Optimization_3D_structure" directories exist (`~/Merope/Optimization_3D_structure/` and `~/Merope/old_files/File originali/Optimization_3D_structure/`). Both contain the same 21 MB zip and old `main.py` / `MOX_structure_generator.py` / `statistical_test_func.py`. The first one's `exp_img/` folder is what `run_optimization.py` and the shell wrappers actually read; **don't delete it** until image paths in scripts are updated.

## 2. `core/` — library

### 2.1 `geometry.py` — `MicrostructureBuilder`

Wraps Mérope and `sac_de_billes`. Constructor takes RVE size `L = [Lx, Ly, Lz]`, voxel count per side `n3D`, and a seed.

**Methods (current convention: phase 1 = matrix, phase 2 = pores):**
- `generate_polycrystal(grain_radius, delta, aspect_ratio)` — Laguerre tessellation, optional grain-boundary layer of thickness `delta`. Aspect ratio for anisotropy.
- `generate_spheres(...)` — Boolean / RSA sphere distribution.
- `generate_mixed_structure(...)` — polycrystal + intra-grain spheres.
- `generate_interconnected_structure(...)` — **legacy convention** (phase 0 = matrix, phase 2 = pores, phase 3 = GB layer); reproduces Mattiuz's original IGB pipeline.

**Phase-convention warning:** `run_keff_vs_delta.py` builds structures by hand (not via the high-level helpers) and uses **0 = grains, 2 = pores, 3 = GB layer (temporary)**. Always check `K_THERMAL` ordering matches the phase IDs in the script you're touching.

Voxelization defaults: `VoxelRule.Average` + `HomogenizationRule.Voigt`. Composite voxels record sub-voxel phase fractions — essential for thin GB cracks at moderate resolution.

### 2.2 `solver.py` — `ThermalSolver`

Thin wrapper around `interface_amitex_fftp.amitex_wrapper.computeThermalCoeff`. Takes a `.vtk` file (already voxelized) and an output path; returns a dict `{Kxx, Kyy, Kzz, Kmean}`. `n_cpus` controls MPI parallelism. Failures return zeros instead of raising — keep this in mind when sweeping parameters.

### 2.3 `statistics.py`

KS and $\chi^2$ tests for slice‑vs‑image comparison. `evaluate_slices` extracts N 2D slices, segments each, computes p-values per slice, returns best/worst/avg. Pore exclusion at 30 px to suppress segmentation noise. `plot_area_distribution` produces the histograms used in `area_distribution.png`.

### 2.4 `pore_analysis.py`

Standalone image-analysis pipeline (also runs as a CLI). Otsu or Sauvola thresholding, optional watershed for touching pores, classifies each pore as **intra** (round, isolated) or **inter** (elongated, grain-boundary) based on circularity and area. Outputs per-pore CSV plus a summary CSV. Stereological correction factor 0.85.

CLI: `python core/pore_analysis.py image.png 0.195 [circularity_thr] [--plot --export-csv --adaptive --sensitivity]`

The `--sensitivity` mode sweeps circularity thresholds — useful when calibrating against a new SEM image.

### 2.5 `utils.py` — `ProjectManager`

Three small helpers: `cleanup_folder` (uses `send2trash`, recoverable), `cd` context manager, `log_results` for tab-separated logging. Keeps experiments hermetic.

## 3. `experiments/` — research scripts

| Script | Study | Inputs | Outputs |
|---|---|---|---|
| `run_keff_vs_porosity.py` | Closed-porosity sweep, $p \in [0.01, 0.30]$. Calibrates Loeb α. | None (constants in file) | `Results_Keff_vs_Porosity/keff_vs_porosity.csv`, `Keff_Validation_Summary.png`, per-case `Phi_*_Nvox_*/` dirs |
| `run_keff_vs_delta.py` | δ-sweep at $p = 0.1, 0.2, 0.3$ (interconnected). Generates the data behind the sigmoidal model. | `--recover` to skip already-computed cases | `Results_Keff_vs_Delta/keff_vs_delta.csv`, `Slide_Keff_vs_Delta.png`, per-case `P_*_Delta_*/` dirs |
| `fit_correction_factor.py` | Sigmoidal fit on the δ-sweep CSV. Linear regressions of K_min, K_max, b, δ_c on p. | `--csv ... --output-dir ...` | `Results_Sigmoidal_Fit/fitted_parameters.csv`, `Sigmoidal_Fits.png`, `Parameters_vs_Porosity.png`, `K_eff_Contour.png` |
| `run_optimization.py` | Bayesian opt to match SEM image. | `--mode {distributed,interconnected,test_*}`, `--exp-image PATH`, `--n-calls`, `--n3d`, `--run-amitex`, `--seed`, `--n-slices` | `Results_Optimization_<mode>/summary.txt`, `area_distribution.png`, `convergence.png`, `best_slice.png`, `best_geometry/structure.vtk`, `final_slices/` |
| `predict_keff_from_optimization.py` | Apply sigmoidal correction to the optimized δ. | Path to `Results_Optimization_*` dir | `keff_prediction.txt` in same dir |
| `compare_optimization_results.py` | Side-by-side bar chart distributed vs interconnected. | None | `comparison_distributed_vs_interconnected.png`, `keff_vs_porosity_comparison.png` (top-level) |
| `run_anisotropy.py` | Directional K vs grain aspect ratio (Mattiuz §3.4.2). | Constants in file | Directional results — **figure not in paper** (commented out) |
| `run_distributed_porosity.py` | Single-config closed-porosity run. | Constants in file | Per-case dirs |
| `run_interconnected_porosity.py` | Single-config interconnected run. | Constants in file | Per-case dirs |
| `run_mixed_porosity.py` | Single-config inter+intra. | Constants in file | Per-case dirs |
| `run_delta_iteration.py` | Older variant of δ-sweep. | Constants in file | Per-case dirs |
| `run_plots.py` | Plot helpers. | — | Plots only |

### 3.1 Phase IDs cheat sheet

When touching a new experiment, find the `K_THERMAL = [...]` line and confirm:

- `run_keff_vs_delta.py`: `K_THERMAL = [1.0, 1.0, 1e-3]`, phases 0 & 1 = solid, phase 2 = pore. Phase 3 is GB layer used only during construction and remapped.
- `run_keff_vs_porosity.py`: typically `[1.0, 1e-3]` — phase 0 = matrix, phase 1 = pore.
- `geometry.py` high-level methods: phase 1 = matrix, phase 2 = pore.

This isn't a footgun in practice (each script is self-contained) but it bites when copy-pasting between scripts.

#### Legacy phase-mapping bug (`old_files/` scripts)

Several scripts in `old_files/File originali/Test porosità/` use the convention `incl_phase = 2`, `delta_phase = 3`, `grains_phase = 0`, then build the structure with a remapping `dictionnaire = {incl_phase: grains_phase, delta_phase: grains_phase}`. If this dictionary is missing or its keys are crossed, the *pores* end up remapped onto the *grain* phase — silently solidifying the network. The simulated $K_\text{eff}$ then matches the Maxwell/Loeb distributed baseline instead of the percolation-crashed regime. **Symptom**: an interconnected δ-sweep that does not show the sigmoidal drop at low δ. **Cause**: phase-mapping bug, not physics. The OOP refactor in `core/geometry.py` fixed this by exposing `generate_interconnected_structure` with an explicit, tested phase contract (legacy convention `0 = matrix`, `2 = pores`). When resurrecting any `old_files/` script, dump the segmented `.vtk` in ParaView before trusting the K_eff number.

## 4. Top-level shell wrappers

Each runs from `~/Merope/` and assumes `project_root/` is on `PYTHONPATH`:

```bash
cd ~/Merope
export PYTHONPATH=$PYTHONPATH:./project_root
source Env_Merope.sh                      # Mérope environment (or activate conda env)
```

| Wrapper | What it does |
|---|---|
| `run_pore_analysis.sh` | Runs `pore_analysis.py` on the SEM images under `Optimization_3D_structure/exp_img/`, writes `pore_analysis_results.csv` at top level |
| `run_keff_vs_porosity.sh` | `python project_root/experiments/run_keff_vs_porosity.py` |
| `run_keff_vs_delta.sh` | `python project_root/experiments/run_keff_vs_delta.py --recover` |
| `run_optimization_distributed.sh` | Reads `pore_analysis_results.csv`, calls `run_optimization.py --mode distributed ...` |
| `run_optimization_interconnected.sh` | Same for interconnected, with $\delta$ + intra params |
| `fit_correction_factor.sh` | Runs `fit_correction_factor.py` then `compare_optimization_results.py` |

The wrappers encode the canonical CLI args for the paper. If you re-run anything, prefer the wrapper over invoking Python directly.

## 5. End-to-end pipeline

```
[ SEM image ] ──► pore_analysis.py ──► p_total, p_intra, p_inter ─┐
                                                                  │
                                                                  ▼
                                                         run_optimization.py
                                                         (Bayesian, KS+χ²)
                                                                  │
                                                                  ▼
                                              best params + structure.vtk
                                                                  │
                              ┌───────────────────────────────────┤
                              ▼                                   ▼
                      run_amitex (in solver)          predict_keff_from_optimization.py
                       full FFT solve                  K_Loeb × K_δ(p,δ) × intra correction
                              │                                   │
                              └────────────► compare_optimization_results.py ◄────────────┘
                                                       │
                                                       ▼
                                  comparison_distributed_vs_interconnected.png
```

In parallel, the **morphology→property law itself** is built by:
```
run_keff_vs_porosity.py ──► closed-porosity α calibration (α = 1.37)
run_keff_vs_delta.py     ──► δ-sweep CSV
fit_correction_factor.py ──► sigmoidal K_δ(p, δ*) parameters
```

## 6. Inputs the framework treats as fixed

These are encoded as constants and rarely changed, but worth knowing:

- **RVE size**: `L = [10, 10, 10]` physical units; voxel count `n3D = 120`-`200` (200 in `run_keff_vs_delta.py`, 134/154/174 in `run_keff_vs_porosity.py`).
- **Grain size**: `LAG_R = 1.0` (so $\delta$ values numerically equal $\delta^*$).
- **Random seed**: 0 or 42 — deterministic by default.
- **Conductivities**: $\kappa_m = 1.0$, $\kappa_g = 10^{-3}$.
- **Optimisation**: 20-50 calls typical, GP surrogate + Expected Improvement.
- **Stereological correction**: 0.85 in pore_analysis.

If you change any of these, the calibrated $\alpha = 1.37$ and the sigmoidal parameters may need to be re-fit.

### 6a. Voxelisation quality floor

Two rules of thumb (see `01_theory.md` §4.1a) bound any new sweep:

- $L_\text{RVE} / R_\text{pore} > 10$ (statistical representativity).
- $R_\text{pore} / \Delta_\text{vox} > 5$ (geometric resolution; same applies to $\delta$).

The second is the binding constraint when extending the δ-sweep below 0.15: $\Delta_\text{vox} = L / n_\text{3D}$, so for $L=10$ and $\delta = 0.05$, $n_\text{3D}$ must be at least $10 \cdot 5 / 0.05 = 1000$ to resolve the GB layer with five voxels — well above the current `n3D = 200`. In practice this means the δ-sweep extension either accepts under-resolved layers and trusts the composite-voxel Voigt rule to cover the gap, or steps `n_\text{3D}` up alongside δ. Worth a sensitivity check before locking in the new fit.

## 7. Mérope and AMITEX-FFTP — the upstream tools

**Mérope** (Josien 2024, CEA/PLEIADES): C++ core with Python bindings. Builds Boolean / RSA sphere structures, Laguerre tessellations, Gaussian random fields; voxelizes with composite-voxel mixing. Output: `.vtk` files. Imported as `import merope`.

**`sac_de_billes`** (CEA): companion library for sphere throwing (RSA). Imported as `import sac_de_billes`. Used to seed Laguerre tessellations.

**AMITEX-FFTP** (Brisard, Dormieux, Willot 2015, CEA Saclay; <https://github.com/amitex/amitex-fftp>): FFT-based homogenization solver. MPI-parallel. Reads the segmented VTK from Mérope, returns the homogenized $3\times 3$ conductivity tensor. Imported as `interface_amitex_fftp.amitex_wrapper`.

**ParaView**: visualization only — not used in the automated pipeline.

## 8. Mérope idioms — quick reference (from `~/Merope/tests/microstructures/`)

These are the canonical patterns used by Marc Josien (Mérope's author). The wrappers in `core/geometry.py` reduce them to a method call, but it helps to know what they unfold to.

### Sphere distribution
```python
import sac_de_billes, merope
spheres = sac_de_billes.throwSpheres_3D(
    sac_de_billes.TypeAlgo.RSA,         # or .Boolean for overlap-allowed
    sac_de_billes.NameShape.Tore,       # periodic cube
    L, seed,
    [[radius, volFrac], ...],            # desired (R, φ) pairs
    [phase_id, ...], minDist=0.01,
)
```

### Polycrystal (Laguerre tessellation)
```python
polyCrystal = merope.LaguerreTess_3D(L, spheres)
multi = merope.MultiInclusions_3D()
multi.setInclusions(polyCrystal)
```

### Layered structure (the δ trick)
```python
multi.addLayer(identifiers, newPhases, widths)   # a list of layers added per inclusion
```
This is exactly what makes the δ‑band of grain-boundary porosity. `geometry.MicrostructureBuilder.generate_polycrystal(..., delta=...)` wraps this.

### Sphere inclusions with a histogram
```python
sph = merope.SphereInclusions_3D()
sph.setLength(L)
sph.fromHisto(seed, sac_de_billes.TypeAlgo.RSA, minDist,
              [[r1, vf1], [r2, vf2]],   # bimodal example
              [phase1, phase2])
```

### Voxelization with composite voxels (the standard recipe)
```python
gridParameters = merope.vox.create_grid_parameters_N_L_3D(nbVox, L)
structure = merope.Structure_3D(multi)
grid = merope.vox.GridRepresentation_3D(structure, gridParameters,
                                         merope.vox.VoxelRule.Average)
grid.apply_homogRule(merope.HomogenizationRule.Voigt, pure_coeffs)
my_printer = merope.vox.vtk_printer_3D()
my_printer.printVTK_segmented(grid, "Composite.vtk", "Coeffs.txt")
```

### AMITEX call (separate `Coeffs.txt` is implicit)
```python
import interface_amitex_fftp.amitex_wrapper as amitex
import interface_amitex_fftp.post_processing as amitex_out
amitex.computeThermalCoeff("Composite.vtk", n_cpus)
amitex_out.printThermalCoeff(".")
```

### Subdirectories worth knowing in `tests/microstructures/`
- `polyCrystal/`, `polyCrystal_2D/`, `polyCrystal_filamentaire/` — tessellation reference
- `multiLayer/` — adds layers around inclusions (mechanism behind δ)
- `inclusions/`, `coated_inclusions/`, `intersectingSpheres/`, `largeSphere/` — sphere variants
- `buildVoxellation/` — `Thermal_amitex.py` / `Thermal_tmfft.py` are the reference solver couplings
- `gaussianCrystal/`, `parallel_gaussian/`, `texture/` — Gaussian random fields
- `optimize_Laguerre_tess/`, `optimize_Laguerre_2D/` — volume-balanced tessellation (relevant if grain-size distribution becomes an optimization variable)
- `non-regression_tests.py`, `non-regression_lambda.py` — run these to verify a fresh build of Mérope before sweeps

## 9. Adding a new study — minimum recipe

1. Drop a script in `experiments/` named `run_<thing>.py`. Keep the `_PROJECT_ROOT` sys.path bootstrap at the top.
2. Use `MicrostructureBuilder` for structure, `ThermalSolver` for AMITEX, `ProjectManager.cd()` per-case directory, `ProjectManager.log_results()` to append to a CSV.
3. Output: a CSV plus per-case dirs under a `Results_<Thing>/` directory at the top of `~/Merope/`.
4. Optionally add a thin wrapper at top-level (`run_<thing>.sh`).
5. Document the script with a docstring and update `project_root/README.md`.
