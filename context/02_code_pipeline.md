# 02 — Code, pipeline, and how to run things

This file is a working manual: where the code lives, what each piece does, and the order in which it actually runs.

## 1. Layout of `~/Merope/` (post-2026-04-30 cleanup)

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
│   ├── experiments/                                                     # 14 research scripts (one per study)
│   └── README.md                                                        # ⚠ Out of date — lists 3 core files / 3 experiments; reality is 5/14
├── *.sh                                                                 # Shell wrappers that call experiments/ scripts
├── Results_Anisotropy/                                                  # ★ ALL CANONICAL RESULTS LIVE AT TOP-LEVEL ★
│   Results_Distributed_Validation/                                      #   (post-2026-04-30 consolidation:
│   Results_Keff_vs_Delta/                                               #    project_root/Results_*/ no longer exists)
│   Results_Keff_vs_Porosity/
│   Results_Optimization_Distributed/
│   Results_Optimization_Interconnected/
│   Results_Sigmoidal_Fit/                                               #   (renamed from Results_Sigmoidal_Fit_Joint)
├── _to_delete/                                                          # 10 stale folders + 3 redundant scripts
│   └── experiments/                                                     #   staged 2026-04-30; rm -rf when satisfied
├── *.png, *.csv (top-level)                                             # Mirrors of paper Images/ — also results
├── Optimization_3D_structure/                                           # Pre-project_root prototype; SEM images live in exp_img/
├── PORE_ANALYSIS_QUICKSTART.md, README_PORE_ANALYSIS.md,                # Up-to-date how-tos for pore_analysis pipeline
│   WORKFLOW_COMPLETE.md
└── old_files/                                                           # Mattiuz-era scripts + thesis (recovered 2026-04-30
                                                                         # WITHOUT the "File originali/" subdir on this machine)
```

**Top-level rule**: `project_root/` is the source of truth for code; top-level `Results_*/` is the source of truth for outputs. Don't restructure without a reason.

⚠ `~/Merope/Optimization_3D_structure/exp_img/` (top-level, **NOT** in `old_files/`) holds the SEM images that `run_optimization.py` and the shell wrappers read. Don't delete unless image paths are updated.

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
| `run_keff_vs_delta.py` | δ-sweep at $p = 0.1, 0.2, 0.3$ (interconnected). Generates the data behind the sigmoidal model. Currently 47 points. | `--recover` to skip already-computed cases | `Results_Keff_vs_Delta/keff_vs_delta.csv`, `Slide_Keff_vs_Delta.png`, per-case `P_*_Delta_*/` dirs |
| `run_anisotropy.py` (rewritten 2026-04-30) | Thesis Fig 16: directional K vs grain aspect ratio. 20 γ × 2 φ = 40 cases. Faithful to `old_files/Test porosità/aniso_delta_calc.py`. | `--recover` (filters K=0 rows), `--no-solver`, `--plot-only` | `Results_Anisotropy/anisotropy.csv`, `anisotropy.png`, per-case `AR_<γ>_Phi_<φ>/` dirs |
| `run_grain_size_distribution.py` (new 2026-04-30) | Thesis Figs 17-18: K_eff vs σ of Gaussian-weighted grain volumes. σ ∈ {0.5, 3.0} at p≈0.20, δ=1.0, n3D=150. Faithful to `vol_distribution_IGB_calc.py`. | `--recover`, `--no-solver`, `--plot-only` | `Results_GrainSizeDistribution/summary.csv`, `volume_histograms.png`, `keff_comparison.png`, per-case `sigma_*/grain_volumes.csv + structure.vtk` |
| `fit_correction_factor_joint.py` (canonical) | Joint linear-in-p sigmoidal fit on the full δ-sweep CSV. 8 free parameters across all p. | `--csv ... --output-dir ...` (default `Results_Sigmoidal_Fit`) | `Results_Sigmoidal_Fit/fitted_parameters.csv`, `linear_coeffs.csv`, `Sigmoidal_Fits.png`, `Parameters_vs_Porosity.png`, `K_eff_Contour.png` |
| `fit_correction_factor.py` (deprecated 2026-04-30) | Per-p sigmoidal fit + linear regression of params on p. Goes degenerate when low-δ plateau is unsampled. K_min lower bound raised to 0.05 to prevent fully-zero fit. Kept for benchmarking against the joint fit only. | `--csv ... --output-dir ...` | Same filenames as joint script (in a different dir if `--output-dir` set) |
| `run_optimization.py` (extended 2026-05-04) | Bayesian opt to match a 2D microstructure target image. **Default reference set is now `synthetic`** (reproducible PNGs from `make_synthetic_targets.py`); `--exp-image-set consortium` falls back to the private ESFR-SIMPLE images for comparison runs. `--exp-image PATH` still overrides both. | `--mode {distributed,interconnected,test_*}`, `--exp-image-set {synthetic,consortium}`, `--exp-image PATH`, `--n-calls`, `--n3d`, `--run-amitex`, `--seed`, `--n-slices` | `Results_Optimization_<mode>/summary.txt`, `area_distribution.png`, `convergence.png`, `best_slice.png`, `best_geometry/structure.vtk`, `final_slices/` |
| `predict_keff_from_optimization.py` (refactored 2026-04-30, reframed 2026-05-04) | Apply sigmoidal correction to optimised δ. Loads coefficients from `Results_Sigmoidal_Fit/linear_coeffs.csv` at runtime. Now prints two K_eff values: AMITEX-comparable (`K_loeb · K_δ`, matches the optimisation RVE) and composite (extra `(1 − 1.37·p_intra)` factor, NOT comparable to AMITEX). Also reports the bare morphology penalty `1 − K_δ` so the asymptote-vs-operating-point distinction is explicit. | Path to `Results_Optimization_*` dir; coeffs path overridable | `keff_prediction.txt` in same dir |
| `make_synthetic_targets.py` (new 2026-05-04, refined 2026-05-05) | Generates reproducible 2D microstructure PNGs as drop-in replacements for the private ESFR-SIMPLE consortium reference images. Distributed = polydisperse RSA spheres (4 radius bins). Interconnected matched-form = Laguerre + δ=1 GB band + monodisperse boundary pores, no intra (matches optimiser's vocabulary, used for recovery test). Interconnected visual-rich = polydisperse boundary + intra (paper figure only, not optimiser target). Realised porosities reproduce the paper's pore-analysis values by construction. | None (constants in file) | `Optimization_3D_structure/exp_img_synthetic/synthetic_distributed.png`, `synthetic_interconnected.png` (matched-form), `synthetic_interconnected_visual.png` (visual-rich), `ground_truth.json` |
| `run_thin_delta_mixed.py` (new 2026-05-06) | Builds the **thin-$\delta^*$ mixed interconnected RVE** directly (boundary pores clipped to GB band of thickness $\delta = 0.3$, $\delta^* = 0.10$ + independent intra-granular RSA pores) and runs AMITEX on it. Used to produce the headline morphology-penalty figure: $K_{\rm eff} = 0.600$ W/m·K vs Loeb 0.708 (15.2 % drop) at $p_{\rm total} = 0.213$. AMITEX needs `--n-cpus ≤ 18` on this hardware. | `--no-solver`, `--n-cpus`, `--inter-phi`, `--intra-phi`, `--delta` | `Results_ThinDelta_Mixed/structure.vtk`, `Coeffs.txt`, `slice.png`, `summary.txt` |
| `make_paper_comparison_figures.py` (new 2026-05-05) | Generates the three paper-side comparison figures: (1) `comparison_distributed_vs_interconnected.png` (porosity composition + AMITEX vs Loeb bars with morphology-penalty annotation); (2) `keff_vs_porosity_comparison.png` (scatter on Loeb baseline with "X % drop below Loeb" arrow); (3) `recovery_test_interconnected.png` (synthetic target ↔ best slice side-by-side). Reads from `Results_Optimization_Distributed/`, `Results_ThinDelta_Mixed/` (for the headline interconnected point), and `Results_Optimization_Interconnected/` (for the matched-form recovery test). | None | Three PNGs in `~/research-manuscripts/Luzzi_et_al___MEROPE__2026/Images/Comparison/` |
| `compare_optimization_results.py` (legacy) | Older side-by-side bar chart. **Superseded by `make_paper_comparison_figures.py` 2026-05-05**; kept for now until verified unused. | None | `comparison_distributed_vs_interconnected.png`, `keff_vs_porosity_comparison.png` (top-level) |
| `run_distributed_porosity.py` | Single-config closed-porosity validation runs (R_pore × φ sweep behind the kept `Results_Distributed_Validation/`). | Constants in file | Per-case dirs |
| `run_interconnected_porosity.py` | Single-config interconnected run. **Last archived run produced K=0 (now in `_to_delete/`); status uncertain.** | Constants in file | Per-case dirs |
| `run_mixed_porosity.py` | Single-config inter+intra. **Same K=0 issue as `run_interconnected_porosity.py`; status uncertain.** | Constants in file | Per-case dirs |

**Removed 2026-04-30** (moved to `_to_delete/experiments/`): `run_delta_iteration.py` (older δ-sweep variant, K_THERMAL bug), `run_keff_vs_delta_p03_extension.py` (one-shot helper for the abandoned low-δ extension), `run_plots.py` (older Mattiuz-era thesis-graph script with K_THERMAL bug, never referenced).

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
