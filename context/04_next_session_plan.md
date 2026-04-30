# 04 — Plan for next session

Ordered to start the long-running compute first (so it runs while you write/read), then writing tasks that don't need compute, then assembly once results are in.

## Phase 1 — kick off compute (do first, then leave running)

### 1.1 Rewrite `run_anisotropy.py` to match thesis Fig 16
**File:** `~/Merope/project_root/experiments/run_anisotropy.py`
**Reference:** `~/Merope/old_files/File originali/Test porosità/aniso_delta_calc.py` (Mattiuz original)
**Key parameters to match thesis:**
- 15-20 aspect ratios from γ = 0.1 to 1.0 (`np.linspace(0.1, 1.0, 18)`)
- Two porosity regimes: low p ≈ 0.05-0.10 and high p ≈ 0.20
- Volume-preserving anisotropy: `aspect_ratio = [1.0, γ, 1.0/γ]`
- Mixed Laguerre + spherical inclusions (NOT just delta-layer)
- Output CSV: AspectRatio, TargetP, RealP, Kxx, Kyy, Kzz, Kmean
- Compute `(K_mean - K_yy)/K_yy` post-hoc for the plot
- n3D = 100 (matches thesis), grain_radius = 3.0, delta = 1.0
- `K_THERMAL = [1.0, 1.0, 1e-3]` consistent with `run_keff_vs_delta.py`
**Expected time to write:** 30-45 min. **Run time on 32 cores:** ~15-20 min (AMITEX with `n_cpus=8` per case, run cases sequentially or 2-4 in parallel via `multiprocessing`).

### 1.2 Write `run_grain_size_distribution.py`
**Reference:** `~/Merope/old_files/File originali/Test porosità/vol_distribution_IGB_calc.py`
**Just two cases** at fixed (p ≈ 0.20, δ = 1.0):
- σ = 0.5 (tight unimodal volume distribution)
- σ = 3.0 (broader)
**Output:**
- `Results_GrainSizeDistribution/grain_volumes_sigma_{0p5,3p0}.csv`
- Histogram PNG of grain volumes (analogue of thesis Fig 17)
- K_eff log next to baseline Loeb (analogue of thesis Fig 18)
- Use `merope.algo_fit_volumes_3D` to seed the Laguerre tessellation with the prescribed volumes
**Expected time to write:** 30 min. **Run time on 32 cores:** ~5 min for both cases at n3D = 150.

### 1.3 (Optional but cheap) Extend δ-sweep below 0.15
Edit `experiments/run_keff_vs_delta.py`: prepend `0.05, 0.07, 0.09, 0.11, 0.13` to `DELTA_VALUES`. Run `bash run_keff_vs_delta.sh` (uses `--recover`, skips already-done cases).
**Run time on 32 cores:** ~5-10 min (15 new cases). Makes the per-p sigmoidal fit non-degenerate and removes the K_min lower-bound clamp now disclosed in §3.4.

⚠ **Resolution caveat** (per `01_theory.md` §4.1a, §02 §6a). With current `L = 10`, `n3D = 200` → $\Delta_\text{vox} = 0.05$, the GB layer at $\delta = 0.05$ is sampled by only **one voxel** (against the rule $\delta/\Delta_\text{vox} > 5$). At $\delta = 0.07$ it is 1.4 voxels. Below δ ≈ 0.25 the simulation relies entirely on the composite-voxel Voigt rule to capture the percolating crack — which is what `run_keff_vs_delta.py` already assumes for the lowest current point, $\delta = 0.15$ (3 voxels). So extending downward is consistent with the existing setup, but the fit-quality floor is the composite-voxel approximation, not the voxel grid. **Recommendation**: run the δ ∈ {0.05, 0.07, 0.09, 0.11, 0.13} extension at `n3D = 200` for consistency with the existing CSV; spot-check one point at `n3D = 400` to confirm the trend isn't a Voigt-rule artefact. If the spot check matches, proceed; if it diverges, the sigmoidal fit's low-δ asymptote $K_\text{min}(p)$ is upper-bounded by Voigt and shouldn't be reported as a converged number.

**At this point, ~4 hr of compute is queued. Walk away or move to Phase 2 in parallel.**

## Phase 2 — writing-only tasks (run while compute is going)

### 2.1 Run the joint sigmoidal fit on existing data
```bash
cd ~/Merope
python project_root/experiments/fit_correction_factor_joint.py \
  --csv Results_Keff_vs_Delta/keff_vs_delta.csv \
  --output-dir Results_Sigmoidal_Fit_Joint
```
Inspect `Sigmoidal_Fits_Joint.png`, `Parameters_vs_Porosity_Joint.png`, `K_eff_Contour_Joint.png`. If the fit looks clean, copy these into `paper/Images/Sigmoidal_Fit/` (replacing the degenerate ones). Update results.tex captions to match the new fit if numerical values changed.
**Time:** 10-15 min.

### 2.2 Reconcile pore-analysis numbers
Decide which set is canonical:
- **Paper text** (recommended): connected_79 = 21.8 % total / 13.8 % boundary / 8.0 % intra; distributed_77 = 23.0 % total / ~0 % boundary / 23.0 % intra
- **Current CSV**: connected_79 = 25.9 % total / 23.2 % inter / 2.8 % intra (inverted!); distributed_77 = 23.2 % / 5.4 % inter / 17.8 % intra

Inspect `core/pore_analysis.py` parameters (`circularity_thr`, `area_inter_thr_um2`, `min_area_inter_um2`) used in the live `run_pore_analysis.sh`. Find the threshold combination that reproduces the paper-text numbers (the sensitivity sweep `python project_root/core/pore_analysis.py <img> 0.195 --sensitivity` is the right tool). Then re-run `bash run_pore_analysis.sh`, copy the new `pore_analysis_results.csv` and the `*_analysis.png` files into `paper/Images/Pore_Analysis/`.
**Time:** 30-60 min including sensitivity sweep.

### 2.3 Recover or update Table 1 optimisation scores
The paper claims distributed avg = 0.901, interconnected avg = 0.676. The archived `summary.txt` files show 0.4895 / 0.3338. Two paths:
- **(a) Find the run that produced 0.901/0.676.** Search git history (`git log` in `~/Merope/`); look for older `Results_Optimization_*` snapshots. If the archived run is in a different branch or stash, recover it.
- **(b) Re-run optimisation** with current scoring; update Table 1 + the captioned best-slice numbers to match. ~4 hr per mode at `--n-calls 50 --n3d 120`. Can run during Phase 1 if there's CPU headroom.

If neither approach works in reasonable time, **drop the avg-score numbers** from Table 1 and report only the per-slice (KS, χ²) p-values — those at least match `summary.txt`.
**Time:** 30 min investigation + decision; **~1-1.5 hr** if re-running both modes on 32 cores (each Bayesian iteration drops from ~4 min to ~1 min when AMITEX uses 8-16 cores).

### 2.4 Port thesis prose for §3.4.2 (Anisotropy) and §3.4.3 (Grain size)
Source: `2025_07_Mattiuz_Thesis_01.pdf` pages 22-24. Translate to paper-quality English (the thesis prose is OK but bears editing). Apply UK English + plain hyphens per `~/research-manuscripts/writing_guidelines.md`.

Drop into `results.tex` at the commented `\subsubsection{Anisotropy and directional effects in interconnected porous microstructures}` and `\subsubsection{Effect of grain size distribution on thermal conductivity}` placeholders. Uncomment those blocks and add proper `\includegraphics` paths to the figures generated in Phase 1. Cross-reference the §6 design-guidelines bullets that mention "anisotropic IGB networks" and "grain size heterogeneity" so the new figures back them up.
**Time:** 1-1.5 hr.

## Phase 3 — assembly (after Phase 1 compute finishes)

### 3.1 Generate paper-quality figures from anisotropy + grain-size CSVs
Add a `make_anisotropy_plot.py` and `make_grain_size_plot.py` either in `experiments/` or as helpers in the existing scripts. Output:
- `paper/Images/Anisotropy/relative_diff_vs_aspect_ratio.png` (thesis Fig 16 analogue)
- `paper/Images/GrainSizeDistribution/volume_histograms.png` (Fig 17)
- `paper/Images/GrainSizeDistribution/keff_vs_porosity_grain_dist.png` (Fig 18)
**Time:** 30-45 min.

### 3.2 Verify figures match prose; commit
Read the uncommented results.tex sections side-by-side with the new figures. Audit symbol consistency (γ for aspect ratio, σ for the grain-volume weighting spread — make sure the legend/axis label/prose all use the same symbol). Apply UK English check on the new prose.
**Time:** 30 min.

### 3.3 Compile and inspect
```bash
cd ~/research-manuscripts/Luzzi_et_al___MEROPE__2026/
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```
Grep `main.log` for `Citation` / `Reference` / `LaTeX Warning` lines. Fix any undefined refs. Skim the rendered PDF for layout issues (figure placement, table overflows).
**Time:** 30 min unless something serious breaks.

## Phase 4 — polish (do last)

### 4.1 Update `project_root/README.md`
Out of date: lists 3 core files + 3 experiment scripts; reality is 8 + 13 (after the joint-fit script). Take 15 min to refresh the file tree and add a "How to reproduce paper figures" mini-guide.

### 4.2 Decide on `old_files/` cleanup
Per `context/03_paper_and_open_issues.md` §6:
- Move thesis PDF + presentation slides into `~/research-manuscripts/Luzzi_et_al___MEROPE__2026/_supporting/` (or a new `references/` dir).
- Tar `Test porosità/` + `Test statistici/` into `mattiuz_legacy.tar.gz` next to the thesis.
- Delete the rest.
- **Keep** `~/Merope/Optimization_3D_structure/exp_img/` (top-level, not `old_files/`) — `run_optimization.py` reads its SEM images.

### 4.3 Decide what to track from `Results_*/` in git

Current state (audited 2026-04-30): six `Results_*` folders exist at top-level, totalling ~355 MB. None are git-tracked, and only `DeltaScan_Results` (a legacy unrelated name) appears in `.gitignore`. So these folders are in limbo: not staged, not ignored.

| Folder | Size | What it contains | Recommendation |
|---|---|---|---|
| `Results_Keff_vs_Delta/` | 232 KB | `keff_vs_delta.csv` + per-case `P_*_Delta_*/thermalCoeff_amitex.txt` | Track everything (small) |
| `Results_Keff_vs_Porosity/` | 336 MB | `keff_vs_porosity.csv` + per-case `Phi_*_Nvox_*/{structure.vtk,Coeffs.txt,...}` | Track CSV + `Keff_Validation_Summary.png` only; gitignore per-case dirs (the `.vtk` files are the bulk) |
| `Results_Optimization_Distributed/` | 8.8 MB | `summary.txt`, `convergence.png`, `area_distribution.png`, `best_slice.png`, `best_geometry/structure.vtk`, `final_slices/`, `work/` | Track top-level files; gitignore `work/` and `final_slices/`; possibly track `best_geometry/structure.vtk` if reasonable size, otherwise skip |
| `Results_Optimization_Interconnected/` | 8.7 MB | Same structure | Same recommendation |
| `Results_Sigmoidal_Fit/` | 604 KB | `fitted_parameters.csv`, 3 PNGs | Track everything |
| `Results_Sigmoidal_Fit_Joint/` | 588 KB | New joint-fit outputs | Track everything |

**Action items:**
1. Decide the policy: track summaries/CSVs/headline figures, gitignore per-case `.vtk` and intermediate `work/` directories.
2. Add appropriate `.gitignore` patterns. Suggested:
   ```
   Results_Keff_vs_Porosity/Phi_*_Nvox_*/
   Results_Keff_vs_Delta/P_*_Delta_*/
   Results_Optimization_*/work/
   Results_Optimization_*/final_slices/
   Results_Optimization_*/best_geometry/structure.vtk
   ```
3. `git add` the surviving CSVs, summaries, and PNGs.
4. Consider archiving the bulky `.vtk` files separately (e.g., a Zenodo data deposit linked from the paper) if reproducibility requires them.

**Why this matters for the paper:** the per-case `.vtk` files are the AMITEX inputs/outputs — 95 % of the mass in the Results dirs is them. They're regenerable from the scripts in `project_root/experiments/`, so tracking them in git is wasteful. Tracking the CSVs + summary PNGs gives a co-author or reviewer enough to inspect numbers without bloating the repo.

## Phase 5 — submission (separate session)

Once Phases 1-4 are done, start fresh: a co-author review pass, journal-specific formatting (Elsevier `cas-sc.cls` if J. Nucl. Mat. has changed templates since the elsarticle preprint we're using), and the cover letter.

---

## What I will NOT touch in the next session unless explicitly asked

- The Mérope library itself (third-party).
- The optimisation scoring formula (any change invalidates Table 1).
- Bibliography entries beyond the one fix already applied (`Torquato2002`).
- The δ\* descriptor or sigmoidal model functional form.

## Quick references

- Open issues catalogue: `~/Merope/context/03_paper_and_open_issues.md`
- Theory: `~/Merope/context/01_theory.md` §6 (sigmoidal correction)
- Code idioms: `~/Merope/context/02_code_pipeline.md` §8 (Mérope test recipes)
- Writing rules: `~/research-manuscripts/writing_guidelines.md`
- Joint fit script: `~/Merope/project_root/experiments/fit_correction_factor_joint.py`

## Estimated timeline on the 32-core machine (revised 2026-04-30)

| Phase | Compute | Writing | Wallclock |
|---|---|---|---|
| 1 (kick off) | ~30-45 min | 1-1.5 hr | 2 hr |
| 2 (parallel) | 0-1.5 hr | 2-3 hr | 3 hr |
| 3 (assembly) | 0 | 1-1.5 hr | 1.5 hr |
| 4 (polish) | 0 | 30-45 min | 45 min |
| **Total** | **~30 min - 2 hr** | **5-7 hr** | **~7 hr active wallclock** |

The compute axis collapses on 32 cores: per-case AMITEX runs are dominated by FFT (near-linear scaling), and a typical voxel grid (n3D = 150-200) finishes in well under a minute when AMITEX gets 8-16 ranks. So Phase 1 is "kick off and wait briefly" rather than "kick off and walk away".
