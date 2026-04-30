# 04 — Plan for next session

Updated 2026-04-30 after a long working session. Previous "kick off compute first" plan has been worked through; this file now reflects the new state and the remaining punch list.

## Snapshot of today (2026-04-30)

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

## Phase 1 — finish the compute initiated today

### 1.1 Wait for the anisotropy sweep to finish

Inside `tmux` ideally — terminal close kills the run. Status checks:
```bash
wc -l ~/Merope/Results_Anisotropy/anisotropy.csv
tail -3 ~/Merope/Results_Anisotropy/anisotropy.csv
```
Expected: 41 lines (header + 40 cases) when complete. The script writes the CSV after each case, so partial progress is preserved.

### 1.2 Run the grain-size-distribution sweep

Only after anisotropy completes (otherwise both compete for cores). Two cases at $n_\text{3D}=150$, ~5–10 min each:
```bash
cd ~/Merope
source Env_Merope.sh
export PYTHONPATH=$PYTHONPATH:./project_root
python project_root/experiments/run_grain_size_distribution.py
```
Outputs land in `Results_GrainSizeDistribution/`. `--recover` works the same as the anisotropy script.

### 1.3 Refresh the optimisation predictions

```bash
python project_root/experiments/predict_keff_from_optimization.py Results_Optimization_Interconnected
python project_root/experiments/predict_keff_from_optimization.py Results_Optimization_Distributed
```
This regenerates `keff_prediction.txt` in each directory using the joint-fit coefficients. **Verify the headline "40 % reduction" claim still holds** at the optimised δ — the new joint fit puts $K_\delta$ near its upper plateau already at δ*=0.5, so the magnitude of the reduction may have shifted. See `03 §5.6`.

## Phase 2 — writing & paper assembly

If neither approach works in reasonable time, **drop the avg-score numbers** from Table 1 and report only the per-slice (KS, χ²) p-values — those at least match `summary.txt`.
**Time:** 30 min investigation + decision; **~1-1.5 hr** if re-running both modes on 32 cores (each Bayesian iteration drops from ~4 min to ~1 min when AMITEX uses 8-16 cores).

Source: `~/Merope/old_files/2025_07_Mattiuz_Thesis_01.pdf` pages 22–24. Translate to UK English + plain hyphens (`writing_guidelines.md`). Drop into `results.tex` at the existing commented `\subsubsection{...}` placeholders. Add the new figure includes:
- `Images/Anisotropy/anisotropy.png`
- `Images/GrainSizeDistribution/volume_histograms.png`
- `Images/GrainSizeDistribution/keff_comparison.png`

Also add **one sentence in `discussion.tex`** on the K_min(p) extrapolation — at p=0.1 and p=0.3 the lower asymptote of the sigmoid is set by the linear-in-p regression, not by direct measurement, because no data was sampled below the plateau at those porosities.

### 2.2 Pore-analysis number reconciliation (`03 §5.2`)

Still open. Likely root cause is a CLI signature change in `core/pore_analysis.py`: paper-text numbers were produced with `... 0.195 80` (third arg seemingly an area filter), current CLI treats the third positional arg as a circularity threshold in [0, 1]. Diff against git history to find when the meaning changed; either restore old behaviour or sweep `circularity_thr × min_area_um2` until the analysis reproduces the paper text (connected_79: 21.8 % / 13.8 % / 8.0 %; distributed_77: 23.0 % / ~0 % / 23.0 %).

### 2.3 Recover or update Table 1 optimisation scores (`03 §5.3`)

Paper Table 1 says distributed avg = 0.901, interconnected avg = 0.676. Archived `summary.txt` files show 0.4895 / 0.3338. The paper's claimed numbers do not appear in any artefact on disk. Either find the run that produced them (search git history, look for older snapshots) or re-run optimisation with current scoring and update Table 1.

### 2.4 Compile and inspect the paper

```bash
cd ~/research-manuscripts/Luzzi_et_al___MEROPE__2026/
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```
Grep `main.log` for `Citation` / `Reference` / `LaTeX Warning`. Skim the rendered PDF.

## Phase 3 — polish and tidy

### 3.1 `core/geometry.py` API gap

`generate_polycrystal()` accepts `aspect_ratio`, but `generate_mixed_structure()` does not. That's why `run_anisotropy.py` had to inline-build today. Add an `aspect_ratio=(1,1,1)` kwarg to `generate_mixed_structure` for symmetry. Also consider deduplicating `generate_delta_structure` and the inline build in `run_keff_vs_delta.py:worker()` (same recipe).

### 3.2 Refresh `project_root/README.md`

Lists 3 core files + 3 experiment scripts; reality is 5 + 12. Add a "How to reproduce paper figures" mini-guide with the canonical command sequence.

### 3.3 Decide on the three `run_*_porosity.py` validation scripts

`run_distributed_porosity.py` produced the data behind the kept `Results_Distributed_Validation/`. `run_interconnected_porosity.py` and `run_mixed_porosity.py` last produced K=0 (those folders are now in `_to_delete/`). Either debug them (~30 min each) or move both to `_to_delete/`.

### 3.4 Clear `_to_delete/`

```bash
rm -rf ~/Merope/_to_delete
```
Once you're satisfied nothing in there is needed. ~49 MB total.

### 3.5 Decide what to track from `Results_*/` in git

Per the policy in the previous version of this file (now relevant since all `Results_*/` are at top-level): track summaries / CSVs / headline figures, gitignore per-case `.vtk` and intermediate `work/` directories. Suggested `.gitignore` patterns:
```
Results_Keff_vs_Porosity/Phi_*_Nvox_*/
Results_Keff_vs_Delta/P_*_Delta_*/
Results_Anisotropy/AR_*_Phi_*/
Results_GrainSizeDistribution/sigma_*/structure.vtk
Results_GrainSizeDistribution/sigma_*/Coeffs.txt
Results_Optimization_*/work/
Results_Optimization_*/final_slices/
Results_Optimization_*/best_geometry/structure.vtk
```

## Phase 4 — submission (separate session)

Co-author review pass, journal-specific formatting check (Elsevier `cas-sc.cls` if NED template requires), cover letter.

---

## Quick references

- Open issues catalogue: `~/Merope/context/03_paper_and_open_issues.md`
- Theory: `~/Merope/context/01_theory.md` §6 (sigmoidal correction)
- Code idioms: `~/Merope/context/02_code_pipeline.md`
- Writing rules: `~/research-manuscripts/writing_guidelines.md`
- Joint fit script: `~/Merope/project_root/experiments/fit_correction_factor_joint.py`
- Anisotropy script: `~/Merope/project_root/experiments/run_anisotropy.py`
- Grain-size script: `~/Merope/project_root/experiments/run_grain_size_distribution.py`

## What I will NOT touch in the next session unless explicitly asked

- The Mérope library itself (third-party).
- The optimisation scoring formula (any change invalidates Table 1).
- The δ\* descriptor or sigmoidal model functional form.
- The joint-fit coefficients in `linear_coeffs.csv` (they are the paper-canonical values).


The compute axis collapses on 32 cores: per-case AMITEX runs are dominated by FFT (near-linear scaling), and a typical voxel grid (n3D = 150-200) finishes in well under a minute when AMITEX gets 8-16 ranks. So Phase 1 is "kick off and wait briefly" rather than "kick off and walk away".
