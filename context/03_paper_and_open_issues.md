# 03 — Paper status and open issues

This file is the working punch-list for finishing the paper. It catalogs (a) section-by-section state, (b) numerical discrepancies that need reconciling before submission, (c) what's worth mining from `old_files/`, and (d) decisions for Giovanni.

## 1. Manuscript inventory

`~/research-manuscripts/Luzzi_et_al___MEROPE__2026/`

| File | Status |
|---|---|
| `main.tex` | OK — frontmatter, abstract, highlights, keywords, includes 5 sections |
| `introduction.tex` | OK — narrative complete |
| `methods.tex` | OK — physics + classical models + tools |
| `results.tex` | **5 subsections commented out** — see §3 below |
| `discussion.tex` | OK — physical interpretation, limitations, comparison with literature |
| `conclusions.tex` | OK |
| `bibliography.bib` | 912 entries; **case-sensitivity issues** — see §4 |
| `Images/` | 12 PNGs across 6 subfolders, all currently-cited figures present |
| `copy_figures.sh` | Helper script to copy figures from `~/Merope/Results_*/` into `Images/` |
| `main.pdf` | Last build artefact |

### 1.1 Figures currently in `Images/`

```
Comparison/        comparison_distributed_vs_interconnected.png, keff_vs_porosity_comparison.png
Keff_vs_Delta/     Slide_Keff_vs_Delta.png, keff_vs_delta.csv
Keff_vs_Porosity/  Keff_Validation_Summary.png, keff_vs_porosity.csv
Optimization_Distributed/    area_distribution.png, best_slice.png, convergence.png, summary.txt, keff_prediction.txt
Optimization_Interconnected/ area_distribution.png, best_slice.png, convergence.png, summary.txt, keff_prediction.txt
Pore_Analysis/     distributed_77_original.png, distributed_77_analysis.png, connected_79_original.png, connected_79_analysis.png, pore_analysis_results.csv
Sigmoidal_Fit/     Sigmoidal_Fits.png, Parameters_vs_Porosity.png, K_eff_Contour.png
```

## 2. Headline numbers in the paper

For sanity-checking against the codebase / data:

- Loeb recalibrated: **α = 1.37** (Loeb_classical: 2.5; matches Morimoto 2008)
- Distributed/Loeb error: **< 2 %** over $p \in [0, 0.30]$
- 40 % $K_{\text{eff}}$ reduction interconnected vs distributed at $p \approx 22\%$ ($K_{\text{eff}} = 0.606$ vs $1.00$ normalized)
- Bayesian optimization scores claimed in Table 1: distributed avg 0.901, interconnected avg 0.676
- Pore analysis: distributed_77 = 23.0 % total / ~0 % boundary / 23.0 % intra; connected_79 = 21.8 % total / 13.8 % boundary / 8.0 % intra

## 3. Commented-out subsections in `results.tex` — disposition

Each is a real study Mattiuz did. Figures in his thesis exist (PDF: `~/Merope/old_files/File originali/2025_07_Mattiuz_Thesis_01.pdf`).

| Section in `results.tex` | Status | Where the data and figures are | Recommendation |
|---|---|---|---|
| **Simulation Setup → RVE representativity** | Commented | Mattiuz Fig 3 (p. 13). No CSV in `Results_*` — likely needs re-running. | Regenerate at current `n3D` if re-included; cheap to redo. |
| **Simulation Setup → Resolution / voxel size** | Commented | Mattiuz Fig 4 (p. 14). Same as above. | Regenerate. Useful for reviewer questions on numerical convergence. |
| **Composite voxel rules** | Commented | Mattiuz Fig 5 (p. 14). Voigt vs Reuss vs Smallest vs Largest. | Probably worth re-including: justifies the Voigt choice. |
| **Closed porosity → Effect of pore radius** | Commented | Mattiuz Fig 7 (p. 16). Result: K_eff ~ independent of inclusion radius at fixed φ (1–2 % scatter). | Re-include. Trivial figure. Old script: `old_files/File originali/Test porosità/sph_incl_conduct_calc.py`. |
| **Closed porosity → Comparison with classical models** | Commented | Mattiuz Figs 8 & 9 (p. 17). Loeb α = 2.5 over-degrades; α = 1.37 fits. | The numbers match the paper's "validation of Loeb" subsection; figure could just be reused from Mattiuz's set. |
| **Anisotropy and directional effects** | Commented | Mattiuz Fig 16 (p. 22). Relative \|K_mean − K_yy\|/K_yy vs aspect ratio γ; up to 1.4 % for high-p. | Decision: include (deepens the story, matches §6 "guidelines") OR drop entirely. Code: `experiments/run_anisotropy.py` already exists; re-running is cheap. |
| **Effect of grain size distribution** | Commented | Mattiuz Figs 17 & 18 (p. 23–24). Result: < 0.6 % effect → **conclusion: negligible**. | Currently mentioned only in the conclusions ("limited sensitivity"). Decision: re-include (clean negative result, supports δ\* dominance) or leave as a one-line reference. |

⚠ **`copy_figures.sh` exists** in the manuscript folder — read it; it may already encode the migration logic.

## 4. Bibliography issues

```
912 entries total.
```

**Confirmed citation→entry resolution:**
- `IAEA2006`, `Morimoto2008`, `Meynard2021`, `Sevostianov2019`, `Magni2021`, `kanit2003`, `SCHNEIDER2022104652`, `amitex2015`, `Josien2024`, `Underwood1970`, `Cecen2016`, `Cheaito2012`, `Millett2011`, `Kinoshita2004` — all present.

**Case-sensitivity bugs (BibTeX is case-sensitive):**
- `methods.tex` cites `\cite{Torquato2002}`, but bib has `torquato2002` (lowercase t). → ❌ undefined reference. **Fix**: rename the bib key to `Torquato2002` or update the cite.
- The Loeb model is described inline but **never `\cite{}`d** in the main text. The bib has `loeb1934`. Decide whether to add an explicit citation for the original Loeb paper; if yes, fix case.

**Action:** run `pdflatex` once and grep the `.log` for `Citation` warnings; fix all undefined refs.

## 5. ⚠ Numerical discrepancies to reconcile

These are the issues most likely to bite during a co-author review or revision.

### 5.1 Sigmoidal fit parameters — three versions on disk

See `01_theory.md` §6.3. In short:
- **Mattiuz thesis (p. 21)** parameters look stable: $K_{\min}(p)$ stays near 1, $K_{\max}(p)$ trends down from 1.10.
- **`WORKFLOW_COMPLETE.md`** has different coefficients (very steep $K_{\min}(p) = -4.74p + 1.26$).
- **`Results_Sigmoidal_Fit/fitted_parameters.csv`** (most recent) is **degenerate**: $K_{\min} = 0$ for $p = 0.2$ and $p = 0.3$, $\delta_c \approx 0$ for $p = 0.1$.

#### Root cause (data-level diagnosis of `keff_vs_delta.csv`)

The independent per-p fit in `experiments/fit_correction_factor.py` is **under-determined**:

| p | min sampled δ | min sampled K_δ | Crack-plateau sampled? |
|---|---|---|---|
| 0.1 | 0.10 | 0.91 | **No** — already on the upper plateau |
| 0.2 | 0.10 | 0.54 | One single point in the transition |
| 0.3 | **0.15** | 0.44 | No δ=0.10 point at all; transition severely under-sampled |

With the crack-dominated plateau never sampled at p=0.1 and barely sampled at p=0.2, K_min has no data to constrain it; the optimizer hits its lower bound (0). Each independent fit pulls K_min wherever the bounds allow, breaking the linear-in-p regression downstream.

#### Two complementary fixes

1. **Methodological** — fit jointly with linear-in-p parameter dependence (8 params for 31 data points instead of 12 for 11/11/9). This is implemented in `experiments/fit_correction_factor_joint.py` (added 2026-04-29). It uses `scipy.optimize.least_squares` with `loss="soft_l1"` and a positive floor on K_min. Run:
   ```bash
   python project_root/experiments/fit_correction_factor_joint.py \
     --csv Results_Keff_vs_Delta/keff_vs_delta.csv \
     --output-dir Results_Sigmoidal_Fit_Joint
   ```
2. **Data-level** — extend `run_keff_vs_delta.py` to sample δ = 0.05, 0.07, 0.09, 0.11, 0.13 for all three porosity levels. That's 15 new simulations, ≈ 1–2 hours on 2 CPUs. With those points the independent fit will work too. Edit `DELTA_VALUES` in `experiments/run_keff_vs_delta.py` (currently starts at 0.15) and re-run with `--recover` to skip already-computed cases.

The cleanest answer for the paper is: do (2), then either fit method works. (1) alone is enough for a defensible figure but does not change the fact that the data is sparse below δ=0.15.

### 5.2 Pore analysis values — three versions

| Source | connected_79 total / inter / intra | distributed_77 total / inter / intra |
|---|---|---|
| Paper (`results.tex` text) | 21.8 % / 13.8 % / 8.0 % | 23.0 % / ~0 % / 23.0 % |
| `WORKFLOW_COMPLETE.md` | 22.3 % / 13.8 % / 8.5 % | 22.7 % / 0 % / 22.7 % |
| `~/Merope/pore_analysis_results.csv` (current) | **25.9 % / 23.2 % / 2.8 %** | 23.2 % / 5.4 % / 17.8 % |

The current CSV's connected_79 has **the inter/intra split flipped vs the paper text** (23.2 % inter vs 13.8 % paper, 2.8 % intra vs 8.0 % paper). Same for distributed_77 (the paper says ~0 % inter; the CSV says 5.4 % inter).

This means the current `pore_analysis.py` parameters (circularity threshold, area thresholds, watershed setting) produce a different classification than the run that fed the paper text. The paper's `Images/Pore_Analysis/pore_analysis_results.csv` has the same "wrong" current numbers — so the figures `*_analysis.png` in the paper are **inconsistent with the paper's own narrative numbers**.

**Action:** lock the segmentation parameters that reproduce the paper's narrative (likely the WORKFLOW values), regenerate `pore_analysis_results.csv` and the per-image PNGs, and verify the targets used in `run_optimization*` shell scripts still hold.

### 5.3 Optimization scores — paper text vs archived summary

Paper Table 1: distributed avg = 0.901, interconnected avg = 0.676.
Archived `Images/Optimization_*/summary.txt`: distributed combined = 0.4895 (image avg 0.2762), interconnected combined = 0.3338 (image avg 0.0535).

The paper's table values do not appear in any `summary.txt` on disk. They may come from an earlier run that wasn't archived, or they may be from a different scoring formula (a different weighting of KS p, χ² p, or porosity penalty).

**Action:** trace which run produced 0.901 / 0.676. Either (a) re-run optimization to current scoring and update Table 1, or (b) update the summary files with the run that matches the paper text.

### 5.4 Grant agreement number

- Paper Acknowledgements: **n°101166386** ❌ (incorrect — typo)
- Mattiuz thesis Italian abstract: **No.101059543** ✓ (correct, confirmed via CORDIS)

The official ESFR-SIMPLE grant is **101059543** (EURATOM, 2021–2025; <https://cordis.europa.eu/project/id/101059543> and <https://esfr-simple.eu/>). The number in the paper Acknowledgements does not exist on CORDIS. **Action**: edit `main.tex` line ~80 to replace `101166386` with `101059543`. Also consider rewording the official project title to *"European Sodium Fast Reactor — Safety by Innovative Monitoring, Power Level flexibility and Experimental research"* (the official long form).

### 5.5 Target porosity for distributed optimization

`run_optimization.py --mode distributed` summary on disk says **target = 24.6 %**, paper text says target = 23 %. Probably a stale run; re-runnable in 2–3 hours.

## 6. Mining `old_files/File originali/`

This is the Mattiuz codebase prior to the project_root refactor. Italian filenames; `phase 1 = porous` convention; many scripts hard-coded to `/home/alessio/Thesis_Merope/...`.

| Folder / File | Content | Useful for paper? |
|---|---|---|
| `2025_07_Mattiuz_Thesis_01.pdf` | The thesis itself, 31 pp, 18 numbered figures | **Yes** — source of every commented-out figure |
| `AM_Final_Thesis_presentation.pdf` (3 MB, July 2025) | Defense slides — likely cleaner versions of key figures | Worth grabbing 2–3 figures if quality > thesis PDF |
| `AM_Thesis_presentation copy.pdf` | Earlier defense draft | Skip |
| `shared image*.png`, `Immagine*.png` | Probably the original SEM images Mattiuz worked from | Verify whether these are the same as `Optimization_3D_structure/exp_img/{connected_79,distributed_77}.png`, or different microstructures worth mentioning |
| `Gauss_dble_lay.py` | Old microstructure generator with double Gaussian layers | Superseded by `geometry.py`; archive only |
| `Test porosità/aniso_delta_calc.py` | Anisotropy sweep — produced **thesis Fig 16** | Re-runnable via current `run_anisotropy.py`; reference for parameter ranges if anisotropy section is re-included |
| `Test porosità/iter_delta_IGB_calc.py` | δ-sweep — produced **thesis Fig 11** | Superseded by `run_keff_vs_delta.py` |
| `Test porosità/IGB_porosity_calc.py` | IGB structure builder + K calc | Superseded by `geometry.generate_interconnected_structure` + `run_interconnected_porosity.py` |
| `Test porosità/vol_distribution_IGB_calc.py` | Grain-size-distribution sweep — produced **thesis Figs 17–18** | If re-included, lift parameter ranges from here |
| `Test porosità/sph_incl_conduct_calc.py` | Sphere inclusion sweep — produced **thesis Fig 7** (radius effect) and **Fig 8** (Maxwell/Loeb classical) | Reference for the closed-porosity radius-effect section |
| `Test porosità/mixed_Intra_inter_calc.py` | Inter+intra mixed — produced **thesis Fig 13** | Superseded by `run_mixed_porosity.py`; reference for parameters |
| `Test porosità/{IGB,Intra_inter_mixed_IGB,Gauss_multi_rad,2_rad_mixed}_*` | Various microstructure generators | All superseded; archive only |
| `Test statistici/compare_structures.py`, `multi_slice_comp_gauss_dble_lay.py` | Older KS/χ² code | Superseded by `core/statistics.py` |
| `Optimization_3D_structure/{main,MOX_structure_generator,statistical_test_func,fft_calc}.py` | The optimization prototype | Superseded by `experiments/run_optimization.py` + `core/statistics.py` |
| `Optimization_3D_structure/EXP IMG/` | 5 SEM images: `exp_distrib_1.png`, `exp_distributed_full.png`, `exp_interconnect_{1,2}.png`, `exp_interconnected_full.png` | Not currently used by paper; worth checking if any have better resolution than the live `connected_79.png` / `distributed_77.png` |

**Recommendation**: before deleting `old_files/`, copy the thesis PDF and the defense slides into the manuscript folder (or `~/Merope/context/references/`) for safekeeping; archive `Test porosità/` and `Test statistici/` as a single `mattiuz_legacy.tar.gz` next to the thesis; then delete the rest. `Optimization_3D_structure/exp_img/` (top-level, NOT in old_files) must remain — scripts reference it.

## 6b. Anisotropy and grain-size sections — Option A (re-run)

Decision (2026-04-29): keep both subsections in the paper. The current `project_root/experiments/run_anisotropy.py` is a stub (5 AR points, single porosity, no CSV) and does NOT reproduce thesis Fig 16. There is no script for grain-size distribution at all.

Plan:
1. Rewrite `experiments/run_anisotropy.py` to match thesis methodology: 15–20 AR values from 0.1 to 1.0, two porosity levels (low p ≈ 0.05–0.1 and high p ≈ 0.2), volume‑preserving anisotropy (`[1, γ, 1/γ]`), mixed Laguerre+spherical morphology, full CSV output.
2. Write new `experiments/run_grain_size_distribution.py` based on `old_files/Test porosità/vol_distribution_IGB_calc.py`: two σ values of the volume‑weighting function (0.5 and 3.0) at fixed (p, δ).
3. Plot helpers under `Results_Anisotropy/` and `Results_GrainSizeDistribution/` producing paper-quality figures.
4. Run both, copy figures to `paper/Images/Anisotropy/` and `paper/Images/GrainSizeDistribution/`.
5. Port thesis prose §3.4.2 and §3.4.3 into `results.tex`, uncomment those subsections.

Estimated compute: ≈ 3 hours (anisotropy: 20 AR × 2 p × ~5 min/case at n3D=100; grain-size: 2 cases at n3D=150).

## 7. Writing-guidelines pass

A `~/research-manuscripts/writing_guidelines.md` file has been authored by Giovanni and applies to this manuscript. Key rules: UK English everywhere in prose (-ise/-isation, -our, centre, modelled, analyse), single hyphens only (no en-dash `--` or em-dash `---`), no `\emph{}` for stress, units in parentheses not brackets, "X et al." author attribution, "selection" vs "calibration" used precisely. The guidelines explicitly prohibit en-dash "fixes" — keep plain `-`.

Sweep done 2026-04-29 across `main.tex`, `introduction.tex`, `methods.tex`, `results.tex`, `discussion.tex`, `conclusions.tex`. Re-apply if new prose is added.

## 8. Decisions for Giovanni

In rough order of impact on time-to-submission. Items marked ✅ done in the 2026-04-29 audit pass.

1. **Sigmoidal fit reconciliation** (§5.1) — joint fit script written (`experiments/fit_correction_factor_joint.py`); needs to be run + plot regenerated. Optionally extend δ-sweep to δ < 0.15 to constrain K_min from data.
2. **Pore-analysis number reconciliation** (§5.2) — paper text contradicts paper figures. Pick one, regenerate the other.
3. **Optimization scores in Table 1** (§5.3) — paper Table 1 says distributed avg=0.901, interconnected avg=0.676; the archived `summary.txt` files show 0.4895/0.3338. The paper's claimed numbers are not reproducible from any artifact on disk.
4. **Commented-out subsections** (§3) — Giovanni confirmed (2026-04-29) that some thesis figures are "too basic" and the commented text can be deleted. Suggested deletions: RVE representativity, voxel resolution, composite voxel rules (3 sections — methodology details that the paper's Methods covers in prose). Keep candidate: closed-porosity Loeb α=1.37 calibration (this is part of the headline result), and possibly anisotropy (narrative weight) and grain-size-distribution (clean negative result).
5. **Bibliography case bugs** (§4) — ✅ `Torquato2002` fixed in bib; check that `loeb1934` is intentionally uncited (not a missing cite).
6. **Grant agreement number** (§5.4) — ✅ confirmed correct value is 101059543 (EURATOM, CORDIS-verified). `main.tex` Acknowledgements still has wrong 101166386 — needs editing.
7. **`project_root/README.md`** — out of date; lists 3 core files + 3 experiments, reality is 8 + 12. Update as part of polish.
8. **Decide on `old_files/` cleanup plan** (§6).

## 8. Things that may still need new simulation

If we re-include the commented sections cleanly:
- **RVE convergence plot** at the n3D values used in the paper's main results (cheap; ~ 1 hour).
- **Voxel resolution sweep** (cheap; ~ 1 hour).
- **Composite voxel rules comparison** (cheap; ~ 30 min if scripted).
- **Anisotropy** at one or two porosity levels matching the paper's δ-sweep (overnight).
- **Grain-size-distribution** at $p = 0.2, \delta = 0.4$ (overnight).
- **Re-fit sigmoidal** with regularization (minutes; just re-run `fit_correction_factor.py` with cleaner input).
- **Re-run pore analysis** with parameters that match paper text (minutes).
- **Re-run Bayesian optimization** with the corrected target porosities (~ 4 hours per mode).

Total worst case: ~ 1 day of compute + ~ 1 day of write-up to fold the missing sections back in. Best case (drop the missing sections, fix only §5 and §4): ~ 2 hours.
