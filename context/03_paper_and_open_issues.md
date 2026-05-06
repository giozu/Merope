# 03 — Paper status and open issues

This file is the working punch-list for finishing the paper. It catalogs (a) section-by-section state, (b) numerical discrepancies that need reconciling before submission, (c) what's worth mining from `old_files/`, and (d) decisions for Giovanni.

## 1. Manuscript inventory

`~/research-manuscripts/Luzzi_et_al___MEROPE__2026/`

| File | Status |
|---|---|
| `main.tex` | OK — frontmatter, abstract, highlights, keywords, includes 5 sections |
| `introduction.tex` | OK — narrative complete |
| `methods.tex` | OK — physics + classical models + tools |
| `results.tex` | 3 subsections still commented out (Anisotropy + Grain-size written 2026-05-04); §"Experimental validation" rewritten 2026-05-04 against current artefacts — see §3 below |
| `discussion.tex` | OK — physical interpretation, limitations, comparison with literature |
| `conclusions.tex` | OK |
| `bibliography.bib` | 912 entries; **case-sensitivity issues** — see §4 |
| `Images/` | 12 PNGs across 6 subfolders, all currently-cited figures present |
| `copy_figures.sh` | Helper script to copy figures from `~/Merope/Results_*/` into `Images/` |
| `main.pdf` | Last build artefact |

### 1.1 Figures currently in `Images/`

```
Anisotropy/        anisotropy.png                                              ← added 2026-05-04
Comparison/        comparison_distributed_vs_interconnected.png, keff_vs_porosity_comparison.png
                                                                                ← FIXME: stale, regenerate from current artefacts
GrainSizeDistribution/  volume_histograms.png, keff_comparison.png             ← added 2026-05-04
Keff_vs_Delta/     Slide_Keff_vs_Delta.png, keff_vs_delta.csv
Keff_vs_Porosity/  Keff_Validation_Summary.png, keff_vs_porosity.csv
Optimization_Distributed/    area_distribution.png, best_slice.png, convergence.png, summary.txt, keff_prediction.txt
Optimization_Interconnected/ area_distribution.png, best_slice.png, convergence.png, summary.txt, keff_prediction.txt
Pore_Analysis/     distributed_77_original.png, distributed_77_analysis.png, connected_79_original.png, connected_79_analysis.png, pore_analysis_results.csv
                                                                                ← consortium-image originals; replace with synthetic in Phase 2.6
Sigmoidal_Fit/     Sigmoidal_Fits.png, Parameters_vs_Porosity.png, K_eff_Contour.png   ← refreshed 2026-04-30 from joint fit;
                   fitted_parameters.csv, linear_coeffs.csv                              also dropped here for traceability.
```

## 2. Headline numbers in the paper

For sanity-checking against the codebase / data:

- Loeb recalibrated: **α = 1.37** (Loeb_classical: 2.5; matches Morimoto 2008)
- Distributed/Loeb error: **< 2 %** over $p \in [0, 0.30]$
- Morphology penalty at the optimised δ\*=0.283 (interconnected, $p_b = 0.138$): $1 - K_\delta \approx 3\%$. The "40 % reduction" claim from earlier drafts only holds at sub-percolation $\delta^* < \delta_c \approx 0.08$ where $K_\delta \to K_{\min}(p) \approx 0.6$. **Updated 2026-05-04**.
- Bayesian optimization scores in Table 1: 0.901 / 0.676 (paper text); will be re-derived from the synthetic-target run (`Phase 1.5` in `04_next_session_plan.md`). See §5.3.
- Pore analysis numbers (`p_b = 0.138`, `p_intra = 0.080`, `p_distributed = 0.230`) now reproduced **by construction** from the synthetic-target generator (`make_synthetic_targets.py`), so they no longer depend on the broken legacy CLI; see §5.2 and §5.7.

## 3. Commented-out subsections in `results.tex` — disposition

Each is a real study Mattiuz did. Figures in his thesis exist (PDF: `~/Merope/old_files/File originali/2025_07_Mattiuz_Thesis_01.pdf`).

| Section in `results.tex` | Status | Where the data and figures are | Recommendation |
|---|---|---|---|
| **Simulation Setup → RVE representativity** | Commented | Mattiuz Fig 3 (p. 13). No CSV in `Results_*` — likely needs re-running. | Regenerate at current `n3D` if re-included; cheap to redo. |
| **Simulation Setup → Resolution / voxel size** | Commented | Mattiuz Fig 4 (p. 14). Same as above. | Regenerate. Useful for reviewer questions on numerical convergence. |
| **Composite voxel rules** | Commented | Mattiuz Fig 5 (p. 14). Voigt vs Reuss vs Smallest vs Largest. | Probably worth re-including: justifies the Voigt choice. |
| **Closed porosity → Effect of pore radius** | Commented | Mattiuz Fig 7 (p. 16). Result: K_eff ~ independent of inclusion radius at fixed φ (1–2 % scatter). | Re-include. Trivial figure. Old script: `old_files/File originali/Test porosità/sph_incl_conduct_calc.py`. |
| **Closed porosity → Comparison with classical models** | Commented | Mattiuz Figs 8 & 9 (p. 17). Loeb α = 2.5 over-degrades; α = 1.37 fits. | The numbers match the paper's "validation of Loeb" subsection; figure could just be reused from Mattiuz's set. |
| **Anisotropy and directional effects** | ✅ DONE 2026-05-04 | `Images/Anisotropy/anisotropy.png` from the 2026-04-30 sweep. | Subsection drafted at `results.tex` §"Anisotropy and directional effects in interconnected porous microstructures"; reports up-to-1.5 % spread, justifies scalar K_eff. |
| **Effect of grain size distribution** | ✅ DONE 2026-05-04 | `Images/GrainSizeDistribution/{volume_histograms,keff_comparison}.png` from the 2026-04-30 sweep. | Subsection drafted at `results.tex` §"Effect of grain size distribution on thermal conductivity"; reports ~0.5 % effect, validates monodisperse Laguerre calibration. |

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

### 5.1 Sigmoidal fit parameters — RESOLVED (2026-04-30)

The joint fit (`experiments/fit_correction_factor_joint.py`) was run on the canonical 47-point `keff_vs_delta.csv` and produced clean coefficients:

| Parameter | a (slope) | b (intercept) |
|---|---|---|
| $K_{\min}(p)$ | -2.500 | +0.850 |
| $K_{\max}(p)$ | -0.203 | +0.996 |
| $b(p)$ | -1.000 | -25.96 |
| $\delta_c(p)$ | +0.530 | +0.008 |

Cost = 0.01014 over 47 points (mean residual ~0.014 in $K_{\text{eff}}$ units; `ftol`-converged in 13 iterations). Outputs are at `Results_Sigmoidal_Fit/` (top-level; renamed from `Results_Sigmoidal_Fit_Joint`) and the three PNGs + two CSVs were copied into `paper/Images/Sigmoidal_Fit/`.

**Caveat to disclose in the paper.** Of the three $K_{\min}$ anchor points, only $K_{\min}(p=0.2) = 0.35$ is data-anchored (the dataset has a $\delta=0.10$ point where $K_\delta$ has dropped to 0.39, near the plateau). $K_{\min}(p=0.1)$ and $K_{\min}(p=0.3)$ are **structural extrapolations** through the linear-in-p regression. One sentence in `discussion.tex` is enough.

**The data-level fix (extend δ < 0.15) was attempted and abandoned (2026-04-30).** A wrapper (`run_keff_vs_delta_p03_extension.py`, now in `_to_delete/`) ran for ~1 h at $n_\text{3D}=200$ without producing a single new row. AMITEX's iterative scheme stalls at $\delta=0.05$ because the GB film is exactly one voxel thick at that resolution and the 10³ contrast jump makes the linear system pathologically ill-conditioned. A genuine anchor would require $n_\text{3D} \geq 400$ — out of scope for this paper. See `01_theory.md` §6.3 for the longer write-up.

**Old per-p fit script kept as a deprecated benchmark.** `fit_correction_factor.py` is annotated at the top with the deprecation notice, and its $K_{\min}$ lower bound was raised from 0 to 0.05 so it can no longer produce the fully-degenerate fit.

### 5.2 Pore analysis values — three versions — SUPERSEDED (2026-05-04)

The pivot to synthetic targets (§5.7) makes this issue largely cosmetic. The
synthetic generator hits `p_b = 0.137`, `p_intra = 0.080`, `p_distributed = 0.230`
**by construction**, matching the paper-text numbers without requiring a working
legacy CLI. The current `core/pore_analysis.py` should still be re-validated
against the synthetic PNGs in Phase 2.6 to confirm a downstream pipeline works
on a controlled input, but the paper text no longer depends on it.

#### Original write-up (kept for the record)



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

### 5.4 Grant agreement number — RESOLVED (verified 2026-04-30)

`main.tex` line 75 already contains `n°101059543` (CORDIS-verified correct). The earlier note about a `101166386` typo refers to a draft state that has since been fixed. Nothing to do.

### 5.5 Target porosity for distributed optimization

`run_optimization.py --mode distributed` summary on disk says **target = 24.6 %**, paper text says target = 23 %. Probably a stale run; re-runnable in 2–3 hours.

### 5.6 Stale sigmoidal coefficients in `predict_keff_from_optimization.py` — RESOLVED (2026-04-30, extended 2026-05-04)

The script previously hardcoded the OLD WORKFLOW coefficients (`k_min = -4.74p + 1.26`, etc.) — meaning any `keff_prediction.txt` it produced was internally inconsistent with the joint-fit figure in `paper/Images/Sigmoidal_Fit/`. Refactor: it now loads `{slope, intercept}` from `Results_Sigmoidal_Fit/linear_coeffs.csv` at runtime via `load_sigmoid_coeffs()`. Verified that the loader returns the joint-fit values exactly.

**Headline-claim verification done (2026-05-04).** The "40 % morphology reduction" claim does NOT hold at the optimised δ\*=0.283: the joint fit yields $K_\delta = 0.97$, so the residual morphology penalty is only ~3 %. The 40 % figure is the sigmoid's lower asymptote ($K_\delta \to K_{\min}(p)$), reachable only for sub-percolation $\delta^* < \delta_c \approx 0.08$. The paper §"Experimental validation" was rewritten 2026-05-04 to reflect this; the original headline was an asymptotic-vs-operating-point conflation. Confirmed by independent per-p fit (`Results_Sigmoidal_Fit_PerP/`) which gives $K_\delta = 0.89$ at the same evaluation point — both fits within ~3 % of each other and far from 0.6.

`predict_keff_from_optimization.py` now also prints two K_eff values: AMITEX-comparable (`K_loeb · K_δ`) and composite (`× (1 − 1.37·p_intra)`), with the morphology penalty `1 − K_δ` reported separately. No more silent conflation between an FFT-comparable prediction and a composite real-material estimate.

### 5.7 Synthetic-targets pivot for the optimisation reference images (2026-05-04)

The two reference images that the Bayesian optimisation matches —
`connected_79.png` and `distributed_77.png` — come from
`ESFR_SIMPLE_Monitoring_Meeting_3_WP8.pptx`, a private consortium deck.
Slides 12-14 confirm they are SEM micrographs of MOX fuel pellets fabricated
by CEA Marcoule for ESFR-SIMPLE Subtask 8.2.2; characterisation was "on going"
at the meeting time and the planned JRC Karlsruhe $K_{\text{eff}}$
measurements never landed (no reply from ATALANTE on this).

**Implication:** the images cannot be reproduced in a journal paper without
rights clearance, and no paired $K_{\text{eff}}$ measurement exists, so the
former §"Experimental validation" framing was overstated even in principle.

**Resolution:** generate fully reproducible synthetic targets that hit the same
nominal porosity values the paper quotes. New script
`project_root/experiments/make_synthetic_targets.py` produces
`Optimization_3D_structure/exp_img_synthetic/{synthetic_distributed,synthetic_interconnected}.png`
with realised porosities (boundary-clipped + intra union for the interconnected case):

| Quantity | Paper claim | Synthetic |
|---|---|---|
| Distributed total | 23.0 % | 23.0 % (3D) |
| Interconnected total | 21.8 % | 20.6 % (3D, after overlap loss) |
| Interconnected boundary | 13.8 % | 13.7 % |
| Interconnected intra | 8.0 % | 8.0 % |

Ground-truth parameters recorded in `exp_img_synthetic/ground_truth.json`.
`run_optimization.py` now has an `--exp-image-set {synthetic, consortium}`
flag (default synthetic). The original consortium results were preserved
under `Results_Optimization_*_consortium/` before re-running with synthetic
targets (run started end-of-2026-05-04 in tmux `opt-synth`).

Methods text in `methods.tex` will need a paragraph in the next session
explaining the synthetic-targets choice (Phase 2.6 in the next-session plan).

## 6. Mining `old_files/`

This is the Mattiuz codebase prior to the project_root refactor. Italian filenames; `phase 1 = porous` convention; many scripts hard-coded to `/home/alessio/Thesis_Merope/...`.

⚠ **Layout on this machine (recovered 2026-04-30)**: `~/Merope/old_files/Test porosità/...` (no `File originali/` subdir as on the previous machine). Adjust paths in this section accordingly when looking for the referenced scripts.

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

## 6b. Anisotropy and grain-size sections — DONE (2026-04-30, 2026-05-04)

Decision (2026-04-29): keep both subsections in the paper. ✅ Now complete:

1. ✅ `experiments/run_anisotropy.py` rewritten to match thesis methodology (20 AR
   values × 2 porosity levels = 40 cases). Sweep ran 2026-04-30. Slipped cases at
   γ ≈ 0.81 / 0.86 patched 2026-05-04 (recovery filter tightened to use
   `min(Kxx,Kyy,Kzz) > 0.01`). Final dataset 40/40 valid.
2. ✅ `experiments/run_grain_size_distribution.py` written and run 2026-04-30.
   Two σ values, ~0.5 % effect on K_eff at p=0.20 (validates monodisperse
   calibration).
3. ✅ Plot helpers in both scripts produce paper-quality figures.
4. ✅ Figures copied to `paper/Images/Anisotropy/` and
   `paper/Images/GrainSizeDistribution/` 2026-05-04.
5. ✅ Subsections drafted in `results.tex` 2026-05-04 with current numbers
   (replacing the commented-out placeholders at lines 60-69).

## 7. Writing-guidelines pass

A `~/research-manuscripts/writing_guidelines.md` file has been authored by Giovanni and applies to this manuscript. Key rules: UK English everywhere in prose (-ise/-isation, -our, centre, modelled, analyse), single hyphens only (no en-dash `--` or em-dash `---`), no `\emph{}` for stress, units in parentheses not brackets, "X et al." author attribution, "selection" vs "calibration" used precisely. The guidelines explicitly prohibit en-dash "fixes" — keep plain `-`.

Sweep done 2026-04-29 across `main.tex`, `introduction.tex`, `methods.tex`, `results.tex`, `discussion.tex`, `conclusions.tex`. Re-apply if new prose is added.

## 8. Decisions for Giovanni

In rough order of impact on time-to-submission. Items marked ✅ done.

1. **Sigmoidal fit reconciliation** (§5.1) — ✅ joint fit run, paper figures refreshed, deprecation note on the per-p script.
2. **Stale coefficients in `predict_keff_from_optimization.py`** (§5.6) — ✅ refactored to load from `linear_coeffs.csv`; ✅ "40 % reduction" claim verified to be an asymptote-vs-operating-point conflation; paper §"Experimental validation" rewritten 2026-05-04.
3. **Pore-analysis number reconciliation** (§5.2) — ✅ SUPERSEDED by the synthetic-targets pivot (§5.7); the paper-text numbers are now reproduced by construction.
4. **Optimization scores in Table 1** (§5.3) — still open. Will be re-derived from the synthetic-target optimisation run started end-of-2026-05-04 (verify in Phase 1.5 of `04_next_session_plan.md`).
5. **Commented-out subsections** (§3) — ✅ Anisotropy and grain-size subsections drafted 2026-05-04. Three remaining subsections (RVE convergence, voxel resolution, composite voxel rules) confirmed 2026-04-29 as "too basic" — leave commented or delete.
6. **Bibliography case bugs** (§4) — ✅ `Torquato2002` fixed; ✅ duplicate `Magni2020` removed 2026-05-04 during compile cleanup.
7. **Grant agreement number** (§5.4) — ✅ verified correct in `main.tex` line 75.
8. **Synthetic-targets pivot** (§5.7) — ✅ generator written, wired into `run_optimization.py`, run started 2026-05-04. Methods text (`methods.tex`) still needs a paragraph explaining the choice — Phase 2.6 in the next-session plan.
9. **Stale comparison figures** — ⚠ `Images/Comparison/{comparison_distributed_vs_interconnected,keff_vs_porosity_comparison}.png` carry FIXME markers in `results.tex`; regenerate from current artefacts in Phase 2.5 of the next-session plan.
10. **`project_root/README.md`** — out of date; lists 3 core files + 3 experiments, reality is 5 + 13 (added `make_synthetic_targets.py`). Update as part of polish.
11. **Final cleanup**: review `_to_delete/` and `rm -rf` once satisfied. Decide whether to keep the three `run_*_porosity.py` validation scripts (interconnected and mixed last produced K=0; status uncertain).

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
