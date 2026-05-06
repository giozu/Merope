# 00 — Big picture

## What this project is

A study of **how pore morphology controls effective thermal conductivity** $K_{\text{eff}}$ of ceramic nuclear fuels (UO$_2$, MOX). The deliverable is a journal paper for *Nuclear Engineering and Design* (target confirmed in `main.tex` line 15: `\journal{Nuclear Engineering and Design}`):

> **Luzzi, Zullo, Mattiuz, Pizzocri** — *Morphological analysis of the thermal conductivity in highly porous ceramic materials* (2026).

Manuscript folder: `~/research-manuscripts/Luzzi_et_al___MEROPE__2026/`.

## Why it matters

For high‑porosity engineered fuels (10–30 %), the classical Loeb correlation $\kappa = \kappa_0(1-\alpha p)$ with the IAEA value $\alpha=2.5$ is wrong: it ignores **pore connectivity**. A fuel with the same total porosity can lose 40 % of its conductivity if pores percolate along grain boundaries instead of staying intragranular. Fuel performance codes need a morphology‑aware correction.

## ESFR‑SIMPLE context

This work is **WP 8 / WP 8.2.3** of the **ESFR‑SIMPLE** Horizon Europe / EURATOM project (European Sodium Fast Reactor — Safety by Innovative Monitoring, Power Level flexibility and Experimental research).

**Grant Agreement: n°101059543** (EURATOM 2021‑2025, confirmed via CORDIS and the project's own site at <https://esfr-simple.eu/>). Verified correct in `main.tex` line 75 (the earlier draft typo `n°101166386` has since been fixed).

⚠ **The two reference SEM images** that drive the Bayesian optimisation (`exp_img/connected_79.png`, `exp_img/distributed_77.png`) come from `~/Merope/old_files/ESFR_SIMPLE_Monitoring_Meeting_3_WP8.pptx` (slides 12-14: real CEA Marcoule MOX specimens fabricated for Subtask 8.2.2; characterisation was "on going" at the meeting; planned JRC Karlsruhe $K_{\text{eff}}$ measurements never landed). The deck is private consortium material — these images **cannot be reproduced in a journal paper without rights clearance**. Pivot 2026-05-04: build fully reproducible synthetic targets that hit the same nominal porosity values (`make_synthetic_targets.py`); see `03_paper_and_open_issues.md` §5.7.

## Authors and history

- **Lelio Luzzi** — advisor (lead PI on POLIMI side)
- **Davide Pizzocri** — co‑advisor / co‑author
- **Giovanni Zullo** — co‑author, current driver of the paper, **Mérope user (not developer)**
- **Alessio Mattiuz** — MSc student (Politecnico di Milano, July 2025); his thesis is the basis of this paper

Mattiuz produced the bulk of the parametric studies in 2024–2025 in `~/Merope/old_files/File originali/` (Italian filenames). Giovanni's job is to **consolidate, reconcile, and publish**. Mattiuz's thesis is `2025_07_Mattiuz_Thesis_01.pdf` (31 pp, 13 main figures).

## Tools and pipeline (one-liner)

`Mérope` (CEA, microstructure generator, Python wrapper over C++) → voxelize → `AMITEX‑FFTP` (CEA, FFT homogenization solver) → ParaView for visualization. Bayesian optimization (`scikit‑optimize`) calibrates synthetic structures against SEM images.

Material constants used throughout: $\kappa_m = 1.0$ (normalized matrix), $\kappa_g = 10^{-3}$ (gas), contrast ratio $10^3$.

## The headline results

1. **Loeb recalibrated**: $\alpha = 1.37$ (vs IAEA 2.5) gives < 2 % error over $p \in [0, 0.30]$ for distributed porosity. Matches Morimoto et al. (2008) for $(U,Pu,Am)O_2$.
2. **Sigmoidal correction factor** $K_\delta(p, \delta^*)$ applied as $K_{\text{eff}} = K_{\text{Loeb}}(p) \cdot K_\delta(p,\delta^*)$, where $\delta^* = \delta / L_{\text{grain}}$ is the normalized grain‑boundary thickness. Captures the smooth transition from crack‑dominated ($\delta^* < 0.2$) to distributed‑like ($\delta^* > 0.5$) regimes. **Canonical joint-fit parameters (2026-04-30, 47-point CSV)**: $K_{\min}(p) = 0.85 - 2.50p$, $K_{\max}(p) = 0.996 - 0.203p$, $b(p) = -25.96 - 1.00p$, $\delta_c(p) = 0.008 + 0.530p$. See `Results_Sigmoidal_Fit/linear_coeffs.csv` (top-level).
3. **Morphology penalty is sharp but gated by $\delta^*$**: at the saturated-regime case ($\delta^* = 0.32$, matched-form synthetic) the residual penalty is only $\sim 3\%$ — indistinguishable from Loeb on the figure. At the morphology-controlled regime ($\delta^* = 0.10$, thin-band mixed-porosity AMITEX run, 2026-05-06), $K_{\rm eff}$ falls **15.2 % below the Loeb baseline** at the same total porosity ($p_{\rm total} = 0.213$), with the joint-fit prediction within 5 % of AMITEX. The earlier "40 %" headline was an asymptote-vs-operating-point conflation; the actual measured penalty in the regime where it bites is 15 %. The design lever is $\delta^*$ itself, not "interconnected vs distributed" as a categorical statement.
4. **Bayesian optimization works on a known-truth recovery test**: against a matched-form synthetic with ground-truth $\delta = 1.0$, the optimiser recovers $\delta = 0.959$ — **4.1 % recovery error** (2026-05-06). Reference targets are reproducible synthetic microstructures (`make_synthetic_targets.py` for slice PNGs; `run_thin_delta_mixed.py` for the thin-$\delta^*$ AMITEX RVE) rather than the private ESFR-SIMPLE consortium SEMs (which remain on disk for comparison runs only).

## What's drafted vs what's loose

The paper is structurally complete: `main.tex` pulls `introduction.tex`, `methods.tex`, `results.tex`, `discussion.tex`, `conclusions.tex`. The `Images/` folder has every figure currently cited.

But **`results.tex` has 5 subsections commented out** ("missing figures"): RVE convergence, voxel resolution, composite voxel rules, closed‑porosity radius effect & Maxwell/Loeb baseline, anisotropy, grain‑size distribution. **Each of these corresponds to a real figure in Mattiuz's thesis** (Figs 3, 4, 5, 7, 8–9, 16, 17–18). The studies happened; only the figures and write‑ups didn't migrate. Decision for Giovanni: re‑include or drop. See `03_paper_and_open_issues.md`.

## Where the code lives

- **Canonical**: `~/Merope/project_root/` (`core/` library + `experiments/` scripts).
- **Top‑level `~/Merope/`**: holds the Mérope C++/Python library (third‑party, do not edit), shell wrappers that call into `project_root/experiments/`, and **all canonical `Results_*` directories** (after the 2026-04-30 cleanup; see `02_code_pipeline.md`).
- **`~/Merope/_to_delete/`** (created 2026-04-30): 10 stale folders + 3 redundant scripts staged for deletion. `rm -rf` when satisfied.
- **Legacy**: `~/Merope/old_files/` — Mattiuz‑era scripts and the thesis PDF. **Note (2026-04-30)**: this machine's recovered copy has the layout `old_files/Test porosità/...` (no `File originali/` subdir as on the previous machine).

## Reading order for new context

1. This file (`00_big_picture.md`)
2. `01_theory.md` — physics and analytical models
3. `02_code_pipeline.md` — what every script does and how to run it
4. `03_paper_and_open_issues.md` — section‑by‑section state, numerical discrepancies to reconcile, decisions Giovanni needs to make before submission
