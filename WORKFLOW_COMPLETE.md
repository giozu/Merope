# Complete Workflow: Microstructure Optimization & K_eff Prediction

## Overview

This document describes the complete workflow for analyzing experimental microstructure images, optimizing 3D structures, and predicting thermal conductivity using morphology-aware models.

---

## Phase 1: Image Analysis (DONE ✅)

### Step 1.1: Analyze experimental images

**Connected morphology (connected_79.png):**
```bash
python project_root/core/pore_analysis.py \
  "Optimization_3D_structure/exp_img/connected_79.png" \
  0.195 80
```

**Results:**
- Total porosity: 22.3% (with stereological correction)
- Boundary pores: 13.8% (62% of total, interconnected)
- Intra pores: 8.5% (38% of total, isolated)

**Distributed morphology (distributed_77.png):**
```bash
python project_root/core/pore_analysis.py \
  "Optimization_3D_structure/exp_img/distributed_77.png" \
  0.195 80
```

**Results:**
- Total porosity: 22.7%
- Boundary pores: 0.0% (no grain boundaries)
- Intra pores: 22.7% (100% isolated spheres)

---

## Phase 2: Bayesian Optimization (IN PROGRESS 🏃)

### Step 2.1: Optimize interconnected morphology

**Target:** Match connected_79.png morphology (boundary pores only)

```bash
python project_root/experiments/run_optimization.py \
  --mode interconnected \
  --exp-image "Optimization_3D_structure/exp_img/connected_79.png" \
  --n-calls 50 \
  --n3d 120
```

**What it optimizes:**
- `delta` (grain boundary thickness): [0.39, 3.0]
- `pore_radius` (pore size): [0.20, 0.50]
- `pore_phi` (volume fraction): [0.05, 0.50]

**Target porosity:** 13.8% (boundary only)

**Estimated time:** 3-4 hours (50 calls × ~4 min/call)

**Output:**
- `Results_Optimization_Interconnected/best_geometry/structure.vtk`
- `Results_Optimization_Interconnected/best_slice.png`
- `Results_Optimization_Interconnected/summary.txt` ← **Contains best delta!**
- `Results_Optimization_Interconnected/convergence.png`

---

### Step 2.2: Optimize distributed morphology

**Target:** Match distributed_77.png morphology (spherical pores)

```bash
python project_root/experiments/run_optimization.py \
  --mode distributed \
  --exp-image "Optimization_3D_structure/exp_img/distributed_77.png" \
  --n-calls 50 \
  --n3d 120
```

**What it optimizes:**
- `mean_radius` (log-normal mean)
- `std_radius` (log-normal std)
- `small_frac` (fraction of nano-pores)

**Target porosity:** 22.7%

**Estimated time:** 2-3 hours

**Output:**
- `Results_Optimization_Distributed/best_geometry/structure.vtk`
- `Results_Optimization_Distributed/best_slice.png`
- `Results_Optimization_Distributed/summary.txt`

---

## Phase 3: K_eff Prediction (PENDING ⏳)

After optimization completes, predict K_eff using the fitted correction factor.

### Step 3.1: Predict for interconnected

```bash
python project_root/experiments/predict_keff_from_optimization.py \
  Results_Optimization_Interconnected
```

**What it does:**
1. Loads best `delta` from optimization
2. Applies correction factor model:
   ```
   K_eff = K_Loeb(p_boundary) × K_δ(p, delta) × (1 - α × p_intra)
   ```
3. Prints prediction breakdown
4. Saves to `Results_Optimization_Interconnected/keff_prediction.txt`

**Expected output:**
```
K_eff PREDICTION FROM OPTIMIZATION RESULTS
==================================================================
Mode: INTERCONNECTED
Results directory: Results_Optimization_Interconnected

Best parameters:
  delta                = 1.234

Input parameters:
  p_boundary  = 13.8% (interconnected)
  p_intra     = 8.5% (isolated)
  p_total     = 22.3%
  delta       = 1.234

----------------------------------------------------------------------
PREDICTION BREAKDOWN:
----------------------------------------------------------------------
1. Loeb model (boundary):      K = 0.8110
2. Morphology correction:       K_δ = 0.6234
3. Boundary contribution:       K = 0.5054
4. Intra correction (×0.883):   K = 0.4463
----------------------------------------------------------------------

✓ PREDICTED K_eff = 0.4463 W/m·K

Comparison with distributed morphology (p=22.3%):
  K_distributed = 0.6949 W/m·K (Loeb classical)
  K_interconnected = 0.4463 W/m·K (with correction)
  Reduction due to morphology: 35.8%
```

---

### Step 3.2: Predict for distributed

```bash
python project_root/experiments/predict_keff_from_optimization.py \
  Results_Optimization_Distributed
```

**Expected output:**
```
K_eff PREDICTION FROM OPTIMIZATION RESULTS
==================================================================
Mode: DISTRIBUTED

Input parameters:
  p_total     = 22.7% (all isolated)

----------------------------------------------------------------------
PREDICTION:
----------------------------------------------------------------------
Classical Loeb model: K = K_matrix × (1 - 1.37 × p)
                      K = 1.0 × (1 - 1.37 × 0.227)
                      K = 0.6890
----------------------------------------------------------------------

✓ PREDICTED K_eff = 0.6890 W/m·K
```

---

### Step 3.3: Compare both morphologies

```bash
python project_root/experiments/compare_optimization_results.py
```

**Output:**
- Console table comparing porosity, K_eff, reduction
- Plot: `comparison_distributed_vs_interconnected.png`

**Expected key finding:**
```
Reduction due to morphology: ~35-45%

Despite having similar total porosity (~22%), the interconnected
sample has 35-45% lower K_eff due to crack-like pore morphology
at grain boundaries.
```

---

## Phase 4: Validation (Optional)

If you want to validate the prediction with full Amitex simulation:

### Step 4.1: Run Amitex on best structure

```bash
cd Results_Optimization_Interconnected/best_geometry

# Amitex simulation (uses existing structure.vtk and Coeffs.txt)
mpirun -np 6 amitex_fftp -n 3 3 3
```

**Compare:**
- Predicted: ~0.44 W/m·K (from correction factor)
- Amitex: ??? W/m·K (full simulation)
- Error: ???%

---

## Summary of Tools Created

| Tool | Purpose | Usage |
|------|---------|-------|
| **pore_analysis.py** | Analyze experimental images → p_boundary, p_intra | `python core/pore_analysis.py image.png 0.195 80` |
| **run_optimization.py** | Bayesian optimization to match morphology | `python experiments/run_optimization.py --mode interconnected --n-calls 50` |
| **predict_keff_from_optimization.py** | Predict K_eff from optimized delta | `python experiments/predict_keff_from_optimization.py Results_Optimization_Interconnected` |
| **compare_optimization_results.py** | Compare distributed vs interconnected | `python experiments/compare_optimization_results.py` |
| **fit_correction_factor.py** | Fit sigmoid model (already done) | `python experiments/fit_correction_factor.py --csv Results_Keff_vs_Delta/keff_vs_delta.csv` |

---

## Key Parameters

### Interconnected mode
- Target porosity: **13.8%** (boundary only, from pore_analysis)
- Additional intra: **8.5%** (applied as correction in prediction)
- Grain radius: **3.0** (from run_keff_vs_delta.py)
- Pore radius: **0.3** (optimized in range [0.20, 0.50])
- Delta: **optimized** (range [0.39, 3.0])

### Distributed mode
- Target porosity: **22.7%** (all intra, from pore_analysis)
- No grain boundaries
- Mean radius: **optimized** (log-normal distribution)
- Std radius: **optimized**

---

## Correction Factor Model

From `run_keff_vs_delta.py` (63 simulations):

```
K_eff(p, δ) = K_Loeb(p) × K_δ(p, δ)

K_δ(p, δ) = k_min + (k_max - k_min) / [1 + exp(b × (δ - δ_c))]
```

**Linear parameter dependencies:**
```
k_min(p) = -4.74 × p + 1.26
k_max(p) = -0.15 × p + 1.00
b(p)     = -1.98 × p - 5.58
δ_c(p)   = -1.08 × p + 0.64
```

**Physical interpretation:**
- **k_min**: K_eff at low δ (interconnected cracks)
- **k_max**: K_eff at high δ (isolated spheres)
- **b**: Steepness of transition
- **δ_c**: Critical δ where transition occurs

---

## Expected Results

| Sample | Morphology | p_total | K_eff (predicted) | Model |
|--------|-----------|---------|-------------------|-------|
| **distributed_77** | Isolated spheres | 22.7% | ~0.69 W/m·K | Loeb classical |
| **connected_79** | Interconnected | 22.3% | ~0.44 W/m·K | Correction factor |

**Key finding:** ~36% reduction in K_eff due to morphology alone!

---

## Next Steps After Optimization Completes

1. ✅ Run `predict_keff_from_optimization.py` on both results
2. ✅ Run `compare_optimization_results.py` to generate comparison plot
3. 📊 Review convergence plots to ensure optimization converged
4. 🔍 Inspect best_slice.png to visually validate morphology match
5. 📝 Document best delta and K_eff predictions
6. (Optional) Run Amitex validation if needed

---

## Troubleshooting

### Optimization not converging?
- Increase `--n-calls` to 100-150
- Check convergence plot (should plateau)
- Verify exp_image path is correct

### K_eff prediction seems wrong?
- Check that best delta is in reasonable range [0.5, 2.5]
- Verify porosity values match pore_analysis output
- Compare with Loeb classical (sanity check)

### Need to restart optimization?
- Results are saved incrementally
- Can resume by reducing `--n-calls` based on how many completed
- Or start fresh by deleting Results_Optimization_* directory

---

## References

1. Loeb, A. L. (1954). "Thermal conductivity: VIII, a theory of thermal conductivity of porous materials." *Journal of the American Ceramic Society*, 37(2), 96-99.

2. `run_keff_vs_delta.py` - 63 simulations across delta=[0.39, 3.0], p=[0.1, 0.2, 0.3]

3. `pore_analysis.py` - Automatic segmentation with stereological correction (factor=0.85)

4. `fit_correction_factor.py` - Sigmoidal fit with linear parameter dependencies

---

**Last updated:** 2025-03-17
**Status:** Phase 2 in progress (interconnected optimization running)
