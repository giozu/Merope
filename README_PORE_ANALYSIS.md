# Pore Analysis Scripts - Documentation

## Overview
Complete toolkit for analyzing porosity in polished ceramic cross-section images (SEM, optical microscopy). Automatically classifies pores as **intragranular** (round, isolated) or **intergranular** (elongated, interconnected).

---

## Quick Start

### 1. Run Batch Analysis
```bash
bash run_pore_analysis.sh
```

### 2. Analyze Single Image
```bash
python project_root/core/pore_analysis.py sample.png 0.195 --plot --export-csv
```

### 3. Find Optimal Threshold
```bash
python project_root/core/pore_analysis.py sample.png 0.195 --sensitivity
```

---

## File Structure

```
Merope/
├── run_pore_analysis.sh              # Batch processing script (MAIN SCRIPT)
├── project_root/core/pore_analysis.py # Python analysis library
├── example_sensitivity_analysis.py    # Example script with 5 use cases
├── PORE_ANALYSIS_QUICKSTART.md       # User guide (START HERE)
├── SENSITIVITY_ANALYSIS_GUIDE.md     # Detailed threshold calibration guide
└── README_PORE_ANALYSIS.md           # This file
```

---

## Documentation Index

### For Beginners
1. **[PORE_ANALYSIS_QUICKSTART.md](PORE_ANALYSIS_QUICKSTART.md)** ← START HERE
   - Complete user guide
   - Parameter explanations
   - Examples and troubleshooting

### For Calibration
2. **[SENSITIVITY_ANALYSIS_GUIDE.md](SENSITIVITY_ANALYSIS_GUIDE.md)**
   - How to choose circularity threshold
   - Interpretation guide
   - Step-by-step workflow

### For Python Users
3. **[example_sensitivity_analysis.py](example_sensitivity_analysis.py)**
   - 5 working examples
   - Python API usage
   - Batch processing templates

### Scripts with Inline Documentation
4. **[run_pore_analysis.sh](run_pore_analysis.sh)**
   - Batch processing script
   - All parameters explained
   - Ready to customize

5. **[project_root/core/pore_analysis.py](project_root/core/pore_analysis.py)**
   - Python library
   - Detailed docstrings
   - Can be used as CLI or imported

---

## Common Use Cases

### Use Case 1: Batch Process Multiple Images
**Edit** `run_pore_analysis.sh` parameters:
```bash
IMAGE_DIR="your/image/folder"
UM_PER_PIXEL=0.195          # Your scale
CIRCULARITY_THR=0.50        # Classification threshold
MIN_AREA_UM2=1.5           # Minimum pore size
```

**Run:**
```bash
bash run_pore_analysis.sh
```

**Output:**
- `pore_analysis_results.csv` - Summary table
- `*_pore_analysis.png` - Diagnostic plots for each image
- `*_pores.csv` - Per-pore data for each image

---

### Use Case 2: Find Best Threshold for Your Material

**Step 1:** Run sensitivity analysis
```bash
python project_root/core/pore_analysis.py sample.png 0.195 --sensitivity
```

**Step 2:** Review output table
```
  circ_thr    p_total    p_intra    p_inter  n_intra  n_inter
      0.40     24.14%      1.74%     22.40%       20       34
      0.50     24.14%      0.98%     23.16%       11       43  ← Choose this
      0.60     24.14%      0.56%     23.58%        7       47
```

**Step 3:** Update `run_pore_analysis.sh`
```bash
CIRCULARITY_THR=0.50  # ← Your chosen value
```

**Step 4:** Apply to all samples
```bash
bash run_pore_analysis.sh
```

**See:** [SENSITIVITY_ANALYSIS_GUIDE.md](SENSITIVITY_ANALYSIS_GUIDE.md) for details

---

### Use Case 3: Single Image Analysis with Custom Parameters

```bash
python project_root/core/pore_analysis.py \
    sample.png \
    0.195 \
    0.55 \
    --adaptive \
    --plot \
    --export-csv
```

Where:
- `sample.png` - Image file
- `0.195` - Scale (μm/pixel)
- `0.55` - Circularity threshold
- `--adaptive` - Use adaptive thresholding (for uneven lighting)
- `--plot` - Generate diagnostic plot
- `--export-csv` - Save per-pore CSV

---

### Use Case 4: Python API Integration

```python
from project_root.core.pore_analysis import analyze_porosity

result = analyze_porosity(
    image_path="sample.png",
    um_per_pixel=0.195,
    circularity_threshold=0.50,
    min_area_um2=2.0,
    use_adaptive_threshold=False,
    use_watershed=True,
    export_csv=True,
    show_plots=True
)

# Access results
print(f"Total porosity:   {result['p_total']:.2%}")
print(f"Intragranular:    {result['p_intra']:.2%}")
print(f"Intergranular:    {result['p_inter']:.2%}")
print(f"Number of pores:  {result['n_total']}")

# Access per-pore data
for pore in result['pore_table']:
    print(f"Pore {pore['label']}: {pore['area_um2']:.2f} μm², "
          f"circularity={pore['circularity']:.3f}, type={pore['type']}")
```

---

## Key Parameters Explained

| Parameter | Default | Description | How to adjust |
|-----------|---------|-------------|---------------|
| **UM_PER_PIXEL** | 0.195 | Spatial resolution | `scale_bar_μm / scale_bar_pixels` |
| **CIRCULARITY_THR** | 0.50 | Classification threshold | Use `--sensitivity` to calibrate |
| **MIN_AREA_UM2** | 1.5 | Noise filter (μm²) | Increase to ignore smaller pores |
| **ADAPTIVE** | false | Thresholding method | `true` for uneven illumination |
| **NO_WATERSHED** | false | Watershed segmentation | `true` to skip (faster, less accurate) |

**Detailed explanations:** See inline comments in [run_pore_analysis.sh](run_pore_analysis.sh)

---

## Output Files

### 1. Summary CSV (`pore_analysis_results.csv`)
One row per image with:
- Total, intragranular, intergranular porosity
- Pore counts
- Mean sizes and circularities
- Processing parameters

### 2. Per-Image Plots (`*_pore_analysis.png`)
6-panel diagnostic plot:
1. Original grayscale image
2. Binarized (detected pores)
3. **Classification map** (blue=intra, orange=inter)
4. Circularity distribution
5. Size distribution histogram
6. Summary statistics

### 3. Per-Pore CSV (`*_pores.csv`)
One row per pore with:
- `area_um2` - Pore area
- `ecd_um` - Equivalent circular diameter
- `circularity` - Shape factor (0-1)
- `solidity` - Convexity
- `centroid_x_px`, `centroid_y_px` - Position
- `type` - "intra" or "inter"

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Scale bar detected as pore | Increase `SCALE_BAR_STRIP_FRACTION` in Python script |
| Too many small pores (noise) | Increase `MIN_AREA_UM2` |
| Results vary with threshold | Run `--sensitivity` to find stable value |
| Touching pores counted as one | Ensure `NO_WATERSHED=false` |
| Uneven detection across image | Set `ADAPTIVE=true` |

**More help:** See Troubleshooting section in [PORE_ANALYSIS_QUICKSTART.md](PORE_ANALYSIS_QUICKSTART.md)

---

## Workflow Summary

```
┌─────────────────────────────────────────────────────────────┐
│ 1. CALIBRATION (Do once per material type)                 │
│    python pore_analysis.py sample.png 0.195 --sensitivity   │
│    → Choose optimal circularity threshold                   │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. CONFIGURATION                                            │
│    Edit run_pore_analysis.sh:                               │
│    - Set CIRCULARITY_THR to chosen value                    │
│    - Set UM_PER_PIXEL for your images                       │
│    - Configure other parameters if needed                   │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│ 3. BATCH PROCESSING                                         │
│    bash run_pore_analysis.sh                                │
│    → Analyzes all images                                    │
│    → Generates plots and CSV files                          │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│ 4. VALIDATION                                               │
│    - Check diagnostic plots                                 │
│    - Verify classification matches SEM observations         │
│    - Compare p_total_binary with Archimedes density         │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│ 5. ANALYSIS & REPORTING                                     │
│    - Use pore_analysis_results.csv for statistics          │
│    - Include diagnostic plots in supplementary materials    │
│    - Report threshold and parameters in methods section     │
└─────────────────────────────────────────────────────────────┘
```

---

## Citation & Methods Section Template

When publishing results, include in your methods section:

> Porosity was analyzed from polished cross-section images using custom Python
> scripts based on scikit-image. Images were thresholded using Otsu's method,
> and touching pores were separated using watershed segmentation. Pores were
> classified as intragranular or intergranular based on circularity
> (4π·Area/Perimeter²) with a threshold of 0.50, calibrated by sensitivity
> analysis to match SEM observations. Pores smaller than 1.5 μm² were excluded
> as noise. Spatial resolution was 0.195 μm/pixel.

**Include in supplementary:**
- Sensitivity analysis table
- Example diagnostic plots
- Parameter settings used

---

## Examples Provided

### [example_sensitivity_analysis.py](example_sensitivity_analysis.py)
Demonstrates 5 use cases:
1. Basic sensitivity analysis (default thresholds)
2. Custom threshold range (fine-tuning)
3. Sensitivity with adaptive thresholding
4. Full analysis after choosing threshold
5. Batch sensitivity analysis (multiple images)

**Run it:**
```bash
python example_sensitivity_analysis.py
```

---

## Dependencies

**Required:**
- Python 3.6+
- numpy
- scipy
- PIL/Pillow

**Optional (recommended):**
- scikit-image (for watershed & adaptive thresholding)
- matplotlib (for diagnostic plots)

**Install all:**
```bash
pip install numpy scipy pillow scikit-image matplotlib
```

---

## Getting Help

1. **Quick reference:** Read inline comments in `run_pore_analysis.sh`
2. **User guide:** [PORE_ANALYSIS_QUICKSTART.md](PORE_ANALYSIS_QUICKSTART.md)
3. **Threshold calibration:** [SENSITIVITY_ANALYSIS_GUIDE.md](SENSITIVITY_ANALYSIS_GUIDE.md)
4. **CLI help:** `python project_root/core/pore_analysis.py` (no arguments)
5. **Examples:** Run `python example_sensitivity_analysis.py`

---

## Summary

✓ **Comprehensive** - Fully documented with guides, examples, and inline comments
✓ **Flexible** - Works as batch script, CLI tool, or Python library
✓ **Validated** - Sensitivity analysis for threshold calibration
✓ **Production-ready** - Used for ceramic microstructure characterization
✓ **Well-tested** - Multiple examples and edge cases covered

**Start here:** [PORE_ANALYSIS_QUICKSTART.md](PORE_ANALYSIS_QUICKSTART.md)
