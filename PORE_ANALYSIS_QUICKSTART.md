# PORE ANALYSIS - Quick Start Guide

## Overview
Analyze porosity in polished ceramic cross-section images (SEM, optical microscopy).
Classifies pores as **intragranular** (round, isolated) or **intergranular** (elongated, interconnected).

---

## Method 1: Batch Analysis (Recommended)

### Using `run_pore_analysis.sh`

**Quick start:**
```bash
bash run_pore_analysis.sh
```

### Adjustable Parameters

Open `run_pore_analysis.sh` and modify these parameters at the top:

| Parameter | Default | Description | How to adjust |
|-----------|---------|-------------|---------------|
| `IMAGE_DIR` | `Optimization_3D_structure/exp_img` | Directory with images | Change to your image folder path |
| `UM_PER_PIXEL` | `0.195` | Spatial resolution (μm/pixel) | Measure scale bar: `bar_length_μm / bar_pixels` |
| `CIRCULARITY_THR` | `0.50` | Classification threshold | 0.45-0.55 for ceramics, 0.40-0.50 for compacts |
| `MIN_AREA_UM2` | `1.5` | Minimum pore size (μm²) | Increase to ignore smaller pores |
| `ADAPTIVE` | `false` | Thresholding method | `true` for uneven illumination |
| `NO_WATERSHED` | `false` | Watershed segmentation | `true` to skip (faster, less accurate) |
| `OUTPUT_CSV` | `pore_analysis_results.csv` | Summary CSV filename | Any name you want |

### Adding More Images

In `run_pore_analysis.sh`, find the section:
```bash
run_analysis "${IMAGE_DIR}/connected_79.png"   "interconnected"
run_analysis "${IMAGE_DIR}/distributed_77.png" "distributed"
```

Add your images:
```bash
run_analysis "${IMAGE_DIR}/your_image.png"   "your_label"
```

### Output Files
- `<image>_pore_analysis.png` - 6-panel diagnostic plot
- `<image>_pores.csv` - Per-pore data (area, ECD, circularity, position, type)
- `pore_analysis_results.csv` - Summary table for all images

---

## Method 2: Single Image Analysis

### Command Line

**Basic usage:**
```bash
python project_root/core/pore_analysis.py sample.png 0.195
```

**With plots and CSV:**
```bash
python project_root/core/pore_analysis.py sample.png 0.195 --plot --export-csv
```

**Custom circularity threshold:**
```bash
python project_root/core/pore_analysis.py sample.png 0.195 0.55 --plot
```

**Adaptive thresholding (uneven illumination):**
```bash
python project_root/core/pore_analysis.py sample.png 0.195 --adaptive --plot
```

**Sensitivity analysis (find best circularity threshold):**
```bash
python project_root/core/pore_analysis.py sample.png 0.195 --sensitivity
```

### Python API

```python
from project_root.core.pore_analysis import analyze_porosity

result = analyze_porosity(
    image_path="sample.png",
    um_per_pixel=0.195,              # Required: spatial resolution
    circularity_threshold=0.50,       # Optional: classification boundary
    min_area_um2=2.0,                 # Optional: minimum pore size
    use_adaptive_threshold=False,     # Optional: True for uneven lighting
    use_watershed=True,               # Optional: separate touching pores
    export_csv=True,                  # Optional: save per-pore CSV
    show_plots=True                   # Optional: generate diagnostic plots
)

# Access results
print(f"Total porosity:   {result['p_total']:.2%}")
print(f"Intragranular:    {result['p_intra']:.2%} ({result['n_intra']} pores)")
print(f"Intergranular:    {result['p_inter']:.2%} ({result['n_inter']} pores)")
```

---

## How to Calculate `um_per_pixel`

1. Open your image in any image editor (ImageJ, GIMP, Photoshop, etc.)
2. Measure the scale bar length in **pixels**
3. Divide by the scale bar length in **micrometers**

**Example:**
- Scale bar shows "100 μm"
- Measured in pixels: 513 px
- Calculation: `100 / 513 = 0.195 μm/pixel`

---

## Understanding the Parameters

### Circularity Threshold (`CIRCULARITY_THR`)

**What it does:** Separates round (intragranular) from elongated (intergranular) pores.

**Formula:** `Circularity = 4π × Area / Perimeter²`
- 1.0 = perfect circle
- 0.0 = straight line

**Guidelines:**
- **Lower threshold (0.40-0.45):** More pores classified as intergranular
- **Higher threshold (0.55-0.60):** More pores classified as intragranular
- **Recommended:** 0.50 for sintered ceramics

**How to calibrate:**
1. Run sensitivity analysis: `python pore_analysis.py sample.png 0.195 --sensitivity`
2. Compare results with your SEM observations
3. Choose threshold that matches your visual assessment

### Adaptive Thresholding (`ADAPTIVE`)

**Use `ADAPTIVE=true` when:**
- Image has brightness gradients
- Uneven illumination across the field
- One side darker than the other

**Use `ADAPTIVE=false` when:**
- Uniform illumination
- Good contrast throughout
- Standard SEM/microscopy images

### Watershed Segmentation (`NO_WATERSHED`)

**Keep `NO_WATERSHED=false` for:**
- Accurate pore counting (separates touching pores)
- Quantitative analysis
- Final results

**Set `NO_WATERSHED=true` for:**
- Quick preview
- Faster processing
- When pores are well-separated

---

## Classification Logic

A pore is classified as **INTERGRANULAR** if:
1. Area > 20 μm² (very large → always intergranular)
   **OR**
2. Area ≥ 5 μm² AND Circularity < threshold (irregular shape)

Otherwise → **INTRAGRANULAR** (small, round pores)

**Advanced:** Adjust these in the shell script Python code:
- `area_inter_thr_um2=20.0` - size threshold for large pores
- `min_area_inter_um2=5.0` - minimum size for irregular pores

---

## Troubleshooting

### Problem: Scale bar detected as pore
**Solution:** Increase `SCALE_BAR_STRIP_FRACTION` in `pore_analysis.py` (default 0.12)

### Problem: Too many small pores (noise)
**Solution:** Increase `MIN_AREA_UM2` (default 1.5 μm²)

### Problem: Results vary with threshold
**Solution:** Run `--sensitivity` analysis to find stable threshold

### Problem: Touching pores counted as one
**Solution:** Ensure `NO_WATERSHED=false` (watershed enabled)

### Problem: Uneven detection across image
**Solution:** Set `ADAPTIVE=true` (use Sauvola thresholding)

---

## Output Interpretation

### Diagnostic Plot (6 panels)

1. **Original** - Grayscale input image
2. **Binarised** - Detected pores (red) and masked regions (green)
3. **Classification** - Blue = intragranular, Orange = intergranular
4. **Circularity distribution** - Scatter plot showing classification boundary
5. **Size distribution** - Histogram of pore sizes (ECD)
6. **Summary** - Text summary with all metrics

### CSV Outputs

**Per-image CSV** (`*_pores.csv`):
- `label` - Pore ID number
- `area_um2` - Pore area in μm²
- `ecd_um` - Equivalent circular diameter
- `circularity` - Shape factor (0-1)
- `solidity` - Convexity measure
- `centroid_x_px`, `centroid_y_px` - Position
- `type` - "intra" or "inter"

**Summary CSV** (`pore_analysis_results.csv`):
- Total, intragranular, and intergranular porosity
- Pore counts
- Mean sizes and circularities
- Processing parameters

---

## Best Practices

1. **Always check the diagnostic plots** - Verify detection quality
2. **Compare multiple circularity thresholds** - Use `--sensitivity`
3. **Validate against Archimedes density** - Check `p_total_binary` value
4. **Use consistent parameters** - Same settings for all samples in a study
5. **Document your settings** - Save parameters with your results

---

## Examples

### Example 1: Standard analysis
```bash
# Edit parameters in run_pore_analysis.sh
UM_PER_PIXEL=0.195
CIRCULARITY_THR=0.50
MIN_AREA_UM2=2.0

# Run
bash run_pore_analysis.sh
```

### Example 2: Find optimal threshold (Sensitivity Analysis)
```bash
# Run sensitivity analysis to test multiple thresholds
python project_root/core/pore_analysis.py sample.png 0.195 --sensitivity

# Or run the detailed example script
python example_sensitivity_analysis.py

# Review output, choose stable threshold, update CIRCULARITY_THR in run_pore_analysis.sh
```

**See [SENSITIVITY_ANALYSIS_GUIDE.md](SENSITIVITY_ANALYSIS_GUIDE.md) for complete tutorial**

### Example 3: Images with uneven lighting
```bash
# In run_pore_analysis.sh, set:
ADAPTIVE=true

bash run_pore_analysis.sh
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

**Install:**
```bash
pip install numpy scipy pillow scikit-image matplotlib
```
