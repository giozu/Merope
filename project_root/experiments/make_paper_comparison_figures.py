"""
make_paper_comparison_figures.py
================================
Regenerate the three paper-side comparison figures from current artefacts:

  1. comparison_distributed_vs_interconnected.png
        Side-by-side bar chart: porosity composition (left) and
        K_eff predicted vs measured (right) for both modes.

  2. keff_vs_porosity_comparison.png
        Scatter of (p_total, K_eff) for both modes, overlaid on the
        calibrated Loeb baseline (alpha=1.37) and the joint sigmoidal
        prediction at the optimised delta_star.

  3. recovery_test_interconnected.png
        Side-by-side image comparison: synthetic interconnected target
        vs. the optimiser's best_slice. Headline visual of the
        known-truth recovery test.

Reads:
  Optimization_3D_structure/exp_img_synthetic/{ground_truth.json,
                                                synthetic_interconnected.png}
  Results_Optimization_{Distributed,Interconnected}/{summary.txt, best_slice.png}

Writes (paper Images/ tree):
  Images/Comparison/comparison_distributed_vs_interconnected.png
  Images/Comparison/keff_vs_porosity_comparison.png
  Images/Comparison/recovery_test_interconnected.png

Run
---
    cd ~/Merope
    source Env_Merope.sh
    export PYTHONPATH=$PYTHONPATH:./project_root
    python project_root/experiments/make_paper_comparison_figures.py
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ALPHA_LOEB = 1.37

PAPER_IMG_DIR = Path("/home/giovanni/research-manuscripts/Luzzi_et_al___MEROPE__2026/Images/Comparison")
PAPER_IMG_DIR.mkdir(parents=True, exist_ok=True)


def parse_summary(path: Path) -> dict:
    """Pull the parameters and AMITEX K_eff out of a Bayesian-opt summary."""
    text = path.read_text()
    out = {}
    for key in ("delta", "pore_phi", "pore_radius", "mean_radius", "std_radius"):
        m = re.search(rf"{key}\s*:\s*([-\d.]+)", text)
        if m:
            out[key] = float(m.group(1))
    m = re.search(r"Real porosity\s*:\s*([\d.]+)%", text)
    if m:
        out["real_porosity"] = float(m.group(1)) / 100.0
    for axis in ("Kxx", "Kyy", "Kzz", "Kmean"):
        m = re.search(rf"{axis}\s*:\s*([\d.]+)", text)
        if m:
            out[axis] = float(m.group(1))
    return out


def parse_prediction(path: Path) -> dict:
    """Pull predicted K_eff(s) from a keff_prediction.txt."""
    text = path.read_text()
    out = {}
    for tag, key in [("AMITEX-comparable", "k_eff_amitex"),
                     ("composite", "k_eff_composite"),
                     ("Classical Loeb", "k_eff_loeb_only")]:
        m = re.search(rf"{tag}.*?=\s*([\d.]+)", text, re.DOTALL)
        if m:
            out[key] = float(m.group(1))
    m = re.search(r"K_delta\s*=?\s*([\d.]+)", text)
    if m:
        out["k_delta"] = float(m.group(1))
    m = re.search(r"delta\*?\s*=\s*([\d.]+)", text)
    if m:
        out["delta_star"] = float(m.group(1))
    return out


def loeb(p, alpha=ALPHA_LOEB):
    return np.maximum(0.0, 1.0 - alpha * p)


# ---------------------------------------------------------------------------
# Load all the numbers
# ---------------------------------------------------------------------------
gt = json.loads(Path("Optimization_3D_structure/exp_img_synthetic/ground_truth.json").read_text())

# Distributed: from the BO recovery run (matched-form synthetic).
dist = parse_summary(Path("Results_Optimization_Distributed/summary.txt"))
dist_pred = parse_prediction(Path("Results_Optimization_Distributed/keff_prediction.txt"))

# Headline interconnected case: thin delta + mixed (boundary + intra) at
# delta_star = 0.10, in the morphology-controlled regime where K_delta < 1
# meaningfully. Run by run_thin_delta_mixed.py.
thin = parse_summary(Path("Results_ThinDelta_Mixed/summary.txt"))
# "thin" exposes Kxx/Kyy/Kzz/Kmean and target inter/intra phi from the script.
# Reconstruct realised porosity from the summary file directly.
thin_summary_text = Path("Results_ThinDelta_Mixed/summary.txt").read_text()
m = re.search(r"Realised pore fraction[^:]*:\s*([\d.]+)", thin_summary_text)
thin_p_total = float(m.group(1)) if m else np.nan
m = re.search(r"delta_star[^:]*:\s*([\d.]+)", thin_summary_text)
thin_delta_star = float(m.group(1)) if m else 0.10
m = re.search(r"inter_phi \(target\)\s*:\s*([\d.]+)", thin_summary_text)
thin_inter_target = float(m.group(1)) if m else np.nan
m = re.search(r"intra_phi \(target\)\s*:\s*([\d.]+)", thin_summary_text)
thin_intra_target = float(m.group(1)) if m else np.nan
# At delta=0.3 / r=0.40 the realised/raw clipping ratio is ~0.22, so the
# realised boundary fraction is approximately:
thin_p_b = 0.22 * thin_inter_target
thin_p_intra = thin_p_total - thin_p_b

# Recovery test (matched-form synthetic at delta=1.0). Used for the third
# figure only.
inter_match = parse_summary(Path("Results_Optimization_Interconnected/summary.txt"))
delta_gt = gt["interconnected"]["DELTA_CANONICAL"]
delta_recovered = inter_match.get("delta", 0.0)

# Synthetic ground-truth porosity for the distributed bar.
p_dist_gt = gt["distributed"]["REALISED_3D_P"]

print(f"Distributed (FFT):    K_AMITEX={dist['Kmean']:.4f} at p={dist['real_porosity']:.4f}")
print(f"Thin-delta mixed (FFT): K_AMITEX={thin['Kmean']:.4f} at p_total={thin_p_total:.4f} "
      f"(p_b~{thin_p_b:.3f}, p_intra~{thin_p_intra:.3f}), delta*={thin_delta_star:.3f}")
print(f"Recovery test (matched-form): delta_recovered={delta_recovered:.3f} (gt={delta_gt})")


# ---------------------------------------------------------------------------
# Figure 1: porosity composition + K_eff bars side-by-side
# ---------------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
modes = ["Distributed", "Interconnected\n(thin $\\delta^*$, mixed)"]
x = np.arange(len(modes))

# Porosity composition: distributed is all intra-granular; thin-delta mixed
# is split into boundary (clipped to GB band) + intra-granular.
p_dist_total = dist["real_porosity"]
ax1.bar(x[0], p_dist_total, width=0.55, color="steelblue",
        edgecolor="black", linewidth=0.5, label="Intra-granular")
# Stacked bar for the interconnected case.
ax1.bar(x[1], thin_p_b, width=0.55, color="darkorange",
        edgecolor="black", linewidth=0.5, label="Boundary (clipped to GB)")
ax1.bar(x[1], thin_p_intra, width=0.55, bottom=thin_p_b,
        color="steelblue", edgecolor="black", linewidth=0.5)
ax1.set_xticks(x)
ax1.set_xticklabels(modes)
ax1.set_ylabel("Porosity")
ax1.set_title("Porosity composition")
ax1.set_ylim(0, 0.30)
ax1.grid(True, axis="y", linestyle="--", alpha=0.4)
ax1.legend(loc="upper left", fontsize=9)

# K_eff comparison: AMITEX measurement vs Loeb baseline at total porosity.
# For distributed at p_total, Loeb is the natural prediction (no morphology
# correction). For the thin-delta mixed case, Loeb at p_total is the
# baseline against which the morphology penalty is measured.
k_amitex = [dist["Kmean"], thin["Kmean"]]
k_loeb = [float(loeb(p_dist_total)), float(loeb(thin_p_total))]
width = 0.35
ax2.bar(x - width/2, k_amitex, width, label="AMITEX (FFT)",
        color="steelblue", edgecolor="black", linewidth=0.5)
ax2.bar(x + width/2, k_loeb, width, label=r"Loeb baseline at $p_{\rm total}$",
        color="lightgray", edgecolor="black", linewidth=0.5)
for i, (m, p) in enumerate(zip(k_amitex, k_loeb)):
    ax2.text(i - width/2, m + 0.005, f"{m:.3f}", ha="center", va="bottom", fontsize=8)
    ax2.text(i + width/2, p + 0.005, f"{p:.3f}", ha="center", va="bottom", fontsize=8)
# Annotate the morphology penalty for the interconnected bar pair.
penalty = 1.0 - k_amitex[1] / k_loeb[1]
ax2.annotate(f"morph. penalty\n{penalty*100:.1f}%",
             xy=(x[1], k_amitex[1]),
             xytext=(x[1] - 0.45, k_loeb[1] + 0.10),
             fontsize=9, ha="left",
             arrowprops=dict(arrowstyle="->", lw=0.8, color="black"))
ax2.set_xticks(x)
ax2.set_xticklabels(modes)
ax2.set_ylabel(r"$K_{\rm eff}$ (W/m$\cdot$K)")
ax2.set_title("Effective conductivity: AMITEX vs Loeb baseline")
ax2.set_ylim(0, 1.0)
ax2.grid(True, axis="y", linestyle="--", alpha=0.4)
ax2.legend(loc="upper right", fontsize=9)

fig.tight_layout()
out1 = PAPER_IMG_DIR / "comparison_distributed_vs_interconnected.png"
fig.savefig(out1, dpi=200)
plt.close(fig)
print(f"  wrote {out1}")


# ---------------------------------------------------------------------------
# Figure 2: K_eff vs porosity scatter on Loeb baseline
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 5))
p_grid = np.linspace(0.0, 0.30, 200)
ax.plot(p_grid, loeb(p_grid), "k-", linewidth=1.5,
        label=r"Loeb baseline, $\alpha = 1.37$")

# Distributed point: lies essentially on Loeb -- demonstrates the calibration
# baseline holds for distributed morphology at this porosity.
ax.scatter([dist["real_porosity"]], [dist["Kmean"]],
           s=140, color="steelblue", edgecolor="black",
           zorder=5,
           label=f"Distributed (FFT, $p={dist['real_porosity']:.3f}$)")

# Thin-delta mixed point: clearly below Loeb -- demonstrates morphology
# penalty in the crack-dominated regime (delta* < delta_c).
ax.scatter([thin_p_total], [thin["Kmean"]],
           s=140, color="darkorange", edgecolor="black", marker="s",
           zorder=5,
           label=(f"Interconnected (FFT, $p_{{\\rm total}}={thin_p_total:.3f}$, "
                  f"$\\delta^* = {thin_delta_star:.2f}$)"))

# Visualise the morphology penalty with a vertical drop arrow.
loeb_at_thin = float(loeb(thin_p_total))
ax.annotate("",
            xy=(thin_p_total, thin["Kmean"]),
            xytext=(thin_p_total, loeb_at_thin),
            arrowprops=dict(arrowstyle="->", lw=1.4, color="darkorange"))
penalty_pct = 100 * (loeb_at_thin - thin["Kmean"]) / loeb_at_thin
ax.text(thin_p_total - 0.04, 0.5 * (loeb_at_thin + thin["Kmean"]),
        f"{penalty_pct:.0f}% drop\nbelow Loeb",
        fontsize=10, color="darkorange", va="center", ha="right")

ax.set_xlabel(r"Porosity $p$")
ax.set_ylabel(r"$K_{\rm eff}$ (W/m$\cdot$K)")
ax.set_xlim(0.0, 0.30)
ax.set_ylim(0.4, 1.05)
ax.grid(True, linestyle="--", alpha=0.4)
ax.legend(loc="lower left", fontsize=9)
ax.set_title("Effective thermal conductivity vs porosity")
fig.tight_layout()
out2 = PAPER_IMG_DIR / "keff_vs_porosity_comparison.png"
fig.savefig(out2, dpi=200)
plt.close(fig)
print(f"  wrote {out2}")


# ---------------------------------------------------------------------------
# Figure 3: recovery-test side-by-side images
# ---------------------------------------------------------------------------
synth_path = "Optimization_3D_structure/exp_img_synthetic/synthetic_interconnected.png"
best_path  = "Results_Optimization_Interconnected/best_slice.png"
img_synth = np.array(Image.open(synth_path).convert("L"))
img_best  = np.array(Image.open(best_path).convert("L"))

fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))
for ax, img, title in [
    (axes[0], img_synth,
     f"Synthetic target  (ground-truth $\\delta = {delta_gt}$, $r = 0.40$)"),
    (axes[1], img_best,
     f"Optimiser best slice  (recovered $\\delta = {delta_recovered:.3f}$, "
     f"$r = {inter_match.get('pore_radius', 0):.3f}$)"),
]:
    ax.imshow(img, cmap="gray", aspect="equal")
    ax.set_title(title, fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])

err_pct = 100 * abs(delta_recovered - delta_gt) / delta_gt
fig.suptitle(
    f"Known-truth recovery: $\\delta$ recovered within {err_pct:.1f}\\%",
    fontsize=13,
)
fig.tight_layout()
out3 = PAPER_IMG_DIR / "recovery_test_interconnected.png"
fig.savefig(out3, dpi=200)
plt.close(fig)
print(f"  wrote {out3}")
