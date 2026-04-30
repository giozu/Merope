"""
fit_correction_factor_joint.py
==============================
Joint sigmoidal fit of K_delta(p, delta*) across all porosity levels.

Why this exists
---------------
`fit_correction_factor.py` fits each porosity independently (one sigmoid per p).
With the current `keff_vs_delta.csv` that approach is *under-determined*:

  - p = 0.1 : delta starts at 0.1, K_delta is already ~0.91. The crack-dominated
              plateau (K_delta -> K_min) is never sampled, so K_min is unconstrained.
  - p = 0.2 : delta = 0.10 gives K_delta = 0.54. Only ONE point sits in the
              transition region. K_min could be anywhere <= 0.54; the optimizer
              hits the lower bound (0) and returns degenerate parameters.
  - p = 0.3 : delta starts at 0.15, K_delta = 0.44. Same issue, worse.

The independent fit therefore returns
    K_min(0.1) = 0.57,  K_min(0.2) = 0,  K_min(0.3) = 0
which is unphysical (K_min should be a smooth, decreasing function of p) and
makes the linear regression K_min(p) = a*p + b nonsensical.

Methodological fix
------------------
Fit ALL data simultaneously, with each parameter constrained to be linear in p:
    K_min(p) = a_kmin * p + b_kmin
    K_max(p) = a_kmax * p + b_kmax
    b(p)     = a_b    * p + b_b
    delta_c(p) = a_dc * p + b_dc

That collapses 12 free parameters (4 per porosity * 3 porosities) to 8 free
parameters total, with ~31 data points. The linear-in-p assumption is exactly
what the paper relies on anyway -- enforcing it during the fit is more honest
than fitting independently and then regressing afterwards.

Physical priors (encoded as bounds):
    K_min(p) in [0.05, 0.95] for p in [0.1, 0.3]   (positive floor; never reaches Loeb)
    K_max(p) in [0.85, 1.05] for p in [0.1, 0.3]   (recovery toward Loeb)
    b(p) negative and not too steep                 (sigmoid transition)
    delta_c(p) in [0, 1.0]                          (transition somewhere in sampled range)

Usage
-----
    cd ~/Merope
    python project_root/experiments/fit_correction_factor_joint.py \
        --csv Results_Keff_vs_Delta/keff_vs_delta.csv \
        --output-dir Results_Sigmoidal_Fit
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

ALPHA_LOEB = 1.37
K_MATRIX = 1.0


def loeb(p: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, K_MATRIX * (1.0 - ALPHA_LOEB * np.asarray(p)))


def k_delta(delta_star, k_min, k_max, b, delta_c):
    return k_min + (k_max - k_min) / (1.0 + np.exp(b * (delta_star - delta_c)))


def linear(p, a, b):
    return a * p + b


def joint_residuals(theta, p, delta_star, y_corr):
    """Residuals of K_delta = y_corr at every data point, parameters linear in p."""
    a_kmin, b_kmin, a_kmax, b_kmax, a_b, b_b, a_dc, b_dc = theta
    kmin = linear(p, a_kmin, b_kmin)
    kmax = linear(p, a_kmax, b_kmax)
    bcoef = linear(p, a_b, b_b)
    dc = linear(p, a_dc, b_dc)
    pred = k_delta(delta_star, kmin, kmax, bcoef, dc)
    return pred - y_corr


def fit_joint(df, p_anchor=(0.1, 0.2, 0.3)):
    """Returns: theta (8,), per-p parameter table, predictions per row."""
    df = df.sort_values(["Target_P", "Delta"]).reset_index(drop=True)
    p = df["Target_P"].values
    lag_r = df["Grain_R"].iloc[0] if "Grain_R" in df.columns else 1.0
    delta_star = df["Delta"].values / lag_r
    k_eff = df["K_eff"].values
    y_corr = k_eff / loeb(p)

    # Initial guess: physically-motivated priors.
    #   K_min(p) is the crack-dominated plateau. At p=0.1 the data only spans
    #   0.91-0.99, so K_min(0.1) is unidentified from data alone -- it must
    #   be constrained by a physical prior (it cannot drop to zero at low p).
    theta0 = np.array([
        -2.0, 0.95,   # K_min:  ~0.75 at p=0.1, ~0.35 at p=0.3
         0.0, 0.97,   # K_max:  near 1
        -30.0, -5.0,  # b:      negative, steeper at high p
         0.5, 0.05,   # delta_c: rises with p
    ])

    # Tight bounds keep the linear-in-p parametrisation in a physically
    # sensible regime over p in [0.05, 0.35]:
    #   K_min(p) in [0.10, 1.05]   (crack-dominated plateau, never < 0.10)
    #   K_max(p) in [0.85, 1.05]   (recovery toward Loeb)
    #   b(p)     < 0                (sigmoid with crack-dominated regime at low delta)
    #   delta_c(p) in [0, 1.0]
    lb = np.array([-2.5, 0.85,    # K_min:  K_min(0.3) >= 0.85 - 0.75 = 0.10
                   -0.5, 0.85,    # K_max
                   -200.0, -50.0,  # b
                   -1.0, -0.2])    # delta_c
    ub = np.array([ 0.0, 1.05,
                    0.2, 1.05,
                   -1.0,  0.0,
                    3.0,  0.5])

    res = least_squares(
        joint_residuals,
        theta0,
        bounds=(lb, ub),
        args=(p, delta_star, y_corr),
        method="trf",
        max_nfev=20000,
        loss="soft_l1",   # robust against any single outlier point
    )
    theta = res.x
    a_kmin, b_kmin, a_kmax, b_kmax, a_b, b_b, a_dc, b_dc = theta

    rows = []
    for pa in p_anchor:
        rows.append({
            "p": pa,
            "k_min": linear(pa, a_kmin, b_kmin),
            "k_max": linear(pa, a_kmax, b_kmax),
            "b": linear(pa, a_b, b_b),
            "delta_c": linear(pa, a_dc, b_dc),
        })
    per_p = pd.DataFrame(rows)
    coeffs = {
        "k_min": (a_kmin, b_kmin),
        "k_max": (a_kmax, b_kmax),
        "b": (a_b, b_b),
        "delta_c": (a_dc, b_dc),
    }
    return theta, per_p, coeffs, res


def plot_fits(df, coeffs, output_dir, lag_r=1.0):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    colors = {0.1: "steelblue", 0.2: "darkorange", 0.3: "forestgreen"}

    fig, ax = plt.subplots(figsize=(8, 6))
    delta_grid = np.linspace(0.01, 1.0, 200)
    for p_val, group in df.groupby("Target_P"):
        col = colors.get(p_val, "black")
        x = group["Delta"].values / lag_r
        y = group["K_eff"].values
        ax.scatter(x, y, color=col, alpha=0.6, label=f"data p={p_val}")
        kmin = linear(p_val, *coeffs["k_min"])
        kmax = linear(p_val, *coeffs["k_max"])
        bcoef = linear(p_val, *coeffs["b"])
        dc = linear(p_val, *coeffs["delta_c"])
        pred = loeb(np.full_like(delta_grid, p_val)) * k_delta(delta_grid, kmin, kmax, bcoef, dc)
        ax.plot(delta_grid, pred, color=col, linestyle="--", label=f"joint fit p={p_val}")

    ax.set_xlabel(r"$\delta^* = \delta / L_{grain}$")
    ax.set_ylabel(r"$K_{eff}$ (W/m·K)")
    ax.set_title("Joint sigmoidal fit (linear-in-p parameters)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    out = output_dir / "Sigmoidal_Fits.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return out


def plot_parameters(per_p, coeffs, output_dir):
    output_dir = Path(output_dir)
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()
    p_grid = np.linspace(0.05, 0.35, 50)
    for i, name in enumerate(["k_min", "k_max", "b", "delta_c"]):
        ax = axes[i]
        a, b0 = coeffs[name]
        ax.plot(p_grid, linear(p_grid, a, b0), "b--", alpha=0.7,
                label=f"{a:+.3f}p {b0:+.3f}")
        ax.scatter(per_p["p"], per_p[name], color="red")
        ax.set_xlabel("p")
        ax.set_ylabel(name)
        ax.set_title(f"{name}(p)")
        ax.legend(loc="best", fontsize=9)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    out = output_dir / "Parameters_vs_Porosity.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return out


def plot_contour(coeffs, output_dir):
    output_dir = Path(output_dir)
    p_grid = np.linspace(0.05, 0.30, 60)
    d_grid = np.linspace(0.05, 1.0, 60)
    P, D = np.meshgrid(p_grid, d_grid)
    Kmin = linear(P, *coeffs["k_min"])
    Kmax = linear(P, *coeffs["k_max"])
    B = linear(P, *coeffs["b"])
    DC = linear(P, *coeffs["delta_c"])
    K = loeb(P) * k_delta(D, Kmin, Kmax, B, DC)

    fig, ax = plt.subplots(figsize=(8, 6))
    cp = ax.contourf(P, D, K, levels=20, cmap="viridis")
    fig.colorbar(cp, label=r"$K_{eff}$")
    ax.set_xlabel("Porosity p")
    ax.set_ylabel(r"$\delta^* = \delta/L_{grain}$")
    ax.set_title("Joint-fit contour")
    out = output_dir / "K_eff_Contour.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, type=str)
    ap.add_argument("--output-dir", default="Results_Sigmoidal_Fit", type=str)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    theta, per_p, coeffs, res = fit_joint(df)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    lag_r = df["Grain_R"].iloc[0] if "Grain_R" in df.columns else 1.0

    print("\n=== Joint fit summary ===")
    print(f"  Cost          : {0.5 * np.sum(res.fun**2):.5f}")
    print(f"  Status        : {res.status} ({res.message})")
    print(f"  Nfev          : {res.nfev}")
    print(f"  Linear coeffs (param = a*p + b):")
    for name, (a, b0) in coeffs.items():
        print(f"    {name:8s}: a = {a:+.4f}, b = {b0:+.4f}")
    print("\n  Per-porosity parameters (anchor points):")
    print(per_p.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    per_p.to_csv(out / "fitted_parameters.csv", index=False)
    pd.DataFrame([{
        "param": k, "slope": v[0], "intercept": v[1]
    } for k, v in coeffs.items()]).to_csv(out / "linear_coeffs.csv", index=False)

    plot_fits(df, coeffs, out, lag_r=lag_r)
    plot_parameters(per_p, coeffs, out)
    plot_contour(coeffs, out)
    print(f"\n  Outputs written to {out}/")


if __name__ == "__main__":
    main()
