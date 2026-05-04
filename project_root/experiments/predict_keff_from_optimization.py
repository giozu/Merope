#!/usr/bin/env python3
"""
Predict K_eff from optimization results using the correction factor model.

Usage:
    python predict_keff_from_optimization.py Results_Optimization_Interconnected
    python predict_keff_from_optimization.py Results_Optimization_Distributed
"""

import csv
import sys
from datetime import date
from pathlib import Path

import numpy as np

# --- Constants ---
LAG_R = 3.0  # Grain radius for delta normalization (must match run_keff_vs_delta.py)
DEFAULT_COEFFS_PATH = Path("Results_Sigmoidal_Fit/linear_coeffs.csv")


def load_sigmoid_coeffs(path=DEFAULT_COEFFS_PATH):
    """Load (slope, intercept) for k_min, k_max, b, delta_c from the joint-fit CSV.

    The CSV is produced by fit_correction_factor_joint.py and has columns
    ``param, slope, intercept``. Returns ``{param: [slope, intercept]}``.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Linear coefficients not found at {path}. "
            "Run fit_correction_factor_joint.py first to generate it."
        )
    coeffs = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            coeffs[row["param"]] = [float(row["slope"]), float(row["intercept"])]
    expected = {"k_min", "k_max", "b", "delta_c"}
    missing = expected - set(coeffs)
    if missing:
        raise ValueError(f"{path} is missing rows for: {sorted(missing)}")
    return coeffs


def loeb_model(p, k_matrix=1.0, alpha=1.37):
    """Classical Loeb model for distributed porosity."""
    return max(0.0, k_matrix * (1.0 - alpha * p))


def sigmoid_correction(delta, p, k_min_coeff, k_max_coeff, b_coeff, delta_c_coeff):
    """
    Sigmoidal correction factor K_δ(p, δ).

    Linear parameter dependencies:
    - k_min(p) = k_min_coeff[0] * p + k_min_coeff[1]
    - k_max(p) = k_max_coeff[0] * p + k_max_coeff[1]
    - b(p) = b_coeff[0] * p + b_coeff[1]
    - δ_c(p) = delta_c_coeff[0] * p + delta_c_coeff[1]
    """
    k_min = k_min_coeff[0] * p + k_min_coeff[1]
    k_max = k_max_coeff[0] * p + k_max_coeff[1]
    b = b_coeff[0] * p + b_coeff[1]
    delta_c = delta_c_coeff[0] * p + delta_c_coeff[1]

    K_delta = k_min + (k_max - k_min) / (1.0 + np.exp(b * (delta - delta_c)))
    return max(0.0, K_delta)


def predict_interconnected(p_boundary, p_intra, delta, coeffs=None):
    """
    Predict K_eff for interconnected morphology.

    Returns two K_eff values:
    - ``K_eff_amitex``: K_loeb(p_boundary) * K_delta. Matches the optimisation
      RVE (boundary phase only), so it is directly comparable to the AMITEX
      output in ``summary.txt``.
    - ``K_eff_composite``: K_eff_amitex * (1 - 1.37 * p_intra). Adds an extra
      Loeb factor for the experimentally observed intra-granular pores. This
      represents the *real* material but is NOT comparable to the AMITEX run
      in this study (which has no intra phase).
    """
    if coeffs is None:
        coeffs = load_sigmoid_coeffs()

    k_min_coeff = coeffs["k_min"]
    k_max_coeff = coeffs["k_max"]
    b_coeff = coeffs["b"]
    delta_c_coeff = coeffs["delta_c"]

    K_loeb_boundary = loeb_model(p_boundary)
    K_delta = sigmoid_correction(
        delta, p_boundary,
        k_min_coeff, k_max_coeff, b_coeff, delta_c_coeff
    )

    K_eff_amitex = K_loeb_boundary * K_delta
    K_eff_composite = K_eff_amitex * (1.0 - 1.37 * p_intra)

    return {
        "K_eff_amitex": K_eff_amitex,
        "K_eff_composite": K_eff_composite,
        "K_loeb_boundary": K_loeb_boundary,
        "K_delta": K_delta,
        "p_boundary": p_boundary,
        "p_intra": p_intra,
        "p_total": p_boundary + p_intra,
        "delta": delta,
    }


def predict_distributed(p_total):
    """
    Predict K_eff for distributed morphology (classical Loeb).

    Parameters
    ----------
    p_total : float
        Total porosity (all pores are isolated)

    Returns
    -------
    dict with K_eff prediction
    """
    K_eff = loeb_model(p_total)

    return {
        "K_eff": K_eff,
        "p_total": p_total,
        "model": "Classical Loeb (no correction needed)",
    }


def load_optimization_results(result_dir):
    """Load best parameters from optimization summary."""
    result_path = Path(result_dir)
    summary_file = result_path / "summary.txt"

    if not summary_file.exists():
        raise FileNotFoundError(f"Summary file not found: {summary_file}")

    params = {}
    with open(summary_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if "delta" in line.lower() and ":" in line:
                parts = line.split(":")
                if len(parts) == 2:
                    try:
                        params["delta"] = float(parts[1].strip())
                    except ValueError:
                        pass
            elif "pore_phi" in line.lower() and ":" in line:
                parts = line.split(":")
                if len(parts) == 2:
                    try:
                        params["pore_phi"] = float(parts[1].strip())
                    except ValueError:
                        pass
            elif "pore_radius" in line.lower() and ":" in line:
                parts = line.split(":")
                if len(parts) == 2:
                    try:
                        params["pore_radius"] = float(parts[1].strip())
                    except ValueError:
                        pass
            elif "mean_radius" in line.lower() and ":" in line:
                parts = line.split(":")
                if len(parts) == 2:
                    try:
                        params["mean_radius"] = float(parts[1].strip())
                    except ValueError:
                        pass
            elif "std_radius" in line.lower() and ":" in line:
                parts = line.split(":")
                if len(parts) == 2:
                    try:
                        params["std_radius"] = float(parts[1].strip())
                    except ValueError:
                        pass

    return params


def main():
    if len(sys.argv) < 2:
        print("Usage: python predict_keff_from_optimization.py <result_directory>")
        print("\nExamples:")
        print("  python predict_keff_from_optimization.py Results_Optimization_Interconnected")
        print("  python predict_keff_from_optimization.py Results_Optimization_Distributed")
        sys.exit(1)

    result_dir = sys.argv[1]
    result_path = Path(result_dir)

    # Detect mode from directory name
    if "interconnected" in result_dir.lower():
        mode = "interconnected"
    elif "distributed" in result_dir.lower():
        mode = "distributed"
    else:
        print("ERROR: Cannot determine mode from directory name.")
        print("Directory should contain 'interconnected' or 'distributed'")
        sys.exit(1)

    print("=" * 70)
    print(f"K_eff PREDICTION FROM OPTIMIZATION RESULTS")
    print("=" * 70)
    print(f"Mode: {mode.upper()}")
    print(f"Results directory: {result_dir}")
    print()

    # Load optimization results
    try:
        params = load_optimization_results(result_dir)
    except Exception as e:
        print(f"ERROR loading results: {e}")
        sys.exit(1)

    print("Best parameters:")
    for key, value in params.items():
        print(f"  {key:20s} = {value:.4f}")
    print()

    # Predict K_eff based on mode
    if mode == "interconnected":
        # From pore_analysis.py on connected_79.png (with stereological correction)
        p_boundary = 0.138  # 13.8% (62% of total)
        p_intra = 0.085     # 8.5% (38% of total)
        delta_abs = params.get("delta", 1.0)  # Absolute delta from optimization
        delta = delta_abs / LAG_R  # Normalize: delta* = delta / L_grain

        print("Input parameters:")
        print(f"  p_boundary  = {p_boundary:.1%} (interconnected)")
        print(f"  p_intra     = {p_intra:.1%} (isolated)")
        print(f"  p_total     = {p_boundary + p_intra:.1%}")
        print(f"  delta (abs) = {delta_abs:.3f}")
        print(f"  delta*      = {delta:.3f} (normalized by L_grain={LAG_R})")
        print()

        result = predict_interconnected(p_boundary, p_intra, delta)
        morphology_penalty_pct = 100.0 * (1.0 - result['K_delta'])

        print("-" * 70)
        print("PREDICTION BREAKDOWN:")
        print("-" * 70)
        print(f"1. Loeb model (boundary):       K_loeb  = {result['K_loeb_boundary']:.4f}")
        print(f"2. Morphology correction:       K_delta = {result['K_delta']:.4f}")
        print(f"3. AMITEX-comparable K_eff:     K_loeb * K_delta = {result['K_eff_amitex']:.4f}")
        print(f"   (boundary phase only — matches optimisation RVE)")
        print(f"4. Composite (with intra Loeb): K = {result['K_eff_composite']:.4f}")
        print(f"   ( * (1 - 1.37 * p_intra={p_intra:.3f}); NOT comparable to AMITEX run)")
        print("-" * 70)
        print(f"\n✓ K_eff (AMITEX-comparable)  = {result['K_eff_amitex']:.4f} W/m·K")
        print(f"  K_eff (composite, w/ intra) = {result['K_eff_composite']:.4f} W/m·K")
        print()

        print(f"Morphology penalty at delta*={delta:.3f}, p_b={p_boundary:.1%}:")
        print(f"  K_loeb (no correction)  = {result['K_loeb_boundary']:.4f}")
        print(f"  K_eff (with K_delta)    = {result['K_eff_amitex']:.4f}")
        print(f"  Penalty (1 - K_delta)   = {morphology_penalty_pct:.1f}%")
        print(f"  (At delta* >> delta_c the sigmoid saturates and the penalty vanishes;")
        print(f"   sub-percolation delta* would be needed for the historical 40% figure.)")

    else:  # distributed
        # From pore_analysis.py on distributed_77.png
        p_total = 0.227  # 22.7% (100% intra, 0% boundary)

        print("Input parameters:")
        print(f"  p_total     = {p_total:.1%} (all isolated)")
        print()

        result = predict_distributed(p_total)

        print("-" * 70)
        print("PREDICTION:")
        print("-" * 70)
        print(f"Classical Loeb model: K = K_matrix × (1 - 1.37 × p)")
        print(f"                      K = 1.0 × (1 - 1.37 × {p_total:.3f})")
        print(f"                      K = {result['K_eff']:.4f}")
        print("-" * 70)
        print(f"\n✓ PREDICTED K_eff = {result['K_eff']:.4f} W/m·K")
        print()

    print("=" * 70)

    # Save results
    output_file = result_path / "keff_prediction.txt"
    with open(output_file, 'w') as f:
        f.write("K_eff PREDICTION\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Mode: {mode}\n")
        f.write(f"Date: {date.today().isoformat()}\n\n")

        if mode == "interconnected":
            f.write(f"Optimized delta: {delta_abs:.3f} (absolute)\n")
            f.write(f"Normalized delta*: {delta:.3f} (delta/L_grain)\n")
            f.write(f"Boundary porosity: {p_boundary:.1%}\n")
            f.write(f"Intra porosity: {p_intra:.1%}\n")
            f.write(f"Total porosity: {p_boundary + p_intra:.1%}\n\n")
            f.write(f"K_eff (AMITEX-comparable)   = {result['K_eff_amitex']:.4f} W/m·K\n")
            f.write(f"K_eff (composite, w/ intra) = {result['K_eff_composite']:.4f} W/m·K\n\n")
            f.write(f"Breakdown:\n")
            f.write(f"  K_Loeb(p_boundary={p_boundary:.3f}) = {result['K_loeb_boundary']:.4f}\n")
            f.write(f"  K_delta(p_boundary, delta*={delta:.3f}) = {result['K_delta']:.4f}\n")
            f.write(f"  K_eff_amitex   = K_Loeb * K_delta = {result['K_eff_amitex']:.4f}\n")
            f.write(f"  K_eff_composite = K_eff_amitex * (1 - 1.37 * p_intra={p_intra:.3f}) = {result['K_eff_composite']:.4f}\n\n")
            f.write(f"Morphology penalty (1 - K_delta): {100.0 * (1.0 - result['K_delta']):.1f}%\n")
        else:
            f.write(f"Total porosity: {p_total:.1%}\n\n")
            f.write(f"K_eff = {result['K_eff']:.4f} W/m·K\n")
            f.write(f"(Classical Loeb model)\n")

    print(f"✓ Results saved to: {output_file}")
    print()


if __name__ == "__main__":
    main()
