#!/usr/bin/env python3
"""
Predict K_eff from optimization results using the correction factor model.

Usage:
    python predict_keff_from_optimization.py Results_Optimization_Interconnected
    python predict_keff_from_optimization.py Results_Optimization_Distributed
"""

import sys
import numpy as np
from pathlib import Path

# --- Constants ---
LAG_R = 3.0  # Grain radius for delta normalization (must match run_keff_vs_delta.py)


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


def predict_interconnected(p_boundary, p_intra, delta):
    """
    Predict K_eff for interconnected morphology.

    Parameters
    ----------
    p_boundary : float
        Boundary/interconnected porosity (from optimization)
    p_intra : float
        Intra-granular porosity (from pore_analysis)
    delta : float
        Grain boundary thickness (optimized parameter)

    Returns
    -------
    dict with K_eff prediction and breakdown
    """
    # Fitted parameters from run_keff_vs_delta.py
    # k_min(p) = -4.74*p + 1.26
    # k_max(p) = -0.15*p + 1.00
    # b(p) = -1.98*p - 5.58
    # δ_c(p) = -1.08*p + 0.64

    k_min_coeff = [-4.74, 1.26]
    k_max_coeff = [-0.15, 1.00]
    b_coeff = [-1.98, -5.58]
    delta_c_coeff = [-1.08, 0.64]

    # Step 1: Loeb model for boundary porosity
    K_loeb_boundary = loeb_model(p_boundary)

    # Step 2: Correction factor for morphology
    K_delta = sigmoid_correction(
        delta, p_boundary,
        k_min_coeff, k_max_coeff, b_coeff, delta_c_coeff
    )

    # Step 3: Boundary contribution
    K_boundary_contribution = K_loeb_boundary * K_delta

    # Step 4: Add correction for intra-pores (simple Loeb)
    # Intra pores are isolated, so use classical Loeb correction
    K_eff = K_boundary_contribution * (1.0 - 1.37 * p_intra)

    return {
        "K_eff": K_eff,
        "K_loeb_boundary": K_loeb_boundary,
        "K_delta": K_delta,
        "K_boundary_contribution": K_boundary_contribution,
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

        print("-" * 70)
        print("PREDICTION BREAKDOWN:")
        print("-" * 70)
        print(f"1. Loeb model (boundary):      K = {result['K_loeb_boundary']:.4f}")
        print(f"2. Morphology correction:      K_δ = {result['K_delta']:.4f}")
        print(f"3. Boundary contribution:      K = {result['K_boundary_contribution']:.4f}")
        print(f"4. Intra correction (×{1 - 1.37*p_intra:.3f}):   K = {result['K_eff']:.4f}")
        print("-" * 70)
        print(f"\n✓ PREDICTED K_eff = {result['K_eff']:.4f} W/m·K")
        print()

        # Compare with distributed (same total porosity)
        p_total = p_boundary + p_intra
        K_distributed = loeb_model(p_total)
        reduction = (K_distributed - result['K_eff']) / K_distributed * 100

        print(f"Comparison with distributed morphology (p={p_total:.1%}):")
        print(f"  K_distributed = {K_distributed:.4f} W/m·K (Loeb classical)")
        print(f"  K_interconnected = {result['K_eff']:.4f} W/m·K (with correction)")
        print(f"  Reduction due to morphology: {reduction:.1f}%")

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
        f.write(f"Date: 2025-03-17\n\n")

        if mode == "interconnected":
            f.write(f"Optimized delta: {delta_abs:.3f} (absolute)\n")
            f.write(f"Normalized delta*: {delta:.3f} (delta/L_grain)\n")
            f.write(f"Boundary porosity: {p_boundary:.1%}\n")
            f.write(f"Intra porosity: {p_intra:.1%}\n")
            f.write(f"Total porosity: {p_boundary + p_intra:.1%}\n\n")
            f.write(f"K_eff = {result['K_eff']:.4f} W/m·K\n\n")
            f.write(f"Breakdown:\n")
            f.write(f"  K_Loeb(boundary) = {result['K_loeb_boundary']:.4f}\n")
            f.write(f"  K_delta = {result['K_delta']:.4f}\n")
            f.write(f"  K_boundary = {result['K_boundary_contribution']:.4f}\n")
            f.write(f"  K_eff (with intra) = {result['K_eff']:.4f}\n")
        else:
            f.write(f"Total porosity: {p_total:.1%}\n\n")
            f.write(f"K_eff = {result['K_eff']:.4f} W/m·K\n")
            f.write(f"(Classical Loeb model)\n")

    print(f"✓ Results saved to: {output_file}")
    print()


if __name__ == "__main__":
    main()
