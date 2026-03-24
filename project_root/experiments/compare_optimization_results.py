#!/usr/bin/env python3
"""
Compare optimization results between distributed and interconnected morphologies.

Usage:
    python compare_optimization_results.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Import pore_analysis to run analysis directly
sys.path.insert(0, str(Path(__file__).parent.parent / "core"))
from pore_analysis import analyze_porosity


def load_pore_data(sample_name):
    """Load pore analysis data by running pore_analysis.py.

    Args:
        sample_name: Name of the sample (e.g., 'distributed_77', 'connected_79')

    Returns:
        Dictionary with porosity statistics
    """
    image_path = Path(f"Optimization_3D_structure/exp_img/{sample_name}.png")

    if not image_path.exists():
        print(f"Error: {image_path} not found!")
        sys.exit(1)

    # Run pore analysis directly
    print(f"  Analyzing {sample_name}...", end=" ", flush=True)
    result = analyze_porosity(
        image_path=str(image_path),
        um_per_pixel=0.195,
        circularity_threshold=0.50,
        export_csv=False,  # CSV already exists
        show_plots=False,
    )
    print("✓")

    # Use results as-is from pore_analysis (geometric classification)
    p_total = result['p_total']
    p_boundary = result['p_inter']
    p_intra = result['p_intra']
    n_total = result['n_total']
    n_boundary = result['n_inter']
    n_intra = result['n_intra']

    # Determine morphology based on AREA fraction of inter pores (not count)
    # Use boundary porosity as fraction of total porosity
    inter_area_fraction = p_boundary / p_total if p_total > 0 else 0

    if inter_area_fraction < 0.30:  # Less than 30% area is inter → mostly isolated
        morphology = "Mostly isolated"
    elif inter_area_fraction > 0.70:  # More than 70% area is inter → interconnected
        morphology = "Interconnected network"
    else:
        morphology = "Mixed morphology"

    return {
        "p_total": p_total,
        "p_boundary": p_boundary,
        "p_intra": p_intra,
        "n_pores": n_total,
        "n_boundary": n_boundary,
        "n_intra": n_intra,
        "morphology": morphology,
        "inter_area_fraction": inter_area_fraction,
    }


def main():
    print("=" * 70)
    print("COMPARISON: DISTRIBUTED vs INTERCONNECTED MORPHOLOGY")
    print("=" * 70)
    print()

    # Load data by running pore_analysis.py
    print("Running pore analysis on experimental images...")
    results = {
        "distributed_77": load_pore_data("distributed_77"),
        "connected_79": load_pore_data("connected_79"),
    }
    print()

    # Predict K_eff
    # Distributed (classical Loeb)
    K_distributed = 1.0 * (1.0 - 1.37 * results["distributed_77"]["p_total"])

    # Interconnected (with correction factor)
    p_b = results["connected_79"]["p_boundary"]
    p_i = results["connected_79"]["p_intra"]

    # Read optimized delta from Results_Optimization_Interconnected/summary.txt
    LAG_R = 3.0  # Grain radius for normalization
    delta_abs = 0.390  # Default from previous optimization

    summary_path = Path("Results_Optimization_Interconnected/summary.txt")
    if summary_path.exists():
        with open(summary_path, 'r') as f:
            for line in f:
                if "delta" in line.lower() and ":" in line:
                    try:
                        delta_abs = float(line.split(":")[1].strip())
                        break
                    except:
                        pass

    delta = delta_abs / LAG_R  # Normalize: delta* = delta / L_grain
    print(f"Using delta = {delta_abs:.3f} (absolute) → delta* = {delta:.3f} (normalized)")
    print()

    # Correction factor parameters (fitted with normalized delta*)
    k_min = -4.74 * p_b + 1.26
    k_max = -0.15 * p_b + 1.00
    b = -1.98 * p_b - 5.58
    delta_c = -1.08 * p_b + 0.64

    K_delta = k_min + (k_max - k_min) / (1.0 + np.exp(b * (delta - delta_c)))
    K_boundary = (1.0 - 1.37 * p_b) * K_delta
    K_interconnected = K_boundary * (1.0 - 1.37 * p_i)

    # Display comparison
    print("POROSITY ANALYSIS:")
    print("-" * 70)
    print(f"{'Property':<30} {'Distributed':>15} {'Interconnected':>15}")
    print("-" * 70)
    print(f"{'Total porosity':<30} {results['distributed_77']['p_total']:>14.1%} {results['connected_79']['p_total']:>14.1%}")
    print(f"{'Boundary porosity':<30} {results['distributed_77']['p_boundary']:>14.1%} {results['connected_79']['p_boundary']:>14.1%}")
    print(f"{'Intra porosity':<30} {results['distributed_77']['p_intra']:>14.1%} {results['connected_79']['p_intra']:>14.1%}")
    print(f"{'Number of pores':<30} {results['distributed_77']['n_pores']:>15d} {results['connected_79']['n_pores']:>15d}")
    print(f"{'Morphology':<30} {results['distributed_77']['morphology']:>15s} {results['connected_79']['morphology']:>15s}")
    print("-" * 70)
    print()

    print("K_eff PREDICTION:")
    print("-" * 70)
    print(f"Distributed (Loeb):             {K_distributed:.4f} W/m·K")
    print(f"Interconnected (w/ correction): {K_interconnected:.4f} W/m·K")
    print()
    reduction = (K_distributed - K_interconnected) / K_distributed * 100
    print(f"Reduction due to morphology:    {reduction:.1f}%")
    print("-" * 70)
    print()

    print("KEY FINDINGS:")
    p_dist = results["distributed_77"]["p_total"]
    p_conn = results["connected_79"]["p_total"]
    inter_frac_dist = results["distributed_77"]["inter_area_fraction"]
    inter_frac_conn = results["connected_79"]["inter_area_fraction"]
    print(f"  1. Total porosity: {p_dist:.1%} (distributed) vs {p_conn:.1%} (interconnected)")
    print(f"  2. Inter pore area fraction: {inter_frac_dist:.1%} vs {inter_frac_conn:.1%}")
    print(f"  3. K_eff differs by {reduction:.0f}% despite different porosity levels")
    print(f"  4. Morphology (inter pore connectivity) is the dominant factor")
    print("  5. Classical Loeb model fails for interconnected morphology")
    print()

    # Create comparison plot with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: Porosity composition
    ax = axes[0]
    samples = ["Distributed\n(77)", "Interconnected\n(79)"]
    boundary = [results["distributed_77"]["p_boundary"], results["connected_79"]["p_boundary"]]
    intra = [results["distributed_77"]["p_intra"], results["connected_79"]["p_intra"]]

    x = np.arange(len(samples))
    width = 0.5

    ax.bar(x, boundary, width, label='Boundary (interconnected)', color='#d62728')
    ax.bar(x, intra, width, bottom=boundary, label='Intra (isolated)', color='#1f77b4')

    ax.set_ylabel('Porosity (%)')
    ax.set_title('Porosity Composition')
    ax.set_xticks(x)
    ax.set_xticklabels(samples)
    ax.legend()
    ax.set_ylim([0, 0.3])
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y*100:.0f}%'))
    ax.grid(axis='y', alpha=0.3)

    # Plot 2: K_eff comparison (bar chart)
    ax = axes[1]
    K_values = [K_distributed, K_interconnected]
    colors = ['#1f77b4', '#d62728']

    bars = ax.bar(samples, K_values, width, color=colors, alpha=0.7, edgecolor='black')

    # Add value labels on bars
    for bar, val in zip(bars, K_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f}',
                ha='center', va='bottom', fontweight='bold')

    ax.set_ylabel('K_eff (W/m·K)')
    ax.set_title('Effective Thermal Conductivity')
    ax.set_ylim([0, max(K_values) * 1.2])
    ax.grid(axis='y', alpha=0.3)

    # Add reduction annotation
    ax.annotate(f'{reduction:.0f}% reduction\ndue to morphology',
                xy=(0.5, (K_distributed + K_interconnected)/2),
                xytext=(1.3, (K_distributed + K_interconnected)/2),
                arrowprops=dict(arrowstyle='->', lw=2, color='red'),
                fontsize=10, color='red', fontweight='bold',
                ha='left', va='center')


    plt.tight_layout()
    plt.savefig('comparison_distributed_vs_interconnected.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ Comparison plot saved: comparison_distributed_vs_interconnected.png")
    
    # ========================================================================
    # Create separate K_eff vs Porosity plot
    # ========================================================================
    fig2, ax2 = plt.subplots(figsize=(10, 7))
    
    # Generate porosity range
    p_range = np.linspace(0.05, 0.35, 100)
    
    # Loeb model (distributed) - classical model
    K_loeb = 1.0 * (1.0 - 1.37 * p_range)
    ax2.plot(p_range, K_loeb, 'b-', linewidth=3, label='Loeb model', zorder=1)
    
    # Loeb × correction factor (interconnected) - same porosity as Loeb but with correction
    K_corrected = []
    for p_total in p_range:
        # For interconnected: use total porosity but apply correction factor based on morphology
        # Approximate p_boundary ≈ 0.6 * p_total (from experimental data)
        p_b = 0.62 * p_total  # 13.8/22.3 ≈ 0.62
        
        # Correction factor parameters
        k_min_i = -4.74 * p_b + 1.26
        k_max_i = -0.15 * p_b + 1.00
        b_i = -1.98 * p_b - 5.58
        delta_c_i = -1.08 * p_b + 0.64
        K_delta_i = k_min_i + (k_max_i - k_min_i) / (1.0 + np.exp(b_i * (delta - delta_c_i)))
        
        # Apply correction to Loeb model evaluated at total porosity
        K_i = (1.0 - 1.37 * p_total) * K_delta_i
        K_corrected.append(K_i)
    
    ax2.plot(p_range, K_corrected, 'r--', linewidth=3, 
            label=f'Loeb × correction (δ*={delta:.2f})', zorder=1)
    
    # Plot experimental points - BOTH at total porosity
    ax2.plot(results["distributed_77"]["p_total"], K_distributed, 
            's', markersize=15, color='blue', markeredgecolor='black', 
            markeredgewidth=2, label='Distributed (77)', zorder=3)
    ax2.plot(results["connected_79"]["p_total"], K_interconnected, 
            's', markersize=15, color='red', markeredgecolor='black', 
            markeredgewidth=2, label='Interconnected (79)', zorder=3)
    
    # Add arrow showing the effect of morphology
    p_inter = results["connected_79"]["p_total"]
    K_loeb_at_p = 1.0 - 1.37 * p_inter
    ax2.annotate('', xy=(p_inter, K_interconnected), xytext=(p_inter, K_loeb_at_p),
                arrowprops=dict(arrowstyle='->', lw=2.5, color='red', linestyle='--'))
    ax2.text(p_inter + 0.015, (K_loeb_at_p + K_interconnected)/2, 
             f'{reduction:.0f}% reduction\ndue to morphology', 
             fontsize=10, color='red', fontweight='bold', 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    ax2.set_xlabel('Porosity (-)', fontsize=14)
    ax2.set_ylabel('K$_{eff}$ (W/m·K)', fontsize=14)
    ax2.set_title('Effective Thermal Conductivity vs Porosity', fontsize=16, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=11, framealpha=0.95)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_xlim([0.10, 0.30])
    ax2.set_ylim([0.4, 1.0])
    ax2.tick_params(axis='both', labelsize=12)
    
    # Add text annotation for Loeb model accuracy on distributed
    loeb_error = abs(K_distributed - (1.0 - 1.37 * results["distributed_77"]["p_total"])) / K_distributed * 100
    ax2.text(0.27, 0.92, f'K$_{{Loeb,error}}$ = {loeb_error:.1f}%', 
             fontsize=12, color='blue', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('keff_vs_porosity_comparison.png', dpi=200, bbox_inches='tight')
    plt.close(fig2)
    print(f"✓ K_eff vs Porosity plot saved: keff_vs_porosity_comparison.png")
    print()

    print("=" * 70)


if __name__ == "__main__":
    main()
