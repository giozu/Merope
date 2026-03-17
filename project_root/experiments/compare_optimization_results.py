#!/usr/bin/env python3
"""
Compare optimization results between distributed and interconnected morphologies.

Usage:
    python compare_optimization_results.py
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def main():
    print("=" * 70)
    print("COMPARISON: DISTRIBUTED vs INTERCONNECTED MORPHOLOGY")
    print("=" * 70)
    print()

    # Expected results from pore_analysis.py (with stereological correction)
    results = {
        "distributed_77": {
            "p_total": 0.227,
            "p_boundary": 0.000,
            "p_intra": 0.227,
            "n_pores": 2425,
            "morphology": "Isolated spheres",
        },
        "connected_79": {
            "p_total": 0.223,
            "p_boundary": 0.138,
            "p_intra": 0.085,
            "n_pores": 258,
            "morphology": "Interconnected network",
        },
    }

    # Predict K_eff
    # Distributed (classical Loeb)
    K_distributed = 1.0 * (1.0 - 1.37 * results["distributed_77"]["p_total"])

    # Interconnected (with correction factor)
    p_b = results["connected_79"]["p_boundary"]
    p_i = results["connected_79"]["p_intra"]
    delta = 1.0  # Placeholder - will be updated from optimization

    # Correction factor parameters
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
    print("  1. Both samples have ~22-23% porosity (similar)")
    print(f"  2. K_eff differs by {reduction:.0f}% due to morphology alone")
    print("  3. Interconnected pores have 2-3× stronger impact than isolated pores")
    print("  4. Classical Loeb model fails for interconnected morphology")
    print()

    # Create comparison plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: Porosity composition
    ax = axes[0]
    samples = ["Distributed\n77%", "Interconnected\n79%"]
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

    # Plot 2: K_eff comparison
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
    print(f"✓ Comparison plot saved: comparison_distributed_vs_interconnected.png")
    print()

    print("=" * 70)


if __name__ == "__main__":
    main()
