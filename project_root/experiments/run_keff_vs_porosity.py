"""
run_keff_vs_porosity.py
=======================
Generates K_eff vs Porosity for RVEs with spherical inclusions, comparing
Mérope+Amitex simulation data against the Maxwell–Eucken and Loeb (1954)
analytical models.

Usage
-----
# Run simulations + plot:
    python experiments/run_keff_vs_porosity.py

# Plot only (from a previous run's CSV):
    python experiments/run_keff_vs_porosity.py --plot-only

# Skip the Amitex solver (geometry only) – useful for testing:
    python experiments/run_keff_vs_porosity.py --no-solver
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make sure project_root/ is on sys.path so that `core` is importable
# regardless of where the script is invoked from.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── Core imports (only needed when running simulations) ──────────────────────
try:
    import merope
    from core.geometry import MicrostructureBuilder
    from core.solver import ThermalSolver
    from core.utils import ProjectManager
    _MEROPE_AVAILABLE = True
except ImportError as _import_err:
    _MEROPE_AVAILABLE = False
    _IMPORT_ERROR_MSG = str(_import_err)


# =============================================================================
# CONFIGURATION – edit here
# =============================================================================
L_DIM   = [10.0, 10.0, 10.0]   # RVE size [μm] (cubic)
N_VOX_BASE = 120                # Starting voxels per side
SEED    = 42

# Adaptive Resolution Settings
ADAPTIVE_VOX   = True
MAX_ERROR_PERC = 2.0            # If error > this %, increase N_VOX
MAX_N_VOX      = 240            # Maximum allowed N_VOX
N_VOX_STEP     = 30             # How much to increase N_VOX each time

# Sphere radius / porosity sweep
SPHERE_R  = 0.5                          # pore radius [same units as L_DIM]
PHI_VALUES = np.linspace(0.01, 0.30, 20) # porosity range to simulate

# Thermal properties
K_MAT  = 1.0    # matrix conductivity  [W/m·K]  (normalised to 1)
K_PORE = 1e-3   # pore conductivity    [W/m·K]

# Amitex phase assignment:  index 0 = matrix, index 2 = pores
K_THERMAL = [K_MAT, K_MAT, K_PORE]

OUTPUT_DIR = Path("Results_Keff_vs_Porosity")


# =============================================================================
# ANALYTICAL MODELS
# =============================================================================

def maxwell_eucken(phi: np.ndarray, k_m: float = K_MAT, k_p: float = K_PORE) -> np.ndarray:
    """Maxwell–Eucken model for dilute spherical inclusions.

    K_eff = K_m * (1 + 2β·φ) / (1 − β·φ)       with  β = (K_p − K_m)/(K_p + 2·K_m)
    """
    beta = (k_p - k_m) / (k_p + 2.0 * k_m)
    return k_m * (1.0 + 2.0 * beta * phi) / (1.0 - beta * phi)


def loeb(phi: np.ndarray, k_m: float = K_MAT) -> np.ndarray:
    """Loeb (1954) model for spherical pores.

    K_eff = K_m · (1 − φ)^(4/3)

    Reference: Loeb, A.L. (1954). "Thermal conductivity: VIII, A theory of
    thermal conductivity of porous materials." Journal of the American
    Ceramic Society, 37(2), 96-99.
    """
    return k_m * (1.0 - phi) ** (4.0 / 3.0)


# =============================================================================
# SIMULATION
# =============================================================================

def recover_results(output_dir: Path) -> pd.DataFrame:
    """Read existing directories to recreate the CSV without recalculating everything."""
    import re
    if not _MEROPE_AVAILABLE:
        raise RuntimeError(
            f"Mérope/core modules not found: {_IMPORT_ERROR_MSG}\n"
            "Activate your Merope environment and re-run."
        )

    print("=== Recovering K_eff vs Porosity from existing folders ===")
    L_RVE = float(L_DIM[0])
    rows: list[dict] = []

    for case_dir in sorted(output_dir.glob("Phi_*_Nvox_*")):
        if not case_dir.is_dir():
            continue
            
        thermal_file = case_dir / "thermalCoeff_amitex.txt"
        if not thermal_file.exists():
            print(f"   [Skipping {case_dir.name}] - No thermalCoeff_amitex.txt found.")
            continue
            
        m = re.match(r"Phi_([0-9\.]+)_Nvox_([0-9]+)", case_dir.name)
        if not m:
            continue
            
        phi_target = float(m.group(1))
        current_n_vox = int(m.group(2))
        
        # Re-initialize builder to get exact phi_real
        builder = MicrostructureBuilder(L=L_DIM, n3D=current_n_vox, seed=SEED)
        multi   = builder.generate_spheres([[SPHERE_R, phi_target]], phase_id=2)
        struct  = merope.Structure_3D(multi)
        
        with ProjectManager().cd(str(case_dir)):
            fractions = builder.voxellate(struct, K_THERMAL)
            phi_real  = fractions.get(2, 0.0)
            
        coeffs = np.loadtxt(thermal_file)
        if coeffs.shape == (3, 3):
            k_eff = np.trace(coeffs) / 3.0
        else:
            k_eff = 0.0
            
        L_voxel = L_RVE / float(current_n_vox)
        ratio_LR    = L_RVE / float(SPHERE_R)
        ratio_Rlvox = float(SPHERE_R) / L_voxel
        
        k_maxw   = float(maxwell_eucken(np.array([phi_real]), K_MAT, K_PORE)[0])
        k_loeb   = float(loeb(np.array([phi_real]), K_MAT)[0])
        error_perc = abs(k_eff - k_maxw) / k_maxw * 100.0 if k_maxw > 0 else 0.0
        
        print(
            f"   [Recovered N={current_n_vox}] φ_real={phi_real:.4f} | "
            f"K_sim={k_eff:.4f} | Err={error_perc:.2f}%"
        )
        
        rows.append({
            "Phi_Target":  phi_target,
            "Phi_Real":    phi_real,
            "K_mean":      k_eff,
            "K_Maxwell":   k_maxw,
            "K_Loeb":      k_loeb,
            "Error_Perc":  error_perc,
            "Ratio_LR":    ratio_LR,
            "Ratio_Rlvox": ratio_Rlvox,
            "N_Vox":       current_n_vox
        })
        
    df = pd.DataFrame(rows)
    if df.empty:
        print("No valid results found to recover.")
        return df
        
    df = df.sort_values(by=["Phi_Target", "N_Vox"])
    csv_path = output_dir / "keff_vs_porosity.csv"
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"\nRecovered {len(df)} results -> {csv_path}")
    return df


def run_simulations(output_dir: Path, no_solver: bool = False) -> pd.DataFrame:
    """Run Mérope + Amitex for each porosity target and return a DataFrame."""
    if not _MEROPE_AVAILABLE:
        raise RuntimeError(
            f"Mérope/core modules not found: {_IMPORT_ERROR_MSG}\n"
            f"  sys.path = {sys.path}\n"
            "Activate your Merope environment and re-run "
            "(or check that PYTHONPATH includes project_root/)."
        )

    pm      = ProjectManager()
    solver  = ThermalSolver(n_cpus=4)

    pm.cleanup_folder(str(output_dir))

    rows: list[dict] = []
    print("=== K_eff vs Porosity – Spherical Inclusions ===")
    
    # ── Numerical Resolution Checks ──────────────────────────────────────────
    L_RVE   = float(L_DIM[0])  # Assuming cubic RVE
    
    print(f"    Geometry: R_pore = {SPHERE_R} | L_RVE = {L_RVE}")
    print(f"    Base Resolution: N_vox = {N_VOX_BASE} (Adaptive up to {MAX_N_VOX})")

    for phi_target in PHI_VALUES:
        print(f"\n→ φ_target = {phi_target:.3f}")
        
        current_n_vox = N_VOX_BASE

        while True:
            # Re-initialize builder with current_n_vox (might have increased)
            builder = MicrostructureBuilder(L=L_DIM, n3D=current_n_vox, seed=SEED)
            multi   = builder.generate_spheres([[SPHERE_R, float(phi_target)]], phase_id=2)
            struct  = merope.Structure_3D(multi)
            
            # Recalculate resolution parameters for logging
            L_voxel = L_RVE / float(current_n_vox)
            ratio_LR    = L_RVE / float(SPHERE_R)
            ratio_Rlvox = float(SPHERE_R) / L_voxel

            case_dir = output_dir / f"Phi_{phi_target:.4f}_Nvox_{current_n_vox}"
            with pm.cd(str(case_dir)):
                fractions = builder.voxellate(struct, K_THERMAL)
                phi_real  = fractions.get(2, 0.0)

                if no_solver:
                    res = {"Kxx": 0.0, "Kyy": 0.0, "Kzz": 0.0, "Kmean": 0.0}
                else:
                    res = solver.solve()

                k_eff    = res["Kmean"]
                k_maxw   = float(maxwell_eucken(np.array([phi_real]), K_MAT, K_PORE)[0])
                k_loeb   = float(loeb(np.array([phi_real]), K_MAT)[0])
                
                error_perc = abs(k_eff - k_maxw) / k_maxw * 100.0 if k_maxw > 0 else 0.0
                
                print(
                    f"   [N={current_n_vox}] φ_real={phi_real:.4f} | R/l_vox={ratio_Rlvox:.2f} | "
                    f"K_sim={k_eff:.4f} | Err={error_perc:.2f}%"
                )
                
                # Check if we should retry with higher resolution
                if ADAPTIVE_VOX and error_perc > MAX_ERROR_PERC and current_n_vox < MAX_N_VOX:
                    current_n_vox += N_VOX_STEP
                    if current_n_vox > MAX_N_VOX:
                        current_n_vox = MAX_N_VOX
                    print(f"   [!] Error {error_perc:.2f}% > {MAX_ERROR_PERC}%. Retrying with N_VOX = {current_n_vox}...")
                    continue
                
                # Save results
                rows.append({
                    "Phi_Target":  phi_target,
                    "Phi_Real":    phi_real,
                    "K_mean":      k_eff,
                    "K_Maxwell":   k_maxw,
                    "K_Loeb":      k_loeb,
                    "Error_Perc":  error_perc,
                    "Ratio_LR":    ratio_LR,
                    "Ratio_Rlvox": ratio_Rlvox,
                    "N_Vox":       current_n_vox
                })
                
                # If error is fine or we hit max N_VOX, we break out of the while loop
                break

    df = pd.DataFrame(rows)
    csv_path = output_dir / "keff_vs_porosity.csv"
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved → {csv_path}")
    return df


# =============================================================================
# PLOTTING
# =============================================================================

def plot_results(df: pd.DataFrame, output_dir: Path) -> None:
    """Enhanced plots including error analysis and resolution checks."""

    phi   = df["Phi_Real"].to_numpy()
    phi_s = np.linspace(0.0, phi.max() * 1.05, 200)

    # 1. Main Plot: K vs Porosity
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10), sharex=True)

    ax1.plot(phi_s, maxwell_eucken(phi_s), "r-",  linewidth=1.5, label="Maxwell–Eucken")
    ax1.plot(phi_s, loeb(phi_s),           "g--", linewidth=1.1, label="Loeb (1954)")
    ax1.scatter(df["Phi_Real"], df["K_mean"], marker="o", s=50, color="steelblue", 
                edgecolor="white", zorder=5, label="Mérope + Amitex")

    ax1.set_ylabel(r"$K_\mathrm{eff}$ [W/m·K]", fontsize=12)
    ax1.set_title("Thermal Conductivity Validation", fontsize=14, fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.3)
    ax1.set_ylim(bottom=0.0, top=1.05)

    # 2. Error Plot: Relative Error vs Porosity
    # Color points differently if N_VOX was increased (Adaptive)
    adaptive_mask = df["N_Vox"] > df["N_Vox"].min()
    base_mask = ~adaptive_mask
    
    # Draw line through all points
    ax2.plot(df["Phi_Real"], df["Error_Perc"], "-", color="lightgray", linewidth=1.0, zorder=1)
    
    # Plot base points
    ax2.plot(df["Phi_Real"][base_mask], df["Error_Perc"][base_mask], "o", 
             color="crimson", linewidth=1.0, markersize=5, zorder=2, label=f"N_Vox = {df['N_Vox'].min()}")
    
    # Plot adaptive points
    if adaptive_mask.any():
        ax2.plot(df["Phi_Real"][adaptive_mask], df["Error_Perc"][adaptive_mask], "s", 
                 color="orange", linewidth=1.0, markersize=5, zorder=3, label="Adaptive N_Vox (Increased)")
        ax2.legend(fontsize=9, loc="upper left")

    ax2.set_ylabel("Rel. Error vs Maxwell [%]", fontsize=12)
    ax2.set_xlabel("Porosity", fontsize=12)
    ax2.grid(True, linestyle="--", alpha=0.3)
    ax2.set_ylim(bottom=0.0)

    # Annotate with Resolution Info
    max_nvox_used = df["N_Vox"].max()
    res_text = (
        f"Resolution Info:\n"
        f"Base N_vox = {df['N_Vox'].min()}\n"
        f"Max used N_vox = {max_nvox_used}\n"
        f"Base R_pore/L_vox = {df['Ratio_Rlvox'].iloc[0]:.2f}\n"
        f"L_RVE / R_pore = {df['Ratio_LR'].iloc[0]:.2f}"
    )
    plt.text(0.05, 0.95, res_text, transform=ax2.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    fig.tight_layout()
    img_path = output_dir / "Keff_Validation_Summary.png"
    fig.savefig(img_path, dpi=300)
    
    # Save a legacy simple plot for compatibility if needed
    plt.close(fig)
    print(f"Summary figure saved → {img_path}")


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--plot-only",  action="store_true", help="Load existing CSV and re-plot (skip simulations).")
    parser.add_argument("--no-solver",  action="store_true", help="Skip Amitex solver (generate geometry only).")
    parser.add_argument("--recover",    action="store_true", help="Recover CSV from existing folders without simulating.")
    args = parser.parse_args()

    csv_path = OUTPUT_DIR / "keff_vs_porosity.csv"

    if args.recover:
        df = recover_results(OUTPUT_DIR)
        if not df.empty:
            plot_results(df, OUTPUT_DIR)
        return

    if args.plot_only:
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {csv_path}. Run without --plot-only first.")
        print(f"Loading existing results from {csv_path}")
        df = pd.read_csv(csv_path)
    else:
        df = run_simulations(OUTPUT_DIR, no_solver=args.no_solver)

    plot_results(df, OUTPUT_DIR)


if __name__ == "__main__":
    main()
