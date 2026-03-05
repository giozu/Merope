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
N_VOX   = 120                   # voxels per side (increase for accuracy)
SEED    = 42

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
    builder = MicrostructureBuilder(L=L_DIM, n3D=N_VOX, seed=SEED)
    solver  = ThermalSolver(n_cpus=4)

    pm.cleanup_folder(str(output_dir))

    rows: list[dict] = []
    print("=== K_eff vs Porosity – Spherical Inclusions ===")
    print(f"    R_pore = {SPHERE_R} | N_vox = {N_VOX} | L = {L_DIM}")

    for phi_target in PHI_VALUES:
        print(f"\n→ φ_target = {phi_target:.3f}")

        multi  = builder.generate_spheres([[SPHERE_R, float(phi_target)]], phase_id=2)
        struct = merope.Structure_3D(multi)

        case_dir = output_dir / f"Phi_{phi_target:.4f}"
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

            print(
                f"   φ_real={phi_real:.4f} | "
                f"K_sim={k_eff:.4f} | K_Maxwell={k_maxw:.4f} | K_Loeb={k_loeb:.4f}"
            )
            rows.append({
                "Phi_Target":  phi_target,
                "Phi_Real":    phi_real,
                "K_mean":      k_eff,
                "K_Maxwell":   k_maxw,
                "K_Loeb":      k_loeb,
            })

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
    """Reproduce the K_eff vs Porosity figure."""

    phi   = df["Phi_Real"].to_numpy()
    phi_s = np.linspace(0.0, phi.max() * 1.05, 200)

    fig, ax = plt.subplots(figsize=(7, 5))

    # Analytical curves
    ax.plot(phi_s, maxwell_eucken(phi_s), "r-",  linewidth=1.5, label="K_Maxwell (Maxwell–Eucken)")
    ax.plot(phi_s, loeb(phi_s),           "g--", linewidth=1.5, label="K_Loeb (Loeb 1954)")

    # Simulation points
    ax.scatter(
        df["Phi_Real"], df["K_mean"],
        marker="o", s=40, color="steelblue", zorder=5, label="K_mean (Mérope + Amitex)"
    )

    ax.set_xlabel("Porosity", fontsize=12)
    ax.set_ylabel(r"$K_\mathrm{eff}$ [W/m·K]", fontsize=12)
    ax.set_title("Effective Thermal Conductivity vs Porosity\n(Spherical Inclusions)", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_xlim(left=0.0)
    ax.set_ylim(bottom=0.0)

    fig.tight_layout()
    img_path = output_dir / "Keff_vs_Porosity.png"
    fig.savefig(img_path, dpi=300)
    plt.close(fig)
    print(f"Figure saved → {img_path}")


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--plot-only",  action="store_true", help="Load existing CSV and re-plot (skip simulations).")
    parser.add_argument("--no-solver",  action="store_true", help="Skip Amitex solver (generate geometry only).")
    args = parser.parse_args()

    csv_path = OUTPUT_DIR / "keff_vs_porosity.csv"

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
