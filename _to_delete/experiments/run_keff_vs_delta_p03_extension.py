"""
run_keff_vs_delta_p03_extension.py
==================================
One-off extension: anchor the low-delta plateau at p=0.3 by running
delta in {0.05, 0.07, 0.09, 0.11, 0.13}.

Reuses worker() from run_keff_vs_delta.py and appends to the canonical
Results_Keff_vs_Delta/keff_vs_delta.csv.

Resolution caveat (see context/04 §1.3): at n3D=200, L=10 -> Delta_vox=0.05,
so delta=0.05 is exactly one voxel thick. The composite-voxel Voigt rule
then sets the floor on K_min(p=0.3); these points should be reported as
Voigt-bounded, not converged.
"""
import sys
from pathlib import Path
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from experiments import run_keff_vs_delta as rk

rk.N_CPUS = 8

NEW_DELTAS = [0.05, 0.07, 0.09, 0.11, 0.13]
P_TARGET = 0.3
CSV_PATH = rk.OUTPUT_DIR / "keff_vs_delta.csv"


def main():
    rk.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if CSV_PATH.exists():
        df = pd.read_csv(CSV_PATH)
        print(f"[INIT] Loaded {len(df)} existing rows from {CSV_PATH}")
    else:
        df = pd.DataFrame(columns=["Target_P", "Delta", "Grain_R", "Real_P", "K_eff"])
        print(f"[INIT] No existing CSV; starting fresh")

    for delta in NEW_DELTAS:
        already = df[
            (abs(df["Target_P"] - P_TARGET) < 1e-6) &
            (abs(df["Delta"] - delta) < 1e-6)
        ]
        if len(already) > 0:
            print(f"[SKIP] p={P_TARGET}, delta={delta} already in CSV")
            continue

        print(f"\n=== Running p={P_TARGET}, delta={delta} (N_CPUS={rk.N_CPUS}) ===")
        result = rk.worker((P_TARGET, delta, False))

        df = pd.concat([df, pd.DataFrame([result])], ignore_index=True)
        df = df.sort_values(by=["Target_P", "Delta"]).reset_index(drop=True)
        df.to_csv(CSV_PATH, index=False)
        print(f"[SAVE] CSV updated -> {len(df)} rows")

    print(f"\n=== Done. Final CSV: {CSV_PATH} ({len(df)} rows) ===")


if __name__ == "__main__":
    main()
