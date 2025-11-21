import numpy as np
import os
from merope_engine import MeropeEngine

# -----------------------------------------------------------
# settings
# -----------------------------------------------------------

P_FIXED = 0.10
SEED = 0

lagR = 3.0
lagPhi = 1.0

K_MATRIX = 1.0
K_GAS = 1e-3

WORK_DIR = "Geometry_Scan"
os.makedirs(WORK_DIR, exist_ok=True)

# -----------------------------------------------------------
# Sampling ranges
# -----------------------------------------------------------

L_RVE = 10

crit1_values = np.linspace(5, 25, 5)
crit2_values = np.linspace(1, 5, 5)

outfile = os.path.join(WORK_DIR, "geometry_scan.txt")

with open(outfile, "w") as f:
    f.write("L_RVE\tR_pore\tn3D\tcrit1\tcrit2\tKmean\tp_por_meas\n")

# -----------------------------------------------------------
# Full 2D scan
# -----------------------------------------------------------

for crit_1 in crit1_values:
    for crit_2 in crit2_values:

        R_PORE = L_RVE / crit_1
        n3D = int(crit_2 * (L_RVE / R_PORE))

        if n3D < 16:
            print(f"Skipping n3D={n3D} (too coarse)")
            continue

        print(f"[SCAN] L={L_RVE:.2f}, n3D={n3D}")

        engine = MeropeEngine(
            L=[L_RVE, L_RVE, L_RVE],
            n3D=n3D,
            lagR=lagR,
            lagPhi=lagPhi,
            inclR_inter=0.0,
            inclR_intra=R_PORE,
            k_matrix=K_MATRIX,
            k_gas=K_GAS,
            work_dir=os.path.join(WORK_DIR,f"L{int(L_RVE)}_n{n3D}"),
            nproc=1
        )

        res = engine.run_distributed_case(seed=SEED, p_target=P_FIXED, safe_geometry=False)

        L = float(L_RVE)
        R = float(R_PORE)
        n = n3D
        L_voxel = L / n

        crit1 = L / R
        crit2 = R / L_voxel

        with open(outfile,"a") as f:
            f.write(
                f"{L:.6f}\t{R:.6f}\t{n}\t"
                f"{crit1:.6f}\t{crit2:.6f}\t"
                f"{res['Kmean']:.6f}\t{res['p_por_meas']:.6f}\n"
            )

print("\nScan complete.")
print("Saved:", outfile)
