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

R_PORE = 0.25

L_values = np.linspace(3.0, 9.0, 8)     # controls crit1
n3D_values = np.arange(20, 130, 20)     # controls crit2

outfile = os.path.join(WORK_DIR, "geometry_scan_full.txt")

with open(outfile, "w") as f:
    f.write("L_RVE\tR_pore\tn3D\tcrit1\tcrit2\tKmean\tp_por_meas\n")

# -----------------------------------------------------------
# Full 2D scan
# -----------------------------------------------------------

for L_RVE in L_values:
    for n3D in n3D_values:

        print(f"[SCAN] L={L_RVE:.2f}, n3D={n3D}")

        engine = MeropeEngine(
            L=[L_RVE, L_RVE, L_RVE],
            n3D=int(n3D),
            lagR=lagR,
            lagPhi=lagPhi,
            inclR_inter=0.0,
            inclR_intra=R_PORE,
            k_matrix=K_MATRIX,
            k_gas=K_GAS,
            work_dir=os.path.join(WORK_DIR,f"L{L_RVE:.2f}_n{n3D}"),
            nproc=1
        )

        # geometry not safe -> we want ALL combinations
        res = engine.run_distributed_case(seed=SEED, p_target=P_FIXED, safe_geometry=False)

        L = float(L_RVE)
        R = float(R_PORE)
        n = int(n3D)
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
