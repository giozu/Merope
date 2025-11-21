import numpy as np
import os
from merope_engine import MeropeEngine

# -----------------------------------------------------------
# settings
# -----------------------------------------------------------

L = [10, 10, 10]
n3D = 80

lagR = 3.0
lagPhi = 1.0

R_intra = 0.3
K_MATRIX = 1.0
K_GAS = 1e-3

WORK_DIR = "Distributed_Test"

p_values = np.linspace(0.02, 0.30, 10)
SEED = 0

# -----------------------------------------------------------
# run
# -----------------------------------------------------------

engine = MeropeEngine(
    L=L,
    n3D=n3D,
    lagR=lagR,
    lagPhi=lagPhi,
    inclR_inter=0.0,      # unused
    inclR_intra=R_intra,  # radius for distributed porosity
    k_matrix=K_MATRIX,
    k_gas=K_GAS,
    work_dir=WORK_DIR,
    nproc=1
)

outfile = os.path.join(WORK_DIR, "distributed_results.txt")

with open(outfile, "w") as f:
    f.write("p_target\tp_por_meas\tKxx\tKyy\tKzz\tKmean\tcrit1\tcrit2\n")

for p in p_values:
    print(f"\n[RUN] distributed porosity p={p}")
    res = engine.run_distributed_case(seed=SEED, p_target=p)

    with open(outfile, "a") as f:
        f.write(
            f"{res['p_target']:.6f}\t{res['p_por_meas']:.6f}\t"
            f"{res['Kxx']:.6f}\t{res['Kyy']:.6f}\t{res['Kzz']:.6f}\t"
            f"{res['Kmean']:.6f}\t"
            f"{res['crit1_LRVE_over_Rpore']:.3f}\t"
            f"{res['crit2_Rpore_over_Lvoxel']:.3f}\n"
        )

print("\nDone. Results saved in:", outfile)
