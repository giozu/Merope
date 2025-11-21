import numpy as np
import os
from merope_engine import MeropeEngine

# =====================================================
# PARAMETRI
# =====================================================

L = [10, 10, 10]
n3D = 200

lagR = 1
lagPhi = 1

R_inter = 0.03      # pori intergranulari
R_intra = 0.10      # pori intragranulari

K_MATRIX = 1.0
K_GAS = 1e-3

WORK_DIR = "Interconnected_Test"
SEED = 0

# -----------------------------------------------------
# SCANSIONE (PUOI MODIFICARE)
# -----------------------------------------------------

p_delta_values = [0.40]   # porosità inter-granulare
p_intra_values = [0.19]   # porosità intra-granulare
delta_phys_values = [0.003]   # thickness film (metri / unita Merope)

# =====================================================
# OUTPUT
# =====================================================

outfile = os.path.join(WORK_DIR, "interconnected_results.txt")
os.makedirs(WORK_DIR, exist_ok=True)

with open(outfile, "w") as f:
    f.write("p_target\tp_meas\tp_delta\tp_intra\tdelta_phys\tKxx\tKyy\tKzz\tKmean\n")

# =====================================================
# MAIN
# =====================================================

engine = MeropeEngine(
    L=L,
    n3D=n3D,
    lagR=lagR,
    lagPhi=lagPhi,
    inclR_inter=R_inter,
    inclR_intra=R_intra,
    k_matrix=K_MATRIX,
    k_gas=K_GAS,
    work_dir=WORK_DIR,
    nproc=1
)

for p_delta in p_delta_values:
    for p_intra in p_intra_values:
        for delta_phys in delta_phys_values:

            print(f"\n[RUN] p_delta={p_delta}, p_intra={p_intra}, delta={delta_phys}")

            res = engine.run_interconnected_case(
                seed=SEED,
                p_delta=p_delta,
                p_intra=p_intra,
                delta_phys=delta_phys
            )

            with open(outfile, "a") as f:
                f.write(
                    f"{res['p_target']:.6f}\t"
                    f"{res['p_meas']:.6f}\t"
                    f"{res['p_delta']:.6f}\t"
                    f"{res['p_intra']:.6f}\t"
                    f"{res['delta_phys']:.6f}\t"
                    f"{res['Kxx']:.6f}\t"
                    f"{res['Kyy']:.6f}\t"
                    f"{res['Kzz']:.6f}\t"
                    f"{res['Kmean']:.6f}\n"
                )

print("\nDone. Results saved in:", outfile)
