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

p_total = 0.20   # porosità totale desiderata

# -----------------------------------------------------
# SCANSIONE (PUOI MODIFICARE)
# -----------------------------------------------------

p_delta_values = [0.30, 0.40]   # porosità inter-granulare
p_intra_values = [0.19]   # porosità intra-granulare
delta_phys_values = [0.003, 0.01]   # thickness film (metri / unita Merope)

def estimate_porosity(p_delta, p_intra, delta_phys, lagR, lagPhi):
    """
    Stima della porosità finale usando:
        p_est = p_intra + p_delta + 3 * phi * delta / R_g
    dove phi ≈ lagPhi (packing dei grani)
    """
    Rg = lagR
    phi = lagPhi

    p_layer_est = 3.0 * phi * (delta_phys / Rg)
    p_est = p_intra + p_delta + p_layer_est

    return p_est, p_layer_est

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

# =====================================================
# OUTPUT
# =====================================================

outfile = os.path.join(WORK_DIR, "interconnected_results.txt")
os.makedirs(WORK_DIR, exist_ok=True)

with open(outfile, "w") as f:
    f.write(
        "p_delta\tp_intra\tdelta_phys\t"
        "p_estimate\tp_layer_estimate\t"
        "p_meas\tKxx\tKyy\tKzz\tKmean\n"
    )
    
for p_delta in p_delta_values:
    for p_intra in p_intra_values:
        for delta_phys in delta_phys_values:

            print(f"\n[RUN] p_delta={p_delta}, p_intra={p_intra}, delta={delta_phys}")

            p_est, p_layer_est = estimate_porosity(
                p_delta, p_intra, delta_phys, lagR, lagPhi
            )

            res = engine.run_interconnected_case(
                seed=SEED,
                p_delta=p_delta,
                p_intra=p_intra,
                delta_phys=delta_phys,
                p_target=p_total,
                amitex=False
            )

            p_meas = res["p_meas"]

            with open(outfile, "a") as f:
                f.write(
                    f"{p_delta:.6f}\t{p_intra:.6f}\t{delta_phys:.6f}\t"
                    f"{p_est:.6f}\t{p_layer_est:.6f}\t"
                    f"{p_meas:.6f}\t"
                    f"{res['Kxx']:.6f}\t{res['Kyy']:.6f}\t{res['Kzz']:.6f}\t"
                    f"{res['Kmean']:.6f}\n"
                )

print("\nDone. Results saved in:", outfile)
