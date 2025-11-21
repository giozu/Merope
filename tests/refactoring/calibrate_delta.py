import numpy as np
import os

# --- IMPORT NECESSARI (mancavano!) ---
import sac_de_billes
import merope

from merope_engine import MeropeEngine


# =====================================================
# USER PARAMETERS
# =====================================================

L = [10, 10, 10]
n3D = 200

lagR = 1.0
lagPhi = 1.0

R_inter = 0.03
R_intra = 0.10

K_MATRIX = 1.0
K_GAS = 1e-3

SEED = 0

# delta to probe
delta_values = [0.001, 0.003, 0.005, 0.010]

OUTFILE = "layer_factor_calibration.txt"


# =====================================================
# HELPER: extract layer porosity
# =====================================================

def measure_layer(engine, delta_phys):
    """
    Costruisce SOLO i grani con il boundary layer.
    Misura la porosità della fase 'boundary layer' PRIMA del merge.
    """
    Lbox = engine.L

    phase_grains   = 0
    phase_intra    = 1
    phase_inter    = 2
    phase_boundary = 3

    # --- 1) Laguerre grains ---
    sph_lag = merope.SphereInclusions_3D()
    sph_lag.setLength(Lbox)
    sph_lag.fromHisto(
        int(SEED),
        sac_de_billes.TypeAlgo.RSA,
        0.0,
        [[lagR, lagPhi]],
        [1],
    )

    tess = merope.LaguerreTess_3D(Lbox, sph_lag.getSpheres())
    grains = merope.MultiInclusions_3D()
    grains.setInclusions(tess)

    # --- 2) Add layer δ ---
    ids = grains.getAllIdentifiers()
    grains.addLayer(ids, phase_boundary, delta_phys)
    grains.changePhase(ids, [1 for _ in ids])

    # --- 3) Voxelize and measure ---
    struct = merope.Structure_3D(grains)
    grid, fracs = engine._voxelize_and_assign(struct)

    p_layer = fracs.get(phase_boundary, 0.0)
    return p_layer


# =====================================================
# MAIN CALIBRATION
# =====================================================

def calibrate():
    engine = MeropeEngine(
        L=L,
        n3D=n3D,
        lagR=lagR,
        lagPhi=lagPhi,
        inclR_inter=R_inter,
        inclR_intra=R_intra,
        k_matrix=K_MATRIX,
        k_gas=K_GAS,
        work_dir="CALIB_LAYER",
        nproc=1
    )

    delta_list = []
    player_list = []

    print("== Layer calibration ==")
    for d in delta_values:
        print(f"Measuring layer porosity for delta={d}...")
        p_layer = measure_layer(engine, d)
        print(f"  p_layer = {p_layer:.6f}")
        delta_list.append(d)
        player_list.append(p_layer)

    # Fit p_layer ≈ C * delta
    delta_arr = np.array(delta_list)
    p_arr = np.array(player_list)

    C, residuals, _, _, _ = np.polyfit(delta_arr, p_arr, deg=1, full=True)
    C = C[0]  # slope only

    print("\n=== RESULT ===")
    print(f"Calibrated C = {C:.6f}")
    print(f"Meaning: p_layer ≈ {C:.6f} * delta")

    with open(OUTFILE, "w") as f:
        f.write("# Calibrated layer coefficient\n")
        f.write("delta\tp_layer\n")
        for d, p in zip(delta_list, player_list):
            f.write(f"{d:.6f}\t{p:.6f}\n")
        f.write("\n# linear fit:\n")
        f.write(f"C = {C:.6f}\n")

    print(f"\nSaved results in {OUTFILE}.")


if __name__ == "__main__":
    calibrate()
