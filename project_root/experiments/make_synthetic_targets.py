"""
make_synthetic_targets.py
=========================

Generate two synthetic 2D microstructure PNGs as drop-in replacements for the
ESFR-SIMPLE consortium reference images (`connected_79.png`, `distributed_77.png`)
that the existing optimisation pipeline targets. The originals come from a
private consortium presentation and cannot be reproduced in a journal paper
without rights clearance; the synthetic targets generated here are fully
reproducible from this script alone, with known ground-truth parameters.

The geometry conventions (Laguerre + grain-boundary band + Boolean spherical
pores clipped to the GB, for the interconnected case; isolated RSA spherical
pores, for the distributed case) match `run_anisotropy.py` and the AMITEX
simulations used to calibrate the joint sigmoidal fit, so the synthetic
targets live on the same parameter space as the rest of the framework.

Outputs (in OUTPUT_DIR)
-----------------------
- synthetic_distributed.png    : isolated spherical pores, p ~ 0.23.
- synthetic_interconnected.png : Laguerre cells + delta=1 GB band + Boolean
                                 spherical pores clipped to the GB band, with
                                 realised boundary porosity ~ 0.14.
                                 Single pore phase only (no separate intra
                                 phase) -- matches the pore phase AMITEX
                                 actually simulates.
- ground_truth.json            : parameters used for both, plus realised
                                 porosity measured on the saved 2D slice.

Run
---
    cd ~/Merope
    source Env_Merope.sh
    export PYTHONPATH=$PYTHONPATH:./project_root
    python project_root/experiments/make_synthetic_targets.py
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import merope
import sac_de_billes


# --- Common ---------------------------------------------------------------
L_DIM = [10.0, 10.0, 10.0]
N_VOX = 400                  # bumped 200 -> 400 (2026-05-05) for crisper paper
                             # figures + richer area-histogram statistics. The
                             # optimisation script auto-rescales via um_per_px,
                             # so simulator n3d can be set independently.
SEED = 42                    # deliberately different from SEED=0 used in production
SLICE_AXIS = 2               # 0=x, 1=y, 2=z; we take the midplane along this axis
PORE_THRESHOLD = 1.5         # phase value above this is rendered as pore (black)

OUTPUT_DIR = Path("Optimization_3D_structure/exp_img_synthetic")

# --- Distributed (closed) porosity ---------------------------------------
# Polydisperse: discretise an approximately lognormal radius distribution
# into N bins, each contributing TARGET_P / N to the total porosity. The
# range spans roughly a factor 2x in radius, well above the voxelisation
# floor (~3 voxels at N_VOX=200 → r >= 0.15) so the resulting pores stay
# resolved on the slice. This makes the area-distribution histogram in the
# downstream pipeline non-degenerate (the previous monodisperse setup
# produced a delta-like distribution that the KS-test scoring couldn't
# discriminate).
DIST_TARGET_P = 0.23
DIST_RADII = [0.15, 0.20, 0.25, 0.32]
DIST_RADIUS_BINS = [[r, DIST_TARGET_P / len(DIST_RADII)] for r in DIST_RADII]

# --- Interconnected morphology -------------------------------------------
# The realised image is a union of two independent pore populations:
#   (a) BOUNDARY pores -- Boolean overlapping spheres, clipped to a
#       grain-boundary band of thickness DELTA. Form the "crack network".
#   (b) INTRA pores    -- isolated RSA spheres, smaller radius, thrown
#       independently. Most fall in grain interiors (some land on GBs by
#       chance; that's fine -- pore_analysis classifies post-hoc by area).
# Both populations are now polydisperse for the same reason as the
# distributed case (non-degenerate slice statistics).
INTER_LAG_R = 3.0
INTER_LAG_PHI = 1.0
INTER_DELTA = 1.0            # canonical (paper-anchor) value; written to
                             # `synthetic_interconnected.png` for backward compat.

# Optional sweep across grain-boundary thicknesses. Each value in this list
# produces a separate `synthetic_interconnected_delta_<tag>.png` file with the
# corresponding ground-truth parameters recorded in `ground_truth.json`. Useful
# for probing the morphology-correction sigmoid across its transition. Default
# is just the canonical anchor (delta=1.0, mirrored to
# `synthetic_interconnected.png`); add e.g. [0.3, 0.6, 1.0, 1.5] to enable a
# sweep across the sigmoid.
INTER_DELTA_VALUES = [1.0]

# (a) Boundary pores. Over-throw the Boolean target because clipping to the
# GB band retains ~60% of the raw volume.
INTER_BOUND_RAW_TARGET_P = 0.22
INTER_BOUND_TARGET_P_NOMINAL = 0.14
INTER_BOUND_RADII = [0.40]                     # 2026-05-05: monodisperse, in
                                                # the middle of the optimiser's
                                                # tightened search range
                                                # [0.30, 0.50]. This makes the
                                                # known-truth recovery test
                                                # well-posed: the synthetic and
                                                # the optimiser now use the
                                                # same parametric form (single
                                                # boundary radius, single
                                                # delta, single phi).
INTER_BOUND_RADIUS_BINS = [[r, INTER_BOUND_RAW_TARGET_P / len(INTER_BOUND_RADII)]
                           for r in INTER_BOUND_RADII]

# (b) Intra pores -- DISABLED in the matched-form target (target_p = 0). The
# optimiser's interconnected structure builder has no intra-pore parameter, so
# leaving an intra population in the optimiser's target would force the BO to
# compensate via delta-inflation. With p_intra = 0 the synthetic and the
# optimiser share the same parametric form and the recovery test is well-posed.
INTER_INTRA_TARGET_P = 0.0
INTER_INTRA_RADII = [0.12]                     # placeholder; not used at p=0
INTER_INTRA_RADIUS_BINS = [[r, INTER_INTRA_TARGET_P / len(INTER_INTRA_RADII)]
                           for r in INTER_INTRA_RADII]

# --- Interconnected, visual-rich variant ---------------------------------
# Used ONLY for the paper figure (illustrative microstructure with realistic
# polydisperse boundary pores plus intra-granular bubbles, matching the visual
# character of real interconnected MOX SEMs). NOT used as the optimiser
# target; the BO would land in the same degenerate corner we just fixed.
# Output: `synthetic_interconnected_visual.png`. Generated at the canonical
# delta only (no sweep needed for a figure).
INTER_VISUAL_BOUND_RADII = [0.20, 0.30, 0.42, 0.55]
INTER_VISUAL_BOUND_RAW_TARGET_P = 0.22
INTER_VISUAL_BOUND_RADIUS_BINS = [
    [r, INTER_VISUAL_BOUND_RAW_TARGET_P / len(INTER_VISUAL_BOUND_RADII)]
    for r in INTER_VISUAL_BOUND_RADII
]
INTER_VISUAL_INTRA_RADII = [0.08, 0.10, 0.13, 0.16]
INTER_VISUAL_INTRA_TARGET_P = 0.08
INTER_VISUAL_INTRA_RADIUS_BINS = [
    [r, INTER_VISUAL_INTRA_TARGET_P / len(INTER_VISUAL_INTRA_RADII)]
    for r in INTER_VISUAL_INTRA_RADII
]


def voxelise_to_array(structure) -> np.ndarray:
    grid_params = merope.vox.create_grid_parameters_N_L_3D([N_VOX] * 3, L_DIM)
    # VoxelRule.Center samples the phase at the voxel centre, giving a single
    # phase ID per voxel directly. compute_PhaseField then returns the integer
    # ID array without any homogenisation rule. Adequate for the binary slice
    # mask we want; we threshold afterwards anyway.
    grid = merope.vox.GridRepresentation_3D(structure, grid_params, merope.vox.VoxelRule.Center)
    conv = merope.vox.NumpyConverter_3D()
    return conv.compute_PhaseField(grid).reshape((N_VOX,) * 3, order='C')


def slice_to_png(arr3d: np.ndarray, png_path: Path) -> float:
    if SLICE_AXIS == 0:
        sl = arr3d[N_VOX // 2, :, :]
    elif SLICE_AXIS == 1:
        sl = arr3d[:, N_VOX // 2, :]
    else:
        sl = arr3d[:, :, N_VOX // 2]
    binary = np.where(sl > PORE_THRESHOLD, 0, 255).astype(np.uint8)
    Image.fromarray(binary).save(png_path)
    return float((binary == 0).sum() / binary.size)


def build_distributed():
    """Polydisperse RSA spherical pores spanning DIST_RADII."""
    incl_phase = 2
    pores = merope.SphereInclusions_3D()
    pores.setLength(L_DIM)
    pores.fromHisto(
        SEED, sac_de_billes.TypeAlgo.RSA, 0.0,
        DIST_RADIUS_BINS, [incl_phase] * len(DIST_RADIUS_BINS),
    )
    multi = merope.MultiInclusions_3D()
    multi.setInclusions(pores)
    return merope.Structure_3D(multi)


def build_interconnected_boundary(delta: float = INTER_DELTA,
                                  bound_radius_bins=None):
    """Laguerre + GB layer + Boolean spherical pores clipped to the GB band.

    Parameters
    ----------
    delta : float
        Absolute grain-boundary thickness (in RVE units). The dimensionless
        descriptor used in the framework is delta_star = delta / INTER_LAG_R.
    bound_radius_bins : list of [radius, volume_fraction] pairs, optional
        Boundary pore radius bins. Defaults to ``INTER_BOUND_RADIUS_BINS``
        (matched-form, monodisperse). Pass ``INTER_VISUAL_BOUND_RADIUS_BINS``
        for the visual-rich variant.
    """
    if bound_radius_bins is None:
        bound_radius_bins = INTER_BOUND_RADIUS_BINS

    incl_phase = 2
    delta_phase = 3
    grains_phase = 0

    # 1. Boolean spherical pores (over-thrown; clipping reduces realised p)
    sph_pores = merope.SphereInclusions_3D()
    sph_pores.setLength(L_DIM)
    sph_pores.fromHisto(
        SEED, sac_de_billes.TypeAlgo.BOOL, 0.0,
        bound_radius_bins, [incl_phase] * len(bound_radius_bins),
    )
    multi_pores = merope.MultiInclusions_3D()
    multi_pores.setInclusions(sph_pores)

    # 2. Laguerre seeds (RSA), tessellation, then a delta-thick GB layer
    sph_grains = merope.SphereInclusions_3D()
    sph_grains.setLength(L_DIM)
    sph_grains.fromHisto(
        SEED, sac_de_billes.TypeAlgo.RSA, 0.0,
        [[INTER_LAG_R, INTER_LAG_PHI]], [1],
    )
    poly = merope.LaguerreTess_3D(L_DIM, sph_grains.getSpheres())
    multi_grains = merope.MultiInclusions_3D()
    multi_grains.setInclusions(poly)
    ids = multi_grains.getAllIdentifiers()
    multi_grains.addLayer(ids, delta_phase, delta)
    multi_grains.changePhase(ids, [1 for _ in ids])

    # 3. Overlay: pores in grain interiors get erased; pores on GB survive
    mapping = {incl_phase: grains_phase, delta_phase: grains_phase}
    return merope.Structure_3D(multi_pores, multi_grains, mapping)


def build_interconnected_intra(intra_radius_bins=None):
    """Polydisperse RSA spherical pores -- the intra-granular population.

    Parameters
    ----------
    intra_radius_bins : list of [radius, volume_fraction] pairs, optional
        Intra-granular pore radius bins. Defaults to ``INTER_INTRA_RADIUS_BINS``
        (matched-form). Pass ``INTER_VISUAL_INTRA_RADIUS_BINS`` for the
        visual-rich variant.
    """
    if intra_radius_bins is None:
        intra_radius_bins = INTER_INTRA_RADIUS_BINS

    incl_phase = 2
    pores = merope.SphereInclusions_3D()
    pores.setLength(L_DIM)
    pores.fromHisto(
        SEED + 1, sac_de_billes.TypeAlgo.RSA, 0.0,
        intra_radius_bins, [incl_phase] * len(intra_radius_bins),
    )
    multi = merope.MultiInclusions_3D()
    multi.setInclusions(pores)
    return merope.Structure_3D(multi)


def union_pore_arrays(arr_a: np.ndarray, arr_b: np.ndarray) -> np.ndarray:
    """Combine two phase-ID voxel arrays: pore (phase >= 2) wins."""
    is_pore = (arr_a >= 2) | (arr_b >= 2)
    return np.where(is_pore, 2, 0).astype(arr_a.dtype)


def _delta_tag(delta: float) -> str:
    """Filesystem-safe tag for a delta value, e.g. 0.3 -> '0p3', 1.5 -> '1p5'."""
    return f"{delta:.2f}".rstrip("0").rstrip(".").replace(".", "p")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ----- Distributed target (single, no delta dependence) -----------------
    print(f"=== Building synthetic distributed target ===")
    arr_d = voxelise_to_array(build_distributed())
    p_d = slice_to_png(arr_d, OUTPUT_DIR / "synthetic_distributed.png")
    p_d_3d = float((arr_d > PORE_THRESHOLD).sum() / arr_d.size)
    print(f"  3D realised porosity = {p_d_3d:.4f}")
    print(f"  2D slice porosity    = {p_d:.4f}")

    # ----- Interconnected targets: one per delta value ----------------------
    # The intra population is independent of delta, so build it once. Skip
    # the build entirely when intra is disabled (Mérope's RSA rejects a
    # zero-volume-fraction call rather than no-oping).
    if INTER_INTRA_TARGET_P > 0:
        arr_intra = voxelise_to_array(build_interconnected_intra())
        p_i_intra_3d = float((arr_intra > PORE_THRESHOLD).sum() / arr_intra.size)
    else:
        arr_intra = None
        p_i_intra_3d = 0.0

    interconnected_runs = []
    for delta in INTER_DELTA_VALUES:
        delta_star = delta / INTER_LAG_R
        tag = _delta_tag(delta)
        png_name = f"synthetic_interconnected_delta_{tag}.png"

        print(f"\n=== Building synthetic interconnected target  "
              f"delta={delta} (delta*={delta_star:.3f}) ===")
        arr_bound = voxelise_to_array(build_interconnected_boundary(delta=delta))
        arr_i = arr_bound if arr_intra is None else union_pore_arrays(arr_bound, arr_intra)
        p_i_bound_3d = float((arr_bound > PORE_THRESHOLD).sum() / arr_bound.size)
        p_i_3d = float((arr_i > PORE_THRESHOLD).sum() / arr_i.size)
        p_i_slice = slice_to_png(arr_i, OUTPUT_DIR / png_name)
        print(f"  3D boundary-only porosity = {p_i_bound_3d:.4f}")
        print(f"  3D intra-only    porosity = {p_i_intra_3d:.4f}")
        print(f"  3D combined      porosity = {p_i_3d:.4f}  (overlap loss: "
              f"{(p_i_bound_3d + p_i_intra_3d - p_i_3d):.4f})")
        print(f"  2D slice combined porosity = {p_i_slice:.4f}")
        print(f"  output: {png_name}")

        # The canonical delta value also writes the legacy filename so the
        # existing run_optimization.py wiring (which points at
        # `synthetic_interconnected.png`) keeps working without changes.
        if abs(delta - INTER_DELTA) < 1e-9:
            slice_to_png(arr_i, OUTPUT_DIR / "synthetic_interconnected.png")

        interconnected_runs.append({
            "DELTA": delta,
            "DELTA_STAR": delta_star,
            "REALISED_3D_P_BOUNDARY": p_i_bound_3d,
            "REALISED_3D_P_INTRA": p_i_intra_3d,
            "REALISED_3D_P_COMBINED": p_i_3d,
            "REALISED_SLICE_P": p_i_slice,
            "output_png": png_name,
        })

    # ----- Visual-rich variant at canonical delta only ---------------------
    # Used solely as the paper figure for the interconnected morphology class.
    # Polydisperse boundary pores + intra-granular bubbles, matching the
    # qualitative content of real interconnected MOX SEMs.
    print(f"\n=== Building VISUAL-RICH interconnected target  "
          f"delta={INTER_DELTA} (paper figure only, NOT used by optimiser) ===")
    arr_v_bound = voxelise_to_array(
        build_interconnected_boundary(delta=INTER_DELTA,
                                      bound_radius_bins=INTER_VISUAL_BOUND_RADIUS_BINS)
    )
    arr_v_intra = voxelise_to_array(
        build_interconnected_intra(intra_radius_bins=INTER_VISUAL_INTRA_RADIUS_BINS)
    )
    arr_v = union_pore_arrays(arr_v_bound, arr_v_intra)
    p_v_bound_3d = float((arr_v_bound > PORE_THRESHOLD).sum() / arr_v_bound.size)
    p_v_intra_3d = float((arr_v_intra > PORE_THRESHOLD).sum() / arr_v_intra.size)
    p_v_3d = float((arr_v > PORE_THRESHOLD).sum() / arr_v.size)
    p_v_slice = slice_to_png(arr_v, OUTPUT_DIR / "synthetic_interconnected_visual.png")
    print(f"  3D boundary-only porosity = {p_v_bound_3d:.4f}")
    print(f"  3D intra-only    porosity = {p_v_intra_3d:.4f}")
    print(f"  3D combined      porosity = {p_v_3d:.4f}  (overlap loss: "
          f"{(p_v_bound_3d + p_v_intra_3d - p_v_3d):.4f})")
    print(f"  2D slice combined porosity = {p_v_slice:.4f}")
    print(f"  output: synthetic_interconnected_visual.png")

    # ----- ground_truth.json ------------------------------------------------
    canonical_run = next((r for r in interconnected_runs
                          if abs(r["DELTA"] - INTER_DELTA) < 1e-9),
                         interconnected_runs[0])
    gt = {
        "common": {
            "L_DIM": L_DIM,
            "N_VOX": N_VOX,
            "SEED": SEED,
            "SLICE_AXIS": SLICE_AXIS,
            "PORE_THRESHOLD": PORE_THRESHOLD,
        },
        "distributed": {
            "RADII": DIST_RADII,
            "TARGET_P_TOTAL": DIST_TARGET_P,
            "RADIUS_BINS": DIST_RADIUS_BINS,
            "REALISED_3D_P": p_d_3d,
            "REALISED_SLICE_P": p_d,
            "output_png": "synthetic_distributed.png",
        },
        "interconnected": {
            "LAG_R": INTER_LAG_R,
            "LAG_PHI": INTER_LAG_PHI,
            "DELTA_CANONICAL": INTER_DELTA,
            "BOUND_RADII": INTER_BOUND_RADII,
            "BOUND_RAW_TARGET_P": INTER_BOUND_RAW_TARGET_P,
            "BOUND_TARGET_P_NOMINAL": INTER_BOUND_TARGET_P_NOMINAL,
            "BOUND_RADIUS_BINS": INTER_BOUND_RADIUS_BINS,
            "INTRA_RADII": INTER_INTRA_RADII,
            "INTRA_TARGET_P": INTER_INTRA_TARGET_P,
            "INTRA_RADIUS_BINS": INTER_INTRA_RADIUS_BINS,
            # Canonical-delta convenience fields (back-compat with old readers).
            "DELTA": canonical_run["DELTA"],
            "REALISED_3D_P_BOUNDARY": canonical_run["REALISED_3D_P_BOUNDARY"],
            "REALISED_3D_P_INTRA": canonical_run["REALISED_3D_P_INTRA"],
            "REALISED_3D_P_COMBINED": canonical_run["REALISED_3D_P_COMBINED"],
            "REALISED_SLICE_P": canonical_run["REALISED_SLICE_P"],
            "output_png": "synthetic_interconnected.png",
            # Full sweep.
            "delta_sweep": interconnected_runs,
        },
        "interconnected_visual": {
            "purpose": ("Visual-rich variant for the paper figure only; NOT "
                        "used as the optimisation target."),
            "LAG_R": INTER_LAG_R,
            "LAG_PHI": INTER_LAG_PHI,
            "DELTA": INTER_DELTA,
            "BOUND_RADII": INTER_VISUAL_BOUND_RADII,
            "BOUND_RAW_TARGET_P": INTER_VISUAL_BOUND_RAW_TARGET_P,
            "BOUND_RADIUS_BINS": INTER_VISUAL_BOUND_RADIUS_BINS,
            "INTRA_RADII": INTER_VISUAL_INTRA_RADII,
            "INTRA_TARGET_P": INTER_VISUAL_INTRA_TARGET_P,
            "INTRA_RADIUS_BINS": INTER_VISUAL_INTRA_RADIUS_BINS,
            "REALISED_3D_P_BOUNDARY": p_v_bound_3d,
            "REALISED_3D_P_INTRA": p_v_intra_3d,
            "REALISED_3D_P_COMBINED": p_v_3d,
            "REALISED_SLICE_P": p_v_slice,
            "output_png": "synthetic_interconnected_visual.png",
        },
    }
    with open(OUTPUT_DIR / "ground_truth.json", "w") as f:
        json.dump(gt, f, indent=2)

    print(f"\n[SAVE] {OUTPUT_DIR}/")
    print(f"       synthetic_distributed.png")
    for r in interconnected_runs:
        print(f"       {r['output_png']}  (delta={r['DELTA']}, "
              f"delta*={r['DELTA_STAR']:.3f}, p_b={r['REALISED_3D_P_BOUNDARY']:.4f})")
    print(f"       synthetic_interconnected.png             (canonical matched-form, "
          f"mirrors delta={INTER_DELTA})")
    print(f"       synthetic_interconnected_visual.png      (visual-rich, paper figure only, "
          f"p_total={p_v_3d:.3f})")
    print(f"       ground_truth.json")


if __name__ == "__main__":
    main()
