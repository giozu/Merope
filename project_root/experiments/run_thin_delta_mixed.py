"""
run_thin_delta_mixed.py
=======================
Generate the thin-delta MIXED interconnected case (boundary + intra-granular
pores) at delta_star = 0.10 -- in the morphology-controlled regime of the
joint sigmoidal fit -- and run AMITEX to measure K_eff.

Purpose
-------
The headline figure of the paper requires an interconnected case where
K_eff falls visibly below the calibrated Loeb baseline. The matched-form
case used for the optimiser recovery test sits at delta_star ~ 0.32, well
past the morphological transition, so its K_eff is only ~3% below Loeb.
This script generates a SEPARATE case (not used for the optimisation
recovery test) at thin delta + mixed pore populations:

  * Boundary phase   :  delta = 0.3,  delta_star = 0.10
                       boundary porosity p_b ~ 0.14 (target)
  * Intra-granular   :  RSA spherical pores, p_intra ~ 0.07 (target)
  * Total porosity   :  p_total ~ 0.21

The geometry follows generate_interconnected_structure() (the inter+intra
+ grains pipeline already used by run_interconnected_porosity.py and the
inco_intra_inter_polycrystal test) so AMITEX can ingest the result directly
through the standard builder.voxellate() pipeline.

Outputs (written to Results_ThinDelta_Mixed/)
---------------------------------------------
  structure.vtk        -- segmented voxel grid for AMITEX.
  Coeffs.txt           -- thermal conductivity table.
  amitex_run/          -- AMITEX run directory.
  summary.txt          -- target/realised porosity, K_eff, parameters.
  slice.png            -- midplane 2D slice for the paper figure.

Run
---
    cd ~/Merope
    source Env_Merope.sh
    export PYTHONPATH=$PYTHONPATH:./project_root
    python project_root/experiments/run_thin_delta_mixed.py
    # add --no-solver to skip AMITEX (geometry-only check)
    # add --n-cpus 32 to use full 32-core hardware

Notes
-----
The boundary-phase target is built as Boolean overlapping spheres + the
Mérope `Structure_3D({2:0, 3:0})` overlay, which clips the spheres to the
grain-boundary band of thickness `delta`. The realised boundary porosity
is therefore much smaller than the raw Boolean target; the script defaults
crank INTER_PHI=0.65 to land at realised ~0.14 at delta=0.3 / r=0.40
(empirical ratio realised/raw ~ 0.22 from make_synthetic_targets.py
ground_truth.json). If the realised fractions deviate from the targets,
re-run with `--inter-phi <new>` and `--intra-phi <new>` to dial them in
before launching AMITEX.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import merope  # noqa: E402

from core.geometry import MicrostructureBuilder  # noqa: E402
from core.solver import ThermalSolver  # noqa: E402
from core.utils import ProjectManager  # noqa: E402


# --- RVE & voxelisation -----------------------------------------------------
L_DIM = [10.0, 10.0, 10.0]
N_VOX = 200
SEED = 0

# --- Geometry: thin delta + mixed pore populations -------------------------
GRAIN_R = 3.0       # matches make_synthetic_targets.py (LAG_R)
GRAIN_PHI = 1.0
DELTA = 0.3         # delta_star = DELTA / GRAIN_R = 0.10  (morphology-controlled)

INTER_R = 0.40      # boundary pore radius (monodisperse)
INTER_PHI = 0.65    # Boolean RAW target. Mérope's Structure_3D({2:0,3:0})
                     # overlay clips these spheres to the GB band of thickness
                     # delta, so the REALISED boundary porosity is much smaller.
                     # Empirical ratio at delta=0.3, r=0.40 is realised/raw ~0.22
                     # (from make_synthetic_targets.py ground_truth.json), so
                     # raw=0.65 -> realised ~0.14 as required.

INTRA_R = 0.10      # intra-granular pore radius
INTRA_PHI = 0.07    # target intra porosity (Boolean, not clipped)

# --- Thermal conductivities (legacy convention) ----------------------------
# Phase 0 = matrix (solid grains), phase 1 = transient intra tag, phase 2 = pores.
K_MATRIX = 1.0
K_GAS = 1e-3
K_THERMAL = [K_MATRIX, K_MATRIX, K_GAS]


def save_midplane_slice(grid_repr, builder: MicrostructureBuilder, out_path: Path) -> float:
    """Save a midplane slice as a binary PNG (matches the synthetic figure look)."""
    conv = merope.vox.NumpyConverter_3D()
    arr3d = conv.compute_RealField(grid_repr).reshape((builder.n3D,) * 3, order="C")
    sl = arr3d[:, :, builder.n3D // 2]
    # K=1 => matrix (white); K=K_GAS => pore (black). Threshold at 0.5.
    binary = np.where(sl > 0.5, 255, 0).astype(np.uint8)
    Image.fromarray(binary).save(out_path)
    return float((binary == 0).sum() / binary.size)


def main() -> None:
    parser = argparse.ArgumentParser(description="Thin-delta mixed-porosity AMITEX run.")
    parser.add_argument("--no-solver", action="store_true",
                        help="Skip AMITEX, just build geometry and report fractions.")
    parser.add_argument("--n-cpus", type=int, default=4,
                        help="CPU cores for AMITEX (default: 4; use 32 on 32-core HW).")
    parser.add_argument("--inter-phi", type=float, default=INTER_PHI,
                        help=f"Boundary-pore target volume fraction (default: {INTER_PHI}).")
    parser.add_argument("--intra-phi", type=float, default=INTRA_PHI,
                        help=f"Intra-pore target volume fraction (default: {INTRA_PHI}).")
    parser.add_argument("--delta", type=float, default=DELTA,
                        help=f"Grain-boundary thickness (default: {DELTA}).")
    args = parser.parse_args()

    output_dir = Path("Results_ThinDelta_Mixed")
    pm = ProjectManager()
    pm.cleanup_folder(str(output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)

    builder = MicrostructureBuilder(L=L_DIM, n3D=N_VOX, seed=SEED)
    delta_star = args.delta / GRAIN_R

    print("=" * 60)
    print("  Thin-delta MIXED interconnected — geometry build")
    print("=" * 60)
    print(f"  grain_radius        : {GRAIN_R}")
    print(f"  delta               : {args.delta}  (delta* = {delta_star:.3f})")
    print(f"  inter_radius / phi  : {INTER_R} / {args.inter_phi}")
    print(f"  intra_radius / phi  : {INTRA_R} / {args.intra_phi}")
    print(f"  N_VOX               : {N_VOX}")
    print(f"  output_dir          : {output_dir}")

    struct = builder.generate_interconnected_structure(
        inter_radius=INTER_R,
        inter_phi=args.inter_phi,
        intra_radius=INTRA_R,
        intra_phi=args.intra_phi,
        grain_radius=GRAIN_R,
        grain_phi=GRAIN_PHI,
        delta=args.delta,
    )

    vtk_path = output_dir / "structure.vtk"
    coeffs_path = output_dir / "Coeffs.txt"
    fractions = builder.voxellate(struct, K_THERMAL, vtk_path=vtk_path, coeffs_path=coeffs_path)
    p_pore = float(fractions.get(2, 0.0))
    print(f"\n  Realised pore fraction (phase 2)  : {p_pore:.4f}")

    # Slice for the figure (rebuild the grid we just voxelised so we can extract the array)
    grid_repr = merope.vox.GridRepresentation_3D(struct, builder.grid_params, merope.vox.VoxelRule.Average)
    grid_repr.apply_homogRule(merope.HomogenizationRule.Voigt, K_THERMAL)
    slice_path = output_dir / "slice.png"
    p_slice = save_midplane_slice(grid_repr, builder, slice_path)
    print(f"  2D slice porosity                   : {p_slice:.4f}")
    print(f"  Slice PNG saved                     : {slice_path}")

    results: dict = {}
    if not args.no_solver:
        print("\n  Running AMITEX...")
        solver = ThermalSolver(n_cpus=args.n_cpus)
        # ThermalSolver doesn't chdir internally; AMITEX resolves Coeffs.txt
        # against cwd. Use pm.cd to ensure cwd is the directory holding both
        # structure.vtk and Coeffs.txt.
        with pm.cd(str(output_dir.resolve())):
            results = solver.solve(vtk_file="structure.vtk")
        print(f"  Kxx={results.get('Kxx', 0):.5f}  "
              f"Kyy={results.get('Kyy', 0):.5f}  "
              f"Kzz={results.get('Kzz', 0):.5f}  "
              f"Kmean={results.get('Kmean', 0):.5f}")

    summary_path = output_dir / "summary.txt"
    with open(summary_path, "w") as fp:
        fp.write("Thin-delta mixed interconnected case\n")
        fp.write("=" * 50 + "\n")
        fp.write(f"grain_radius           : {GRAIN_R}\n")
        fp.write(f"delta (absolute)       : {args.delta}\n")
        fp.write(f"delta_star (= d/L)     : {delta_star:.4f}\n")
        fp.write(f"inter_radius           : {INTER_R}\n")
        fp.write(f"inter_phi (target)     : {args.inter_phi}\n")
        fp.write(f"intra_radius           : {INTRA_R}\n")
        fp.write(f"intra_phi (target)     : {args.intra_phi}\n")
        fp.write(f"N_VOX                  : {N_VOX}\n")
        fp.write(f"\nRealised pore fraction (phase 2): {p_pore:.4f}\n")
        fp.write(f"2D slice porosity              : {p_slice:.4f}\n")
        if results:
            fp.write("\nAMITEX K_eff:\n")
            for k in ("Kxx", "Kyy", "Kzz", "Kmean"):
                fp.write(f"  {k}: {results.get(k, 0):.6f}\n")

    print(f"\nSummary written -> {summary_path}")
    print("Done.")


if __name__ == "__main__":
    main()
