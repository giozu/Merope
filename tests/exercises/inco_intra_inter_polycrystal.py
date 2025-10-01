"""
intra_inter_polycrystal.py

Build a polycrystal with:
- inter-granular spherical pores (phase 2),
- intra-granular spherical pores (phase 1),
- Laguerre grains (phase 0) with a boundary layer (phase 3).

Then voxelize with Merope and (optionally) run Amitex to get the effective
thermal conductivity. Use this to study K vs total porosity, intra/inter
porosity, and boundary thickness (delta).

Notes
-----
- Homogenization uses the Voigt rule (requires voxel_rule = Average).
- Phase indices used here:
    phase_grains   = 0
    phase_intra    = 1   (pores inside grains)
    phase_inter    = 2   (pores between grains)
    phase_boundary = 3   (grain-boundary layer before remapping)
"""

import os
import shutil
import sac_de_billes
import merope
import interface_amitex_fftp.amitex_wrapper as amitex
import interface_amitex_fftp.post_processing as amitex_out


# -----------------------------------------------------------------------------
# USER INPUTS
# -----------------------------------------------------------------------------

# RVE and voxelization
domain_size = [10, 10, 10]    # Lx, Ly, Lz of the RVE
num_voxels  = 200             # grid resolution → num_voxels^3 voxels (heavy if 300)
seed        = 0

# voxel & homogenization rules
voxel_rule = merope.vox.VoxelRule.Average
homog_rule = merope.HomogenizationRule.Voigt  # requires voxel_rule = Average

# output names (created under ./intra_inter_polycrystal/)
results_folder = "intra_inter_polycrystal"
vtk_filename   = "structure.vtk"
coeff_filename = "Coeffs.txt"

# materials (matrix duplicated in list to match printer expectations)
k_matrix = 1.0
k_gas    = 1e-3
K_list   = [k_matrix, k_matrix, k_gas]

# microstructure parameters
# inter-granular pores (global porosity channels)
inter_R   = 0.03
inter_phi = 0.40

# intra-granular pores (inside grains)
intra_R   = 0.10
intra_phi = 0.19

# grains (Laguerre)
grain_R   = 1.0      # controls grain size through sphere histogram
grain_phi = 1.0      # must be 1.0 to fill the RVE with grains before porosity

# grain-boundary thickness
delta = 0.003        # physical thickness of boundary layer

# toggle Amitex run
USE_AMITEX = True   # set True to run Amitex after voxelization


# phase indices (keep consistent!)
phase_grains   = 0
phase_intra    = 1
phase_inter    = 2
phase_boundary = 3


# -----------------------------------------------------------------------------
# HELPERS
# -----------------------------------------------------------------------------

def fresh_folder(path: str):
    """Create an empty folder at `path` (remove if exists)."""
    if os.path.exists(path):
        shutil.rmtree(path)
    os.mkdir(path)


# -----------------------------------------------------------------------------
# CORE: build structure and voxelize
# -----------------------------------------------------------------------------

def build_and_voxelize():
    """
    1) Make inter-granular pore field (phase 2) as independent MultiInclusions.
    2) Make intra-granular pores (phase 1) as independent MultiInclusions.
    3) Build Laguerre tessellation (grains, phase 0), add boundary layer (phase 3),
       then remap boundary to phase 0 (so final grain phase includes boundaries).
    4) Combine: mask inter-pores onto grains, then overlay intra-pores into the
       result; finally voxelize and export VTK + coefficients.
    """

    # -- inter-granular pores --------------------------------------------------
    sph_inter = merope.SphereInclusions_3D()
    sph_inter.setLength(domain_size)
    sph_inter.fromHisto(
        seed,
        sac_de_billes.TypeAlgo.BOOL,
        0.0,
        [[inter_R, inter_phi]],
        [phase_inter],
    )
    inter_pores = merope.MultiInclusions_3D()
    inter_pores.setInclusions(sph_inter)

    # -- intra-granular pores --------------------------------------------------
    sph_intra = merope.SphereInclusions_3D()
    sph_intra.setLength(domain_size)
    sph_intra.fromHisto(
        seed,
        sac_de_billes.TypeAlgo.BOOL,
        0.0,
        [[intra_R, intra_phi]],
        [phase_intra],
    )
    intra_pores = merope.MultiInclusions_3D()
    intra_pores.setInclusions(sph_intra)

    # -- grains via Laguerre tessellation -------------------------------------
    sph_for_laguerre = merope.SphereInclusions_3D()
    sph_for_laguerre.setLength(domain_size)
    sph_for_laguerre.fromHisto(
        seed,
        sac_de_billes.TypeAlgo.RSA,
        0.0,
        [[grain_R, grain_phi]],
        [1],  # temporary id for Laguerre seeds; not a material phase
    )
    poly = merope.LaguerreTess_3D(domain_size, sph_for_laguerre.getSpheres())
    grains = merope.MultiInclusions_3D()
    grains.setInclusions(poly)

    # Add boundary layer around all grain cells (phase 3)
    ids = grains.getAllIdentifiers()
    grains.addLayer(ids, phase_boundary, delta)

    # Set all grain cells (the cores) to phase 1 (arbitrary tag),
    # then later we remap phase 1 and phase 3 both to phase 0 in the structure.
    grains.changePhase(ids, [1 for _ in ids])

    # -- combine: inter-pores + grains (remap) --------------------------------
    # First, create a structure where inter-pores (2) and boundary (3) are remapped to grains (0).
    # This makes a "solid-with-interpores" structure.
    remap_inter_and_boundary_to_grains = {phase_inter: phase_grains, phase_boundary: phase_grains}
    struct_inter_on_grains = merope.Structure_3D(inter_pores, grains, remap_inter_and_boundary_to_grains)

    # Then, overlay intra-pores (phase 1) onto that structure, but finally remap
    # the temporary '1' to true pore phase (2). We also ensure anything '0' stays '2' if asked.
    # Here we simply convert phase 0 coming from the overlay to 2 to force pores dominance
    # where intra inclusions land.
    final_overlay_remap = {0: 2}
    structure = merope.Structure_3D(struct_inter_on_grains, intra_pores, final_overlay_remap)

    # -- voxelization ----------------------------------------------------------
    grid_params = merope.vox.create_grid_parameters_N_L_3D(
        [num_voxels, num_voxels, num_voxels], domain_size
    )
    grid = merope.vox.GridRepresentation_3D(structure, grid_params, voxel_rule)

    analyzer = merope.vox.GridAnalyzer_3D()
    phase_fracs = analyzer.compute_percentages(grid)
    analyzer.print_percentages(grid)

    # store homogenization coefficients onto grid (useful for vtk export)
    grid.apply_homogRule(homog_rule, K_list)

    # export segmented VTK + coeffs file (for Amitex)
    printer = merope.vox.vtk_printer_3D()
    printer.printVTK_segmented(grid, vtk_filename, coeff_filename, nameValue="MaterialId")

    return phase_fracs


# -----------------------------------------------------------------------------
# AMITEX
# -----------------------------------------------------------------------------

def run_amitex():
    """Run Amitex and return the homogenized conductivity matrix."""
    np = 2
    amitex.computeThermalCoeff(vtk_filename, np)
    return amitex_out.printThermalCoeff(".")


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------

def main():
    # set up ./Result/<num_voxels>/ as working folder
    fresh_folder(results_folder)
    os.chdir(results_folder)
    fresh_folder(str(num_voxels))
    os.chdir(str(num_voxels))

    # build and voxelize
    phase_fracs = build_and_voxelize()
    print("Phase fractions:", phase_fracs)

    # optional homogenization
    if USE_AMITEX:
        H = run_amitex()
        print("Homogenized conductivity matrix:\n", H)

    # back to original directory (optional)
    os.chdir("..")
    os.chdir("..")


if __name__ == "__main__":
    main()
