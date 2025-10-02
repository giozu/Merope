###############################################################
# MICROSTRUCTURE GENERATION & THERMAL CONDUCTIVITY ANALYSIS
#
# This script builds a porous polycrystal RVE (Representative 
# Volume Element) using the Merope library. It voxelizes the 
# structure, analyzes porosity and phase fractions, and can 
# compute effective thermal conductivity via Amitex-FFTP.
#
# Outputs:
#   - Thermal conductivity (homogenized)
#   - Porosity fraction
#   - Delta layer thickness
#   - Grain volume distribution
#
# Notes:
#   - Phase 1 = porous phase
#   - Delta layer = thin boundary region at grain interfaces
###############################################################

# ---------------------------------------------------------------------------
# IMPORTS
# ---------------------------------------------------------------------------
import os
import sac_de_billes
import merope
import interface_amitex_fftp.amitex_wrapper as amitex
import interface_amitex_fftp.post_processing as amitex_out

USE_AMITEX = False   # set True to run Amitex after voxelization

# ---------------------------------------------------------------------------
# GLOBAL PARAMETERS
# ---------------------------------------------------------------------------

# --- Geometry and discretization
L = [10, 10, 10]        # RVE dimensions (cube 10x10x10 units)
n3D = 300               # Number of voxels per axis → grid = 300³
seed = 0                # RNG seed for reproducibility

# --- Rules for voxelization and homogenization
voxel_rule = merope.vox.VoxelRule.Average
homogRule  = merope.HomogenizationRule.Voigt  # Voigt = upper bound

# --- Output settings
folder_name = "polycrystal"   # Top-level results folder
vtkname     = "Zone.vtk"      # VTK file of voxelized structure
fileCoeff   = "Coeffs.txt"    # Material coefficients output

# --- Microstructure parameters
inclR   = 0.03    # Intergranular pore radius
inclPhi = 0.4     # Intergranular pore volume fraction

# Intragranular pores: list of [volume fraction, radius]
intraInclRphi = [
    [0.05, 0.03],  # small pores
    [0.30, 0.19]   # large pores
]

# Laguerre tessellation (grains)
lagR   = 1.0
lagPhi = 1.0      # must be 1 → fill matrix before porosity insertion

# Boundary layer
delta = 0.003     # thickness of grain-boundary delta layer

# Target porosity
porosity = 0.1    # overall target porosity (fraction)

# --- Phase IDs
grains_phase = 0
incl_phase   = 2
delta_phase  = 3

# --- Thermal conductivities of phases
Kmatrix = 1.0     # solid
Kgases  = 1e-3    # pores
K       = [Kmatrix, Kmatrix, Kgases]   # one entry per phase


# ---------------------------------------------------------------------------
# UTILS
# ---------------------------------------------------------------------------

def go_to_dir(name_dir: str):
    """Change working directory to `name_dir` and print confirmation."""
    os.chdir(name_dir)
    print("→ Entered directory:", name_dir)


# ---------------------------------------------------------------------------
# STRUCTURE GENERATION & VOXELIZATION
# ---------------------------------------------------------------------------

def Crack_structure_Voxellation(n3D, L, seed, inclRphi, lagRphi,
                                incl_phase, grains_phase, delta_phase,
                                delta, voxel_rule, K, vtkname, fileCoeff):
    """
    Build a polycrystal with:
        - Intergranular pores
        - Intragranular pores
        - Grain-boundary delta layer
    Then voxelize the structure and export as VTK.

    Parameters
    ----------
    n3D : int
        Number of voxels per axis.
    L : list
        Physical size of the RVE [Lx, Ly, Lz].
    seed : int
        RNG seed for reproducibility.
    inclRphi : [R, phi]
        Intergranular pore radius and volume fraction.
    lagRphi : [R, phi]
        Laguerre tessellation parameters for grains.
    delta : float
        Thickness of boundary layer.
    voxel_rule : merope.vox.VoxelRule
        Rule for voxelization.
    K : list
        Thermal conductivity per phase.
    vtkname : str
        Output VTK filename.
    fileCoeff : str
        Output coefficients filename.
    """

    # --- Step 1: Intergranular spherical inclusions (large pores)
    sphIncl2 = merope.SphereInclusions_3D()
    sphIncl2.setLength(L)
    sphIncl2.fromHisto(seed, sac_de_billes.TypeAlgo.BOOL, 0., [inclRphi], [2])
    multiInclusions2 = merope.MultiInclusions_3D()
    multiInclusions2.setInclusions(sphIncl2)

    # --- Step 2: Intragranular spherical inclusions (small pores)
    sphIncl4 = merope.SphereInclusions_3D()
    sphIncl4.setLength(L)
    sphIncl4.fromHisto(seed, sac_de_billes.TypeAlgo.BOOL, 0., intraInclRphi, [1, 1])
    multiInclusions4 = merope.MultiInclusions_3D()
    multiInclusions4.setInclusions(sphIncl4)

    # --- Step 3: Laguerre tessellation for grains
    sphIncl = merope.SphereInclusions_3D()
    sphIncl.setLength(L)
    sphIncl.fromHisto(seed, sac_de_billes.TypeAlgo.RSA, 0., [lagRphi], [1])
    polyCrystal = merope.LaguerreTess_3D(L, sphIncl.getSpheres())

    multiInclusions = merope.MultiInclusions_3D()
    multiInclusions.setInclusions(polyCrystal)

    # Add boundary layer of thickness delta
    multiInclusions.addLayer(multiInclusions.getAllIdentifiers(), delta_phase, delta)
    # Assign all grains as solid initially
    multiInclusions.changePhase(multiInclusions.getAllIdentifiers(),
                                [1 for _ in multiInclusions.getAllIdentifiers()])

    # --- Step 4: Merge structures
    structure3 = merope.Structure_3D(multiInclusions2, multiInclusions, {2: 0, 3: 0})
    structure  = merope.Structure_3D(structure3, multiInclusions4, {0: 2})

    # --- Step 5: Voxelization
    gridParams = merope.vox.create_grid_parameters_N_L_3D([n3D, n3D, n3D], L)
    grid = merope.vox.GridRepresentation_3D(structure, gridParams, voxel_rule)

    # Phase fraction analysis
    analyzer = merope.vox.GridAnalyzer_3D()
    analyzer.compute_percentages(grid)
    analyzer.print_percentages(grid)

    # Homogenization with conductivity
    grid.apply_homogRule(homogRule, K)

    # Export to VTK
    printer = merope.vox.vtk_printer_3D()
    printer.printVTK_segmented(grid, vtkname, fileCoeff, nameValue="MaterialId")


# ---------------------------------------------------------------------------
# THERMAL CONDUCTIVITY (AMITEX)
# ---------------------------------------------------------------------------

def ThermalAmitex():
    """Run Amitex-FFTP solver to compute effective thermal conductivity."""
    nproc = 2
    amitex.computeThermalCoeff(vtkname, nproc)
    return amitex_out.printThermalCoeff(".")


# ---------------------------------------------------------------------------
# MAIN EXECUTION
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    # --- Create results directory
    os.mkdir(folder_name)
    go_to_dir(folder_name)
    os.mkdir(str(n3D))
    go_to_dir(str(n3D))

    # --- Build & voxelize structure
    Crack_structure_Voxellation(
        n3D, L, seed, [inclR, inclPhi], [lagR, lagPhi],
        incl_phase, grains_phase, delta_phase,
        delta, voxel_rule, K, vtkname, fileCoeff
    )

    # --- Compute conductivity
    if USE_AMITEX:
        ThermalAmitex()
