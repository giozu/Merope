###############################################################
# FILE USED TO CALCULATE:
#   - Thermal conductivity K
#   - Porosity
#   - Delta layer thickness
#   - Grain volume distribution
#
# NOTE: Phase 1 is defined as the porous phase
###############################################################

import os
import sac_de_billes
import merope
import shutil
import numpy as np

import archi_merope as arch
import interface_amitex_fftp.amitex_wrapper as amitex
import interface_amitex_fftp.post_processing as amitex_out

# ---------------------------------------------------------------------------
# VOXEL & CELL INPUTS
# ---------------------------------------------------------------------------

L = [10, 10, 10]                 # RVE physical dimensions (cube of 10x10x10)
n3D = 300                        # number of voxels in each direction → grid = 300³
seed = 0                         # random seed for reproducibility

# voxelization rule: how to assign properties if voxel is mixed
voxel_rule = merope.vox.VoxelRule.Average

# homogenization rule: Voigt average (upper bound)
homogRule = merope.HomogenizationRule.Voigt  

# ---------------------------------------------------------------------------
# OUTPUT FOLDERS AND FILES
# ---------------------------------------------------------------------------

folder_name = 'results_2_rad_mixed_gen_0' # folder where results will be stored

vtkname = "crack_structure.vtk"           # voxelized structure output
fileCoeff = "Coeffs.txt"                  # material coefficients output

def go_to_dir(name_dir):
    """Change working directory and print the new path"""
    os.chdir(name_dir)
    print("→ Entered directory:", name_dir)

# ---------------------------------------------------------------------------
# INPUT PARAMETERS (microstructure)
# ---------------------------------------------------------------------------

# Intergranular inclusion radius (controls pores size)
inclR = 0.03    

# Volume fraction of inclusions (controls porosity fraction)
inclPhi = 0.4    

# Intragranular inclusions: [volume fraction, radius]
# Two sets here: small and large pores inside grains
intraInclRphi = [[0.05, 0.03], [0.3, 0.19]]

# Parameters for Laguerre tessellation (grains)
lagR = 1
lagPhi = 1  # must be 1 to fill the RVE with matrix before inserting porosity

# Fit parameters (from separate fitting file alpha_beta_gamma.py)
alpha = -0.5484
beta  =  1.9214
gamma =  0.3777

# Material inputs
porosity = 0.1          # target porosity (fraction)
delta = 0.003           # thickness of grain-boundary layer

# Group parameters into convenient structures
inclRphi = [inclR, inclPhi]
lagRphi  = [lagR, lagPhi]

# ---------------------------------------------------------------------------
# PHASE DEFINITIONS
# ---------------------------------------------------------------------------

incl_phase   = 2  # pores
delta_phase  = 3  # grain-boundary delta layer
grains_phase = 0  # solid grains

# Thermal conductivity of each phase
Kmatrix = 1.0     # solid matrix
Kgases  = 1e-3    # pore (gas-filled)
K = [Kmatrix, Kmatrix, Kgases]   # one entry per phase

# ---------------------------------------------------------------------------
# MAIN FUNCTION: BUILD MICROSTRUCTURE AND VOXELIZE
# ---------------------------------------------------------------------------

def Crack_structure_Voxellation(n3D, L, seed, inclRphi, lagRphi,
                                incl_phase, grains_phase, delta_phase,
                                delta, voxel_rule, K, vtkname, fileCoeff):
    """
    Build a porous polycrystal structure with cracks and intragranular porosity,
    then voxelize it and export to VTK.
    """

    # -------------------------------
    # Step 1. Add INTER-granular spherical inclusions (large pores)
    # -------------------------------
    sphIncl2 = merope.SphereInclusions_3D()
    sphIncl2.setLength(L)
    sphIncl2.fromHisto(seed, sac_de_billes.TypeAlgo.BOOL, 0., [inclRphi], [2])
    multiInclusions2 = merope.MultiInclusions_3D()
    multiInclusions2.setInclusions(sphIncl2)

    # -------------------------------
    # Step 2. Add INTRA-granular spherical inclusions (small pores)
    # -------------------------------
    sphIncl4 = merope.SphereInclusions_3D()
    sphIncl4.setLength(L)
    sphIncl4.fromHisto(seed, sac_de_billes.TypeAlgo.BOOL, 0., intraInclRphi, [1, 1])
    multiInclusions4 = merope.MultiInclusions_3D()
    multiInclusions4.setInclusions(sphIncl4)
    structure4 = merope.Structure_3D(multiInclusions4)

    # -------------------------------
    # Step 3. Build Laguerre tessellation (polycrystal grains)
    # -------------------------------
    sphIncl = merope.SphereInclusions_3D()
    sphIncl.setLength(L)
    sphIncl.fromHisto(seed, sac_de_billes.TypeAlgo.RSA, 0., [lagRphi], [1])
    polyCrystal = merope.LaguerreTess_3D(L, sphIncl.getSpheres())

    # Wrap into MultiInclusions object
    multiInclusions = merope.MultiInclusions_3D()
    multiInclusions.setInclusions(polyCrystal)

    # Number of grains
    N = len(multiInclusions.getAllIdentifiers())

    # Add grain-boundary layer of thickness delta
    multiInclusions.addLayer(multiInclusions.getAllIdentifiers(), 3, delta)

    # Set all inclusions as phase "1" initially
    multiInclusions.changePhase(multiInclusions.getAllIdentifiers(), 
                                [1 for _ in multiInclusions.getAllIdentifiers()])

    # -------------------------------
    # Step 4. Combine all structures
    # -------------------------------
    structure1 = merope.Structure_3D(multiInclusions)
    structure2 = merope.Structure_3D(multiInclusions2)

    # Map of phase transformations (dict: old_phase → new_phase)
    dictionnaire1 = {2: 0, 3: 0}  # pores & boundary layer → matrix
    structure3 = merope.Structure_3D(multiInclusions2, multiInclusions, dictionnaire1)

    # Final merge: combine polycrystal+pores+intragranular porosity
    dictionnaire = {0: 2}
    structure = merope.Structure_3D(structure3, structure4, dictionnaire)

    # -------------------------------
    # Step 5. Voxelization
    # -------------------------------
    gridParameters = merope.vox.create_grid_parameters_N_L_3D([n3D, n3D, n3D], L)
    grid = merope.vox.GridRepresentation_3D(structure, gridParameters, voxel_rule)

    # Analyze phase fractions
    analyzer = merope.vox.GridAnalyzer_3D()
    analyzer.compute_percentages(grid)
    analyzer.print_percentages(grid)

    # Apply homogenization rule with conductivities
    grid.apply_homogRule(homogRule, K)

    # Export voxelized structure as VTK
    my_printer = merope.vox.vtk_printer_3D()
    my_printer.printVTK_segmented(grid, vtkname, fileCoeff, nameValue="MaterialId")

# ---------------------------------------------------------------------------
# FUNCTION TO CALCULATE THERMAL CONDUCTIVITY USING AMITEX
# ---------------------------------------------------------------------------

def ThermalAmitex():
    """Run Amitex-FFTP solver to compute effective thermal conductivity"""
    number_of_processors = 2
    voxellation_of_zones = vtkname
    amitex.computeThermalCoeff(voxellation_of_zones, number_of_processors)
    homogenized_matrix = amitex_out.printThermalCoeff(".")
    return homogenized_matrix

# ---------------------------------------------------------------------------
# EXECUTION
# ---------------------------------------------------------------------------

# Create result directories
os.mkdir(folder_name)
go_to_dir(folder_name)
os.mkdir(str(n3D))
go_to_dir(str(n3D))

# Build structure and voxelize
Crack_structure_Voxellation(
    n3D, L, seed, inclRphi, lagRphi,
    incl_phase, grains_phase, delta_phase,
    delta, voxel_rule, K, vtkname, fileCoeff
)

# Uncomment this line to compute conductivity with Amitex
ThermalAmitex()

# Reference output
'''
results_2_rad_mixed_gen
300
Volume fraction : 0.510826
----------------------------
Volume fraction : 0.248397
----------------------------
----------------------------
AlgoRSA::proceed()
Fully packed
Nb of spheres : 93
Volume fraction : 0.389557
----------------------------
phase 0 : 77.294607%
phase 2 : 22.705393%
../
'''
