import os
import sac_de_billes
import merope
import csv
import time
from utils_microstructure import run_amitex, go_to_dir

USE_AMITEX = False   # set True to run Amitex after voxelization

# --- Geometry and discretization
L_RVE = 10
n3D = 100                       # Number of voxels per axis

seed = 0                        # RNG seed for reproducibility

# --- Rules for voxelization and homogenization
voxel_rule = merope.vox.VoxelRule.Average
homogRule  = merope.HomogenizationRule.Voigt  # Voigt = upper bound

# --- Output settings
folder_name = "polycrystal"   # Top-level results folder
vtkname     = "Zone.vtk"      # VTK file of voxelized structure
fileCoeff   = "Coeffs.txt"    # Material coefficients output

# --- Microstructure parameters
R_pores_GB = 0.03    # Intergranular pore radius
P_pores_GB = 0.40    # Intergranular pore volume fraction

R_pores_IG1 = 0.05
P_pores_IG1 = 0.03

R_pores_IG2 = 0.30
P_pores_IG2 = 0.19

# Laguerre tessellation (grains)
R_grain = 1.0
P_grain = 1.0

# Boundary layer
grain_boundary_layer = 0.003     # thickness of grain-boundary delta layer

# --- Phase IDs
grain_phase  = 0
temp_phase   = 1
incl_phase   = 2
delta_phase  = 3

# --- Thermal conductivities of phases
Kmatrix = 1.0     # solid
Kgases  = 1e-3    # pores
K       = [Kmatrix, Kmatrix, Kgases]   # one entry per phase

# ---------------------------------------------------------------------------
# STRUCTURE GENERATION & VOXELIZATION
# ---------------------------------------------------------------------------

def cracked_structure(n3D, length, seed, delta, voxel_rule, K, vtkname, fileCoeff, INTRA=False, INTER=False):

    """
    Build a polycrystal with:
        - Intergranular pores
        - Intragranular pores
        - Grain-boundary delta layer
    Then voxelize the structure and export as VTK.

    """

    # --- Step 1: Intergranular spherical multiInc_intergranular (large pores)
    if INTER:
        intergranular = merope.SphereInclusions_3D()
        intergranular.setLength(length)
        intergranular.fromHisto(
            seed, 
            sac_de_billes.TypeAlgo.BOOL, 
            0., 
            [[R_pores_GB, P_pores_GB]], 
            [incl_phase]
        )

        # pores of incl_phase)
        multiInc_intergranular = merope.MultiInclusions_3D()
        multiInc_intergranular.setInclusions(intergranular)

    # --- Step 2: Intragranular spherical multiInc_intergranular (small pores)
    if INTRA:
        intragranular = merope.SphereInclusions_3D()
        intragranular.setLength(length)
        intragranular.fromHisto(
            seed, 
            sac_de_billes.TypeAlgo.BOOL, 
            0., 
            [[R_pores_IG1, P_pores_IG1],  # R, P --> phase 1
            [R_pores_IG2, P_pores_IG2]],  # R, P --> phase 1
            [temp_phase, temp_phase]      # phase 1, phase 1
        )

        # pores of phase 1 (temp_phase)
        multiInc_intragranular = merope.MultiInclusions_3D()
        multiInc_intragranular.setInclusions(intragranular)

    # --- Step 3: Laguerre tessellation for grains
    grain = merope.SphereInclusions_3D()
    grain.setLength(length)
    grain.fromHisto(
        seed, 
        sac_de_billes.TypeAlgo.RSA, 
        0., 
        [[R_grain, P_grain]], 
        [temp_phase] # anche i grani in phase 1 (temp)
    )

    polyCrystal = merope.LaguerreTess_3D(length, grain.getSpheres())

    multiInc_grain = merope.MultiInclusions_3D()
    multiInc_grain.setInclusions(polyCrystal)

    # Add boundary layer of thickness delta to the grains, and phase 1 (temp)
    multiInc_grain.addLayer(multiInc_grain.getAllIdentifiers(), delta_phase, delta) # layer in delta phase
    multiInc_grain.changePhase(multiInc_grain.getAllIdentifiers(), [temp_phase for _ in multiInc_grain.getAllIdentifiers()]) # grani tutti nella stessa fase
    # multiInc_grain.changePhase(multiInc_grain.getAllIdentifiers(),multiInc_grain.getAllIdentifiers()) # grain in fasi diverse (per vederli su paraview)

    # --- Step 4: Merge structures
    if INTER:
        structure3 = merope.Structure_3D(
            multiInc_intergranular, # inter-granulari, phase 2 (incl)
            multiInc_grain,  # grain (phase 1) + delta (phase 3)
            {incl_phase: grain_phase,  # old phase --> new phase
            delta_phase: grain_phase} # old phase --> new phase
        )

    structure  = merope.Structure_3D(
        structure3, 
        multiInc_intragranular, 
        {grain_phase: incl_phase}
    )

    # --- Step 5: Voxelization
    voxellation = [n3D, n3D, n3D]
    gridParams = merope.vox.create_grid_parameters_N_L_3D(voxellation, [L_RVE, L_RVE, L_RVE])
    grid = merope.vox.GridRepresentation_3D(structure, gridParams, voxel_rule)

    # --- Step 6: Print resolution parameters
    voxel_size = [length[i] / n3D for i in range(3)]

    print("\n==================== SIMULATION SUMMARY ====================\n")

    print("--- Resolution parameters ---")
    print(f"Grid size      : {n3D} x {n3D} x {n3D} → {n3D**3:,} voxels")
    print(f"RVE dimensions : {length} (units)")
    print(f"Voxel size     : {voxel_size} (units per axis)")
    print(f"Delta layer    : {delta} (thickness)\n")

    # --- Phase fractions ---
    print("--- Phase fractions ---")
    analyzer = merope.vox.GridAnalyzer_3D()
    analyzer.compute_percentages(grid)

    print("\n=============================================================\n")


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
# MAIN EXECUTION
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    results = []
    start_time = time.time()

    # --- Create results directory
    os.makedirs(folder_name, exist_ok=True)
    go_to_dir(folder_name)
    os.makedirs(str(n3D), exist_ok=True)
    go_to_dir(str(n3D))

    # --- Build & voxelize structure
    cracked_structure(
        n3D, [L_RVE, L_RVE, L_RVE], seed, 
        grain_boundary_layer, voxel_rule, K, vtkname, fileCoeff
    )

    if USE_AMITEX:
        k_eff = run_amitex()
        print("Effective thermal conductivity:", k_eff)
        error = 0.0  # replace with real error if available
    else:
        k_eff = None
        error = None

    elapsed = time.time() - start_time

    # Compute resolution values
    a = seed      # or whatever parameter “a” refers to
    num_voxels = n3D**3
    L_voxel = L_RVE / n3D

    # Example porosity calculations
    porosity_calc = None  # could parse from analyzer.get_percentages(grid)
    R_pore = R_pores_GB

    # Append summary row
    results.append((
        "Voigt",        # rule_name
        L_RVE,
        a,
        num_voxels,
        L_voxel,
        k_eff,
        error,
        porosity_calc,
        R_pore,
        elapsed
    ))

    with open("summary.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "Rule", "L_RVE", "a", "N_voxel", "L_voxel",
            "K_mean", "Error", "Porosity_calc", "Porosity", "R_pore", "Elapsed_s"
        ])
        writer.writerows(results)

    print("→ Summary written to summary.csv")
