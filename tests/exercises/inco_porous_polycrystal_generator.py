import os
import sac_de_billes
import merope
import csv
import time
from utils_microstructure import run_amitex

USE_AMITEX = False   # set True to run Amitex after voxelization

# --- Geometry and discretization
L_RVE = 10
n3D = 300                       # Number of voxels per axis

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

def Crack_structure_Voxellation(n3D, length, seed,
                                incl_phase, grains_phase, delta_phase,
                                delta, voxel_rule, K, vtkname, fileCoeff):
    """
    Build a polycrystal with:
        - Intergranular pores
        - Intragranular pores
        - Grain-boundary delta layer
    Then voxelize the structure and export as VTK.

    """

    # --- Step 1: Intergranular spherical inclusions (large pores)
    intergranular = merope.SphereInclusions_3D()
    intergranular.setLength(length)
    intergranular.fromHisto(
        seed, 
        sac_de_billes.TypeAlgo.BOOL, 
        0., 
        [[R_pores_GB, P_pores_GB]], 
        [2]
    )
    multiInclusions2 = merope.MultiInclusions_3D()
    multiInclusions2.setInclusions(intergranular)

    # --- Step 2: Intragranular spherical inclusions (small pores)
    intragranular = merope.SphereInclusions_3D()
    intragranular.setLength(length)
    intragranular.fromHisto(
        seed, 
        sac_de_billes.TypeAlgo.BOOL, 
        0., 
        [[R_pores_IG1, P_pores_IG1],  # R, P --> phase 1
        [R_pores_IG2, P_pores_IG2]],  # R, P --> phase 1
        [1, 1]          # phase 1, phase 1
    )
    multiInclusions4 = merope.MultiInclusions_3D()
    multiInclusions4.setInclusions(intragranular)

    # --- Step 3: Laguerre tessellation for grains
    grain = merope.SphereInclusions_3D()
    grain.setLength(length)
    grain.fromHisto(
        seed, 
        sac_de_billes.TypeAlgo.RSA, 
        0., 
        [[R_grain, P_grain]], 
        [1]
    )
    polyCrystal = merope.LaguerreTess_3D(length, grain.getSpheres())

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
    gridParams = merope.vox.create_grid_parameters_N_L_3D([n3D, n3D, n3D], [L_RVE, L_RVE, L_RVE])
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
    Crack_structure_Voxellation(
        n3D, [L_RVE, L_RVE, L_RVE], seed, 
        incl_phase, grains_phase, delta_phase,
        delta, voxel_rule, K, vtkname, fileCoeff
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
        porosity,
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
