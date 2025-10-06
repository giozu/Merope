import os
import sac_de_billes
import merope
import csv
import time
from utils_microstructure import run_amitex, go_to_dir

USE_AMITEX = False

INTRA = True
INTER = True
GRAINS = True

# --- Microstructure parameters
R_pores_GB = 0.10       # Intergranular pore radius
P_pores_GB = 0.175      # Intergranular pore volume fraction

R_pores_IG1 = 0.1       # L_RVE / R_pore = L_RVE
P_pores_IG1 = 0.0175

# Laguerre tessellation (grains)
R_grain = 1.0
P_grain = 1.0

# Boundary layer: 
# spessore di una zona che poi scomparira
# è solo una zona che deve contenere le inclusioni a bordo grano, 
# la zona in sè dopo sparisce, torna ad essere nei grani
grain_boundary_layer = 0.2 # thickness of grain-boundary delta layer 

# --- Phase IDs
grain_phase  = 0
temp_phase   = 1 # se metto 1, non vedo niente in paraview
incl_phase   = 2
delta_phase  = 3

# --- Geometry and discretization
# L_RVE = 10*R_pores_IG1
L_RVE = 5
resolution = 20
# n_voxels = int(resolution * L_RVE / R_pores_IG1)
n_voxels = 150
L_voxel = L_RVE / n_voxels
a_param = R_pores_IG1 / L_voxel

print("--- Geometry and discretization ---")
print(f"L_voxel = {L_voxel}")
print(f"a_param = {a_param}")
print(f"L_RVE / R_pore_IG = {L_RVE / R_pores_IG1}")

seed = 0

# --- Rules for voxelization and homogenization
voxel_rule = merope.vox.VoxelRule.Average
homogRule  = merope.HomogenizationRule.Voigt  # Voigt = upper bound

# --- Output settings
results_folder = "polycrystal"   # Top-level results folder
if not os.path.exists(results_folder):
    os.makedirs(results_folder)

vtkname     = "Zone.vtk"      # VTK file of voxelized structure
fileCoeff   = "Coeffs.txt"    # Material coefficients output

# --- Thermal conductivities of phases
Kmatrix = 1.0     # solid
Kgases  = 1e-3    # pores
K       = [Kmatrix, Kmatrix, Kgases]   # one entry per phase

# ---------------------------------------------------------------------------
# STRUCTURE GENERATION & VOXELIZATION
# ---------------------------------------------------------------------------

def cracked_structure(n_voxels, length, seed, layer, voxel_rule, K, vtkname, fileCoeff, INTRA=False, INTER=False, GRAINS=True):

    """
    Build a polycrystal with:
        - Intergranular pores
        - Intragranular pores
        - Grain-boundary layer
    Then voxelize the structure and export as VTK.

    """
    
    print("--- Cracked structure ---")

    print(f"INTRA  = {INTRA}")
    print(f"INTER  = {INTER}")
    print(f"GRAINS = {GRAINS}")

    # --- Step 1: Intergranular spherical multiInc_intergranular (large pores)
    if INTER:
        print(f"\nIntergranular inclusions:")
        print(f"Radius = {R_pores_GB}")
        print(f"Porosity = {P_pores_GB}")
        print(f"Phase = {incl_phase}")

        intergranular = merope.SphereInclusions_3D()
        intergranular.setLength(length)
        intergranular.fromHisto(
            seed, 
            sac_de_billes.TypeAlgo.BOOL, 
            0., 
            [[R_pores_GB, P_pores_GB]], 
            [incl_phase]
        )

        print("→ Number of inclusions generated:", len(intergranular.getSpheres()))

        # pores of incl_phase)
        multiInc_intergranular = merope.MultiInclusions_3D()
        multiInc_intergranular.setInclusions(intergranular)

    # --- Step 2: Intragranular spherical multiInc_intergranular (small pores)
    if INTRA:
        print(f"Intragranular inclusions:")
        print(f"Radius = {R_pores_IG1}")
        print(f"Porosity = {P_pores_IG1}")
        print(f"Phase = {temp_phase}")

        intragranular = merope.SphereInclusions_3D()
        intragranular.setLength(length)
        intragranular.fromHisto(
            seed, 
            sac_de_billes.TypeAlgo.BOOL, 
            0., 
            [[R_pores_IG1, P_pores_IG1]],   # R, P --> phase 1
            [temp_phase]                    # phase 1, phase 1
            # [[R_pores_IG1, P_pores_IG1],  # R, P --> phase 1
            # [R_pores_IG2, P_pores_IG2]],  # R, P --> phase 1
            # [temp_phase, temp_phase]      # phase 1, phase 1
        )

        print("→ Number of inclusions generated:", len(intragranular.getSpheres()))

        # inclusions / pores in phase "temp_phase"
        multiInc_intragranular = merope.MultiInclusions_3D()
        multiInc_intragranular.setInclusions(intragranular)
        # multiInc_intragranular.changePhase(multiInc_intragranular.getAllIdentifiers(), [temp_phase for _ in multiInc_intragranular.getAllIdentifiers()])

    # --- Step 3: Laguerre tessellation for grains
    if GRAINS:
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

    # if GRAINS:
    #     # Add boundary layer of thickness delta to the grains, and phase 0 (grains)
    #     multiInc_grain.addLayer(multiInc_grain.getAllIdentifiers(), delta_phase, delta) # layer in delta phase
    #     multiInc_grain.changePhase(multiInc_grain.getAllIdentifiers(), [grain_phase for _ in multiInc_grain.getAllIdentifiers()])
    #     multiInc_grain.changePhase(multiInc_grain.getAllIdentifiers(),multiInc_grain.getAllIdentifiers()) # grain in fasi diverse (per vederli su paraview)

    # --- Step 4: Merge structures
    if INTER and GRAINS and INTRA:
        print(f"\nIntra, inter and grains")
        # Add boundary layer of thickness delta to the grains, and phase 1 (temp)
        print("Adding grain-boundary layer")
        multiInc_grain.addLayer(multiInc_grain.getAllIdentifiers(), delta_phase, layer) # layer in delta phase
        multiInc_grain.changePhase(multiInc_grain.getAllIdentifiers(), [temp_phase for _ in multiInc_grain.getAllIdentifiers()]) # grani tutti nella stessa fase (la temp_phase in questo caso)

        structure_temp = merope.Structure_3D(
            multiInc_intergranular, # inter-granulari, phase 2 (incl)
            multiInc_grain,  # grain (phase 1) + delta (phase 3)
            {incl_phase: grain_phase, # old phase --> new phase
            delta_phase: grain_phase} # old phase --> new phase
        )

        structure = merope.Structure_3D(
            structure_temp, 
            multiInc_intragranular, 
            {grain_phase: incl_phase}
        )

    elif GRAINS and not INTER and not INTRA:
        print(f"\nOnly grains:")
        structure = merope.Structure_3D(
            multiInc_grain,
        )

    elif INTRA and not GRAINS and not INTER:
        print(f"\nOnly intra:")
        structure = merope.Structure_3D(multiInc_intragranular)


    elif INTER and not GRAINS and not INTRA:
        print(f"\nOnly inter:")
        structure = merope.Structure_3D(multiInc_intergranular)

    # --- Step 5: Voxelization
    voxellation = [n_voxels, n_voxels, n_voxels]
    domain_size = [L_RVE, L_RVE, L_RVE]
    grid_params = merope.vox.create_grid_parameters_N_L_3D(voxellation, domain_size)
    grid = merope.vox.GridRepresentation_3D(structure, grid_params, voxel_rule)

    # --- Step 6: Print resolution parameters
    voxel_size = [length[i] / n_voxels for i in range(3)]

    print("\n==================== SIMULATION SUMMARY ====================\n")

    print("--- Resolution parameters ---")
    print(f"Grid size      : {n_voxels} x {n_voxels} x {n_voxels} → {n_voxels**3:,} voxels")
    print(f"RVE dimensions : {length} (units)")
    print(f"Voxel size     : {voxel_size} (units per axis)")
    print(f"GB layer       : {layer} (thickness)\n")

    # --- Phase fractions ---
    print("--- Phase fractions ---")
    analyzer = merope.vox.GridAnalyzer_3D()
    analyzer.compute_percentages(grid)

    # Phase fraction analysis
    analyzer.print_percentages(grid)

    # Homogenization with conductivity
    grid.apply_homogRule(merope.HomogenizationRule.Voigt, K)

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
    os.makedirs(results_folder, exist_ok=True)
    go_to_dir(results_folder)
    os.makedirs(str(n_voxels), exist_ok=True)
    go_to_dir(str(n_voxels))

    # --- Build & voxelize structure
    cracked_structure(
        n_voxels, [L_RVE, L_RVE, L_RVE], seed, 
        grain_boundary_layer, voxel_rule, K, vtkname, fileCoeff,
        INTRA, INTER, GRAINS
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
    num_voxels = n_voxels**3
    L_voxel = L_RVE / n_voxels

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
