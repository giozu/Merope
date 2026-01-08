# ======================================================================================
# PROJECT: Master's Thesis in Nuclear Engineering
# AUTHOR : [Alessio Mattiuz]
# CREATED: 2024/25
# FILE PURPOSE: 
# Functions for generating distributed or interconnected 3D porous structures.
# Includes:
#   - Intra- and inter-granular inclusions,
#   - Log-normal radius distributions,
#   - Control of topology via the delta/grain ratio,
#   - Mérope-based voxelization and export for homogenization.
# ======================================================================================
# THE PARAMETER WHICH DETERMINATES IF THE STRUCTURE IS DISTRIBUTED OR INTERCONNECTED IS DELTA/GRAIN
# IF DELTA/GRAIN ~ 1 -> DISTIBUTED
# IF DELTA/GRAIN < 1 -> INTERCONNECTED 


import numpy as np
import sac_de_billes
import merope
import archi_merope as arch


# Output file output for Mérope
vtkname = "double_layer_structure.vtk"
fileCoeff = "Coeffs.txt"

def generate_spheres(target_porosity, mean_radius, std_radius, num_radius):
    # Genera i raggi con distribuzione lognormale
    radii = np.abs(np.random.lognormal(mean_radius, std_radius, num_radius))
    radii = radii[(radii >= 0.005) & (radii <= 2)]
    
    # Genera frazioni: in questo caso uguali per ciascuna inclusione, normalizzate sul target
    fractions = [target_porosity / num_radius for _ in range(len(radii))]
    fractions = (np.array(fractions) / np.sum(fractions)) * target_porosity
    
    return [[r, f] for r, f in zip(radii, fractions)]

def Crack_structure_Voxellation(fixed_params, var_params):

    # Extracting all params 
    
    # RVE
    L = fixed_params["L"]
    n3D = fixed_params["n3D"] 
    seed = fixed_params["seed"]
    voxel_rule = fixed_params["voxel_rule"]
    homogRule = fixed_params["homogRule"] 
    grid_size = fixed_params["grid_size"] 
    
    # 3D structure params
    inclRphi = fixed_params["inclRphi"]
    intraInclRphi = fixed_params["intraInclRphi"]
    target_porosity = fixed_params["target_porosity"]
    num_radius = fixed_params["num_radius"]
    incl_phase = fixed_params["incl_phase"]
    delta_phase = fixed_params["delta_phase"]
    grains_phase = fixed_params["grains_phase"]
    lagR = fixed_params["lagR"]
    lagPhi = fixed_params["lagPhi"]
    Kmatrix = fixed_params["Kmatrix"]
    Kgases = fixed_params["Kgases"]
    K = [Kmatrix, Kgases]
   
    delta = [fixed_params["delta0"], fixed_params["delta1"]]
    
    # Parameters for Inter porosity (log-normal distribution):
    mean_radius = var_params["mean_radius"]
    std_radius = var_params["std_radius"]
    
    ### 3D structure building

    # --- Inter-granular inclusions for small Grain Boundary that represent the nano fracture situated between every grain
    sphIncl2 = merope.SphereInclusions_3D()
    sphIncl2.setLength(L)
    sphIncl2.fromHisto(seed, sac_de_billes.TypeAlgo.BOOL, 0., [inclRphi], [2])
    multiInclusions2 = merope.MultiInclusions_3D()
    multiInclusions2.setInclusions(sphIncl2)
    
    # --- Intra-granular inclusions
    sphIncl4 = merope.SphereInclusions_3D()
    sphIncl4.setLength(L)
    sphIncl4.fromHisto(seed, sac_de_billes.TypeAlgo.BOOL, 0., [intraInclRphi], [1])
    multiInclusions4 = merope.MultiInclusions_3D()
    multiInclusions4.setInclusions(sphIncl4)
    structure4 = merope.Structure_3D(multiInclusions4)
    
    # --- Inter-granular inclusions with log-normal distribution that represent the main porosity contribution, based on delta parameter this inclusions could be near grain boundaries or distributed in all the solid matrix
    sphIncl5 = merope.SphereInclusions_3D()
    sphIncl5.setLength(L)
    spheres_intra = generate_spheres(target_porosity, mean_radius, std_radius, num_radius)
    sphIncl5.fromHisto(seed, sac_de_billes.TypeAlgo.BOOL, 0., spheres_intra, [1 for _ in range(num_radius)])
    multiInclusions5 = merope.MultiInclusions_3D()
    multiInclusions5.setInclusions(sphIncl5)
    structure5 = merope.Structure_3D(multiInclusions5)
    
    # --- Laguerre 1
    sphIncl = merope.SphereInclusions_3D()
    sphIncl.setLength(L)
    sphIncl.fromHisto(seed, sac_de_billes.TypeAlgo.RSA, 0., [[lagR, lagPhi]], [1])
    polyCrystal = merope.LaguerreTess_3D(L, sphIncl.getSpheres())
    multiInclusions = merope.MultiInclusions_3D()
    multiInclusions.setInclusions(polyCrystal)
    multiInclusions.addLayer(multiInclusions.getAllIdentifiers(), 3, delta[0])
    multiInclusions.changePhase(multiInclusions.getAllIdentifiers(), [1 for _ in multiInclusions.getAllIdentifiers()])
    
    # --- Laguerre 2
    polyCrystal = merope.LaguerreTess_3D(L, sphIncl.getSpheres())
    multiInclusionsLag = merope.MultiInclusions_3D()
    multiInclusionsLag.setInclusions(polyCrystal)
    multiInclusionsLag.addLayer(multiInclusions.getAllIdentifiers(), 4, delta[1])
    multiInclusionsLag.changePhase(multiInclusions.getAllIdentifiers(), [1 for _ in multiInclusions.getAllIdentifiers()])
    
    # --- Structures combination
    structure1 = merope.Structure_3D(multiInclusions)         # Laguerre
    structure2 = merope.Structure_3D(multiInclusions2)          # Inclusions BOOL 1
    dictionnaire1 = {2: 0, 3: 0}
    structure3 = merope.Structure_3D(structure2, structure1, dictionnaire1)
    
    Lag2_str = merope.Structure_3D(multiInclusionsLag)    
    dictionnaire2 = {1: 0, 4: 1}
    structure6 = merope.Structure_3D(structure5, Lag2_str, dictionnaire2)
    
    dictionnaire = {0: 2}
    structure_a = merope.Structure_3D(structure3, structure6, dictionnaire)
    
    
    structure_final = merope.Structure_3D(structure_a, structure4, dictionnaire)
    
    # --- Voxellization
    gridParameters = merope.vox.create_grid_parameters_N_L_3D([n3D, n3D, n3D], L)
    grid = merope.vox.GridRepresentation_3D(structure_final, gridParameters, voxel_rule)
    analyzer = merope.vox.GridAnalyzer_3D()
    por = analyzer.compute_percentages(grid)
    analyzer.print_percentages(grid)
    porosity = por[2]
    grid.apply_homogRule(homogRule, K)
    
    converter = merope.vox.NumpyConverter_3D()
    array3d = converter.compute_RealField(grid)
    array3d = array3d.reshape((n3D, n3D, n3D), order='C')
    
    my_printer = merope.vox.vtk_printer_3D()
    my_printer.printVTK_segmented(grid, vtkname, fileCoeff, nameValue="MaterialId")
    
    return array3d, porosity


