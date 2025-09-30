###############################################################

### FILE USED TO CLAC K VS POROSITY,DELTA,GRAINS VOL ###

###############################################################

## N:B PHASE 1 IS THE POROUS PHASE ##

# BISOGNA OTTIMIZZARE VOLUME GRANI #

import os
import sac_de_billes
import merope
import time
import shutil
from math import sqrt

import archi_merope as arch
import interface_amitex_fftp.amitex_wrapper as amitex
import interface_amitex_fftp.post_processing as amitex_out



# Voxel & Cell INPUTS #

L = [10, 10, 10]
n3D = 300
seed = 0
voxel_rule = merope.vox.VoxelRule.Average
homogRule = merope.HomogenizationRule.Voigt  ## If i want to use homogRule, voxel_rule must be = merope.vox.VoxelRule.Average

# Names of folders that will contain results #

folder_name = 'Result' #nome della cartella che conterrà i risultati
    
def go_to_dir(name_dir):
    time.sleep(0.1)
    os.chdir(name_dir)
    print(name_dir)
    time.sleep(0.1)
    
vtkname = "crack_structure.vtk"
fileCoeff = "Coeffs.txt"

tic0 = time.time()

### EQUAL FILES PARAMETERS FOR THE DETERMINATION OF ALPHA BETA GAMMA PARAMETERS ###

###########################################
inclR = 0.1 # POSSIBLE INPUT to determine K
###########################################

###########################################
inclPhi = 0.2 # Used to determine porosity
###########################################

lagR = 4 # Parameter to determine the grains volumes
lagPhi = 1 # Must be 1 to fill the entire RVE with solid matrix before putting porosities

# fit parameters to be calculated with alpha_beta_gamma file.py #

alpha = -0.5484
beta = 1.9214
gamma = 0.3777
porosity = 0.5  # total porosity

# MATERIAL INPUTS #

##############
# 0.135
delta = 0.001 # thickness of layer POSSIBLE INPUT
delta_ratio = delta/lagR
##############

#inclPhi = (porosity - beta*delta - gamma)/alpha

inclRphi = [inclR, inclPhi]
lagRphi = [lagR, lagPhi]



incl_phase = 2 # 0
delta_phase = 3 # 1
grains_phase = 0 # 2


Kmatrix = 1  #thermal conductivity of the matrix
Kgases = 1e-03  #thermal conductivity of gases in pore

K = [Kmatrix, Kmatrix, Kgases]

def Crack_structure_Voxellation(n3D, L, seed, inclRphi, lagRphi, incl_phase, grains_phase, delta_phase, delta, voxel_rule, K, vtkname, fileCoeff):
    
    ### Add the spherical inclusions 
    sphIncl2 = merope.SphereInclusions_3D()
    sphIncl2.setLength(L)
    sphIncl2.fromHisto(seed, sac_de_billes.TypeAlgo.BOOL, 0., [inclRphi], [incl_phase])

    multiInclusions2 = merope.MultiInclusions_3D()
    multiInclusions2.setInclusions(sphIncl2)
       
    ### Laguerre
    sphIncl = merope.SphereInclusions_3D()
    sphIncl.setLength(L)
    sphIncl.fromHisto(seed, sac_de_billes.TypeAlgo.RSA, 0., [lagRphi], [1])

    polyCrystal = merope.LaguerreTess_3D(L, sphIncl.getSpheres())
    multiInclusions = merope.MultiInclusions_3D()
    multiInclusions.setInclusions(polyCrystal)
    N = len(multiInclusions.getAllIdentifiers())
    multiInclusions.addLayer(multiInclusions.getAllIdentifiers(), delta_phase, delta) # addLayer(identifiers, newPhase, width)
    multiInclusions.changePhase(multiInclusions.getAllIdentifiers(), [1 for i in multiInclusions.getAllIdentifiers()])
    
    ### Final structureGet the polyCrystal
    structure1 = merope.Structure_3D(multiInclusions)
    structure2 = merope.Structure_3D(multiInclusions2)
    phase = [N]
    dictionnaire = { incl_phase:grains_phase , delta_phase: grains_phase} # {2:3} dice che le sfere dentro i bordi diventano pori
    #dictionnaire = {delta_phase: delta_phase + 1, incl_phase: grains_phase}
    
    structure = merope.Structure_3D(multiInclusions2, multiInclusions, dictionnaire)
    ## PHASE 1 OF THE SECOND STRUCTURE ACTIVATES THE MASK ##
    
    ### Voxellation
    gridParameters = merope.vox.create_grid_parameters_N_L_3D([n3D,n3D,n3D], L)
    grid = merope.vox.GridRepresentation_3D(structure, gridParameters, voxel_rule)
    analyzer = merope.vox.GridAnalyzer_3D()
    analyzer.compute_percentages(grid)
    analyzer.print_percentages(grid)
    grid.apply_homogRule(homogRule, K)
    #allPhases = structure.getAllPhases()
    #grid.apply_coefficients(K)
    #my_printer.printVTK(grid, vtkname, nameValue = "Materialid")
    my_printer = merope.vox.vtk_printer_3D()
    my_printer.printVTK_segmented(grid, vtkname, fileCoeff, nameValue = "MaterialId")
    

### function to calculate the thermal conductivity

def ThermalAmitex():
    number_of_processors = 2                   # for parallel computing
    voxellation_of_zones = vtkname
    amitex.computeThermalCoeff(voxellation_of_zones, number_of_processors)
    homogenized_matrix = amitex_out.printThermalCoeff(".")
    
os.mkdir(folder_name)
go_to_dir(folder_name)
os.mkdir(str(n3D))
go_to_dir(str(n3D))
Crack_structure_Voxellation(n3D, L, seed, inclRphi, lagRphi, incl_phase, grains_phase, delta_phase, delta, voxel_rule, K, vtkname, fileCoeff)
#ThermalAmitex()
go_to_dir("../")

'''
porosity = 0.3
delta = 0.08
[0.48470641, -0.00027252675, -0.00061701866]
[-0.00027254867, 0.4851352, 0.00050689383]
[-0.00061708994, 0.00050689858, 0.48461078]
'''


