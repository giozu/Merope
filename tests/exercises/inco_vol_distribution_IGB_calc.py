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
import math as ma
from math import sqrt
import matplotlib.pyplot as plt
import numpy as np

import archi_merope as arch
import interface_amitex_fftp.amitex_wrapper as amitex
import interface_amitex_fftp.post_processing as amitex_out



# Voxel & Cell INPUTS #

L = [10, 10, 10]
total_volume = ma.prod(L)
n3D = 150
seed = 0
nameShape = sac_de_billes.NameShape.Tore
voxel_rule = merope.vox.VoxelRule.Average
homogRule = merope.HomogenizationRule.Voigt  ## If i want to use homogRule, voxel_rule must be = merope.vox.VoxelRule.Average


# Names of folders that will contain results #

folder_name = 'Result' #nome della cartella che conterrà i risultati
folder_path = os.path.join("/home/giovanni/nuclear/merope/tests/mattiuz", folder_name)

    
def go_to_dir(name_dir):
    time.sleep(0.1)
    os.chdir(name_dir)
    print(name_dir)
    time.sleep(0.1)
    
vtkname = "Zone.vtk"
fileCoeff = "Coeffs.txt"

tic0 = time.time()

# fit parameters to be calculated with alpha_beta_gamma file.py #

alpha = -0.5484
beta = 1.9214
gamma = 0.3777

# MATERIAL INPUTS #

##############
#porosity = 0.3  # total porosity


##########################
##########################
inclPhi = 0.2
delta = 1 # thickness of layer
NbSpheres = 200
#########################
#########################


### EQUAL FILES PARAMETERS ###

inclR = 0.25
#inclPhi = 0.47
lagR = 1
tabRadii = [lagR for i in range(NbSpheres)]
lagPhi = 1

####

inclRphi = [inclR, inclPhi]
lagRphi = [lagR, lagPhi]

# Phases #
incl_phase = 2 # 0
delta_phase = 3 # 1
grains_phase = 0 # 2

# Thermal conductivity of the phases #
Kmatrix = 1  #thermal conductivity of the matrix
Kgases = 1e-03  #thermal conductivity of gases in pore

K = [Kmatrix, Kgases]
    
### Function that set the grains volume distribution

mean = 5
std_dev = 1.2 # if = 0.5 more difference between volumes, if = 3 more equal volumes ~1% ## CHANGING STD_DEV DOESNT CHANGE POROSITY

def weighting_function(center, mean, std_dev):
    distance = sqrt(sum((coord - 0.5 * dim)**2 for coord, dim in zip(center, L)))
    return np.exp(-0.5 * (distance - mean)**2 / (std_dev**2))

def cracked_structure(n3D, L, total_volume, seed, NbSpheres, inclRphi, nameShape, tabRadii, incl_phase, grains_phase, delta_phase, delta, voxel_rule, K, vtkname, fileCoeff):
    
    ### Add the spherical inclusions 
    sphIncl2 = merope.SphereInclusions_3D()
    sphIncl2.setLength(L)
    sphIncl2.fromHisto(seed, sac_de_billes.TypeAlgo.BOOL, 0., [inclRphi], [incl_phase])

    multiInclusions2 = merope.MultiInclusions_3D()
    multiInclusions2.setInclusions(sphIncl2)
       
    ### Laguerre
    ### Optimize the volumes with all equal volumes
    tabPhases = [1 for i in range(NbSpheres)]
    mindist = 0
    
    theSpheres = sac_de_billes.throwSpheres_3D(sac_de_billes.TypeAlgo.RSA, nameShape, L, seed, tabRadii, tabPhases, mindist) 
    desiredVolumes = [weighting_function(sphere.center, mean, std_dev) for sphere in theSpheres]
    total_weight = 0
    for weight in desiredVolumes:
        total_weight += weight
    desiredVolumes = [weight / total_weight * total_volume for weight in desiredVolumes]
    
    algo = merope.algo_fit_volumes_3D(L, theSpheres, desiredVolumes)
    max_delta_Volume = 1e-6 * total_volume
    max_iter = 3000
    verbose = True  
    algo.proceed(max_delta_Volume, max_iter, verbose)
    the_new_spheres = algo.getCenterTessels()
    
    polyCrystal = merope.LaguerreTess_3D(L, the_new_spheres)
    volume_distr = [100* i /total_volume for i in desiredVolumes]
    print("Vgrains/Vtot = ", [f"{i:.2f}%" for i in volume_distr])
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
    
    return volume_distr
### function to calculate the thermal conductivity

def ThermalAmitex():
    number_of_processors = 6                   # for parallel computing
    voxellation_of_zones = vtkname
    amitex.computeThermalCoeff(voxellation_of_zones, number_of_processors)
    homogenized_matrix = amitex_out.printThermalCoeff(".")
    
os.mkdir(folder_name)
go_to_dir(folder_name)
os.mkdir(str(n3D))
go_to_dir(str(n3D))
volume_distr = cracked_structure(n3D, L, total_volume, seed, NbSpheres, inclRphi, nameShape, tabRadii, incl_phase, grains_phase, delta_phase, delta, voxel_rule, K, vtkname, fileCoeff)

numeri = volume_distr
plt.hist(numeri, bins= 20, edgecolor='black', alpha=0.7)
# Aggiunta di etichette
plt.title("Volume Distribution")
plt.xlabel("% Volume")
plt.ylabel("Frequency")
# Save the distribution istogram
plt.savefig("volume_distribution.png", format="png", dpi=300)

#ThermalAmitex()
go_to_dir("../")

'''
porosity = 0.3
delta = 0.08
[0.48470641, -0.00027252675, -0.00061701866]
[-0.00027254867, 0.4851352, 0.00050689383]
[-0.00061708994, 0.00050689858, 0.48461078]

p = 0.27
inclPhi = 0.45
delta = 0.3 # thickness of layer
NbSpheres = 200
mean = 5
std_dev = 3
[0.58056617, -0.00049252528, -0.002885129]
[-0.0004925225, 0.58331701, 0.0015727888]
[-0.002885125, 0.001572786, 0.58302629]

p = 0.24
inclPhi = 0.45
delta = 0.3 # thickness of layer
NbSpheres = 200
mean = 5
std_dev = 0.5

'''


