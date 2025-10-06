###############################################################

### FILE USED TO CLAC K VS POROSITY,DELTA,GRAINS VOL ###

###############################################################

## N:B PHASE 1 IS THE POROUS PHASE ##

# BISOGNA OTTIMIZZARE VOLUME GRANI #

import os
import sac_de_billes
import merope
import shutil
import numpy as np

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
folder_path = os.path.join("/home/giovanni/nuclear/merope/tests/mattiuz", folder_name)

    
def go_to_dir(name_dir):
    os.chdir(name_dir)
    print(name_dir)
    
    
vtkname = "Zone.vtk"
fileCoeff = "Coeffs.txt"


###########################################
inclR = 0.03 # POSSIBLE INPUT to determine K
###########################################

###########################################
inclPhi = 0.4 # Used to determine porosity
###########################################

# INTRAGRANULAR INCLUSIONS #
target_porosity = 0.23
mean_radius, std_radius = 0.3, 0.13
num_radius = 5

lagR = 1 # Parameter to determine the grains volumes
lagPhi = 1 # Must be 1 to fill the entire RVE with solid matrix before putting porosities

# MATERIAL INPUTS #

##############
delta = 0.003 # thickness of layer POSSIBLE INPUT
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


def generate_spheres(target_porosity, mean_radius, std_radius, num_radius):
    # Genera i raggi con distribuzione gaussiana
    radii = np.abs(np.random.normal(mean_radius, std_radius, num_radius))  
      
    # Genera frazioni casuali e normalizza per rispettare il target
    fractions = np.random.rand(num_radius)
    fractions = (fractions / np.sum(fractions)) * target_porosity

    # Formatta nel formato richiesto per la funzione
    return [[r, f] for r, f in zip(radii, fractions)]


def cracked_structure(n3D, L, seed, inclRphi, lagRphi, incl_phase, target_porosity, mean_radius, std_radius, num_radius, grains_phase, delta_phase, delta, voxel_rule, K, vtkname, fileCoeff):
    
    ### Add the spherical inclusions INTER-POROSITY
    sphIncl2 = merope.SphereInclusions_3D()
    sphIncl2.setLength(L)
    sphIncl2.fromHisto(seed, sac_de_billes.TypeAlgo.BOOL, 0., [inclRphi], [2])

    multiInclusions2 = merope.MultiInclusions_3D()
    multiInclusions2.setInclusions(sphIncl2)
    
    ### Add the second spherical inclusions INTRA-POROSITY
    sphIncl4 = merope.SphereInclusions_3D()
    sphIncl4.setLength(L)
    sphIncl4.fromHisto(seed, sac_de_billes.TypeAlgo.BOOL, 0., generate_spheres(target_porosity, mean_radius, std_radius, num_radius), [1 for i in range(num_radius)])

    multiInclusions4 = merope.MultiInclusions_3D()
    multiInclusions4.setInclusions(sphIncl4)
    structure4 = merope.Structure_3D(multiInclusions4)
       
    ### Laguerre
    sphIncl = merope.SphereInclusions_3D()
    sphIncl.setLength(L)
    sphIncl.fromHisto(seed, sac_de_billes.TypeAlgo.RSA, 0., [lagRphi], [1])

    polyCrystal = merope.LaguerreTess_3D(L, sphIncl.getSpheres())
    multiInclusions = merope.MultiInclusions_3D()
    multiInclusions.setInclusions(polyCrystal)
    N = len(multiInclusions.getAllIdentifiers())
    multiInclusions.addLayer(multiInclusions.getAllIdentifiers(), 3, delta) # addLayer(identifiers, newPhase, width)
    multiInclusions.changePhase(multiInclusions.getAllIdentifiers(), [1 for i in multiInclusions.getAllIdentifiers()])
    
    ### Final structureGet the polyCrystal
    structure1 = merope.Structure_3D(multiInclusions)
    structure2 = merope.Structure_3D(multiInclusions2)
    phase = [N]
    dictionnaire1 = { 2:0 , 3: 0} # {2:3} dice che le sfere dentro i bordi diventano pori
    #dictionnaire = {delta_phase: delta_phase + 1, incl_phase: grains_phase}
    
    structure3 = merope.Structure_3D(multiInclusions2, multiInclusions, dictionnaire1)
    dictionnaire = {0: 2}
    structure = merope.Structure_3D(structure3, structure4, dictionnaire)
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
cracked_structure(n3D, L, seed, inclRphi, lagRphi, incl_phase, target_porosity, mean_radius, std_radius, num_radius, grains_phase, delta_phase, delta, voxel_rule, K, vtkname, fileCoeff)
#ThermalAmitex()
go_to_dir("../")




