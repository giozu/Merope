### this script works just for porosity with one type of spheres

import os
import sac_de_billes
import merope
import time
import shutil
from math import sqrt

import archi_merope as arch
import interface_amitex_fftp.amitex_wrapper as amitex
import interface_amitex_fftp.post_processing as amitex_out
from send2trash import send2trash

folder_name = 'ResultV' #nome della cartella che conterrà i risultati
folder_path = os.path.join("/home/alessio/Merope/Esercitazione", folder_name)

# Verifica se la cartella dei risultati esiste già in caso la sposta nel cestino
if os.path.exists(folder_path):
    send2trash(folder_path)

### Define input setting


Porosity = 0.175
R = 1

Kmatrix = 1    #thermal conductivity of the matrix
Kgases = 1e-09   #thermal conductivity of gases in pore

LVoxel = [2,3,5,10,20,30,50,100,200]
singleVoxel = 100

NameFolder = '/home/alessio/Merope/OneDrive_merope/'  #the parent folder name


### Calcule some values

C = 10         # the size of the side of the cube eauivqlent of 1mm
SizeRVE = [C,C,C]

K = [Kmatrix , Kgases]

Seed = 0


### function to create the voxellation in the case of spherical inclusions

def VoxellationInclusionV(SizeRVE, Seed, R, Porosity, K, Voxellation):
    tic0 = time.time()

    sphIncl = merope.SphereInclusions_3D()
    sphIncl.setLength(SizeRVE)
    sphIncl.fromHisto(Seed, sac_de_billes.TypeAlgo.RSA, 0., [[R, Porosity]], [1])  #genero la microstruttura con inclusioni sferiche

    #sphIncl.printDump("inclusions.dump")
    #sphIncl.printPos("inclusions.pos")
    #sphIncl.printCSV("inclusions.csv")
    #sphIncl.printFracVol("volume_fractions.txt")

    multiInclusions = merope.MultiInclusions_3D()
    multiInclusions.setInclusions(sphIncl)

    grid = merope.Voxellation_3D(multiInclusions)  #come argomento della funzione posso mettere multiInclusion_3D oppure structure_3D se sto considerando un insieme di multiInclusions
    grid.setPureCoeffs(K)
    grid.setHomogRule(merope.HomogenizationRule.Voigt) # Reuss, Voigt, Smallest, Largest
    grid.setVoxelRule(merope.VoxelRule.Average)  #average o center
    grid.proceed(Voxellation) # voxellation = [larghezza, NMin e NMax] larghezza = [N1,N2,N3] Nmin e Nmax usati insieme selezionano solo un certo intervallo da considerare
    grid.printFile("Zone.vtk","Coeffs.txt") #file.vtk per fft e paraview, file.txt per fft

    print(time.time() - tic0)


### function to go in and go out to a folder

def go_to_dir(name_dir):
    time.sleep(0.1)
    os.chdir(name_dir)
    print(name_dir)
    time.sleep(0.1)


### function to calculate the thermal conductivity

def ThermalAmitex():
    number_of_processors = 2                   # for parallel computing
    voxellation_of_zones = "Zone.vtk"
    amitex.computeThermalCoeff(voxellation_of_zones, number_of_processors)
    homogenized_matrix = amitex_out.printThermalCoeff(".")

### funzone che crea file di testo con i dati

def CollectData(name, singleVoxel):
    # Crea o sovrascrive il file con il nome fornito
    fichier = open(name + '.txt', "w")
    fichier.close()

    # Sposta nella directory del singolo caso
    go_to_dir(str(singleVoxel))
   
    # Apre il file dei coefficienti termici
    file = open("thermalCoeff_amitex.txt", "r", encoding="utf8")
    coeff = open("thermalCoeff_amitex.txt", "r")
    
    # Legge il file riga per riga
    L1 = []
    for ligne in coeff:
        L1.append(ligne.strip())
    
    file.close()
    coeff.close()
    
    # Torna indietro alla directory precedente
    go_to_dir("../")
    
    # Estrae i numeri e calcola Value ed Error
    L2 = []
    for i in range(len(L1)):
        liste_de_chiffres = [float(chiffre) for chiffre in L1[i].split(' ')]
        L2.append(liste_de_chiffres)
    
    # Calcola Value ed Error
    Value = sqrt(L2[0][0]**2 + L2[1][1]**2 + L2[2][2]**2)

    Error = sqrt(L2[0][1]**2 + L2[1][0]**2 + L2[0][2]**2 + L2[2][0]**2 + L2[1][2]**2 + L2[2][1]**2)

    
    # Scrive i risultati in un file
    texte = open(name + '.txt', "w")  #"w" dice di sovrascrivere se il file esiste gia
    texte.write(f"Nb of Voxel {singleVoxel}:   {Value}  ,  {Error}\n") # \n dice di andare a capo
    texte.close()




### Generation of folders with vtk files and txt files (of thermal conductivity)

os.mkdir(folder_name)
go_to_dir(folder_name)

os.mkdir(str(singleVoxel))
go_to_dir(str(singleVoxel))
Voxellation = [singleVoxel, singleVoxel, singleVoxel]
VoxellationInclusionV(SizeRVE, Seed, R, Porosity, K, Voxellation)
ThermalAmitex()
go_to_dir("../")

# Collezione dei dati
name = 'V'
CollectData(name, singleVoxel)  





































