###############################################################

### FILE USED TO CLAC K VS porDelta,DELTA,GRAINS VOL ###

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
from send2trash import send2trash



# Voxel & Cell INPUTS #

L = [10, 10, 10]
RVEsize = L[1]
n3D = 100
seed = 0
NbSeed = 1 #Number of calculations with the same physical quantities but duffrent randomness alghoritm (seed)
voxel_rule = merope.vox.VoxelRule.Average
homogRule = merope.HomogenizationRule.Voigt  ## If i want to use homogRule, voxel_rule must be = merope.vox.VoxelRule.Average


# Names of folders that will contain results #

folder_name = 'Result_mixed_intra_inter' #nome della cartella che conterrà i risultati
folder_path = os.path.join("/home/giovanni/Merope/tests/test_por/Ale_py_files", folder_name)
file_output_path = "porDelta_conduct_results.txt"

os.makedirs(folder_path, exist_ok=True)
if os.path.exists(folder_path):
    send2trash(folder_path)
os.makedirs(folder_path, exist_ok=True)
    
vtkname = "crack_structure.vtk"
fileCoeff = "Coeffs.txt"

### EQUAL FILES PARAMETERS FOR THE DETERMINATION OF ALPHA BETA GAMMA PARAMETERS ###

inclR = 0.3 # POSSIBLE INPUT to determine K

incl2R = 0.05
#incl2Phi = 0.05 # porDelta to add to the one due to cracks

lagR = 3 # Parameter to determine the grains volumes
lagPhi = 1 # Must be 1 to fill the entire RVE with solid matrix before putting porosities


# MATERIAL INPUTS #

##############
##############

inclPhi = np.linspace(0.02, 0.20, 10)
incl2Phi = np.linspace(0.01, 0.031, 10) # porDelta to add to the one due to cracks
delta = 3 # thickness of layer POSSIBLE INPUT
delta_ratio = delta/lagR

##############
##############

#inclPhi = (porDelta - beta*delta - gamma)/alpha

inclRphi = [inclR, inclPhi]
lagRphi = [lagR, lagPhi]



incl_phase = 2 # 0
delta_phase = 3 # 1
grains_phase = 0 # 2


Kmatrix = 1  #thermal conductivity of the matrix
Kgases = 1e-03  #thermal conductivity of gases in pore

K = [Kmatrix, Kmatrix, Kgases]

def Crack_structure_Voxellation(n3D, L, seed, inclR, inclPhi, lagRphi, incl2R , incl2Phi, incl_phase, grains_phase, delta_phase, delta, voxel_rule, K, vtkname, fileCoeff):
    
    ### Add the spherical inclusions 
    sphIncl2 = merope.SphereInclusions_3D()
    sphIncl2.setLength(L)
    sphIncl2.fromHisto(seed, sac_de_billes.TypeAlgo.BOOL, 0., [[inclR, inclPhi]], [incl_phase])

    multiInclusions2 = merope.MultiInclusions_3D()
    multiInclusions2.setInclusions(sphIncl2)
    
    ### Add the second spherical inclusions 
    sphIncl4 = merope.SphereInclusions_3D()
    sphIncl4.setLength(L)
    sphIncl4.fromHisto(seed, sac_de_billes.TypeAlgo.RSA, 0., [[incl2R , incl2Phi]], [1])

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
    multiInclusions.addLayer(multiInclusions.getAllIdentifiers(), delta_phase, delta) # addLayer(identifiers, newPhase, width)
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
    phases_fracts = analyzer.compute_percentages(grid)
    analyzer.print_percentages(grid)
    grid.apply_homogRule(homogRule, K)
    #allPhases = structure.getAllPhases()
    #grid.apply_coefficients(K)
    #my_printer.printVTK(grid, vtkname, nameValue = "Materialid")
    my_printer = merope.vox.vtk_printer_3D()
    my_printer.printVTK_segmented(grid, vtkname, fileCoeff, nameValue = "MaterialId")
    
    # I extrapolate the porDelta
    porDelta = phases_fracts[2]
    porIntra = incl2Phi
    return (porDelta, porIntra)

### function to calculate the thermal conductivity

def ThermalAmitex():
    number_of_processors = 6                   # for parallel computing
    voxellation_of_zones = vtkname
    amitex.computeThermalCoeff(voxellation_of_zones, number_of_processors)
    homogenized_matrix = amitex_out.printThermalCoeff(".")
    
    
    
    
    # Funzione per estrarre e scrivere i valori di conduttività nel file di output
def scrivi_valori_su_file(file_output, valori, porDelta, porIntra, seed):
    media = sum(valori) / len(valori)
    num_decimali = len(str(valori[0]).split('.')[1]) if '.' in str(valori[0]) else 0
    media_formattata = f"{media:.{num_decimali}f}"
    with open(file_output, 'a') as f:
        linea = f"{porDelta:.4f}\t{porIntra}\tSeed_{seed}\t{valori[0]}\t{valori[1]}\t{valori[2]}\t{media_formattata}\n"
        f.write(linea)

# Funzione per leggere la matrice dal file thermalCoeff_amitex.txt
def leggi_matrice_da_file(file_path):
    with open(file_path, 'r') as f:
        matrice = [list(map(float, line.split())) for line in f.readlines()]
    return matrice

# Funzione per estrarre i valori principali (diagonale principale) dalla matrice
def estrai_valori_principali(matrice):
    return [matrice[i][i] for i in range(3)]

# Funzione per aggiornare il file aggregato con i risultati di ogni singolo file di conduttività
def aggiorna_file_aggregato(file_aggregato, porDelta, porIntra, seed, valori, inclR, lagR, RVEsize, n3D, Kmatrix, Kgases):
    media = sum(valori) / len(valori)
    num_decimali = len(str(valori[0]).split('.')[1]) if '.' in str(valori[0]) else 0
    media_formattata = float(f"{media:.{num_decimali}f}")  # Converti media_formattata in float
    
    # Se il file non esiste, crea l'intestazione e scrivi i parametri di input
    if not os.path.exists(file_aggregato):
        with open(file_aggregato, 'w') as f:
            #f.write("Por2: {incl2Phi}\n")
            f.write("Input Parameters:\n")
            f.write(f"Delta: {delta}\n")
            f.write(f"size RVE: {RVEsize}\n")
            f.write(f"INclusions radius: {inclR}\n")
            f.write(f"Mean grains size: {lagR}\n")
            f.write(f"NVoxel: {n3D}\n")
            f.write(f"Kmatrix: {Kmatrix}\n")
            f.write(f"Kgases: {Kgases}\n")
            f.write("PorDelta\tPorIntra\tPorosity\tSeed_index\tK_xx\tK_yy\tK_zz\tK_mean\n")  # Intestazione delle colonne

    # Aggiungi i risultati
    with open(file_aggregato, 'a') as f:
        linea = f"{porDelta:.4f}\t\t{porIntra:.4f}\t\t{porDelta+porIntra:.4f}\t\t{seed}\t\t{valori[0]:.4f}\t\t{valori[1]:.4f}\t\t{valori[2]:.4f}\t\t{media_formattata:.4f}\n"
        f.write(linea)

# Funzione principale per generare strutture e salvare i risultati
def main():
    os.makedirs(folder_path, exist_ok=True)
    file_aggregato = os.path.join(folder_path, "aggregated_results.txt")

    if os.path.exists(file_output_path):
        open(file_output_path, 'w').close()

    # Crea la cartella principale 'Result' solo una volta
    if os.path.exists(folder_name):
        shutil.rmtree(folder_name)
    os.mkdir(folder_name)  # Creiamo la cartella 'Result' una sola volta
    os.chdir(folder_name)  # Entra nella cartella 'Result'

    # Inizializza il valore della porosità
    for j in range(len(incl2Phi)):
        for i in range(len(inclPhi)):
            for seed in range(NbSeed):
                seed_folder = f'Seed_{seed}'  # Nome della cartella per ogni seed
                if not os.path.exists(seed_folder):
                    os.mkdir(seed_folder)  # Crea la cartella per il seed
                os.chdir(seed_folder)  # Entra nella cartella per il seed

                # Calcola la porosità
                porDelta, porIntra = Crack_structure_Voxellation(
                    n3D, L, seed, inclR, inclPhi[i], lagRphi, incl2R , incl2Phi[j], incl_phase, grains_phase,
                    delta_phase, delta, voxel_rule, K, vtkname, fileCoeff
                )

                ThermalAmitex()  # Calcola la conduttività termica

                # Estrai e salva i dati di conduttività
                matrice = leggi_matrice_da_file("thermalCoeff_amitex.txt")
                valori_estratti = estrai_valori_principali(matrice)
                scrivi_valori_su_file(file_output_path, valori_estratti, porDelta, porIntra, seed)

                # Aggiungi i risultati al file aggregato
                aggiorna_file_aggregato(file_aggregato, porDelta, porIntra, seed, valori_estratti, inclR, lagR, RVEsize, n3D, Kmatrix, Kgases)

                os.chdir("../")  # Torna alla cartella del seed

    os.chdir("../")  # Torna alla cartella principale 'Result'

if __name__ == "__main__":
    main()


