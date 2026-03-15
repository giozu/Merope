### CALCOLO K PER DIVERSE POROSITA' E CONFRONTO CON I MODELLI ###


import os
import sac_de_billes
import merope
import shutil
from math import sqrt

import archi_merope as arch
import interface_amitex_fftp.amitex_wrapper as amitex
import interface_amitex_fftp.post_processing as amitex_out
from send2trash import send2trash  #lo lascio per magari aggiungere codice per togliere risultati se già presenti nella cartella



# Parametri principali
PorosityMin = 0.05
PorosityMax = 0.3
NbPorosity = 5 #numero di valori di porosità da studiare
R = 0.8
Kmatrix = 1
Kgases = 1e-3
NbSeed = 1 # numero calcoli per ogni valore di porosità
NbVoxellation = 40
DeltaP = (PorosityMax - PorosityMin) / (NbPorosity - 1)
C = 20
SizeRVE = [C, C, C]
Voxellation = [NbVoxellation, NbVoxellation, NbVoxellation]
K = [Kmatrix, Kgases]

Cmm = 3 #lunghezza RVE in mm
Rmicron = R/C *Cmm*1000 #raggio porosità in micron

file_output_path = "risultato_conduttivita.txt"  # Unico file di output
file_aggregato_path = os.path.join("/home/alessio/Merope/Esercitazione", "conductivity_results.txt")  # File aggregato nella directory principale

# Funzione per generare la voxellation e la struttura inclusa
def VoxellationInclusion(SizeRVE, Seed, R, Porosity, K, Voxellation):
    sphIncl = merope.SphereInclusions_3D()
    sphIncl.setLength(SizeRVE)
    sphIncl.fromHisto(Seed, sac_de_billes.TypeAlgo.RSA, 0., [[R, Porosity]], [1])
    multiInclusions = merope.MultiInclusions_3D()
    multiInclusions.setInclusions(sphIncl)
    grid = merope.Voxellation_3D(multiInclusions)
    grid.setPureCoeffs(K)
    grid.setHomogRule(merope.HomogenizationRule.Voigt)
    grid.setVoxelRule(merope.VoxelRule.Average)
    grid.proceed(Voxellation)
    grid.printFile("Zone.vtk", "Coeffs.txt")

# Funzione per calcolare la conducibilità termica
def ThermalAmitex():
    number_of_processors = 2
    amitex.computeThermalCoeff("Zone.vtk", number_of_processors)
    amitex_out.printThermalCoeff(".")

# Funzione per estrarre e scrivere i valori di conduttività nel file di output
def scrivi_valori_su_file(file_output, valori, porosity, seed_index):
    media = sum(valori) / len(valori)
    num_decimali = len(str(valori[0]).split('.')[1]) if '.' in str(valori[0]) else 0
    media_formattata = f"{media:.{num_decimali}f}"
    with open(file_output, 'a') as f:
        linea = f"Porosity_{porosity:.2f}\tSeed_{seed_index}\t{valori[0]}\t{valori[1]}\t{valori[2]}\t{media_formattata}\n"
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
def aggiorna_file_aggregato(file_aggregato, porosity, seed_index, valori, C, Cmm, Rmicron, NbVoxellation, Kmatrix, Kgases):
    media = sum(valori) / len(valori)
    num_decimali = len(str(valori[0]).split('.')[1]) if '.' in str(valori[0]) else 0
    media_formattata = float(f"{media:.{num_decimali}f}")  # Converti media_formattata in float
    
    # Se il file non esiste, crea l'intestazione e scrivi i parametri di input
    if not os.path.exists(file_aggregato):
        with open(file_aggregato, 'w') as f:
            f.write("Input Parameters:\n")
            f.write(f"C: {C}\n")
            f.write(f"Cmm: {Cmm}\n")
            f.write(f"Rmicron: {Rmicron:.2f} micrometri\n")
            f.write(f"NbVoxellation: {NbVoxellation}\n")
            f.write(f"Kmatrix: {Kmatrix}\n")
            f.write(f"Kgases: {Kgases}\n")
            f.write("Porosity\tSeed_Index\tK_xx\tK_yy\tK_zz\tK_mean\n")  # Intestazione delle colonne

    # Aggiungi i risultati
    with open(file_aggregato, 'a') as f:
        linea = f"{porosity:.4f}\t{seed_index}\t{valori[0]:.4f}\t{valori[1]:.4f}\t{valori[2]:.4f}\t{media_formattata:.4f}\n"
        f.write(linea)

# Funzione principale per generare strutture e salvare i risultati
def main():
    if os.path.exists(file_output_path):
        open(file_output_path, 'w').close()

    # Inizializza il valore della porosità
    porosity = PorosityMin
    for i in range(NbPorosity):
        os.mkdir(f'Porosity of {porosity:.4f}')
        os.chdir(f'Porosity of {porosity:.4f}')
        for seed in range(NbSeed):
            os.mkdir(f'Seed of {seed}')
            os.chdir(f'Seed of {seed}')
            VoxellationInclusion(SizeRVE, seed, R, porosity, K, Voxellation)
            ThermalAmitex()

            # Estrai e salva i dati di conduttività
            matrice = leggi_matrice_da_file("thermalCoeff_amitex.txt")
            valori_estratti = estrai_valori_principali(matrice)
            scrivi_valori_su_file(file_output_path, valori_estratti, porosity, seed)

            # Aggiungi i risultati al file aggregato, includendo i parametri di input
            aggiorna_file_aggregato(file_aggregato_path, porosity, seed, valori_estratti, C, Cmm, Rmicron, NbVoxellation, Kmatrix, Kgases)

            os.chdir("../")
        os.chdir("../")
        porosity += DeltaP

if __name__ == "__main__":
    main()

  ## Questa parte serve in caso questo codice debba essere importato, senza questa se il modulo fosse importato verrebbe anche eseguito, in questo modo ciò non avviene


