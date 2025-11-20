import os
import shutil
import numpy as np

import sac_de_billes
import merope

import archi_merope as arch
import interface_amitex_fftp.amitex_wrapper as amitex
import interface_amitex_fftp.post_processing as amitex_out

# -----------------------------------------------------------
# Configuration
# -----------------------------------------------------------

L = [10, 10, 10]  # RVE size
n3D = 100         # voxels per side
seed = 0
NbSeed = 1

voxel_rule = merope.vox.VoxelRule.Average
homogRule = merope.HomogenizationRule.Voigt

folder_name = "Result_0"
file_output_path = "porDelta_conduct_results.txt"

vtkname = "Zone.vtk"
fileCoeff = "Coeffs.txt"

inclR = 0.3       # intergranular pore radius
incl2R = 0.05     # intragranular pore radius

lagR = 3     # grain size
lagPhi = 1   # keep = 1 to fill RVE

inclPhi = np.linspace(0.02, 0.20, 10)   # porDelta, intergranular porosity
incl2Phi = np.linspace(0.01, 0.031, 10) # porIntra, distributed porosity

delta = 3 # grain boundary layer thickness
delta_ratio = delta / lagR # layer thickness / grain size

delta_phase = 3   # phase id for grain boundary layer
incl_phase = 2    # phase id for intergranular pores
grains_phase = 0  # phase id for grains

Kmatrix = 1
Kgases = 1e-03
K = [Kmatrix, Kmatrix, Kgases]


# -----------------------------------------------------------
# FUNZIONI
# -----------------------------------------------------------

def build_microstructure(n3D, L, seed, inclR, inclPhi, lagRphi,
                         incl2R, incl2Phi, incl_phase, grains_phase,
                         delta_phase, delta, voxel_rule, K, vtkname, fileCoeff):
    """
    Genera una microstruttura mista (intergranulare + intragranulare)
    e ritorna porDelta + porIntra.
    """

    # -------------------------------
    # 1. Pori intragranulari (sferici)
    # -------------------------------
    sph_intra = merope.SphereInclusions_3D()
    sph_intra.setLength(L)
    sph_intra.fromHisto(seed, sac_de_billes.TypeAlgo.BOOL, 0.,
                        [[inclR, inclPhi]], [incl_phase])
    multi_intra = merope.MultiInclusions_3D()
    multi_intra.setInclusions(sph_intra)

    # -------------------------------
    # 2. Secondo set di pori intragranulari
    # -------------------------------
    sph_intra2 = merope.SphereInclusions_3D()
    sph_intra2.setLength(L)
    sph_intra2.fromHisto(seed, sac_de_billes.TypeAlgo.RSA, 0.,
                         [[incl2R, incl2Phi]], [1])
    multi_intra2 = merope.MultiInclusions_3D()
    multi_intra2.setInclusions(sph_intra2)
    structure_intra2 = merope.Structure_3D(multi_intra2)

    # -------------------------------
    # 3. Grain structure (Laguerre)
    # -------------------------------
    sph_lag = merope.SphereInclusions_3D()
    sph_lag.setLength(L)
    sph_lag.fromHisto(seed, sac_de_billes.TypeAlgo.RSA, 0., [lagRphi], [1])

    poly = merope.LaguerreTess_3D(L, sph_lag.getSpheres())
    multi_poly = merope.MultiInclusions_3D()
    multi_poly.setInclusions(poly)

    # num grains
    N = len(multi_poly.getAllIdentifiers())

    # aggiungi layer ai grain boundary (fase delta)
    multi_poly.addLayer(multi_poly.getAllIdentifiers(), delta_phase, delta)

    # converte tutti i grain in fase 1 (solido)
    multi_poly.changePhase(multi_poly.getAllIdentifiers(),
                           [1 for _ in multi_poly.getAllIdentifiers()])

    structure_grains = merope.Structure_3D(multi_poly)
    structure_intra = merope.Structure_3D(multi_intra)

    # -------------------------------
    # 4. Fusioni tra strutture
    # -------------------------------

    # rende i pori "dentro i bordi" vuoto (fase 0)
    dict1 = {2: 0, 3: 0}
    structure_mix1 = merope.Structure_3D(multi_intra, multi_poly, dict1)

    # combina crack-layer + pore-spheres
    dict2 = {0: 2}
    final_structure = merope.Structure_3D(structure_mix1, structure_intra2, dict2)

    # -------------------------------
    # 5. Voxellation
    # -------------------------------
    grid_params = merope.vox.create_grid_parameters_N_L_3D(
        [n3D, n3D, n3D], L
    )

    grid = merope.vox.GridRepresentation_3D(final_structure,
                                            grid_params,
                                            voxel_rule)

    analyzer = merope.vox.GridAnalyzer_3D()
    phase_fracs = analyzer.compute_percentages(grid)
    analyzer.print_percentages(grid)

    # applica la regola Voigt
    grid.apply_homogRule(homogRule, K)

    # esporta la voxelizzazione
    printer = merope.vox.vtk_printer_3D()
    printer.printVTK_segmented(grid, vtkname, fileCoeff, nameValue="MaterialId")

    porDelta = phase_fracs[2]   # pori intergranulari
    porIntra = incl2Phi         # pori intragranulari

    return porDelta, porIntra


def run_amitex(vtkname):
    """ Lancia AMITEX e ritorna la matrice 3×3. """
    number_of_processors = 6
    amitex.computeThermalCoeff(vtkname, number_of_processors)
    matrix = amitex_out.printThermalCoeff(".")
    return matrix


def extract_diagonal(matrix):
    """ Estrae kxx, kyy, kzz. """
    return [matrix[i][i] for i in range(3)]


def append_seed_results(file_output, porDelta, porIntra, seed, kvals):
    mean_k = np.mean(kvals)
    with open(file_output, 'a') as f:
        f.write(f"{porDelta:.4f}\t{porIntra:.4f}\tSeed_{seed}\t"
                f"{kvals[0]:.6f}\t{kvals[1]:.6f}\t{kvals[2]:.6f}\t{mean_k:.6f}\n")


def append_aggregated(file_path, params, porDelta, porIntra, seed, kvals):
    mean_k = np.mean(kvals)

    # crea header se file non esiste
    if not os.path.exists(file_path):
        with open(file_path, "w") as f:
            f.write("Input Parameters:\n")
            for k, v in params.items():
                f.write(f"{k}: {v}\n")
            f.write("PorDelta\tPorIntra\tPorTot\tSeed\tKxx\tKyy\tKzz\tKmean\n")

    # append valori
    with open(file_path, "a") as f:
        f.write(f"{porDelta:.4f}\t{porIntra:.4f}\t{porDelta+porIntra:.4f}\t{seed}\t"
                f"{kvals[0]:.6f}\t{kvals[1]:.6f}\t{kvals[2]:.6f}\t{mean_k:.6f}\n")


# -----------------------------------------------------------
# MAIN
# -----------------------------------------------------------

def main():

    root = os.getcwd()  # salva directory principale

    # Path assoluto della cartella risultati
    result_dir = os.path.join(root, folder_name)

    # crea cartella risultati da zero
    if os.path.exists(result_dir):
        shutil.rmtree(result_dir)
    os.mkdir(result_dir)

    aggregated_file = os.path.join(result_dir, "aggregated_results.txt")

    # parametri da scrivere nell’header dell’aggregato
    params = {
        "Delta": delta,
        "RVE size": L,
        "inclR": inclR,
        "grain size lagR": lagR,
        "n3D": n3D,
        "Kmatrix": Kmatrix,
        "Kgases": Kgases,
    }

    # reset file seed-wise (scrivo nella directory root)
    open(os.path.join(root, file_output_path), 'w').close()

    # loop combinazioni
    for j, phi2 in enumerate(incl2Phi):
        for i, phi1 in enumerate(inclPhi):
            for seed in range(NbSeed):

                case_folder = os.path.join(result_dir, f"Seed_{seed}_i{i}_j{j}")
                os.mkdir(case_folder)

                # lavoro dentro la cartella del caso
                os.chdir(case_folder)

                # build microstructure
                porDelta, porIntra = build_microstructure(
                    n3D, L, seed, inclR, phi1,
                    [lagR, lagPhi], incl2R, phi2,
                    incl_phase, grains_phase,
                    delta_phase, delta, voxel_rule,
                    K, vtkname, fileCoeff
                )

                # run AMITEX
                matrix = run_amitex(vtkname)
                kvals = extract_diagonal(matrix)

                # path sicuri assoluti
                append_seed_results(
                    os.path.join(root, file_output_path),
                    porDelta, porIntra, seed, kvals
                )

                append_aggregated(
                    aggregated_file,
                    params,
                    porDelta, porIntra, seed, kvals
                )

                # ritorno SEMPRE alla root
                os.chdir(root)

    print("Simulazioni completate.")

if __name__ == "__main__":
    main()
