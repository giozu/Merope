import os
import shutil
import numpy as np
import sac_de_billes
import merope

import archi_merope as arch
import interface_amitex_fftp.amitex_wrapper as amitex
import interface_amitex_fftp.post_processing as amitex_out

# -----------------------------------------------------------
# Simulation parameters
# -----------------------------------------------------------

L = [20, 20, 20]       # RVE size
n3D = 200              # voxel resolution
lagR = 3               # grain size (used also in normalization)
lagPhi = 1             # full RVE with grains (100% grains)

voxel_rule = merope.vox.VoxelRule.Average
homogRule = merope.HomogenizationRule.Voigt

Kmatrix = 1.0
Kgases  = 1e-3
K = [Kmatrix, Kmatrix, Kgases]

inclR    = 0.5   # base radius for phase 2 inclusions (pores)
incl2R   = 0.2  # intragranular pore radius

incl_phase = 2 # porosity_phase = 2
delta_phase = 3
grains_phase = 0

# porosities to test
p_values = [0.1, 0.2, 0.3]

# delta ratios to test
delta_ratio_values = np.linspace(0.1, 0.9, 10)

# result folder
output_dir = "DeltaScan_Results"

# -----------------------------------------------------------
# Geometry diagnostics
# -----------------------------------------------------------

voxel_size = L[0] / n3D

def print_geometry_diagnostics(name, R):
    ratio_RVE = L[0] / R
    ratio_voxel = R / voxel_size
    print(f"\nGeometry diagnostic for {name}:")
    print(f"  - L_RVE / R_pore   = {ratio_RVE:.2f}")
    print(f"  - R_pore / L_voxel = {ratio_voxel:.2f}")
    if ratio_voxel < 1.0:
        print("  WARNING: pore radius < voxel size → unresolved geometry!")
    elif ratio_voxel < 2.0:
        print("  WARNING: pore radius ≈ 1–2 voxel → very low resolution")
    elif ratio_voxel < 4.0:
        print("  CAUTION: pore radius ≈ 2–4 voxel → borderline resolution")
    else:
        print("  OK: pore well resolved (>4 voxels per radius)")

print_geometry_diagnostics("intergranular pores", inclR)
print_geometry_diagnostics("intragranular pores", incl2R)

print(f"\nGlobal voxel size = {voxel_size:.4f}")
print("-----------------------------------------------------------\n")


# -----------------------------------------------------------
# Microstructure builder
# -----------------------------------------------------------

def build_microstructure(n3D, L, seed, p_delta, p_intra,
                         delta_phys, voxel_rule, K):

    # --- Intragranular pores (fase 2) ---
    sph_intra = merope.SphereInclusions_3D()
    sph_intra.setLength(L)
    sph_intra.fromHisto(seed,
                        sac_de_billes.TypeAlgo.BOOL,
                        0.,
                        [[incl2R, p_intra]],
                        [2])    # fase 2 = poro
    multi_intra = merope.MultiInclusions_3D()
    multi_intra.setInclusions(sph_intra)
    struct_intra = merope.Structure_3D(multi_intra)

    # --- Intergranular pores (fase 2) ---
    sph_gb = merope.SphereInclusions_3D()
    sph_gb.setLength(L)
    sph_gb.fromHisto(seed,
                     sac_de_billes.TypeAlgo.RSA,
                     0.,
                     [[inclR, p_delta]],
                     [2])
    multi_gb = merope.MultiInclusions_3D()
    multi_gb.setInclusions(sph_gb)

    # --- Laguerre grains (fase 1) + layer fase 3 ---
    sph_lag = merope.SphereInclusions_3D()
    sph_lag.setLength(L)
    sph_lag.fromHisto(seed,
                      sac_de_billes.TypeAlgo.RSA,
                      0.,
                      [[lagR, lagPhi]],
                      [1])

    poly = merope.LaguerreTess_3D(L, sph_lag.getSpheres())
    multi_poly = merope.MultiInclusions_3D()
    multi_poly.setInclusions(poly)

    multi_poly.addLayer(multi_poly.getAllIdentifiers(), 3, delta_phys)
    multi_poly.changePhase(multi_poly.getAllIdentifiers(),
                           [1 for _ in multi_poly.getAllIdentifiers()])

    struct_poly = merope.Structure_3D(multi_poly)

    # ============================================================
    #  MERGE "ALLA TESISTA"
    #
    #   primo merge:
    #       intergranular pores (2) → 0
    #       film layer (3)         → 0
    #
    #   secondo merge:
    #       fase 0 → 2 (vero poro)
    # ============================================================

    # grains+layer con inter-granular pores
    struct_inter_poly = merope.Structure_3D(
        multi_gb,
        multi_poly,
        {2: 0, 3: 0}
    )

    # aggiungi pori intra e rimappa:
    final_struct = merope.Structure_3D(
        struct_inter_poly,
        struct_intra,
        {0: 2}
    )

    # voxelization
    grid_params = merope.vox.create_grid_parameters_N_L_3D([n3D]*3, L)
    grid = merope.vox.GridRepresentation_3D(final_struct,
                                            grid_params,
                                            voxel_rule)

    analyzer = merope.vox.GridAnalyzer_3D()
    phase_fracs = analyzer.compute_percentages(grid)

    print("\nDEBUG -- phase fractions:")
    for pid, frac in phase_fracs.items():
        print(f"  phase {pid}: {frac:.6f}")
    print("------------\n")

    grid.apply_homogRule(homogRule, K)

    return grid, phase_fracs


def build_microstructure_gauss(n3D, L, seed,
                               p_inter,          # porDelta (pori ai grain boundaries)
                               p_intra,          # porIntra (pori nei grani)
                               delta_phys,       # spessore film GB
                               voxel_rule, K,
                               mean_radius, std_radius, num_radius):

    # ============================================================
    # 1) PORI INTERGRANULARI (fase 2)
    # ============================================================
    sph_inter = merope.SphereInclusions_3D()
    sph_inter.setLength(L)
    sph_inter.fromHisto(seed,
                        sac_de_billes.TypeAlgo.BOOL,
                        0.,
                        [[inclR, p_inter]],   # r = inclR, volume = p_inter
                        [2])                  # phase 2 = poro
    multi_inter = merope.MultiInclusions_3D()
    multi_inter.setInclusions(sph_inter)


    # ============================================================
    # 2) PORI INTRAGRANULARI GAUSSIANI (fase 1, verranno poi mappati)
    # ============================================================
    hist_intra = generate_spheres(p_intra, mean_radius, std_radius, num_radius)

    sph_intra = merope.SphereInclusions_3D()
    sph_intra.setLength(L)
    sph_intra.fromHisto(seed,
                        sac_de_billes.TypeAlgo.BOOL,
                        0.,
                        hist_intra,
                        [1 for _ in range(num_radius)])  # fase 1 → poi rimappata

    multi_intra = merope.MultiInclusions_3D()
    multi_intra.setInclusions(sph_intra)
    struct_intra = merope.Structure_3D(multi_intra)


    # ============================================================
    # 3) LAGUERRE GRAINS + FILM δ AI GRAIN BOUNDARIES
    # ============================================================
    sph_lag = merope.SphereInclusions_3D()
    sph_lag.setLength(L)
    sph_lag.fromHisto(seed,
                      sac_de_billes.TypeAlgo.RSA,
                      0.,
                      [[lagR, lagPhi]],
                      [1])  # grani = fase 1

    poly = merope.LaguerreTess_3D(L, sph_lag.getSpheres())
    multi_poly = merope.MultiInclusions_3D()
    multi_poly.setInclusions(poly)

    # aggiunge film GB in fase 3
    multi_poly.addLayer(multi_poly.getAllIdentifiers(), 3, delta_phys)

    # rimette i grani nella fase 1
    multi_poly.changePhase(multi_poly.getAllIdentifiers(),
                           [1 for _ in multi_poly.getAllIdentifiers()])

    struct_poly = merope.Structure_3D(multi_poly)


    # ============================================================
    # 4) MERGE COME NEL CODICE DEL TESISTA
    #
    #   Primo merge:
    #       fase 2 (inter-pori) → 0
    #       fase 3 (film GB)    → 0
    #
    #   Secondo merge:
    #       fase 0 → fase 2 (poro finale)
    #
    # ============================================================

    # inter + grains+layer
    struct_inter_poly = merope.Structure_3D(
        multi_inter,     # contiene pori inter (fase 2)
        multi_poly,      # contiene grani (1) + layer (3)
        {2: 0, 3: 0}     # mappa 2 → 0, 3 → 0 (vuoto temporaneo)
    )

    # ora combina con pori intra gaussiani
    final_struct = merope.Structure_3D(
        struct_inter_poly,
        struct_intra,
        {0: 2}     # tutto ciò che era fase 0 diventa PORO (fase 2)
    )

    # RISULTATO FINALE:
    #   fase 1 → solido (matrix)
    #   fase 2 → poro   (inter + intra + film GB)
    #   fase 3 → non presente (riassorbita nel mapping)
    #   fase 0 → temporanea (sparita)


    # ============================================================
    # 5) VOXELLIZZAZIONE
    # ============================================================
    grid_params = merope.vox.create_grid_parameters_N_L_3D([n3D]*3, L)
    grid = merope.vox.GridRepresentation_3D(final_struct,
                                            grid_params,
                                            voxel_rule)

    analyzer = merope.vox.GridAnalyzer_3D()
    phase_fracs = analyzer.compute_percentages(grid)  # lista: frazioni di fase 0,1,2,...

    print("\nDEBUG -- phase fractions:")
    for pid, frac in phase_fracs.items():
        print(f"  phase {pid}: {frac:.6f}")
    print("------------\n")


    grid.apply_homogRule(homogRule, K)

    # Non stampo VTK (lo fai nel main)
    return grid, phase_fracs




def generate_spheres(target_porosity, mean_radius, std_radius, num_radius):
    radii = np.abs(np.random.normal(mean_radius, std_radius, num_radius))  
      
    fractions = np.random.rand(num_radius)
    fractions = (fractions / np.sum(fractions)) * target_porosity

    return [[r, f] for r, f in zip(radii, fractions)]


# -----------------------------------------------------------
# AMITEX
# -----------------------------------------------------------

def run_amitex(vtkname):
    amitex.computeThermalCoeff(vtkname, 6)
    Kmat = amitex_out.printThermalCoeff(".")
    return [Kmat[i][i] for i in range(3)]


# -----------------------------------------------------------
# MAIN
# -----------------------------------------------------------

def main():

    # reset output folder
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.mkdir(output_dir)

    outfile = os.path.join(output_dir, "delta_scan_results.txt")
    with open(outfile, "w") as f:
        f.write("p_target\tp_por_meas\tp_delta\tp_intra\tdelta_ratio\tdelta_phys\t"
                "Kxx\tKyy\tKzz\tKmean\n")

    seed = 0

    for p_tot in p_values:
        for delta_ratio in delta_ratio_values:

            delta_phys = delta_ratio * lagR

            # porosità totale fissa: per delta piccolo tutta nei GB,
            # per delta grande quasi tutta intragranulare
            p_delta = p_tot * (1.0 - delta_ratio)   # porosità nei grain boundaries
            p_intra = p_tot - p_delta               # porosità intragranulare

            # build microstructure
            grid, phases = build_microstructure(
                n3D, L, seed,
                p_delta, p_intra,
                delta_phys,
                voxel_rule, K
            )

            # grid, phases = build_microstructure_gauss(
            #     n3D, L, seed,
            #     p_inter=p_delta,
            #     p_intra=p_intra,
            #     delta_phys=delta_phys,
            #     voxel_rule=voxel_rule,
            #     K=K,
            #     mean_radius=0.3,
            #     std_radius=0.13,
            #     num_radius=5
            # )

            p_solid = phases.get(grains_phase, 0.0)      # frazione di matrice
            p_por   = phases.get(incl_phase, 0.0)    # frazione porosa effettiva

            # export VTK for AMITEX
            printer = merope.vox.vtk_printer_3D()
            printer.printVTK_segmented(grid, "Zone.vtk", "Coeffs.txt",
                                       nameValue="MaterialId")

            # run AMITEX
            Kvals = run_amitex("Zone.vtk")
            Kmean = np.mean(Kvals)

            print("phase_fracs:", phases)
            print("available phases:", phases.keys())

            # save result
            with open(outfile, "a") as f:
                f.write(f"{p_tot:.4f}\t{p_por:.4f}\t{p_delta:.4f}\t{p_intra:.4f}\t"
                        f"{delta_ratio:.4f}\t{delta_phys:.4f}\t"
                        f"{Kvals[0]:.6f}\t{Kvals[1]:.6f}\t{Kvals[2]:.6f}\t{Kmean:.6f}\n")

    print("Delta scan completed.")


if __name__ == "__main__":
    main()
