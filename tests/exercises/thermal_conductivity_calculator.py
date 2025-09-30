"""
Script to compute homogenized thermal conductivity as a function of porosity.
Mérope builds the microstructure and voxelization, AMITEX computes the
homogenized conductivity, and results are stored in structured output files.
"""

import os
import sac_de_billes
import merope

import archi_merope as arch
import interface_amitex_fftp.amitex_wrapper as amitex
import interface_amitex_fftp.post_processing as amitex_out


# ----------------------------------------------------------------------
# Main input parameters
# ----------------------------------------------------------------------

PorosityMin = 0.01         # minimum porosity
PorosityMax = 99.00        # maximum porosity
NbPorosity = 10            # number of porosity values to investigate
R = 0.8                    # pore radius (normalized by cell size)
Kmatrix = 1                # thermal conductivity of the solid matrix
Kgases = 1e-3              # thermal conductivity of the pores
NbSeed = 1                 # number of seeds (random structures) for each porosity
NbVoxellation = 40         # voxel grid resolution
DeltaP = (PorosityMax - PorosityMin) / (NbPorosity - 1)

C = 20                     # size of RVE (arbitrary units)
SizeRVE = [C, C, C]        # cubic RVE dimensions
Voxellation = [NbVoxellation, NbVoxellation, NbVoxellation]
K = [Kmatrix, Kgases]

Cmm = 3                    # physical size of RVE in mm
Rmicron = R / C * Cmm * 1000  # pore radius in microns


# ----------------------------------------------------------------------
# Output root directory
# ----------------------------------------------------------------------
root_dir = os.path.join(os.getcwd(), "thermal_conductivity_calculator")
os.makedirs(root_dir, exist_ok=True)

results_dir = os.path.join(root_dir, "results")
os.makedirs(results_dir, exist_ok=True)

file_output_path = os.path.join(results_dir, "conductivity_results_single.txt")
file_aggregated_path = os.path.join(results_dir, "conductivity_results_all.txt")


# ----------------------------------------------------------------------
# Functions
# ----------------------------------------------------------------------

def VoxellationInclusion(SizeRVE, Seed, R, Porosity, K, Voxellation):
    """Generate voxelized structure with spherical pores."""
    sphIncl = merope.SphereInclusions_3D()
    sphIncl.setLength(SizeRVE)
    sphIncl.fromHisto(Seed, sac_de_billes.TypeAlgo.RSA, 0., [[R, Porosity]], [1])

    multiInclusions = merope.MultiInclusions_3D()
    multiInclusions.setInclusions(sphIncl)

    grid = merope.Voxellation_3D(multiInclusions)
    grid.setPureCoeffs(K)
    grid.setHomogRule(merope.HomogenizationRule.Voigt)
    grid.setVoxelRule(merope.vox.VoxelRule.Average)
    grid.proceed(Voxellation)

    grid.printFile("Zone.vtk", "Coeffs.txt")


def ThermalAmitex():
    """Run AMITEX to compute homogenized thermal conductivity."""
    number_of_processors = 2
    amitex.computeThermalCoeff("Zone.vtk", number_of_processors)
    amitex_out.printThermalCoeff(".")


def write_results_single(file_output, values, porosity, seed_index):
    """Write single porosity/seed conductivity results to file."""
    mean_val = sum(values) / len(values)
    num_decimals = len(str(values[0]).split('.')[1]) if '.' in str(values[0]) else 0
    mean_fmt = f"{mean_val:.{num_decimals}f}"

    with open(file_output, 'a') as f:
        line = (
            f"porosity_{porosity:.2f}\tseed_{seed_index}\t"
            f"{values[0]}\t{values[1]}\t{values[2]}\t{mean_fmt}\n"
        )
        f.write(line)


def read_matrix(file_path):
    """Read conductivity matrix from AMITEX output file."""
    with open(file_path, 'r') as f:
        matrix = [list(map(float, line.split())) for line in f.readlines()]
    return matrix


def extract_diagonal(matrix):
    """Extract main diagonal values (Kxx, Kyy, Kzz) from conductivity matrix."""
    return [matrix[i][i] for i in range(3)]


def update_aggregated_file(file_aggregated_path, porosity, seed_index, values,
                           C, Cmm, Rmicron, NbVoxellation, Kmatrix, Kgases):
    """Append conductivity results to aggregated file, with input parameters."""
    mean_val = sum(values) / len(values)
    num_decimals = len(str(values[0]).split('.')[1]) if '.' in str(values[0]) else 0
    mean_fmt = float(f"{mean_val:.{num_decimals}f}")

    if not os.path.exists(file_aggregated_path):
        with open(file_aggregated_path, 'w') as f:
            f.write("Input Parameters:\n")
            f.write(f"C: {C}\n")
            f.write(f"Cmm: {Cmm}\n")
            f.write(f"Rmicron: {Rmicron:.2f} microns\n")
            f.write(f"NbVoxellation: {NbVoxellation}\n")
            f.write(f"Kmatrix: {Kmatrix}\n")
            f.write(f"Kgases: {Kgases}\n")
            f.write("Porosity\tSeed_Index\tK_xx\tK_yy\tK_zz\tK_mean\n")

    with open(file_aggregated_path, 'a') as f:
        line = (
            f"{porosity:.4f}\t{seed_index}\t"
            f"{values[0]:.4f}\t{values[1]:.4f}\t{values[2]:.4f}\t{mean_fmt:.4f}\n"
        )
        f.write(line)


# ----------------------------------------------------------------------
# Main routine
# ----------------------------------------------------------------------

def main():
    if os.path.exists(file_output_path):
        open(file_output_path, 'w').close()

    porosity = PorosityMin
    for i in range(NbPorosity):
        porosity_dir = os.path.join(root_dir, f'porosity_{porosity:.4f}')
        os.makedirs(porosity_dir, exist_ok=True)
        os.chdir(porosity_dir)

        for seed in range(NbSeed):
            seed_dir = os.path.join(porosity_dir, f'seed_{seed}')
            os.makedirs(seed_dir, exist_ok=True)
            os.chdir(seed_dir)

            VoxellationInclusion(SizeRVE, seed, R, porosity, K, Voxellation)
            ThermalAmitex()

            matrix = read_matrix("thermalCoeff_amitex.txt")
            values = extract_diagonal(matrix)

            write_results_single(file_output_path, values, porosity, seed)
            update_aggregated_file(
                file_aggregated_path, porosity, seed, values,
                C, Cmm, Rmicron, NbVoxellation, Kmatrix, Kgases
            )

            os.chdir(porosity_dir)
        os.chdir(root_dir)
        porosity += DeltaP


if __name__ == "__main__":
    main()
