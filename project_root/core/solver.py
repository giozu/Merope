import interface_amitex_fftp.amitex_wrapper as amitex
import interface_amitex_fftp.post_processing as amitex_out
import numpy as np
import os

class ThermalSolver:
    def __init__(self, n_cpus=4):
        self.n_cpus = n_cpus

    def solve(self, vtk_file="structure.vtk"):
        res_file = "thermalCoeff_amitex.txt"
        
        # Clean up old results to avoid reading stale data
        if os.path.exists(res_file):
            os.remove(res_file)

        print(f"--- Running Amitex Solver on {vtk_file} ---")
        try:
            amitex.computeThermalCoeff(vtk_file, self.n_cpus)
            amitex_out.printThermalCoeff(".")
        except Exception as e:
            print(f"Critical Solver Error: {e}")

        if os.path.exists(res_file):
            print(f"Successfully generated {res_file}")
            return self._parse_results(res_file)
        else:
            print(f"ERROR: Amitex failed to produce {res_file}. Check your Amitex installation.")
            return {"Kxx": 0, "Kyy": 0, "Kzz": 0, "Kmean": 0}

    def _parse_results(self, file_path):
        try:
            data = np.loadtxt(file_path)
            if data.size == 0: return {"Kxx":0,"Kyy":0,"Kzz":0,"Kmean":0}
            # Amitex outputs a 3x3 matrix
            matrix = data[:3, :3]
            return {
                "Kxx": matrix[0, 0], "Kyy": matrix[1, 1], "Kzz": matrix[2, 2],
                "Kmean": np.trace(matrix) / 3.0
            }
        except Exception as e:
            print(f"Error parsing results: {e}")
            return {"Kxx": 0, "Kyy": 0, "Kzz": 0, "Kmean": 0}