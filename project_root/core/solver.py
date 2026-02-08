import interface_amitex_fftp.amitex_wrapper as amitex
import interface_amitex_fftp.post_processing as amitex_out
import numpy as np
import os

class ThermalSolver:
    def __init__(self, n_cpus=4):
        self.n_cpus = n_cpus

    def solve(self, vtk_file="structure.vtk"):
        """Runs the Amitex solver and returns the homogenized thermal conductivity."""
        if not os.path.exists(vtk_file):
            raise FileNotFoundError(f"VTK file {vtk_file} not found.")
            
        amitex.computeThermalCoeff(vtk_file, self.n_cpus)
        return self._parse_results("thermalCoeff_amitex.txt")

    def _parse_results(self, file_path):
        """Reads the 3x3 matrix from Amitex output."""
        with open(file_path, 'r') as f:
            lines = f.readlines()
            matrix = np.array([list(map(float, line.split())) for line in lines])
        
        return {
            "Kxx": matrix[0, 0],
            "Kyy": matrix[1, 1],
            "Kzz": matrix[2, 2],
            "Kmean": np.trace(matrix) / 3.0
        }