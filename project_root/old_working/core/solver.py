from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Union

import interface_amitex_fftp.amitex_wrapper as amitex
import interface_amitex_fftp.post_processing as amitex_out
import numpy as np


class ThermalSolver:
    """Thin wrapper around Amitex-FFTP for thermal conductivity problems.

    This class is intentionally minimal: it assumes that a segmented VTK file
    compatible with Amitex has already been written to disk (typically via
    :meth:`core.geometry.MicrostructureBuilder.voxellate`).
    """

    def __init__(self, n_cpus: int = 4) -> None:
        """Create a solver instance.

        Parameters
        ----------
        n_cpus :
            Number of MPI processes (or threads, depending on your Amitex
            build) to use when calling ``computeThermalCoeff``.
        """
        if n_cpus <= 0:
            raise ValueError(f"n_cpus must be positive, got {n_cpus}")
        self.n_cpus: int = int(n_cpus)

    def solve(
        self,
        vtk_file: Union[str, Path] = "structure.vtk",
        results_file: Union[str, Path] = "thermalCoeff_amitex.txt",
    ) -> Dict[str, float]:
        """Run Amitex on a given VTK file and parse the homogenized tensor.

        Parameters
        ----------
        vtk_file :
            Path to the segmented VTK file exported by Mérope. The file must
            exist and be readable by Amitex.
        results_file :
            Path where Amitex writes the effective conductivity matrix (ASCII).
            The default corresponds to the standard ``printThermalCoeff`` output.

        Returns
        -------
        dict[str, float]
            A dictionary with keys ``\"Kxx\"``, ``\"Kyy\"``, ``\"Kzz\"`` and
            ``\"Kmean\"`` (mean of the diagonal). If anything goes wrong
            (missing files, parse errors, etc.), all values are set to zero.
        """
        vtk_path = Path(vtk_file).resolve()
        res_path = Path(results_file)

        if not vtk_path.is_file():
            print(f"ERROR: VTK file not found, skipping Amitex run: {vtk_path}")
            return {"Kxx": 0.0, "Kyy": 0.0, "Kzz": 0.0, "Kmean": 0.0}

        # Amitex reads Coeffs.txt and writes thermalCoeff_amitex.txt relative to
        # the current working directory — so we must run from the VTK's directory.
        work_dir = vtk_path.parent
        if not res_path.is_absolute():
            res_path = work_dir / res_path

        # Clean up old results to avoid reading stale data.
        if res_path.exists():
            res_path.unlink()

        prev_dir = os.getcwd()
        print(f"--- Running Amitex Solver on {vtk_path} ---")
        try:
            os.chdir(work_dir)
            amitex.computeThermalCoeff(str(vtk_path), self.n_cpus)
            amitex_out.printThermalCoeff(".")
        except Exception as e:  # pragma: no cover - external solver failure
            print(f"Critical Solver Error: {e}")
        finally:
            os.chdir(prev_dir)

        if res_path.is_file():
            print(f"Successfully generated {res_path}")
            return self._parse_results(res_path)

        print(
            f"ERROR: Amitex failed to produce {res_path}. "
            "Check your Amitex installation and input VTK file."
        )
        return {"Kxx": 0.0, "Kyy": 0.0, "Kzz": 0.0, "Kmean": 0.0}

    def _parse_results(self, file_path: Union[str, Path]) -> Dict[str, float]:
        """Parse the 3x3 conductivity matrix written by Amitex.

        Parameters
        ----------
        file_path :
            Path to the ASCII file produced by ``printThermalCoeff``.
        """
        path = Path(file_path)
        try:
            data = np.loadtxt(path)
            if data.size == 0:
                return {"Kxx": 0.0, "Kyy": 0.0, "Kzz": 0.0, "Kmean": 0.0}

            # Amitex outputs (at least) a 3x3 conductivity tensor.
            matrix = np.asarray(data)[:3, :3]
            kxx = float(matrix[0, 0])
            kyy = float(matrix[1, 1])
            kzz = float(matrix[2, 2])
            kmean = float(np.trace(matrix) / 3.0)

            return {"Kxx": kxx, "Kyy": kyy, "Kzz": kzz, "Kmean": kmean}
        except Exception as e:  # pragma: no cover - defensive
            print(f"Error parsing results from {path}: {e}")
            return {"Kxx": 0.0, "Kyy": 0.0, "Kzz": 0.0, "Kmean": 0.0}