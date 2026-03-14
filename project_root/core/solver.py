import os
import numpy as np
import vtk
import subprocess
from pathlib import Path
from typing import Dict, Union
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
import interface_amitex_fftp.amitex_xml_writer as ami_xml

class ThermalSolver:
    def __init__(self, n_cpus: int = 4) -> None:
        self.n_cpus: int = int(n_cpus)
        self.amitex_bin = "/usr/lib/amitex_fftp-v8.17.14/libAmitex/bin/amitex_fftp"

    def solve(
        self,
        vtk_file: Union[str, Path] = "structure.vtk",
        results_file: Union[str, Path] = "thermalCoeff_amitex.txt",
    ) -> Dict[str, float]:
        vtk_path = Path(vtk_file).resolve()
        res_path = Path(results_file)
        if not vtk_path.is_file(): return {"Kmean": 0.0}

        work_dir = vtk_path.parent
        if not res_path.is_absolute(): res_path = work_dir / res_path
        if res_path.exists(): res_path.unlink()

        prev_dir = os.getcwd()
        try:
            os.chdir(work_dir)
            
            # 1. READ AND SANITIZE VTK
            reader = vtk.vtkDataSetReader()
            reader.SetFileName(str(vtk_path))
            reader.Update()
            grid = reader.GetOutput()
            
            scalars = grid.GetCellData().GetScalars("MaterialId")
            if not scalars: scalars = grid.GetCellData().GetScalars()
            
            # Collapse IDs: Everything > 3 goes to 1. 0 goes to 1.
            arr = vtk_to_numpy(scalars).copy().astype(np.uint16)
            arr[arr > 3] = 1
            arr[arr == 0] = 1
            
            # Detect what we actually have
            active_phases = sorted([int(p) for p in np.unique(arr)])
            print(f"  [Solver] Sanitized VTK - Phases detected: {active_phases}")

            # 2. WRITE BINARY VTK (Using standard VTK library)
            vtk_arr = numpy_to_vtk(arr, deep=True)
            vtk_arr.SetName("MaterialId")
            
            new_grid = vtk.vtkStructuredPoints()
            new_grid.SetDimensions(grid.GetDimensions())
            new_grid.SetSpacing(grid.GetSpacing())
            new_grid.SetOrigin(grid.GetOrigin())
            new_grid.GetCellData().SetScalars(vtk_arr)
            
            writer = vtk.vtkDataSetWriter()
            writer.SetFileName(str(vtk_path))
            writer.SetInputData(new_grid)
            writer.SetFileTypeToBinary()
            writer.Write()
            
            # 3. DYNAMIC XML for detected phases
            self._write_xml(work_dir, active_phases)
            
            # 4. SIMULATE
            for i in range(1, 4):
                load_vals = [0., 0., 0.]; load_vals[i-1] = 1.0
                load_conf = ami_xml.Loading_diffusion(direction_values=load_vals)
                load_conf.write_into(f"load_{i}.xml")
                
                cmd = ["mpirun", "-np", str(self.n_cpus), self.amitex_bin, 
                       "-nz", vtk_path.name, "-m", "mat.xml", "-a", "algo.xml", 
                       "-c", f"load_{i}.xml", "-s", f"res_{i}"]
                subprocess.run(cmd, cwd=work_dir)

            # 5. PARSE
            ks = []
            for i in range(1, 4):
                sf = work_dir / f"res_{i}.std"
                if sf.exists():
                    try:
                        d = np.loadtxt(sf)
                        ks.append(d[0] if d.ndim <= 1 else d[0,0])
                    except: pass
            
            k_eff = float(np.mean(ks)) if ks else 0.0
            with open(res_path, "w") as f:
                f.write(f"{k_eff} 0 0\n0 {k_eff} 0\n0 0 {k_eff}\n")
            
            return {"Kmean": k_eff}
            
        except Exception as e:
            print(f"Solver error: {e}")
        finally:
            os.chdir(prev_dir)
        return {"Kmean": 0.0}

    def _write_xml(self, work_dir: Path, phases: list):
        K_val = {1: 1.0, 2: 1e-3, 3: 1.0}
        params = []
        with open(work_dir / "Z.txt", "w") as f: f.write("0.0\n")
        for p in phases:
            k = K_val.get(p, 1.0)
            k_file = f"K_{p}.txt"
            with open(work_dir / k_file, "w") as f: f.write(f"{k}\n")
            px = ami_xml.Parameters_Fourier_iso(coeff_fileName=k_file, flux_fileNames=(k_file, "Z.txt", "Z.txt"), numM=p)
            params.append(px)
        for i, px in enumerate(params): px.numM = phases[i]
        mat_conf = ami_xml.Material(coeff_K=1.0, list_of_param_single_mat=params)
        mat_conf.write_into("mat.xml")
        ami_xml.Algo_diffusion.write_into("algo.xml", convergence_criterion_value=1e-4)