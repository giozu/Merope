from __future__ import annotations

import os
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Union, Any
import contextlib

import merope
import sac_de_billes
import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk

@contextlib.contextmanager
def temp_dir():
    """Context manager for a temporary directory."""
    name = tempfile.mkdtemp()
    try:
        yield name
    finally:
        shutil.rmtree(name)

def load_vtk(path: Union[str, Path]) -> np.ndarray:
    """Load a VTK file and return its cell data as a NumPy array."""
    reader = vtk.vtkDataSetReader()
    reader.SetFileName(str(path))
    reader.Update()
    array = reader.GetOutput().GetCellData().GetArray("MaterialId")
    if array is None:
        raise ValueError(f"No 'MaterialId' array found in {path}")
    return vtk_to_numpy(array).astype(np.int32)

class ConfinementWrapper:
    """Helper to pass grains and unconfined pores to voxellate for manual clipping."""
    def __init__(self, struct_base, struct_pores, delta):
        self.struct_base = struct_base
        self.struct_pores = struct_pores
        self.delta = delta


class MicrostructureBuilder:
    """Factory for Mérope polycrystals and porous microstructures."""

    def __init__(self, L: Sequence[float], n3D: int, seed: int = 0) -> None:
        if len(L) != 3:
            raise ValueError(f"L must be a 3-component iterable, got {L!r}")
        if n3D <= 0:
            raise ValueError(f"n3D must be a positive integer, got {n3D}")

        self.L: List[float] = [float(L[0]), float(L[1]), float(L[2])]
        self.n3D: int = int(n3D)
        self.seed: int = int(seed)
        self.grid_params = merope.vox.create_grid_parameters_N_L_3D([self.n3D] * 3, self.L)

    def generate_polycrystal(
        self,
        grain_radius: float,
        delta: float = 0.0,
        aspect_ratio: Sequence[float] = (1.0, 1.0, 1.0),
    ) -> "merope.Structure_3D":
        sph = merope.SphereInclusions_3D()
        sph.setLength(self.L)
        sph.fromHisto(self.seed, sac_de_billes.TypeAlgo.RSA, 0.0, [[float(grain_radius), 1.0]], [1])
        poly = merope.LaguerreTess_3D(self.L, sph.getSpheres())
        poly.setAspRatio(list(aspect_ratio))
        multi = merope.MultiInclusions_3D()
        multi.setInclusions(poly)
        ids = multi.getAllIdentifiers()
        multi.changePhase(ids, [1 for _ in ids])
        if delta > 0.0:
            multi.addLayer(ids, 2, float(delta))
        return merope.Structure_3D(multi)

    def generate_spheres(
        self,
        radii_phi_list: Sequence[Sequence[float]],
        phase_id: int = 2,
    ) -> "merope.MultiInclusions_3D":
        sph = merope.SphereInclusions_3D()
        sph.setLength(self.L)
        sph.fromHisto(self.seed, sac_de_billes.TypeAlgo.BOOL, 0.0,
                      [[float(r), float(phi)] for r, phi in radii_phi_list],
                      [int(phase_id)] * len(radii_phi_list))
        multi = merope.MultiInclusions_3D()
        multi.setInclusions(sph)
        return multi

    def generate_interconnected_structure(
        self,
        inter_radius: float, inter_phi: float,
        intra_radius: float, intra_phi: float,
        grain_radius: float, grain_phi: float,
        delta: float,
    ) -> "merope.Structure_3D":
        sph_lag = merope.SphereInclusions_3D()
        sph_lag.setLength(self.L)
        sph_lag.fromHisto(int(self.seed + 2), sac_de_billes.TypeAlgo.RSA, 0.0, [[float(grain_radius), float(grain_phi)]], [0])
        spheres = sph_lag.getSpheres()

        tess_base = merope.LaguerreTess_3D(self.L, spheres)
        m_base = merope.MultiInclusions_3D()
        m_base.setInclusions(tess_base)
        ids = m_base.getAllIdentifiers()
        m_base.changePhase(ids, [0 for _ in ids])
        m_base.addLayer(ids, 2, float(delta))
        struct_base = merope.Structure_3D(m_base)

        tess_mask = merope.LaguerreTess_3D(self.L, spheres)
        m_mask = merope.MultiInclusions_3D()
        m_mask.setInclusions(tess_mask)
        ids_m = m_mask.getAllIdentifiers()
        m_mask.changePhase(ids_m, [0 for _ in ids_m])
        m_mask.addLayer(ids_m, 1, float(delta))
        struct_mask = merope.Structure_3D(m_mask)

        sph_intra = merope.SphereInclusions_3D()
        sph_intra.setLength(self.L)
        if intra_phi > 0.0:
            sph_intra.fromHisto(int(self.seed + 1), sac_de_billes.TypeAlgo.BOOL, 0.0, [[float(intra_radius), float(intra_phi)]], [1])
        m_i = merope.MultiInclusions_3D()
        m_i.setInclusions(sph_intra)
        s_intra = merope.Structure_3D(m_i)

        s_intra_restricted = merope.Structure_3D(s_intra, struct_mask, {1: 0})
        return merope.Structure_3D(struct_base, s_intra_restricted, {0: 1})

    def generate_boundary_confined_structure(
        self,
        grain_radius: float,
        delta: float,
        pore_radius: float,
        pore_phi: float,
        confinement_mode: str = "standard"
    ) -> Union["merope.Structure_3D", ConfinementWrapper]:
        """Generate structure with spherical pores confined to grain boundary layer."""
        L = self.L
        # Standardization: 1=Matrix, 2=Mask, 3=Pores
        sph_lag = merope.SphereInclusions_3D()
        sph_lag.setLength(L)
        sph_lag.fromHisto(int(self.seed + 2), sac_de_billes.TypeAlgo.RSA, 0.0, [[float(grain_radius), 1.0]], [1])
        tess = merope.LaguerreTess_3D(L, sph_lag.getSpheres())
        m_base = merope.MultiInclusions_3D()
        m_base.setInclusions(tess)
        ids = m_base.getAllIdentifiers()
        m_base.changePhase(ids, [1 for _ in ids])
        if delta > 0.0:
            m_base.addLayer(ids, 2, float(delta))
        s_base = merope.Structure_3D(m_base)

        sph_pores = merope.SphereInclusions_3D()
        sph_pores.setLength(L)
        if pore_phi > 0.0:
            sph_pores.fromHisto(int(self.seed + 1), sac_de_billes.TypeAlgo.BOOL, 0.0, [[float(pore_radius), float(pore_phi)]], [3])
        m_pores = merope.MultiInclusions_3D()
        m_pores.setInclusions(sph_pores)
        s_pores = merope.Structure_3D(m_pores)

        if confinement_mode == "standard":
            return merope.Structure_3D(s_base, s_pores, {})
        return ConfinementWrapper(s_base, s_pores, delta)

    def voxellate(
        self,
        structure: Union["merope.Structure_3D", ConfinementWrapper],
        K_values: Sequence[float],
        vtk_path: Union[str, Path] = "structure.vtk",
        coeffs_path: Union[str, Path] = "Coeffs.txt",
    ) -> Dict[int, float]:
        """Convert structure to VTK using robust NumPy extraction."""
        vtk_path = Path(vtk_path); coeffs_path = Path(coeffs_path)
        vtk_path.parent.mkdir(parents=True, exist_ok=True)
        coeffs_path.parent.mkdir(parents=True, exist_ok=True)

        if isinstance(structure, ConfinementWrapper):
            s_base = structure.struct_base; s_pore = structure.struct_pores; delta = structure.delta
        else:
            s_base = structure; s_pore = structure; delta = 999.0

        def extract(s_obj):
            grid = merope.vox.GridRepresentation_3D(s_obj, self.grid_params, merope.vox.VoxelRule.Average)
            grid.apply_homogRule(merope.HomogenizationRule.Voigt, [float(i) for i in range(20)])
            field = merope.vox.NumpyConverter_3D().compute_RealField(grid)
            return np.round(field).astype(np.int32).reshape((self.n3D, self.n3D, self.n3D))

        arr_b = extract(s_base); arr_p = extract(s_pore)
        
        if delta > 10.0:
            final = np.where(arr_p == 3, 2, 1)
        else:
            final = np.where(np.logical_and(arr_b == 2, arr_p == 3), 2, 1)

        print(f"  DEBUG: final unique: {np.unique(final)}")
        return self._write_manual_vtk(vtk_path, final, coeffs_path)

    def _write_manual_vtk(self, path: Path, arr: np.ndarray, coeffs_path: Path) -> Dict[int, float]:
        nx = ny = nz = self.n3D
        dx, dy, dz = self.L[0]/nx, self.L[1]/ny, self.L[2]/nz
        
        # Use VTK library to write a proper BINARY file
        vtk_arr = numpy_to_vtk(arr.flatten(), deep=True, array_type=vtk.VTK_INT)
        vtk_arr.SetName("MaterialId")
        
        grid = vtk.vtkStructuredPoints()
        grid.SetDimensions(nx+1, ny+1, nz+1)
        grid.SetSpacing(dx, dy, dz)
        grid.SetOrigin(0.0, 0.0, 0.0)
        grid.GetCellData().SetScalars(vtk_arr)
        
        writer = vtk.vtkDataSetWriter()
        writer.SetFileName(str(path))
        writer.SetFileTypeToBinary()
        writer.SetInputData(grid)
        writer.Write()
        
        fracs = {}
        uids, counts = np.unique(arr, return_counts=True)
        for u, c in zip(uids, counts): fracs[int(u)] = float(c) / arr.size
        
        with open(coeffs_path, "w") as f:
            f.write("1.0\n1.0\n0.001\n")
        return fracs