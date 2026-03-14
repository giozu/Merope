from __future__ import annotations

import os
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Union, Any
import contextlib

import merope
import sac_de_billes
import vtk
from vtk.util.numpy_support import vtk_to_numpy
from typing import Union, Sequence, Dict
from pathlib import Path

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
        
    def voxellate(
        self,
        arr: np.ndarray,
        vtk_path: Union[str, Path] = "structure.vtk",
    ) -> Dict[int, float]:
        vtk_path = Path(vtk_path)
        vtk_path.parent.mkdir(parents=True, exist_ok=True)
        
        nx = ny = nz = self.n3D + 1
        spacing = [float(self.L[i])/self.n3D for i in range(3)]
        
        header = (
            f"# vtk DataFile Version 4.0\r\n"
            f"Amitex Sanitized\r\n"
            f"BINARY\r\n"
            f"DATASET STRUCTURED_POINTS\r\n"
            f"DIMENSIONS {nx} {ny} {nz}\r\n"
            f"ORIGIN  0. 0. 0.\r\n"
            f"SPACING {spacing[0]} {spacing[1]} {spacing[2]}\r\n"
            f"CELL_DATA {arr.size}\r\n"
            f"SCALARS MaterialId unsigned_short\r\n"
            f"LOOKUP_TABLE default\r\n"
        ).encode()
        
        with open(vtk_path, "wb") as f:
            f.write(header)
            # Use Big-Endian uint16
            f.write(arr.astype('>u2').tobytes())
        uids, counts = np.unique(arr, return_counts=True)
        return {int(u): float(c)/arr.size for u, c in zip(uids, counts)}