from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Union

import merope
import sac_de_billes
import numpy as np


class MicrostructureBuilder:
    """Factory for Mérope polycrystals and porous microstructures.

    Notes
    -----
    Default conventions (kept compatible with the new `experiments/` scripts):

    - Phase 1: dense grain / matrix.
    - Phase 2: porous phase (grain-boundary film, spherical pores, etc.).
    - Phase 3: helper phase used for intermediate overlays (e.g. intra-granular pores)
      that is usually remapped to phase 2 in the final `Structure_3D`.

    The older scripts in `tests/exercises/` often use
    ``phase_grains = 0``, ``phase_intra = 1``, ``phase_inter = 2``,
    ``phase_boundary = 3``. The current builder implements:

    - high-level helpers (`generate_polycrystal`, `generate_spheres`,
      `generate_mixed_structure`) using the *new* convention (1 = matrix,
      2 = pores),
    - a dedicated method :meth:`generate_interconnected_structure` that
      reproduces the legacy inter+intra pipeline with ``0 = matrix`` and
      ``2 = pores``, so that the phase semantics match the original scripts
      exactly (useful when cross-validating results).
    """

    def __init__(self, L: Sequence[float], n3D: int, seed: int = 0) -> None:
        """Initialize a microstructure builder.

        Parameters
        ----------
        L :
            Domain size \([L_x, L_y, L_z]\) of the RVE in physical units.
        n3D :
            Number of voxels along each direction. The total grid will have
            \(n3D^3\) voxels.
        seed :
            Integer seed passed to `sac_de_billes` for reproducible sphere
            throwing.

        Raises
        ------
        ValueError
            If ``L`` does not contain exactly three positive components or
            if ``n3D`` is not strictly positive.
        """
        if len(L) != 3:
            raise ValueError(f"L must be a 3-component iterable, got {L!r}")

        if n3D <= 0:
            raise ValueError(f"n3D must be a positive integer, got {n3D}")

        self.L: List[float] = [float(L[0]), float(L[1]), float(L[2])]
        self.n3D: int = int(n3D)
        self.seed: int = int(seed)

        # Pre-compute grid parameters for subsequent voxelizations.
        self.grid_params = merope.vox.create_grid_parameters_N_L_3D(
            [self.n3D] * 3, self.L
        )

    # ------------------------------------------------------------------
    # Basic building blocks
    # ------------------------------------------------------------------
    def generate_polycrystal(
        self,
        grain_radius: float,
        delta: float = 0.0,
        aspect_ratio: Sequence[float] = (1.0, 1.0, 1.0),
    ) -> "merope.Structure_3D":
        """Create a Laguerre polycrystal with an optional grain-boundary layer.

        This is the core building block used by the anisotropy and delta-sweep
        experiments.

        Parameters
        ----------
        grain_radius :
            Radius parameter used in the RSA seed histogram (in the same units
            as ``L``). This controls the typical grain size.
        delta :
            Physical thickness of the grain-boundary layer. A value of ``0.0``
            disables the layer.
        aspect_ratio :
            Aspect ratio \([a_x, a_y, a_z]\) applied to the Laguerre tessellation
            to create anisotropic grains. All components must be positive.

        Returns
        -------
        merope.Structure_3D
            A structure where:

            - phase 1 contains the grain interiors (matrix),
            - phase 2 (if ``delta > 0``) contains the grain-boundary film.

        Notes
        -----
        The implementation mirrors the Laguerre construction used in
        ``tests/exercises/inco_intra_inter_polycrystal.py`` but without the
        inter-/intra-pore overlays. Only the seed generation part is shared.
        """
        if grain_radius <= 0.0:
            raise ValueError(f"grain_radius must be positive, got {grain_radius}")
        if delta < 0.0:
            raise ValueError(f"delta must be non-negative, got {delta}")
        if len(aspect_ratio) != 3:
            raise ValueError(
                f"aspect_ratio must have three components, got {aspect_ratio!r}"
            )
        if any(a <= 0.0 for a in aspect_ratio):
            raise ValueError(f"aspect_ratio components must be > 0, got {aspect_ratio}")

        sph = merope.SphereInclusions_3D()
        sph.setLength(self.L)
        sph.fromHisto(
            self.seed,
            sac_de_billes.TypeAlgo.RSA,
            0.0,
            [[float(grain_radius), 1.0]],
            [1],
        )

        poly = merope.LaguerreTess_3D(self.L, sph.getSpheres())
        poly.setAspRatio(list(aspect_ratio))

        multi = merope.MultiInclusions_3D()
        multi.setInclusions(poly)

        ids = multi.getAllIdentifiers()
        if delta > 0.0:
            # Phase 2 = grain-boundary film (porous).
            multi.addLayer(ids, 2, float(delta))

        # Phase 1 = grain interiors (matrix).
        multi.changePhase(ids, [1 for _ in ids])

        return merope.Structure_3D(multi)

    def generate_spheres(
        self,
        radii_phi_list: Sequence[Sequence[float]],
        phase_id: int = 2,
    ) -> "merope.MultiInclusions_3D":
        """Create distributed spherical inclusions with a single phase id.

        Parameters
        ----------
        radii_phi_list :
            Sequence of ``[radius, volume_fraction]`` pairs passed directly to
            ``SphereInclusions_3D.fromHisto``. Radii must be positive and the
            corresponding target porosities in \([0, 1]\).
        phase_id :
            Phase index assigned to *all* generated spheres. By convention:

            - 2 → porous phase (default, used in new ``experiments/`` scripts),
            - other values are possible but must be consistent with the
              homogenization coefficients passed to :meth:`voxellate`.

        Returns
        -------
        merope.MultiInclusions_3D
            A `MultiInclusions_3D` object containing the generated spheres.

        Raises
        ------
        ValueError
            If ``radii_phi_list`` is empty.
        """
        if not radii_phi_list:
            raise ValueError("radii_phi_list must not be empty.")

        sph = merope.SphereInclusions_3D()
        sph.setLength(self.L)
        sph.fromHisto(
            self.seed,
            sac_de_billes.TypeAlgo.BOOL,
            0.0,
            [[float(r), float(phi)] for r, phi in radii_phi_list],
            [int(phase_id)] * len(radii_phi_list),
        )

        multi = merope.MultiInclusions_3D()
        multi.setInclusions(sph)
        return multi

    # ------------------------------------------------------------------
    # Mixed microstructures (grains + pores)
    # ------------------------------------------------------------------
    def generate_mixed_structure(
        self,
        grain_radius: float,
        delta: float,
        intra_pore_list: Sequence[Sequence[float]],
    ) -> "merope.Structure_3D":
        """Build a simplified “mixed porosity” structure.

        This mirrors, in a simplified form, the logic of
        ``tests/exercises/inco_intra_inter_polycrystal.build_and_voxelize``:

        - a Laguerre polycrystal (grain seeds generated via RSA),
        - an optional grain-boundary film of thickness ``delta``,
        - intra-granular spherical pores.

        Compared to the full legacy script, **inter-granular pore channels
        are not generated here**. If you need both inter- and intra-granular
        porosity, the construction in ``tests/refactoring/merope_engine.py``
        (`MeropeEngine.build_interconnected_structure`) is the authoritative
        reference to follow.

        Parameters
        ----------
        grain_radius :
            Radius parameter used for the Laguerre seeds.
        delta :
            Physical thickness of the grain-boundary film. Set to ``0.0`` to
            disable the film.
        intra_pore_list :
            Histogram for intra-granular pores, as a sequence of
            ``[radius, volume_fraction]`` pairs.

        Returns
        -------
        merope.Structure_3D
            A structure where:

            - phase 1 = grains (matrix),
            - phase 2 = grain-boundary film (if ``delta > 0``),
            - phase 3 = raw intra-granular spheres, remapped to phase 2
              in the final structure.
        """
        if not intra_pore_list:
            raise ValueError("intra_pore_list must not be empty.")
        if grain_radius <= 0.0:
            raise ValueError(f"grain_radius must be positive, got {grain_radius}")
        if delta < 0.0:
            raise ValueError(f"delta must be non-negative, got {delta}")

        # 1. Polycrystal (grains + optional boundary film)
        sph_grains = merope.SphereInclusions_3D()
        sph_grains.setLength(self.L)
        sph_grains.fromHisto(
            self.seed,
            sac_de_billes.TypeAlgo.RSA,
            0.0,
            [[float(grain_radius), 1.0]],
            [1],
        )
        poly = merope.LaguerreTess_3D(self.L, sph_grains.getSpheres())

        m_grains = merope.MultiInclusions_3D()
        m_grains.setInclusions(poly)
        grain_ids = m_grains.getAllIdentifiers()

        # Phase 1 = grain interiors.
        m_grains.changePhase(grain_ids, [1 for _ in grain_ids])
        if delta > 0.0:
            # Phase 2 = grain-boundary film.
            m_grains.addLayer(grain_ids, 2, float(delta))

        # 2. Intra-granular spherical pores (phase 3, later remapped to 2)
        m_spheres = merope.MultiInclusions_3D()
        sph_obj = merope.SphereInclusions_3D()
        sph_obj.setLength(self.L)
        sph_obj.fromHisto(
            self.seed,
            sac_de_billes.TypeAlgo.BOOL,
            0.0,
            [[float(r), float(phi)] for r, phi in intra_pore_list],
            [3],
        )
        m_spheres.setInclusions(sph_obj)

        # 3. Combine: convert both to Structure_3D and remap phase 3 → 2
        #    in the overlaid spheres. This yields a single porous phase (2).
        s_grains = merope.Structure_3D(m_grains)
        s_spheres = merope.Structure_3D(m_spheres)

        return merope.Structure_3D(s_grains, s_spheres, {3: 2, 0: -1})

    def generate_interconnected_structure(
        self,
        inter_radius: float,
        inter_phi: float,
        intra_radius: float,
        intra_phi: float,
        grain_radius: float,
        grain_phi: float,
        delta: float,
    ) -> "merope.Structure_3D":
        """Generate interconnected structure matching MOX_structure_generator logic."""
        import sac_de_billes
        import merope

        np.random.seed(self.seed)
        L = self.L

        phase_matrix = 0
        phase_intra = 1  # Red spheres
        phase_inter = 2  # Blue boundary network

        # -----------------------------------------------------------------
        # 1) Generate identical grains for Base and Mask
        # -----------------------------------------------------------------
        # Base laguerre: Core = 0 (white matrix), Boundary = 2 (blue inter-pores)
        sph_lag = merope.SphereInclusions_3D()
        sph_lag.setLength(L)
        sph_lag.fromHisto(
            int(self.seed + 2), sac_de_billes.TypeAlgo.RSA, 0.0, [[float(grain_radius), float(grain_phi)]], [0]
        )
        spheres = sph_lag.getSpheres()

        tess_base = merope.LaguerreTess_3D(L, spheres)
        grains_base = merope.MultiInclusions_3D()
        grains_base.setInclusions(tess_base)
        ids = grains_base.getAllIdentifiers()
        grains_base.addLayer(ids, phase_inter, float(delta))  # outer layer = 2
        grains_base.changePhase(ids, [phase_matrix for _ in ids])  # core = 0
        struct_base = merope.Structure_3D(grains_base)

        # Mask laguerre: Core = 0, Boundary = 1 (Activating mask)
        tess_mask = merope.LaguerreTess_3D(L, spheres)
        grains_mask = merope.MultiInclusions_3D()
        grains_mask.setInclusions(tess_mask)
        ids_m = grains_mask.getAllIdentifiers()
        grains_mask.addLayer(ids_m, 1, float(delta))  # outer layer = 1
        grains_mask.changePhase(ids_m, [phase_matrix for _ in ids_m])  # core = 0
        struct_mask = merope.Structure_3D(grains_mask)

        # -----------------------------------------------------------------
        # 2) Generate Intra-granular pores (Red spheres, phase 1)
        # -----------------------------------------------------------------
        sph_intra = merope.SphereInclusions_3D()
        sph_intra.setLength(L)
        if intra_phi > 0.0:
            sph_intra.fromHisto(
                int(self.seed + 1),
                sac_de_billes.TypeAlgo.BOOL,
                0.0,
                [[float(intra_radius), float(intra_phi)]],
                [phase_intra],
            )
        m_intra = merope.MultiInclusions_3D()
        m_intra.setInclusions(sph_intra)
        struct_intra = merope.Structure_3D(m_intra)

        # -----------------------------------------------------------------
        # 3) Restrict intra-pores to grain interiors
        # -----------------------------------------------------------------
        # Where struct_mask == 1 (the boundary network), map intra (1) -> matrix (0)
        # This deletes all intra-pores that touch or overlap the inter-pore boundary,
        # ensuring they stay strictly inside the grain cores.
        struct_intra_restricted = merope.Structure_3D(
            struct_intra, struct_mask, {phase_intra: phase_matrix}
        )

        # -----------------------------------------------------------------
        # 4) Composite final image
        # -----------------------------------------------------------------
        # Now struct_intra_restricted has Phase 1 strictly inside the cores.
        # We overlay this onto the base (which has 0 and 2).
        # Where intra is 1, we replace matrix (0) with intra (1).
        final_struct = merope.Structure_3D(
            struct_base, struct_intra_restricted, {phase_matrix: phase_intra}
        )

        return final_struct

    def generate_boundary_confined_structure(
        self,
        grain_radius: float,
        delta: float,
        pore_radius: float,
        pore_phi: float,
    ) -> "merope.Structure_3D":
        """Generate structure with spherical pores confined to grain boundary layer.

        This implements the δ-parameter model from the slide:
        - Small δ → pores squeezed into thin interconnected cracks → low K_eff
        - Large δ → pores expand into isolated spheres → high K_eff

        Parameters
        ----------
        grain_radius :
            Radius parameter for Laguerre grain seeds.
        delta :
            Thickness of the grain-boundary layer where pores are confined.
        pore_radius :
            Radius of spherical pores to be clipped.
        pore_phi :
            Target volume fraction of pores.

        Returns
        -------
        merope.Structure_3D
            Structure with:
            - phase 0 = grain interior (matrix)
            - phase 2 = pores confined to grain boundary layer
        """
        if grain_radius <= 0.0:
            raise ValueError(f"grain_radius must be positive, got {grain_radius}")
        if delta <= 0.0:
            raise ValueError(f"delta must be positive, got {delta}")
        if pore_radius <= 0.0:
            raise ValueError(f"pore_radius must be positive, got {pore_radius}")
        if not 0.0 <= pore_phi <= 1.0:
            raise ValueError(f"pore_phi must be in [0,1], got {pore_phi}")

        L = self.L
        phase_matrix = 0
        phase_pore = 2

        # -----------------------------------------------------------------
        # 1) Generate grain structure with boundary layer mask
        # -----------------------------------------------------------------
        # Base structure: grain cores (phase 0) only, no boundary layer yet
        sph_lag = merope.SphereInclusions_3D()
        sph_lag.setLength(L)
        sph_lag.fromHisto(
            int(self.seed + 2),
            sac_de_billes.TypeAlgo.RSA,
            0.0,
            [[float(grain_radius), 1.0]],
            [0],
        )
        spheres = sph_lag.getSpheres()

        tess_base = merope.LaguerreTess_3D(L, spheres)
        grains_base = merope.MultiInclusions_3D()
        grains_base.setInclusions(tess_base)
        ids = grains_base.getAllIdentifiers()
        grains_base.changePhase(ids, [phase_matrix for _ in ids])
        struct_base = merope.Structure_3D(grains_base)

        # -----------------------------------------------------------------
        # 2) Generate spherical pores
        # -----------------------------------------------------------------
        sph_pores = merope.SphereInclusions_3D()
        sph_pores.setLength(L)
        if pore_phi > 0.0:
            sph_pores.fromHisto(
                int(self.seed + 1),
                sac_de_billes.TypeAlgo.BOOL,
                0.0,
                [[float(pore_radius), float(pore_phi)]],
                [phase_pore],
            )
        m_pores = merope.MultiInclusions_3D()
        m_pores.setInclusions(sph_pores)
        struct_pores = merope.Structure_3D(m_pores)

        # -----------------------------------------------------------------
        # 3) Confine pores to boundary layer
        # -----------------------------------------------------------------
        # Merope overlay semantics: Structure_3D(A, B, {x: y})
        #   → where B != 0, map A's phase x to y
        #
        # Create deletion mask: grain cores = phase 1, boundary = phase 0
        # (so pores are DELETED in cores, KEPT in boundary)
        tess_del_mask = merope.LaguerreTess_3D(L, spheres)
        grains_del = merope.MultiInclusions_3D()
        grains_del.setInclusions(tess_del_mask)
        ids_del = grains_del.getAllIdentifiers()
        # First add boundary layer (phase 0), THEN change cores (phase 1)
        # This way boundary stays 0, cores become 1
        grains_del.addLayer(ids_del, 0, float(delta))  # boundary = 0
        grains_del.changePhase(ids_del, [1 for _ in ids_del])  # cores = 1
        struct_del_mask = merope.Structure_3D(grains_del)

        # Where del_mask == 1 (grain cores), delete pores (map phase 2 → 0)
        struct_pores_confined = merope.Structure_3D(
            struct_pores, struct_del_mask, {phase_pore: phase_matrix}
        )

        # -----------------------------------------------------------------
        # 4) Return confined structure
        # -----------------------------------------------------------------
        # struct_pores_confined already has:
        #   - phase 0 where there's no pores (matrix)
        #   - phase 2 where pores survived (confined to boundary layer)
        # No need for additional overlay!
        return struct_pores_confined

    # ------------------------------------------------------------------
    # Voxellation + homogenization
    # ------------------------------------------------------------------
    def voxellate(
        self,
        structure: "merope.Structure_3D",
        K_values: Sequence[float],
        vtk_path: Union[str, Path] = "structure.vtk",
        coeffs_path: Union[str, Path] = "Coeffs.txt",
    ) -> Dict[int, float]:
        """Voxelize a structure and export a VTK file for Amitex.

        Parameters
        ----------
        structure :
            Mérope 3D structure to voxelize.
        K_values :
            List of thermal conductivities indexed by phase. The conventions
            must match the phases present in ``structure``. For example, in
            the mixed-porosity case used by the new experiments:

            - index 1 → matrix conductivity,
            - index 2 → pore conductivity.
        vtk_path :
            Target path for the segmented VTK file. Can be a string or
            :class:`pathlib.Path`. Parent directories are created if missing.
        coeffs_path :
            Path of the text file where homogenized coefficients are written
            by Mérope. Passed unchanged to the VTK printer.

        Returns
        -------
        dict[int, float]
            Mapping ``phase_id → volume_fraction`` as estimated by
            ``GridAnalyzer_3D.compute_percentages``. If the analyzer fails,
            a dictionary with phases 1 and 2 set to zero is returned.
        """
        if structure is None:
            raise ValueError("structure must be a valid merope.Structure_3D instance.")
        if not K_values:
            raise ValueError("K_values must contain at least one conductivity value.")

        vtk_path = Path(vtk_path)
        coeffs_path = Path(coeffs_path)
        vtk_path.parent.mkdir(parents=True, exist_ok=True)
        coeffs_path.parent.mkdir(parents=True, exist_ok=True)

        grid_repr = merope.vox.GridRepresentation_3D(
            structure, self.grid_params, merope.vox.VoxelRule.Average
        )

        # SAFE ANALYZER LOGIC
        fractions: Dict[int, float] = {1: 0.0, 2: 0.0}
        try:
            analyzer = merope.vox.GridAnalyzer_3D()
            fractions = analyzer.compute_percentages(grid_repr)
            print(f"Voxellation complete. Detected Phases: {fractions}")
        except Exception as e:  # pragma: no cover - defensive logging only
            print(f"Analyzer skipped due to: {e}")

        # Apply thermal coefficients (bakes the phases into the grid).
        grid_repr.apply_homogRule(merope.HomogenizationRule.Voigt, list(K_values))

        # Export for Amitex
        printer = merope.vox.vtk_printer_3D()
        printer.printVTK_segmented(
            grid_repr, str(vtk_path), str(coeffs_path), nameValue="MaterialId"
        )

        return fractions