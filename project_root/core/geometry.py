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
        """Build a full inter+intra porous polycrystal (legacy-equivalent).

        This method is the object-oriented counterpart of the geometry
        pipeline implemented in
        ``tests/exercises/inco_intra_inter_polycrystal.build_and_voxelize``
        and, in refactored form, in
        ``tests/refactoring/merope_engine.MeropeEngine.build_interconnected_structure``.

        Geometry
        --------
        1. Inter-granular pores (phase 2) as an independent `MultiInclusions_3D`.
        2. Intra-granular pores (phase 1) as another independent
           `MultiInclusions_3D`.
        3. Laguerre grains built from RSA seeds, with a boundary layer
           (phase 3) of physical thickness ``delta``.
        4. Combination:

           - First, inter-pores + grains are merged, remapping both
             inter-pores (2) and boundary (3) to grains (0).
           - Then intra-pores are overlaid, with a remap that converts the
             grain phase (0) into the pore phase (2) wherever intra-pores
             are active.

        Phase conventions
        -----------------
        To stay fully consistent with the legacy scripts, this method
        adopts the *old* phase convention:

        - 0 → matrix (solid grains, including boundary film),
        - 1 → temporary intra-pore tag,
        - 2 → final pore phase (used to compute porosity and to assign `K_gas`),
        - 3 → temporary grain-boundary layer before remapping.

        You should therefore pass thermal conductivities as::

            K = [K_matrix, K_matrix, K_gas]

        when voxelizing the returned structure.

        Parameters
        ----------
        inter_radius :
            Sphere radius for inter-granular pores.
        inter_phi :
            Target volume fraction of inter-granular porosity (0 ≤ φ ≤ 1).
        intra_radius :
            Sphere radius for intra-granular pores.
        intra_phi :
            Target volume fraction of intra-granular porosity (0 ≤ φ ≤ 1).
        grain_radius :
            Radius parameter for the Laguerre seeds.
        grain_phi :
            Target “packing” volume fraction for the Laguerre seeds (typically 1.0
            to fill the RVE with grains).
        delta :
            Physical thickness of the grain-boundary layer.

        Returns
        -------
        merope.Structure_3D
            A structure where pores (both inter- and intra-granular) are in
            phase 2, and the solid matrix is in phase 0, matching the
            original Mérope/Amitex scripts.

        Raises
        ------
        ValueError
            If any radius is non-positive, any porosity is outside [0, 1],
            or ``delta`` is negative.
        """
        if inter_radius <= 0.0:
            raise ValueError(f"inter_radius must be positive, got {inter_radius}")
        if intra_radius <= 0.0:
            raise ValueError(f"intra_radius must be positive, got {intra_radius}")
        if grain_radius <= 0.0:
            raise ValueError(f"grain_radius must be positive, got {grain_radius}")
        if not (0.0 <= inter_phi <= 1.0):
            raise ValueError(f"inter_phi must be in [0, 1], got {inter_phi}")
        if not (0.0 <= intra_phi <= 1.0):
            raise ValueError(f"intra_phi must be in [0, 1], got {intra_phi}")
        if not (0.0 < grain_phi <= 1.0):
            raise ValueError(f"grain_phi must be in (0, 1], got {grain_phi}")
        if delta < 0.0:
            raise ValueError(f"delta must be non-negative, got {delta}")

        L = self.L

        # Legacy-compatible phase indices
        phase_grains = 0
        phase_intra = 1
        phase_inter = 2
        phase_boundary = 3

        # 1) Inter-granular pores ----------------------------------------
        sph_inter = merope.SphereInclusions_3D()
        sph_inter.setLength(L)
        if inter_phi > 0.0:
            sph_inter.fromHisto(
                int(self.seed),
                sac_de_billes.TypeAlgo.BOOL,
                0.0,
                [[float(inter_radius), float(inter_phi)]],
                [phase_inter],
            )
        inter_pores = merope.MultiInclusions_3D()
        inter_pores.setInclusions(sph_inter)

        # 2) Intra-granular pores ----------------------------------------
        sph_intra = merope.SphereInclusions_3D()
        sph_intra.setLength(L)
        if intra_phi > 0.0:
            sph_intra.fromHisto(
                int(self.seed),
                sac_de_billes.TypeAlgo.BOOL,
                0.0,
                [[float(intra_radius), float(intra_phi)]],
                [phase_intra],
            )
        intra_pores = merope.MultiInclusions_3D()
        intra_pores.setInclusions(sph_intra)

        # 3) Laguerre grains + boundary layer ----------------------------
        sph_lag = merope.SphereInclusions_3D()
        sph_lag.setLength(L)
        sph_lag.fromHisto(
            int(self.seed),
            sac_de_billes.TypeAlgo.RSA,
            0.0,
            [[float(grain_radius), float(grain_phi)]],
            [1],  # temporary IDs for seeds
        )

        tess = merope.LaguerreTess_3D(L, sph_lag.getSpheres())
        grains = merope.MultiInclusions_3D()
        grains.setInclusions(tess)

        ids = grains.getAllIdentifiers()
        grains.addLayer(ids, phase_boundary, float(delta))
        grains.changePhase(ids, [1 for _ in ids])  # temporary, see mapping below

        # 4) Merge inter-pores + grains ----------------------------------
        map_inter_boundary_to_grains = {
            phase_inter: phase_grains,
            phase_boundary: phase_grains,
        }
        struct_inter_on_grains = merope.Structure_3D(
            inter_pores,
            grains,
            map_inter_boundary_to_grains,
        )

        # 5) Overlay intra-pores, force grains (0) → pores (2) where intra act
        final_overlay_remap = {phase_grains: phase_inter}
        final_struct = merope.Structure_3D(
            struct_inter_on_grains,
            intra_pores,
            final_overlay_remap,
        )

        return final_struct

    def generate_delta_structure(
        self,
        pore_radius: float,
        pore_phi: float,
        grain_radius: float,
        grain_phi: float,
        delta: float,
    ) -> "merope.Structure_3D":
        """Build a polycrystal with grain-boundary clipped pores (single overlay).

        This matches the CORRECT pattern from iter_delta_IGB_calc.py and run_keff_vs_delta.py:
        - Create pores (phase 2) as spheres
        - Create Laguerre grains with boundary layer (phase 3)
        - SINGLE overlay: {2:0, 3:0} → pores and boundaries both map to grains (0)

        This is simpler and more physically correct than the double overlay pattern.

        Parameters
        ----------
        pore_radius : float
            Radius of spherical pores
        pore_phi : float
            Target volume fraction of pores (adjusted iteratively in caller)
        grain_radius : float
            Laguerre grain size
        grain_phi : float
            Should be 1.0 to fill RVE
        delta : float
            Grain boundary layer thickness

        Returns
        -------
        merope.Structure_3D
            Structure with phase 0=solid, phase 2=pores
        """
        import merope
        import sac_de_billes

        L = self.L
        seed = int(self.seed)

        # Phase IDs (same as iter_delta_IGB_calc.py)
        incl_phase = 2      # Pores
        delta_phase = 3     # Grain boundary layer (temporary)
        grains_phase = 0    # Grains (final solid phase)

        # 1. Create spherical pore inclusions (phase 2)
        sphIncl_pores = merope.SphereInclusions_3D()
        sphIncl_pores.setLength(L)
        sphIncl_pores.fromHisto(
            seed,
            sac_de_billes.TypeAlgo.BOOL,
            0.0,
            [[float(pore_radius), float(pore_phi)]],
            [incl_phase]
        )
        multiInclusions_pores = merope.MultiInclusions_3D()
        multiInclusions_pores.setInclusions(sphIncl_pores)

        # 2. Create Laguerre tessellation for grains
        sphIncl_grains = merope.SphereInclusions_3D()
        sphIncl_grains.setLength(L)
        sphIncl_grains.fromHisto(
            seed,
            sac_de_billes.TypeAlgo.RSA,
            0.0,
            [[float(grain_radius), float(grain_phi)]],
            [1]  # Temporary phase
        )
        polyCrystal = merope.LaguerreTess_3D(L, sphIncl_grains.getSpheres())

        multiInclusions_grains = merope.MultiInclusions_3D()
        multiInclusions_grains.setInclusions(polyCrystal)

        # Add grain boundary layer (phase 3) and set cores to phase 1
        ids = multiInclusions_grains.getAllIdentifiers()
        multiInclusions_grains.addLayer(ids, delta_phase, float(delta))
        multiInclusions_grains.changePhase(ids, [1 for _ in ids])

        # 3. SINGLE OVERLAY: pores on grains
        dictionnaire = {incl_phase: grains_phase, delta_phase: grains_phase}
        structure = merope.Structure_3D(
            multiInclusions_pores,
            multiInclusions_grains,
            dictionnaire
        )

        return structure

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