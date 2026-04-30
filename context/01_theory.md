# 01 — Theory and analytical models

This file collects the physics and the analytical scaffolding behind the paper, in roughly the order it appears in `methods.tex` (which doubles as the paper's theory section). It also records the equations that the code actually implements, with pointers.

## 1. Thermal transport in ceramic nuclear fuels

### 1.1 Heat conduction modes

Total thermal conductivity in oxide fuels splits into:
$$ \kappa = \kappa_{\text{cond}} + \kappa_{\text{rad}} + \kappa_{\text{conv}} $$

- $\kappa_{\text{cond}}$ — phonon‑mediated, **dominant**.
- $\kappa_{\text{rad}}$ — radiative, only matters above ~1800–2300 K.
- $\kappa_{\text{conv}}$ — convective, **negligible** in solids and for pores < 5 mm (Smith 2013). Always dropped.

### 1.2 Phonon transport (kinetic-theory analogue)

$$ \kappa_{\text{cond}} = \frac{1}{3} \int_0^{\omega_D} c(\omega)\, v(\omega)\, \ell(\omega)\, d\omega $$

with $c$ spectral heat capacity, $v$ phonon group velocity, $\ell$ mean free path, $\omega_D$ Debye frequency. The mean free path $\ell$ is shortened by:

- **phonon–phonon** Umklapp scattering (rises with T)
- **phonon–defect** scattering (vacancies, fission products, irradiation damage)
- **phonon–boundary** scattering (grain boundaries, pores) — the lever this work pulls on

### 1.3 Practical fits

- $\kappa(T) = A/T + B$ (Ronchi-style)
- **Magni 2021** for fresh MOX (used in fuel performance codes; cited but not implemented in this codebase):
$$ \kappa_0(T,x,[\text{Pu}],p) = \frac{1}{A_0 + A_x x + A_{\text{Pu}}[\text{Pu}] + (B_0 + B_{\text{Pu}}[\text{Pu}])T + (D/T^2)e^{-E/T}} (1-p)^n $$
- **Magni irradiation decay**:
$$ \kappa_{\text{irr}} = \kappa_\infty + (\kappa_0 - \kappa_\infty) e^{-B/\varphi} $$
with $\kappa_\infty \approx 2.0$ W/m·K and $\varphi = 75$ GWd/tHM.

### 1.4 Knudsen suppression in narrow pores

When the gas mean free path is comparable to the pore size, gas-phase conduction is killed. Relevant in narrow open porosity (Nichenko 2014). Effectively built into the choice of $\kappa_g = 10^{-3}$ in this work — gas conduction is treated as essentially zero.

## 2. Porosity classification

Two morphology classes are studied:

| Class | Where pores live | Typical shape | Connectivity |
|---|---|---|---|
| **Closed / distributed** | Intragranular | Spherical | Isolated |
| **Open / interconnected** | Along grain boundaries (IGB) | Crack‑like, layered | Percolating |

Under irradiation closed porosity tends to grow, coalesce, and reopen, drifting toward interconnected. Sintering can be tuned to favor closed (HIP, SPS, slow ramps) or leave open networks (insufficient sintering).

## 3. Effective thermal conductivity — analytical models

Defined via volume-averaged Fourier:
$$ \langle \mathbf{q} \rangle = -\kappa_{\text{eff}} \langle \nabla T \rangle $$
on a Representative Volume Element (RVE).

### 3.1 Bounding models

- **Voigt** (parallel): $\kappa_V = \phi \kappa_g + (1-\phi)\kappa_m$
- **Reuss** (series): $\kappa_R = (\phi/\kappa_g + (1-\phi)/\kappa_m)^{-1}$
- **Hashin–Shtrikman** (variational, tighter): for $\kappa_m > \kappa_g$,
$$ \kappa_{HS}^{-} = \kappa_g + \frac{1}{\frac{1-\phi}{3(\kappa_m-\kappa_g)} + \frac{1}{\kappa_m}} $$

### 3.2 Models for porous media

- **Maxwell–Eucken** (dilute spherical inclusions, valid $\phi < 0.1$):
$$ \frac{\kappa_{\text{eff}}}{\kappa_m} = \frac{2\kappa_m + \kappa_g - 2\phi(\kappa_m - \kappa_g)}{2\kappa_m + \kappa_g + \phi(\kappa_m - \kappa_g)} $$
- **Loeb** (IAEA-recommended for nuclear fuels):
$$ \kappa_{\text{eff}} = \kappa_m (1 - \alpha \phi) $$
  - **IAEA**: $\alpha = 2.5$ for UO$_2$
  - **This work**: $\alpha = 1.37$ recalibrated against FFT data (matches Morimoto 2008 for $(U,Pu,Am)O_2$). Implemented in `experiments/fit_correction_factor.py` as `ALPHA_LOEB = 1.37`, `K_MATRIX = 1.0`.
- **Meredith–Tobias** (extends Maxwell to higher $\phi$, $\beta = (\kappa_g/\kappa_m - 1)/(\kappa_g/\kappa_m + 2)$):
$$ \frac{\kappa_{\text{eff}}}{\kappa_m} = \frac{(2 + 2\beta\phi)(2 + (2\beta-1)\phi - (\beta+1)\phi)}{(2 - \beta\phi)^2} $$

### 3.3 Crack-density models for IGB porosity (PCW family)

Classical PCW assumes random crack locations. **Sevostianov & Kachanov (2019)** modified it to enforce that intergranular cracks saturate when all grain boundaries are cracked — for cubic grains, $\rho^* \approx 0.54$.

$$ k_{\text{eff}} = k_0 (1 - \bar{\rho})^{(8/9)\rho^*}, \qquad \bar{\rho} = \rho/\rho^* $$

where $\rho = N_c R_c^3 / V$ is the dimensionless crack density. Predicts the sharp drop observed at the percolation threshold. **Not directly implemented**, but cited as the theoretical justification for the sigmoidal shape in §6.

## 4. RVE and homogenization

### 4.1 RVE concept (Kanit et al. 2003)

An RVE is statistically representative when:
- $L \gg \ell$ (scale separation; rule of thumb $L > 10\ell$)
- contains hundreds of features
- variance reduced by averaging over realisations

Mattiuz thesis §3.1.1: $K_{\text{eff}}$ stabilises for $L_{\text{RVE}}/R \geq 20$ (within 1 %); paper used 25 to be conservative.

### 4.1a Two quantitative quality criteria for FFT homogenisation

These are the rules of thumb the framework was built around. Both must be satisfied jointly, otherwise results become voxelisation artefacts rather than physics:

1. **Statistical representativity**: $L_{\text{RVE}} / R_{\text{pore}} > 10$. RVE side at least 10 times the pore radius. Sensitivity sweep in thesis Fig 3 shows $K_{\text{eff}}$ stabilises around the boundary; paper uses ratio $\geq 25$ to be conservative.
2. **Geometric resolution**: $R_{\text{pore}} / \Delta_{\text{vox}} > 5$. Each pore must be sampled by at least five voxels along its radius. The same criterion applies to the grain-boundary thickness $\delta$: when $\delta$ approaches $\Delta_{\text{vox}}$, the percolating-crack regime is replaced by a staircase of disconnected voxels, and the simulated $K_{\text{eff}}$ silently jumps back toward the Loeb (distributed) baseline. **This is the dominant failure mode of the δ-sweep at low δ** and is directly relevant to the proposed extension below δ=0.15: the new sweep must keep $R_{\text{pore}}/\Delta_{\text{vox}} > 5$, which sets a floor on $n_\text{3D}$ for any chosen δ.

### 4.2 Homogenization equation

Local steady-state heat conduction:
$$ \nabla \cdot (\kappa(\mathbf{x}) \nabla T(\mathbf{x})) = 0 $$
Effective tensor extracted from spatial average:
$$ \kappa_{\text{eff}} \cdot \nabla T^{\text{macro}} = \langle \kappa(\mathbf{x}) \nabla T(\mathbf{x}) \rangle $$
For asymmetric microstructures $\kappa_{\text{eff}}$ is anisotropic; `solver.py` returns the diagonal $K_{xx}, K_{yy}, K_{zz}$ and a mean.

### 4.3 FFT solver and BCs

Lippmann–Schwinger solved on a regular voxel grid in Fourier space → near-linear scaling, no meshing. **Periodic BCs** are the default (other choices are biased): naturally enforced by AMITEX-FFTP. Errors typically 2–3 % when RVEs are sized properly (Kanit, Moutin, Schneider).

## 5. Microstructure parameters

In Mérope:
- **Distributed**: spheres via Boolean (overlap allowed) or RSA (non-overlapping). Fixed radius or log-normal. Volume fraction → porosity (exact for RSA, lower than imposed for Boolean due to overlap).
- **Interconnected**: Laguerre tessellation (weighted Voronoi from polydisperse sphere packings via `sac_de_billes`) plus thin shells of pore phase laid on grain boundaries. Layer thickness $\delta$ is the key knob.
- **Composite voxels**: voxels straddling phase boundaries get sub-voxel mixing (Voigt rule by default). This is essential for thin features like grain-boundary cracks at moderate resolution.
- **Mixing laws** (paper's Eq. for composite voxels):
$$ \lambda_{\text{Voigt}} = \sum_i \phi_i \lambda_i, \qquad \lambda_{\text{Reuss}} = \left(\sum_i \phi_i / \lambda_i\right)^{-1} $$

Voigt converges fastest for this contrast ratio (Mattiuz thesis Fig 5); used throughout. `core/geometry.py` enforces `voxel_rule = merope.vox.VoxelRule.Average` + `homogRule = merope.HomogenizationRule.Voigt`.

## 6. The δ\* descriptor and sigmoidal correction (this paper's contribution)

### 6.1 Normalized grain-boundary thickness

$$ \delta^* = \delta / L_{\text{grain}} $$

makes the descriptor scale-independent so distinct microstructures with different $L_{\text{grain}}$ can be compared.

| Regime | Behavior |
|---|---|
| $\delta^* < 0.2$ | Crack-like, percolating, severe degradation |
| $\delta^* \approx 0.2$–$0.5$ | Sharp transition |
| $\delta^* > 0.5$ | Effectively distributed, recovery toward Loeb |

### 6.2 Sigmoidal correction factor

Multiplicative correction on top of Loeb:
$$ K_{\text{eff}}(p, \delta^*) = K_{\text{Loeb}}(p) \cdot K_\delta(p, \delta^*) $$

**Naming note**: earlier conversations and exploratory writeups (some authored via Gemini) refer to the same quantity as **η** (eta). The paper and the codebase use **$K_\delta$**. If a stray "η correction factor" or "η vs δ" plot surfaces, treat it as a synonym — there is no separate η.

with
$$ K_\delta(p, \delta^*) = K_{\min}(p) + \frac{K_{\max}(p) - K_{\min}(p)}{1 + \exp\!\left[b(p)\,(\delta^* - \delta_c(p))\right]} $$

Each parameter linear in $p$. Implemented in `experiments/fit_correction_factor.py` (`sigmoidal_correction`, `full_model`).

### 6.3 ⚠ Three different parameter sets exist on disk

This is a real reconciliation issue — see `03_paper_and_open_issues.md` for the action item.

| Source | $K_{\min}(p)$ | $K_{\max}(p)$ | $b(p)$ | $\delta_c(p)$ |
|---|---|---|---|---|
| Mattiuz thesis p. 21 | $1.0049 - 0.157\,p$ | $1.098 - 4.116\,p$ | $10.86 + 89.98\,p$ | $-0.083 + 0.862\,p$ |
| `WORKFLOW_COMPLETE.md` | $-4.74\,p + 1.26$ | $-0.15\,p + 1.00$ | $-1.98\,p - 5.58$ | $-1.08\,p + 0.64$ |
| `Results_Sigmoidal_Fit/fitted_parameters.csv` (current, per-p rows) | 0.569 / ~0 / ~0 | 0.975 / 0.962 / 0.949 | −16.2 / −20 / −19.5 | ~0 / 0.082 / 0.151 |

The current CSV has $K_{\min} \to 0$ for $p = 0.2, 0.3$ (the fit went degenerate there). The paper figures (`Sigmoidal_Fits.png`, `Parameters_vs_Porosity.png`, `K_eff_Contour.png` in `paper/Images/Sigmoidal_Fit/`) need to be re-generated from a clean fit, or the canonical parameter set has to be chosen and locked in.

### 6.4 Key physical predictions

- $K_{\min}(p)$ decreases with $p$ → stronger crack-dominated degradation at higher porosity
- $K_{\max}(p) \to 1$ → recovery to Loeb at high $\delta^*$
- $\delta_c(p)$ rises with $p$ → higher porosity needs thicker GB shells before percolation breaks
- $b(p)$ controls sharpness of transition

## 7. Image-based inverse problem

### 7.1 Statistics

For each candidate 3D RVE: extract N 2D slices, segment, compute pore-size CDF and pore-density grid, run KS and $\chi^2$ tests against the experimental SEM image. Combined score:
- $w_{\text{data}} \cdot (\text{KS} + \chi^2)$ + $w_{\text{porosity}} \cdot |p_{\text{sim}} - p_{\text{target}}|$

Pores below 30 px excluded as segmentation noise. Stereological correction factor 0.85 applied to convert 2D area fraction to 3D volume fraction (Underwood 1970).

### 7.2 Surrogate optimization

Bayesian optimization (`skopt.gp_minimize`, Gaussian-process surrogate, Expected Improvement acquisition). Each function call is expensive (full 3D voxelization + slicing + tests, optionally + AMITEX). 20–80 calls typical.

Optimized parameters (current scope):
- **Distributed**: log-normal $\mu, \sigma$ of intra pore radius (and `small_frac` in newer code)
- **Interconnected**: $\delta$, intra pore $\phi$, intra pore radius

Fixed inputs (could be unlocked): RVE size, voxel count, grain size, conductivities.

## 8. Quick reference — canonical numerical values used

| Quantity | Value | Origin |
|---|---|---|
| $\kappa_m$ (matrix) | 1.0 W/m·K (normalized) | `K_THERMAL[0]` everywhere |
| $\kappa_g$ (gas) | 10⁻³ W/m·K | `K_THERMAL[2]` everywhere |
| Loeb $\alpha$ (recalibrated) | 1.37 | `ALPHA_LOEB` in `fit_correction_factor.py` |
| Stereo correction | 0.85 | Pore analysis pipeline |
| Pore exclusion threshold | 30 px (statistics), 1.5 µm² (pore_analysis) | `statistics.py`, `pore_analysis.py` |
| Sevostianov saturation | $\rho^* \approx 0.54$ | Cubic-grain estimate, cited not implemented |
| Composite voxel rule | Voigt (= `HomogenizationRule.Voigt`) | `geometry.py` |
| Voxel rule | `VoxelRule.Average` | `geometry.py` |
