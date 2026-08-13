# Compressible SPH Example Gallery

This page summarizes the 13 benchmark examples in this folder (numbered 01-14;
07 was folded into 06's own directory alongside its 1D/2D variants, and 15 was
folded into 14's own directory alongside its equal-mass/equal-spacing variants).

- Static PNG previews are used to keep the page lightweight.
- Each case also includes an embedded MP4 (plus a direct link).
- Notebooks are linked for quick access.

## Quick Index

| Case | Notebook | Preview |
|---|---|---|
| 01. Sod Shock Tube (1D) | [01-sod/sod_1d.ipynb](01-sod/sod_1d.ipynb) | ![](01-sod/outputs/01-Sod_Shock_Tube.png) |
| 02. Linear Wave | [02-Linear_Wave.ipynb](02-Linear_Wave.ipynb) | ![](outputs/02-Linear_wave.png) |
| 03. Kidder Isentropic Compression | [03-Kidder_Isentropic_Compression.ipynb](03-Kidder_Isentropic_Compression.ipynb) | ![](outputs/03-Kidder_Isentropic_compression.png) |
| 04. Noh Implosion | [04-Noh_Implosion.ipynb](04-Noh_Implosion.ipynb) | ![](outputs/04-Noh_Implosion.png) |
| 05. Woodward-Colella Double Blastwave | [05-Woodward_Colella.ipynb](05-Woodward_Colella.ipynb) | ![](outputs/05-Wodward_Colella_Double_Blastwave.png) |
| 06. Sedov-Taylor Blastwave (1D/2D/3D) | [06-sedov/sedov_1d.ipynb](06-sedov/sedov_1d.ipynb) | ![](06-sedov/outputs/06-Sedov_Taylor_Blastwave_1D.png) |
| 08. Hydrostatic | [08-Hydrostatic.ipynb](08-Hydrostatic.ipynb) | ![](outputs/08-Hydrostatic.png) |
| 09. Gresho-Chan Vortex | [09-Gresho_Chan_Vortex.ipynb](09-Gresho_Chan_Vortex.ipynb) | ![](outputs/09-Gresho_Chan_Vortex.png) |
| 10. Yee Vortex | [10-Yee_Vortex.ipynb](10-Yee_Vortex.ipynb) | ![](outputs/10-Yee_Vortex.png) |
| 11. Shearing Noh Implosion (2D) | [11-Shearing_Noh_Implosion_2D.ipynb](11-Shearing_Noh_Implosion_2D.ipynb) | ![](outputs/11-Shearing_Noh_2D.png) |
| 12. Kelvin-Helmholtz | [12-Kelvin-Helmholtz.ipynb](12-Kelvin-Helmholtz.ipynb) | ![](outputs/12-Kelvin_Helmholtz.png) |
| 13. Rayleigh-Taylor | [13-Rayleigh_Taylor.ipynb](13-Rayleigh_Taylor.ipynb) | ![](outputs/13-Rayleigh_Taylor.png) |
| 14. Triple Point (Equal Spacing/Mass) | [14-triplePoint/triplePoint_equalSpacing.ipynb](14-triplePoint/triplePoint_equalSpacing.ipynb) | ![](14-triplePoint/outputs/14-Triple_Point_equal_resolution.png) |

## Case Details (Preview + MP4)

### 01. Sod Shock Tube (1D)
Classical 1D Riemann problem with rarefaction, contact discontinuity, and shock. Own directory
([01-sod/](01-sod/)) with a resume script/notebook and a demo of the trajectory export scheme.

![](01-sod/outputs/01-Sod_Shock_Tube.png)

<video src="01-sod/outputs/01-Sod_Shock_Tube.mp4" controls width="900"></video>

[Open MP4](01-sod/outputs/01-Sod_Shock_Tube.mp4)

### 02. Linear Wave
Small-amplitude acoustic wave propagation test for phase and dissipation behavior.

![](outputs/02-Linear_wave.png)

<video src="outputs/02-Linear_wave.mp4" controls width="900"></video>

[Open MP4](outputs/02-Linear_wave.mp4)

### 03. Kidder Isentropic Compression
Smooth compression/expansion benchmark for entropy and adiabatic consistency.

![](outputs/03-Kidder_Isentropic_compression.png)

<video src="outputs/03-Kidder_Isentropic_compression.mp4" controls width="900"></video>

[Open MP4](outputs/03-Kidder_Isentropic_compression.mp4)

### 04. Noh Implosion
Strong converging-flow shock benchmark for robustness under high compression.

![](outputs/04-Noh_Implosion.png)

<video src="outputs/04-Noh_Implosion.mp4" controls width="900"></video>

[Open MP4](outputs/04-Noh_Implosion.mp4)

### 05. Woodward-Colella Double Blastwave
Interacting strong shocks and contacts in 1D.

![](outputs/05-Wodward_Colella_Double_Blastwave.png)

<video src="outputs/05-Wodward_Colella_Double_Blastwave.mp4" controls width="900"></video>

[Open MP4](outputs/05-Wodward_Colella_Double_Blastwave.mp4)

### 06. Sedov-Taylor Blastwave (1D/2D/3D)
Localized energy deposition driving a strong, self-similar blast wave -- a compressible
scheme's strong-shock and energy-conservation test under an extreme initial gradient. Own
directory ([06-sedov/](06-sedov/)) with 1D/2D/3D variants of the same case, and a smoothed
(`'hat'`) as well as a raw single-particle-spike (`'singular'`) initial condition.

![](06-sedov/outputs/06-Sedov_Taylor_Blastwave_1D.png)

<video src="06-sedov/outputs/06-Sedov_Taylor_Blastwave_1D.mp4" controls width="900"></video>

[Open MP4](06-sedov/outputs/06-Sedov_Taylor_Blastwave_1D.mp4)

![](06-sedov/outputs/06-Sedov_Taylor_Blastwave_2D.png)

<video src="06-sedov/outputs/06-Sedov_Taylor_Blastwave_2D.mp4" controls width="900"></video>

[Open MP4](06-sedov/outputs/06-Sedov_Taylor_Blastwave_2D.mp4)

![](06-sedov/outputs/06-Sedov_Taylor_Blastwave_3D.png)

<video src="06-sedov/outputs/06-Sedov_Taylor_Blastwave_3D.mp4" controls width="900"></video>

[Open MP4](06-sedov/outputs/06-Sedov_Taylor_Blastwave_3D.mp4)

### 08. Hydrostatic
Gravity-balanced static fluid test for hydrostatic equilibrium preservation.

![](outputs/08-Hydrostatic.png)

<video src="outputs/08-Hydrostatic.mp4" controls width="900"></video>

[Open MP4](outputs/08-Hydrostatic.mp4)

### 09. Gresho-Chan Vortex
Steady rotating vortex test for low-dissipation and angular-momentum behavior.

![](outputs/09-Gresho_Chan_Vortex.png)

<video src="outputs/09-Gresho_Chan_Vortex.mp4" controls width="900"></video>

[Open MP4](outputs/09-Gresho_Chan_Vortex.mp4)

### 10. Yee Vortex
Smooth isentropic vortex benchmark for long-time advection quality.

![](outputs/10-Yee_Vortex.png)

<video src="outputs/10-Yee_Vortex.mp4" controls width="900"></video>

[Open MP4](outputs/10-Yee_Vortex.mp4)

### 11. Shearing Noh Implosion (2D)
Converging shock with shear, stressing shock capture and symmetry in 2D.

![](outputs/11-Shearing_Noh_2D.png)

<video src="outputs/11-Shearing_Noh_2D.mp4" controls width="900"></video>

[Open MP4](outputs/11-Shearing_Noh_2D.mp4)

### 12. Kelvin-Helmholtz
Shear-layer instability with vortex roll-up and mixing.

![](outputs/12-Kelvin_Helmholtz.png)

<video src="outputs/12-Kelvin_Helmholtz.mp4" controls width="900"></video>

[Open MP4](outputs/12-Kelvin_Helmholtz.mp4)

### 13. Rayleigh-Taylor
Buoyancy-driven instability with bubble and spike growth.

![](outputs/13-Rayleigh_Taylor.png)

<video src="outputs/13-Rayleigh_Taylor.mp4" controls width="900"></video>

[Open MP4](outputs/13-Rayleigh_Taylor.mp4)

### 14. Triple Point (Equal Spacing/Mass)
Multi-region 2D Riemann interaction with shocks, contacts, and slip lines -- a
compressible scheme's test for mass-resolution handling at a density jump. Own
directory ([14-triplePoint/](14-triplePoint/)) with equal-spacing and
equal-mass sampling variants of the same case.

![](14-triplePoint/outputs/14-Triple_Point_equal_resolution.png)

<video src="14-triplePoint/outputs/14-Triple_Point_equal_resolution.mp4" controls width="900"></video>

[Open MP4](14-triplePoint/outputs/14-Triple_Point_equal_resolution.mp4)

![](14-triplePoint/outputs/14-Triple_Point_equal_mass.png)

<video src="14-triplePoint/outputs/14-Triple_Point_equal_mass.mp4" controls width="900"></video>

[Open MP4](14-triplePoint/outputs/14-Triple_Point_equal_mass.mp4)
