# Video description boilerplate

Copy-paste source for the descriptions attached to the rendered case videos when
they are published outside this repository (YouTube and similar). Nothing reads
this file programmatically — it is a writing aid, not configuration.

Use it as: the general block below, with the `<PLACEHOLDER>` run metadata filled
in, preceded by the matching per-case snippet from the end of this file. The
metadata it asks for (case, final time, step count, particle count, dimension,
precision, device) is exactly what a run's own banner and completion report
print, so the easiest source is the console output of the run that produced the
video — or the `caseSpec.json` written next to its export.

Two caveats before reusing it:

- **The numbering here is the original 01–15 scheme**, where 07 was Sedov 2D and
  15 was the equal-mass triple point. The examples tree has since folded 07 into
  [`examples/compressible/06-sedov/`](examples/compressible/06-sedov/) and 15
  into [`examples/compressible/14-triplePoint/`](examples/compressible/14-triplePoint/),
  so the snippet numbers no longer match the directory names. The case *names*
  are still correct.
- **The Case Data table is only partly filled in.** Blank rows are simply rows
  nobody has published a video for yet.

---

## Video Description Boilerplate (General)

This video shows a GPU-based compressible SPH simulation from the warpSPH benchmark suite.

The render includes every exported simulation time step. If adaptive time stepping is enabled, note that playback time is not the same as physical simulation time.

Simulation stack:
- Codebase: https://github.com/wi-re/warpSPH
- Warp backend: https://github.com/wi-re/warpSPHCore
- Timesteppers: https://github.com/wi-re/warpSPHIntegrators

Method summary:
- SPH Scheme: CRKSPH
- Time Integration: Runge-Kutta 2 (warpSPHIntegrators backend)
- Kernel: B7 spline
- Neighbor Search: Warp compact-hash GPU neighbor search
- Equation of State: Ideal gas

Run metadata:
- Case: <CASE_NAME>
- Final simulated time: <T_FINAL>
- Total Number of steps: 
- Particle count: <N_PARTICLES>
- Dimensionality: <DIM>
- Precision: <PRECISION>
- Device: <GPU_NAME>

Reproducibility notes:
- Exported states and media are generated directly from the benchmark notebooks.
- The full example set and outputs are available in this repository.

References:
[1] diffSPH: Differentiable smoothed particle hydrodynamics for hybrid machine learning solutions in fluid mechanics.
Rene Winchenbach and Nils Thuerey, Journal of Computational Physics.
https://doi.org/10.1016/j.jcp.2026.114769

[2] CRKSPH: A Conservative Reproducing Kernel Smoothed Particle Hydrodynamics Scheme.
Nicholas Frontiere, Cody D. Raskin, J. Michael Owen, Journal of Computational Physics.
https://doi.org/10.1016/j.jcp.2016.12.004

[3] A new class of accurate, mesh-free hydrodynamic simulation methods.
Philip F. Hopkins, Monthly Notices of the Royal Astronomical Society.
https://doi.org/10.1093/mnras/stv195

License:
The implementation and examples are open source (MIT and/or Apache-2.0, depending on repository component).

---

Case Data:
Case | $t$ | num steps  | num particles | D | precision 
Sod Shock Tube | 0.3 | 998 | 500 | 1 | single
Linear Wave | 1.0 | 1000 | 200 | 1 | double
Kidder Isentropic Compression | 28ms (0.99 tau) | 17450 | 100 | 1 | double
Noh Implosion | 
Woodward-Colella Double Blastwave | 
Sedov-Taylor Blastwave | 
Sedov-Taylor Blastwave | 
Hydrostatic | 
Gresho-Chan Vortex | 
Yee Vortex | 
Shearing Noh Implosion | 
Kevin Helmholtz Instability | 
Rayleight Taylor Instability | 
Triple Point (equal resolution sampling) | 
Triple Point (equal mass sampling) | 10.0 | 5673 | 190K | 2D | single


## Short Case Intro Snippets (Place At Top Of Description)

### 01. Sod Shock Tube (1D)
This video shows the 1D Sod shock tube benchmark, a classical Riemann problem with a left-going rarefaction, contact discontinuity, and right-going shock.

### 02. Linear Wave
This video shows the 1D linear wave benchmark, tracking propagation of a small-amplitude acoustic perturbation to evaluate phase accuracy and dissipation.

### 03. Kidder Isentropic Compression
This video shows the Kidder isentropic compression benchmark, a smooth adiabatic compression-expansion flow used to assess entropy and energy consistency.

### 04. Noh Implosion
This video shows the Noh implosion benchmark, where a converging flow produces a strong central shock that stresses shock-capturing robustness.

### 05. Woodward-Colella Double Blastwave
This video shows the Woodward-Colella double blastwave benchmark, featuring interacting strong shocks and contact discontinuities in 1D.

### 06. Sedov-Taylor Blastwave (1D)
This video shows the 1D Sedov-Taylor blastwave benchmark, where localized energy deposition drives a strong self-similar shock.

### 07. Sedov-Taylor Blastwave (2D)
This video shows the 2D Sedov-Taylor blastwave benchmark, focusing on radial blast expansion, isotropy, and shock-front quality.

### 08. Hydrostatic
This video shows the hydrostatic benchmark, testing whether pressure gradients and gravity remain in near-equilibrium with minimal spurious motion.

### 09. Gresho-Chan Vortex
This video shows the Gresho-Chan vortex benchmark, a nominally steady rotating flow used to evaluate low-Mach dissipation and angular-momentum behavior.

### 10. Yee Vortex
This video shows the Yee vortex benchmark, a smooth isentropic vortex advection case used to measure long-time transport accuracy.

### 11. Shearing Noh Implosion (2D)
This video shows the 2D shearing Noh implosion benchmark, combining strong compression and shear to challenge stability and symmetry.

### 12. Kelvin-Helmholtz
This video shows the Kelvin-Helmholtz instability benchmark, where shear layers roll up into vortices and develop mixing structures over time.

### 13. Rayleigh-Taylor
This video shows the Rayleigh-Taylor instability benchmark, where a heavy fluid over a light fluid produces characteristic bubble-and-spike growth.

### 14. Triple Point (Equal Resolution)
This video shows the triple-point benchmark (equal-resolution setup), a multi-region Riemann interaction with shocks, contacts, and slip lines.

### 15. Triple Point (Equal Mass)
This video shows the triple-point benchmark (equal-mass setup), highlighting wave interaction behavior under an equal-mass particle sampling strategy.