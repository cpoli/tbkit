# Roadmap

What tbkit does not do yet, in priority order. The comparison is with
[Kwant](https://kwant-project.org) (quantum transport) and
[PythTB](https://www.physics.rutgers.edu/pythtb/) (band topology), the two
packages closest to tbkit's scope.

tbkit already covers more physics than either package: band topology,
Hall and optical response, non-Hermitian and Floquet models, mean field,
moiré systems. What it lacks is mostly transport machinery and
scalability. Items 1-4 would make tbkit credible for mesoscopic device
work. Until they land, Kwant remains the right tool for serious transport.
Items 10-13 are smaller. Section 12 (plotting and analysis) holds the
cheapest wins, and section 10 the most valuable additions on the physics
side.

Each item lists why it matters, the current state, and a sketch of the
approach. The project conventions (validation in `error_handling.py`, 100%
line coverage, an asserted gallery example, a history entry when there is
a breakthrough to tell) apply to all of them.

## 1. Scattering-matrix transport

**Why.** Mode-resolved transmission, reflection, Andreev and spin-resolved
conductances, and scattering wavefunctions inside the device are what
transport studies actually need. Kwant's `smatrix` and `wave_function` are
the reference.

**Now.** `Transport` builds the dense device Green's function by inverting
the whole device, O(N^3) per energy
(`transport.py`, `Transport.__init__` densifies `ham`). Leads use the
Lopez Sancho-Rubio decimation with a finite `eta`, which loses accuracy at
band edges. There are no lead modes, no S-matrix and no wavefunctions.
`RecursiveTransport` avoids the full inversion only for quasi-1D devices.

**Approach.**
- Lead modes from the generalized eigenproblem of the lead's transfer
  matrix. That gives exact propagating and evanescent modes and their
  velocities, with no `eta`.
- The S-matrix from the sparse linear system of the device plus the
  modes' boundary conditions (the Kwant/Groth et al. 2014 formulation),
  solved with `scipy.sparse.linalg.splu`.
- Scattering wavefunctions and the local density of states from the same
  factorization.
- Keep `Transport.transmission` as a thin wrapper, for backward
  compatibility.

## 2. Automatic lead attachment

**Why.** Today a lead is attached by hand: `add_lead(h0, v, coupling,
sites)` with an explicit coupling matrix and list of device sites. That is
error-prone for anything but a straight strip.

**Approach.** `attach_lead(system, lead_kspace, direction)`: find the
device sites that the lead's translation maps onto, build the coupling
from the lead's hoppings, and (optionally) extend the device until the
interface is complete, as Kwant's `attach_lead` does. `lead_from_kspace`
and `bridges.py` already hold most of the pieces.

## 3. Conservation laws, symmetries and finite temperature in transport

**Why.** NS junctions (Andreev reflection), spin-resolved transport and
valley filters need transmission blocks resolved by a conserved quantity.
Thermoelectrics and realistic conductances need finite temperature.

**Now.** `KSpace.symmetry_error` and `tenfold_class` detect symmetries
but nothing uses them to block-diagonalize. All Landauer quantities are at
zero temperature.

**Approach.** A `conservation_law` operator on `Transport` (and on leads)
projects the lead modes onto its eigenspaces and labels the S-matrix
blocks. Add a `temperature` argument to the conductances
(`-df/dE`-weighted integrals), plus the thermopower and thermal
conductance.

## 4. Parametrized Hamiltonians

**Why.** Sweeps over field, gate voltage or disorder seed currently
rebuild the whole model at every step. Kwant's value functions with
`params=` make such sweeps cheap and the code short.

**Now.** Hopping and onsite values are numbers. The only exception is
`set_peierls_phase`, which takes a callable.

**Approach.** Accept callables `t(site_i, site_j, **params)` and
`onsite(site, **params)` in `System.set_hopping`/`set_onsite` and
`KSpace.set_hopping`. Store the structure (indices, bond vectors) once and
evaluate the values per call with `get_ham(**params)`, vectorized over
bonds. It must stay backward compatible: plain numbers keep working
unchanged.

## 5. Continuum discretization (k·p to tight-binding)

**Why.** Semiconductor devices, BHZ and Bi2Se3 k·p models and
Bogoliubov-de Gennes nanowires are written as continuum Hamiltonians.
Kwant's `kwant.continuum` turns them into lattice models automatically.

**Approach.** `discretize(expr, a)` on a sympy expression in `k_x, k_y,
k_z` (finite differences on a square or cubic grid, with symmetrized
products). It returns a `KSpace` and the matching `System` hoppings. sympy
would be an optional dependency (`pip install tbkit[continuum]`).

## 6. Wannierization

**Why.** Building localized Wannier functions from a band group is the
bridge between band topology and real-space pictures. It is also where the
obstruction of Chern bands shows up. I believe PythTB 2.x has
projection-based and maximally localized Wannier functions; check its
current documentation before relying on this comparison.

**Now.** tbkit reads Wannier90 `_hr.dat` models and computes Wannier
centres and flow from Wilson loops, but cannot construct Wannier functions
itself.

**Approach.** Projection onto trial orbitals (Löwdin orthogonalization of
the projected Bloch states on a mesh), then the Marzari-Vanderbilt spread
minimization for isolated band groups, and disentanglement later. It
returns the Wannier functions on a finite `System` and their centres and
spreads.

## 7. Performance and scale

**Why.** Many workflows (Wilson loops on fine meshes, `System.get_eig` on
large flakes, transport of 2D devices) are dense and single-threaded.

**Now.** The sparse paths (KPM, the k-d tree neighbour search,
shift-invert, `RecursiveTransport`) cover part of this. Most of the
k-space topology works on dense matrices, looping over k.

**Approach.**
- Batch the diagonalizations over k with stacked `numpy.linalg.eigh`
  (as `_bloch_sum` already does for the Hamiltonians).
- Optional parallelism over k-points and energies (`concurrent.futures`,
  no new required dependency).
- Sparse solvers in transport (item 1).
- Keep a benchmark script in `tests/` so regressions are visible.

## 8. Interoperability and API limits

- Import and export Kwant `Builder`s, PythTB `tb_model`s, pybinding
  models and ASE `Atoms` geometries. That makes cross-checks against these
  packages one line.
- Write Wannier90 `_hr.dat` files (not only read them), and read
  `_tb.dat`.
- Sublattice tags are dtype `'U1'` (one character). Allow longer tags
  (for example `'Mo'`, `'Se1'`) without breaking existing models: widen
  the dtype and keep one-character tags working.
- Optional physical units (eV, Å, tesla) for magnetic fields and
  conductances. The current dimensionless conventions stay the default.

## 9. Time-dependent transport

Transient currents and pulses through a device, like tkwant. This is
niche, and it depends on items 1 and 4.

## 10. Missing topological invariants

**Why.** These are standard in the literature and in other codes, and each
one reuses machinery tbkit already has (Wilson loops, BdG models,
projectors).

- **Spin Chern number**: the Chern number of the two sectors of the
  projected spin operator P s_z P (Prodan 2009). It covers quantum spin
  Hall systems where s_z is not conserved, for example Kane-Mele with
  Rashba coupling.
- **Mirror Chern numbers**: Chern numbers of the mirror-eigenvalue sectors
  on a mirror-invariant plane, for topological crystalline insulators
  (SnTe). It needs a `mirror` operator argument, like `parity_z2`'s
  `inversion`.
- **BdG topology**: the Pfaffian Z2 invariant and Kitaev's Majorana number
  sign Pf[H(k=0)] Pf[H(k=pi)] in the Majorana basis. `bdg.py` builds the
  Hamiltonians but computes no invariant.
- **Bott index**: a real-space Chern number for disordered samples and
  quasicrystals (Loring and Hastings 2010), complementing
  `get_local_chern_marker`.
- **All four 3D Z2 indices in one call**: `z2_indices_3d(bands)` returning
  (nu0; nu1 nu2 nu3) from the six time-reversal-invariant planes. Today it
  takes six `z2_invariant(..., k_fixed=...)` calls combined by hand, which
  is how the weak phase of the 3D topological insulator example has to be
  identified.
- **Entanglement spectrum**: the eigenvalues of the correlation matrix
  restricted to half of a ribbon or flake. It is a basis-independent
  diagnostic of topology and cheap to compute.

## 11. Berry-phase response beyond the Hall conductivity

- **Orbital magnetization** in the modern theory (Thonhauser, Ceresoli,
  Vanderbilt and Resta 2005), from the same Bloch derivatives as
  `hall_conductivity`. It can be checked against the Streda formula
  dM/dmu = sigma_xy on a `System`.
- **Anomalous Nernst and thermal Hall conductivities**: energy-weighted
  integrals of the Berry curvature (Xiao et al. 2006).
- **Axion angle theta** of 3D insulators, for magnetic topological
  insulators. Low priority.

## 12. Analysis and plotting

These are cheap wins, about a day each, and they remove a lot of
boilerplate from the examples.

- **Fat bands and spin textures**: `KSpace.plot_bands(weights=...)` coloring
  or sizing the bands by orbital, sublattice or spin projection, and a
  spin-texture arrow plot on a constant-energy contour. `KSpace` currently
  plots only `plot_bands` and `plot_dos`.
- **Fermi surfaces and constant-energy contours** in 2D (contours on a
  mesh) and 3D (marching cubes, optional scikit-image). The Weyl semimetal
  example builds its Fermi-arc map by hand.
- **Automatic high-symmetry k-paths**: `high_symmetry_path(lat)` for the 2D
  Bravais lattices and the cubic, hexagonal and tetragonal lattices, with
  labels ready for `plot_bands(node_labels=...)`. Every example
  hard-codes Gamma, K and M today.
- **Tetrahedron-method density of states**: more accurate than Gaussian or
  Lorentzian broadening at van Hove singularities and band edges.
- **Plot helpers for the newer tools**: the surface spectral function, the
  Wilson loop and Wannier-centre flow, and the Berry curvature map. The
  examples draw all of these with raw matplotlib.

## 13. Project infrastructure

- **Docs and gallery in CI**: build the docs, which re-runs every asserted
  example, on each push. Today a code change that breaks an example passes
  CI and is only found at the next local docs build.
- **Release automation**: a workflow triggered by a `v*` tag that builds
  the wheel and sdist, publishes to PyPI with trusted publishing (no API
  token), creates the GitHub Release from the CHANGELOG section, and
  rebuilds gh-pages.
- **A DOI**: connect the repository to Zenodo so each release gets a
  citable DOI, and add it to `CITATION.cff` and the README.
- **Static checks**: ruff for linting and mypy for the existing type hints,
  as a CI job.

## Done

- **Surface spectral functions of semi-infinite crystals** (0.4.2):
  `KSpace.surface_spectral_function`, the iterative surface Green's
  function, on either side, surface or bulk. Example:
  `examples/topology/plot_3d_topological_insulator.py`.

## Where tbkit stands

| | tbkit | Kwant | PythTB |
|---|---|---|---|
| Model building | Real-space `System` and Bloch `KSpace`, with bridges between them | Graph `Builder` with symmetries and value functions | One `tb_model`, any periodic and real-space dimensions |
| Transport | Landauer, multi-terminal, currents, shot noise, recursive Green's function (dense, finite `eta`) | S-matrix, lead modes, wavefunctions, conservation laws, sparse solver | None |
| Band topology | Chern, Wilson loops, 2D/3D Z2, Fu-Kane parities, quantum geometry, tenfold way, higher order, local Chern marker | Minimal (via qsymm) | Berry phases, Wilson loops, hybrid Wannier centres |
| Surface states | Ribbons and slabs, semi-infinite surface spectral function | Leads and `kwant.physics` | Finite slabs (`cut_piece`) |
| Response | Anomalous and spin Hall, Kubo-Bastin, optics, Berry curvature dipole, shift current | Kubo conductivity (KPM) | None |
| Non-Hermitian, Floquet | Extensive | None | None |
| Scale | NumPy/SciPy, mostly dense | Cython and MUMPS, about 10^6 sites | Pure Python, small models |
