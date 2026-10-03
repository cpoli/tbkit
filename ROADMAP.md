# Roadmap

What tbkit does not do yet, in priority order. The comparison is with
[Kwant](https://kwant-project.org) (quantum transport) and
[PythTB](https://www.physics.rutgers.edu/pythtb/) (band topology), the two
packages closest to tbkit's scope.

tbkit already covers more physics than either package: band topology,
Hall and optical response, non-Hermitian and Floquet models, mean field,
moiré systems. What it lacks is mostly transport machinery and
scalability. The scattering matrix, conservation laws, finite
temperature and automatic lead attachment are in place (see Done). What
remains for mesoscopic device work is completing lead interfaces (item 1)
and, above all, scale (item 2). Until then, Kwant remains the right tool
for large transport calculations. Items 4 and 6 are smaller, and item 5
(interactions) holds the next additions on the physics side.

Each item lists why it matters, the current state, and a sketch of the
approach. The project conventions (validation in `error_handling.py`, 100%
line coverage, an asserted gallery example, a history entry when there is
a breakthrough to tell) apply to all of them.

## 1. Completing lead interfaces

**Why.** `Transport.attach_lead` (see Done) needs the device to end with
a complete lead cell. A device whose edge is ragged relative to the lead,
such as a graphene flake under a zigzag lead or a disc under a strip,
has to be trimmed or padded by hand first.

**Now.** `attach_lead` refuses an incomplete interface cell and names the
missing lead orbitals.

**Approach.** An `extend=True` option that adds the missing lead sites to
the device, as Kwant's `attach_lead` does. Their bonds come from the
lead's hoppings, and the method returns the enlarged Hamiltonian and site
positions. Repeat cell by cell until every lead orbital with a hopping
into the next cell is present.

## 2. Performance and scale

**Why.** Many workflows (Wilson loops on fine meshes, `System.get_eig` on
large flakes, transport of 2D devices) are dense and single-threaded.

**Now.** The sparse paths (KPM, the k-d tree neighbour search,
shift-invert, `RecursiveTransport`) cover part of this. The k-space
diagonalizations are batched over k and optionally threaded
(`KSpace.set_workers`, see Done), and `tests/benchmark_kspace.py` times
them. The scattering matrix (`Transport.smatrix` and the default
`Transport.transmission`) factorizes the sparse device. Real-space
`System.get_eig` and the Green's-function methods of `Transport`
(`get_green`, the conductance matrix, currents and noise) remain dense
and single-threaded.

**Approach.**
- Parallelism over energies in transport and in
  `surface_spectral_function` (`concurrent.futures`, as for k).
- The rest of `Transport` on the sparse factorization: currents, noise
  and the conductance matrix from the scattering states and the S-matrix.
- Batch the remaining per-k loops: `biorthogonal_berry_curvature`, the
  quantum geometric tensor, and the Floquet `get_ham` (one evolution
  per k-point today).

## 3. Interoperability and API limits

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

## 4. Time-dependent transport

Transient currents and pulses through a device, like tkwant. This is
niche. It can build on the scattering states (`Transport.wave_function`).

## 5. Interactions beyond on-site Hubbard mean field

**Why.** Magnetic order in a bulk crystal, charge-density waves, and the
correlated phases of moiré bands all need mean-field theory in k-space or
interactions longer than on-site U. Devices need the charge to be
self-consistent with the electrostatics.

**Now.** `hubbard_mean_field`, `hubbard_mean_field_noncollinear` and
`bdg.s_wave_gap` work only on a finite `System`, with on-site
interactions. A bulk phase has to be approximated by a large flake or
torus.

**Approach.**
- Hartree-Fock on a `KSpace` mesh: the density matrix per k, a
  (possibly enlarged) magnetic unit cell, and the same linear mixing as
  the real-space solver. The result is a `KSpace`, so every band and
  topology tool applies to it.
- Extended interactions: nearest-neighbour V, then a screened Coulomb
  potential, with Hartree and Fock terms on the bonds.
- Self-consistent Hartree-Poisson for gated devices, on the sparse
  scattering states (`Transport.ldos`).

## 6. Project infrastructure

- **Activate releases and the DOI**: the workflow is in place
  (`.github/workflows/release.yml`, see Done); register it as a PyPI
  trusted publisher, create the `pypi` environment on GitHub, and enable
  the repository on Zenodo, then add the DOI to `CITATION.cff` and the
  README.
- **Make the type check blocking**: `mypy tbkit` reports about 260 errors,
  mostly from the strictness of the NumPy stubs. Fix them module by module,
  then drop `continue-on-error` from the `typecheck` job.

## Done

- **Project infrastructure for review** (unreleased): the docs build,
  which re-runs every asserted gallery example, and the paper's code
  listings run in CI (job `docs`); ruff lints `tbkit` (job `lint`) and
  mypy runs as an advisory job; a release workflow builds on a `v*` tag,
  publishes to PyPI by trusted publishing, creates the GitHub Release from
  the CHANGELOG section and rebuilds gh-pages; `GrapheneLattice` and
  `GrapheneSystem` have docstrings.

- **Automatic lead attachment** (unreleased):
  `Transport.attach_lead(sys, ks, direction)` takes a lead as a 1D
  `KSpace` (for example a ribbon) placed in the device's coordinates. It
  matches the device sites to lead orbitals by position and tag, takes
  the outermost matched cell along `direction` as the interface, and
  couples it with the lead's own hoppings (`lead_from_kspace`). No
  coupling matrix or site list to write by hand, and a lead may cover only
  part of a wider device. Example:
  `examples/transport/plot_automatic_lead_attachment.py` (narrow leads at
  different heights on a wide cavity).

- **Wannier disentanglement and interpolation** (unreleased):
  `wannier.wannierize` takes more bands than trial orbitals, an outer
  energy `window` and a `frozen` inner window, and first selects at each
  k the `n_wann`-dimensional subspace that minimizes Ω_I (Souza, Marzari
  and Vanderbilt 2001, with linear mixing), then runs the spread
  minimization in it. `WannierFunctions.kspace()` returns the
  Wannier-basis Hamiltonian H_mn(R) as a `KSpace` (Wigner-Seitz images,
  built by the same code as `io.read_wannier90`), which reproduces the
  frozen bands on the mesh and interpolates between its points. Example:
  `models/plot_disentangling_entangled_bands.py` (the π bands of flat
  and buckled sp3 graphene).

- **Wannierization** (unreleased): `wannier.wannierize(ks, bands,
  trial, nk)` projects an isolated band group onto trial orbitals (Löwdin
  orthogonalization on the k-mesh), then minimizes the Marzari-Vanderbilt
  spread by conjugate gradients. The overlaps include the orbital
  positions, so the centres agree with the Wilson-loop ones. It returns a
  `WannierFunctions` with the functions on the sites of a finite `System`
  (`wf.system`), their centres and spreads, and Ω_I, Ω_D and Ω_OD. A
  Chern band shows the obstruction: the projection becomes singular and
  the spread grows with the mesh.
  Examples: `models/plot_building_maximally_localized_wannier_functions.py`,
  `topology/plot_wannier_obstruction_of_chern_bands.py`.

- **Berry-phase response beyond the Hall conductivity** (unreleased):
  `KSpace.orbital_magnetization` (the modern theory, Thonhauser et al.
  and Xiao et al. 2005, at any Fermi level and temperature; its slope in a
  gap is the Chern number, checked against the Streda formula on a
  `System` flake), `anomalous_nernst_conductivity` and
  `thermal_hall_conductivity` (entropy- and c2-weighted Berry curvature,
  Xiao et al. 2006 and Qin, Niu and Shi 2011; they satisfy the Mott and
  Wiedemann-Franz relations exactly), all sharing the Kubo pair sums of
  `hall_conductivity`, and `KSpace.axion_angle` (the second Chern form
  integrated along a gapped parameter path). Finite temperature in the
  k-space Kubo formulas was already in place. Examples:
  `magnetic_field/plot_orbital_magnetization.py`,
  `hall_effects/plot_anomalous_nernst_effect.py`,
  `plot_thermal_hall_effect.py`, `topology/plot_axion_angle.py`.

- **Continuum discretization** (unreleased): `continuum.discretize`
  turns a k·p Hamiltonian (a sympy expression, matrix or string in
  `k_x, k_y, k_z`) into a `KSpace` on a chain, square or cubic grid, with
  finite differences as in `kwant.continuum`. Its free symbols become
  value-function parameters, `discretize_symbolic` shows the hoppings,
  and `bridges.finite_system` gives the `System`. sympy is optional
  (`tbkit[continuum]`). Position-dependent coefficients (a potential
  V(x), a gate) are not supported: they need a real-space builder, not a
  `KSpace`. Example: `examples/models/plot_kp_theory_on_a_lattice.py`.

- **Analysis and plotting** (unreleased): fat bands
  (`KSpace.band_weights`, `plot_bands(weights=...)`, by orbital,
  sublattice or any operator such as `spin_operator`), spin textures
  (`spin_texture`, `plot_spin_texture`), Fermi surfaces in 2D and 3D
  (`fermi_surface`, `plot_fermi_surface`, by marching triangles and
  tetrahedra, without scikit-image), standard k-paths for the 1D and 2D
  Bravais lattices and the cubic, tetragonal and hexagonal ones
  (`high_symmetry_path`), the tetrahedron density of states
  (`dos.tetrahedron_dos`, `plot_dos(kernel='tetrahedron')`), and plot
  helpers for the surface spectral function, the Wannier-centre flow and
  the Berry curvature. Examples:
  `tight_binding/plot_fermi_surfaces.py`,
  `plot_projected_bands_and_spin_textures.py`, `plot_tetrahedron_method.py`.

- **Conservation laws and finite temperature in transport** (unreleased):
  `add_lead(..., conservation_law=Q)` finds the lead modes in each
  eigenspace of Q, and `SMatrix` addresses blocks as `(lead, block)`, for
  Andreev reflection (tau_z) and spin-resolved transport (sigma_z).
  `Transport.conductance` and `thermoelectric` (conductance, thermopower,
  thermal conductance) integrate over the Fermi window, and
  `conductance_matrix`/`four_terminal_resistance` take a `temperature`.
  The symmetry detectors (`symmetry_error`, `tenfold_class`) are not yet
  used to find Q automatically. Examples:
  `examples/superconductivity/plot_andreev_reflection.py`,
  `examples/transport/plot_thermoelectric_transport.py`.

- **Scattering-matrix transport** (unreleased): exact lead modes from the
  generalized eigenproblem of the transfer matrix (`lead_modes`, QZ, no
  `eta`, singular couplings allowed), the S-matrix from one sparse LU
  factorization of the device plus the modes' boundary conditions
  (`Transport.smatrix`, the Groth et al. 2014 formulation), and the
  scattering states and the LDOS (`Transport.wave_function`,
  `Transport.ldos`). `Transport.transmission` wraps it by default, and
  passing `eta` keeps the Caroli formula. Example:
  `examples/transport/plot_scattering_matrix.py`.

- **Missing topological invariants** (unreleased): spin Chern numbers
  (`KSpace.spin_chern_number`, on `sector_chern_numbers`), mirror Chern
  numbers (`KSpace.mirror_chern_number`), the four 3D Z2 indices in one
  call (`KSpace.z2_indices_3d`), Kitaev's Majorana number
  (`bdg.majorana_number`, `bdg.pfaffian`), and the new `tbkit.topology`
  module (the Bott index, the entanglement spectrum, also k-resolved as
  `KSpace.entanglement_spectrum`, and `find_weyl_points` with
  chiralities). Examples: `topology/plot_spin_chern_number.py`,
  `plot_mirror_chern_number.py`, `plot_bott_index.py`,
  `plot_entanglement_spectrum.py`, plus the updated Kitaev chain, 3D
  topological insulator and Weyl semimetal examples.

- **Batched and threaded k-space diagonalization** (unreleased): one
  Bloch sum and stacked NumPy eigensolvers per memory-bounded chunk of
  k-points, in `get_bands`, `mesh_bands`, `berry_curvature` and the Wilson
  loops, across threads with `KSpace.set_workers(n)`. About 10x faster
  on one thread. Benchmark: `tests/benchmark_kspace.py`. Example:
  `examples/large_scale/plot_batched_k_space.py`.

- **Parametrized Hamiltonians** (unreleased): value functions
  `t(site_i, site_j, **params)` and `onsite(site, **params)` in
  `System.set_hopping`/`set_onsite` and `KSpace.set_hopping`, evaluated
  per call by `get_ham(**params)` (`KSpace.set_params` for the other
  methods). Example: `examples/tight_binding/plot_parametrized_hamiltonians.py`.

- **Packaging and CI** (unreleased): a `py.typed` marker, Python 3.14 in
  the CI matrix, and a CI job on the minimum supported NumPy, SciPy and
  matplotlib, which found two NumPy 2-only calls.

- **Surface spectral functions of semi-infinite crystals** (0.4.2):
  `KSpace.surface_spectral_function`, the iterative surface Green's
  function, on either side, surface or bulk. Example:
  `examples/topology/plot_3d_topological_insulator.py`.

## Where tbkit stands

| | tbkit | Kwant | PythTB |
|---|---|---|---|
| Model building | Real-space `System` and Bloch `KSpace`, with bridges between them, value functions (`get_ham(**params)`) and k·p discretization | Graph `Builder` with symmetries and value functions | `TBModel` (2.0; `tb_model` kept as an alias), any periodic and real-space dimensions, parametrized terms |
| Transport | S-matrix from exact lead modes (sparse), automatic lead attachment, conservation-law blocks, scattering states and LDOS, finite temperature and thermoelectrics; Landauer, multi-terminal, currents, shot noise (dense, finite `eta`), recursive Green's function | S-matrix, lead modes, wavefunctions, conservation laws, sparse solver | None |
| Band topology | Chern, spin and mirror Chern, Wilson loops, 2D Z2 and the four 3D indices, Fu-Kane parities, Majorana number, Weyl points, quantum geometry, tenfold way, higher order, local Chern marker, Bott index, entanglement spectrum | Minimal (via qsymm) | Berry phases, Wilson loops, hybrid Wannier centres; since 2.0 also Chern numbers, Berry curvature, quantum metric, local Chern marker, axion angle, Wannier functions |
| Surface states | Ribbons and slabs, semi-infinite surface spectral function | Leads and `kwant.physics` | Finite slabs (`cut_piece`) |
| Response | Anomalous and spin Hall, orbital magnetization, anomalous Nernst and thermal Hall, axion angle, Kubo-Bastin, optics, Berry curvature dipole, shift current | Kubo conductivity (KPM) | Berry curvature and velocities, no conductivity routines |
| Non-Hermitian, Floquet | Extensive | None | None |
| Scale | NumPy/SciPy, mostly dense; k-meshes batched and threaded; sparse LU for the S-matrix | Cython and MUMPS, about 10^6 sites | Vectorized NumPy since 2.0 |

Measured on one thread (`paper/benchmarks/run_benchmarks.py`, Kwant 1.5.0,
PythTB 2.0.2): on k-meshes tbkit and PythTB 2 are within a factor of two of
each other, tbkit using 5-14x less memory; for the scattering matrix Kwant
is faster from about 10^3 sites, and ten times faster at 4x10^5 sites.
