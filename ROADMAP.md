# Roadmap

What tbkit does not do yet, in priority order. The comparison is with
[Kwant](https://kwant-project.org) (quantum transport) and
[PythTB](https://www.physics.rutgers.edu/pythtb/) (band topology), the two
packages closest to tbkit's scope.

tbkit already covers more physics than either package: band topology,
Hall and optical response, non-Hermitian and Floquet models, mean field,
moiré systems. What it lacks is mostly transport machinery and
scalability. The scattering matrix, conservation laws, finite
temperature and automatic lead attachment, including the completion of
ragged interfaces, are in place (see Done). What remains for mesoscopic
device work is, above all, scale (item 1). Until then, Kwant remains the
right tool for large transport calculations. Items 3 and 5 are smaller,
and item 4 (interactions) holds the next additions on the physics side.
Item 6 lists history breakthroughs still missing from the docs.

Each item lists why it matters, the current state, and a sketch of the
approach. The project conventions (validation in `error_handling.py`, 100%
line coverage, an asserted gallery example, a history entry when there is
a breakthrough to tell) apply to all of them.

## 1. Performance and scale

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

## 2. Interoperability and API limits

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

## 3. Time-dependent transport

Transient currents and pulses through a device, like tkwant. This is
niche. It can build on the scattering states (`Transport.wave_function`).

## 4. Interactions beyond on-site Hubbard mean field

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

## 5. Project infrastructure

- **Activate releases and the DOI**: the workflow is in place
  (`.github/workflows/release.yml`, see Done); register it as a PyPI
  trusted publisher, create the `pypi` environment on GitHub, and enable
  the repository on Zenodo, then add the DOI to `CITATION.cff` and the
  README.
- **Make the type check blocking**: `mypy tbkit` reports about 260 errors,
  mostly from the strictness of the NumPy stubs. Fix them module by module,
  then drop `continue-on-error` from the `typecheck` job.

## 6. History breakthroughs to add

**Why.** Two pages of `docs/source/history/` have fewer than 15
breakthroughs: interactions (11) and driven (11).
The candidates below have no entry of their own on any page yet. The aim
is to tell the missing stories, not to reach a count.

**Now.** Unless marked *new code*, the functionality is already in
tbkit. What is missing is the history entry and a dedicated gallery
example per breakthrough (no example shared between two entries).

**Flat bands and interactions** (`interactions_breakthroughs.rst`)
- Optional, *new code*: 2015, flat-band superconductivity and the
  superfluid weight bounded by the quantum metric (Peotta-Törmä).
  `quantum_geometric_tensor` exists; a superfluid-weight function does
  not.

**Driven and open systems** (`driven_breakthroughs.rst`)
- Optional, *new code*: 2018/2019, the 38-fold non-Hermitian symmetry
  classification (Kawabata et al.; Zhou-Lee), mirroring
  `KSpace.tenfold_class`.

**Approach.** One breakthrough per change: the entry (references, link to
the tbkit functions, `.. minigallery::`) and a new asserted example whose
title names the breakthrough. Where an entry today mentions the subject in
passing, move that text into the new entry rather than repeat it.

## Done

- **Driven and open systems history breakthroughs** (unreleased): five
  entries in `driven_breakthroughs.rst` (now 11), each with its own
  asserted example: Landau-Zener tunnelling and Zener breakdown
  (1932/1934), dynamic localization and coherent destruction of
  tunnelling (1986/1991), discrete optics in waveguide arrays
  (1999/2003), the photovoltaic Hall effect (2009) and the generalized
  Brillouin zone (2018/2019). Oka-Aoki moved out of the Floquet
  topological insulator entry (now 2011), and the GBZ out of the skin
  effect entry; `non_hermitian/plot_skin_effect.py` dropped its GBZ
  section for `non_hermitian/plot_generalized_brillouin_zone.py`.

- **Fields history breakthroughs** (unreleased): six entries in
  `fields_breakthroughs.rst` (now 13), each with its own asserted
  example: Rashba spin-orbit coupling (1959/1984), Zak's magnetic
  translation group (1964), Lifshitz tails (1964), the inverse
  participation ratio and multifractality (1970/1980), Aubry-André
  localization (1980) and vacancy zero modes in graphene (2006). The IPR
  definition moved from the Anderson entry into its own.

- **Flat bands and interactions history breakthroughs** (unreleased):
  four entries in `interactions_breakthroughs.rst` (now 11), each with
  its own asserted example: the Bogoliubov-de Gennes equations (1966,
  Andreev bound states), Nagaoka and Mielke-Tasaki flat-band
  ferromagnetism (1966/1991-1992, exact diagonalization of one-hole
  clusters and the kagome flat band), magnetism of triangular graphene
  nanoislands (2007) and Majorana nanowires (2010). The de Gennes text of
  the BCS entry and the nanowire sentence of the Kitaev entry moved to
  the new entries. `correlations/plot_lieb_theorem.py` now shows the
  ferrimagnet of the Lieb lattice, leaving the graphene triangles to
  `correlations/plot_triangular_graphene_nanoislands.py`.

- **Models history breakthroughs** (unreleased): six entries in
  `models_breakthroughs.rst` (now 16), each with its own asserted
  example: Brillouin zones and the Setyawan-Curtarolo paths (1930/2010),
  Löwdin orthogonalization (1950), Monkhorst-Pack grids (1976), fermion
  doubling (1981), Lopez Sancho decimation (1985) and zigzag edge states
  of graphene ribbons (1996). The zigzag half of
  `topology/plot_edge_states.py` moved to
  `tight_binding/plot_zigzag_edge_states.py`. New API along the way:
  `mesh_grid`/`mesh_bands(..., monkhorst_pack=True)` and a public
  `kspace.cubic_path`.

- **Completing lead interfaces** (unreleased):
  `Transport.attach_lead(..., extend=True)` adds to the device the lead
  sites it lacks, as Kwant's `attach_lead` does: a flood fill from the
  lead's first cell over the lead's hoppings, cell by cell towards the
  device, until every site reached is present (refused if it would pass
  the device's far end). The added sites carry the lead's onsite energies
  and hoppings and come after the device's, so earlier leads keep their
  sites; the method returns the enlarged Hamiltonian and the site
  positions. Example: `examples/transport/plot_completing_lead_interfaces.py`
  (a ragged strip gives back the full one; a graphene disc between zigzag
  leads).

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
| Transport | S-matrix from exact lead modes (sparse), automatic lead attachment (completing ragged interfaces), conservation-law blocks, scattering states and LDOS, finite temperature and thermoelectrics; Landauer, multi-terminal, currents, shot noise (dense, finite `eta`), recursive Green's function | S-matrix, lead modes, wavefunctions, conservation laws, sparse solver | None |
| Band topology | Chern, spin and mirror Chern, Wilson loops, 2D Z2 and the four 3D indices, Fu-Kane parities, Majorana number, Weyl points, quantum geometry, tenfold way, higher order, local Chern marker, Bott index, entanglement spectrum | Minimal (via qsymm) | Berry phases, Wilson loops, hybrid Wannier centres; since 2.0 also Chern numbers, Berry curvature, quantum metric, local Chern marker, axion angle, Wannier functions |
| Surface states | Ribbons and slabs, semi-infinite surface spectral function | Leads and `kwant.physics` | Finite slabs (`cut_piece`) |
| Response | Anomalous and spin Hall, orbital magnetization, anomalous Nernst and thermal Hall, axion angle, Kubo-Bastin, optics, Berry curvature dipole, shift current | Kubo conductivity (KPM) | Berry curvature and velocities, no conductivity routines |
| Non-Hermitian, Floquet | Extensive | None | None |
| Scale | NumPy/SciPy, mostly dense; k-meshes batched and threaded; sparse LU for the S-matrix | Cython and MUMPS, about 10^6 sites | Vectorized NumPy since 2.0 |

Measured on one thread (`paper/benchmarks/run_benchmarks.py`, Kwant 1.5.0,
PythTB 2.0.2): on k-meshes tbkit and PythTB 2 are within a factor of two of
each other, tbkit using 5-14x less memory; for the scattering matrix Kwant
is faster from about 10^3 sites, and ten times faster at 4x10^5 sites.
