# Roadmap

What tbkit does not do yet. The comparison is first with
[Kwant](https://kwant-project.org) (quantum transport) and
[PythTB](https://www.physics.rutgers.edu/pythtb/) (band topology), the two
packages closest to tbkit's scope. Items 6 to 12 come from an audit
(2026-11-15) of the whole package against those two and against tkwant,
qsymm, pybinding, KITE, TBPLaS, sisl, Z2Pack, WannierTools, WannierBerri
and TBmodels, and against the history pages: what would someone doing or
teaching tight-binding physics look for and not find?

tbkit already covers more physics than Kwant or PythTB: band topology,
Hall and optical response, non-Hermitian and Floquet models, mean field,
moiré systems. The scattering matrix, conservation laws, finite
temperature and automatic lead attachment, including the completion of
ragged interfaces, are in place (see Done). What remains for mesoscopic
device work is, above all, scale (item 1). Until then, Kwant remains the
right tool for large transport calculations. On the physics side the
audit found three holes that the other packages fill in part and tbkit
not at all: crystal symmetry (item 6, now done), response at a finite wave vector
(item 7) and the semiclassical conductivity of a metal (item 11). It
found four more that none of them fills but a teaching package should:
exact diagonalization as the reference for mean field (item 9, now
done), spin observables in transport (item 8), lattices without a unit
cell (item 10) and the scaling fit of a localization transition
(item 12).

The items keep their numbers, because other notes cite them, so the
numbers are no longer the priority order. The order is:

1. **Item 1**, performance and scale. Still what most limits tbkit
   against Kwant (dense `System.get_eig` and `Transport.get_green` with
   an explicit `eta`). The Chebyshev time evolution, spectral function
   and optical conductivity that KITE and TBPLaS have are done.
2. **Item 4**, interactions beyond on-site Hubbard mean field. Its exact
   references come from `tbkit.manybody` (item 9, done).
3. **Item 7**, the Lindhard function and RPA. It says which orders item 4
   should look for, and in which enlarged cell.
4. **Item 8**, spin densities, spin currents and torques.
5. **Item 10**, quasicrystals and amorphous lattices.
6. **Item 2**, interoperability. The cross-checks of items 8 and 11
   get shorter once it is done, but none of them waits for it.
7. **Item 11**, semiclassical transport.
8. **Item 12**, finite-size scaling. Small. Item 5 needs it.
9. **Item 5**, disorder projected onto a flat band (research).
10. **Item 3**, time-dependent transport (niche).

Item 6, crystal symmetry, is done (see Done).

Each item lists why it matters, the current state, and a sketch of the
approach. The project conventions (validation in `error_handling.py`, 100%
line coverage, an asserted gallery example, a history entry when there is
a breakthrough to tell) apply to all of them. In items 6 to 12 and in the
additions to items 1, 2 and 4, a reference or a number followed by
*(unverified)* was written from memory: check it against the primary
source before a test or a history entry relies on it.

## 1. Performance and scale

**Why.** Many workflows (Wilson loops on fine meshes, `System.get_eig` on
large flakes, transport of 2D devices) are dense and single-threaded.

**Now.** The sparse paths (KPM, the k-d tree neighbour search,
shift-invert, `RecursiveTransport`) cover part of this. The k-space
diagonalizations are batched over k and optionally threaded
(`KSpace.set_workers`, see Done), and `tests/benchmark_kspace.py` times
them. The energies of transport and of `surface_spectral_function` are
threaded too (`set_workers`, see Done). The scattering matrix and the
scattering states factorize the sparse device, and by default every
transmission, conductance matrix, current and noise of `Transport`
comes from them (see Done). With an explicit broadening `eta`, the
same methods take only the columns of the Green's function on the lead
sites from one sparse factorization (see Done); `Transport.get_green`
and `self_energy`, which return full matrices, remain dense.
Real-space `System.get_eig` is a full dense diagonalization, by
definition `O(N^3)` (LAPACK, threaded through BLAS); the scalable paths
are `get_eig_sparse` (shift-invert near a target energy) and KPM.
Still open: all the eigenpairs in an energy window of a large flake
(spectrum slicing: several shift-invert windows, counted by Sylvester
inertia or a KPM estimate of the number of states).

**At scale.** `tbkit.kpm` now has Chebyshev time evolution, the
spectral function `A(k, E)` and the optical conductivity besides `dos`,
`ldos`, `conductivity` and `hall_conductivity`, and
`optical_conductivity` has its reactive part and Hall components (see
Done). The Crank-Nicolson steps of `Propagation` factorize a sparse
matrix (see Done). Still missing next to KITE: nonlinear optical
responses (second-order conductivities from three-index Chebyshev
moments, KITE's `conductivity_optical_nonlinear`).

## 2. Interoperability and API limits

- Import and export Kwant `Builder`s, PythTB `tb_model`s, pybinding
  models and ASE `Atoms` geometries. That makes cross-checks against these
  packages one line.
- Write Wannier90 `_hr.dat` files (not only read them), and read
  `_tb.dat`. Once `_tb.dat` is read, use its position matrix elements in
  the optical and Berry-curvature responses, which today know only the
  Wannier centres (`positions=True`).
- Export a TBmodels `Model` (and its HDF5 file) and the callable `H(k)`
  that `z2pack.hm.System` takes, so that Z2 invariants and Chern numbers
  cross-check against Z2Pack. Export a qsymm `Model`, for the
  cross-checks of item 6.
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
torus. `manybody.exact_diagonalization` already takes a
nearest-neighbour V (or any two-body interaction) on a cluster, so the
exact answer exists before the mean field does.

**Approach.**
- Hartree-Fock on a `KSpace` mesh: the density matrix per k, a
  (possibly enlarged) magnetic unit cell, and the same linear mixing as
  the real-space solver. The result is a `KSpace`, so every band and
  topology tool applies to it.
- Extended interactions: nearest-neighbour V, then a screened Coulomb
  potential, with Hartree and Fock terms on the bonds.
- Self-consistent Hartree-Poisson for gated devices, on the sparse
  scattering states (`Transport.ldos`).
- Pairing beyond on-site s-wave. `bdg.bdg_kspace` and `bdg.bdg_ham`
  accept any pairing, but the only gap equation is `bdg.s_wave_gap`
  (on-site, real space). Add the gap equation on bonds for a
  nearest-neighbour attraction, in real space and on the `KSpace` mesh
  of the first bullet. Check: on the square lattice near half filling
  the d-wave solution has the lowest free energy, extended s-wave at low
  filling (Micnas, Ranninger and Robaszkiewicz, Rev. Mod. Phys. 62, 113
  (1990), *unverified*); the density of states of the d-wave state is
  linear in energy at the nodes; with on-site attraction the k-space
  solver gives the gap of `s_wave_gap` on the torus of the same mesh.
- Electron-lattice coupling in the adiabatic limit (Peierls 1955;
  Su, Schrieffer and Heeger 1979). Hoppings `t - alpha u_ij`, an elastic
  energy `K u_ij^2 / 2`, and the bond displacements iterated to
  self-consistency with the bond density matrix that the Fock term of the
  second bullet already needs. Check: the half-filled chain dimerizes
  for any coupling, with a gap that closes as `exp(-1/lambda)` at weak
  coupling; an odd chain relaxes to a domain wall with one midgap state,
  spread over several sites, where today's SSH example imposes a sharp
  one. This is the only part of electron-phonon physics the audit found
  worth an item (see "Looked at and left out" at the end).

Each bullet is checked on a cluster small enough for the exact
diagonalization of `tbkit.manybody` (item 9).

## 5. Disorder projected onto a flat band

**Why.** The localization of a disordered flat Chern band, and how its
quantum geometry enters, is a current research question (the critical
exponent of a projected Chern band, the role of the quantum metric). It
needs disorder restricted to one band, `PVP`, whose matrix
elements couple several slices.

**Now.** `localization.lyapunov_exponents` (see Done) propagates strips
with couplings to several slices, but only with onsite disorder added to
a fixed slice Hamiltonian.

**Approach.** Maximally localized hybrid Wannier functions of one band
of a `KSpace` (parallel transport along the transfer direction, the
twist spread uniformly), white-noise disorder projected onto them, and
the coupling blocks `<w_x|V|w_{x+l}>` streamed slice
by slice into the transfer matrix. Validated against exact
diagonalization of `PVP` on a torus, and the Wannier variances
against `quantum_geometric_tensor`. The critical exponent then comes
from the scaling fit of item 12.

## 6. Crystal symmetry: operations, irreps, indicators, symmetrization

Done (unreleased): see "Crystal symmetry" under Done. Left out of this
pass, each small once needed: the double groups (irreps of spinful
models), nonsymmorphic little groups and the 3D point groups other than
the cubic ones (D4h at X of a cubic crystal, the hexagonal groups); the
corner charge of C2, C3 and C6 insulators (the C2 formula of
Benalcazar, Li and Hughes did not match flakes of the 2D SSH model at
zero polarization, and `higher_order.corner_charges` cuts quadrants,
not the thirds and sixths of C3 and C6 flakes); the models a group
allows (`qsymm.continuum_hamiltonian`, `bloch_family`), and integrals
over the irreducible wedge of the zone (WannierBerri).

## 7. Response at a finite wave vector: Lindhard function, RPA, plasmons

**Why.** Every response function in tbkit is at q = 0 (Hall, optical,
Berry-curvature responses). The susceptibility `chi_0(q, omega)` is what
says where a Fermi surface is unstable (nesting, the Stoner criterion,
the Peierls instability at 2k_F), how a charge is screened (Friedel
oscillations) and where the plasmons are. It also tells item 4 which
order to look for and which enlarged cell to give it.
`tight_binding/plot_fermi_surfaces.py` shows the nesting of the
half-filled square lattice and can only say in words what it drives.
TBPLaS computes the Lindhard function, the dielectric function and the
plasmon dispersion.

**Now.** `KSpace._eigs` on a mesh and `occupation.fermi_dirac` are all
the Lindhard sum needs. `optics.optical_conductivity` is its q = 0
limit, and `optics.joint_dos` its interband weight.
`hubbard_mean_field` gives the Stoner cross-check.

**Approach.**
- The Lindhard function
  `chi_0(q, w) = (1/N) sum_k sum_nm (f_n(k) - f_m(k+q)) |<u_n(k)|u_m(k+q)>|^2 / (w + E_n(k) - E_m(k+q) + i eta)`,
  for q on the mesh, so that `k + q` is a roll of the arrays of one
  `_eigs` call; also resolved by orbital, as the matrix a multi-orbital
  RPA needs. The overlaps take the orbital positions, as the Wilson
  loops do. Check: the closed form for the cosine band of a chain, with
  its logarithmic peak at 2k_F; the peak at (π, π) of the half-filled
  square lattice; as q goes to zero at a fixed frequency,
  `chi_0 = D q^2 / w^2` with D the Drude weight of
  `optical_conductivity` (up to its convention for D); the polarization
  of doped
  graphene at low energy (Wunsch, Stauber, Sols and Guinea, New J.
  Phys. 8, 318 (2006); Hwang and Das Sarma, Phys. Rev. B 75, 205418
  (2007); both *unverified*); TBPLaS on the same model.
- RPA. The spin susceptibility `chi_0 / (1 - U chi_0)` and the Stoner
  criterion `U_c chi_0(Q) = 1`: on the torus of the same mesh and at the
  same temperature, `hubbard_mean_field` must leave the paramagnet at
  exactly that U, at that Q. The dielectric function
  `1 - V(q) chi_0(q, w)`, the loss function and the plasmon dispersion:
  proportional to `sqrt(q)` in 2D, and to `n^(1/4)` in graphene (Hwang
  and Das Sarma 2007).
- A point impurity. The T-matrix from the lattice Green's function of
  the clean crystal, the Friedel oscillations of the local density of
  states around it, and their Fourier transform, the quasiparticle
  interference map that WannierTools computes for surface states.
  Check: `System.get_ldos` around one impurity in a large flake, and
  the vacancy limit of `disorder/plot_vacancy_zero_modes.py`.

## 8. Spin densities, spin currents and torques

**Why.** `Transport` gives charge only: `ldos`, `bond_currents`,
`local_currents`. Spin enters only when it is conserved, through the
blocks of a `conservation_law`. With spin-orbit coupling it is not
conserved, and the observables are the spin density a current induces,
the spin current, and the torque. Kwant evaluates all three for any
onsite operator (`kwant.operator.Density`, `Current`, `Source`). The
spin Hall entry of the history shows the Kubo number but not the spin
accumulation at the edges, which is what was measured in 2004. The
current-induced spin polarization (Edelstein, Solid State Commun. 73,
233 (1990), *unverified*) and the spin-transfer torque (Slonczewski,
J. Magn. Magn. Mater. 159, L1 (1996), *unverified*) cannot be told at
all.

**Now.** `Transport.wave_function` returns the scattering states, and
`ldos`, `bond_currents` and `local_currents` contract them on the
sites and on `_bond_list` with the identity. `KSpace.spin_operator`,
`spin_texture` and `spin_hall_conductivity` (the Kubo sum `_kubo`, which
already takes a `spin_op`) are the bulk side. `OrbitalSystem` builds the
spinful devices (`set_rashba`, `set_spin_orbit`, `set_zeeman`), and
`slater_koster.orbital_angular_momentum` gives L.

**Approach.**
- `Transport.density(energy, lead, op)`, `current(energy, lead, op)`
  and `source(energy, lead, op)` for an operator `op` on the orbitals
  of a site: `psi^+ M psi` per site, the current of M on each bond, and
  `psi^+ i[H, M] psi` per site, which is the torque when M is a spin.
  Check: `op = 1` returns `ldos` and `bond_currents`; the divergence of
  the current equals the source on every site, to rounding; with s_z
  conserved the spin current is the difference of the two
  `conservation_law` blocks; Kwant's three operators on the same Rashba
  strip (in the `paper/` environment).
- Two examples for the history: the antisymmetric s_z accumulation at
  the edges of a Rashba strip carrying a current (Kato et al., Science
  306, 1910 (2004); Nikolić, Souma, Zârbo and Sinova, Phys. Rev. Lett.
  95, 046601 (2005); both *unverified*), and the torque on the free
  layer of a ferromagnet/normal/ferromagnet chain, which vanishes for
  parallel and antiparallel magnetizations.
- The bulk counterpart in `KSpace`. The Edelstein susceptibility, the
  Fermi-surface integral of `<s_a> v_b` with a constant relaxation time
  (the integral of item 11 with one velocity replaced by a spin).
  Check: the Rashba model, linear in the Rashba coupling and independent
  of the Fermi level when both subbands are occupied (Edelstein 1990;
  prefactor *unverified*). Open `spin_hall_conductivity` to any
  operator, which gives the orbital Hall effect with
  `orbital_angular_momentum` (Bernevig, Hughes and Zhang, Phys. Rev.
  Lett. 95, 066601 (2005); Go, Jo, Kim and Lee, Phys. Rev. Lett. 121,
  086602 (2018); both *unverified*, and a published number to check
  against is still to be found).

## 9. Exact diagonalization of small interacting clusters

Done (unreleased): see "Exact diagonalization" under Done.

## 10. Aperiodic lattices: quasicrystals and amorphous networks

**Why.** The docstrings of `topology.bott_index` and
`System.get_local_chern_marker` say they apply to quasicrystals and
amorphous solids, and tbkit cannot build either. A `Lattice` is a unit
cell repeated, and `System.set_hopping` selects bonds by neighbour
order, which means nothing when every distance is different. TBPLaS
lists quasicrystals and fractals among its models. The history cannot
tell the Cantor spectrum of the Fibonacci chain (Kohmoto, Kadanoff and
Tang, Phys. Rev. Lett. 50, 1870 (1983); Ostlund et al., Phys. Rev.
Lett. 50, 1873 (1983); both *unverified*), the confined states of the
Penrose tiling (Kohmoto and Sutherland, Phys. Rev. Lett. 56, 2740
(1986), *unverified*), or amorphous topological insulators (Agarwala
and Shenoy, Phys. Rev. Lett. 118, 236402 (2017); Mitchell et al., Nat.
Phys. 14, 380 (2018); both *unverified*).

**Now.** `Lattice.add_sites` adds arbitrary positions, but only to a
crystal. `System._near_pairs` finds every pair closer than a radius
with a k-d tree, periodic images included. Value functions
`t(site_i, site_j)` and `set_hopping_manual` can set any bond once it
is selected. `moire.pz_hopping` is a hopping as a function of the bond
vector, used in `KSpace` only. The Bott index, the local Chern marker,
`kpm` and `Transport` take the resulting Hamiltonian as it is.

**Approach.**
- A `Lattice` from a set of points, with no unit cell (`get_lattice`
  refused), and `System.set_hopping` by distance: every bond shorter
  than a cutoff, with a number or a function of the bond vector, through
  `_near_pairs` and `_bond_vectors`, on a flake or a torus.
- Generators in `lattices.py`: the Fibonacci chain, the Penrose and
  Ammann-Beenker tilings by cut and project (sites on the vertices,
  bonds on the tile edges), and random point sets with a minimum
  distance.
- Check: on the Fibonacci chain the integrated density of states in
  every gap is `p + q/tau` modulo 1 with integers p and q (the
  gap-labelling theorem). On the Penrose vertex model the fraction of
  zero-energy states tends to `81 - 50 tau`, about 9.8% (Kohmoto and
  Sutherland 1986; Arai, Tokihiro, Fujiwara and Kohmoto 1988; value
  *unverified*). On a random point set with the hoppings of Agarwala
  and Shenoy the Bott index is 1, it equals the mean local Chern
  marker, and `Transport` gives one quantized edge channel.

## 11. Semiclassical transport: conductivity, thermopower, Hall coefficient

**Why.** The first transport calculation of a solid-state course is
`sigma = e^2 tau <v v>` on the Fermi surface, then the thermopower from
the Mott formula, the Hall coefficient `1/ne` with its sign change
between electron and hole surfaces, and the magnetoresistance. tbkit
has the intrinsic Berry-curvature responses and phase-coherent Landauer
transport, and between them only the Drude term inside
`optics.optical_conductivity`, in 2D. WannierBerri has
`conductivity_Ohmic`, and WannierTools the ordinary magnetoresistance.
The response page of the history starts in 1954 with the anomalous
velocity: Bloch's conductivity, the Mott formula and the
magnetoresistance of open orbits are missing.

**Now.** `KSpace._bloch_derivatives` gives the band velocities and
inverse masses analytically. `dos._mesh_simplices` and
`tetrahedron_dos` integrate over a constant-energy surface, and
`fermi_surface` returns it. `Transport.thermoelectric` computes the
same three Onsager integrals for a Landauer transmission
(`_onsager`).

**Approach.**
- The transport distribution
  `Sigma_ab(E) = sum_n int v_a v_b delta(E - E_n(k))` by the
  tetrahedron method, in 2D and 3D, then the conductivity, the
  thermopower and the electronic thermal conductivity at a constant
  relaxation time from its three moments. Check: `n e^2 tau / m*` at
  the bottom of a square or cubic band; the Mott formula and the
  Wiedemann-Franz ratio at low temperature; the Drude weight of
  `optical_conductivity` in 2D; `conductivity_Ohmic` of WannierBerri on
  one `_hr.dat` model read by both.
- The weak-field Hall conductivity, of order `tau^2 B`, and the Hall
  coefficient. Check: `-1/ne` at low filling of the square lattice, a
  sign change at half filling and `+1/n_h e` near the top of the band;
  in 2D, the area swept by the mean-free-path vector around the Fermi
  line (Ong, Phys. Rev. B 43, 193 (1991), *unverified*).
- The magnetoresistance at any field, by integrating the orbits on the
  Fermi surface (Chambers 1952; Zhang, Wu, Liu et al., Phys. Rev. B 99,
  035142 (2019), the method of WannierTools; both *unverified*). Check:
  none for one isotropic band; quadratic and unsaturated for a
  compensated two-band model; saturating or not with the direction of
  the field when the surface has open orbits; WannierTools on the same
  model.

## 12. Finite-size scaling of a localization transition

**Why.** `lyapunov_exponents` returns `Lambda = lambda_M / M` with its
error, and the two examples show the curves of two sizes crossing. The
numbers a paper reports are the critical disorder and the exponent ν
with their errors, and they come from a fit to a scaling function with
corrections, which is where the method is hard. The 1981 entry of the
history quotes ν ≈ 1.57 and cannot compute it. Item 5 asks
for the exponent of a projected Chern band, so it needs this fit.

**Now.** `localization.lyapunov_exponents` (several energies on one
strip, errors from pieces of the strip). The mean ratio of consecutive
level spacings is a function inside
`examples/three_dimensions/plot_anderson_transition.py`, not in the
library. `System.get_ipr`.

**Approach.** `localization.scaling_fit`: Λ as a polynomial in the
relevant variable `(W - W_c) M^(1/nu)` and in one irrelevant variable
`M^y`, as Slevin and Ohtsuki (1999) do, fitted by least squares with
the errors of the data, with the goodness of fit and bootstrap
intervals for W_c, ν and y. Move the level-spacing ratio into
`localization`, so the same fit applies to it. Check: data drawn from a
known scaling function give back its W_c and ν within the intervals;
the cubic Anderson model with box disorder at E = 0 gives W_c ≈ 16.5
and ν ≈ 1.57 (Slevin and Ohtsuki 1999, cited on the history page; the
digits beyond these are *unverified*). The example states the cross
sections and the strip lengths it could afford and the interval it
obtained, and asserts only that.

## Done

- **Crystal symmetry** (unreleased, item 6): `tbkit.symmetry`.
  `Operation` (rotation or reflection, translation, orbital matrix,
  antiunitary flag; `Operation.about` a centre, products and powers)
  and its representation `U_g(k)` on the Bloch basis (the permutation of
  the orbitals, with the phase of the lattice vector that brings each
  image back into the cell, and the spin-1/2 rotation with `spin=True`);
  `KSpace.symmetry_error(op)` takes one. `find_symmetries` tries the
  operations of the Bravais lattice with the translations that map the
  positions onto themselves, about a common centre when the structure is
  symmorphic. `irreps`: the little group, the characters of each level
  and their decomposition on the tables of the ten 2D point groups and of
  T, Th, O, Td, Oh (spinless, symmorphic); `plot_bands(irreps=)`.
  `rotation_indicator` (C mod n from the C_n eigenvalues, Fang, Gilbert
  and Bernevig 2012), `inversion_indicators` (z4 and the weak z2, Khalaf,
  Po, Vishwanath and Watanabe 2018), `corner_charge` (C4, Benalcazar, Li
  and Hughes 2019); `symmetrize` (group average of `H(R)`, as TBmodels);
  `higher_order.corner_charges(..., background=)`. Checked: 12 unitary
  operations for graphene (24 with time reversal), 6 + 6 with a
  sublattice mass and with the Haldane flux, 24 for Kane-Mele with or
  without Rashba, the same counts from qsymm 1.4 for graphene, mass and
  Haldane; `H(gk) = U H(k) U^dagger` for each; the Dirac point is E of
  C3v and splits into ¹E, ²E of C3 with the mass; compatibility of the
  characters along Γ-M-K-Γ; s and p of a cubic crystal A1g and T1u at Γ;
  orthonormal tables; C mod 4, 2 of Qi-Wu-Zhang and C mod 3, 6 of
  Haldane against `chern_number`; `(z4 mod 2; z2)` against
  `z2_indices_3d` in four phases, and z4 = 2 for two strong copies whose
  Z2 indices vanish; the corner charge 1/4, 0, 1/2 against the quadrants
  of 8x8 flakes; `symmetrize` leaves symmetric models unchanged, makes a
  noisy `read_wannier90` file exactly degenerate at K with zero
  `symmetry_error`, and equals TBmodels 1.4 to 1e-15. Examples
  `tight_binding/plot_irreps_of_graphene_bands.py` (history entry 1936
  in `models_breakthroughs.rst`), `tight_binding/plot_symmetrizing_a_wannier90_model.py`,
  `topology/plot_chern_number_from_rotation_eigenvalues.py` (entry 2012)
  and `topology/plot_symmetry_indicators.py` with
  `higher_order/plot_corner_charge_from_rotation_eigenvalues.py` (entry
  2017, symmetry indicators and topological quantum chemistry, in
  `topology_breakthroughs.rst`). The references were written from memory
  and are not yet checked against the publishers' records.

- **Sparse broadened transport, sparse Crank-Nicolson, complex KPM
  optical conductivity** (unreleased, item 1). `Transport`: with an
  explicit `eta`, the Caroli transmissions, transmission eigenvalues,
  conductance matrix and bond currents take the columns of `G^r` on the
  lead sites from one `splu` of `E + i eta - H - sum Sigma`
  (`Transport._green_columns`); checked against the dense `get_green` to
  1e-10, and `Transport.ham` is never built. `Propagation`: one `splu`
  per Hamiltonian instead of a dense inverse, checked against the dense
  step to 1e-12 (also non-Hermitian and along a pumping cycle); 100 steps
  on 22,500 sites in 0.3 s. `kpm.optical_conductivity(..., vel_b=,
  part=)`: Hall components and the reactive part, a principal value over
  the transition spectral function `S_ab(nu)`; on a 12x12 Haldane torus
  with the exact trace, `Re sigma_yx(omega -> 0)` is `C` to 1% and
  `Re sigma_yx`, `Im sigma_xx` below the gap match
  `optics.optical_conductivity` within 1-2%. Example
  `large_scale/plot_kpm_optical_hall_conductivity.py` (7,200 sites,
  clean and disordered).

- **Chebyshev time evolution, spectral function, optical conductivity**
  (unreleased, item 1): `kpm.time_evolution` (Bessel coefficients,
  truncated at `tol`; Tal-Ezer and Kosloff, J. Chem. Phys. 81, 3967
  (1984), checked against the publisher's record), also
  `Propagation.get_propagation(..., method='chebyshev')`;
  `kpm.spectral_function` (plane wave per sublattice, moments in pairs);
  `kpm.optical_conductivity` (`Re sigma_xx`, Kubo-Greenwood with the
  two-index moments of `conductivity`, any Fermi level and temperature).
  Checked: `expm` to 1e-12 and Crank-Nicolson to its O(dt^2) error; the
  chain's `<x^2> = 2 t^2 tau^2` to 1e-10; `moire.unfold` on a 6x6
  disordered supercell to 1e-10; the exact Chebyshev-broadened
  Kubo-Greenwood sum to 1e-3 (the energy quadrature); the universal
  `pi/4 e^2/h` and `optics.optical_conductivity` within 3-5% on a
  45,000-site torus (16 random vectors; 4 give about 10% of noise).
  Examples `large_scale/plot_chebyshev_time_evolution.py` (history
  entry 1984 in `models_breakthroughs.rst`, now 17),
  `large_scale/plot_kpm_spectral_function.py`,
  `large_scale/plot_kpm_optical_conductivity.py`.

- **Exact diagonalization** (unreleased, item 9): `tbkit.manybody`.
  `FockSector` holds the basis of a sector (N_up, N_down), or of spinless
  fermions, as bit patterns; `exact_diagonalization(ham, n_up, n_dn, U,
  V)` builds the sparse Hamiltonian with one vectorized pass per bond
  (fermion signs by popcount, no loop over configurations) and returns
  the lowest states (dense below 1500 states, else `eigsh`; 12 sites at
  half filling in about 2 s), with densities, double occupancy, density
  and spin correlations and the total spin. For flat bands,
  `project_interaction` projects V onto Bloch states, and
  `exact_diagonalization(..., W=, momentum=)` takes the resulting
  two-body interaction in one sector of total momentum.
  `KSpace.finite_ham(..., twist=)` gives twisted boundary conditions.
  Checked: the dimer energy and `4t^2/U`; U = 0 against the filled
  levels; a brute-force Jordan-Wigner build; the Nagaoka example, whose
  inline Fock-space code it replaced. Examples:
  `correlations/plot_exact_diagonalization_vs_mean_field.py` (the
  energy and local moment of `hubbard_mean_field` against the exact ones
  on an 8-site ring, the Heisenberg limit), and two history entries in
  `interactions_breakthroughs.rst` (now 14):
  `correlations/plot_lieb_wu_hubbard_chain.py` (1968; rings of up to 12
  sites against the Lieb-Wu energy and Mott gap, which they reproduce
  within 0.5% once extrapolated) and `correlations/plot_fractional_chern_insulator.py`
  (2011; checkerboard flat band at filling 1/3 on a 4x6 torus, three
  ground states at momenta (0, 0), (0, 2), (0, 4) and their spectral
  flow under flux insertion). The references of both entries were
  checked against the publishers' records.

- **Transfer-matrix localization length** (unreleased):
  `localization.lyapunov_exponents(h0, v, energies, disorder, n_slices)`
  returns the Lyapunov exponents and the localization length of a long
  strip with box disorder (MacKinnon and Kramer 1981), with errors from
  pieces of the strip, several energies on the same strip, and couplings
  to more than one slice. Slices are drawn and discarded, never stored.
  Example: `disorder/plot_transfer_matrix_localization_length.py`
  (Thouless's formula, and the crossing of the 3D Anderson transition);
  history entry 1981 in `fields_breakthroughs.rst`.

- **Longer sublattice tags** (unreleased): tags of up to 8 characters
  (`'Mo'`, `'Se1'`) in `Lattice.coor` (`'U8'`) and `System.hop`
  (`'U16'`). Bond selectors take a pair of tags, `('Mo', 'S')`, matched
  on the tags of the two sites, so concatenations cannot collide;
  `'ab'` strings, `'U1'` arrays for `add_sites` and old archives keep
  working.

- **Blocking type check and live releases** (unreleased): `mypy tbkit`
  passes (the NumPy >= 2.3 stub errors were fixed module by module), so
  the `typecheck` job no longer has `continue-on-error`. The release
  workflow is active: PyPI trusted publishing, the `pypi` environment and
  Zenodo, which published 0.7.0 with its DOI (in `CITATION.cff` and the
  README).

- **38-fold non-Hermitian symmetry classes history breakthrough**
  (0.7.0): `KSpace.non_hermitian_class` names the class of
  Kawabata, Shiozaki, Ueda and Sato (2019) from up to seven operators
  (TRS, PHS, CS and their dagger partners, SLS, pseudo-Hermiticity),
  closed under products, as `tenfold_class` does for the Hermitian
  classes; `canonical=True` folds the 54 names of their tables into the
  38 classes. `symmetry_error(..., transpose=True)` checks symmetries
  with H^T or H^dagger. The 2018/2019 entry in `driven_breakthroughs.rst`
  (now 12) has its asserted example `non_hermitian/plot_38_fold_way.py`.
  This was the last candidate of the former item 6 (history
  breakthroughs); interactions and driven have 12 entries each.

- **Flat-band superconductivity history breakthrough** (0.7.0):
  `bdg.superfluid_weight` (curvature of the BdG grand potential in a
  vector potential) and the 2015 Peotta-Törmä entry in
  `interactions_breakthroughs.rst` (now 12), with its asserted example
  `superconductivity/plot_flat_band_superconductivity.py`. The passing
  mention in the 1980 quantum-metric entry moved into it.

- **The rest of `Transport` on the sparse factorization** (0.7.0):
  `transmission_matrix`, `conductance_matrix`,
  `four_terminal_resistance`, `bond_currents`, `local_currents`,
  `transmission_eigenvalues`, `shot_noise` and `fano_factor` default to
  `eta=None`: S-matrix blocks and scattering states from one `splu` per
  energy, never the dense device. An explicit `eta` keeps the dense
  Caroli path.

- **Batched per-k loops** (0.7.0): `biorthogonal_berry_curvature`
  (one stacked `eig` and link determinants over the mesh),
  `quantum_geometric_tensor` (one batched diagonalization, now also over
  many k-points) and the Floquet `get_ham` (`DrivenKSpace._hams`: all
  k-points evolved together, one stacked exponential per step).

- **Threads over energies** (0.7.0): `Transport.set_workers` and
  `RecursiveTransport.set_workers` run the energies of `transmission`,
  `shot_noise`, `fano_factor` and the Fermi-window integrals in a
  `ThreadPoolExecutor`; `surface_spectral_function` spreads its k-points
  (and chunks of energies) over the `KSpace` workers.

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
| Model building | Real-space `System` and Bloch `KSpace`, with bridges between them, value functions (`get_ham(**params)`) and k·p discretization | Graph `Builder` with symmetries and value functions; k·p discretization, since 1.5 also in a Landau-level basis (`discretize_landau`) | `TBModel` (2.0; `tb_model` kept as an alias), any periodic and real-space dimensions, parametrized terms, predefined models (`pythtb.models`) |
| Transport | S-matrix from exact lead modes (sparse), automatic lead attachment (completing ragged interfaces), conservation-law blocks, scattering states and LDOS, finite temperature and thermoelectrics; Landauer, multi-terminal, currents and shot noise, on the sparse S-matrix by default since 0.7.0 (dense with an explicit `eta`); recursive Green's function. Charge only: no spin densities, spin currents or torques (item 8) | S-matrix, lead modes, wavefunctions, conservation laws, sparse solver; densities, currents and sources of any onsite operator (`kwant.operator`); Peierls phases with leads (`magnetic_gauge`, `add_peierls_phase`) | None |
| Band topology | Chern, spin and mirror Chern, Wilson loops, 2D Z2 and the four 3D indices, Fu-Kane parities, Majorana number, Weyl points, quantum geometry, tenfold way, higher order, local Chern marker, Bott index, entanglement spectrum | Minimal (via qsymm) | Berry phases, Wilson loops, hybrid Wannier centres; since 2.0 also Chern numbers, Berry curvature, quantum metric, local Chern marker, axion angle, second Chern number, Wannier functions with disentanglement |
| Crystal symmetry | Space-group operations and their Bloch representation, a symmetry finder, irreps at high-symmetry points (2D and cubic point groups, spinless, symmorphic), rotation and inversion indicators, the C4 corner charge, symmetrization (`tbkit.symmetry`); no generator of symmetric models, no double groups (item 6) | qsymm: finds the symmetries of a model, generates the models a group allows | None |
| Surface states | Ribbons and slabs, semi-infinite surface spectral function | Leads and `kwant.physics` | Finite slabs (`cut_piece`) |
| Response | Anomalous and spin Hall, orbital magnetization, anomalous Nernst and thermal Hall, axion angle, Kubo-Bastin, optics, Berry curvature dipole, shift current | Kubo conductivity (KPM) | Berry curvature and velocities, no conductivity routines |
| Non-Hermitian, Floquet | Extensive | None | None |
| Scale | NumPy/SciPy, mostly dense; k-meshes batched and threaded; sparse LU for the S-matrix | Cython and MUMPS, about 10^6 sites | Vectorized NumPy since 2.0 |

Measured on one thread (`paper/benchmarks/run_benchmarks.py`, Kwant 1.5.0,
PythTB 2.0.2): on k-meshes tbkit and PythTB 2 are within a factor of two of
each other, tbkit using 5-14x less memory; for the scattering matrix Kwant
is faster from about 10^3 sites, and ten times faster at 4x10^5 sites.
Kwant 1.5.0 (June 2024) and PythTB 2.0.2 were still the latest releases
on 2026-11-15.

### The other packages

What each has that tbkit lacks, from their feature lists as read on
2026-11-15.

- **tkwant** 1.1.0 (June 2023): time-dependent one-body and many-body
  states of an open device, and a time-dependent mean field. Item 3.
- **qsymm** 1.4.0 (December 2023): the generator of symmetric k·p and
  Bloch models (its symmetry finder is in `tbkit.symmetry`, item 6).
- **pybinding** 0.9.5 (2020), continued as pybinding-dev 1.0.6 (March
  2025): KPM spectral densities of any operator (item 1), and a material
  repository (graphene, phosphorene, group 6 TMDs; left out below).
- **KITE** 1.1 (July 2022): `arpes`, `conductivity_optical`,
  `conductivity_optical_nonlinear` and `gaussian_wave_packet` by
  Chebyshev expansion, on 10^9 and more sites. Item 1.
- **TBPLaS** 2.0.1 (July 2025): the Lindhard function, the dielectric
  function and plasmons (item 7), time propagation at scale and GPUs
  (item 1), quasicrystals and fractals (item 10).
- **WannierTools** 2.6.2: the ordinary magnetoresistance (item 11) and
  the quasiparticle interference of surface states (item 7). Its bulk,
  slab and ribbon bands, surface spectral function, Fermi arcs, Wilson
  loops, Z2, Chern and mirror Chern numbers, Weyl chiralities and
  anomalous Hall conductivity are all in tbkit. Its search for nodes
  also finds nodal lines; `find_weyl_points` finds points only.
- **WannierBerri** (version not confirmed): `conductivity_Ohmic`
  (item 11) and symmetry-reduced k-meshes (left out of item 6); the gyrotropic
  responses and the frequency-dependent spin Hall conductivity (left
  out below).
- **TBmodels** 1.4: its HDF5 format (item 2; `symmetrize` is in
  `tbkit.symmetry`, item 6), and
  `construct_kdotp`, the inverse of `continuum.discretize` (left out).
- **Z2Pack** 2.2.1 (February 2025): nothing missing. `wannier_flow`,
  `z2_invariant`, `z2_indices_3d` and `chern_number` cover it; item 2
  adds the export for cross-checks.
- **sisl** 0.20.0 (October 2026): post-processing of DFT (Siesta)
  Hamiltonians and of TBtrans output. A different scope; nothing taken.

### Looked at and left out

Checked on 2026-11-15 and not given an item, so that the next audit does
not start from them again.

- **Already computable, missing only an example or a history entry.**
  Quantum oscillations (`magnetic_supercell` and the density of states
  against 1/B, with the Onsager area from `fermi_surface`); d-wave and
  chiral p-wave superconductors (`bdg_kspace` takes any pairing, and
  `chern_number` applies to the result); the current-phase relation of a
  Josephson junction (the phase derivative of the `bdg_ham` spectrum,
  which the BdG example mentions and does not plot); weak localization
  and universal conductance fluctuations (`Transport` with disorder);
  the phonon bands of a network of springs (a `KSpace` whose eigenvalues
  are ω²). These are candidates for history entries, not missing
  functionality.
- **Phonons from force constants and electron-phonon scattering.** The
  dynamical matrix is the case above. Scattering rates and the
  Eliashberg function belong to first-principles codes. The adiabatic
  Peierls distortion went into item 4.
- **Magnons and spin models.** Linear spin-wave theory needs a
  classical ground state of a spin Hamiltonian and a bosonic Bogoliubov
  transformation. Neither is in tbkit and no package compared here has
  them (SpinW and Sunny do). A separate domain: undecided.
- **A plane-wave continuum model of moiré bands**
  (Bistritzer-MacDonald). `moire.twisted_bilayer` is atomistic: 11908
  orbitals at 1.05°, 2.3 GB per Bloch matrix, so the magic-angle example
  rescales the interlayer hopping at 6°. Item 4 names moiré bands as a
  motivation and would need this to reach them. Undecided.
- **Nonlinear propagation** (the discrete nonlinear Schrödinger
  equation, for the discrete solitons the waveguide-array entry
  mentions). A small addition to `Propagation`; photonics only.
- **A library of named models** (Haldane, Kane-Mele, BHZ, the three-band
  TMD model). PythTB 2, pybinding and TBPLaS ship one. A convenience,
  not a capability.
- **More nonlinear and gyrotropic responses** (injection current, second
  harmonic generation, the gyrotropic magnetic effect), and
  `optics` in 3D. Research tools with few users; the shift current and
  the Berry curvature dipole are in.
- **Nodal lines, DMFT and impurity solvers, excitons (Bethe-Salpeter),
  Lindblad dynamics.** Outside the single-particle and mean-field scope.
