# Changelog

## Unreleased

### Added

New physics, each with tests, a gallery example that asserts its claims, and
an entry in `docs/source/history.rst`:

- **Three dimensions.** `Lattice` accepts 3D positions and three primitive
  vectors (`get_lattice(n1, n2, n3)`, `slab`, `shift_z`, `change_sign_z`,
  `space_dim`); `System` finds 3D neighbours; `KSpace` takes
  three-component `'R'`, `chern_number(..., plane=, k_fixed=)` works on any
  plane of a 3D Brillouin zone, and `ribbon()` cuts slabs. Examples:
  `three_dimensions/plot_anderson_transition.py` (the 3D metal-insulator
  transition from level statistics) and
  `three_dimensions/plot_weyl_semimetal.py` (Weyl points and Fermi arcs).
- **Topology tools** on `KSpace`: `berry_phase` (Zak phase),
  `wannier_centers`, `wannier_flow`, `z2_invariant` (Soluyanov-Vanderbilt
  Wilson-loop flow), `parity_z2` (Fu-Kane), `orbital_positions`,
  `quantum_geometric_tensor`, `symmetry_error`, `tenfold_class`; the
  real-space `System.get_local_chern_marker` (Bianco-Resta); and
  `kspace.magnetic_supercell` for rational fluxes `p/q` (TKNN). Examples in
  `topology/`: `plot_zak_phase`, `plot_kane_mele_z2`, `plot_fu_kane_parity`,
  `plot_tknn_hofstadter`, `plot_quantum_geometry`, `plot_tenfold_way`,
  `plot_local_chern_marker`; and `magnetic_field/plot_peierls_substitution`.
- **Non-Hermitian bands.** `KSpace.set_hopping(..., hermitian=False)` for
  non-reciprocal hoppings, `is_hermitian`, `finite_ham` (open or periodic
  finite samples), `get_ham_beta`, `spectral_winding`, and `gbz` (the
  generalized Brillouin zone). Example: `non_hermitian/plot_skin_effect.py`.
- **Green's functions and occupations.** `System.get_green`, `get_ldos`,
  `get_fermi_level`, `get_occupations`, `get_charge_density`,
  `get_total_energy`, `KSpace.get_fermi_level`, and the new
  `tbkit.occupation` (`fermi_dirac`, `fermi_level`).
- **Transport.** New `tbkit.transport`: `surface_green` (Sancho-Rubio
  decimation), `lead_from_kspace`, and `Transport` (leads, self-energies,
  Green's function, Caroli/Landauer transmission). Example:
  `transport/plot_landauer_conductance.py` (quantized point-contact steps).
- **Large lattices.** `System` finds neighbours with a k-d tree above
  `System.dense_max` (5000) sites; `System.get_eig_sparse` (shift-invert);
  new `tbkit.kpm` (kernel polynomial method: `dos`, `ldos`, `conductivity`,
  `dos_from_levels`, Jackson and Lorentz kernels). Example:
  `large_scale/plot_kernel_polynomial_method.py`.
- **Several orbitals per site.** New `tbkit.slater_koster` (`sk_block`
  for s, p and d orbitals, `sk_kspace`, `orbital_angular_momentum`),
  overlap matrices (`KSpace.set_overlap`, generalized eigenproblems, Loewdin
  orthogonalization for the topology tools), and the new
  `tbkit.orbital.OrbitalSystem` for spinful multi-orbital real-space models
  (Zeeman, atomic and Kane-Mele spin-orbit, Rashba, Peierls phases,
  Slater-Koster hoppings). Example: `orbitals/plot_slater_koster.py`.
- **Interactions.** New `tbkit.meanfield.hubbard_mean_field` (collinear
  unrestricted Hartree-Fock). Examples:
  `correlations/plot_hubbard_edge_magnetism.py` and
  `correlations/plot_lieb_theorem.py`.
- **Superconductivity.** New `tbkit.bdg` (`bdg_ham`, `pairing_bonds`,
  `pairing_s_wave`, `bdg_kspace`, `particle_hole`). Example:
  `superconductivity/plot_kitaev_chain.py`.
- **Floquet.** New `tbkit.floquet` (`evolution_operator`, `quasienergies`,
  `effective_hamiltonian`, `harmonics`, `sambe_hamiltonian`, and
  `FloquetKSpace` for driven Bloch models); `KSpace.get_ham_peierls`.
  Example: `floquet/plot_floquet_chern_insulator.py`.
- **Anomalous Floquet phases.** `floquet.DrivenKSpace` (a driven Bloch
  model from any `H(k, t)`; `FloquetKSpace` is now its subclass),
  `floquet.step_drive` (piecewise-constant drives of `KSpace` models,
  ribbons, `System` flakes or matrices, evolved exactly as a product of
  step exponentials; `floquet.StepDrive` in real space),
  `DrivenKSpace.winding_number` (the Rudner-Lindner-Berg-Levin invariant,
  spectrally discretized, with its tolerance documented) and
  `DrivenKSpace.edge_state_count` (chiral states of a driven ribbon, per
  edge). `effective_hamiltonian`, `quasienergies` and `FloquetKSpace` take
  an `epsilon` branch cut; the default (`None`) computes exactly what it
  did before (pinned by a test). Example:
  `floquet/plot_anomalous_floquet_phases.py`.
- **Hall conductivities at any Fermi level.** `KSpace.hall_conductivity`:
  the intrinsic anomalous Hall conductivity from the Kubo formula (the
  finite-at-degeneracies form, with `dH/dk` built analytically from the
  hoppings), in a gap or in a band, at zero or finite temperature, in 2D
  (units `e^2/h`), on a plane of a 3D model, or as the full 3D Hall vector
  (`e^2/(h length)`); `positions=True` (bond vectors `R + tau_j - tau_i`,
  the physical velocity) or `False` (periodic gauge), which agree for full
  bands only; Loewdin orthogonalization with an overlap; optional adaptive
  mesh refinement (Wang, Yates, Souza and Vanderbilt 2006). In a gap it
  equals `chern_number` (TKNN's sign; the Ohm's-law `sigma_xy` is its
  negative, as stated in the docstring). `KSpace.spin_hall_conductivity`:
  the spin current `{s, v_x}/2`, in units of `e/2pi`. `KSpace.finite_velocity`
  (velocity operators from the bond vectors, valid on a torus) and
  `finite_ham(..., sparse=True)`. `tbkit.kpm.hall_conductivity`: the
  Kubo-Bastin Hall conductivity by the kernel polynomial method (Garcia,
  Covaci and Rappoport 2015), its antisymmetric (Hall) part, at any Fermi
  level and temperature. Examples: `hall_effects/plot_anomalous_hall_effect.py`,
  `hall_effects/plot_anomalous_hall_disorder.py`,
  `hall_effects/plot_intrinsic_spin_hall_effect.py`.
- **Exceptional points of 2D bands.** New `tbkit.exceptional`, for a 2D
  `KSpace` or any callable `H(p)` of a parameter plane:
  - `track_eigenvalues` follows the eigenvalues around a loop by
    continuation (adaptive steps, never sorting), and `vorticity` gives the
    eigenvalue vorticity of Shen, Zhen and Fu (+-1/2 at an EP, 0 at a DP).
  - `discriminant` computes `prod (E_m - E_n)^2` from the resultant of the
    characteristic polynomial, without eigenvalues, and
    `discriminant_winding` its winding (`W = -2 sum nu`).
  - `find_exceptional_points` returns an `ExceptionalPoints` result:
    plaquette windings refined by Newton's method, charges, Jordan-block
    orders from the Petermann factor, exceptional rings when the
    discriminant is real, and `total_charge` (zero by the doubling
    theorem).
  - `fermi_arcs`, `petermann_factors` and `circle`.
  - `encircle` returns an `Encircling` result: biorthogonal parallel
    transport, the state swap and the four-loop return at an EP, Berry
    phase pi at a DP.

  `KSpace.biorthogonal_berry_curvature` and `biorthogonal_chern_number`
  give the LR, RL, RR and LL Chern numbers of a band separated by a real or
  imaginary line gap. Examples: `topology/plot_diabolical_points.py`,
  `non_hermitian/plot_exceptional_points_fermi_arcs.py`,
  `non_hermitian/plot_exceptional_point_vorticity.py`,
  `non_hermitian/plot_exceptional_point_encircling.py` and
  `non_hermitian/plot_exceptional_points_chern_number.py`.
- `tight_binding/plot_isolation_of_graphene.py`, so the 2004 graphene entry
  of the history has an example of its own.
- `tbkit` exports `OrbitalSystem`, `Transport`, `FloquetKSpace`,
  `DrivenKSpace`, `StepDrive`, `step_drive`,
  `MeanFieldResult`, `hubbard_mean_field`, `ExceptionalPoints`,
  `Encircling` and `find_exceptional_points`; the API reference, tutorial,
  index and README cover the new modules.
- History entries: Landauer (1957/1988), Hubbard (1963), scaling theory of
  localization (1979), quantum metric (1980), Zak phase (1989), skin effect
  (1996/2018), tenfold way (1997), Kitaev chain (2001), kernel polynomial
  method (2006), Fu-Kane (2007), Floquet topological insulators
  (2009/2011), anomalous Floquet phases (2010-2013), anomalous Hall effect
  (1954-2004), intrinsic spin Hall effect (2003/2004), local Chern marker (2011), Weyl semimetals (2011/2015),
  diabolical points (1984), exceptional points in momentum space
  (2015-2018). The
  Peierls, Slater-Koster, TKNN and 2004 graphene entries now link examples
  of their own, so no example is shared between two entries.

### Fixed

Bugs that produced silently wrong numbers:

- `System.get_ham()` built a non-Hermitian Hamiltonian with half of its bonds
  missing whenever the sites were not sorted by `(y, x)`: after
  `Lattice.rotation()` (by 90 or -45 degrees, for example), `clean_coor()`, or
  `lat += other`. `set_hopping` took "`i < j`" to mean "bond angle in
  `[0, 180)`", which only sorted sites guarantee. Once the stored bonds had
  mixed-sign angles, `get_ham` treated them as a deliberate non-Hermitian
  upper + lower fill and skipped the Hermitian conjugate. A 3x3 honeycomb flake
  rotated by 90 degrees got 21 nonzero entries instead of 42. Bonds are now
  oriented by their angle (new private `System.get_bonds`), and the
  "replace a previous hopping" masks test the angle sign instead of `i < j`.
  Sorted lattices, including every lattice straight from `get_lattice`, get the
  same bonds, in the same order, as before.
- `GrapheneSystem.set_hop_linear_strain()` had the same site-order dependence.
  It also oriented each bond "outwards" by matching the angles 30 and 150
  degrees, so a rotated flake got a non-Hermitian Hamiltonian with the wrong
  strain pattern. Bonds are now oriented from their `'b'` site to their `'a'`
  site, which is what the angle matching did for an unrotated flake. A circular
  flake rotated by 120 degrees now has exactly the spectrum of the unrotated one.
- `KSpace.berry_curvature()` / `chern_number()` flipped sign for a left-handed
  `prim_vec` (with `a1 x a2 < 0`), because the plaquette loop then runs
  clockwise. The same Haldane model gave `C = +1` or `C = -1` depending only on
  the order of its two primitive vectors. The flux is now multiplied by the
  orientation of `prim_vec`.
- The Haldane gallery example, and the Kane-Mele term in the edge-state
  example and in the tutorial, put the complex second-neighbour hopping on
  the vectors `a1, a2, a1 - a2`. That set is not related by 120-degree
  rotations, so it breaks the lattice's C3 symmetry and opens a gap at K and
  K' three times smaller than the model's: `2*sqrt(3)*t2` instead of Haldane's
  `6*sqrt(3)*t2`. The example then "pinned down numerically" a critical mass of
  `sqrt(3)*t2*|sin(phi)|` in place of Haldane's `3*sqrt(3)*t2*|sin(phi)|`. The
  examples now use `a2, -a1, a1 - a2`, and the Haldane example asserts the gaps
  `2|M +- 3 sqrt(3) t2|` at K and K'.
- `KSpace.set_onsite()` accepted complex onsite energies (and non-Hermitian 2x2
  blocks), but `get_bands` diagonalized with `eigvalsh`, which silently dropped
  the imaginary part: a chain with onsite `1j` gave `E(k=0) = 2` instead of
  `2 + 1j`. A non-Hermitian `H(k)` now goes to the general eigensolver:
  `get_bands` returns complex energies sorted by real part, `berry_curvature`
  uses the right eigenvectors, and `plot_bands` plots the real part. Hermitian
  models still get real energies from `eigh` (new `KSpace.is_hermitian()`).
- `System.set_hopping()` with both `'ang'` and `'tag'` selected bonds within
  1 degree, but sized the result with the `1e-3` degree tolerance, so it crashed
  when two bonds of the same order were less than a degree apart. Both now
  use `ATOL`.
- `System.set_hopping_def()` silently ignored a key given as `(j, i)` when
  `sys.hop` stores the bond as `(i, j)`, which is always the case for a
  Hermitian model built by `set_hopping`. It now sets `H_ji = val` (that is,
  `t_ij = conj(val)`). A pair that is not a hopping at all is still ignored.
- `Propagation.get_pumping()` with `steps < len(hams) + 1` gave every stage zero
  steps, and its last loop overwrote the initial state `prop[:, 0]` with an
  uninitialized column (`nan` with `norm=True`). The initial state is now kept,
  and the state evolves under `hams[-1]`. With enough steps the result is
  unchanged.
- `Propagation.get_animation_nb(prop_type='imag')` animated the real part. It
  also scaled the colormap by the last frame only, so earlier frames saturated.
- `Propagation.get_animation()` took its axis limits from the first and last
  site. Those are the extreme `y` of a `(y, x)`-sorted lattice, not the extreme
  `x`, so sites of a hexagonal flake fell outside the frame.
- `Plot.spectrum()` / `spectrum_complex()` defaulted to the y-range
  `[-max(E), max(E)]`. An all-negative spectrum got an inverted range with
  every point off screen. The range now spans `[min, max]`.
- `System.get_intensity_en()` compared complex energies with the limits instead
  of their real part (the same fix 0.3.0 made in `Plot`).

Broken on correct usage:

- `error_handling.hop_sites` compared `max(i)` with `max(i)`, never checking
  `j`, and tested `sites < max` instead of `sites <= max`. After
  `remove_sites`, stale hoppings reached scipy's `index 8 exceeds matrix
  dimension 8` instead of the intended message, which also named a method that
  does not exist (`clean_hopping`, not `clear_hopping`).
- `KSpace.plot_dos()` (through `mesh_bands`) overwrote the band structure that
  `k_path` had stored, so a later `plot_bands()` crashed on mismatched shapes.
  `get_bands()` never set the distances that `plot_bands()` plots against,
  despite `plot_bands` documenting it as a valid source. `mesh_bands` no longer
  touches the stored bands, and `get_bands` stores the cumulative k-distance.
- `Plot.butterfly()` read `sys.betas` instead of its own `betas` argument, so it
  failed for anything but a `GrapheneSystem` that had run `get_butterfly`. Its
  default title was also overwritten by the empty `title=''`.
- `Plot.intensity_area()` ignored `figsize`, and `Plot.lattice(plt_hop=True)`
  drew each bond with linewidth `c*Re(t)`, which is negative for negative
  hoppings. It is now `c*|t|`.
- `error_handling.hop_n1` tested `len(mask) == 0` (always the number of
  hoppings) instead of whether any hopping has `n == 1`.
- A four-key hopping dictionary needed only one of `'ang'`/`'tag'` to pass
  validation, then crashed with a bare `KeyError`.

### Changed

- `KSpace.berry_curvature()` / `chern_number()` on a non-Hermitian model now
  raise a ValueError when the chosen bands, ordered by Re E, are not
  separated from the others by a real line gap. That covers an imaginary
  line gap, and exceptional points or overlapping real parts, which the new
  checks `line_gap` and `band_continuity` detect on the mesh. Before, the
  labels jumped between bands and the result was a meaningless number:
  - `i*H` of the Qi-Wu-Zhang model (C = 1) gave -3, -7 or 5 depending on
    `nk`.
  - QWZ plus `2.5i*sigma_z` gave 1, while the bands isolated by its
    imaginary line gap have C = 0.

  `biorthogonal_chern_number(..., gap='imaginary')` handles these.
  Line-gapped non-Hermitian models, such as the Haldane model with gain or
  loss, and every Hermitian model give bit-for-bit the same curvature as
  before; this was checked against the previous implementation.
- `System.set_hopping_manual()` now refuses the inputs that used to build a
  malformed `hop` array and fail later (out-of-range or non-integer indices,
  non-numeric values). Everything that worked before still does: NumPy
  integer indices and NumPy values, `(i, i)` keys, a non-`bool` `upper_part`.
  `error_handling.set_hopping_def` accepts NumPy integers and numbers too.
  `KSpace.get_bands()` checks that `ks` has shape `(nk, dim)`.
- No existing call breaks: the unmodified 0.3.0 test suite passes against this
  version, except the two tests asserting that 3-tuple positions and
  primitive vectors are rejected -- they are now valid 3D input.
- Beyond `System.dense_max` sites, `get_distances` uses a k-d tree instead
  of the dense distance matrix; smaller lattices take the unchanged path.
- `Plot.spectrum_hist()` returns its figure, like the other plotting methods.
- `KSpace` spin blocks accept NumPy scalars (`np.ndim(t) == 0`), not only
  Python numbers.
- Documentation fixes:
  - `set_peierls_phase`'s Landau-gauge example called `2*pi*alpha` "B"
    while claiming the same field as `set_magnetic_field(alpha=B)`.
  - The strain formulas wrote `B_s = beta/2` and `sqrt(beta n)`. The gallery
    uses `beta = -0.05`, so these are now `|beta|`.
  - The tutorial claimed a real-space cross-check of the Kane-Mele edge states
    that the example never performs.
  - The negative-angle range in `print_distances` was mislabelled.
  - `ellipse_in`/`ellipse_out` used radii `a`, `b` for parameters named
    `rx`, `ry`.
  - Assorted wrong defaults and missing parameters, plus typos ("plackets",
    "stength", "conjugaison", "Fist").
  - `Propagation` now states the Crank-Nicolson step it implements.
- The test suite is back to 100% line coverage (`Plot.plt_hopping` without an
  `ax` was untested).

## 0.3.0 -- 2026-09-23

### Fixed

Bugs that produced silently wrong numbers:

- `System.set_hopping_def()` assigned a scalar to `hop['ang']` and
  `hop['tag']` without a mask, so setting one hopping's value flattened
  *every* bond's angle and tag onto a single value. `get_coor_hop()` then
  collapsed a square lattice onto a line. It also read `vec_hop`, which
  `set_hopping_manual()` never fills.
- `System.set_hopping()`'s tag branch relied on `self.hop['n'] == dic['n'] &
  (...)`, but `&` binds tighter than `==`. The mask that clears previously-set
  hoppings therefore misfired for every hopping order except `n == 1`:
  re-setting an `n >= 2` hopping appended a duplicate instead of replacing it,
  and `get_ham()` summed the two (t = 1 then t = 9 gave 10).
- `Lattice.rotation()` converted degrees with `PI/360` instead of `PI/180`,
  halving every rotation, and re-applied the rotation once per unit-cell site,
  each about a different centre -- so a kagome lattice asked for 90 degrees was
  turned by 135 and translated. It now rotates once, about the origin.
- `KSpace` dropped the y-component of a 1D primitive vector, in both
  `get_ham()` and `mesh_grid()`. A chain not aligned with x got a
  k-independent (flat) Hamiltonian, and a honeycomb ribbon cut with
  `direction=0` sampled only 25% of the Brillouin zone. In 1D, `k` is now the
  crystal momentum along the primitive vector (identical to `k_x` for an
  x-aligned chain, so x-aligned models are unaffected).
- `Propagation`'s `norm=True` divided by `sum|psi|` (the L1 norm), leaving
  `sum|psi|^2` far from 1 -- the Crank-Nicolson step is a Cayley transform and
  already unitary, so the flag only did damage. It now uses the L2 norm.
- `System.get_eig(eigenvec=True, left=False)` left the left eigenvectors from
  an earlier `left=True` call in place and applied a second sort permutation to
  them; `get_petermann()` then silently returned `inf`. `ln` is now cleared on
  every call.
- `System.change_hopping_square()` / `change_hopping_ellipse()` matched the
  `'ang'` key with exact float equality. Angles come from `arctan2`
  (29.999999999999996, ...), so on a honeycomb lattice these silently changed
  no hoppings at all. They now use `np.isclose(..., atol=ATOL)`, as the rest of
  the package does.
- `GrapheneSystem.get_beta_lims()` derived the strain limits from one
  sublattice's extreme y coordinate and ignored the x-dependence entirely, so
  the range it returned did not keep the hoppings positive (for an uncentred
  flake the second endpoint was off by more than an order of magnitude). It now
  computes the bound from every bond's actual strain projection and returns
  `[beta_min, beta_max]` ascending.
- `GrapheneSystem.get_butterfly()` accepted a hopping `t` but hard-coded
  `t = 1` in the sweep.
- `System.get_petermann()` returned the wrong Petermann factor for some
  eigenstates. `scipy.linalg.eig` normalizes the left and right
  eigenvectors but does not fix their *relative phase*, so the overlap
  `<L|R>` carries an arbitrary phase; taking its real part instead of
  its modulus made `K_n` gauge-dependent, and it came out as `K_n^2`
  for modes whose overlap happened to pick up a phase. For the
  PT-symmetric dimer, where `K = 1/(1-g^2/t^2)` is known exactly, one
  of the two modes was wrong at every `g > 0`.
- `System.get_eig(eigenvec=True, left=True)` unpacked `scipy.linalg.eig`'s
  `(w, vl, vr)` as `(en, rn, ln)`, so `rn` held the *left* eigenvectors and
  `ln` the right ones. `get_petermann()` was unaffected (`|<L|R>|` is
  symmetric under the swap), but `rn`, `intensity`, `pola` and `ipr`
  described the wrong state. Only non-reciprocal models show it: for a
  complex *symmetric* Hamiltonian (a PT-symmetric gain/loss chain) the left
  eigenvectors are the conjugates of the right ones, so `|rn|^2` is the same
  either way -- but for a Hatano-Nelson chain the skin mode came out
  localized at the wrong end.
- `System.get_ipr()`'s docstring wrote the inverse participation ratio
  as `|sum_i psi_i|^4` instead of `sum_i |psi_i|^4`. The code was
  already correct.

Broken on correct usage:

- `Plot.petermann()` guarded on `sys.ipr` rather than `sys.petermann`, raising
  `AttributeError` even after a correct `get_eig(left=True)` +
  `get_petermann()`. `System.__init__` now also initializes `ipr`, so its guard
  gives a clean `RuntimeError` instead of `AttributeError`.
- `Plot.spectrum()`'s keyword was spelled `peterman`; its docstring documented
  `petermann`. Renamed to `petermann` (**breaking**: pass `petermann=True`).
- `Save` built its path by string concatenation, so
  `Save(dir_name='run', dir_main='out')` created `outrun/` instead of
  `out/run/`. It now uses `os.path.join`, and the default `dir_main` is `figs`
  rather than `figs/`.

### Changed

- `Plot.lattice()` and `Plot.lattice_hop()` accept an `ax` argument, drawing
  onto a caller-supplied axis instead of always creating a figure of their own,
  so several lattices can be placed side by side in one figure. `Plot`'s
  lattice drawing now goes through that axis explicitly rather than through
  pyplot's current-figure state; passing no `ax` behaves exactly as before.
- `KSpace.set_onsite()` now accepts a 2x2 matrix when `spin=True`, so
  spin-off-diagonal onsite terms (an in-plane Zeeman field, an onsite Rashba
  term) can be expressed. Previously `set_hopping` refused `i == j` with
  `R == 0` and pointed at `set_onsite`, which only took a number or an
  `(E_up, E_down)` pair -- leaving no way to build such a term.
- `error_handling.ellipse` removed: it was never called, and named its
  parameters `a`/`b` while the methods it claimed to check use `rx`/`ry`.
- `GrapheneSystem.get_beta_lims()` no longer prints to stdout.
- Clearer error messages: `prim_vec` validation no longer reports the loop
  variable name (`coor`) or splits a sentence across two messages ("...must be
  a list.\nof length 1 ... fro 2D lattices"); the hopping-order message now
  substitutes the actual maximum instead of the literal `nmax"`; assorted typos
  ("emptynumpy", double spaces, missing full stops).
- Documentation fixes: `set_peierls_phase`'s Landau-gauge example integrated
  along the wrong coordinate, making it a pure gauge (zero field); the
  `set_hopping` and `set_onsite_def` examples were not valid Python (`t: 1.`
  instead of `'t': 1.`, missing braces); `System.set_onsite` documented a
  parameter that does not exist; `get_propagation`'s and `spectrum_hist`'s
  documented defaults contradicted their signatures; plus assorted copy-paste
  leftovers (`shift_y` "shift the x coordinates", `Lattice.plot` "in hopping
  space", a `:param list_hop:` on `ellipse_in`/`ellipse_out`, `find_ellipse`'s
  `y0` described as the x centre, `get_intensity_pola_min` returning the "max"
  state).

- `Plot`'s `lims` filters compared the raw (complex) eigenenergies against the
  limits, while plotting `en.real`. NumPy orders complex values
  lexicographically, so the filter disagreed with the plotted axis at ties, and
  `spectrum_hist` handed the complex array straight to `plt.hist`, which cast
  it away with a `ComplexWarning`. All six `lims` comparisons, and the
  histogram data, now use `.real`.

### Added (docs)

- The narrative pages and the gallery now *show* the lattice they are about.
  `tutorial.rst` and `history.rst` grew fourteen figures, rendered at build
  time by matplotlib's `.. plot::` directive from the docs-only helpers in
  `docs/source/lattice_figures.py`: each draws the unit cell (solid, labelled
  by tag), the primitive vectors, and a patch of the lattice they generate, so
  the repeating motif is visible rather than only described. Thirteen gallery
  examples gained a corresponding cell, built with the package's own
  `Plot.lattice(plt_hop=True)` so the scripts stay self-contained and the bond
  widths show the hopping amplitudes (the SSH and Rice-Mele dimerizations, for
  instance, are now visible rather than asserted).
- `examples/topology/plot_ssh_model.py` gained a section on the localized
  state that lives *inside* the chain: with strong bonds terminating both
  edges (so neither end is topological) and a defect where two weak bonds
  meet, the odd site count leaves one unpaired sublattice site and chiral
  symmetry pins a single exact zero mode to the defect, decaying by
  -weak/strong per dimer on either side.
- The 1D-chain figures in the Bloch-oscillation and Anderson-localization
  examples drew all 161 / 200 sites at `set_aspect('equal')`, so the markers
  merged into a featureless solid bar. They now draw a 16-site stretch, with
  the onsite energies (the linear tilt, the disorder realization) plotted
  underneath -- since in both examples the geometry is what stays fixed and
  the onsite term is what the physics lives in.
- `examples/tight_binding/plot_square_lattice_bands.py` -- the square
  lattice as the smallest complete illustration of Bloch's theorem: a
  one-orbital cell whose whole band structure is
  `E(k) = -2t(cos kx a + cos ky a)`, checked against the Bloch sum at 200
  random k-points, plotted along Gamma-X-M-Gamma and over the Brillouin
  zone (the nested E=0 Fermi surface), with the van Hove singularity in its
  density of states. The minigallery under "1928 -- Bloch's Theorem and
  Band Theory" pointed at the graphene example; it now points here, matching
  the square lattice that section already draws.
- Ribbons are drawn as level strips. `ribbon()` stacks its rows along the
  primitive vector that, for a honeycomb lattice, has a component along the
  periodic direction too, so tiling its unit cell as-is draws a slanted
  parallelogram. Sliding each site back by whole multiples of **a**\ :sub:`1`
  -- the same site of the same cell, relabelled by a different **R** -- gives
  the strip with two open zigzag edges that one actually pictures.

### Added (tests)

- `tests/test_regressions.py`, pinning each of the above bugs.
- The test suite is now warning-free, and `filterwarnings = ["error"]` in
  `pyproject.toml` keeps it that way. Two test-side sources were fixed rather
  than silenced: `show()` is now patched out (there is nothing to show under
  the Agg backend), and animations are drawn before being dropped, which also
  runs `get_animation`'s per-frame callback for the first time.

### Added
- `examples/strain/plot_pseudo_magnetic_field.py` -- linear triaxial
  strain as a pseudo-magnetic field, verified against graphene's
  relativistic Landau ladder and against a real field of the same
  strength.
- `examples/non_hermitian/plot_pt_symmetry.py` -- PT symmetry breaking,
  the exceptional point of a gain/loss dimer, the Petermann factor, and
  the selective amplification of an SSH edge mode.
- Two new entries in `docs/source/history.rst` (strain as a gauge
  field, 2010; PT symmetry and exceptional points, 1998/2015).


## 0.2.1

First release published to PyPI (`pip install tbkit`).

### Fixed
- README's logo image used a path relative to the repository
  (`docs/source/_static/image/tbkit_logo.png`), which GitHub resolves
  against the repo automatically but PyPI does not -- PyPI renders the
  README standalone, so the logo was broken on the PyPI project page.
  Switched to an absolute `raw.githubusercontent.com` URL, which
  renders correctly on both.

### Added
- PyPI links: an icon in the docs navbar next to GitHub, and a link
  from the docs home page and the README.
- A test coverage badge/link in the README, pointing at an HTML
  coverage report (100% line coverage) published alongside the docs.

## 0.2.0

**tbkit** now runs on current Python/NumPy/SciPy, and gained a new
reciprocal-space band-structure feature.

### Fixed
- The package could not be imported at all on Python >= 3.12
  (`tbkit/__init__.py` imported the removed `distutils` module and ran
  `distutils.core.setup()` as an import side effect).
- `System.get_eig()` and `System.get_petermann()` crashed on current SciPy
  (`sparse_matrix.H` was removed; replaced with `.conj().T`).
- `numpy.core.defchararray` (deprecated, later removed) replaced with the
  public `numpy.char`.
- `Save`: `dir_main` was assigned the wrong argument, the `dir_name` method
  was shadowed by its own return value, and the `params={}` mutable default
  argument was shared across instances.
- `error_handling.ani()` checked the wrong variable (`fig` instead of `ani`),
  so it never actually validated anything.
- `error_handling.angle()` had inverted boolean logic and the wrong sign
  convention, so it silently accepted invalid angles for `upper_part=False`
  and used the wrong tolerance (machine epsilon instead of the package's
  `ATOL=1e-3`).
- `System.get_intensity_pola_max/min()` crashed on current NumPy (`float()`
  on a 1-element array is no longer implicit).
- Assorted invalid `\x` escape sequences in docstrings (`SyntaxWarning`,
  soon to be `SyntaxError`).
- Inverted "upper part"/"lower part" wording in `System.set_hopping()`'s
  docstring.
- `tests/test_system.py` called `set_hopping(..., low=True)`, a parameter
  that no longer existed (renamed to `upper_part` at some point without
  updating the tests) — these tests were not actually exercising that code
  path.
- `tests/test_system.py::test_get_intensity_pola_min` called
  `get_intensity_pola_max` throughout (copy-paste bug) and so never actually
  exercised `get_intensity_pola_min`.
- `error_handling.py` had two functions (`empty_coor`, `empty_coor_hop`) each
  defined twice; the first definition of each was permanently unreachable
  (silently shadowed by the second). Removed the dead pair.
- `error_handling.prim_vec()` had an unreachable branch
  (`if 1 > len(prim_vec) > 2`, impossible given the preceding length check).
  Removed.
- `error_handling.set_hopping()`'s 3-key-dict branch checked
  `not ('tag' not in dic or 'ang' not in dic)`, which requires both `'tag'`
  and `'ang'` to be present simultaneously — impossible in a 3-key dict, so
  the branch could never fire and a bogus third key was silently accepted.
  Fixed to mirror the (correct) 4-key branch: reject when neither key is
  present.
- `System.get_coor_hop()` crashed (or silently produced wrong coordinates)
  for any lattice needing more than one BFS step: `i_visit = explored[0]`
  returned a 1-element array instead of a scalar index. Fixed to
  `explored[0, 0]`.
- `Plot.polarization/ipr/petermann()` crashed on current Matplotlib
  (`Tick.label` was removed in favor of `Tick.label1`).
- `fig.set_tight_layout(True)` (pending-deprecated) replaced with
  `fig.set_layout_engine('tight')` throughout `plot.py` and `kspace.py`.
- `docs/conf.py` referenced `sphinx.ext.pngmath`, removed from Sphinx years
  ago; replaced with `sphinx.ext.mathjax`. `docs/graphene.rst` was an orphaned
  file duplicating content already generated from `tbkit.rst`'s
  `tbkit.graphene` automodule; removed. Several docstrings
  (`System.set_hopping`, `System.set_onsite_dis`, `error_handling.set_onsite`,
  `error_handling.angle`) had reST formatting bugs (missing blank lines,
  unspaced emphasis markup) that produced docutils parse errors; fixed. The
  docs now build with zero warnings.
- All five notebooks in `examples/` updated for the 0.2 API (`'a'` instead of
  `b'a'`, `'U1'`/`'U2'` instead of `'S1'`/`'S2'` in any hand-built structured
  arrays) and re-executed end to end to confirm they run cleanly on the
  current package.
- `tbkit.dos.density_of_states` cast complex energies to `float64` before
  taking their real part, triggering a spurious `ComplexWarning`; fixed the
  cast order.

### Added
- `KSpace.berry_curvature` and `KSpace.chern_number`: Berry curvature and
  Chern number of a group of bands, by the gauge-invariant
  Fukui-Hatsugai-Suzuki lattice method. Verified against the Haldane model's
  topological phase transition (Chern number 1 -> 0 as the sublattice mass
  crosses the analytic critical value) and against the sum rule that the
  Chern numbers of all bands together must vanish. See
  [`examples/topology/plot_haldane_topology.py`](examples/topology/plot_haldane_topology.py).
- `KSpace(lat, spin=True)`: an optional spin-1/2 degree of freedom on every
  site, with `set_onsite`/`set_hopping` accepting 2x2 (Pauli) matrices in
  addition to plain numbers, for spin-orbit coupling and Zeeman terms. New
  `PAULI` dict of Pauli matrices. Verified that a spinful Kane-Mele model
  matches two decoupled spinless models with opposite next-nearest-neighbor
  chirality exactly, at every k-point tested, and that Rashba coupling
  splits bands away from (but not at) time-reversal-invariant momenta, as
  required by Kramers' theorem.
- `tbkit.kspace.ribbon`: cut a ribbon (periodic in one direction, finite in
  the other) out of any periodic model, to see edge states. Verified
  against a large real-space flake (`System`), and reproduces the zigzag
  graphene ribbon's zero-energy edge flat band and the Kane-Mele ribbon's
  helical edge states crossing the bulk gap. See
  [`examples/topology/plot_edge_states.py`](examples/topology/plot_edge_states.py).
- `tbkit.dos.density_of_states`, `Plot.dos`, `KSpace.mesh_bands`/`.plot_dos`:
  Gaussian/Lorentzian-broadened density of states, from a real-space
  spectrum or a Brillouin-zone mesh. Verified that the broadened DOS
  integrates back to the exact number of input levels.
- `tbkit.lattices`: ready-made `chain`, `square`, `triangular`, `honeycomb`,
  `kagome`, and `lieb` lattices. The kagome and Lieb hopping patterns are
  verified against their famous exactly-flat bands (at E=-2t and E=0
  respectively).
- `System.set_peierls_phase` and `System.set_magnetic_field`: apply an
  orbital magnetic field to the existing hoppings via the Peierls
  substitution (the latter is a symmetric-gauge convenience wrapper around
  the former for a uniform field). See
  [`examples/magnetic_field/plot_magnetic_field.py`](examples/magnetic_field/plot_magnetic_field.py) for an
  Aharonov-Bohm ring example, verified against the analytic zero-field
  spectrum and the flux periodicity of the textbook result.
- `tbkit.kspace`: build the Bloch Hamiltonian `H(k)` of a periodic lattice
  from a small set of intra-unit-cell hoppings (`KSpace.set_hopping`,
  `.set_onsite`), diagonalize it at a single k-point or over a k-path
  through high-symmetry points (`.get_bands`, `.k_path`), and plot the
  resulting band structure (`.plot_bands`). Includes a
  `reciprocal_vectors()` helper.
- `pyproject.toml`, replacing the old `distutils`-based `setup.py`.
- `LICENSE` (BSD-3-Clause, matching the license named in the old README).
- `.gitignore`, and removal of previously committed build artifacts
  (`build/`, `dist/`, `*.egg-info`, `*.pyc`, `docs/_build`,
  `.ipynb_checkpoints`).
- GitHub Actions CI running the test suite on Python 3.10-3.13.
- `examples/tight_binding/plot_graphene_bands.py`, a from-scratch example covering both the
  real-space flake workflow and the new k-space band-structure workflow.
- Full unit test suite (100% line coverage of the `tbkit` package):
  `tests/test_error_handling.py`, `tests/test_kspace.py`,
  `tests/test_graphene.py`, `tests/test_plot.py`, `tests/test_save.py`, and
  a rewritten `tests/test_propagation.py` (previously had no tests at all).
- `docs/source/tutorial.rst`: a narrative walkthrough (lattice -> real-space system
  -> reciprocal space -> disorder/strain/field -> spin-orbit -> topology ->
  edge states -> DOS), linked from the docs home page, complementing the
  auto-generated API reference. Every code snippet in it is exercised
  end-to-end (not just eyeballed) before being included.
- Type hints on every public function/method signature across the package
  (`from __future__ import annotations` + `numpy.typing`), except
  `error_handling.py`'s ~90 small validators, which intentionally accept
  values of any type to check them and so gain little from hinting.
- `docs/source/history.rst`: a chronology of Tight-Binding breakthroughs (Bloch's
  theorem 1928 through the Kane-Mele model 2005-2007), each entry linked to
  the corresponding `tbkit` functionality, real references, and a
  numerically-verified example. Backed by four new example scripts:
  `examples/topology/plot_ssh_model.py` (bulk gap = 2|v-w|, edge states present only for
  w>v and exponentially localized), `examples/disorder/plot_anderson_localization.py`
  (IPR vs. disorder strength), `examples/magnetic_field/plot_hofstadter_butterfly.py` (checked
  against the exact, flux-independent Gershgorin bound, particle-hole
  symmetry, and flux periodicity), and `examples/flat_bands/plot_flat_bands.py` (kagome and
  Lieb bandwidths below 1e-8t).
- Three more `docs/source/history.rst` entries: Wallace's 1947 tight-binding
  prediction of graphene's linear (Dirac) dispersion, decades before its
  isolation (verified in `examples/tight_binding/plot_graphene_bands.py` to within 2% for
  |q|<=0.05 from the Dirac point); Bloch oscillations and the Wannier-Stark
  ladder (`examples/dynamics/plot_bloch_oscillations.py`, new: exact ladder spacing,
  localization strengthening monotonically with tilt, and a wavepacket's
  oscillation amplitude matching the semiclassical 4t/F to within 0.3%);
  and the 1983 Thouless quantum pump / Rice-Mele model
  (`examples/topology/plot_thouless_pump.py`, new: built as a genuine 2D `KSpace` Bloch
  Hamiltonian with the pump parameter as a synthetic second momentum,
  verifying both a quantized Chern number and an independently computed
  polarization (Zak phase) winding of exactly 1 per cycle).
- `sphinx.ext.intersphinx`, configured for python/numpy/scipy/matplotlib, so
  the type hints added above resolve to their official docs instead of
  dangling; caught and fixed a real docstring bug in the process
  (`error_handling.hop_n1` documented a nonexistent `RunTimeError`, when
  the function actually raises `ValueError`).
- Docs now use `pydata_sphinx_theme` (light mode), the same theme as the
  sibling physicskit/mathematicskit/chemistrykit projects, with a navbar
  logo (`docs/source/_static/image/tbkit_logo.png`).
- Every `docs/source/history.rst` breakthrough entry and
  `docs/source/tutorial.rst` example mention is now backed by an actual
  rendered `.. minigallery::` thumbnail, not just a plain-text filename.
- Two new `docs/source/history.rst` entries, each with a new,
  numerically-verified example:
  - 1930/2005, Landau levels: `examples/magnetic_field/plot_landau_levels.py` builds a
    square-lattice flake and a triangular graphene flake at the same weak
    flux, confirming exact particle-hole symmetry in both, the square
    lattice's non-relativistic Landau ladder to within 15% of
    `E_0+omega_c/2`, and graphene's relativistic n=1 Landau level to
    within 1% of `v_F*sqrt(2eB)`.
  - 2000/2011, a topological flat band on the kagome lattice:
    `examples/topology/plot_kagome_chern_band.py` confirms the flat band and
    middle band are exactly degenerate at Gamma at zero field, confirms a
    real gap opens across the whole Brillouin zone once the
    nearest-neighbor hopping is made complex, and confirms the resulting
    three bands' Chern numbers are exactly -1, 0, +1.
- `examples/tight_binding/plot_visualizing_a_model.py`: the first example to actually
  exercise `tbkit.plot.Plot` (lattice, spectrum with sublattice
  polarization, density of states, eigenstate intensity), verifying that
  every state's sublattice weights sum to exactly 1 and that the
  broadened DOS integrates back to the exact site count.

### Changed
- `docs/` restructured into `docs/source/` (Sphinx source) and `docs/build/`
  (output), matching the sibling physicskit/mathematicskit/chemistrykit
  layout; `docs/Makefile`/`docs/make.bat` updated from the old
  hardcoded-source-directory sphinx-quickstart (2016) versions to the
  modern minimal `SOURCEDIR`/`BUILDDIR`-based Makefile.
- `examples/` restructured into a Sphinx-Gallery source tree: each script
  moved into a topic subfolder (`tight_binding/`, `magnetic_field/`,
  `disorder/`, `topology/`, `flat_bands/`, `dynamics/`) with a
  `README.rst` blurb, renamed to the `plot_*.py` convention, and rewritten
  with an RST module docstring plus `# %%`-delimited narrative/code cells.
  `sphinx_gallery.gen_gallery` (new `docs` optional dependency, along with
  `sphinx-gallery`) renders these into an executed, thumbnailed gallery
  under `docs/source/api/gallery/` on every docs build, so the rendered docs and
  the runnable scripts can never drift out of sync -- the script is the
  single source of truth for both. Manual `fig.savefig(...)` calls were
  dropped from every script in favor of Sphinx-Gallery's automatic figure
  capture.
- Sublattice tags are now plain one-character strings (`'a'`) instead of
  byte strings (`b'a'`).
- Classes renamed to `PascalCase` (`Lattice`, `System`, `Plot`,
  `Propagation`, `Save`, `KSpace`, `GrapheneLattice`, `GrapheneSystem`) to
  avoid the previous class name matching its module name (which made
  `import tbkit.lattice as lattice` silently bind the class rather than the
  submodule, once `tbkit/__init__.py` wildcard-imported it). The pre-0.2
  lowercase/camelCase names remain available as aliases for continuity.
