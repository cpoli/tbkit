# Changelog

## Unreleased

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
