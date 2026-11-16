# CLAUDE.md

tbkit builds and solves 1D/2D/3D Tight-Binding models in vectorized NumPy/SciPy: finite real-space lattices (`System`) and periodic Bloch Hamiltonians (`KSpace`), Hermitian and non-Hermitian. It targets teaching as much as research, so keep the mechanics explicit and inspectable.

## Commands

```bash
pip install -e ".[test]"          # tests
pip install -e ".[docs]"          # docs (sphinx, pydata-sphinx-theme, sphinx-gallery)

pytest tests/                                        # full suite, in parallel (pytest-xdist, set in pyproject addopts)
pytest tests/ --cov=tbkit --cov-report=term-missing  # what CI runs (Python 3.10-3.14)
pytest tests/test_kspace.py::test_name -q -n0        # a single test, without worker startup

python examples/<section>/plot_<name>.py             # run one gallery example
cd docs && make html                                 # re-executes every examples/*/plot_*.py; output in docs/build/html
```

## Tests and CI

- `filterwarnings = ["error"]`: any warning fails the suite. A warning from tbkit is a defect. Add a targeted `ignore::` only for dependency warnings outside our control.
- Coverage of `tbkit` is 100% of lines. New code needs tests that keep it there.
- `test-minimum-versions` runs on the oldest NumPy/SciPy/matplotlib that `pyproject.toml` allows (Python 3.10). No NumPy 2-only APIs: use `scipy.integrate.trapezoid` and `np.char`, not `np.trapezoid` or `import numpy.char`.
- `lint`: `ruff check tbkit` (no formatter). `typecheck`: `mypy tbkit`, must pass. `docs`: builds the docs (every gallery example) and runs `paper/listings/*.py`.
- `tests/conftest.py` forces the matplotlib `Agg` backend.
- `paper/` is the SciPost manuscript with its Kwant/PythTB cross-checks and benchmarks (`paper/README.md`, `paper/environment.yml`). Not in CI except the listings.
- The default branch is `master`.

## Architecture invariants

Two independent pipelines, both starting from `Lattice(unit_cell, prim_vec)`.

### Real space (`System`)

- `get_lattice(n1, n2, n3)` fills `lat.coor`, a structured array `[('x','f8'), ('y','f8'), ('tag','U8')]` (plus `('z','f8')` for 3D, see `lat.space_dim`). Geometry edits mutate it in place.
- `set_hopping` selects bonds by neighbour order, angle or tag pair, stored in `sys.hop` (`i`, `j`, `t`, `ang`, `tag`, ...). Tag pairs are matched on the two site tags (`System._tag_mask`), never on the concatenated `hop['tag']`, which can collide.
- Modifiers (disorder, dimerization, defects, `set_peierls_phase`, `set_magnetic_field`) must run *after* `set_hopping`/`set_hopping_manual`.
- `get_ham()` builds sparse CSR from one triangle plus its conjugate transpose, unless bond angles mix signs.
- Value functions (`t(site_i, site_j, **params)`, `onsite(site, **params)`, helpers in `values.py`): their `sys.hop` rows hold `t = 1` and `sys._hop_values` keeps `(directed bond keys, function)`, so filtering or concatenating `hop` never desyncs them. `get_ham(**params)` multiplies them in; assigning a number to those bonds (`_assign_hopping`) drops the function.
- `io.py` persists `HOP_DTYPE` (`load_model` casts older archives to it): change it only compatibly.
- Above `System.dense_max` (5000) sites, distances use a k-d tree.
- Periodic Systems (`periodic=True | (bool, ...)`): every bond displacement goes through `System._min_image`. In new code use `_bond_vectors(i, j)`, never `coords[j] - coords[i]`. `get_bonds` refuses an order reaching half-way around the torus. `set_magnetic_field` and `get_local_chern_marker` need open boundaries; `get_bott_index` needs a 2D torus. `io` writes `periodic` (archive version 2) only for periodic Systems.

### Reciprocal space (`KSpace`)

- Uses only `unit_cell` and `prim_vec`, never `get_lattice`. Hoppings are `{'i', 'j', 'R', 't'}`. With `spin=True` values may be 2x2 (`kspace.PAULI`).
- Every Bloch matrix comes from `KSpace._bloch_sum` (or `_bloch_ham`). Never add another loop over `_hop`. Its two paths (`_bloch_grouped` when the blocks are at least `KSpace.group_fill` full, else one exponential per hopping plus `bincount`) give identical results; keep them in sync (`TestGroupedBlochSum`).
- Every diagonalization over many k-points goes through `KSpace._eigs` (`_subspaces` for band projectors), never a loop over k. `tests/benchmark_kspace.py` times it.
- `KSpace._hop` is a property: `_hop_const` plus `_hop_values` evaluated with `ks.params`. Assigning it replaces both, so `bdg`, `moire` and `io` keep reading and writing `_hop` unchanged.
- Bloch convention: periodic gauge (orbital positions ignored). Wilson-loop tools add positions back with `positions=True`.
- `set_hopping(..., hermitian=False)` stores no conjugate. `set_overlap` makes every eigenproblem generalized.
- Non-Hermitian: `berry_curvature`/`chern_number` raise ValueError for a band group without a real line gap (`error_handling.line_gap`, `band_continuity`); keep the Hermitian path untouched. `tbkit.exceptional` follows eigenvalues by continuation (`_track`), never by sorting. Sign convention: W = -2 sum nu.
- `magnetic_supercell`, `slater_koster.sk_kspace`, `bdg.bdg_kspace`, `floquet.FloquetKSpace`/`DrivenKSpace` and `kspace.ribbon` all return `KSpace` objects, so every band and topology tool applies to them.
- Berry-curvature responses (`hall_conductivity`, `spin_hall_conductivity`, `orbital_magnetization`, `anomalous_nernst_conductivity`, `thermal_hall_conductivity`) share `_hall`/`_kubo` with pair weights in `kspace._response_weights`: add new ones there. `hall_conductivity` uses the TKNN sign (equals `chern_number` in a gap, the negative of the Ohm's-law `sigma_xy`).

### Transport

- `Transport` stores the device sparse; `ham` is a lazily built dense property. Don't build `self.ham` on the sparse path.
- `Transport._exact(eta)` picks the path. Default (`eta=None`): `Transport._scattering` (exact lead modes from `_lead_modes`, one `splu` of device plus lead boundary conditions, Groth et al. 2014). Explicit `eta`, non-Hermitian leads, `lead_in == lead_out`: Caroli formula with Sancho-Rubio leads, on the columns of `G^r` at the lead sites from one `splu` (`Transport._green_columns`). Only `get_green`/`self_energy` (full matrices) are dense.
- A lead's `conservation_law` blocks live in `Transport._laws`, so `self.leads` keeps its 4-tuples; `SMatrix` addresses `(lead, block)`.
- `transport._green_surface_bulk` is shared by `surface_green` and `KSpace.surface_spectral_function`.

### Shared helpers to build on

- Mesh integrals and contours: `dos._mesh_simplices`/`_simplex_values` (behind `tetrahedron_dos` and `kspace._level_set`); `KSpace._fold` maps into the first Brillouin zone.
- k-paths: `kspace._path` (`high_symmetry_path` for standard ones). Band expectation values: `KSpace._expectations`.
- Sector, spin and mirror Chern numbers: `KSpace._sector_cherns` on the FHS product `KSpace._flux`.
- Exact references for interacting results: `tbkit.manybody` (Fock sectors as bit patterns, `_hop`/`_two_body_matrix` vectorized over all states). Use it rather than ad-hoc Fock-space code in examples or tests.
- Wannier90-style `KSpace` construction: `io._hr_kspace` (used by `WannierFunctions.kspace` and `read_wannier90`).
- Spatial symmetry: every space-group operation acts through `symmetry._Action` (position blocks, their permutation, lattice vectors `L_s`, orbital matrices) and `symmetry._representations` (`U_g(k)` in the periodic gauge, vectorized over k); `KSpace.symmetry_error` imports them lazily (`symmetry` imports `kspace`). Characters and rotation eigenvalues depend on the translation's lattice representative: a rotation about `c` must have translation `c - R c` exactly (`Operation.about`), which `find_symmetries` gives for symmorphic structures.
- `continuum.py`: sympy is optional (extra `continuum`) and imported inside the functions, so `import tbkit` must never import it. Its parametrized onsite terms go straight into `_hop_values` as an `i -> i`, `R = 0`, `hermitian=False` bond, because `set_hopping` refuses that bond.

### Validation

All argument checks live in `tbkit/error_handling.py`, called by every public method before doing anything else, and are exercised by `tests/test_error_handling.py`. Put new checks there, not inline. Its `ATOL = 1e-3` is the distance/angle tolerance used throughout `system.py`.

## Conventions

- Sublattice tags are `str` of 1 to `error_handling.TAG_LEN` (8) characters (`'a'`, `'Mo'`), not bytes (the pre-0.2 API used `b'a'`). A bond selector `'tag'` is a two-character string (two one-character tags) or a pair of tags; turn it into `(tag_i, tag_j)` with `lattice._tag_pair`. The old lowercase class names (`lattice`, `system`, `plot`, `propagation`, `save`) remain as backward-compatible aliases.
- `tbkit/__init__.py` imports classes explicitly, never with `from tbkit.<module> import *`: a wildcard would rebind the `tbkit.lattice` submodule to the alias class and break `import tbkit.lattice as lattice`.
- A bug fix gets a pinning test (see `tests/test_regressions.py`) and a `CHANGELOG.md` entry saying what numbers were wrong and why.
- Gallery examples (`examples/<topic>/plot_*.py`, one topic per folder, each with a `README.rst`) are self-contained and `assert` their key numeric claims before plotting. The docs claim nothing the code doesn't verify.
- `docs/source/history/` is a chronology of Tight-Binding breakthroughs: `index.rst` plus one `<theme>_breakthroughs.rst` per examples subsection of `docs/source/index.rst` (models, fields, topology, response, interactions, driven). Each breakthrough links to its tbkit functionality and gallery examples.
  - Every breakthrough links to one or more gallery examples, and no example is linked from two breakthroughs.
  - An entry has a single `.. minigallery::` directive, with one example path per line when there are several, so the thumbnails sit side by side in one row.
  - Each example's title and content must show that breakthrough's subject.
  - Split an example covering several breakthroughs into focused ones, one per breakthrough, and delete the combined file.
  - When the match is unclear, change the example, not the breakthrough's title or text.
