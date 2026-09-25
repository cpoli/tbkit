# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

tbkit builds and solves 2D Tight-Binding models in vectorized NumPy/SciPy: finite real-space lattices (flakes, ribbons, disorder, defects, magnetic fields, strain) and periodic Bloch Hamiltonians (band structures, Berry curvature, Chern numbers, spin-orbit coupling, ribbons for edge states). Hermitian and non-Hermitian Hamiltonians are both supported. It targets teaching as much as research, so the mechanics are kept explicit and inspectable.

## Commands

```bash
pip install -e ".[test]"          # tests
pip install -e ".[docs]"          # docs (sphinx, pydata-sphinx-theme, sphinx-gallery)

pytest tests/                                        # full suite, in parallel (pytest-xdist, set in pyproject addopts)
pytest tests/ --cov=tbkit --cov-report=term-missing  # what CI runs (Python 3.10-3.13)
pytest tests/test_kspace.py::test_name -q -n0        # a single test, without worker startup

python examples/<section>/plot_<name>.py             # run one gallery example
cd docs && make html                                 # re-executes every examples/*/plot_*.py; output in docs/build/html
```

- `pyproject.toml` sets `filterwarnings = ["error"]`. Any warning fails the suite. A warning from tbkit is treated as a defect (for example, complex energies reaching matplotlib). Only add a targeted `ignore::` for dependency warnings outside our control.
- The suite has 100% line coverage of `tbkit` (published report linked from the README). New code needs tests that keep it there.
- `tests/conftest.py` forces the matplotlib `Agg` backend.
- No linter or formatter is configured.
- The default branch is `master`.

## Architecture

There are two independent pipelines. Both start from a `Lattice`:

- **Real space.** `Lattice(unit_cell, prim_vec)` → `get_lattice(n1, n2, n3)` fills `lat.coor`, a NumPy structured array with dtype `[('x','f8'), ('y','f8'), ('tag','U1')]` (plus `('z','f8')` for 3D positions; `lat.space_dim` says which). Geometry edits (`remove_sites`, `ellipse_in`, `boundary_line`, `rotation`, ...) mutate that array in place.
  - `System(lat)` then computes pairwise distances. `set_hopping` addresses bonds by neighbour order `n` (1st, 2nd, ...), optionally narrowed by angle or sublattice-pair tag. It stores them in `sys.hop`, a structured array with fields including `i`, `j`, `t`, `ang` and `tag`.
  - Modifiers act on `sys.hop`/`sys.onsite` and must run *after* `set_hopping`/`set_hopping_manual`: disorder, dimerization, defects, `set_peierls_phase`, `set_magnetic_field`.
  - `get_ham()` builds a sparse CSR matrix. It stores only one triangle of hoppings and adds the conjugate transpose, unless the bond angles mix signs.
  - Above `System.dense_max` (5000) sites, `get_distances` uses a k-d tree instead of the dense N² distance matrix.
  - `get_eig()` diagonalizes densely with `scipy.linalg`. It can also compute left eigenvectors for non-Hermitian cases. `get_eig_sparse` uses shift-invert. `get_green`, `get_ldos`, `get_fermi_level`/`get_occupations` (via `occupation.py`) and `get_local_chern_marker` build on it.
  - `Plot(sys)`, `Propagation(lat)` (Crank-Nicolson wavepackets and pumping) and `Save` consume the result. `graphene.py` subclasses `Lattice`/`System` with graphene-specific shapes, triaxial strain and the spectrum-versus-strain "butterfly".
- **Reciprocal space.** `KSpace(lat, spin=False)` uses only `unit_cell` and `prim_vec`, never `get_lattice`.
  - Hoppings are given as `{'i', 'j', 'R', 't'}`: an intra-cell orbital pair plus a lattice-vector offset `R`.
  - `get_ham(k)` returns the Bloch matrix. `k_path`, `mesh_grid`, `berry_curvature` and `chern_number` (Fukui-Hatsugai-Suzuki) build on it.
  - Every Bloch matrix (`get_ham`, `get_overlap`, `get_ham_peierls`, `_bloch_derivatives`, `floquet._bloch_grid`) comes from the one vectorized `KSpace._bloch_sum` (many k at once, optional derivatives, `positions`, vector potential `A`). Build new ones through it (or `_bloch_ham`), not with another loop over `_hop`.
  - With `spin=True`, onsite and hopping values may be 2x2 matrices, using `kspace.PAULI`.
  - `kspace.ribbon(lat, list_hop, width, direction)` returns a new 1D `KSpace` that is finite across the ribbon.
  - The Bloch convention is the periodic gauge (orbital positions ignored). The Wilson-loop tools (`berry_phase`, `wannier_centers`, `wannier_flow`, `z2_invariant`) add the positions back with `positions=True`.
  - `set_hopping(..., hermitian=False)` stores a hopping without its conjugate. `finite_ham`, `spectral_winding` and `gbz` serve the non-Hermitian case. `set_overlap` turns every eigenproblem into a generalized one.
  - Non-Hermitian band topology in 2D: `biorthogonal_berry_curvature`/`biorthogonal_chern_number` (LR, RL, RR, LL; real or imaginary line gap). `berry_curvature`/`chern_number` label bands by Re E and raise a ValueError for a non-Hermitian band group without a real line gap (checks `error_handling.line_gap` and `band_continuity`). The Hermitian path is untouched. `tbkit.exceptional` takes a 2D `KSpace` or any callable `H(p)` and a closed loop (`circle`, polygon vertices, or a callable of s in [0, 1]). It follows eigenvalues by continuation (`_track`: linear prediction, Hungarian matching, halving ambiguous steps, never sorting) for `vorticity`, `track_eigenvalues` and `encircle` (midpoint biorthogonal parallel transport). `discriminant` comes from the Sylvester resultant of the characteristic polynomial (no eigenvalues), which feeds `discriminant_winding` and `find_exceptional_points` (plaquette windings on a shifted mesh, Newton refinement, charges, Petermann-based orders, exceptional lines when Delta is real) and `fermi_arcs`. Sign conventions: W = -2 sum nu.
  - `kspace.magnetic_supercell` (rational flux), `slater_koster.sk_kspace` (multi-orbital), `bdg.bdg_kspace` (holes as extra orbitals) and `floquet.FloquetKSpace` (a dynamic subclass whose `get_ham` is the Floquet Hamiltonian) all return `KSpace` objects, so every band and topology tool applies to them.
  - `floquet.DrivenKSpace` is the base of those driven models: a copy of a `KSpace` whose `get_ham` is `H_F(k)` for any `H(k, t)` (branch cut `epsilon`). `floquet.step_drive` builds exact piecewise-constant drives from `KSpace` models or ribbons (a `DrivenKSpace`) or from `System`s/matrices (a real-space `StepDrive`). `DrivenKSpace.winding_number` (Rudner et al. 2013, spectral k-derivatives, returns the raw value) and `edge_state_count` (ribbons) detect anomalous Floquet phases that the Chern numbers of `H_F` miss.
  - `hall_conductivity` and `spin_hall_conductivity` evaluate the Kubo formula over a mesh at any Fermi level (vectorized `_bloch_derivatives`, analytic `dH/dk` from the hoppings; `positions=True` uses bond vectors `R + tau_j - tau_i`). In a gap they equal `chern_number` (TKNN sign, the negative of the Ohm's-law `sigma_xy`). `finite_velocity` gives `i[H, r]` from bond vectors for `kpm.hall_conductivity` (Kubo-Bastin) on a torus.
- **Other modules.** `orbital.OrbitalSystem` handles multi-orbital, spinful real-space models. `transport.Transport` computes Landauer transmission with Sancho-Rubio leads. `kpm.py` implements the kernel polynomial method. `meanfield.hubbard_mean_field` solves the Hubbard model by linear mixing.
- **Shared helpers.** `dos.density_of_states` is used by both `System` and `KSpace` plotting. `lattices.py` returns ready-made `Lattice` objects (chain, square, triangular, honeycomb, kagome, Lieb).
- **Validation is centralized in `tbkit/error_handling.py`.** Every public method validates its arguments by calling `error_handling.*` checks before doing anything else, and `tests/test_error_handling.py` exercises those paths. Put new argument checks there, not inline. Its `ATOL = 1e-3` is the distance/angle tolerance used throughout `system.py`.

## Conventions

- Sublattice tags are one-character `str` (`'a'`), not bytes (the pre-0.2 API used `b'a'`). The old lowercase class names (`lattice`, `system`, `plot`, `propagation`, `save`) remain as backward-compatible aliases.
- `tbkit/__init__.py` imports classes explicitly, never with `from tbkit.<module> import *`. A wildcard would rebind the `tbkit.lattice` submodule attribute to the alias class of the same name and break `import tbkit.lattice as lattice`. Keep it that way.
- A bug fix gets a pinning test (see `tests/test_regressions.py`) and an entry in `CHANGELOG.md` saying what numbers were wrong and why.
- Gallery examples (`examples/<topic>/plot_*.py`, one topic per folder, each with a `README.rst`) are self-contained and check their key numeric claims with `assert` before plotting. Nothing is claimed in the docs that the code doesn't verify. The five Jupyter notebooks in `examples/` predate the 0.2 API and use byte-string tags.
- `docs/source/history.rst` is a chronology of Tight-Binding breakthroughs, each linked to the tbkit functionality and to gallery examples via `.. minigallery::`.
  - Every breakthrough links to one or more gallery examples, and no example is linked from two breakthroughs.
  - Each example's title and content must show that breakthrough's subject, so a reader can tell at a glance why it illustrates that entry.
  - Split an example that covers several breakthroughs into focused ones, one per breakthrough, and delete the combined file.
  - When the match is unclear, change the example, not the breakthrough's title or text.
