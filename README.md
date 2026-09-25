# tbkit — a Tight-Binding package

[![tests](https://github.com/cpoli/tbkit/actions/workflows/tests.yml/badge.svg)](https://github.com/cpoli/tbkit/actions/workflows/tests.yml)
[![coverage](https://img.shields.io/badge/coverage-100%25-brightgreen.svg)](https://cpoli.github.io/tbkit/coverage/)
[![docs](https://img.shields.io/badge/docs-cpoli.github.io%2Ftbkit-blue.svg)](https://cpoli.github.io/tbkit/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)
[![License: BSD 3-Clause](https://img.shields.io/badge/License-BSD--3--Clause-blue.svg)](LICENSE)

![tbkit logo](https://raw.githubusercontent.com/cpoli/tbkit/master/docs/source/_static/image/tbkit_logo.png)

**tbkit** is a Python package to build and solve Tight-Binding models, written
in vectorized NumPy/SciPy. It aims to make the mechanics of Tight-Binding
models — lattices, hoppings, Hamiltonians, spectra, band structures —
explicit and easy to inspect, so it works as well for teaching as for
research prototyping.

For computational physics beyond tight-binding — quantum mechanics,
classical mechanics, statistical physics, general relativity, and more —
see [physicskit](https://github.com/cpoli/physicskit), whose
`physicskit.condensed` subpackage covers tight-binding band theory and
topology alongside superconductivity.

## Features

* **Real space**: build arbitrarily complex finite lattices (flakes,
  ribbons, disordered structures, defects) site by site or via boolean
  selections (ellipses, half-planes, unions/differences of lattices), then
  diagonalize the resulting Hamiltonian directly.
* **Reciprocal space**: build the Bloch Hamiltonian `H(k)` of the infinite
  periodic lattice from a small set of intra-unit-cell hoppings, and compute
  band structures along a k-path through high-symmetry points.
* **Topology**: Berry curvature and Chern numbers of a group of bands, by
  the gauge-invariant Fukui-Hatsugai-Suzuki lattice method; Zak/Berry
  phases, Wannier centres and their flow, the Z2 invariant (Wilson loops
  and Fu-Kane parities), the quantum geometric tensor, symmetry checks and
  the tenfold way, magnetic supercells (TKNN), and the real-space local
  Chern marker.
* **Hall conductivities**: the intrinsic anomalous Hall conductivity at
  any Fermi level (Kubo formula over the Brillouin zone, in 2D and 3D,
  with adaptive mesh refinement), the spin Hall conductivity, and the
  real-space Kubo-Bastin Hall conductivity of large disordered samples by
  the kernel polynomial method.
* **3D**: lattices, slabs, and Bloch Hamiltonians in three dimensions
  (Anderson transition, Weyl semimetals).
* **Orbitals**: several orbitals per site with Slater-Koster bond
  integrals (s, p, d), non-orthogonal bases, and spinful multi-orbital
  real-space models (`OrbitalSystem`: atomic/Kane-Mele spin-orbit,
  Rashba, Zeeman, Peierls phases).
* **Large lattices**: sparse neighbour search, shift-invert sparse
  eigensolvers, and the kernel polynomial method (DOS, LDOS, Kubo-Greenwood
  conductivity).
* **Green's functions and transport**: retarded Green's functions, LDOS,
  Fermi levels and occupations, and Landauer transmission through a device
  between semi-infinite leads.
* **Interactions and superconductivity**: the Hubbard model in mean field,
  and Bogoliubov-de Gennes Hamiltonians in real and reciprocal space.
* **Non-Hermitian bands**: non-reciprocal hoppings, spectral winding, the
  skin effect, and the generalized Brillouin zone.
* **Exceptional points** of 2D bands: eigenvalue vorticity, discriminant
  winding, an EP finder (charges, Jordan-block orders, exceptional rings,
  bulk Fermi arcs, the doubling theorem), encircling exceptional versus
  diabolical points, and biorthogonal Chern numbers of line-gapped bands.
* **Floquet**: quasienergies, Floquet and Sambe Hamiltonians, driven
  Bloch models (`FloquetKSpace`, `DrivenKSpace`), exact step drives of
  Bloch models, ribbons and flakes (`step_drive`), and anomalous Floquet
  phases: the Rudner winding number and edge-state counts.
* **Spin**: optional spin-1/2 degree of freedom on every site, with 2x2
  (Pauli-matrix) hoppings/onsite terms, for spin-orbit coupling (Rashba,
  Kane-Mele) and Zeeman splitting.
* **Edge states**: cut a ribbon (periodic in one direction, finite in the
  other) out of any periodic model, to see edge/surface physics.
* **Density of states**, Gaussian- or Lorentzian-broadened, from either a
  real-space spectrum or a Brillouin-zone mesh.
* A small library of ready-made lattices (chain, square, triangular,
  honeycomb, kagome, Lieb).
* Complex-valued onsite energies and hoppings; **Hermitian and non-Hermitian**
  Tight-Binding Hamiltonians.
* Multiple sublattices, with hoppings addressed by neighbor order (1st,
  2nd, 3rd-nearest neighbor, ...), by angle, or by sublattice-pair tag.
* Built-in patterns for onsite disorder, hopping disorder, dimerization,
  strain, and an orbital magnetic field (Peierls substitution).
* Time propagation of a wavepacket under the Tight-Binding Hamiltonian
  (Crank-Nicolson).

**tbkit** is organized as a small set of composable classes and modules:

| Class / module            | Purpose                                                |
|----------------------------|---------------------------------------------------------|
| `tbkit.Lattice`             | Define and manipulate site positions and sublattices.   |
| `tbkit.System`              | Build the real-space Hamiltonian from a `Lattice` and solve it. |
| `tbkit.KSpace`              | Build and solve the Bloch Hamiltonian of a periodic `Lattice`; bands, Berry curvature/Chern numbers, anomalous and spin Hall conductivities, ribbons, DOS. |
| `tbkit.Plot`                | Plot lattices, spectra, eigenstates, and the density of states. |
| `tbkit.Propagation`         | Time-evolve a wavepacket.                               |
| `tbkit.Save`                | Save figures/animations to disk.                        |
| `tbkit.lattices`            | Ready-made common lattices.                              |
| `tbkit.dos`                 | Broadened density of states from a set of eigenenergies. |
| `tbkit.OrbitalSystem`       | Real-space models with several orbitals and spin per site (`tbkit.orbital`). |
| `tbkit.Transport`           | Landauer transmission through a device between leads (`tbkit.transport`). |
| `tbkit.occupation`          | Fermi-Dirac occupations and Fermi levels.               |
| `tbkit.kpm`                 | Kernel polynomial method: DOS, LDOS, longitudinal and Hall conductivities of very large lattices. |
| `tbkit.slater_koster`       | Slater-Koster bond integrals and multi-orbital `KSpace` models. |
| `tbkit.meanfield`           | The Hubbard model in (unrestricted Hartree-Fock) mean field. |
| `tbkit.bdg`                 | Bogoliubov-de Gennes Hamiltonians and pairings.         |
| `tbkit.exceptional`         | Exceptional and diabolical points of 2D bands: vorticity, discriminant winding, EP finder, Fermi arcs, encircling. |
| `tbkit.floquet`             | Floquet theory; `FloquetKSpace`/`DrivenKSpace` for driven Bloch models, `step_drive`, and the winding number of anomalous Floquet phases. |

## Install

Requires Python >= 3.10.

```bash
pip install tbkit
```

or, for an editable install from a clone (e.g. to run the test suite or
work on tbkit itself):

```bash
git clone https://github.com/cpoli/tbkit
cd tbkit
pip install -e .
```

or, to also install the tools needed to run the test suite:

```bash
pip install -e ".[test]"
pytest tests/
```

The test suite has 100% line coverage of the `tbkit` package; see the
[HTML coverage report](https://cpoli.github.io/tbkit/coverage/).

## Quick start

Real-space flake, nearest-neighbor square lattice:

```python
from tbkit.lattice import Lattice
from tbkit.system import System

lat = Lattice(unit_cell=[{'tag': 'a', 'r0': (0., 0.)}],
                       prim_vec=[(1., 0.), (0., 1.)])
lat.get_lattice(n1=10, n2=10)

sys = System(lat)
sys.set_hopping([{'n': 1, 't': 1.}])
sys.get_ham()
sys.get_eig()
print(sys.en)
```

Graphene band structure (reciprocal space):

```python
import numpy as np
from tbkit.lattice import Lattice
from tbkit.kspace import KSpace, reciprocal_vectors

DX, DY = 0.5 * 3 ** 0.5, 0.5
unit_cell = [{'tag': 'a', 'r0': (0., 0.)}, {'tag': 'b', 'r0': (DX, DY)}]
prim_vec = [(2 * DX, 0.), (DX, 1.5)]
lat = Lattice(unit_cell=unit_cell, prim_vec=prim_vec)

gra = KSpace(lat)
gra.set_hopping([{'i': 0, 'j': 1, 'R': (0, 0), 't': 1.},
                        {'i': 0, 'j': 1, 'R': (-1, 0), 't': 1.},
                        {'i': 0, 'j': 1, 'R': (0, -1), 't': 1.}])

b1, b2 = (np.array(v) for v in reciprocal_vectors(prim_vec))
Gamma, K, M = np.zeros(2), (b1 - b2) / 3, b1 / 2
gra.k_path([Gamma, K, M, Gamma], nk=60)
fig = gra.plot_bands(node_labels=[r'$\Gamma$', 'K', 'M', r'$\Gamma$'])
fig.savefig('graphene_bands.png')
```

A magnetic field is added with `System.set_magnetic_field` (uniform field) or
`System.set_peierls_phase` (arbitrary vector potential), applied to the
hoppings *after* `set_hopping`/`set_hopping_manual`:

```python
sys.set_hopping_manual(hop_dict)
sys.set_magnetic_field(alpha=0.01)  # flux quanta per unit area
sys.get_ham()
```

See [`examples/magnetic_field/plot_magnetic_field.py`](examples/magnetic_field/plot_magnetic_field.py) for a full
worked example (an Aharonov-Bohm ring, reproducing the textbook result that
the spectrum is periodic in the enclosed flux with period one flux quantum).

A Chern number is the Berry curvature of a group of bands, integrated over
the Brillouin zone:

```python
chern = gra.chern_number(bands=[0], nk=40)   # ~0 for plain graphene
```

See [`examples/topology/plot_haldane_topology.py`](examples/topology/plot_haldane_topology.py) for the
Haldane model (the first Chern insulator), reproducing its topological
phase transition (Chern number 1 -> 0) and Berry-curvature map.

A spin-1/2 degree of freedom is added with `KSpace(lat, spin=True)`; onsite
values and hoppings then also accept 2x2 (Pauli) matrices:

```python
from tbkit.kspace import KSpace, PAULI

kmele = KSpace(lat, spin=True)
kmele.set_hopping([{'i': 0, 'j': 0, 'R': (1, 0), 't': 1j*lam*PAULI['z']}])  # intrinsic SOC
```

A ribbon -- periodic in one direction, finite in the other, the standard
way to see edge states -- is cut out of a periodic model with
`tbkit.kspace.ribbon`:

```python
from tbkit.kspace import ribbon

rib = ribbon(lat, list_hop, width=30, direction=1)   # 30 unit cells wide
fig = rib.plot_bands()
```

See [`examples/topology/plot_edge_states.py`](examples/topology/plot_edge_states.py) for the zigzag
graphene ribbon's zero-energy edge band, and the Kane-Mele ribbon's helical
edge states crossing a spin-orbit gap.

## Examples

`examples/` is organized as a [Sphinx-Gallery](https://sphinx-gallery.github.io/)
source tree, one topic per subfolder, each with a `README.rst` blurb. Every
`plot_*.py` script is self-contained and runnable directly
(`python examples/<section>/<script>.py`), and checks its own key numeric
claims with `assert` before plotting -- nothing is asserted in the docs
that isn't also verified in code. Building the docs (`pip install -e ".[docs]"`
then `cd docs && make html`) renders these same scripts into an executed,
thumbnailed example gallery under `docs/source/api/gallery/`.

| Script                                                                  | What it shows |
|---------------------------------------------------------------------------|----------------|
| [`tight_binding/plot_square_lattice_bands.py`](examples/tight_binding/plot_square_lattice_bands.py) | The square lattice: Bloch's theorem at its simplest, bands along Gamma-X-M-Gamma, the nested Fermi surface, and the van Hove singularity. |
| [`tight_binding/plot_graphene_bands.py`](examples/tight_binding/plot_graphene_bands.py) | Real-space flake + reciprocal-space band structure; graphene's Dirac point and Wallace's 1947 linear dispersion. |
| [`tight_binding/plot_visualizing_a_model.py`](examples/tight_binding/plot_visualizing_a_model.py) | `tbkit.plot.Plot`: lattice, spectrum with sublattice polarization, density of states, eigenstate intensity. |
| [`magnetic_field/plot_magnetic_field.py`](examples/magnetic_field/plot_magnetic_field.py) | Peierls substitution; an Aharonov-Bohm ring's flux-periodic spectrum. |
| [`magnetic_field/plot_hofstadter_butterfly.py`](examples/magnetic_field/plot_hofstadter_butterfly.py) | The fractal spectrum of a lattice threaded by a continuously swept flux. |
| [`magnetic_field/plot_landau_levels.py`](examples/magnetic_field/plot_landau_levels.py) | Landau levels: a square lattice's non-relativistic ladder vs. graphene's relativistic sqrt(n) ladder and zero mode. |
| [`topology/plot_ssh_model.py`](examples/topology/plot_ssh_model.py) | The SSH model: bulk gap closing and topologically protected edge states. |
| [`disorder/plot_anderson_localization.py`](examples/disorder/plot_anderson_localization.py) | Anderson localization: IPR vs. disorder strength, extended vs. localized states. |
| [`topology/plot_haldane_topology.py`](examples/topology/plot_haldane_topology.py) | The Haldane model: Berry curvature, Chern number, topological phase transition. |
| [`flat_bands/plot_flat_bands.py`](examples/flat_bands/plot_flat_bands.py) | Exactly flat bands on the kagome and Lieb lattices. |
| [`topology/plot_kagome_chern_band.py`](examples/topology/plot_kagome_chern_band.py) | Gapping the kagome flat band into a Chern insulator (C=-1) with complex nearest-neighbor hopping. |
| [`topology/plot_edge_states.py`](examples/topology/plot_edge_states.py) | Zigzag graphene ribbon edge band; Kane-Mele helical edge states. |
| [`dynamics/plot_bloch_oscillations.py`](examples/dynamics/plot_bloch_oscillations.py) | Wannier-Stark ladder, its localization, and Bloch oscillations under a uniform tilt. |
| [`topology/plot_thouless_pump.py`](examples/topology/plot_thouless_pump.py) | The Rice-Mele model as a Thouless quantum pump: quantized Chern number and polarization winding. |
| [`strain/plot_pseudo_magnetic_field.py`](examples/strain/plot_pseudo_magnetic_field.py) | Triaxial strain as a gauge field: pseudo-Landau levels in graphene from a purely real Hamiltonian. |
| [`non_hermitian/plot_pt_symmetry.py`](examples/non_hermitian/plot_pt_symmetry.py) | PT symmetry, exceptional points, the Petermann factor, and a selectively amplified topological edge mode. |
| [`tight_binding/plot_isolation_of_graphene.py`](examples/tight_binding/plot_isolation_of_graphene.py) | The physics behind the isolation of graphene: flakes, zigzag edge states, and the Dirac cones' Berry phase of pi. |
| [`magnetic_field/plot_peierls_substitution.py`](examples/magnetic_field/plot_peierls_substitution.py) | The Peierls substitution: gauge-invariant plaquette fluxes, in two different gauges. |
| [`topology/plot_zak_phase.py`](examples/topology/plot_zak_phase.py) | The SSH chain's Zak phase and Wannier centres. |
| [`topology/plot_tknn_hofstadter.py`](examples/topology/plot_tknn_hofstadter.py) | TKNN: Chern numbers of the Hofstadter bands from magnetic supercells, and the Diophantine equation. |
| [`topology/plot_kane_mele_z2.py`](examples/topology/plot_kane_mele_z2.py) | The Kane-Mele Z2 invariant from Wannier-centre flow. |
| [`topology/plot_fu_kane_parity.py`](examples/topology/plot_fu_kane_parity.py) | The Fu-Kane parity criterion on the BHZ model, checked against the Wannier flow. |
| [`topology/plot_quantum_geometry.py`](examples/topology/plot_quantum_geometry.py) | The quantum metric and Berry curvature of a Bloch band. |
| [`topology/plot_tenfold_way.py`](examples/topology/plot_tenfold_way.py) | Symmetry classes of tight-binding models: the tenfold way. |
| [`topology/plot_local_chern_marker.py`](examples/topology/plot_local_chern_marker.py) | The local Chern marker of a finite, disordered Haldane flake. |
| [`three_dimensions/plot_anderson_transition.py`](examples/three_dimensions/plot_anderson_transition.py) | The 3D Anderson metal-insulator transition from level statistics. |
| [`three_dimensions/plot_weyl_semimetal.py`](examples/three_dimensions/plot_weyl_semimetal.py) | A Weyl semimetal: Chern number jumps across Weyl points, and Fermi arcs on a slab. |
| [`orbitals/plot_slater_koster.py`](examples/orbitals/plot_slater_koster.py) | Slater-Koster sp3 graphene: the decoupled pi bands, a non-orthogonal basis, and the same model in real space. |
| [`large_scale/plot_kernel_polynomial_method.py`](examples/large_scale/plot_kernel_polynomial_method.py) | The kernel polynomial method on a very large (disordered) graphene flake. |
| [`transport/plot_landauer_conductance.py`](examples/transport/plot_landauer_conductance.py) | Landauer conductance, and the quantized steps of a quantum point contact. |
| [`correlations/plot_hubbard_edge_magnetism.py`](examples/correlations/plot_hubbard_edge_magnetism.py) | Hubbard mean field: magnetism of graphene's zigzag edges. |
| [`correlations/plot_lieb_theorem.py`](examples/correlations/plot_lieb_theorem.py) | Lieb's theorem: the total spin of bipartite Hubbard lattices. |
| [`superconductivity/plot_kitaev_chain.py`](examples/superconductivity/plot_kitaev_chain.py) | The Kitaev chain: BdG bands, the topological phase, and Majorana end modes. |
| [`non_hermitian/plot_skin_effect.py`](examples/non_hermitian/plot_skin_effect.py) | The non-Hermitian skin effect, the spectral winding, and the generalized Brillouin zone. |
| [`topology/plot_diabolical_points.py`](examples/topology/plot_diabolical_points.py) | Diabolical points: an accidental conical intersection in a two-parameter family, avoided crossings, and the eigenvector's sign change. |
| [`non_hermitian/plot_exceptional_points_fermi_arcs.py`](examples/non_hermitian/plot_exceptional_points_fermi_arcs.py) | Exceptional points in momentum space: Re E and Im E, EP charges, and the bulk Fermi arc. |
| [`non_hermitian/plot_exceptional_point_vorticity.py`](examples/non_hermitian/plot_exceptional_point_vorticity.py) | A Dirac point splits into an EP pair (vorticity conserved), or becomes an exceptional ring. |
| [`non_hermitian/plot_exceptional_point_encircling.py`](examples/non_hermitian/plot_exceptional_point_encircling.py) | Encircling an EP (state swap, four-loop return) versus a Dirac point (Berry phase pi). |
| [`non_hermitian/plot_exceptional_points_chern_number.py`](examples/non_hermitian/plot_exceptional_points_chern_number.py) | Biorthogonal Chern numbers (LR = RL = RR = LL) until the line gap closes at exceptional points. |
| [`floquet/plot_floquet_chern_insulator.py`](examples/floquet/plot_floquet_chern_insulator.py) | A Floquet Chern insulator from circularly polarized light on graphene. |
| [`floquet/plot_anomalous_floquet_phases.py`](examples/floquet/plot_anomalous_floquet_phases.py) | Anomalous Floquet phases: edge states with zero Chern numbers, the winding number, and the five-step model. |
| [`hall_effects/plot_anomalous_hall_effect.py`](examples/hall_effects/plot_anomalous_hall_effect.py) | The anomalous Hall effect: sigma_xy(E_F) of the Haldane model, quantized at C in the gap, not in the bands. |
| [`hall_effects/plot_anomalous_hall_disorder.py`](examples/hall_effects/plot_anomalous_hall_disorder.py) | The anomalous Hall plateau of a disordered Haldane torus, by the Kubo-Bastin kernel polynomial method. |
| [`hall_effects/plot_intrinsic_spin_hall_effect.py`](examples/hall_effects/plot_intrinsic_spin_hall_effect.py) | The intrinsic spin Hall effect: the Kane-Mele plateau at e/2pi, and its departure from quantization under Rashba coupling. |

The `examples/` directory also has five older Jupyter notebooks (graphene
flakes, kagome/Lieb/dumbbell lattices, disorder, strain, time propagation)
predating the 0.2 API refresh below.

## Documentation

Rendered docs (tutorial, API reference, example gallery): https://cpoli.github.io/tbkit/

* [`docs/source/tutorial.rst`](docs/source/tutorial.rst) -- a narrative walkthrough of the
  package, from building a lattice through topology, spin-orbit coupling,
  and edge states.
* [`docs/source/history.rst`](docs/source/history.rst) -- a chronology of the breakthroughs
  behind Tight-Binding theory (Bloch's theorem through the Kane-Mele model),
  each one linked to the corresponding **tbkit** functionality and example
  above.
* `docs/source/tbkit.rst` -- the API reference (auto-generated from docstrings).

Build the HTML docs with `cd docs && make html` (output in `docs/build/html`).

## A note on the API

Version 0.2 modernized the package to run on current Python/NumPy/SciPy and
cleaned up the API:

* Sublattice tags are plain one-character **strings** (`'a'`) rather than
  byte strings (`b'a'`).
* Classes are named in `PascalCase` (`Lattice`, `System`, ...) rather than
  lowercase names identical to their module (`lattice.lattice`,
  `system.system`, ...), which used to make `import tbkit.lattice as lattice`
  silently bind the wrong object.

For continuity, the pre-0.2 lowercase class names (`lattice`, `system`,
`plot`, `propagation`, `save`) remain available as aliases of the new
classes, so `from tbkit.lattice import lattice` still works. Example
notebooks predating 0.2 still use byte-string tags (`b'a'`) and will need
that one mechanical change to run on the current version.

## License

BSD 3-Clause, see [LICENSE](LICENSE).
