# tbkit — a Tight-Binding package

| | |
|:--|:-:|
| Package | [![PyPI version](https://img.shields.io/pypi/v/tbkit)](https://pypi.org/project/tbkit/) [![Python versions](https://img.shields.io/pypi/pyversions/tbkit)](https://pypi.org/project/tbkit/) [![DOI](https://zenodo.org/badge/1381398706.svg)](https://zenodo.org/badge/latestdoi/1381398706) |
| Quality | [![License](https://img.shields.io/github/license/cpoli/tbkit)](https://github.com/cpoli/tbkit/blob/master/LICENSE) [![CI](https://github.com/cpoli/tbkit/actions/workflows/tests.yml/badge.svg)](https://github.com/cpoli/tbkit/actions/workflows/tests.yml) [![Coverage](https://img.shields.io/badge/coverage-100%25-brightgreen)](https://cpoli.github.io/tbkit/coverage/) |
| Documentation | [![Docs](https://img.shields.io/badge/docs-cpoli.github.io%2Ftbkit-blue)](https://cpoli.github.io/tbkit/) |
| Downloads | [![Downloads](https://static.pepy.tech/badge/tbkit)](https://pepy.tech/project/tbkit) [![Downloads/Month](https://static.pepy.tech/badge/tbkit/month)](https://pepy.tech/project/tbkit) |
| Community | [![GitHub Stars](https://img.shields.io/github/stars/cpoli/tbkit?style=social)](https://github.com/cpoli/tbkit) [![GitHub Forks](https://img.shields.io/github/forks/cpoli/tbkit?style=social)](https://github.com/cpoli/tbkit) [![Contributors](https://img.shields.io/github/contributors/cpoli/tbkit)](https://github.com/cpoli/tbkit/graphs/contributors) [![Last Commit](https://img.shields.io/github/last-commit/cpoli/tbkit)](https://github.com/cpoli/tbkit/commits/master) |

![tbkit logo](https://raw.githubusercontent.com/cpoli/tbkit/master/docs/source/_static/image/tbkit_logo.png)

**tbkit** is a Python package to build and solve Tight-Binding models, written
in vectorized NumPy/SciPy. It aims to make the mechanics of Tight-Binding
models — lattices, hoppings, Hamiltonians, spectra, band structures —
explicit and easy to inspect, so it works as well for teaching as for
research prototyping.

![Graphene's Dirac cone, Hofstadter's butterfly, and helical edge states of a Kane-Mele ribbon, all computed with tbkit](https://raw.githubusercontent.com/cpoli/tbkit/master/docs/source/_static/images/readme_hero.png)

For computational physics beyond tight-binding — quantum mechanics,
classical mechanics, statistical physics, general relativity, and more —
see [physicskit](https://github.com/cpoli/physicskit), whose
`physicskit.condensed` subpackage covers tight-binding band theory and
topology alongside superconductivity.

## Examples

### Building and solving models

Finite lattices, defects, several orbitals per site, real space to k-space and back, k·p models discretized into lattice models, saved models and Wannier90 imports, maximally localized Wannier functions, and solvers for very large lattices.

![An impurity bound state, Slater-Koster sp3 graphene bands with overlaps, and the kernel polynomial method on 45,000 sites](https://raw.githubusercontent.com/cpoli/tbkit/master/docs/source/_static/images/readme_models.png)

### Fields, strain and disorder

What happens to a lattice under a magnetic field (Peierls substitution), under strain, and with disorder.

![Graphene's sqrt(n) Landau ladder, triaxial strain as a pseudo-magnetic field, and Anderson localization](https://raw.githubusercontent.com/cpoli/tbkit/master/docs/source/_static/images/readme_fields.png)

### Band topology

Chern numbers, Berry phases, the Z2 invariant (and the four 3D indices), spin and mirror Chern numbers, the Bott index and entanglement spectrum, Weyl points and their chirality, Kitaev's Majorana number, quantum geometry, symmetry classes, higher-order and 3D topological phases.

![Berry curvature of the Haldane model, the Kane-Mele Z2 Wannier flow, and the four corner states of a quadrupole insulator](https://raw.githubusercontent.com/cpoli/tbkit/master/docs/source/_static/images/readme_topology.png)

### Response and transport

Hall conductivities at any Fermi level (with the orbital magnetization, Nernst and thermal Hall responses), Landauer transport between leads, and linear and nonlinear optical response.

![The anomalous Hall conductivity of the Haldane model, the conductance steps of a quantum point contact, and graphene's universal absorption](https://raw.githubusercontent.com/cpoli/tbkit/master/docs/source/_static/images/readme_response.png)

### Flat bands and interactions

Flat bands from lattice geometry and from a moiré twist, the Hubbard model in mean field, and superconductivity.

![The flat bands of twisted bilayer graphene, magnetic zigzag edges in Hubbard mean field, and the Majorana end modes of a Kitaev chain](https://raw.githubusercontent.com/cpoli/tbkit/master/docs/source/_static/images/readme_interactions.png)

### Driven and open systems

Wavepacket dynamics, periodically driven (Floquet) lattices, and non-Hermitian bands with exceptional points.

![Bloch oscillations in a tilted chain, the Floquet bands of graphene in circularly polarized light, and the non-Hermitian skin effect](https://raw.githubusercontent.com/cpoli/tbkit/master/docs/source/_static/images/readme_driven.png)

## Install

Requires Python >= 3.10 (tested on 3.10-3.15).

```bash
pip install tbkit
```

`tbkit.continuum` (k·p models to tight-binding) also needs sympy:

```bash
pip install "tbkit[continuum]"
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
* [`ROADMAP.md`](ROADMAP.md) -- what is planned next, and what tbkit
  does not do yet (compared with Kwant and PythTB).

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
classes, so `from tbkit.lattice import lattice` still works.

## License

BSD 3-Clause, see [LICENSE](LICENSE).
