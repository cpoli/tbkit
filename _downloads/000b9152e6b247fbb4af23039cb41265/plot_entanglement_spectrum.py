r"""
The Entanglement Spectrum: Edge States Without an Edge
=============================================================

Cut a system in two halves :math:`A` and :math:`B`, and ask how the ground
state entangles them. The answer is the reduced density matrix
:math:`\rho_A = e^{-H_E}/Z`, and H. Li and F. D. M. Haldane (2008) showed
that the spectrum of the *entanglement Hamiltonian* :math:`H_E` of a
topological phase looks like its edge spectrum, as if the cut were a
physical edge. For free fermions :math:`H_E` is quadratic (Peschel 2003):
its single-particle levels follow from the eigenvalues :math:`\xi_n` of the
correlation matrix :math:`C_{ij} = \langle c_i^\dagger c_j\rangle` restricted
to :math:`A`. States that live entirely in :math:`A` or in :math:`B` give
:math:`\xi = 1` or 0. Values inside :math:`(0, 1)` come from states shared
across the cuts, and topology forces some of them there (Fidkowski 2010;
Turner, Zhang and Vishwanath 2010).

* **SSH ring.** Each cut of a topological chain leaves one entanglement
  mode pinned at :math:`\xi = 1/2` exactly, half a fermion on each side.
  Chiral symmetry pins it, much as it pins the zero modes at the physical
  ends of an open chain.
* **Chern insulator on a cylinder.** Resolved by the momentum along the
  cut, the entanglement spectrum of the Haldane model holds branches that
  flow across the whole interval (0, 1) as the momentum crosses the zone,
  like the chiral edge state at a physical edge.

There is no physical edge in either case: both samples are periodic. The
diagnostic depends on the ground state alone, and not on the basis of
the Hamiltonian. :func:`~tbkit.topology.entanglement_spectrum` handles finite
samples, :meth:`~tbkit.kspace.KSpace.entanglement_spectrum` the k-resolved
case.
"""
import numpy as np
import matplotlib.pyplot as plt

import tbkit.lattices as lattices
from tbkit.kspace import KSpace
from tbkit.lattice import Lattice
from tbkit.moire import supercell
from tbkit.topology import entanglement_spectrum


def ssh(v, w):
    lat = Lattice(unit_cell=[{'tag': 'a', 'r0': (0., 0.)}, {'tag': 'b', 'r0': (0.5, 0.)}],
                         prim_vec=[(1., 0.)])
    chain = KSpace(lat)
    chain.set_hopping([{'i': 0, 'j': 1, 'R': (0,), 't': v}, {'i': 1, 'j': 0, 'R': (1,), 't': w}])
    return chain


def haldane(t2=0.2, M=0.):
    hal = KSpace(lattices.honeycomb())
    hal.set_hopping([{'i': 0, 'j': 1, 'R': R, 't': 1.} for R in [(0, 0), (-1, 0), (0, -1)]])
    for R in [(0, 1), (-1, 0), (1, -1)]:
        hal.set_hopping([{'i': 0, 'j': 0, 'R': R, 't': 1j*t2}, {'i': 1, 'j': 1, 'R': R, 't': -1j*t2}])
    hal.set_onsite({'a': M, 'b': -M})
    return hal


# %%
# The SSH ring: a mode at 1/2 at each cut
# -------------------------------------------
# A ring of 40 cells (periodic, so no end states), cut into two halves of
# 20 cells: two cuts.

n_cells = 40
half = list(range(n_cells))  # the 40 orbitals of the first 20 cells
xi_ssh = {}
for (v, w), name in (((1., 0.4), 'trivial'), ((0.4, 1.), 'topological')):
    ring = ssh(v, w).finite_ham(n_cells, periodic=True)
    xi_ssh[name] = entanglement_spectrum(ring, half)
mid = np.sort(np.abs(xi_ssh['topological'] - 0.5))
print('topological: |xi - 1/2| of the two closest modes {:.1e}, next {:.3f}'.format(mid[1], mid[2]))
print('trivial: closest mode to 1/2 at |xi - 1/2| = {:.3f}'.format(np.min(np.abs(xi_ssh['trivial'] - 0.5))))
assert mid[1] < 1e-10 and mid[2] > 0.4
assert np.min(np.abs(xi_ssh['trivial'] - 0.5)) > 0.4

# %%
# The Haldane cylinder: spectral flow
# ---------------------------------------
# A supercell of 12 cells along :math:`\mathbf{a}_2`
# (:func:`~tbkit.moire.supercell`, periodic in both directions), cut in two
# halves of 6 cells. As :math:`k` runs along :math:`\mathbf{a}_1`, the
# entanglement modes of the two cuts (never more than two at a time
# between 0 and 1) flow across the interval, and together fill all of it.
# In the trivial phase (:math:`M = 2`) the spectrum stays away from 1/2.

n_super = 12
fracs = np.linspace(-0.5, 0.5, 201)
xi_cyl = {}
for M, name in ((0., 'Chern'), (2., 'trivial')):
    sc = supercell(haldane(M=M), [[1, 0], [0, n_super]])
    k = fracs[:, None] * sc.rec_vec_k[0][None, :]
    occupied = list(range(sc.norb // 2))
    region = list(range(sc.norb // 2))  # the orbitals of the first 6 cells
    xi_cyl[name] = sc.entanglement_spectrum(k, region, occupied)
window = (xi_cyl['Chern'] > 0.1) & (xi_cyl['Chern'] < 0.9)
flow = np.sort(xi_cyl['Chern'][window])
hole = np.max(np.diff(np.concatenate([[0.1], flow, [0.9]])))
print('Chern phase: the modes inside (0.1, 0.9) leave no hole wider than {:.3f}.'.format(hole))
assert hole < 0.05 and window.sum(axis=1).max() == 2
in_trivial = np.sum((xi_cyl['trivial'] > 0.1) & (xi_cyl['trivial'] < 0.9))
print('trivial phase: {} modes inside (0.1, 0.9).'.format(in_trivial))
assert in_trivial == 0

fig, axes = plt.subplots(1, 3, figsize=(13, 4))
for name, marker, color in (('trivial', 's', 'gray'), ('topological', 'o', 'b')):
    axes[0].plot(np.sort(xi_ssh[name]), marker, color=color, ms=4, label=name)
axes[0].axhline(0.5, color='r', lw=0.5)
axes[0].set_xlabel('index')
axes[0].set_ylabel(r'$\xi$')
axes[0].set_title('SSH ring, half the chain')
axes[0].legend()
for ax, name in zip(axes[1:], ('Chern', 'trivial')):
    ax.plot(fracs * 2 * np.pi, xi_cyl[name], '.', color='b', ms=1.5)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel(r'$k_1$')
    ax.set_title('Haldane cylinder: {} phase'.format(name))
axes[1].set_ylabel(r'$\xi(k_1)$')
fig.set_layout_engine('tight')
