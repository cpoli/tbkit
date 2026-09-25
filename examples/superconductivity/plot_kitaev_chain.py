r"""
The Kitaev Chain: Majorana Zero Modes
===========================================

A. Kitaev (2001) wrote down the simplest topological superconductor: a
chain of spinless fermions with nearest-neighbour hopping :math:`t` and
p-wave pairing :math:`\Delta`,

.. math::

    H = \sum_j\Big[t\,c_j^\dagger c_{j+1} + \Delta\,c_j^\dagger c_{j+1}^\dagger + h.c.\Big]
        - \mu\sum_j c_j^\dagger c_j\, .

In the Bogoliubov-de Gennes form its bulk spectrum is
:math:`E(k) = \pm\sqrt{(2t\cos k - \mu)^2 + 4|\Delta|^2\sin^2k}`, gapped
except at :math:`|\mu| = 2|t|`. For :math:`|\mu| < 2|t|` the chain is
topological: an open chain hosts one *Majorana* zero mode at each end
-- a fermion split in two halves, a building block proposed for
topological quantum computation, and chased in semiconductor nanowires
since 2012. The bulk invariant is the Berry (Zak) phase of the occupied
BdG band, :math:`\pi` in the topological phase and 0 otherwise (class
BDI).

:mod:`tbkit.bdg` builds the chain in real space (*bdg_ham*) and in
reciprocal space (*bdg_kspace*).
"""
import numpy as np
import matplotlib.pyplot as plt

import tbkit.lattices as lattices
from tbkit.bdg import bdg_ham, bdg_kspace, pairing_bonds, particle_hole
from tbkit.kspace import KSpace
from tbkit.system import System

t, delta = 1., 0.6


def kitaev(mu):
    chain = KSpace(lattices.chain())
    chain.set_hopping([{'i': 0, 'j': 0, 'R': (1,), 't': t}])
    return bdg_kspace(chain, [{'i': 0, 'j': 0, 'R': (1,), 'delta': delta}], mu=mu)


# %%
# The bulk spectrum and the invariant
# ---------------------------------------

for mu in (0.5, 1.5, 3.):
    for k in (0.4, 1.3, 2.7):
        e = np.sqrt((2*t*np.cos(k) - mu)**2 + 4*delta**2*np.sin(k)**2)
        assert np.allclose(np.linalg.eigvalsh(kitaev(mu).get_ham([k])), [-e, e])
assert kitaev(0.5).symmetry_error(particle_hole(1), antiunitary=True, anti=True) < 1e-12
print('Class:', kitaev(0.5).tenfold_class(time_reversal=np.eye(2), particle_hole=particle_hole(1)))

mus = np.linspace(-3.5, 3.5, 29)
zak = np.array([abs(kitaev(mu).berry_phase(0, nk=200)) for mu in mus])
topological = np.abs(mus) < 2 * abs(t)
assert np.allclose(zak[topological], np.pi) and np.allclose(zak[~topological], 0., atol=1e-8)
print('Berry phase pi for |mu| < 2|t|, 0 outside.')

# %%
# Majorana zero modes at the ends of an open chain
# ------------------------------------------------------

n = 40
lat = lattices.chain()
lat.get_lattice(n)
sys = System(lat)
sys.set_hopping([{'n': 1, 't': t}])
sys.get_ham()
spectra = {}
for mu in (0.5, 3.):
    en, vec = np.linalg.eigh(bdg_ham(sys.ham, pairing_bonds(sys, delta), mu=mu).toarray())
    spectra[mu] = en
    assert np.allclose(en, -en[::-1])  # particle-hole symmetric
zero = np.argsort(np.abs(spectra[0.5]))[:2]
print('mu = 0.5: two states at |E| = {:.1e}; next at {:.3f}.'
          .format(np.max(np.abs(spectra[0.5][zero])), np.sort(np.abs(spectra[0.5]))[2]))
assert np.max(np.abs(spectra[0.5][zero])) < 1e-8 and np.min(np.abs(spectra[3.])) > 0.5
# the zero modes live at the ends (particle and hole components summed)
en, vec = np.linalg.eigh(bdg_ham(sys.ham, pairing_bonds(sys, delta), mu=0.5).toarray())
weight = np.abs(vec[:n, zero]) ** 2 + np.abs(vec[n:, zero]) ** 2
ends = weight[:5].sum() + weight[-5:].sum()
print('Weight of the two zero modes on the outer 5 sites of each end: {:.0%}.'.format(ends / 2))
assert ends / 2 > 0.95
# and the same chain from the Bloch model
assert np.allclose(spectra[0.5], np.linalg.eigvalsh(kitaev(0.5).finite_ham(n)))

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].plot(mus, zak / np.pi, 'o-b')
axes[0].set_xlabel(r'$\mu/t$')
axes[0].set_ylabel(r'Berry phase / $\pi$')
axes[1].plot(np.arange(n), weight.sum(axis=1), 'o-r')
axes[1].set_xlabel('site')
axes[1].set_ylabel('zero-mode weight')
axes[1].set_title(r'Majorana end modes, $\mu = 0.5t$')
fig.set_layout_engine('tight')
