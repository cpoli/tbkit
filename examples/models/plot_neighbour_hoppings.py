r"""
Hoppings by Neighbour Order in k-Space: Third-Nearest-Neighbour Graphene
=========================================================================

Real-space models in tbkit address bonds by neighbour order: ``{'n': 1}``
for nearest neighbours, ``{'n': 2}`` for next-nearest ones, optionally
narrowed by bond angle ``'ang'`` or sublattice pair ``'tag'``.
:meth:`~tbkit.kspace.KSpace.set_hopping` (and
:meth:`~tbkit.kspace.KSpace.set_overlap`) accept the same selectors: the
bonds of the infinite lattice are found from the unit cell and the
primitive vectors (:func:`tbkit.neighbours.neighbour_bonds`), and turned
into the explicit ``{'i', 'j', 'R', 't'}`` list
(:func:`tbkit.neighbours.neighbour_hoppings`).

This makes a realistic model a three-line affair. The example builds the
third-nearest-neighbour tight-binding fit of graphene's :math:`\pi` bands
by S. Reich, J. Maultzsch, C. Thomsen and P. Ordejon, Phys. Rev. B 66,
035412 (2002), with its non-orthogonal overlaps, and checks it against the
closed-form energies at the high-symmetry points.
"""
import numpy as np
import matplotlib.pyplot as plt

from tbkit import lattices
from tbkit.kspace import KSpace
from tbkit.neighbours import neighbour_bonds, neighbour_hoppings


# %%
# The neighbour shells of the honeycomb lattice
# ---------------------------------------------------
# :func:`~tbkit.neighbours.neighbour_bonds` lists every bond of the first
# shells once (oriented so that its angle lies in [0, 180), as in
# :class:`~tbkit.system.System`): 3, 6 and 3 neighbours per site at
# distances :math:`a`, :math:`\sqrt{3}a` and :math:`2a` -- the nearest and
# third neighbours on the other sublattice, the second on the same one.

lat = lattices.honeycomb()  # nearest-neighbour distance a = 1
bonds = neighbour_bonds(lat, 3)
per_cell = [int(np.sum(bonds['n'] == n)) for n in (1, 2, 3)]
lengths = [float(bonds['dis'][bonds['n'] == n][0]) for n in (1, 2, 3)]
print('bonds per unit cell in shells 1, 2, 3:', per_cell)
print('shell distances:', np.round(lengths, 4))
# 2 sites per cell, each bond shared by 2 sites: neighbours per site = bonds per cell
assert per_cell == [3, 6, 3]
assert np.allclose(lengths, [1., np.sqrt(3.), 2.])
assert set(bonds['tag'][bonds['n'] == 2]) == {'aa', 'bb'}

# %%
# Nearest neighbours: the same as the explicit list
# ----------------------------------------------------
# ``{'n': 1, 't': t}`` generates exactly the three bonds one would write by
# hand.

g0 = -2.97
explicit = KSpace(lat)
explicit.set_hopping([{'i': 0, 'j': 1, 'R': (0, 0), 't': g0},
                             {'i': 0, 'j': 1, 'R': (-1, 0), 't': g0},
                             {'i': 0, 'j': 1, 'R': (0, -1), 't': g0}])
nn = KSpace(lat)
nn.set_hopping([{'n': 1, 't': g0}])
print(neighbour_hoppings(lat, [{'n': 1, 't': g0}]))
rng = np.random.default_rng(0)
for k in rng.uniform(-4., 4., (20, 2)):
    assert np.allclose(nn.get_ham(k), explicit.get_ham(k), atol=1e-14)

# %%
# The third-nearest-neighbour fit of Reich et al.
# ----------------------------------------------------
# Hoppings :math:`\gamma_0, \gamma_1, \gamma_2` and overlaps
# :math:`s_0, s_1, s_2` for the first three shells, and the on-site energy
# :math:`\epsilon_{2p}` (eV):

eps, (g1, g2) = -0.28, (-0.073, -0.33)
s0, s1, s2 = 0.073, 0.018, 0.026
gra = KSpace(lat)
gra.set_onsite({'a': eps, 'b': eps})
gra.set_hopping([{'n': 1, 't': g0}, {'n': 2, 't': g1}, {'n': 3, 't': g2}])
gra.set_overlap([{'n': 1, 't': s0}, {'n': 2, 't': s1}, {'n': 3, 't': s2}])

# %%
# At :math:`\Gamma` every Bloch sum is the number of neighbours, so the
# generalized eigenproblem :math:`Hv = ESv` gives
# :math:`E_\pm = (\epsilon + 6\gamma_1 \pm 3(\gamma_0+\gamma_2)) / (1 + 6s_1 \pm 3(s_0+s_2))`.
# At K the first- and third-neighbour sums vanish (the Dirac point
# survives), and the six second neighbours add up to :math:`-3`: both
# bands meet at :math:`E_K = (\epsilon - 3\gamma_1)/(1 - 3s_1)`, which is
# no longer zero -- second neighbours break electron-hole symmetry.

b1, b2 = np.array(gra.rec_vec)
Gamma, K, M = np.zeros(2), (b1 - b2) / 3, b1 / 2
E_gamma = np.sort([(eps + 6*g1 + s * 3*(g0 + g2)) / (1 + 6*s1 + s * 3*(s0 + s2)) for s in (1, -1)])
E_K = (eps - 3*g1) / (1 - 3*s1)
en_gamma = gra.get_bands(Gamma)[0]
en_K = gra.get_bands(K)[0]
print('E(Gamma) = {} eV (closed form {})'.format(np.round(en_gamma, 4), np.round(E_gamma, 4)))
print('E(K)     = {} eV (closed form {:.4f})'.format(np.round(en_K, 4), E_K))
assert np.allclose(en_gamma, E_gamma, atol=1e-12)
assert np.allclose(en_K, E_K, atol=1e-12)
assert abs(E_K - (-0.0645)) < 1e-4

# %%
# Band structures
# ---------------------
# Nearest neighbours only (orthogonal, :math:`\pm 3|\gamma_0|` at
# :math:`\Gamma`) against the full fit: the :math:`\pi^*` band is pushed up
# and the :math:`\pi` band compressed by the overlaps.

points = [Gamma, K, M, Gamma]
labels = [r'$\Gamma$', 'K', 'M', r'$\Gamma$']
dist, en_nn = nn.k_path(points, nk=80)
_, en_3nn = gra.k_path(points, nk=80)
assert np.allclose(en_nn[0], [3*g0, -3*g0])
# at Gamma: pi* above +3|g0|, pi above -3|g0| (a narrower valence band)
assert en_3nn[0, 1] > -3*g0 and en_3nn[0, 0] > 3*g0

fig, ax = plt.subplots(figsize=(6, 4.5))
ax.plot(dist, en_nn, color='0.6', lw=1.5)
ax.plot(dist, en_3nn, color='C3', lw=2)
ax.plot([], [], color='0.6', label='nearest neighbours, orthogonal')
ax.plot([], [], color='C3', label='3rd neighbours + overlaps (Reich et al.)')
for x in gra.nodes:
    ax.axvline(x, color='k', lw=0.5)
ax.set_xticks(gra.nodes)
ax.set_xticklabels(labels)
ax.set_xlim(dist[0], dist[-1])
ax.set_ylabel('$E$ (eV)')
ax.set_title('Graphene by neighbour order in k-space')
ax.legend(loc='lower right', fontsize=8)
plt.show()
