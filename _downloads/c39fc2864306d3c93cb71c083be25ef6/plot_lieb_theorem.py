r"""
Lieb's Theorem: The Total Spin of Graphene Triangles
==========================================================

E. Lieb proved in 1989 one of the few exact results on the Hubbard
model: on a bipartite lattice at half filling, for any repulsion
:math:`U > 0`, the ground state has total spin

.. math::

    S = \frac{\bigl||A| - |B|\bigr|}{2}\, ,

set by the imbalance between the two sublattices alone. A triangular
graphene flake with zigzag edges has all three edges on the same
sublattice, and an imbalance that grows with its size: it must be a
high-spin magnet, from nothing but its shape -- the physics of the
"triangulenes", synthesized and seen to be magnetic in 2017-2021.

The unrestricted mean-field solution
(:func:`~tbkit.meanfield.hubbard_mean_field`) is not exact, but its
lowest-energy state obeys Lieb's count: :math:`S_z = ||A| - |B||/2`
exactly, for every size and every :math:`U`. (Other self-consistent
solutions exist, of lower spin and higher energy: random starts often
find those, so several starts are compared.)
"""
import numpy as np
import matplotlib.pyplot as plt

from tbkit.graphene import GrapheneLattice
from tbkit.meanfield import hubbard_mean_field
from tbkit.system import System


def triangle(n):
    lat = GrapheneLattice()
    lat.triangle_zigzag(n)
    sys = System(lat)
    sys.set_hopping([{'n': 1, 't': -1.}])
    sys.get_ham()
    return lat, sys


def ground_state(sys, U, starts=4):
    '''Lowest-energy mean-field solution over several starts: random ones,
    and a staggered magnetization (up on A, down on B).'''
    a = sys.lat.coor['tag'] == 'a'
    up = np.where(a, 0.7, 0.3)
    results = [hubbard_mean_field(sys.ham, U, sys.lat.sites, n_up=up, n_dn=1 - up)]
    results += [hubbard_mean_field(sys.ham, U, sys.lat.sites, seed=seed) for seed in range(starts)]
    return min(results, key=lambda r: r.energy)


# %%
# Lieb's count holds for every size and U
# ---------------------------------------

rows = []
for n in (2, 3, 4, 5):
    lat, sys = triangle(n)
    n_a = int(np.sum(lat.coor['tag'] == 'a'))
    lieb = abs(2 * n_a - lat.sites) / 2
    for U in (1., 3.):
        s_z = abs(ground_state(sys, U).total_spin)
        rows.append((n, lat.sites, n_a, lat.sites - n_a, U, s_z))
        assert np.isclose(s_z, lieb, atol=1e-6)
print('  n  sites  |A|  |B|    U    S_z')
for row in rows:
    print('{:3d} {:6d} {:4d} {:4d} {:4.1f} {:6.2f}'.format(*row))

# %%
# Where the moment sits
# ------------------------

lat, sys = triangle(5)
m = ground_state(sys, 2.).magnetization
fig, ax = plt.subplots(figsize=(5.5, 5))
sc = ax.scatter(lat.coor['x'], lat.coor['y'], c=m, cmap='RdBu', vmin=-0.3, vmax=0.3, s=60)
fig.colorbar(sc, ax=ax, label='$m_i$')
ax.set_aspect('equal')
ax.axis('off')
ax.set_title('Triangular flake, {} sites, $U = 2|t|$: $S = {:.1f}$'.format(lat.sites, abs(m.sum())))
n_a = int(np.sum(lat.coor['tag'] == 'a'))
assert np.isclose(abs(m.sum()), abs(2 * n_a - lat.sites) / 2, atol=1e-6)
# the moment sits mostly on the majority sublattice, which carries the edges
assert m[lat.coor['tag'] == 'a'].sum() * np.sign(m.sum()) > abs(m.sum())
