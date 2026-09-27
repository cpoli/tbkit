r"""
Defects and Impurities in a Graphene Flake: Vacancy Zero Modes and Bound States
===================================================================================

Local modifications of a finite model -- the business of
:meth:`~tbkit.lattice.Lattice.remove_sites`,
:meth:`~tbkit.system.System.set_onsite_def` and
:meth:`~tbkit.system.System.change_hopping_ellipse` -- and two exact
consequences on a bipartite lattice:

* **A vacancy binds a zero mode.** With nearest-neighbour hoppings the
  spectrum of a bipartite lattice is symmetric, and at least
  :math:`\bigl||A|-|B|\bigr|` states sit exactly at :math:`E = 0`, on the
  majority sublattice. Removing one site of a balanced flake leaves one.
* **A deep impurity binds a state below the band.** An onsite energy
  :math:`\epsilon \ll -3|t|` pulls one state out of the band
  (:math:`|E| \le 3|t|`), localized on the impurity, near
  :math:`E \approx \epsilon + 3t^2/\epsilon`.

(With *plot_building_finite_lattices.py* and
*plot_wavepacket_interference.py*, this script replaces the pre-0.2
notebooks ``examples_system.ipynb`` and ``examples_graphene.ipynb``.)
"""
import numpy as np
import matplotlib.pyplot as plt

import tbkit.lattices as lattices
from tbkit.system import System


def flake():
    lat = lattices.honeycomb()
    lat.get_lattice(n1=8, n2=8)
    return lat


# %%
# A vacancy
# -----------

lat = flake()
assert np.sum(lat.coor['tag'] == 'a') == np.sum(lat.coor['tag'] == 'b')
center = np.array([lat.coor['x'].mean(), lat.coor['y'].mean()])
vac = int(np.argmin(np.hypot(lat.coor['x'] - center[0], lat.coor['y'] - center[1])))
removed_tag = lat.coor['tag'][vac]
lat.remove_sites([vac])
sys = System(lat)
sys.set_hopping([{'n': 1, 't': -1.}])
sys.get_ham()
en, vec = np.linalg.eigh(sys.ham.toarray())
n_a, n_b = np.sum(lat.coor['tag'] == 'a'), np.sum(lat.coor['tag'] == 'b')
zero = np.abs(en) < 1e-9
assert np.allclose(en, -en[::-1], atol=1e-10)  # chiral symmetry
assert zero.sum() >= abs(n_a - n_b) == 1
zm = (np.abs(vec[:, zero]) ** 2).sum(axis=1)
majority = 'b' if removed_tag == 'a' else 'a'
assert np.isclose(zm[lat.coor['tag'] == majority].sum(), zero.sum())
print('{} zero mode(s), entirely on sublattice {}'.format(zero.sum(), majority))

# %%
# A deep onsite impurity, and a region of stronger bonds
# ---------------------------------------------------------

lat2 = flake()
sys2 = System(lat2)
sys2.set_hopping([{'n': 1, 't': -1.}])
sys2.set_onsite({'a': 0., 'b': 0.})
imp = int(np.argmin(np.hypot(lat2.coor['x'] - center[0], lat2.coor['y'] - center[1])))
eps = -8.
sys2.set_onsite_def({imp: eps})
sys2.get_ham()
en2, vec2 = np.linalg.eigh(sys2.ham.toarray())
bound = np.abs(vec2[:, 0]) ** 2
assert en2[0] < -3. and en2[1] > -3.
assert abs(en2[0] - (eps + 3. / eps)) < 0.1 and bound[imp] > 0.8
print('bound state at E = {:.3f} (estimate {:.3f}), weight {:.2f} on the impurity'.format(
    en2[0], eps + 3. / eps, bound[imp]))

# stronger hoppings inside a disk shift the band edge, but keep the symmetry
sys3 = System(flake())
sys3.set_hopping([{'n': 1, 't': -1.}])
sys3.change_hopping_ellipse([{'n': 1, 't': -1.5}], rx=3., ry=3., x0=center[0], y0=center[1])
sys3.get_ham()
en3 = np.linalg.eigvalsh(sys3.ham.toarray())
sys0 = System(flake())
sys0.set_hopping([{'n': 1, 't': -1.}])
sys0.get_ham()
assert np.allclose(en3, -en3[::-1], atol=1e-10)
assert en3.max() > np.linalg.eigvalsh(sys0.ham.toarray()).max()

fig, axes = plt.subplots(1, 3, figsize=(13, 4))
sc = axes[0].scatter(lat.coor['x'], lat.coor['y'], c=zm, s=25, cmap='viridis')
axes[0].plot(*center, 'rx', ms=10)
axes[0].set_title('vacancy: zero-mode weight')
fig.colorbar(sc, ax=axes[0])
sc = axes[1].scatter(lat2.coor['x'], lat2.coor['y'], c=bound, s=25, cmap='viridis', vmax=0.05)
axes[1].set_title('impurity bound state, $E = {:.2f}$'.format(en2[0]))
fig.colorbar(sc, ax=axes[1])
for ax in axes[:2]:
    ax.set_aspect('equal')
    ax.axis('off')
axes[2].plot(en2, 'k.', ms=3, label=r'impurity $\epsilon = -8$')
axes[2].plot(en3, '.', c='C1', ms=3, label='stronger bonds in a disk')
axes[2].set_xlabel('state')
axes[2].set_ylabel('$E/|t|$')
axes[2].legend()
fig.tight_layout()
