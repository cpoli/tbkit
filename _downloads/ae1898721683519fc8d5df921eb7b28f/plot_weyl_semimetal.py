r"""
Weyl Semimetals: Monopoles of Berry Curvature and Fermi Arcs
==================================================================

A 3D band touching where two bands meet linearly in every direction,
:math:`H \approx \mathbf{v}\cdot\mathbf{q}\,\boldsymbol\sigma`, is a
*Weyl point*: a source or sink of Berry curvature, of charge
:math:`\pm1`. Wan, Turner, Vishwanath and Savrasov (2011) predicted Weyl
semimetals in real materials, and in 2015 Xu et al. and Lv et al. observed
them in TaAs by photoemission -- through their unmistakable surface
signature, open *Fermi arcs*.

Both follow from topology. Every plane :math:`k_z = \mathrm{const}` of the
Brillouin zone is a 2D insulator with a Chern number, which jumps by the
monopole charge (the chirality) as the plane crosses a Weyl point: here
:math:`|C(k_z)| = 1` between the two Weyl points and 0 outside. Each plane
with :math:`C = 1` contributes one chiral edge state to a surface; together
they draw a line of zero-energy surface states that ends at the
projections of the Weyl points -- the Fermi arc.

The model: :math:`H(\mathbf{k}) = \sin k_x\,\sigma_x + \sin k_y\,\sigma_y
+ (2 + \cos k_0 - \cos k_x - \cos k_y - \cos k_z)\,\sigma_z`, Weyl points at
:math:`(0, 0, \pm k_0)`.
"""
import numpy as np
import matplotlib.pyplot as plt

from tbkit.lattice import Lattice
from tbkit.kspace import KSpace, PAULI, ribbon
from tbkit.topology import find_weyl_points

k0 = np.pi / 2
cubic = [(1., 0., 0.), (0., 1., 0.), (0., 0., 1.)]
lat = Lattice(unit_cell=[{'tag': 'a', 'r0': (0., 0., 0.)}], prim_vec=cubic)
# the two components are the "spin" of a spinful site
hops = [{'i': 0, 'j': 0, 'R': (1, 0, 0), 't': -0.5j*PAULI['x'] - 0.5*PAULI['z']},
            {'i': 0, 'j': 0, 'R': (0, 1, 0), 't': -0.5j*PAULI['y'] - 0.5*PAULI['z']},
            {'i': 0, 'j': 0, 'R': (0, 0, 1), 't': -0.5*PAULI['z']}]
onsite = {'a': (2. + np.cos(k0)) * PAULI['z']}
weyl = KSpace(lat, spin=True)
weyl.set_hopping(hops)
weyl.set_onsite(onsite)

# %%
# The Weyl points and their chirality
# ---------------------------------------
# :func:`~tbkit.topology.find_weyl_points` scans the gap between the two
# bands over a mesh of the zone, refines its minima by Newton's method, and
# measures the chirality of each touching: the Berry flux of the lower band
# out of a small sphere around it, divided by :math:`2\pi`. The two
# chiralities cancel, as they must over a whole zone (Nielsen-Ninomiya).

points = find_weyl_points(weyl, bands=0)
order = np.argsort(points.k[:, 2])
k_weyl, chirality = points.k[order], points.chirality[order]
print('Weyl points (k_x, k_y, k_z):', np.round(k_weyl, 6).tolist(), 'chiralities:', chirality.tolist())
assert np.allclose(k_weyl, [[0., 0., -k0], [0., 0., k0]], atol=1e-8)
assert np.all(points.gap < 1e-8) and np.allclose(points.energy, 0., atol=1e-8)
assert chirality.tolist() == [-1, 1] and points.total_chirality == 0
# linear in every direction around them
q = 1e-4
for d in np.eye(3):
    gap = np.diff(np.linalg.eigvalsh(weyl.get_ham(k_weyl[1] + q * d)))[0]
    assert np.isclose(gap / (2 * q), 1., atol=1e-3)
print('Two Weyl points at k = (0, 0, +-pi/2), linear in every direction, of chirality -1 and +1.')

# %%
# The Chern number of the k_z planes
# ----------------------------------------
# By Gauss's law, the Chern number of the plane jumps by the chirality as
# the plane crosses a Weyl point upwards: :math:`-1` at :math:`-k_0`, back up
# by :math:`+1` at :math:`+k_0`.

kz_fracs = np.linspace(-0.475, 0.475, 20)
chern = np.array([weyl.chern_number(bands=[0], nk=20, k_fixed=f) for f in kz_fracs])
inside = np.abs(kz_fracs) < k0 / (2 * np.pi)
print('C(k_z):', np.round(chern).astype(int))
assert np.allclose(chern[inside], chirality[0], atol=1e-6) and np.allclose(chern[~inside], 0., atol=1e-6)

# %%
# The Fermi arc
# ----------------
# A slab, finite along y (:func:`~tbkit.kspace.ribbon` of the 3D model):
# at :math:`k_x = 0`, zero-energy states exist only for :math:`|k_z| < k_0`,
# and live on the two surfaces.

width = 30
slab = ribbon(lat, hops, width=width, direction=1, onsite=onsite, spin=True)
kzs = np.linspace(-np.pi, np.pi, 61)
en, vec = slab.get_bands(np.array([[0., kz] for kz in kzs]), eigenvec=True)
lowest = np.min(np.abs(en), axis=1)
arc = np.abs(kzs) < k0 - 0.2
far = np.abs(kzs) > k0 + 0.3
assert np.all(lowest[arc] < 1e-2) and np.all(lowest[far] > 0.2)
n = np.argmin(np.abs(en[30]))  # k_z = 0
weight = np.abs(vec[30][:, n]) ** 2
rows = weight.reshape(width, 2).sum(axis=1)
surface = rows[:3].sum() + rows[-3:].sum()
print('Zero-energy slab state at k_z = 0: {:.0%} of its weight on the outer 3 layers.'.format(surface))
assert surface > 0.9

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(kz_fracs * 2 * np.pi, chern, 'o-b')
for kz, chi in zip(k_weyl[:, 2], chirality):
    ax.axvline(kz, color='r', ls='--')
    ax.annotate(r'$\chi = {:+d}$'.format(chi), (kz, 0.1), color='r', ha='center')
ax.set_xlabel('$k_z$')
ax.set_ylabel('Chern number of the plane')
fig.set_layout_engine('tight')

# %%
# The Fermi surface of the slab
# ---------------------------------
# :meth:`~tbkit.kspace.KSpace.fermi_surface` of the slab at a small energy
# :math:`E = 0.1`: each surface carries an arc of states with
# :math:`k_x = \pm E` (velocity 1, opposite on the two surfaces) for
# :math:`|k_z| < k_0`, and the two arcs join through the small bulk
# pockets around the projections of the Weyl points -- one closed contour,
# half on each surface.

energy = 0.1
middle = slab.norb // 2  # the lowest band above zero energy
contour = slab.fermi_surface(energy, nk=(40, 80), bands=middle)[0].reshape(-1, 2)
on_arc = np.abs(contour[:, 1]) < k0 - 0.3
assert np.allclose(np.abs(contour[on_arc, 0]), energy, atol=1e-3)
assert np.abs(contour[:, 1]).max() < k0 + 0.15  # it ends at the Weyl points
print('Fermi arcs at E = {}: |k_x| = E for |k_z| < k0, closing at k_z = +-k0.'.format(energy))
fig2 = slab.plot_fermi_surface(energy, nk=(40, 80), bands=middle, fs=14, figsize=(5, 5))
fig2.axes[0].set_title('Slab Fermi surface: the Fermi arcs', fontsize=14)
