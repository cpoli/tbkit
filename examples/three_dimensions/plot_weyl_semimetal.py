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
monopole charge as the plane crosses a Weyl point: here
:math:`C(k_z) = 1` between the two Weyl points and 0 outside. Each plane
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
# The Weyl points
# ------------------

for kz in (k0, -k0):
    assert np.allclose(np.linalg.eigvalsh(weyl.get_ham([0., 0., kz])), 0.)
# linear in every direction around them
q = 1e-4
for d in np.eye(3):
    gap = np.diff(np.linalg.eigvalsh(weyl.get_ham(np.array([0., 0., k0]) + q * d)))[0]
    assert np.isclose(gap / (2 * q), 1., atol=1e-3)
print('Two Weyl points at k = (0, 0, +-pi/2), linear in every direction.')

# %%
# The Chern number of the k_z planes
# ----------------------------------------

kz_fracs = np.linspace(-0.475, 0.475, 20)
chern = np.array([weyl.chern_number(bands=[0], nk=20, k_fixed=f) for f in kz_fracs])
inside = np.abs(kz_fracs) < k0 / (2 * np.pi)
print('C(k_z):', np.round(chern).astype(int))
assert np.allclose(np.abs(chern[inside]), 1., atol=1e-6) and np.allclose(chern[~inside], 0., atol=1e-6)
assert len(set(np.round(chern[inside]))) == 1

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

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].plot(kz_fracs * 2 * np.pi, chern, 'o-b')
axes[0].set_xlabel('$k_z$')
axes[0].set_ylabel('Chern number of the plane')
kx = np.linspace(-np.pi, np.pi, 81)
grid = np.array([[a, b] for a in kx for b in kx])
e_grid = np.min(np.abs(slab.get_bands(grid)), axis=1).reshape(len(kx), len(kx))
axes[1].imshow(np.log10(e_grid.T + 1e-4), origin='lower', extent=[-np.pi, np.pi, -np.pi, np.pi], cmap='magma')
axes[1].set_xlabel('$k_x$')
axes[1].set_ylabel('$k_z$')
axes[1].set_title('slab: $\\log_{10}\\min|E|$ (the Fermi arc)')
fig.set_layout_engine('tight')
