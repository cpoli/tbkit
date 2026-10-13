r"""
Rashba Spin-Orbit Coupling: Spin-Split Bands from Broken Inversion Symmetry
=================================================================================

An electron moving through an electric field sees, in its rest frame, a
magnetic field :math:`\propto \mathbf{E}\times\mathbf{v}`. At a surface or
an asymmetric quantum well the field points out of the plane, and
Bychkov and Rashba (1984) wrote the coupling it produces for a 2D
electron gas as

.. math::

    H_R = \lambda\,(\boldsymbol\sigma\times\mathbf{k})\cdot\hat{\mathbf{z}}\, .

It needs no magnet and keeps time-reversal symmetry, but splits the spin
degeneracy of every band away from :math:`\mathbf{k} = 0` and locks the
spin perpendicular to the momentum. On a square lattice with
nearest-neighbour hopping :math:`t`,
:meth:`~tbkit.orbital.OrbitalSystem.set_rashba` gives the bands

.. math::

    E_\pm(\mathbf{k}) = 2t(\cos k_x + \cos k_y)
    \pm \sqrt{b_z^2 + 4\lambda^2(\sin^2 k_x + \sin^2 k_y)}\, ,

with :math:`b_z` a Zeeman field (:meth:`~tbkit.orbital.OrbitalSystem.set_zeeman`)
that breaks time reversal and opens a gap at :math:`\mathbf{k} = 0`.
"""
import numpy as np
import matplotlib.pyplot as plt

import tbkit.lattices as lattices
from tbkit.orbital import OrbitalSystem


L, t, lam = 12, -1., 0.3
SX = np.array([[0, 1], [1, 0]])
SY = np.array([[0, -1j], [1j, 0]])


def rashba_torus(bz=0.):
    lat = lattices.square()
    lat.get_lattice(n1=L, n2=L)
    sys = OrbitalSystem(lat, spin=True, periodic=True)
    sys.set_hopping([{'n': 1, 't': t}])
    sys.set_onsite({'a': 0.})
    sys.set_rashba(lam)
    if bz:
        sys.set_zeeman((0., 0., bz))
    sys.get_ham()
    return sys


def bands(kx, ky, bz=0.):
    eps = 2 * t * (np.cos(kx) + np.cos(ky))
    split = np.sqrt(bz**2 + 4 * lam**2 * (np.sin(kx)**2 + np.sin(ky)**2))
    return eps - split, eps + split


def bloch_block(sys, k):
    r'''The 2x2 spin block of H at momentum k, projected from the real-space torus.'''
    rows = np.array([[sys.row(i, 's', s) for s in (0, 1)] for i in range(sys.lat.sites)])
    wave = np.exp(1j * (k[0] * sys.lat.coor['x'] + k[1] * sys.lat.coor['y'])) / np.sqrt(sys.lat.sites)
    u = np.zeros((sys.ham.shape[0], 2), complex)
    u[rows[:, 0], 0] = wave
    u[rows[:, 1], 1] = wave
    return u.conj().T @ (sys.ham @ u)


# %%
# The spectrum of the real-space model is the Rashba dispersion
# -----------------------------------------------------------------
# On a 12 x 12 torus the momenta are :math:`2\pi n/12`; the
# :math:`2 \times 144` eigenvalues of the real-space Hamiltonian are
# exactly :math:`E_\pm` at those momenta.

k = 2 * np.pi * np.arange(L) / L
KX, KY = np.meshgrid(k, k)
for bz in (0., 0.4):
    en = np.linalg.eigvalsh(rashba_torus(bz).ham.toarray())
    assert np.allclose(en, np.sort(np.concatenate([b.ravel() for b in bands(KX, KY, bz)])), atol=1e-10)
print('real-space spectrum = E_+-(k) on the torus, with and without Zeeman: OK')

# %%
# Kramers pairs, and the Zeeman field that splits them
# --------------------------------------------------------
# Rashba coupling is odd in :math:`\mathbf{k}` and in spin, so time
# reversal survives: :math:`(\mathbf{k}, \uparrow)` and
# :math:`(-\mathbf{k}, \downarrow)` stay degenerate and every level of the
# torus is (at least) doubly degenerate. A Zeeman field breaks time
# reversal and lifts it, opening a gap :math:`2b_z` at
# :math:`\mathbf{k} = 0`.

en = np.linalg.eigvalsh(rashba_torus().ham.toarray())
assert np.allclose(en[0::2], en[1::2], atol=1e-10)
en_z = np.linalg.eigvalsh(rashba_torus(0.4).ham.toarray())
assert not np.allclose(en_z[0::2], en_z[1::2], atol=1e-6)
gamma = np.linalg.eigvalsh(bloch_block(rashba_torus(0.4), (0., 0.)))
assert np.isclose(gamma[1] - gamma[0], 2 * 0.4)
print('Kramers degeneracy of every level without Zeeman, lifted by b_z (gap 2 b_z at k = 0): OK')

# %%
# Spin-momentum locking
# -------------------------
# Projecting the torus Hamiltonian on plane waves gives its
# :math:`2 \times 2` Bloch block at each momentum. The spin of both
# eigenstates lies in the plane, perpendicular to
# :math:`(\sin k_x, \sin k_y)` (to :math:`\mathbf{k}` near
# :math:`\Gamma`), and opposite for the two bands.

sys = rashba_torus()
spins = {}
for kx, ky in zip(KX.ravel(), KY.ravel()):
    if np.hypot(np.sin(kx), np.sin(ky)) < 1e-9:
        continue  # time-reversal-invariant momenta: no splitting, no locking
    e, v = np.linalg.eigh(bloch_block(sys, (kx, ky)))
    assert np.allclose(e, bands(kx, ky))
    s = np.array([[np.vdot(v[:, n], m @ v[:, n]).real for m in (SX, SY)] for n in range(2)])
    assert np.allclose(np.linalg.norm(s, axis=1), 1.)  # fully in-plane
    assert np.allclose(s @ [np.sin(kx), np.sin(ky)], 0., atol=1e-10)  # perpendicular to k
    assert np.allclose(s[0], -s[1])
    spins[(kx, ky)] = s[0]
print('spin in-plane, perpendicular to sin k, opposite on the two bands at {} momenta: OK'.format(len(spins)))

# %%
# The two spin-split bands
# ----------------------------

kc = np.linspace(-np.pi, np.pi, 401)
fig, axes = plt.subplots(1, 2, figsize=(10, 4.3))
for bz, ls in ((0., '-'), (0.4, '--')):
    lo, hi = bands(kc, 0., bz)
    axes[0].plot(kc, lo, 'C0' + ls, kc, hi, 'C3' + ls)
fold = k - 2 * np.pi * (k > np.pi)  # torus momenta in [-pi, pi)
axes[0].plot(fold, bands(k, 0.)[0], 'C0o', fold, bands(k, 0.)[1], 'C3o', ms=4)
axes[0].set_xlim(-np.pi / 2, np.pi / 2)
axes[0].set_ylim(-4.6, -1.2)
axes[0].set_xlabel('$k_x$ ($k_y = 0$)')
axes[0].set_ylabel('$E/|t|$')
axes[0].set_title('Rashba (solid), with Zeeman $b_z = 0.4$ (dashed)')
q = np.array(list(spins))
s = np.array(list(spins.values()))
q = q - 2 * np.pi * (q > np.pi)
near = np.hypot(*q.T) < 1.6
axes[1].quiver(q[near, 0], q[near, 1], s[near, 0], s[near, 1], color='C0', pivot='mid')
axes[1].set_aspect('equal')
axes[1].set_xlabel('$k_x$')
axes[1].set_ylabel('$k_y$')
axes[1].set_title('spin of the lower band (torus momenta)')
fig.tight_layout()
