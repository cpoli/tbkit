r"""
Fractional Corner Charges from Rotation Eigenvalues
====================================================

W. A. Benalcazar, T. Li and T. L. Hughes (2019) showed that the corner
charge of a :math:`C_4`-symmetric insulator is a symmetry indicator: it
follows from the rotation eigenvalues of the occupied bands,

.. math::

    Q = \tfrac{1}{4}\left([X_1^{(2)}] + 2[M_1^{(4)}] + 3[M_2^{(4)}]\right) \bmod 1\, ,

where :math:`[\Pi_p^{(m)}]` is the number of bands of :math:`C_m`
eigenvalue :math:`e^{2\pi i(p-1)/m}` at :math:`\Pi` minus that at
:math:`\Gamma`. A Wannier centre pinned to the corner of the unit cell,
which a flake cut from whole cells splits between four cells, leaves a
quarter of an electron at each corner of the flake.

The model is the 2D Su-Schrieffer-Heeger lattice: four sites around the
centre of a square cell, intracell hopping :math:`\gamma`, intercell
:math:`\lambda`. For :math:`\gamma < \lambda` the lowest band is
obstructed: its Wannier centre sits at the cell corner.
:func:`~tbkit.symmetry.corner_charge` (the bulk eigenvalues, with
:math:`C_4` about the cell centre) is compared with the charge of a
quadrant of an :math:`8\times8` flake
(:func:`~tbkit.higher_order.corner_charges`), with the bulk band and the
edge states below the gap filled, on the ionic background of the bulk band.
"""
import numpy as np
import matplotlib.pyplot as plt

from tbkit.higher_order import corner_charges
from tbkit.kspace import KSpace
from tbkit.lattice import Lattice
from tbkit.symmetry import Operation, corner_charge, rotation

POS = [(0.25, 0.25), (0.75, 0.25), (0.75, 0.75), (0.25, 0.75)]


def ssh_2d(gamma, lam, copies=1, d=0.1):
    '''*copies* copies of the 2D SSH model, coupled on site by *d*.'''
    cell = [{'tag': t + str(c), 'r0': p} for t, p in zip('abcd', POS) for c in range(copies)]
    ks = KSpace(Lattice(unit_cell=cell, prim_vec=[(1., 0.), (0., 1.)]))
    hops = []
    for c in range(copies):
        def s(i):
            return copies * i + c
        hops += [{'i': s(i), 'j': s((i + 1) % 4), 'R': (0, 0), 't': -gamma} for i in range(4)]
        hops += [{'i': s(1), 'j': s(0), 'R': (1, 0), 't': -lam}, {'i': s(2), 'j': s(3), 'R': (1, 0), 't': -lam},
                     {'i': s(3), 'j': s(0), 'R': (0, 1), 't': -lam}, {'i': s(2), 'j': s(1), 'R': (0, 1), 't': -lam}]
        if c:
            hops += [{'i': copies * i, 'j': s(i), 'R': (0, 0), 't': d} for i in range(4)]
    ks.set_hopping(hops)
    return ks


C4 = Operation.about(rotation(np.pi / 2), (0.5, 0.5))  # about the centre of the cell
N = 8
E_GAP = -0.6  # between the edge states and the zero modes


def flake(ks, n_bands):
    en = np.linalg.eigvalsh(ks.finite_ham(N))
    assert np.min(np.abs(en - E_GAP)) > 0.05
    return corner_charges(ks, N, n_electrons=int(np.sum(en < E_GAP)), background=n_bands / ks.norb)


# %%
# Bulk indicator and flake, in three phases
# -----------------------------------------

cases = [('obstructed', ssh_2d(0.2, 1.), [0], 0.25), ('trivial', ssh_2d(1., 0.2), [0], 0.),
            ('two obstructed copies', ssh_2d(0.2, 1., copies=2), [0, 1], 0.5)]
results = []
for name, ks, bands, q_expected in cases:
    q = corner_charge(ks, C4, bands)
    res = flake(ks, len(bands))
    print('{:22s} Q (eigenvalues) = {:.2f}   quadrants of the flake = {}'.format(
        name, q, np.round(res.charges.ravel(), 6)))
    assert q == q_expected
    np.testing.assert_allclose((res.charges - q + 0.5) % 1. - 0.5, 0., atol=1e-8)
    results.append(res)

# %%
# Where the charge sits
# ---------------------
# The excess density of the obstructed flake: the edges carry the
# filled edge states, and each corner keeps a quarter electron modulo 1.

res = results[0]
fig, ax = plt.subplots(figsize=(4.6, 4))
excess = res.density - res.background
sc = ax.scatter(res.positions[:, 0], res.positions[:, 1], c=excess, cmap='RdBu_r', s=40,
                        vmin=-np.abs(excess).max(), vmax=np.abs(excess).max())
ax.axvline(N / 2, color='k', lw=0.5)
ax.axhline(N / 2, color='k', lw=0.5)
fig.colorbar(sc, ax=ax, label=r'$\rho_i - \bar\rho$')
ax.set_aspect('equal')
ax.set_title('Q = {:.2f} per quadrant (mod 1)'.format(float(res.charges[0, 0] % 1)))
fig.set_layout_engine('tight')
