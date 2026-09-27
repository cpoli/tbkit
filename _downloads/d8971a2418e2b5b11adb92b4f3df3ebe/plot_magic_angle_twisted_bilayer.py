r"""
Magic-Angle Twisted Bilayer Graphene: Flat Moiré Bands
===========================================================

Twist two graphene sheets by a small angle :math:`\theta` and a moiré
pattern appears, with a period :math:`a/(2\sin\frac\theta2)` of a hundred
lattice constants. R. Bistritzer and A. H. MacDonald (2011) showed that the
interlayer tunnelling :math:`w` slows the Dirac electrons down,

.. math::

    \frac{v^*}{v} = \frac{1-3\alpha^2}{1+6\alpha^2}\, ,\qquad
    \alpha = \frac{w}{\hbar v k_\theta}\, ,\quad k_\theta = 2|\mathbf{K}|\sin\frac\theta2\, ,

until, at the *magic angle* :math:`\alpha = 1/\sqrt3` (about
:math:`1.1°`), the two bands at charge neutrality become almost flat. In
2018 Y. Cao, P. Jarillo-Herrero and co-workers found correlated
insulators and superconductivity in exactly those flat bands.

:func:`tbkit.moire.twisted_bilayer` builds the commensurate moiré cell of
the atomistic model (the Slater-Koster :math:`p_z` hoppings of Moon and
Koshino, :func:`~tbkit.moire.pz_hopping`, within and between the layers).

**Why not 1.05° directly?** The commensurate cell at :math:`\theta = 1.05°`
(:math:`m = 31`) holds 11908 orbitals: one dense Bloch matrix takes 2.3 GB,
too much for a gallery script. The physics depends on :math:`\alpha` only,
so we (i) check :math:`v^*/v` against Bistritzer-MacDonald at larger angles,
where :math:`\alpha` is small, and (ii) reach the magic value of
:math:`\alpha` at :math:`\theta = 6°` (364 orbitals) by scaling the
interlayer hoppings (*interlayer_scale*), the continuum-scaling argument of
the Bistritzer-MacDonald model.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq

from tbkit.moire import commensurate_angle, magic_angle_parameter, twisted_bilayer


def moire_k(tbl):
    '''Corner K of the moire Brillouin zone.'''
    b1, b2 = tbl.rec_vec_k
    c = b2 if b1 @ b2 > 0 else b1 + b2
    return np.linalg.solve(2 * np.array([b1, c]), np.array([b1 @ b1, c @ c]))


def central(tbl, ks):
    '''The four bands at charge neutrality (two valleys, two bands).'''
    n = tbl.norb // 2
    return np.linalg.eigvalsh(tbl._bloch_ham(ks @ tbl.k_basis.T))[:, n - 2:n + 2]


def dirac_velocity(tbl, dq=1e-2):
    k = moire_k(tbl)
    e0, e1 = central(tbl, np.array([k, k * (1 + dq)]))
    return np.mean(np.abs(e1 - e0.mean())) / (dq * np.linalg.norm(k))


# %%
# Commensurate angles and the magic angle
# -----------------------------------------
# :math:`m = 31` gives :math:`\theta = 1.05°`. The Bistritzer-MacDonald
# tunnelling of the Moon-Koshino hoppings is :math:`w \approx 110` meV, and
# :math:`v^* = 0` at :math:`\theta \approx 1.1°`.

theta_31 = np.degrees(commensurate_angle(31))
w, _, _ = magic_angle_parameter(np.radians(1.05))
magic = brentq(lambda th: magic_angle_parameter(np.radians(th))[2], 0.8, 2.)
assert abs(theta_31 - 1.05) < 0.01 and abs(w - 0.11) < 0.005 and abs(magic - 1.1) < 0.1
print('m = 31: theta = {:.3f} deg, {} orbitals'.format(theta_31, 4 * (3 * 31**2 + 3 * 31 + 1)))
print('w = {:.1f} meV, first magic angle {:.2f} deg'.format(1e3 * w, magic))

# %%
# The Dirac velocity follows Bistritzer and MacDonald
# ------------------------------------------------------
# The velocity at the moiré K point, relative to that of decoupled layers,
# against :math:`(1-3\alpha^2)/(1+6\alpha^2)` for the same model.

rows = []
for m in (3, 5, 7):
    tbl = twisted_bilayer(m)
    v0 = dirac_velocity(twisted_bilayer(m, interlayer_scale=1e-9))
    ratio_tb = dirac_velocity(tbl) / v0
    _, alpha, ratio_bm = magic_angle_parameter(tbl.theta, t=-v0 / (1.5 * 0.142))
    assert abs(ratio_tb - ratio_bm) < 0.03
    rows.append((np.degrees(tbl.theta), alpha, ratio_tb, ratio_bm))
    print('theta = {:5.2f} deg  alpha = {:.3f}  v*/v: TB {:.3f}  BM {:.3f}'.format(*rows[-1]))

# %%
# Flat bands at the magic value of alpha
# -----------------------------------------
# At :math:`\theta = 6.01°` (:math:`m = 5`), the interlayer hoppings scaled
# by 4.6 bring :math:`\alpha` to 0.50, near the magic value of the lattice
# model: the bandwidth of the four central bands drops by more than an
# order of magnitude.

fig, axes = plt.subplots(1, 2, figsize=(9, 4.5), sharey=False)
widths = []
for ax, s, lim in zip(axes, (1., 4.6), (900, 400)):
    tbl = twisted_bilayer(5, interlayer_scale=s)
    k = moire_k(tbl)
    m_pt = tbl.rec_vec_k[0] / 2
    nodes = [k, 0 * k, m_pt, k]
    ks = np.concatenate([np.linspace(nodes[i], nodes[i + 1], 15, endpoint=False) for i in range(3)]
                                   + [nodes[-1:]])
    en = np.linalg.eigvalsh(tbl._bloch_ham(ks @ tbl.k_basis.T))
    n = tbl.norb // 2
    mid = en[:, n - 2:n + 2]
    widths.append(mid.max() - mid.min())
    e0 = mid.mean()
    x = np.arange(len(ks))
    ax.plot(x, 1e3 * (en[:, n - 8:n + 8] - e0), c='0.6', lw=1)
    ax.plot(x, 1e3 * (mid - e0), c='C3', lw=2)
    _, alpha, _ = magic_angle_parameter(tbl.theta, interlayer_scale=s)
    assert s == 1. or abs(alpha - 0.50) < 0.01
    ax.set_title(r'$\alpha = {:.2f}$: bandwidth {:.0f} meV'.format(alpha, 1e3 * widths[-1]))
    ax.set_xticks([0, 15, 30, 45])
    ax.set_xticklabels([r'$K$', r'$\Gamma$', r'$M$', r'$K$'])
    ax.set_ylim(-lim, lim)
axes[0].set_ylabel('$E - E_0$ (meV)')
assert widths[0] / widths[1] > 15
print('bandwidth: {:.0f} meV -> {:.0f} meV'.format(1e3 * widths[0], 1e3 * widths[1]))
fig.suptitle(r'Twisted bilayer, $\theta = 6.01°$, interlayer hopping $\times 1$ and $\times 4.6$')
fig.tight_layout()
