r"""
Lieb and Wu: the Exact Solution of the Hubbard Chain
=========================================================

E. H. Lieb and F. Y. Wu (1968) solved the one-dimensional Hubbard model
exactly by the Bethe ansatz. At half filling the ground-state energy per
site is

.. math::

    \frac{E_0}{N} = -4|t|\int_0^\infty\frac{J_0(\omega)J_1(\omega)}
    {\omega\,(1 + e^{\omega U/2|t|})}\,d\omega\, ,

which falls from :math:`-4|t|/\pi` at :math:`U = 0` to :math:`-4\ln 2\,t^2/U`
at large *U* (the Heisenberg chain). The charge gap,

.. math::

    \Delta = E_0(N+1) + E_0(N-1) - 2E_0(N) = U - 4|t| + 8|t|\int_0^\infty
    \frac{J_1(\omega)}{\omega\,(1 + e^{\omega U/2|t|})}\,d\omega\, ,

is positive for *every* :math:`U > 0`: the half-filled chain is a Mott
insulator however weak the repulsion -- there is no Mott transition at a
finite *U*.

Both are checked here by :func:`~tbkit.manybody.exact_diagonalization` of
rings of 6 to 12 sites (up to 853776 states), extrapolated in the length.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate, special

import tbkit.lattices as lattices
from tbkit.kspace import KSpace
from tbkit.manybody import exact_diagonalization


def lieb_wu_energy(U):
    '''Ground-state energy per site of the half-filled chain, |t| = 1.'''
    if U == 0:
        return -4 / np.pi
    # 1/(1 + e^x) = expit(-x) is below 1e-26 beyond x = 60
    f = lambda w: special.j0(w) * special.j1(w) / w * special.expit(-w * U / 2)
    return -4 * integrate.quad(f, 0, 120 / U, limit=1000)[0]


def lieb_wu_gap(U):
    '''Charge gap of the half-filled chain, |t| = 1 (U > 0).'''
    f = lambda w: special.j1(w) / w * special.expit(-w * U / 2)
    return U - 4 + 8 * integrate.quad(f, 0, 120 / U, limit=1000)[0]


# %%
# Rings with closed shells
# ---------------------------
# A ring of *L* sites (:meth:`~tbkit.kspace.KSpace.finite_ham` of a chain,
# ``periodic=True``) has a unique free ground state at half filling when
# its :math:`L/2` electrons of each spin fill closed shells: periodic
# boundaries for :math:`L/2` odd, antiperiodic ones (a flux
# :math:`\theta = \pi`, ``twist``) for :math:`L/2` even. At :math:`U = 0`
# the exact energy is that of the filled levels.

chain = KSpace(lattices.chain())
chain.set_hopping([{'n': 1, 't': -1.}])


def ring(L):
    L = int(L)
    return chain.finite_ham(L, periodic=True, twist=None if (L // 2) % 2 else [np.pi]).real


lengths = np.array([6, 8, 10, 12])
for L in lengths:
    levels = np.linalg.eigvalsh(ring(L))
    assert levels[L // 2] - levels[L // 2 - 1] > 0.1   # closed shell
    assert np.isclose(exact_diagonalization(ring(L), L // 2, L // 2).energies[0], 2 * levels[:L // 2].sum())

# %%
# The energy per site
# ----------------------
# The finite rings approach the Lieb-Wu energy as :math:`1/L^2`; the two
# longest give the extrapolation.

us = np.array([0., 2., 4., 8.])
per_site = np.array([[exact_diagonalization(ring(L), L // 2, L // 2, U=U).energies[0] / L for L in lengths]
                     for U in us])
extrapolated = (lengths[-1] ** 2 * per_site[:, -1] - lengths[-2] ** 2 * per_site[:, -2]) \
    / (lengths[-1] ** 2 - lengths[-2] ** 2)
exact = np.array([lieb_wu_energy(U) for U in us])
for U, e, x in zip(us, extrapolated, exact):
    print('U = {:.0f}|t|: E0/N = {:.4f}|t| (rings, L -> infinity), {:.4f}|t| (Lieb-Wu)'.format(U, e, x))
assert np.allclose(extrapolated, exact, rtol=5e-3)

# %%
# The Mott gap
# ---------------
# The charge gap of a ring falls as :math:`1/L` towards the Lieb-Wu gap;
# a fit :math:`\Delta_L = \Delta + a/L + b/L^2` over the four rings gives
# it within a few tenths of a percent at :math:`U = 8|t|` and :math:`16|t|`.
# At weak coupling the gap is
# exponentially small, :math:`\Delta\propto e^{-2\pi|t|/U}`, far below the
# level spacing of any ring that can be diagonalized: there, the
# finite-size gap says nothing about it.


def charge_gap(L, U):
    L = int(L)
    e = [exact_diagonalization(ring(L), up, dn, U=U).energies[0]
         for up, dn in ((L // 2, L // 2), (L // 2 + 1, L // 2), (L // 2, L // 2 - 1))]
    return e[1] + e[2] - 2 * e[0]


gap_us = np.array([8., 16.])
gaps = np.array([[charge_gap(L, U) for L in lengths] for U in gap_us])
fit = np.vstack([np.ones(len(lengths)), 1 / lengths, 1 / lengths ** 2]).T
gap_extrapolated = np.linalg.lstsq(fit, gaps.T, rcond=None)[0][0]
gap_exact = np.array([lieb_wu_gap(U) for U in gap_us])
for U, g, x in zip(gap_us, gap_extrapolated, gap_exact):
    print('U = {:.0f}|t|: gap {:.3f}|t| (rings, L -> infinity), {:.3f}|t| (Lieb-Wu)'.format(U, g, x))
assert np.allclose(gap_extrapolated, gap_exact, rtol=5e-3)
assert all(lieb_wu_gap(U) > 0 for U in (0.5, 1., 2.))

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
u_fine = np.linspace(0., 12., 61)
axes[0].plot(u_fine, [lieb_wu_energy(U) for U in u_fine], 'k', label='Lieb-Wu')
for L, e in zip(lengths, per_site.T):
    axes[0].plot(us, e, 'o', label='ring, L = {}'.format(L))
axes[0].plot(us, extrapolated, 'rx', ms=10, mew=2, label=r'$L\to\infty$')
axes[0].set_xlabel(r'$U/|t|$')
axes[0].set_ylabel(r'$E_0/N$ ($|t|$)')
axes[0].set_title('Ground-state energy per site')
axes[0].legend()
u_gap = np.linspace(0.5, 18., 60)
axes[1].plot(u_gap, [lieb_wu_gap(U) for U in u_gap], 'k', label='Lieb-Wu: positive for all U > 0')
axes[1].plot(u_gap, u_gap - 4, 'k:', label=r'$U - 4|t|$')
axes[1].plot(gap_us, gap_extrapolated, 'rx', ms=10, mew=2, label=r'rings, $L\to\infty$')
axes[1].set_xlabel(r'$U/|t|$')
axes[1].set_ylabel(r'$\Delta$ ($|t|$)')
axes[1].set_ylim(bottom=-0.5)
axes[1].set_title('Mott gap of the half-filled chain')
axes[1].legend()
fig.set_layout_engine('tight')
