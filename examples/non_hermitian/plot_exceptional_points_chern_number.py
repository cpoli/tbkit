r"""
Exceptional Points in Momentum Space: the Chern Number until the Line Gap Closes
=====================================================================================

A non-Hermitian band has two eigenvectors, a right one
:math:`H|R\rangle = E|R\rangle` and a left one
:math:`H^\dagger|L\rangle = E^*|L\rangle`, so there are four Berry
connections :math:`i\langle u^\alpha|\nabla u^\beta\rangle`,
:math:`\alpha,\beta\in\{L,R\}`. Shen, Zhen and Fu (Phys. Rev. Lett. 120,
146402 (2018)) showed that their Chern numbers :math:`C^{LR}, C^{RL},
C^{RR}, C^{LL}` are one and the same integer, as long as the band stays
separated from the others -- here by a **line gap**, a line
:math:`\mathrm{Re}\,E = 0` that no band crosses. The gap can close only at
**exceptional points**; until then the Chern number cannot change.

Here the Haldane model (:math:`t_2 = 0.2`, :math:`\phi = \pi/2`, :math:`M = 0`,
Chern number 1) is given balanced gain and loss :math:`\pm i\gamma` on its two
sublattices:

.. math::

    H(\mathbf{k}) = \mathbf{d}\cdot\boldsymbol\sigma + i\gamma\sigma_z\, ,\qquad
    E^2 = |f|^2 + d_z^2 - \gamma^2 + 2i\gamma d_z\, .

An eigenvalue reaches :math:`\mathrm{Re}\,E = 0` only where :math:`E^2` is real
and negative: :math:`d_z = 0` and :math:`|f| < \gamma`. On the lines
:math:`d_z = 0` the smallest :math:`|f|` is 1 (at the M points), so the real
line gap closes at :math:`\gamma_c = t = 1`, at three pairs of exceptional
points born at M.

:meth:`~tbkit.kspace.KSpace.biorthogonal_chern_number` computes the four
Chern numbers; :meth:`~tbkit.kspace.KSpace.chern_number` (right
eigenvectors only) agrees with them while the line gap is open, and past it
both refuse to return a number, since the band ordering by real part is no
longer continuous.
"""
import numpy as np
import matplotlib.pyplot as plt

import tbkit.lattices as lattices
from tbkit.kspace import KSpace
import tbkit.exceptional as ex

KINDS = ('LR', 'RL', 'RR', 'LL')


def haldane(gamma, t2=0.2):
    hal = KSpace(lattices.honeycomb())
    hal.set_hopping([{'i': 0, 'j': 1, 'R': (0, 0), 't': 1.}, {'i': 0, 'j': 1, 'R': (-1, 0), 't': 1.},
                     {'i': 0, 'j': 1, 'R': (0, -1), 't': 1.}])
    nnn = [(0, 1), (-1, 0), (1, -1)]
    hal.set_hopping([{'i': 0, 'j': 0, 'R': R, 't': 1j * t2} for R in nnn]
                    + [{'i': 1, 'j': 1, 'R': R, 't': -1j * t2} for R in nnn])
    hal.set_onsite({'a': 1j * gamma, 'b': -1j * gamma})
    return hal


b1, b2 = (np.array(b) for b in haldane(0.).rec_vec_k)
M_points = [b1 / 2, b2 / 2, (b1 + b2) / 2]

# %%
# Where the line gap closes
# ------------------------------
# The real line gap is :math:`2\min_{\mathbf{k}}|\mathrm{Re}\,E|`. At an M
# point :math:`d_z = 0` and :math:`|f| = 1`, so :math:`E = \pm\sqrt{1 - \gamma^2}`
# there: the gap closes exactly at :math:`\gamma = 1`.

hal0 = haldane(0.)
for M in M_points:
    h = hal0.get_ham(M)
    assert abs(h[0, 0] - h[1, 1]) < 1e-12 and np.isclose(abs(h[0, 1]), 1.)
_, ks = hal0.mesh_grid(60)


def line_gap(gamma):
    m = haldane(gamma)
    en = [np.linalg.eigvals(m.get_ham(k)) for k in list(ks) + M_points]
    return 2 * np.min(np.abs(np.real(en)))


gammas = np.linspace(0., 1.4, 29)
gaps = np.array([line_gap(g) for g in gammas])
for g in (0.5, 0.9, 0.99):
    assert np.isclose(line_gap(g), 2 * np.sqrt(1 - g ** 2), rtol=1e-8)
assert line_gap(1.05) < 1e-12

# %%
# Four Chern numbers, one integer
# ---------------------------------------

print('gamma   C_LR    C_RL    C_RR    C_LL    chern_number')
for gamma in (0., 0.3, 0.6, 0.9):
    m = haldane(gamma)
    chern = [m.biorthogonal_chern_number(0, 24, kind=kind) for kind in KINDS]
    c_right = m.chern_number(0, 24)
    print('{:.1f}  '.format(gamma) + '  '.join('{:+.4f}'.format(c) for c in chern + [c_right]))
    assert np.allclose(chern, 1., atol=1e-8)
    assert np.isclose(c_right, 1., atol=1e-8)
    assert len(ex.find_exceptional_points(m, nk=30).k) == 0

# %%
# Past gamma_c: exceptional points, and no Chern number
# --------------------------------------------------------------
# At :math:`\gamma = 1.2` the finder reports three pairs of EPs of opposite
# charge around the three M points, and neither Chern number is defined any
# more: both methods raise a ValueError instead of returning a number.

beyond = haldane(1.2)
eps = ex.find_exceptional_points(beyond, nk=40)
print('gamma = 1.2: {} EPs, charges {}, orders {}'
      .format(len(eps.k), eps.charge.tolist(), eps.order.tolist()))
assert len(eps.k) == 6 and eps.total_charge == 0 and np.all(eps.order == 2)
nearest = []
for k in eps.k:
    h = hal0.get_ham(k)  # the Hermitian part: d_z and f
    dist = [min(np.linalg.norm(k - M - m * b1 - n * b2) for m in (-1, 0, 1) for n in (-1, 0, 1))
            for M in M_points]
    nearest.append(int(np.argmin(dist)))
    print('  EP at ({:+.4f}, {:+.4f}): d_z = {:.1e}, |f| = {:.10f}, {:.4f} from M{}'
          .format(*k, abs(h[0, 0] - h[1, 1]) / 2, abs(h[0, 1]), min(dist), nearest[-1] + 1))
    assert abs(h[0, 0] - h[1, 1]) < 1e-8 and abs(abs(h[0, 1]) - 1.2) < 1e-8
assert sorted(nearest) == [0, 0, 1, 1, 2, 2]  # one pair around each M point
for method in (beyond.chern_number, beyond.biorthogonal_chern_number):
    try:
        method(0, 24)
    except ValueError:
        pass
    else:
        raise AssertionError('the Chern number should not be defined past gamma_c')
print('gamma = 1.2: chern_number and biorthogonal_chern_number raise ValueError (no line gap).')

# %%
# Line gap and exceptional points
# ------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
axes[0].plot(gammas, gaps, 'k.-', label='line gap (numerical)')
axes[0].plot(gammas[gammas <= 1], 2 * np.sqrt(1 - gammas[gammas <= 1] ** 2), 'r-', lw=0.8,
             label=r'$2\sqrt{1-\gamma^2}$')
axes[0].axvline(1., color='gray', ls='--')
axes[0].text(0.3, 0.3, '$C^{LR} = C^{RL} = C^{RR} = C^{LL} = 1$')
axes[0].text(1.03, 1.2, 'EPs,\nno $C$')
axes[0].set_xlabel(r'$\gamma$')
axes[0].set_ylabel(r'$2\min|\mathrm{Re}\,E|$')
axes[0].legend()

en_re = np.array([np.max(np.linalg.eigvals(beyond.get_ham(k)).real) for k in ks])
axes[1].tricontourf(ks[:, 0], ks[:, 1], en_re, 30, cmap='viridis')
for q, color in ((1, 'red'), (-1, 'cyan')):
    pts = eps.k[eps.charge == q]
    axes[1].plot(pts[:, 0], pts[:, 1], 'x', color=color, ms=10, mew=3,
                 label='EP, charge {:+d}'.format(q))
axes[1].set_aspect('equal')
axes[1].set_xlabel('$k_x$')
axes[1].set_ylabel('$k_y$')
axes[1].set_title(r'max Re $E$ at $\gamma = 1.2$')
axes[1].legend(fontsize=8)
fig.tight_layout()
