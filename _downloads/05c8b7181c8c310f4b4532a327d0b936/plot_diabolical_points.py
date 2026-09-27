r"""
Diabolical Points: Conical Intersections in a Two-Parameter Family
======================================================================

Berry and Wilkinson (1984) asked when two energy levels of a real
symmetric Hamiltonian -- the quantum billiard in a triangle, in their case
-- can be degenerate. The answer goes back to von Neumann and Wigner: a
real symmetric :math:`2\times2` block has *two* conditions for degeneracy
(equal diagonal elements, vanishing off-diagonal one), so without a
symmetry one parameter is not enough. Levels **avoid crossing** along any
one-parameter path, and meet only at isolated points of a two-parameter
plane. There the two sheets of the spectrum touch like the tips of two
cones -- a double cone, the "diabolo" that gave **diabolical points** their
name.

A diabolical point is also a source of geometry: the real eigenvector of
either level changes **sign** after one loop around it, a Berry phase of
:math:`\pi` (the same mechanism as around graphene's Dirac points, but
here with no lattice symmetry pinning the degeneracy in place).

This example builds the tight-binding version of a *scalene triangle*: a
triangular flake whose hoppings along two bond directions, :math:`t_1`
and :math:`t_2`, play the role of the triangle's two angles, with a small
potential on one site that breaks every mirror symmetry. It finds a
diabolical point, checks that it is a cone, that one-parameter paths
miss it, that the eigenvector flips sign around it
(:func:`tbkit.exceptional.encircle`), and that it *moves* -- rather than
disappears -- when the potential changes.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

import tbkit.lattices as lattices
from tbkit.system import System
from tbkit.plot import Plot
import tbkit.exceptional as ex

# %%
# A triangular flake with three bond directions
# -----------------------------------------------------
# Ten sites of the triangular lattice cut into an equilateral triangle.
# The Hamiltonian is split by bond direction (0, 60 and 120 degrees), so
# that the hoppings along two of them, :math:`t_1, t_2`, can be tuned; the
# third stays at :math:`t = -1`. A potential :math:`V` on one site next to a
# corner removes the remaining mirror symmetries.

lat = lattices.triangular()
lat.get_lattice(n1=4, n2=4)
frac_i = lat.coor['x'] - lat.coor['y'] / np.sqrt(3)
frac_j = 2 * lat.coor['y'] / np.sqrt(3)
lat.remove_sites([int(i) for i in np.nonzero(np.rint(frac_i + frac_j) > 3)[0]])
assert lat.sites == 10

blocks = []
for ang in (0., 60., 120.):
    sys = System(lat)
    sys.set_hopping([{'n': 1, 'ang': ang, 't': -1.}])
    sys.get_ham()
    blocks.append(sys.ham.toarray().real)
defect = int(np.argmin(np.hypot(lat.coor['x'] - 1., lat.coor['y'])))
assert np.allclose(sum(blocks), sum(blocks).T)


def flake(p, v=0.3):
    '''H(t1, t2): real symmetric, no symmetry left for v != 0.'''
    ham = blocks[0] + p[0] * blocks[1] + p[1] * blocks[2]
    ham[defect, defect] += v
    return ham


sys_plot = System(lat)
sys_plot.set_hopping([{'n': 1, 't': -1.}])
fig_lat = Plot(sys_plot).lattice(plt_hop=True, ms=14, figsize=(4, 3.6))

# %%
# Finding the diabolical point of levels 5 and 6
# -------------------------------------------------------
# Levels are counted from 0, in increasing energy. The gap
# :math:`E_6 - E_5` is minimized over the plane (its square is smooth: a
# quadratic form at the cone). It reaches zero -- to machine
# precision -- at an isolated point well away from the three lines
# :math:`t_1 = 1`, :math:`t_2 = 1`, :math:`t_1 = t_2` on which the
# unperturbed triangle would have a mirror symmetry.

N = 5


def gap(p, v=0.3):
    en = np.linalg.eigvalsh(flake(p, v))
    return en[N + 1] - en[N]


def find_dp(v, guess):
    res = minimize(lambda p: gap(p, v) ** 2, guess, method='Nelder-Mead',
                   options={'xatol': 1e-13, 'fatol': 1e-32, 'maxiter': 6000})
    return res.x


dp = find_dp(0.3, (1.15, 0.45))
print('diabolical point at (t1, t2) = ({:.6f}, {:.6f}), gap = {:.1e}'.format(*dp, gap(dp)))
assert gap(dp) < 1e-10
assert min(abs(dp[0] - 1.), abs(dp[1] - 1.), abs(dp[0] - dp[1])) > 0.1

# %%
# A cone: the gap grows linearly in every direction
# ----------------------------------------------------------
# Near a diabolical point the two levels are :math:`\bar E \pm |\mathbf{q}|`
# for some linear map :math:`\mathbf{q}` of the displacement, so the gap is
# proportional to the distance -- with a slope that depends on the
# direction (an elliptic cone).

for theta in np.linspace(0., np.pi, 4, endpoint=False):
    u = np.array([np.cos(theta), np.sin(theta)])
    slopes = [gap(dp + r * u) / r for r in (1e-3, 1e-4)]
    print('direction {:5.1f} deg: gap/r = {:.5f} (r = 1e-3), {:.5f} (r = 1e-4)'
          .format(np.degrees(theta), *slopes))
    assert slopes[0] > 0.01
    assert np.isclose(slopes[0], slopes[1], rtol=1e-2)

# %%
# One parameter is not enough: avoided crossings
# ------------------------------------------------------
# Scanning :math:`t_1` at fixed :math:`t_2` gives an avoided crossing
# unless the line happens to go through the diabolical point exactly. The
# smallest gap along the line grows with the distance by which it misses.

t1_scan = np.linspace(dp[0] - 0.25, dp[0] + 0.25, 1001)
min_gaps = {}
for offset in (0., 0.01, 0.03):
    gaps = np.array([gap((t1, dp[1] + offset)) for t1 in t1_scan])
    min_gaps[offset] = gaps.min()
    print('t2 = t2* + {:.2f}: smallest gap along the line = {:.4f}'.format(offset, gaps.min()))
assert min_gaps[0.] < 2e-3 < min_gaps[0.01] < min_gaps[0.03]

# %%
# Berry phase pi: the eigenvector changes sign
# -----------------------------------------------------
# Carry the eigenvector of level 5 around a circle by parallel transport
# (:func:`tbkit.exceptional.encircle`). Around the diabolical point it
# comes back as :math:`-|u\rangle`: the phase is :math:`\pi`, for either
# level. Around a circle that misses it, it comes back unchanged. For a
# real symmetric matrix this is literally a sign change of the real
# eigenvector, which cannot be removed by any choice of phases.

around = ex.encircle(flake, ex.circle(dp, 0.05), band=N, nk=200)
upper = ex.encircle(flake, ex.circle(dp, 0.05), band=N + 1, nk=200)
missing = ex.encircle(flake, ex.circle(dp + (0.2, 0.), 0.05), band=N, nk=200)
print('phase around the DP: {:.6f} (level 5), {:.6f} (level 6); around nothing: {:.2e}'
      .format(around.phase, upper.phase, missing.phase))
assert around.final_band == N  # the eigenvalue comes back: no swap at a DP
assert np.isclose(abs(around.phase), np.pi, atol=1e-8)
assert np.isclose(abs(upper.phase), np.pi, atol=1e-8)
assert np.isclose(missing.phase, 0., atol=1e-8)
# the vorticity of the pair is 0 (unlike an exceptional point's 1/2)
assert abs(ex.vorticity(flake, ex.circle(dp, 0.05), (N, N + 1))) < 1e-10

# %%
# Not pinned by symmetry: the point moves with the potential
# --------------------------------------------------------------------
# A degeneracy forced by symmetry sits on a symmetric line or point. This
# one is *accidental*: change :math:`V` and it moves continuously across
# the plane, and it cannot disappear by a small perturbation -- it can
# only move, until it meets another diabolical point.

track = {}
guess = dp
for v in (0.3, 0.35, 0.4, 0.45, 0.5):
    track[v] = find_dp(v, guess)
    guess = track[v]
    print('V = {:.2f}: DP at ({:.4f}, {:.4f}), gap {:.1e}'.format(v, *track[v], gap(track[v], v)))
    assert gap(track[v], v) < 1e-10
assert np.linalg.norm(track[0.5] - track[0.3]) > 0.01

# %%
# The double cone, and the flip of the eigenvector
# ------------------------------------------------------

fig = plt.figure(figsize=(12, 4.6))
ax1 = fig.add_subplot(1, 3, 1, projection='3d')
r = np.linspace(0., 0.08, 25)
phi = np.linspace(0., 2 * np.pi, 49)
R, PHI = np.meshgrid(r, phi)
T1, T2 = dp[0] + R * np.cos(PHI), dp[1] + R * np.sin(PHI)
levels = np.array([[np.linalg.eigvalsh(flake((a, b)))[N:N + 2] for a, b in zip(ra, rb)]
                   for ra, rb in zip(T1, T2)])
for n, color in ((0, 'tab:blue'), (1, 'tab:red')):
    ax1.plot_surface(T1, T2, levels[..., n], color=color, alpha=0.7, linewidth=0)
ax1.set_xlabel('$t_1$')
ax1.set_ylabel('$t_2$')
ax1.set_zlabel('$E$')
ax1.set_title('A diabolo: levels 5 and 6')

ax2 = fig.add_subplot(1, 3, 2)
for offset, style in ((0., 'k-'), (0.01, 'b--'), (0.03, 'g:')):
    ens = np.array([np.linalg.eigvalsh(flake((t1, dp[1] + offset)))[N:N + 2] for t1 in t1_scan])
    ax2.plot(t1_scan, ens, style, lw=1.2)
ax2.set_xlabel('$t_1$')
ax2.set_ylabel('$E$')
ax2.set_title('Crossing only through the DP')

ax3 = fig.add_subplot(1, 3, 3)
overlap = [np.real(np.vdot(around.vector[0], v) / np.vdot(around.vector[0], around.vector[0]))
           for v in around.vector]
ax3.plot(around.s, overlap, 'k')
ax3.axhline(0., color='gray', lw=0.5)
ax3.set_xlabel('loop parameter $s$')
ax3.set_ylabel(r'$\langle u(0)|u(s)\rangle$')
ax3.set_title('Eigenvector returns as $-u$')
fig.tight_layout()
