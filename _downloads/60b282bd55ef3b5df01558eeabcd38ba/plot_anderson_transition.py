r"""
The Scaling Theory of Localization: the 3D Anderson Transition
=====================================================================

Abrahams, Anderson, Licciardello and Ramakrishnan (1979) argued that the
conductance of a disordered sample flows, as its size :math:`L` grows,
according to a single scaling function of the conductance itself. The
consequence: in one and two dimensions every state is localized by any
disorder, but in three dimensions a genuine metal-insulator transition
survives, at a critical disorder -- :math:`W_c \approx 16.5\,t` for the
Anderson model on the cubic lattice, with onsite energies uniform in
:math:`[-W/2, W/2]`.

The transition shows in the statistics of the energy levels. Extended
states repel each other, and their spacings follow random-matrix (GOE)
statistics; localized states are independent, with Poisson statistics.
The mean ratio of consecutive spacings
:math:`r_n = \min(s_n, s_{n+1})/\max(s_n, s_{n+1})` tells them apart --
:math:`\langle r\rangle \approx 0.5307` (GOE), :math:`0.3863` (Poisson)
-- and, as :math:`L` grows, flows towards the first value in the metal
and towards the second in the insulator: the curves for two sizes cross
at the transition.
"""
import numpy as np
import matplotlib.pyplot as plt

from tbkit.lattice import Lattice
from tbkit.system import System

R_GOE, R_POISSON = 0.5307, 0.3863
cubic = [(1., 0., 0.), (0., 1., 0.), (0., 0., 1.)]


def anderson(L):
    lat = Lattice(unit_cell=[{'tag': 'a', 'r0': (0., 0., 0.)}], prim_vec=cubic)
    lat.get_lattice(L, L, L)
    sys = System(lat)
    sys.set_hopping([{'n': 1, 't': 1.}])
    return sys


def mean_ratio(sys, W, samples, seed=0):
    '''<r> over the central 40% of the spectrum, averaged over disorder samples.'''
    np.random.seed(seed)
    ratios = []
    for _ in range(samples):
        sys.set_onsite({'a': 0.})
        sys.set_onsite_dis(W / 2)  # uniform in [-W/2, W/2]
        sys.get_ham()
        sys.get_eig()
        en = np.sort(sys.en)
        mid = en[int(0.3 * len(en)):int(0.7 * len(en))]
        s = np.diff(mid)
        ratios.append(np.mean(np.minimum(s[1:], s[:-1]) / np.maximum(s[1:], s[:-1])))
    return np.mean(ratios), np.std(ratios) / np.sqrt(samples)


# %%
# Level statistics across the transition
# -------------------------------------------

disorders = [4., 8., 12., 16.5, 20., 25., 35.]
sizes = {6: 60, 10: 20}  # L: disorder samples
curves = {L: np.array([mean_ratio(anderson(L), W, n) for W in disorders]) for L, n in sizes.items()}
for L, curve in curves.items():
    print('L = {:2d}: <r> ='.format(L), np.round(curve[:, 0], 3))

# weak disorder: extended states, random-matrix statistics; strong
# disorder: localized, close to Poisson
assert abs(curves[10][0, 0] - R_GOE) < 0.015 and abs(curves[6][0, 0] - R_GOE) < 0.015
assert curves[6][-1, 0] < 0.415 and curves[10][-1, 0] < 0.415


def grows(W):
    """Does <r> grow from L = 6 to L = 10 at disorder W (by more than 2 sigma)?"""
    (r6, e6), (r10, e10) = curves[6][disorders.index(W)], curves[10][disorders.index(W)]
    return (r10 - r6) / np.hypot(e6, e10)


# the scaling signature: below W_c the larger sample is more metallic,
# above it more insulating -- the curves cross in between
assert grows(12.) > 2 and grows(20.) < -2
print('<r> grows with L at W = 12 ({:+.1f} sigma), shrinks at W = 20 ({:+.1f} sigma): '
          'the curves cross between them, around W_c = 16.5.'.format(grows(12.), grows(20.)))

fig, ax = plt.subplots(figsize=(6.2, 4))
for L, curve in curves.items():
    ax.errorbar(disorders, curve[:, 0], yerr=curve[:, 1], fmt='o-', ms=4, label='$L = {}$'.format(L))
ax.axhline(R_GOE, color='k', ls=':', lw=1)
ax.axhline(R_POISSON, color='k', ls=':', lw=1)
ax.axvline(16.5, color='0.5', ls='--', lw=1)
ax.text(4.5, R_GOE + 0.005, 'GOE (metal)')
ax.text(4.5, R_POISSON + 0.005, 'Poisson (insulator)')
ax.set_xlabel('disorder $W/t$')
ax.set_ylabel(r'$\langle r\rangle$')
ax.legend()
ax.set_title('3D Anderson transition ($W_c \\approx 16.5\\,t$)')
fig.set_layout_engine('tight')
