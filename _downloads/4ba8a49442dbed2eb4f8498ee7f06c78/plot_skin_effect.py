r"""
The Non-Hermitian Skin Effect
===================================

Hatano and Nelson (1996) studied a chain whose hoppings to the right
and to the left differ, :math:`(H\psi)_n = t_R\psi_{n+1} + t_L\psi_{n-1}`
-- a non-reciprocal, non-Hermitian lattice. Twenty years later it became
the paradigm of a phenomenon with no Hermitian counterpart, the *skin
effect* (Yao and Wang; Kunst, Edvardsson, Budich and Bergholtz, 2018):

* on a ring, the spectrum :math:`E(k) = t_Re^{ik} + t_Le^{-ik}` is an
  ellipse, winding once around every energy inside it (spectral winding
  number :math:`W = \pm1`);
* on an open chain it collapses onto a real segment,
  :math:`|E| < 2\sqrt{t_Rt_L}`, and *every* eigenstate piles up at one
  end -- the bulk becomes a skin.

:meth:`~tbkit.kspace.KSpace.spectral_winding` and
:meth:`~tbkit.kspace.KSpace.finite_ham` compute both. The open-chain
bands live on a generalized Brillouin zone, see
``plot_generalized_brillouin_zone.py``.
"""
import numpy as np
import matplotlib.pyplot as plt

import tbkit.lattices as lattices
from tbkit.bridges import finite_system
from tbkit.kspace import KSpace
from tbkit.plot import Plot

t_right, t_left = 1., 0.5
hn = KSpace(lattices.chain())
hn.set_hopping([{'i': 0, 'j': 0, 'R': (1,), 't': t_right},
                        {'i': 0, 'j': 0, 'R': (-1,), 't': t_left}], hermitian=False)

# %%
# The lattice
# -----------
# Ten sites of the Hatano-Nelson chain
# (:func:`~tbkit.bridges.finite_system`): one site per cell and
# nearest-neighbour bonds, non-reciprocal -- each carries :math:`t_R` one
# way and :math:`t_L` the other. The widths drawn are those of
# :math:`t_R`.

fig_lat = Plot(finite_system(hn, 10)).lattice(plt_hop=True, ms=12, figsize=(7, 1.2))

# %%
# Ring and open chain
# -----------------------

n = 60
ring = np.linalg.eigvals(hn.finite_ham(n, periodic=True))
en, vec = np.linalg.eig(hn.finite_ham(n))
# compared as sets: E(k = pi) is real, and its angle (+pi or -pi) depends on
# the sign of a roundoff imaginary part
expected = [t_right*np.exp(1j*k) + t_left*np.exp(-1j*k) for k in 2*np.pi*np.arange(n)/n]
assert np.allclose(np.sort_complex(np.round(ring, 8)), np.sort_complex(np.round(expected, 8)))
assert np.max(np.abs(en.imag)) < 1e-8 and np.max(np.abs(en.real)) < 2*np.sqrt(t_right*t_left)
winding = hn.spectral_winding(0.)
print('Spectral winding about E = 0: {:.3f}; open-chain spectrum real, within +-{:.3f}.'
          .format(winding, 2*np.sqrt(t_right*t_left)))
assert np.isclose(winding, 1.)

# every eigenstate on the left end
weight = np.abs(vec) ** 2 / np.sum(np.abs(vec) ** 2, axis=0)
left = weight[:n // 4].sum(axis=0)
print('Weight on the leftmost quarter: at least {:.0%} for every eigenstate.'.format(left.min()))
assert left.min() > 0.9

# sphinx_gallery_thumbnail_number = 2
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].plot(ring.real, ring.imag, 'ob', ms=3, label='ring')
axes[0].plot(en.real, en.imag, 'or', ms=3, label='open chain')
axes[0].set_xlabel(r'Re $E$')
axes[0].set_ylabel(r'Im $E$')
axes[0].legend()
axes[0].set_aspect('equal')
axes[1].semilogy(np.arange(n), weight[:, np.argsort(en.real)[::10]], lw=1)
axes[1].set_xlabel('site')
axes[1].set_ylabel(r'$|\psi_n|^2$')
axes[1].set_title('open chain: every state on the left end')
fig.set_layout_engine('tight')
