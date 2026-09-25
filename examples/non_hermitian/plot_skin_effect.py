r"""
The Non-Hermitian Skin Effect and the Generalized Brillouin Zone
======================================================================

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

The open-chain spectrum is not given by Bloch waves on the unit circle
:math:`|\beta| = |e^{ik}| = 1`, but on a *generalized* Brillouin zone,
here a circle of radius :math:`|\beta| = \sqrt{t_L/t_R}`; the
eigenstates decay as :math:`|\beta|^n`.
:meth:`~tbkit.kspace.KSpace.spectral_winding`,
:meth:`~tbkit.kspace.KSpace.finite_ham` and
:meth:`~tbkit.kspace.KSpace.gbz` compute all three.
"""
import numpy as np
import matplotlib.pyplot as plt

import tbkit.lattices as lattices
from tbkit.kspace import KSpace

t_right, t_left = 1., 0.5
hn = KSpace(lattices.chain())
hn.set_hopping([{'i': 0, 'j': 0, 'R': (1,), 't': t_right},
                        {'i': 0, 'j': 0, 'R': (-1,), 't': t_left}], hermitian=False)

# %%
# Ring and open chain
# -----------------------

n = 60
ring = np.linalg.eigvals(hn.finite_ham(n, periodic=True))
en, vec = np.linalg.eig(hn.finite_ham(n))
assert np.allclose(sorted(ring, key=np.angle), sorted(
    [t_right*np.exp(1j*k) + t_left*np.exp(-1j*k) for k in 2*np.pi*np.arange(n)/n], key=np.angle))
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

# %%
# The generalized Brillouin zone
# ----------------------------------

beta = hn.gbz(en)
print('|beta| on the GBZ: {:.6f} (sqrt(t_L/t_R) = {:.6f})'.format(np.abs(beta).mean(), np.sqrt(t_left/t_right)))
assert np.allclose(np.abs(beta), np.sqrt(t_left / t_right))
# the open-chain spectrum is H(beta) on it
for e, b in zip(en[:10], beta[:10, 0]):
    assert np.isclose(hn.get_ham_beta(b)[0, 0], e)

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
