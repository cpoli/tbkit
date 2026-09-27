r"""
Band Unfolding: the Effective Band Structure of a Disordered Supercell
==========================================================================

Disorder, defects and superstructures need a supercell, and a supercell
folds the bands into a smaller Brillouin zone: :math:`N` times more bands,
most of them spectators. W. Ku, T. Berlijn and C.-C. Lee (2010), and V.
Popescu and A. Zunger, unfolded them back: each supercell state
:math:`|\mathbf{K}J\rangle` gets the weight
:math:`W_J(\mathbf{k}) = \sum_o|\langle\mathbf{k}o|\mathbf{K}J\rangle|^2` at
every primitive :math:`\mathbf{k}` folding onto :math:`\mathbf{K}`, and the
spectral function :math:`A(\mathbf{k},\omega) = \sum_J W_J(\mathbf{k})
\delta(\omega-E_J)` -- what angle-resolved photoemission measures --
is drawn in the primitive zone again.

* A pristine supercell unfolds *exactly* onto the primitive bands (weights
  0 or 1).
* A superlattice potential mixes :math:`\mathbf{k}` with
  :math:`\mathbf{k}+\mathbf{G}`: on a chain with a staggered potential
  :math:`\pm V`, the two bands :math:`\pm E_k = \pm\sqrt{\epsilon_k^2+V^2}`
  carry the coherence factors :math:`(1\pm\epsilon_k/E_k)/2`.
* Random onsite energies smear the bands of graphene into a broadened
  :math:`A(\mathbf{k},\omega)`, sharp near the Dirac point and wide where
  the density of states is large.

:mod:`tbkit.moire`: :func:`~tbkit.moire.supercell`,
:func:`~tbkit.moire.unfold`, :func:`~tbkit.moire.spectral_function`.
"""
import numpy as np
import matplotlib.pyplot as plt

import tbkit.lattices as lattices
from tbkit.kspace import KSpace
from tbkit.moire import spectral_function, supercell, unfold

t = -1.
gra = KSpace(lattices.honeycomb())
gra.set_hopping([{'i': 0, 'j': 1, 'R': (0, 0), 't': t}, {'i': 0, 'j': 1, 'R': (-1, 0), 't': t},
                          {'i': 0, 'j': 1, 'R': (0, -1), 't': t}])

# %%
# A pristine supercell unfolds exactly
# ---------------------------------------

sc = supercell(gra, [[3, 0], [0, 3]])
b1, b2 = gra.rec_vec_k
K = (2 * b1 + b2) / 3
M = b1 / 2
nodes = [np.zeros(2), K, M, np.zeros(2)]
path = np.concatenate([np.linspace(nodes[i], nodes[i + 1], 40, endpoint=False) for i in range(3)]
                               + [nodes[-1:]])
en, w = unfold(sc, path)
assert np.allclose(w.sum(axis=1), 2.)
for k, e, ww in zip(path, en, w):
    # the weight sits on the primitive bands only (degenerate folded states,
    # e.g. at Gamma and K, share it)
    prim = np.linalg.eigvalsh(gra.get_ham(k))
    on = np.min(np.abs(e[:, None] - prim[None, :]), axis=1) < 1e-9
    assert np.allclose(ww[~on], 0., atol=1e-10)
    for p in prim:
        assert np.isclose(ww[np.abs(e - p) < 1e-9].sum(), np.sum(np.abs(prim - p) < 1e-9))
print('{} folded bands; all the weight on the 2 graphene bands.'.format(sc.norb))

# %%
# Coherence factors of a staggered chain
# -----------------------------------------

chain = KSpace(lattices.chain())
chain.set_hopping([{'i': 0, 'j': 0, 'R': (1,), 't': 1.}])
cdw = supercell(chain, 2)
V = 0.4
cdw.onsite = np.array([V, -V], dtype='c16')
kc = np.linspace(-np.pi, np.pi, 201)[:, None]
en_c, w_c = unfold(cdw, kc)
eps = 2 * np.cos(kc[:, 0])
assert np.allclose(en_c[:, 1], np.sqrt(eps**2 + V**2))
assert np.allclose(w_c[:, 1], (1 + eps / np.sqrt(eps**2 + V**2)) / 2)

# %%
# Anderson disorder in a 6 x 6 graphene supercell
# --------------------------------------------------
# Onsite energies uniform in :math:`[-W/2, W/2]`, averaged over eight
# configurations. The broadening of the peak at a given :math:`\mathbf{k}`
# grows with :math:`W`, and the integrated weight stays 2 (sum rule).

omega = np.linspace(-3.5, 3.5, 351)
big = supercell(gra, [[6, 0], [0, 6]])
rng = np.random.default_rng(0)
maps, widths = {}, {}
for W in (0., 0.8, 2.):
    acc = np.zeros((len(path), len(omega)))
    for _ in range(8 if W else 1):
        big.onsite = rng.uniform(-W / 2, W / 2, big.norb).astype('c16')
        acc += spectral_function(big, path, omega, broadening=0.04)
    maps[W] = acc / (8 if W else 1)
    # half width at half maximum of the upper peak at k = M
    row = maps[W][80, omega > 0]
    widths[W] = 0.5 * (omega[1] - omega[0]) * np.sum(row > row.max() / 2)
fine = np.linspace(-40., 40., 40001)
norm = np.trapezoid(spectral_function(big, path[:1], fine, 0.04), fine)[0]
assert abs(norm - 2.) < 2e-3
assert widths[0.] < widths[0.8] < widths[2.]
print('half widths at M: ' + ', '.join('W = {}: {:.3f}'.format(W, h) for W, h in widths.items()))

fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
x = np.arange(len(path))
for ax, (W, a) in zip(axes, maps.items()):
    ax.pcolormesh(x, omega, a.T, cmap='magma', vmax=np.percentile(a, 99), shading='auto')
    ax.plot(x, [np.linalg.eigvalsh(gra.get_ham(k)) for k in path], c='w', lw=0.5, ls='--')
    ax.set_xticks([0, 40, 80, 120])
    ax.set_xticklabels([r'$\Gamma$', '$K$', '$M$', r'$\Gamma$'])
    ax.set_title('$W = {}|t|$'.format(W))
axes[0].set_ylabel(r'$\omega/|t|$')
fig.suptitle(r'Unfolded $A(\mathbf{k},\omega)$ of a $6\times6$ graphene supercell')
fig.tight_layout()
