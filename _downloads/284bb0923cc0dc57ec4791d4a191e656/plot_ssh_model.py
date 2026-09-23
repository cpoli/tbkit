r"""
The SSH Model: Bulk-Boundary Correspondence
=================================================

A 1D dimerized chain -- alternating intracell hopping v and intercell
hopping w -- is the earliest and simplest example of a topological
insulator: for w > v the chain is topological and an *open* finite chain
hosts a pair of exponentially localized, near-zero-energy edge modes, one
per end. For v > w it is trivial and no such modes exist. The bulk gap
(:math:`2|v-w|`) closes exactly at the transition, v = w.
"""
import numpy as np
import matplotlib.pyplot as plt

from tbkit.lattice import Lattice
from tbkit.system import System
from tbkit.plot import Plot
from tbkit.kspace import KSpace


unit_cell = [{'tag': 'a', 'r0': (0., 0.)}, {'tag': 'b', 'r0': (0.5, 0.)}]
prim_vec = [(1., 0.)]


# %%
# Bulk (k-space): the gap closes at v = w
# --------------------------------------------
# The two bands are :math:`E(k)=\pm|v+we^{-ik}|`, so the gap is
# :math:`2\min_k|v+we^{-ik}| = 2|v-w|` (minimized at :math:`k=\pi`),
# closing exactly at the transition v = w.

def ssh_bulk(v, w):
    lat = Lattice(unit_cell=unit_cell, prim_vec=prim_vec)
    ssh = KSpace(lat)
    ssh.set_hopping([{'i': 0, 'j': 1, 'R': (0,), 't': v},
                            {'i': 0, 'j': 1, 'R': (-1,), 't': w}])
    return ssh


for v, w in [(1., 0.5), (0.5, 1.), (1., 1.)]:
    en = ssh_bulk(v, w).mesh_bands(nk=200)
    gap = en[:, 1].min() - en[:, 0].max()
    assert np.isclose(gap, 2*abs(v - w), atol=1e-6)
print('Bulk gap = 2|v - w| confirmed for several (v, w).')

ssh = ssh_bulk(v=0.6, w=1.)
ssh.k_path([(-np.pi,), (np.pi,)], nk=200)
fig = ssh.plot_bands()

# %%
# Open finite chain: edge states in the topological phase only
# --------------------------------------------------------------------
# Edge states appear only for w > v, decaying exponentially into the bulk
# from each end.

def ssh_open_chain(v, w, n_cells):
    lat = Lattice(unit_cell=unit_cell, prim_vec=prim_vec)
    lat.get_lattice(n1=n_cells)
    sys = System(lat)
    hop = {}
    for n in range(n_cells):
        hop[(2*n, 2*n + 1)] = v
        if n < n_cells - 1:
            hop[(2*n + 1, 2*n + 2)] = w
    sys.set_hopping_manual(hop)
    sys.get_ham()
    return sys


# %%
# The two dimerizations, drawn. :meth:`tbkit.plot.Plot.lattice` draws each
# bond with a width proportional to its amplitude, so which bond sits at
# the chain's ends -- the whole content of the bulk-boundary
# correspondence here -- is visible directly.

fig_dim, axes_dim = plt.subplots(2, 1, figsize=(7.5, 3.4))
Plot(ssh_open_chain(v=0.4, w=1., n_cells=8)).lattice(plt_hop=True, ms=14,
                                                                             ax=axes_dim[0])
axes_dim[0].set_title('topological ($w>v$): ends on a weak bond', fontsize=12)
Plot(ssh_open_chain(v=1., w=0.4, n_cells=8)).lattice(plt_hop=True, ms=14,
                                                                             ax=axes_dim[1])
axes_dim[1].set_title('trivial ($v>w$): ends on a strong bond', fontsize=12)
fig_dim.tight_layout()

# %%
# Only the first of those two has edge states:

n_cells = 30
sys_trivial = ssh_open_chain(v=1., w=0.5, n_cells=n_cells)
sys_trivial.get_eig()
n_zero_trivial = np.sum(np.abs(sys_trivial.en.real) < 1e-6)
print('Trivial phase (v>w): {} exactly-zero-energy states (expect 0).'.format(n_zero_trivial))
assert n_zero_trivial == 0

sys_topological = ssh_open_chain(v=0.5, w=1., n_cells=n_cells)
sys_topological.get_eig(eigenvec=True)
n_zero_topological = np.sum(np.abs(sys_topological.en.real) < 1e-6)
print('Topological phase (w>v): {} exactly-zero-energy states (expect 2).'.format(n_zero_topological))
assert n_zero_topological == 2

# The two near-zero modes must be exponentially localized at the two ends.
idx = np.argsort(np.abs(sys_topological.en.real))[:2]
intensity = np.abs(sys_topological.rn[:, idx[0]]) ** 2
left_weight = intensity[:6].sum()
bulk_weight = intensity[2*n_cells//2 - 3: 2*n_cells//2 + 3].sum()
print('Edge-mode weight on the first 6 sites: {:.4f}; on 6 sites at the '
           'chain center: {:.6f}.'.format(left_weight, bulk_weight))
assert left_weight > 100 * bulk_weight

fig2, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, sys_, title in zip(axes, [sys_trivial, sys_topological],
                                       ['Trivial (v > w)', 'Topological (w > v)']):
    ax.plot(np.sort(sys_.en.real), 'o', ms=3)
    ax.axhline(0., color='k', lw=0.5)
    ax.set_title(title)
    ax.set_xlabel('state index')
axes[0].set_ylabel('$E$')
fig2.suptitle('SSH open chain ({} unit cells): spectrum'.format(n_cells))

# %%
# A localized state *inside* the chain: a two-weak-bond defect
# --------------------------------------------------------------------
# The zero mode does not have to sit at an end. Terminate the chain on
# **strong** bonds at *both* edges -- so neither end is topological and
# the bulk-boundary argument above gives no edge modes -- and place a
# defect in the middle, where two **weak** bonds meet. The site between
# those two weak bonds belongs to no strong dimer, and chiral symmetry
# pins a single exact zero mode to it, exponentially localized deep
# inside the bulk.
#
# The counting makes it unavoidable: strong bonds at both ends with one
# doubled weak bond leaves an odd number of sites, hence one more site on
# one sublattice than on the other, hence one unpaired -- zero-energy --
# state.

def ssh_defect_chain(strong, weak, n_cells):
    '''Open chain of 2*n_cells + 1 sites carrying a bond defect.

    The bond pattern is strong at both edges and alternates up to the
    centre site, where two weak bonds meet. *n_cells* must be even for
    both edges to end on a strong bond.
    '''
    assert n_cells % 2 == 0, 'n_cells must be even'
    # Build one cell too many and drop the last site: the chain needs an
    # odd number of sites, which no whole number of two-site cells gives.
    lat = Lattice(unit_cell=unit_cell, prim_vec=prim_vec)
    lat.get_lattice(n1=n_cells + 1)
    lat.remove_sites([2*(n_cells + 1) - 1])
    sys = System(lat)
    centre = n_cells    # bonds `centre - 1` and `centre` are both weak
    hop = {}
    for n in range(2*n_cells):
        is_strong = (n % 2 == 0) if n < centre else ((n - centre) % 2 == 1)
        hop[(n, n + 1)] = strong if is_strong else weak
    sys.set_hopping_manual(hop)
    sys.get_ham()
    return sys


# %%
# Drawn with bond widths proportional to the amplitudes, the two weak
# bonds at the centre and the strong bonds closing both ends are directly
# visible:

fig_def = Plot(ssh_defect_chain(strong=1., weak=0.4, n_cells=6)).lattice(
    plt_hop=True, ms=11, c=8., figsize=(8., 1.4))

# %%
# One state, and only one, sits exactly at zero -- well separated from
# the bulk bands -- and it lives at the defect, not at the edges.

sys_defect = ssh_defect_chain(strong=1., weak=0.4, n_cells=14)
sys_defect.get_eig(eigenvec=True)
en_defect = sys_defect.en.real
n_zero = np.sum(np.abs(en_defect) < 1e-10)
print('Defect chain: {} exactly-zero-energy state (expect 1).'.format(n_zero))
assert n_zero == 1

i0 = np.argmin(np.abs(en_defect))
intensity = np.abs(sys_defect.rn[:, i0]) ** 2
centre = sys_defect.lat.sites // 2
print('Zero mode: {:.1%} of its weight on the 3 sites around the defect, '
           '{:.1e} on the 3 sites at each end.'.format(
               intensity[centre-1:centre+2].sum(),
               intensity[:3].sum() + intensity[-3:].sum()))
assert np.argmax(intensity) == centre
assert intensity[centre-1:centre+2].sum() > 1e3 * (intensity[:3].sum()
                                                                        + intensity[-3:].sum())

# %%
# Its amplitude is confined to a single sublattice -- the one carrying
# the extra site -- and falls off by exactly -weak/strong from one dimer
# to the next on either side of the defect, the same exponential law that
# governs the edge modes above.

psi = sys_defect.rn[:, i0].real
assert np.allclose(psi[1::2], 0., atol=1e-10)
ratios = psi[centre+2::2] / psi[centre:-2:2]
print('Amplitude ratio per dimer: {:.3f} (expect {:.3f}).'.format(
           ratios[0], -0.4 / 1.))
assert np.allclose(ratios, -0.4, atol=1e-6)

fig3, axes3 = plt.subplots(1, 2, figsize=(10, 4))
axes3[0].plot(np.sort(en_defect), 'o', ms=3)
axes3[0].axhline(0., color='k', lw=0.5)
axes3[0].set_title('spectrum: one state at $E=0$')
axes3[0].set_xlabel('state index')
axes3[0].set_ylabel('$E$')
axes3[1].semilogy(np.arange(sys_defect.lat.sites), intensity + 1e-18, '-o', ms=4)
axes3[1].set_ylim(1e-12, 1.)
axes3[1].set_title(r'zero mode: $|\psi_j|^2$')
axes3[1].set_xlabel('site $j$')
fig3.suptitle('SSH chain with a two-weak-bond defect '
                     '(strong bonds at both edges)')
fig3.tight_layout()
