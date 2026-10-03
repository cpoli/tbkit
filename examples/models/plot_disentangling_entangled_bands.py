r"""
Disentanglement: Wannier Functions of the π Bands of Graphene
================================================================

Maximal localization needs an isolated group of bands. Most bands of
interest are not isolated: in graphene with its four valence orbitals
:math:`s, p_x, p_y, p_z` per carbon, the π bands cross the σ bands, so no
set of band indices follows the π bands over the whole Brillouin zone.

I. Souza, N. Marzari and D. Vanderbilt (2001) separated the two steps. At
each k, the states of an *outer energy window* span more than the
:math:`n_W` Wannier functions need. Among their :math:`n_W`-dimensional
subspaces, choose the one that varies least from k to k, the one that
minimizes the gauge-invariant spread

.. math::

    \Omega_I = \frac{1}{N}\sum_{\mathbf{k},\mathbf{b}} w_b
    \left(n_W - \mathrm{Tr}\,[P_{\mathbf{k}}P_{\mathbf{k}+\mathbf{b}}]\right)\, ,

then localize within it as for an isolated group. The states of an inner
*frozen window* are kept as they are, so the Wannier functions reproduce
the bands there exactly.

:func:`tbkit.wannier.wannierize` disentangles when it gets more bands than
trial orbitals (here all eight, with two :math:`p_z` trial orbitals).
:meth:`~tbkit.wannier.WannierFunctions.kspace` then returns the
Hamiltonian in the Wannier basis as a two-orbital
:class:`~tbkit.kspace.KSpace`: Wannier interpolation of the π bands.
"""
import numpy as np
import matplotlib.pyplot as plt

import tbkit.lattices as lattices
from tbkit.kspace import high_symmetry_path
from tbkit.lattice import Lattice
from tbkit.slater_koster import sk_kspace
from tbkit.wannier import wannierize

# eV; energies relative to the 2p level (Saito, Dresselhaus and Dresselhaus)
orbitals = {'a': ['s', 'px', 'py', 'pz'], 'b': ['s', 'px', 'py', 'pz']}
onsite = {'a': {'s': -8.87}, 'b': {'s': -8.87}}
hopping = {'ss_sigma': -6.77, 'sp_sigma': 5.58, 'pp_sigma': 5.04, 'pp_pi': -3.03}


def graphene(buckling):
    '''sp3 graphene, with sublattice b raised by *buckling* out of the plane.'''
    honeycomb = lattices.honeycomb()
    a, b = (tuple(site['r0']) for site in honeycomb.unit_cell)
    lat = Lattice(unit_cell=[{'tag': 'a', 'r0': a + (0.,)}, {'tag': 'b', 'r0': b + (buckling,)}],
                       prim_vec=[tuple(p) + (0.,) for p in honeycomb.prim_vec])
    return sk_kspace(lat, orbitals, {1: hopping}, onsite=onsite)


flat = graphene(0.)
pz = [n for n, (_, o) in enumerate(flat.sk_orbitals) if o == 'pz']
points, labels = high_symmetry_path(flat.lat)

# %%
# Flat graphene: the π bands cross the σ bands
# ------------------------------------------------
# The π band starts 9 eV below the Dirac point at Γ, under the top of the
# σ bands at -3 eV, and crosses them on its way up to K. Sorted by energy,
# band 3 is σ near Γ and π near K.

dist, en = flat.k_path(points, 40)
kpath, nodes = flat.ks.copy(), flat.nodes
# the pi bands alone: the pz block of H(k), which flat graphene decouples
pi_bands = np.linalg.eigvalsh(np.array([flat.get_ham(k)[np.ix_(pz, pz)] for k in kpath]))
print('pi band at Gamma: {:.2f} eV, top of the sigma bands: {:.2f} eV'.format(pi_bands[0, 0], en[0, 2]))
assert pi_bands[0, 0] < en[0, 2] - 6.

# %%
# The isolated-group recipe fails: band 3 meets the σ band below it on the
# mesh, so :func:`~tbkit.wannier.wannierize` refuses it. Given all eight
# bands, the two :math:`p_z` trial orbitals, and a frozen window of
# :math:`\pm 2` eV around the Dirac point (where only π states live), it
# disentangles them. In flat graphene, :math:`p_z` is odd under the mirror
# :math:`z\to -z` and the σ orbitals are even, so the smoothest subspace is
# exactly the :math:`p_z` one: :math:`\Omega_I = 0` (point orbitals), the
# Wannier functions are the :math:`p_z` orbitals, and their model is
# Wallace's, with the single hopping :math:`V_{pp\pi}`, exact everywhere.

try:
    wannierize(flat, [3, 4], pz, nk=12)
    raise AssertionError('the pi bands are not isolated')
except ValueError as err:
    print('isolated group:', str(err).strip().split(':')[0])

wf = wannierize(flat, list(range(8)), pz, nk=12, frozen=(-2., 2.))
weight = np.abs(wf.functions.reshape(2, -1, flat.norb)) ** 2
print('Omega_I = {:.1e}, weight on pz = {}'.format(wf.omega_i, np.sum(weight[:, :, pz], axis=(1, 2))))
assert abs(wf.omega_i) < 1e-10 and np.allclose(np.sum(weight[:, :, pz], axis=(1, 2)), 1.)
wallace = wf.kspace()
t = np.array([h[3] for h in wallace._hop])
print('Wannier model: |t| = {} (the three bonds and their conjugates)'.format(np.round(np.abs(t), 6)))
assert np.allclose(np.abs(t), 3.03) and np.allclose(wallace.onsite, 0.)
assert np.allclose(wallace.k_path(points, 40)[1], pi_bands, atol=1e-10)

# %%
# Buckled graphene: the π and σ states mix
# --------------------------------------------
# Raising sublattice ``b`` by 0.25 out of the plane (as in silicene)
# breaks the mirror: the :math:`p_z` orbitals now hybridize with the σ
# ones, and the π-like subspace is no longer a set of orbitals. The
# iteration lowers :math:`\Omega_I` from its projected value at every
# step. The frozen bands are reproduced exactly on the k-mesh, and
# between its points the interpolation converges as the mesh is refined
# (the error is largest at the top of the frozen window, where states
# leave it).

buckled = graphene(0.25)
_, en_b = buckled.k_path(points, 40)
window = (-1.5, 1.5)
inside = (en_b > window[0]) & (en_b < window[1])
results = {}
for nk in (6, 12):
    wf_b = wannierize(buckled, list(range(8)), pz, nk=nk, frozen=window)
    wk = wf_b.kspace()
    _, mesh = buckled.mesh_grid(nk)
    e_mesh, e_wann = buckled._eigs(mesh), wk._eigs(mesh)
    frozen_mesh = (e_mesh >= window[0]) & (e_mesh <= window[1])
    err_mesh = max(np.abs(e_wann[k][:, None] - e_mesh[k][frozen_mesh[k]][None]).min(axis=0).max()
                          for k in range(len(mesh)) if frozen_mesh[k].any())
    e_path = wk.k_path(points, 40)[1]
    err_path = max(np.abs(e_path[k][:, None] - en_b[k][inside[k]][None]).min(axis=0).max()
                          for k in range(len(kpath)) if inside[k].any())
    results[nk] = wf_b, e_path, err_path
    print('nk = {:2d}: Omega_I {:.4f} -> {:.4f} in {} steps; frozen bands on the mesh to {:.0e}, '
           'on the path to {:.3f} eV'.format(nk, wf_b.dis_history[0], wf_b.omega_i,
                                                   len(wf_b.dis_history) - 1, err_mesh, err_path))
    assert wf_b.dis_converged and np.all(np.diff(wf_b.dis_history) <= 1e-12)
    assert wf_b.omega_i < wf_b.dis_history[0] / 2
    assert err_mesh < 1e-10
assert results[12][2] < 0.05 and results[12][2] < results[6][2] / 4

# %%
# Plots: the buckled bands (grey) with the Wannier-interpolated π bands
# from the 12 x 12 mesh, and the descent of :math:`\Omega_I`.

wf_b, e_path, _ = results[12]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))
ax1.plot(dist, en_b, color='0.6', lw=1)
ax1.plot(dist, e_path, '--', color='C0', lw=1.5)
ax1.plot([], [], color='0.6', label='sp$^3$ model (8 bands)')
ax1.plot([], [], '--', color='C0', label='Wannier interpolation (2 bands)')
ax1.axhspan(*window, color='C0', alpha=0.12, label='frozen window')
for x in nodes:
    ax1.axvline(x, color='k', lw=0.5)
ax1.set_xticks(nodes)
ax1.set_xticklabels(labels)
ax1.set_xlim(dist[0], dist[-1])
ax1.set_ylim(-12, 8)
ax1.set_ylabel('$E$ (eV)')
ax1.set_title('Buckled graphene (0.25)')
ax1.legend(fontsize=8, loc='lower right')
for nk, (w, _, _) in results.items():
    ax2.semilogy(w.dis_history[:-1] - w.omega_i, label='{0} x {0} mesh'.format(nk))
ax2.set_xlabel('disentanglement step')
ax2.set_ylabel(r'$\Omega_I - \Omega_I^{\min}$')
ax2.set_title(r'Minimization of $\Omega_I$')
ax2.legend(fontsize=8)
fig.set_layout_engine('tight')
plt.show()
