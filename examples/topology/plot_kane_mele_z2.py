r"""
The Kane-Mele Z2 Invariant from Wannier-Centre Flow
=========================================================

Kane and Mele's quantum spin Hall insulator has zero Chern number -- its
two spins carry opposite ones -- yet it is topological: it carries a
:math:`\mathbb{Z}_2` invariant :math:`\nu`, protected by time reversal,
that survives even when spin-orbit terms mix the two spins.

The cleanest way to see :math:`\nu` is the flow of the hybrid Wannier
centres: take Wilson loops along :math:`\mathbf{b}_1` and move them along
:math:`\mathbf{b}_2` across half the Brillouin zone. Time reversal pairs
the centres at the two ends (Kramers pairs); in the topological phase the
pairs *swap partners* on the way, so any horizontal line is crossed an
odd number of times. :meth:`~tbkit.kspace.KSpace.z2_invariant` counts
those crossings (the largest-gap method of Soluyanov and Vanderbilt,
2011): :math:`\nu = 1` below the critical staggered mass
:math:`|M| = 3\sqrt3\lambda`, :math:`\nu = 0` above it -- with or without
Rashba coupling.
"""
import numpy as np
import matplotlib.pyplot as plt

import tbkit.lattices as lattices
from tbkit.kspace import KSpace, PAULI
from tbkit.orbital import OrbitalSystem


lat = lattices.honeycomb()
lam = 0.06
M_c = 3 * np.sqrt(3) * lam


def kane_mele(M=0., rashba=0.):
    km = KSpace(lat, spin=True)
    km.set_hopping([{'i': 0, 'j': 1, 'R': R, 't': 1.} for R in [(0, 0), (-1, 0), (0, -1)]])
    for R in [(0, 1), (-1, 0), (1, -1)]:
        km.set_hopping([{'i': 0, 'j': 0, 'R': R, 't': 1j*lam*PAULI['z']},
                                {'i': 1, 'j': 1, 'R': R, 't': -1j*lam*PAULI['z']}])
    if rashba:
        tau = np.array([d['r0'] for d in lat.unit_cell])
        a = np.array(lat.prim_vec)
        for R in [(0, 0), (-1, 0), (0, -1)]:
            d = tau[1] + R[0]*a[0] + R[1]*a[1] - tau[0]
            d = d / np.linalg.norm(d)
            km.set_hopping([{'i': 0, 'j': 1, 'R': R,
                                     't': 1j*rashba*(PAULI['x']*d[1] - PAULI['y']*d[0])}])
    km.set_onsite({'a': M, 'b': -M})
    return km


# %%
# The flow: Kramers pairs swap partners
# ------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
for ax, M, title in [(axes[0], 0.1, 'topological'), (axes[1], 0.5, 'trivial')]:
    fracs, centres = kane_mele(M).wannier_flow([0, 1], nk=60, nk_perp=41, k_range=(0., 0.5),
                                                                   positions=False)
    ax.plot(fracs, centres, 'o', color='b', ms=3)
    ax.set_title(r'$M = {:.2f}$: {}'.format(M, title))
    ax.set_xlabel(r'$k_2/|\mathbf{b}_2|$')
axes[0].set_ylabel('hybrid Wannier centres')
fig.set_layout_engine('tight')

# %%
# The invariant across the phase diagram
# -------------------------------------------

masses = np.array([0., 0.1, 0.2, 0.4, 0.5])
nu = [kane_mele(M).z2_invariant([0, 1], nk=60, nk_perp=41) for M in masses]
print('M / M_c:', np.round(masses / M_c, 2), '-> nu =', nu)
assert nu == [int(M < M_c) for M in masses]

# Rashba coupling mixes the spins (S_z is no longer conserved, so there is
# no spin Chern number), but nu does not change
nu_r = kane_mele(0.1, rashba=0.05).z2_invariant([0, 1], nk=60, nk_perp=41)
assert nu_r == 1
print('With Rashba coupling: nu = {}.'.format(nu_r))
# and the total Chern number is zero throughout
assert abs(kane_mele(0.1, rashba=0.05).chern_number([0, 1], nk=20)) < 1e-6

# %%
# The same model in real space: a spinful flake
# ------------------------------------------------
# :class:`~tbkit.orbital.OrbitalSystem` builds the same Hamiltonian on a
# finite flake (Kane-Mele's :math:`\nu_{ij}` from the geometry of each
# second-neighbour path). Its spectrum equals that of the Bloch model cut
# to the same open parallelogram, and shows in-gap states on the edges in
# the topological phase.

flake = lattices.honeycomb()
flake.get_lattice(12, 12)
sys = OrbitalSystem(flake, spin=True)
sys.set_hopping([{'n': 1, 't': 1.}])
sys.set_spin_orbit(lam)
sys.set_onsite({'a': 0.1, 'b': -0.1})
sys.get_ham()
sys.get_eig(eigenvec=True)
assert np.allclose(sys.en, np.linalg.eigvalsh(kane_mele(0.1).finite_ham((12, 12))))
bands = kane_mele(0.1).mesh_bands(60)
bulk_gap = np.min(bands[:, 2] - bands[:, 1])
in_gap = np.abs(sys.en) < bulk_gap / 2
x, y = flake.coor['x'], flake.coor['y']
edge = (x - x.min() < 1.5) | (x.max() - x < 1.5) | (y - y.min() < 1.) | (y.max() - y < 1.)
weight_on_edge = sys.intensity[edge][:, in_gap].sum(axis=0)
print('{} in-gap states, {:.0%} of their weight on the edge sites ({:.0%} of the sites).'
          .format(in_gap.sum(), weight_on_edge.mean(), edge.mean()))
assert in_gap.sum() > 0 and weight_on_edge.mean() > 2 * edge.mean()
