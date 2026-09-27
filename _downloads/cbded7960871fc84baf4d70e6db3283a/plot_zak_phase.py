r"""
The Zak Phase: Where the SSH Chain's Electrons Sit
=======================================================

J. Zak showed in 1989 that the Berry phase acquired by a Bloch state
carried once across the Brillouin zone,

.. math::

    \gamma = \oint \mathbf{A}\cdot d\mathbf{k}\, ,\qquad
    \mathbf{A} = i\langle u_k|\partial_k u_k\rangle\, ,

is a property of the whole band -- the *Zak phase* -- and that
:math:`\gamma/2\pi` is the centre of the band's Wannier functions in
units of the lattice constant: the electronic polarization. With
inversion symmetry it is quantized to 0 or :math:`\pi`.

In the SSH chain (sites at :math:`x = 0` and :math:`x = 1/2`, intracell
hopping :math:`v`, intercell :math:`w`), the occupied band's Wannier
function sits on the stronger bond: at the intracell bond centre
:math:`x = 1/4` when :math:`v > w`, at the intercell one :math:`x = 3/4`
when :math:`w > v`. The two phases differ by :math:`\pi` -- half an
electron per cell moved by half a cell -- and the jump is what leaves
the end charges and zero modes of an open topological chain.
:meth:`~tbkit.kspace.KSpace.berry_phase` and
:meth:`~tbkit.kspace.KSpace.wannier_centers` compute both, from the
gauge-invariant discrete Wilson loop.
"""
import numpy as np
import matplotlib.pyplot as plt

from tbkit.lattice import Lattice
from tbkit.kspace import KSpace


def ssh(v, w):
    lat = Lattice(unit_cell=[{'tag': 'a', 'r0': (0., 0.)}, {'tag': 'b', 'r0': (0.5, 0.)}],
                         prim_vec=[(1., 0.)])
    chain = KSpace(lat)
    chain.set_hopping([{'i': 0, 'j': 1, 'R': (0,), 't': v},
                                {'i': 1, 'j': 0, 'R': (1,), 't': w}])
    return chain


# %%
# The Wannier centre sits on the strong bond
# -------------------------------------------------

for v, w, centre in [(1., 0.4, 0.25), (0.4, 1., 0.75)]:
    x = ssh(v, w).wannier_centers(0, nk=200)[0]
    print('v = {}, w = {}: Wannier centre x = {:.6f}'.format(v, w, x))
    assert np.isclose(x, centre, atol=1e-8)

# %%
# Across the transition: a jump of pi
# -------------------------------------
# The Zak phase is pinned while the gap stays open, and jumps by
# :math:`\pi` where it closes, at :math:`v = w`.

ratios = np.linspace(0.2, 2., 37)
zak = np.array([ssh(r, 1.).berry_phase(0, nk=200) for r in ratios])
centres = (zak / (2*np.pi)) % 1.
topo = ratios < 1.
assert np.allclose(centres[topo], 0.75, atol=1e-8)
assert np.allclose(centres[~topo & ~np.isclose(ratios, 1.)], 0.25, atol=1e-8)
print('Wannier centre 3/4 for v < w, 1/4 for v > w: a jump of half a cell (Zak phase pi).')

fig, ax = plt.subplots(figsize=(6, 3.6))
ax.plot(ratios, centres, 'o-b')
ax.axvline(1., color='k', ls='--', lw=1)
ax.set_xlabel('$v/w$')
ax.set_ylabel(r'Wannier centre $\gamma/2\pi$')
ax.set_title('Zak phase of the SSH chain')
fig.set_layout_engine('tight')

# %%
# The polarization is a property of the band, not of the gauge
# -------------------------------------------------------------------
# Placing both orbitals on the cell origin (``positions=False``) shifts
# each phase, but not the quantized difference between the two phases.

diff = (ssh(0.4, 1.).berry_phase(0, positions=False)
            - ssh(1., 0.4).berry_phase(0, positions=False)) % (2*np.pi)
assert np.isclose(diff, np.pi)
print('Zak-phase difference in the periodic gauge: {:.6f} = pi.'.format(diff))
