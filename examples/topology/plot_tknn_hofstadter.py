r"""
TKNN: The Quantized Hall Conductance of the Hofstadter Bands
===================================================================

Thouless, Kohmoto, Nightingale and den Nijs (1982) computed the Hall
conductance of electrons on a lattice in a magnetic field, and found it
to be a topological integer -- the first Chern number of the occupied
bands. For a flux :math:`\alpha = p/q` per plaquette, the square
lattice's band splits into :math:`q` Hofstadter subbands, and the Hall
conductance with :math:`r` bands filled, :math:`\sigma_{xy} = t_r\,e^2/h`,
solves the Diophantine equation

.. math::

    r = q\,s_r + p\,t_r\, ,\qquad |t_r| \le q/2\, ,

so that each band carries :math:`C_r = t_r - t_{r-1}`.

:func:`~tbkit.kspace.magnetic_supercell` builds the magnetic unit cell of
:math:`q` plaquettes, with the Peierls phases of the field, and
:meth:`~tbkit.kspace.KSpace.chern_number` integrates each subband's
Berry curvature: the numbers come out as TKNN's integers.
"""
import numpy as np
import matplotlib.pyplot as plt

import tbkit.lattices as lattices
from tbkit.kspace import magnetic_supercell


square = [{'i': 0, 'j': 0, 'R': (1, 0), 't': 1.}, {'i': 0, 'j': 0, 'R': (0, 1), 't': 1.}]


def tknn(p, q, r):
    '''t_r from the Diophantine equation r = q s + p t, |t| <= q/2.'''
    return next(t for t in range(-q, q + 1) if (r - p*t) % q == 0 and abs(t) <= q/2)


# %%
# Chern numbers of the subbands, against the Diophantine equation
# -----------------------------------------------------------------------

for p, q in [(1, 3), (1, 5), (2, 5), (3, 7)]:
    mag = magnetic_supercell(lattices.square(), square, p, q)
    chern = np.array([mag.chern_number([n], nk=20) for n in range(q)])
    expected = [tknn(p, q, r + 1) - tknn(p, q, r) for r in range(q)]
    print('alpha = {}/{}: C = {}  (TKNN: {})'.format(p, q, np.round(chern).astype(int), expected))
    assert np.allclose(chern, expected, atol=1e-6)
    assert np.isclose(chern.sum(), 0., atol=1e-6)

# %%
# The Hall conductance in each gap
# ------------------------------------
# :math:`\sigma_{xy}` with the lowest :math:`r` bands filled is the sum of
# their Chern numbers, :math:`t_r`: for :math:`\alpha = 2/5`, the plateaus
# :math:`-2, 1, -1, 2`.

p, q = 2, 5
mag = magnetic_supercell(lattices.square(), square, p, q)
bands = mag.mesh_bands(20)
sigma = np.cumsum([mag.chern_number([n], nk=20) for n in range(q)])[:-1]
assert np.allclose(sigma, [tknn(p, q, r) for r in range(1, q)], atol=1e-6)
assert np.allclose(sigma, [-2, 1, -1, 2], atol=1e-6)
fig, ax = plt.subplots(figsize=(6, 4))
for n in range(q):
    ax.fill_between([0, 1], bands[:, n].min(), bands[:, n].max(), color='b', alpha=0.3)
for r in range(1, q):
    e_gap = (bands[:, r - 1].max() + bands[:, r].min()) / 2
    ax.text(0.5, e_gap, r'$\sigma_{xy} = %d\,e^2/h$' % round(sigma[r - 1]), ha='center', va='center')
ax.set_xticks([])
ax.set_ylabel('$E$')
ax.set_title(r'Hofstadter subbands at $\alpha = 2/5$ and their Hall plateaus')
fig.set_layout_engine('tight')
