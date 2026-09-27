r"""
The Fu-Kane Parity Criterion: Z2 from Inversion Eigenvalues
=================================================================

L. Fu and C. Kane showed in 2007 that for a crystal with inversion
symmetry the :math:`\mathbb{Z}_2` invariant needs no Wilson loop at all:
it is fixed by the parities :math:`\xi = \pm1` of the occupied Kramers
pairs at the four time-reversal-invariant momenta :math:`\Gamma_i`,

.. math::

    (-1)^\nu = \prod_{i=1}^{4}\delta_i\, ,\qquad
    \delta_i = \prod_{m}\xi_{2m}(\Gamma_i)\, .

A band inversion at one :math:`\Gamma_i` flips :math:`\delta_i` and makes
the insulator topological. The criterion led directly to the prediction
of the 3D topological insulators (Bi\ :sub:`1-x`\ Sb\ :sub:`x`, then Bi\ :sub:`2`\ Se\ :sub:`3`).

Here it is applied to the Bernevig-Hughes-Zhang (BHZ) model of HgTe
quantum wells: s-like and p-like orbitals of opposite parity on a square
lattice, with the mass term
:math:`d_z(\mathbf{k}) = m - 2(2 - \cos k_x - \cos k_y)`. The parity
product is compared, across the phase diagram, with the invariant from
the Wannier-centre flow (:meth:`~tbkit.kspace.KSpace.z2_invariant`),
which knows nothing about inversion: they agree everywhere.
"""
import numpy as np
import matplotlib.pyplot as plt

from tbkit.lattice import Lattice
from tbkit.kspace import KSpace, PAULI


# orbitals: (s, up), (s, down), (p, up), (p, down) -> two "sites" with spin
lat = Lattice(unit_cell=[{'tag': 's', 'r0': (0., 0.)}, {'tag': 'p', 'r0': (0., 0.)}],
                     prim_vec=[(1., 0.), (0., 1.)])
SZ = PAULI['z']
INVERSION = np.kron(np.diag([1., -1.]), np.eye(2))  # s even, p odd
TIME_REVERSAL = np.kron(np.eye(2), 1j * PAULI['y'])


def bhz(m, a=1.):
    r'''
    H(k) = sin kx tau_x s_z + sin ky tau_y + d_z(k) tau_z  (tau: orbital, s: spin),
    i.e. two time-reversed Chern insulators, one per spin.
    '''
    model = KSpace(lat, spin=True)
    model.set_onsite({'s': (m - 4.) * PAULI['0'], 'p': -(m - 4.) * PAULI['0']})
    for R in [(1, 0), (0, 1)]:
        model.set_hopping([{'i': 0, 'j': 0, 'R': R, 't': PAULI['0']},
                                    {'i': 1, 'j': 1, 'R': R, 't': -PAULI['0']}])
    # tau_x s_z sin kx + tau_y sin ky between s and p
    model.set_hopping([{'i': 0, 'j': 1, 'R': (1, 0), 't': -0.5j * a * SZ},
                                {'i': 0, 'j': 1, 'R': (-1, 0), 't': 0.5j * a * SZ},
                                {'i': 0, 'j': 1, 'R': (0, 1), 't': -0.5 * a * PAULI['0']},
                                {'i': 0, 'j': 1, 'R': (0, -1), 't': 0.5 * a * PAULI['0']}])
    return model


# %%
# Both symmetries hold
# --------------------------

model = bhz(1.)
assert model.symmetry_error(INVERSION) < 1e-12
assert model.symmetry_error(TIME_REVERSAL, antiunitary=True) < 1e-12
print('Inversion and time reversal (T^2 = -1): class', model.tenfold_class(time_reversal=TIME_REVERSAL))

# %%
# Parities at the TRIMs, and the phase diagram
# -------------------------------------------------
# The occupied pair has the parity of the lower of the two orbitals, set
# by the sign of :math:`d_z` at each TRIM: :math:`m` at :math:`\Gamma`,
# :math:`m - 4` at X and Y, :math:`m - 8` at M. So :math:`\nu = 1` for
# :math:`0 < m < 4` and :math:`4 < m < 8`, and 0 otherwise.

masses = np.array([-1., 1., 3., 5., 7., 9.])
nu_parity = [bhz(m).parity_z2(INVERSION, [0, 1]) for m in masses]
nu_flow = [bhz(m).z2_invariant([0, 1], nk=40, nk_perp=31) for m in masses]
expected = [int(0 < m < 4 or 4 < m < 8) for m in masses]
print('m:', masses, '\n  Fu-Kane parity:', nu_parity, '\n  Wannier flow:  ', nu_flow)
assert nu_parity == expected and nu_flow == expected

fig, ax = plt.subplots(figsize=(6, 3.4))
ax.plot(masses, nu_parity, 'ob', ms=9, label='Fu-Kane parity')
ax.plot(masses, nu_flow, 'xr', ms=9, mew=2, label='Wannier-centre flow')
ax.set_xlabel('$m$')
ax.set_ylabel(r'$\nu$')
ax.set_yticks([0, 1])
ax.legend()
ax.set_title('BHZ model: two routes to the same Z2 invariant')
fig.set_layout_engine('tight')
