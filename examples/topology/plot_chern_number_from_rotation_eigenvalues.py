r"""
Chern Numbers from Rotation Eigenvalues
=======================================

C. Fang, M. J. Gilbert and B. A. Bernevig (2012) showed that in a crystal
with an n-fold rotation :math:`C_n` the Chern number of the occupied
bands is fixed modulo :math:`n` by their rotation eigenvalues at a few
points, without any integral of the Berry curvature:

.. math::

    e^{2\pi i C/n} = \prod_{\mathbf{k}_*}\ \prod_{m\in\mathrm{occ}}
    \lambda_m\big(\mathbf{k}_*, C_n^{s}\big)\, ,

over one point :math:`\mathbf{k}_*` of each orbit of rotation-invariant
momenta, with the power :math:`C_n^s` that keeps it: for :math:`C_4`
the :math:`C_4` eigenvalues at :math:`\Gamma` and M and the :math:`C_2`
eigenvalue at X, :math:`i^C = \xi(\Gamma)\,\xi(\mathrm{M})\,\zeta(\mathrm{X})`;
for :math:`C_3`, the eigenvalues at :math:`\Gamma`, K and K'.

:func:`~tbkit.symmetry.rotation_indicator` computes it. It is compared
with :meth:`~tbkit.kspace.KSpace.chern_number` (an integral over a
:math:`30\times30` mesh) across the phase diagrams of the Qi-Wu-Zhang
model (:math:`C_4`, two orbitals on one site) and of the Haldane model
(:math:`C_3`, and :math:`C_6` without a sublattice mass).
"""
import numpy as np
import matplotlib.pyplot as plt

from tbkit.symmetry import Operation, find_symmetries, rotation, rotation_indicator
from tbkit import models


# %%
# The rotation of the Qi-Wu-Zhang model
# -------------------------------------
# :math:`C_4` sends :math:`(\sigma_x, \sigma_y)` to :math:`(\sigma_y, -\sigma_x)`:
# it acts on the two orbitals as ``diag(1, i)``, with :math:`C_4^4 = 1`.

C4 = Operation(rotation(np.pi / 2), orbitals=np.diag([1., 1j]))
assert models.qi_wu_zhang(-1.).symmetry_error(C4) < 1e-12

masses = np.linspace(-2.9, 2.9, 30)  # gapped: avoids m = 0, +-2
c_qwz = [models.qi_wu_zhang(m).chern_number(0) for m in masses]
c4_qwz = [rotation_indicator(models.qi_wu_zhang(m), C4, [0]) for m in masses]
c2_qwz = [rotation_indicator(models.qi_wu_zhang(m), C4 ** 2, [0]) for m in masses]
for c, c4, c2 in zip(c_qwz, c4_qwz, c2_qwz):
    assert round(c) % 4 == c4 and round(c) % 2 == c2
print('QWZ  C:', [int(round(c)) for c in c_qwz])
print('     C mod 4 from C4 eigenvalues:', c4_qwz)

# %%
# Haldane model
# -------------
# :math:`C_3` about the centre of a hexagon, found by
# :func:`~tbkit.symmetry.find_symmetries`; :math:`C_6` as well without a mass.

phases = [(0.2, 0.), (-0.2, 0.), (0.2, 0.5), (0.2, 1.5), (-0.1, 0.2), (0.1, 1.)]
for t2, mass in phases:
    model = models.haldane(t2=t2, M=mass)
    ops = {op.name: op for op in find_symmetries(model, antiunitary=False)}
    c = int(round(model.chern_number(0)))
    c3 = rotation_indicator(model, ops['C3'], [0])
    c6 = rotation_indicator(model, ops['C6'], [0]) if 'C6' in ops else None
    print('Haldane t2 = {:5.2f}, M = {:4.2f}: C = {:2d}, C mod 3 = {}, C mod 6 = {}'.format(t2, mass, c, c3, c6))
    assert c3 == c % 3 and (c6 is None or c6 == c % 6)

# %%
# The same Chern numbers, two ways
# --------------------------------

fig, ax = plt.subplots(figsize=(6, 3.4))
ax.plot(masses, np.round(c_qwz).astype(int) % 4, 'ob', ms=8, label=r'$C\ \mathrm{mod}\ 4$ (Berry curvature)')
ax.plot(masses, c4_qwz, 'xr', ms=9, mew=2, label=r'from the $C_4$ eigenvalues')
ax.set_xlabel('$m$')
ax.set_ylabel(r'$C\ \mathrm{mod}\ 4$')
ax.set_yticks(range(4))
ax.legend(loc='center right')
ax.set_title('Qi-Wu-Zhang model')
fig.set_layout_engine('tight')
