r"""
Symmetrizing a Wannier90 Model
==============================

A tight-binding model read from Wannier90 inherits the numerical noise of
the first-principles calculation and of the Wannierization: its hoppings
break the crystal symmetry by a little, so that symmetry-protected
degeneracies are split and every symmetry check fails. The cure, as in
TBmodels, is the group average of the hoppings,

.. math::

    H_{\mathrm{sym}} = \frac{1}{|G|}\sum_{g\in G} g\,H\,g^{-1}\, ,

which leaves a symmetric model unchanged and projects a noisy one onto
the closest symmetric one (:func:`~tbkit.symmetry.symmetrize`).

Here a graphene :math:`p_z` model with second neighbours is written as a
``_hr.dat`` file with noise of :math:`2\cdot10^{-4}` on every hopping,
read back by :func:`~tbkit.io.read_wannier90`, and symmetrized with the
operations found, at a loose tolerance, by
:func:`~tbkit.symmetry.find_symmetries`.
"""
import os
import tempfile

import numpy as np
import matplotlib.pyplot as plt

from tbkit.io import read_wannier90
from tbkit.kspace import high_symmetry_path
from tbkit.symmetry import find_symmetries, irreps, symmetrize

T, T2, NN = -2.8, 0.1, 1.42  # eV, eV, Angstrom
A = NN * np.sqrt(3)
PRIM = [(A, 0.), (A / 2, 1.5 * NN)]
POSITIONS = [[0., 0.], [A / 2, 0.5 * NN]]


def hr_file(path, noise, seed=0):
    '''Graphene p_z (t, t2) as a Wannier90 seedname_hr.dat, with Hermitian noise.'''
    rng = np.random.default_rng(seed)
    hop = {}
    for R in [(0, 0), (-1, 0), (0, -1)]:
        hop.setdefault(R, np.zeros((2, 2), complex))[0, 1] += T
        hop.setdefault((-R[0], -R[1]), np.zeros((2, 2), complex))[1, 0] += T
    for R in [(1, 0), (0, 1), (1, -1), (-1, 0), (0, -1), (-1, 1)]:
        hop.setdefault(R, np.zeros((2, 2), complex))[[0, 1], [0, 1]] += T2
    for R in sorted(hop):
        m = (-R[0], -R[1])
        if R < m:
            hop[R] = hop[R] + noise * (rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2)))
            hop[m] = hop[R].conj().T
    hop[(0, 0)] = hop[(0, 0)] + noise * np.diag(rng.normal(size=2))
    lines = ['written by plot_symmetrizing_a_wannier90_model.py', '2', str(len(hop)), ' '.join(['1'] * len(hop))]
    for R, h in hop.items():
        for n in range(2):
            for m in range(2):
                lines.append('{} {} 0 {} {} {:.8f} {:.8f}'.format(R[0], R[1], m + 1, n + 1, h[m, n].real,
                                                                                h[m, n].imag))
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')


with tempfile.TemporaryDirectory() as tmp:
    path = os.path.join(tmp, 'graphene_hr.dat')
    hr_file(path, noise=2e-4)
    noisy = read_wannier90(path, prim_vec=PRIM, positions=POSITIONS, dim=2)

# %%
# The operations, found at a loose tolerance
# ------------------------------------------
# C6v and time reversal (24 operations) leave the noisy model invariant
# only to the noise level.

ops = find_symmetries(noisy, tol=1e-2)
errors = [noisy.symmetry_error(op) for op in ops]
print(len(ops), 'operations; largest symmetry error {:.1e}'.format(max(errors)))
assert len(ops) == 24 and max(errors) > 1e-5

# %%
# Symmetrize
# ----------

sym = symmetrize(noisy, ops)
errors_sym = [sym.symmetry_error(op) for op in ops]
print('after symmetrize: largest symmetry error {:.1e}'.format(max(errors_sym)))
assert max(errors_sym) < 1e-13
assert len(find_symmetries(sym, tol=1e-12)) == 24

points, labels = high_symmetry_path(sym.lat)
K = points[2]
split = [float(np.ptp(m.get_bands([K]))) for m in (noisy, sym)]
print('splitting of the Dirac point: {:.1e} eV -> {:.1e} eV'.format(*split))
assert split[0] > 1e-5 and split[1] < 1e-13
print('irrep at K after symmetrize:', irreps(sym, K).labels)
assert irreps(sym, K).labels == ['E']
# a symmetric model is unchanged
assert np.abs(symmetrize(sym, ops).get_ham(K) - sym.get_ham(K)).max() < 1e-13

# %%
# The Dirac point, before and after
# ---------------------------------

line = [K + s * (points[1] - K) for s in np.linspace(-0.02, 0.02, 81)]
fig, ax = plt.subplots(figsize=(6, 3.6))
for model, style, name in ((noisy, 'r-', 'read from the file'), (sym, 'b--', 'symmetrized')):
    en = model.get_bands(line)
    ax.plot(np.linspace(-0.02, 0.02, 81), en, style, lw=1.5)
    ax.plot([], [], style, label=name)
ax.set_xlabel(r'distance from K along K-M (fraction of $|KM|$)')
ax.set_ylabel('$E$ (eV)')
ax.legend()
ax.set_title('the gap that noise opens at K')
fig.set_layout_engine('tight')
