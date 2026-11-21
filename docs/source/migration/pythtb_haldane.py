"""The worked model of "Coming from PythTB", ported line by line to tbkit.

The Haldane model, built as PythTB 2.0 builds it (``pythtb.models.haldane``).
The expected numbers are those PythTB 2.0 prints for the same model
(``original/pythtb_haldane.py``).
"""
import numpy as np

from tbkit.kspace import KSpace, reciprocal_vectors
from tbkit.lattice import Lattice

delta, t1, t2 = 0.2, -1., 0.15

a1, a2 = np.array([1., 0.]), np.array([1 / 2, np.sqrt(3) / 2])
lat = Lattice(unit_cell=[{'tag': 'a', 'r0': tuple((a1 + a2) / 3)},       # Cartesian positions
                         {'tag': 'b', 'r0': tuple(2 * (a1 + a2) / 3)}],
              prim_vec=[tuple(a1), tuple(a2)])
model = KSpace(lat)
model.set_onsite({'a': -delta, 'b': delta})
model.set_hopping([{'i': 0, 'j': 1, 'R': R, 't': t1} for R in [(0, 0), (-1, 0), (0, -1)]]
                  + [{'i': 0, 'j': 0, 'R': R, 't': t2 * 1j} for R in [(1, 0), (-1, 1), (0, -1)]]
                  + [{'i': 1, 'j': 1, 'R': R, 't': t2 * 1j} for R in [(-1, 0), (1, -1), (0, 1)]])

# Bands at Gamma, K and M (tbkit takes k in Cartesian coordinates)
B = np.array(reciprocal_vectors(lat.prim_vec))       # rows b1, b2
k_red = np.array([[0, 0], [2 / 3, 1 / 3], [1 / 2, 0]])
energies = np.array([np.linalg.eigvalsh(model.get_ham(k)) for k in k_red @ B])

# Chern number of the lower band
C = model.chern_number(0, nk=60)

# Berry phase of the lower band along k1, for five values of k2
k2 = np.arange(5) / 5
phases = np.array([model.berry_phase(0, nk=81, direction=0, k_perp=q) for q in k2])

# PythTB 2.0: model.solve_ham(k_red), model.chern_number(...), wfa.berry_phase(0, ...)
assert np.allclose(energies, [[-3.0066592756746, 3.0066592756746], [-0.579422863406, 0.579422863406],
                              [-1.0198039027186, 1.0198039027186]], atol=1e-12)
assert round(C) == -1 and abs(C + 1) < 1e-12
pythtb_phases = [2.9774265573673, 3.6265248471583, 5.4395118710551, 7.5722415357587,
                 8.6817552893025]
assert np.allclose(np.exp(1j * phases), np.exp(1j * np.array(pythtb_phases)), atol=1e-12)
print('E(Gamma, K, M) =', np.round(energies, 13).tolist())
print('C =', round(C, 13))
print('Berry phase =', np.round(np.unwrap(phases), 13).tolist())
