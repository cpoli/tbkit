"""PythTB's half of the worked model of "Coming from PythTB" (not run in CI).

The Haldane model, built as PythTB 2.0 builds it (``pythtb.models.haldane``).
Run it with PythTB 2.0 installed; it prints the numbers that the tbkit
version (``docs/source/migration/pythtb_haldane.py``) asserts.
"""
import numpy as np
from pythtb import Lattice, Mesh, TBModel, WFArray

delta, t1, t2 = 0.2, -1., 0.15

lat = Lattice([[1, 0], [1 / 2, np.sqrt(3) / 2]],   # lattice vectors
              [[1 / 3, 1 / 3], [2 / 3, 2 / 3]],    # orbitals, reduced coordinates
              periodic_dirs=[0, 1])
model = TBModel(lattice=lat, spinful=False)
model.set_onsite([-delta, delta])
for R in ([0, 0], [-1, 0], [0, -1]):
    model.set_hop(t1, 0, 1, R)
for R in ([1, 0], [-1, 1], [0, -1]):
    model.set_hop(t2 * 1j, 0, 0, R)
for R in ([-1, 0], [1, -1], [0, 1]):
    model.set_hop(t2 * 1j, 1, 1, R)

# Bands at Gamma, K and M (k in reduced coordinates)
k_red = [[0, 0], [2 / 3, 1 / 3], [1 / 2, 0]]
print('E(Gamma, K, M) =', np.round(model.solve_ham(k_red), 13).tolist())

# Chern number of the lower band
print('C =', round(model.chern_number((0, 1), (60, 60), occ_idxs=[0]), 13))

# Berry phase of the lower band along k1, for five values of k2
mesh = Mesh(['k', 'k'])
mesh.build_grid((81, 5))
wfa = WFArray(model.lattice, mesh, spinful=False)
wfa.solve_model(model)
print('k2 =', mesh.get_k_points()[0, :, 1].tolist())
print('Berry phase =', np.round(wfa.berry_phase(0, state_idx=[0]), 13).tolist())
