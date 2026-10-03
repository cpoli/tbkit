"""The worked example of the paper: one model, from its bands to its transport."""
import numpy as np

import tbkit.lattices as lattices
from tbkit.bridges import finite_system
from tbkit.kspace import KSpace, high_symmetry_path, ribbon
from tbkit.transport import Transport

# 1. The Haldane model: graphene, plus a complex second-neighbour hopping
#    (zero net flux) and a staggered sublattice potential M.
t1, t2, M = 1., 0.2, 0.3
hops = [{'i': 0, 'j': 1, 'R': R, 't': t1} for R in [(0, 0), (-1, 0), (0, -1)]]
for R in [(0, 1), (-1, 0), (1, -1)]:
    hops += [{'i': 0, 'j': 0, 'R': R, 't': 1j * t2}, {'i': 1, 'j': 1, 'R': R, 't': -1j * t2}]
lat = lattices.honeycomb()
haldane = KSpace(lat)
haldane.set_hopping(hops)
haldane.set_onsite({'a': M, 'b': -M})

# 2. Bands along Gamma-M-K-Gamma, and the gaps at the two valleys K and K' = -K:
#    2|M -+ 3 sqrt(3) t2|, the smaller one closing at the transition.
points, labels = high_symmetry_path(lat)
dist, bands = haldane.k_path(points, nk=200)
K = points[labels.index('K')]
gaps = sorted(np.ptp(np.linalg.eigvalsh(haldane.get_ham(k))) for k in (K, -K))
assert np.allclose(gaps, sorted(2 * abs(M + s * 3 * np.sqrt(3) * t2) for s in (1, -1)))
gap = gaps[0]

# 3. Topology and response: the Chern number of the lower band, and the
#    Hall conductivity (e^2/h) with the Fermi level in the gap.
chern = haldane.chern_number(0, nk=60)
sigma = haldane.hall_conductivity(0., nk=60)
assert abs(abs(chern) - 1) < 1e-9 and abs(sigma - chern) < 1e-6

# 4. Edge states: a ribbon, 40 cells wide, has states inside the bulk gap.
strip = ribbon(lat, hops, width=40, direction=1, onsite={'a': M, 'b': -M})
_, edge = strip.k_path([(-np.pi / np.sqrt(3),), (np.pi / np.sqrt(3),)], nk=301)
assert np.sum(np.abs(edge) < gap / 4) > 0

# 5. Transport: a finite device cut from the same model, with ribbon leads.
#    In the gap, one chiral edge channel carries T = 1 from left to right.
device = finite_system(haldane, (30, 40))
device.get_ham()
tr = Transport(device.ham)
tr.attach_lead(device, strip, -1)
tr.attach_lead(device, strip, 1)
transmission = tr.transmission([0.])[0]
assert abs(transmission - 1) < 1e-6
print(f'gap {gap:.3f}, C = {chern:+.0f}, sigma_xy = {sigma:+.6f} e^2/h, T(0) = {transmission:.6f}')
