"""The usage guide of the paper: one lattice, solved in real space and in reciprocal space."""
import numpy as np

import tbkit.lattices as lattices
from tbkit.kspace import KSpace, high_symmetry_path
from tbkit.system import System

# 1. Real space: a 20 x 20 square flake with nearest-neighbour hopping t.
#    Its levels are 2t[cos(pi m / 21) + cos(pi l / 21)], m, l = 1..20.
t, n = 1., 20
lat = lattices.square()
lat.get_lattice(n1=n, n2=n)
flake = System(lat)
flake.set_onsite({'a': 0.})
flake.set_hopping([{'n': 1, 't': t}])
flake.get_ham()
flake.get_eig()
c = np.cos(np.pi * np.arange(1, n + 1) / (n + 1))
assert np.allclose(np.sort(flake.en.real), np.sort((2 * t * (c[:, None] + c[None, :])).ravel()))

# 2. Reciprocal space: graphene, from the three bonds leaving the 'a' site
#    of the cell (the reverse bonds are added as Hermitian conjugates).
gra = KSpace(lattices.honeycomb())
gra.set_hopping([{'i': 0, 'j': 1, 'R': R, 't': t} for R in [(0, 0), (-1, 0), (0, -1)]])
points, labels = high_symmetry_path(gra.lat)
dist, bands = gra.k_path(points, nk=100)
gamma, K = (points[labels.index(p)] for p in ('$\\Gamma$', 'K'))
assert np.allclose(np.linalg.eigvalsh(gra.get_ham(gamma)), [-3 * t, 3 * t])
assert np.allclose(np.linalg.eigvalsh(gra.get_ham(K)), 0., atol=1e-12)
print(f'flake: {flake.lat.sites} levels; graphene: Dirac point at K, bandwidth {np.ptp(bands):.1f} t')
