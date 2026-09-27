'''
Bridges between System (real space) and KSpace (Bloch Hamiltonians).
'''
import unittest

import numpy as np

from tbkit import lattices
from tbkit.bridges import cell_orbitals, kspace_from_system, finite_model, finite_system
from tbkit.kspace import KSpace, PAULI
from tbkit.lattice import Lattice
from tbkit.orbital import OrbitalSystem
from tbkit.system import System


KS2 = np.random.default_rng(5).uniform(-4, 4, (6, 2))


def haldane(t1=1., t2=0.2, phi=np.pi / 2, m=0.1):
    hal = KSpace(lattices.honeycomb())
    hal.set_hopping([{'i': 0, 'j': 1, 'R': R, 't': t1} for R in [(0, 0), (-1, 0), (0, -1)]])
    for R in [(1, 0), (-1, 1), (0, -1)]:  # a1, a2 - a1, -a2: 120 degrees apart
        hal.set_hopping([{'i': 0, 'j': 0, 'R': R, 't': t2 * np.exp(1j * phi)},
                                {'i': 1, 'j': 1, 'R': R, 't': t2 * np.exp(-1j * phi)}])
    hal.set_onsite({'a': m, 'b': -m})
    return hal


def cubic():
    return Lattice([{'tag': 'a', 'r0': (0., 0., 0.)}], [(1., 0., 0.), (0., 1., 0.), (0., 0., 1.)])


class TestTorusMatchesMesh(unittest.TestCase):
    '''For the same model, the System spectrum on an N1 x N2 torus is the
    set of KSpace bands on the N1 x N2 mesh.'''

    def check(self, ks, n_cells):
        sys = finite_system(ks, n_cells, periodic=True)
        sys.get_ham()
        np.testing.assert_allclose(sys.ham.toarray(), ks.finite_ham(n_cells, periodic=True), atol=1e-14)
        sys.get_eig()
        mesh = ks.mesh_bands(n_cells).ravel()
        if ks.is_hermitian():
            np.testing.assert_allclose(np.sort(sys.en), np.sort(mesh), atol=1e-11)
        else:
            # every mesh eigenvalue is a torus eigenvalue (as multisets)
            dist = np.abs(sys.en[:, None] - mesh[None, :])
            self.assertLess(dist.min(axis=0).max(), 1e-9)
            self.assertLess(dist.min(axis=1).max(), 1e-9)
        return sys

    def test_haldane_large_torus(self):
        hal = haldane()
        sys = self.check(hal, (24, 21))
        self.assertEqual(sys.lat.sites, 2 * 24 * 21)
        # the torus spectrum has the Haldane gap 2|m - 3 sqrt(3) t2| at K
        self.assertTrue(np.all(sys.hop['ang'] >= 0))

    def test_neighbour_order_graphene(self):
        gra = KSpace(lattices.honeycomb())
        gra.set_hopping([{'n': 1, 't': -1.}, {'n': 2, 't': 0.1}, {'n': 3, 't': -0.05}])
        sys = self.check(gra, (15, 15))
        self.assertEqual(sorted(set(sys.hop['n'].tolist())), [1, 2, 3])
        self.assertEqual(set(sys.hop['tag'][sys.hop['n'] == 2].tolist()), {'aa', 'bb'})

    def test_cubic_3d(self):
        cub = KSpace(cubic())
        cub.set_hopping([{'n': 1, 't': 1.}, {'n': 2, 't': 0.2}])
        self.check(cub, (5, 6, 7))

    def test_hatano_nelson(self):
        '''Periodic: E = tR e^{ik} + tL e^{-ik}; open: real, the skin effect
        spectrum 2 sqrt(tR tL) cos(pi m / (N + 1)).'''
        tr, tl, n = 1.2, 0.5, 12
        hn = KSpace(lattices.chain())
        hn.set_hopping([{'n': 1, 'ang': 0., 't': tr}, {'n': 1, 'ang': -180., 't': tl}], hermitian=False)
        self.check(hn, (n,))
        sys = finite_system(hn, n)
        self.assertTrue(np.any(sys.hop['ang'] < 0) and np.any(sys.hop['ang'] >= 0))
        sys.get_ham()
        sys.get_eig()
        exact = 2 * np.sqrt(tr * tl) * np.cos(np.pi * np.arange(1, n + 1) / (n + 1))
        np.testing.assert_allclose(np.sort(sys.en.real), np.sort(exact), atol=1e-9)
        np.testing.assert_allclose(sys.en.imag, 0., atol=1e-9)

    def test_self_wrapping_bond_goes_onsite(self):
        '''A 1-cell torus: the hopping to the next cell wraps onto the site
        itself, E = 2t cos(0) + onsite.'''
        ch = KSpace(lattices.chain())
        ch.set_hopping([{'n': 1, 't': 1.}])
        ch.set_onsite({'a': 0.5})
        sys = finite_system(ch, 1, periodic=True)
        self.assertEqual(len(sys.hop), 0)
        np.testing.assert_allclose(sys.onsite, [2.5])

    def test_coinciding_orbitals(self):
        '''Two orbitals on the same site (a zero-length bond).'''
        lat = Lattice([{'tag': 'a', 'r0': (0., 0.)}, {'tag': 'b', 'r0': (0., 0.)}], [(1., 0.), (0., 1.)])
        ks = KSpace(lat)
        ks.set_hopping([{'i': 0, 'j': 1, 'R': (0, 0), 't': 0.3j},
                              {'i': 0, 'j': 0, 'R': (1, 0), 't': 1.}])
        sys = self.check(ks, (4, 5))
        self.assertEqual(int(np.sum(sys.hop['n'] == 0)), 20)


class TestOpenRoundTrip(unittest.TestCase):

    def test_system_to_kspace_neighbour_order(self):
        '''A System built the real-space way gives the KSpace built by
        neighbour order from the same list.'''
        list_hop = [{'n': 1, 't': -1.}, {'n': 2, 't': 0.1j}, {'n': 3, 'tag': 'ab', 't': 0.05}]
        lat = lattices.honeycomb()
        lat.get_lattice(7, 7)
        sys = System(lat)
        sys.set_hopping(list_hop)
        sys.set_onsite({'a': 0.3, 'b': -0.3})
        ks = kspace_from_system(sys)
        ref = KSpace(lattices.honeycomb())
        ref.set_hopping(list_hop)
        ref.set_onsite({'a': 0.3, 'b': -0.3})
        self.assertTrue(ks.is_hermitian())
        for k in KS2:
            np.testing.assert_allclose(ks.get_ham(k), ref.get_ham(k), atol=1e-14)

    def test_kspace_to_system_to_kspace(self):
        '''KSpace -> open System -> KSpace is the identity; the open System
        is that of get_lattice + set_hopping, up to the site order.'''
        hal = haldane()
        sys = finite_system(hal, (6, 6))
        back = kspace_from_system(sys)
        for k in KS2:
            np.testing.assert_allclose(back.get_ham(k), hal.get_ham(k), atol=1e-14)
        # same sites as get_lattice(6, 6)
        lat = lattices.honeycomb()
        lat.get_lattice(6, 6)
        a = np.lexsort((sys.lat.coor['x'].round(6), sys.lat.coor['y'].round(6)))
        b = np.lexsort((lat.coor['x'].round(6), lat.coor['y'].round(6)))
        np.testing.assert_allclose(sys.lat.coor['x'][a], lat.coor['x'][b], atol=1e-12)
        np.testing.assert_array_equal(sys.lat.coor['tag'][a], lat.coor['tag'][b])

    def test_graphene_flake_equals_system(self):
        '''Nearest neighbours: finite_system reproduces System.set_hopping
        on the same flake, matrix element by matrix element.'''
        gra = KSpace(lattices.honeycomb())
        gra.set_hopping([{'n': 1, 't': -1.}])
        sys = finite_system(gra, (5, 4))
        sys.get_ham()
        lat = lattices.honeycomb()
        lat.get_lattice(5, 4)
        ref = System(lat)
        ref.set_hopping([{'n': 1, 't': -1.}])
        ref.get_ham()
        key = lambda c: np.lexsort((c['x'].round(6), c['y'].round(6)))
        p, q = key(sys.lat.coor), key(lat.coor)
        h1 = sys.ham.toarray()[np.ix_(p, p)]
        h2 = ref.ham.toarray()[np.ix_(q, q)]
        np.testing.assert_allclose(h1, h2, atol=1e-14)

    def test_non_hermitian_system(self):
        '''A non-reciprocal System (upper and lower parts) gives a
        non-reciprocal KSpace.'''
        lat = lattices.chain()
        lat.get_lattice(8)
        sys = System(lat)
        sys.set_hopping([{'n': 1, 't': 1.2}])
        sys.set_hopping([{'n': 1, 't': 0.5}], upper_part=False)
        ks = kspace_from_system(sys)
        self.assertFalse(ks.is_hermitian())
        k = 0.7
        self.assertAlmostEqual(complex(ks.get_ham((k,))[0, 0]),
                                      1.2 * np.exp(1j * k) + 0.5 * np.exp(-1j * k), places=12)

    def test_torus_system_to_kspace(self):
        hal = haldane()
        sys = finite_system(hal, (5, 4), periodic=True)
        back = kspace_from_system(sys, periodic=True)
        for k in KS2:
            np.testing.assert_allclose(back.get_ham(k), hal.get_ham(k), atol=1e-14)

    def test_3d_system(self):
        lat = cubic()
        lat.get_lattice(4, 4, 4)
        sys = System(lat)
        sys.set_hopping([{'n': 1, 't': 1.}])
        ks = kspace_from_system(sys)
        k = np.array([0.3, -1., 2.])
        self.assertAlmostEqual(ks.get_ham(k)[0, 0].real, 2 * np.cos(k).sum(), places=12)

    def test_cell_orbitals(self):
        lat = lattices.honeycomb()
        lat.get_lattice(3, 2)
        cells, orb = cell_orbitals(lat)
        a = np.array(lat.prim_vec)
        tau = np.array([d['r0'] for d in lat.unit_cell])
        pos = np.stack([lat.coor['x'], lat.coor['y']], axis=1)
        np.testing.assert_allclose(cells @ a + tau[orb], pos, atol=1e-12)
        self.assertEqual(sorted(map(tuple, cells[orb == 0].tolist())),
                               [(i, j) for i in range(3) for j in range(2)])


class TestFiniteModel(unittest.TestCase):

    def test_spinful(self):
        lam = 0.2
        ks = KSpace(lattices.square(), spin=True)
        ks.set_hopping([{'n': 1, 't': 1.}, {'n': 1, 'ang': 0., 't': PAULI['0'] + 1j * lam * PAULI['y']}])
        ham, pos, tags = finite_model(ks, 3, periodic=True)
        np.testing.assert_allclose(ham.toarray(), ks.finite_ham(3, periodic=True), atol=0)
        self.assertEqual(pos.shape, (18, 2))
        np.testing.assert_allclose(pos[0], pos[1])
        np.testing.assert_allclose(pos[2], [1., 0.])
        np.testing.assert_allclose(pos[6], [0., 1.])
        self.assertEqual(list(tags[:2]), ['a', 'a'])


class TestErrors(unittest.TestCase):

    def test_disorder_breaks_translation(self):
        lat = lattices.square()
        lat.get_lattice(4, 4)
        sys = System(lat)
        sys.set_hopping([{'n': 1, 't': 1.}])
        sys.set_hopping_def({(0, 1): 2.})
        self.assertRaises(ValueError, kspace_from_system, sys)

    def test_rotated_lattice(self):
        lat = lattices.square()
        lat.get_lattice(4, 4)
        lat.rotation(10.)
        sys = System(lat)
        sys.set_hopping([{'n': 1, 't': 1.}])
        self.assertRaises(ValueError, kspace_from_system, sys)

    def test_arguments(self):
        lat = lattices.square()
        lat.get_lattice(4, 4)
        sys = System(lat)
        sys.set_hopping([{'n': 1, 't': 1.}])
        self.assertRaises(TypeError, kspace_from_system, 'sys')
        self.assertRaises(TypeError, kspace_from_system, OrbitalSystem(lat))
        self.assertRaises(TypeError, kspace_from_system, sys, periodic=1)
        self.assertRaises(ValueError, kspace_from_system, sys, tol=0.)
        lat2 = lattices.square()
        lat2.get_lattice(2, 4)
        sys2 = System(lat2)
        sys2.set_hopping([{'n': 1, 't': 1.}])
        self.assertRaises(ValueError, kspace_from_system, sys2, periodic=True)
        ks = KSpace(lattices.square(), spin=True)
        self.assertRaises(TypeError, finite_model, 'ks', 3)
        self.assertRaises(ValueError, finite_model, ks, 0)
        self.assertRaises(TypeError, finite_model, ks, 3, periodic=1)
        self.assertRaises(ValueError, finite_system, ks, 3)
        self.assertRaises(TypeError, cell_orbitals, 'lat')
        self.assertRaises(RuntimeError, cell_orbitals, lattices.square())


if __name__ == '__main__':
    unittest.main()
