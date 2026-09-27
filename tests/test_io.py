'''
tbkit.io: save and load models; round trips reproduce the Hamiltonians exactly.
'''
import os
import pathlib
import tempfile
import unittest

import numpy as np

from tbkit import lattices
from tbkit.bridges import finite_system
from tbkit.graphene import GrapheneSystem, GrapheneLattice
from tbkit.io import save_model, load_model, FORMAT, VERSION
from tbkit.kspace import KSpace, PAULI
from tbkit.lattice import Lattice
from tbkit.orbital import OrbitalSystem
from tbkit.system import System


KS2 = np.random.default_rng(7).uniform(-4, 4, (5, 2))


class TestIO(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()

    def roundtrip(self, model, name='model'):
        path = save_model(model, os.path.join(self.dir, name))
        self.assertTrue(path.endswith('.npz'))
        return load_model(path)

    def test_lattice(self):
        lat = lattices.kagome()
        lat.get_lattice(4, 3)
        lat.remove_sites([0, 5])
        back = self.roundtrip(lat)
        self.assertIs(type(back), Lattice)
        np.testing.assert_array_equal(back.coor, lat.coor)
        self.assertEqual(back.unit_cell, lat.unit_cell)
        self.assertEqual(back.prim_vec, [tuple(map(float, a)) for a in lat.prim_vec])
        self.assertEqual((back.n1, back.n2, back.sites), (4, 3, lat.sites))
        # a bare lattice (no get_lattice) too
        back = self.roundtrip(lattices.square(), 'bare')
        self.assertEqual(back.sites, 0)
        self.assertEqual(back.coor.dtype, np.dtype(back.dtype))

    def test_system_exact(self):
        lat = lattices.honeycomb()
        lat.get_lattice(6, 5)
        sys = System(lat)
        sys.set_hopping([{'n': 1, 't': -1.}, {'n': 2, 't': 0.1j}])
        sys.set_onsite({'a': 0.3, 'b': -0.3})
        sys.set_hopping_dis(0.2)
        sys.set_magnetic_field(0.01)
        sys.get_ham()
        back = self.roundtrip(sys)
        back.get_ham()
        np.testing.assert_array_equal(back.ham.toarray(), sys.ham.toarray())
        np.testing.assert_array_equal(back.hop, sys.hop)
        sys.get_eig()
        back.get_eig()
        np.testing.assert_array_equal(back.en, sys.en)

    def test_non_hermitian_3d_system(self):
        lat = Lattice([{'tag': 'a', 'r0': (0., 0., 0.)}], [(1., 0., 0.), (0., 1., 0.), (0., 0., 2.)])
        lat.get_lattice(3, 3, 2)
        sys = System(lat)
        sys.set_hopping([{'n': 1, 't': 1.2}])
        sys.set_hopping([{'n': 1, 't': 0.4}], upper_part=False)
        sys.get_ham()
        back = self.roundtrip(sys)
        back.get_ham()
        np.testing.assert_array_equal(back.ham.toarray(), sys.ham.toarray())
        self.assertEqual(back.lat.space_dim, 3)

    def test_subclass_saved_as_base(self):
        lat = GrapheneLattice()
        lat.triangle_zigzag(n=4)
        sys = GrapheneSystem(lat)
        sys.set_hopping([{'n': 1, 't': -1.}])
        sys.get_ham()
        back = self.roundtrip(sys)
        self.assertIs(type(back), System)
        back.get_ham()
        np.testing.assert_array_equal(back.ham.toarray(), sys.ham.toarray())

    def test_kspace_exact(self):
        gra = KSpace(lattices.honeycomb())
        gra.set_hopping([{'n': 1, 't': -2.97}, {'n': 2, 't': -0.073}, {'n': 3, 't': -0.33}])
        gra.set_overlap([{'n': 1, 't': 0.073}, {'n': 2, 't': 0.018}])
        gra.set_onsite({'a': -0.28, 'b': -0.28})
        back = self.roundtrip(gra, 'gra.npz')
        self.assertIs(type(back), KSpace)
        for k in KS2:
            np.testing.assert_array_equal(back.get_ham(k), gra.get_ham(k))
            np.testing.assert_array_equal(back.get_overlap(k), gra.get_overlap(k))
            np.testing.assert_array_equal(back.get_bands(k), gra.get_bands(k))

    def test_kspace_spin_nonreciprocal(self):
        ks = KSpace(lattices.square(), spin=True)
        ks.set_onsite({'a': 0.3 * PAULI['x'] + 0.1 * PAULI['z']})
        ks.set_hopping([{'n': 1, 't': PAULI['0'] + 0.2j * PAULI['y']}])
        ks.set_hopping([{'i': 0, 'j': 0, 'R': (1, 1), 't': 0.05}], hermitian=False)
        back = self.roundtrip(ks, pathlib.Path(self.dir) / 'spin')
        self.assertTrue(back.spin)
        self.assertEqual(back.is_hermitian(), ks.is_hermitian())
        self.assertFalse(back.is_hermitian())
        for k in KS2:
            np.testing.assert_array_equal(back.get_ham(k), ks.get_ham(k))

    def test_kspace_3d_and_finite(self):
        cub = KSpace(Lattice([{'tag': 'a', 'r0': (0., 0., 0.)}, {'tag': 'b', 'r0': (.5, .5, .5)}],
                                        [(1., 0., 0.), (0., 1., 0.), (0., 0., 1.)]))
        cub.set_hopping([{'n': 1, 't': 1.}, {'n': 2, 't': 0.3j}])
        back = self.roundtrip(cub)
        np.testing.assert_array_equal(back.finite_ham((2, 3, 2), periodic=True),
                                                    cub.finite_ham((2, 3, 2), periodic=True))
        # a System built by the bridges saves too
        sys = finite_system(KSpace(lattices.chain()), 3)
        back = self.roundtrip(sys, 'chain')
        self.assertEqual(back.lat.sites, 3)

    def test_errors(self):
        path = os.path.join(self.dir, 'x.npz')
        self.assertRaises(TypeError, save_model, 'model', path)
        lat = lattices.square()
        lat.get_lattice(2, 2)
        self.assertRaises(TypeError, save_model, OrbitalSystem(lat), path)
        self.assertRaises(TypeError, save_model, lat, 1)
        self.assertRaises(TypeError, load_model, 1)
        np.savez(path, a=np.zeros(2))
        self.assertRaises(ValueError, load_model, path)
        np.savez(path, format=np.array(FORMAT), version=np.array(VERSION + 1))
        self.assertRaises(ValueError, load_model, path)


if __name__ == '__main__':
    unittest.main()
