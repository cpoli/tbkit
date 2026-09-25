"""
Large lattices: the k-d tree neighbour search used beyond System.dense_max
sites gives exactly the bonds of the dense all-pairs path.
"""
import contextlib
import io
import unittest
from unittest import mock

import numpy as np

import tbkit.lattices as lattices
from tbkit.graphene import GrapheneLattice, GrapheneSystem
from tbkit.system import System
from tests.test_three_d import cubic


def build(lat, list_hop, sparse):
    with mock.patch.object(System, 'dense_max', 0 if sparse else 10**9):
        sys = System(lat)
        sys.set_hopping(list_hop)
        sys.get_ham()
    return sys


class TestSparseNeighbours(unittest.TestCase):

    def test_same_bonds_2d(self):
        lat = lattices.honeycomb()
        lat.get_lattice(6, 5)
        lat.rotation(17.)
        hop = [{'n': 1, 't': 1.}, {'n': 2, 't': 0.1j}, {'n': 3, 't': 0.05}]
        dense, sparse = build(lat, hop, False), build(lat, hop, True)
        self.assertTrue(np.array_equal(dense.hop, sparse.hop))
        self.assertEqual((dense.ham != sparse.ham).nnz, 0)
        self.assertEqual(sparse.vec_hop.size, 0)
        self.assertTrue(np.allclose(sparse.dist_uni, dense.dist_uni[:4]))

    def test_same_bonds_3d(self):
        lat = cubic(3, 3, 3)
        hop = [{'n': 1, 't': 1.}, {'n': 2, 't': 0.2}]
        dense, sparse = build(lat, hop, False), build(lat, hop, True)
        self.assertTrue(np.array_equal(dense.hop, sparse.hop))

    def test_angle_and_tag_selection(self):
        lat = lattices.honeycomb()
        lat.get_lattice(4, 4)
        hop = [{'n': 1, 'ang': 90., 't': 2.}, {'n': 2, 'tag': 'aa', 't': 0.3}]
        dense, sparse = build(lat, hop, False), build(lat, hop, True)
        self.assertTrue(np.array_equal(dense.hop, sparse.hop))

    def test_print_distances(self):
        lat = lattices.square()
        lat.get_lattice(4, 4)
        outs = []
        for dense_max in (10**9, 0):
            with mock.patch.object(System, 'dense_max', dense_max):
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    System(lat).print_distances(2)
                outs.append(buf.getvalue())
        self.assertEqual(outs[0].split('Distances')[1], outs[1].split('Distances')[1])

    def test_whole_lattice_within_reach(self):
        # asking for more orders than exist: every distance is found
        lat = lattices.chain()
        lat.get_lattice(4)
        with mock.patch.object(System, 'dense_max', 0):
            sys = System(lat)
            sys.get_distances(10)
        self.assertTrue(np.allclose(sys.dist_uni, [0., 1., 2., 3.]))

    def test_graphene_strain(self):
        lat = GrapheneLattice()
        lat.circle(n=6)
        spectra = []
        for dense_max in (10**9, 0):
            with mock.patch.object(System, 'dense_max', dense_max):
                sys = GrapheneSystem(lat)
                sys.set_hop_linear_strain(t=1., beta=0.1)
                sys.get_ham()
                sys.get_eig()
                spectra.append(sys.en)
        self.assertTrue(np.allclose(*spectra))

    def test_large_lattice(self):
        # a 3D lattice of 8000 sites, past dense_max (6.4e7 pairs otherwise)
        lat = cubic(20, 20, 20)
        sys = System(lat)
        sys.set_hopping([{'n': 1, 't': 1.}])
        self.assertEqual(len(sys.hop), 3 * 19 * 400)
        self.assertEqual(sys.vec_hop.size, 0)
        sys.get_ham()
        sys.get_eig_sparse(n_eig=2, sigma=6.)
        self.assertAlmostEqual(sys.en[-1], 6 * np.cos(np.pi / 21), places=8)


class TestRemoveDangling(unittest.TestCase):

    def test_matches_all_pairs(self):
        # the k-d tree version against the original O(N^2) algorithm
        def reference(lat):
            while True:
                dx = lat.coor['x'] - lat.coor['x'][:, None]
                dy = lat.coor['y'] - lat.coor['y'][:, None]
                dis = np.sqrt(dx**2 + dy**2)
                ind = np.argwhere(np.isclose(dis, np.unique(dis)[1]))
                dang = [i for i in range(lat.sites) if (ind[:, 0] == i).sum() == 1]
                lat.coor = np.delete(lat.coor, dang, axis=0)
                lat.sites -= len(dang)
                if not dang:
                    return lat
        for n in (4, 5, 7):
            a, b = GrapheneLattice(), GrapheneLattice()
            for lat in (a, b):
                lat.get_lattice(n1=2*n, n2=2*n)
                lat.center()
                lat.ellipse_in(rx=0.866*(n+1), ry=0.866*(n+1), x0=0., y0=0.)
            a.remove_dangling()
            reference(b)
            self.assertTrue(np.array_equal(a.coor, b.coor))


if __name__ == '__main__':
    unittest.main()
