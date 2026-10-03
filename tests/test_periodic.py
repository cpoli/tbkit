"""
Periodic boundaries in real space: System(lat, periodic=...) on tori and
cylinders, checked against the KSpace bands on the matching k-mesh.
"""
import os
import tempfile
import unittest

import matplotlib.pyplot as plt
import numpy as np

import tbkit.lattices as lattices
from tbkit.bridges import finite_system, kspace_from_system
from tbkit.io import load_model, save_model
from tbkit.kspace import KSpace, ribbon
from tbkit.lattice import Lattice
from tbkit.orbital import OrbitalSystem
from tbkit.plot import Plot
from tbkit.system import System
from tests.test_topology import haldane, kane_mele

CUBIC = [(1., 0., 0.), (0., 1., 0.), (0., 0., 1.)]
GRAPHENE_HOP = [{'n': 1, 't': -1.}, {'n': 2, 't': 0.1}, {'n': 3, 't': -0.05}]


def spectrum(sys, **params):
    sys.get_ham(**params)
    return np.sort(np.linalg.eigvalsh(sys.ham.toarray()))


def graphene_kspace():
    gra = KSpace(lattices.honeycomb())
    gra.set_hopping(GRAPHENE_HOP)
    return gra


def haldane_torus(n=10, t2=0.2):
    lat = lattices.honeycomb()
    lat.get_lattice(n, n)
    sys = System(lat, periodic=True)
    sys.set_hopping([{'n': 1, 't': 1.}])
    for tag, sign in (('aa', 1), ('bb', -1)):
        sys.set_hopping([{'n': 2, 'ang': 60., 'tag': tag, 't': 1j*t2*sign},
                                {'n': 2, 'ang': 0., 'tag': tag, 't': -1j*t2*sign},
                                {'n': 2, 'ang': 120., 'tag': tag, 't': -1j*t2*sign}])
    sys.set_onsite({'a': 0., 'b': 0.})
    return sys


class TestTorus(unittest.TestCase):

    def test_graphene_skewed_torus(self):
        # the honeycomb cell is skewed (60 degrees): three neighbour orders,
        # every bond counted once, the spectrum that of the k-mesh
        for n1, n2 in ((6, 6), (7, 5)):
            lat = lattices.honeycomb()
            lat.get_lattice(n1, n2)
            sys = System(lat, periodic=True)
            sys.set_hopping(GRAPHENE_HOP)
            np.testing.assert_allclose(spectrum(sys), np.sort(graphene_kspace().mesh_bands((n1, n2)).ravel()),
                                                 atol=1e-12)
            counts = np.bincount(sys.hop['n'].astype(int))[1:]
            self.assertEqual(counts.tolist(), [3 * n1 * n2, 6 * n1 * n2, 3 * n1 * n2])

    def test_sparse_neighbour_search(self):
        lat = lattices.honeycomb()
        lat.get_lattice(9, 7)
        sys = System(lat, periodic=True)
        sys.dense_max = 10  # force the k-d tree
        sys.set_hopping(GRAPHENE_HOP)
        np.testing.assert_allclose(spectrum(sys), np.sort(graphene_kspace().mesh_bands((9, 7)).ravel()),
                                             atol=1e-12)

    def test_cubic_3d(self):
        lat = Lattice([{'tag': 'a', 'r0': (0., 0., 0.)}], CUBIC)
        lat.get_lattice(4, 5, 6)
        sys = System(lat, periodic=True)
        sys.set_hopping([{'n': 1, 't': 1.}, {'n': 2, 't': 0.2}])
        cub = KSpace(Lattice([{'tag': 'a', 'r0': (0., 0., 0.)}], CUBIC))
        cub.set_hopping([{'n': 1, 't': 1.}, {'n': 2, 't': 0.2}])
        np.testing.assert_allclose(spectrum(sys), np.sort(cub.mesh_bands((4, 5, 6)).ravel()), atol=1e-12)

    def test_cylinder_is_a_ribbon(self):
        lat = lattices.honeycomb()
        lat.get_lattice(8, 6)
        sys = System(lat, periodic=(True, False))
        sys.set_hopping([{'n': 1, 't': 1.}])
        rib = ribbon(lattices.honeycomb(), [{'i': 0, 'j': 1, 'R': R, 't': 1.}
                                                       for R in [(0, 0), (-1, 0), (0, -1)]], width=6)
        k = (np.arange(8) / 8)[:, None] * rib.rec_vec_k[0][None]
        np.testing.assert_allclose(spectrum(sys), np.sort(rib.get_bands(k).ravel()), atol=1e-12)

    def test_angle_and_tag_selectors_include_wrapped_bonds(self):
        sys = haldane_torus()
        np.testing.assert_allclose(spectrum(sys), np.sort(haldane().mesh_bands((10, 10)).ravel()), atol=1e-12)
        self.assertTrue(np.all(sys.hop['ang'] >= 0.))
        # wrapped bonds keep their short bond vector
        self.assertTrue(np.any(sys._wrapped(sys.hop['i'].astype(int), sys.hop['j'].astype(int))))
        self.assertTrue(np.allclose(np.linalg.norm(sys._bond_vectors(sys.hop['i'].astype(int),
                                                                                         sys.hop['j'].astype(int)), axis=1)[sys.hop['n'] == 1], 1.))

    def test_vacancies(self):
        # removing sites keeps the torus: same as removing them from the full torus
        lat = lattices.square()
        lat.get_lattice(6, 6)
        full = System(lat, periodic=True)
        full.set_hopping([{'n': 1, 't': 1.}])
        full.get_ham()
        keep = np.setdiff1d(np.arange(36), [3, 17])
        expected = np.sort(np.linalg.eigvalsh(full.ham.toarray()[np.ix_(keep, keep)]))
        lat.remove_sites([3, 17])
        sys = System(lat, periodic=True)
        sys.set_hopping([{'n': 1, 't': 1.}])
        np.testing.assert_allclose(spectrum(sys), expected, atol=1e-12)

    def test_manual_wrapped_bond(self):
        lat = lattices.chain()
        lat.get_lattice(5)
        sys = System(lat, periodic=True)
        sys.set_hopping_manual({(4, 0): 1.})  # wraps around: points along +x
        self.assertAlmostEqual(sys.hop['ang'][0], 0.)
        open_sys = System(lat)
        open_sys.set_hopping_manual({(4, 0): 1.})
        self.assertAlmostEqual(open_sys.hop['ang'][0], 180.)  # the long way, across the chain
        sys.set_hopping_manual({(4, 0): 1.}, upper_part=False)
        self.assertAlmostEqual(sys.hop['ang'][1], -180.)

    def test_checks(self):
        lat = lattices.square()
        self.assertRaises(ValueError, System, lat, True)  # no get_lattice
        lat.get_lattice(4, 4)
        self.assertRaises(TypeError, System, lat, 1)
        self.assertRaises(TypeError, System, lat, (True, 1))
        self.assertRaises(ValueError, System, lat, (True,))
        self.assertEqual(System(lat).periodic, (False, False))
        small = lattices.square()
        small.get_lattice(2, 5)
        sys = System(small, periodic=True)
        self.assertRaises(ValueError, sys.set_hopping, [{'n': 1, 't': 1.}])
        # open along the short direction: fine
        System(small, periodic=(False, True)).set_hopping([{'n': 1, 't': 1.}])
        sq = System(lat, periodic=True)
        sq.set_hopping([{'n': 1, 't': 1.}])
        self.assertRaises(ValueError, sq.set_hopping, [{'n': 3, 't': 1.}])  # length 2 on a 4 x 4 torus
        self.assertRaises(ValueError, sq.set_magnetic_field, 0.1)
        sq.get_ham()
        self.assertRaises(ValueError, sq.get_local_chern_marker)


class TestPeierlsOnTorus(unittest.TestCase):

    def test_twisted_boundary(self):
        # a uniform A along x threads a flux theta through the torus: the
        # k-mesh shifted by theta / L_x
        lat = lattices.square()
        lat.get_lattice(8, 6)
        sys = System(lat, periodic=True)
        sys.set_hopping([{'n': 1, 't': 1.}])
        theta = 0.7
        sys.set_peierls_phase(lambda xi, yi, xj, yj: theta * (xj - xi) / 8)
        sq = KSpace(lattices.square())
        sq.set_hopping([{'n': 1, 't': 1.}])
        k = np.array([[(2*np.pi*m + theta) / 8, 2*np.pi*l / 6] for m in range(8) for l in range(6)])
        np.testing.assert_allclose(spectrum(sys), np.sort(sq.get_bands(k).ravel()), atol=1e-12)

    def test_orbital_blocks(self):
        # Kane-Mele with Rashba coupling on a torus, twisted along a1 (along x)
        n, theta = 6, 0.9
        lat = lattices.honeycomb()
        lat.get_lattice(n, n)
        sys = OrbitalSystem(lat, spin=True, periodic=True)
        sys.set_hopping([{'n': 1, 't': 1.}])
        sys.set_spin_orbit(0.06)
        sys.set_rashba(0.05)
        sys.set_onsite({'a': 0.1, 'b': -0.1})
        km = kane_mele(M=0.1, rashba=0.05)
        np.testing.assert_allclose(spectrum(sys), np.sort(km.mesh_bands((n, n)).ravel()), atol=1e-12)
        length = n * lat.prim_vec[0][0]
        sys.set_peierls_phase(lambda xi, yi, xj, yj: theta * (xj - xi) / length)
        b1, b2 = km.rec_vec_k
        k = np.array([m / n * b1 + l / n * b2 + np.array([theta / length, 0.])
                          for m in range(n) for l in range(n)])
        np.testing.assert_allclose(spectrum(sys), np.sort(km.get_bands(k).ravel()), atol=1e-12)


class TestBottIndexSystem(unittest.TestCase):

    def test_haldane(self):
        sys = haldane_torus()
        sys.get_ham()
        self.assertAlmostEqual(sys.get_bott_index(), 1., places=8)
        sys = haldane_torus(t2=-0.2)
        sys.get_ham()
        self.assertAlmostEqual(sys.get_bott_index(), -1., places=8)

    def test_orbital_and_checks(self):
        lat = lattices.honeycomb()
        lat.get_lattice(6, 6)
        sys = OrbitalSystem(lat, spin=True, periodic=True)
        sys.set_hopping([{'n': 1, 't': 1.}])
        sys.set_spin_orbit(0.06)
        sys.get_ham()
        self.assertAlmostEqual(sys.get_bott_index(), 0., places=8)  # time reversal: C = 0
        cyl = System(lat, periodic=(True, False))
        cyl.set_hopping([{'n': 1, 't': 1.}])
        cyl.get_ham()
        self.assertRaises(ValueError, cyl.get_bott_index)
        self.assertRaises(RuntimeError, System(lat, periodic=True).get_bott_index)  # no get_ham


class TestBridgesIoPlot(unittest.TestCase):

    def test_finite_system_is_periodic(self):
        sys = finite_system(haldane(), (6, 6), periodic=True)
        self.assertEqual(sys.periodic, (True, True))
        self.assertEqual(finite_system(haldane(), (6, 6)).periodic, (False, False))
        sys.get_ham()
        self.assertAlmostEqual(sys.get_bott_index(), 1., places=8)
        # kspace_from_system reads the flag
        self.assertAlmostEqual(kspace_from_system(haldane_torus()).chern_number(0, 30), 1.)

    def test_save_load(self):
        sys = haldane_torus(n=6)
        cyl = System(sys.lat, periodic=(True, False))
        cyl.set_hopping([{'n': 1, 't': 1.}])
        with tempfile.TemporaryDirectory() as tmp:
            for model in (sys, cyl):
                path = save_model(model, os.path.join(tmp, 'torus'))
                with np.load(path) as data:
                    self.assertEqual(int(data['version']), 2)
                back = load_model(path)
                self.assertEqual(back.periodic, model.periodic)
                np.testing.assert_allclose(spectrum(back), spectrum(model), atol=1e-12)
            open_sys = System(sys.lat)
            open_sys.set_hopping([{'n': 1, 't': 1.}])
            with np.load(save_model(open_sys, os.path.join(tmp, 'open'))) as data:
                self.assertEqual(int(data['version']), 1)
                self.assertNotIn('periodic', data.files)
            self.assertEqual(load_model(os.path.join(tmp, 'open.npz')).periodic, (False, False))

    def test_plot_skips_wrapped_bonds(self):
        lat = lattices.square()
        lat.get_lattice(4, 4)
        sys = System(lat, periodic=True)
        sys.set_hopping([{'n': 1, 't': 1.}])
        sys.get_ham()
        fig = Plot(sys).lattice(plt_hop=True)
        self.assertEqual(len(sys.hop), 32)
        drawn = [l for l in fig.axes[0].lines if len(l.get_xdata()) == 2]
        self.assertEqual(len(drawn), 24)  # the 8 wrapped bonds are left out
        plt.close(fig)
        sys.get_eig(eigenvec=True)
        fig = Plot(sys).intensity_area(sys.intensity[:, 0], plt_hop=True)
        plt.close(fig)
