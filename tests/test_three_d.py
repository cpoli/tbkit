"""
Lattices in 3D space: real-space Lattice/System, and 3D (or 2D-in-3D) KSpace.
"""
import unittest

import numpy as np

from tbkit.kspace import KSpace, PAULI, reciprocal_vectors, ribbon
from tbkit.lattice import Lattice, COOR_DTYPE_3D
from tbkit.system import System


CUBIC = [(1., 0., 0.), (0., 1., 0.), (0., 0., 1.)]


def cubic(n1=3, n2=3, n3=3):
    lat = Lattice(unit_cell=[{'tag': 'a', 'r0': (0., 0., 0.)}], prim_vec=CUBIC)
    lat.get_lattice(n1, n2, n3)
    return lat


def weyl_hop(k0=np.pi/2):
    '''
    H(k) = sin kx sx + sin ky sy + (2 + cos k0 - cos kx - cos ky - cos kz) sz:
    two Weyl points at k = (0, 0, +-k0).
    '''
    hop = []
    for d, sigma in ((0, 'x'), (1, 'y')):
        R = [0, 0, 0]
        R[d] = 1
        hop.append({'i': 0, 'j': 0, 'R': tuple(R), 't': -0.5j*PAULI[sigma] - 0.5*PAULI['z']})
    hop.append({'i': 0, 'j': 0, 'R': (0, 0, 1), 't': -0.5*PAULI['z']})
    return hop, (2. + np.cos(k0)) * PAULI['z']


def weyl(k0=np.pi/2):
    lat = Lattice(unit_cell=[{'tag': 'a', 'r0': (0., 0., 0.)}], prim_vec=CUBIC)
    ks = KSpace(lat, spin=True)
    hop, onsite = weyl_hop(k0)
    ks.set_hopping(hop)
    ks.set_onsite({'a': onsite})
    return ks


class TestLattice3D(unittest.TestCase):

    def test_get_lattice(self):
        lat = cubic(2, 3, 4)
        self.assertEqual(lat.sites, 24)
        self.assertEqual(lat.coor.dtype, np.dtype(COOR_DTYPE_3D))
        self.assertEqual(lat.space_dim, 3)
        # sorted by (z, y, x)
        order = np.lexsort((lat.coor['x'], lat.coor['y'], lat.coor['z']))
        self.assertTrue(np.array_equal(order, np.arange(24)))
        self.assertAlmostEqual(lat.coor['z'].max(), 3.)
        self.assertRaises(ValueError, lat.get_lattice, 2, 2, 0)
        lat2 = Lattice(unit_cell=[{'tag': 'a', 'r0': (0., 0.)}], prim_vec=[(1., 0.), (0., 1.)])
        self.assertRaises(ValueError, lat2.get_lattice, 2, 2, 2)

    def test_two_d_lattice_in_three_d_space(self):
        lat = Lattice(unit_cell=[{'tag': 'a', 'r0': (0., 0., 0.)}],
                             prim_vec=[(1., 0., 0.), (0., 0., 1.)])  # a vertical sheet
        lat.get_lattice(3, 2)
        self.assertEqual(lat.sites, 6)
        self.assertTrue(np.allclose(lat.coor['y'], 0.))

    def test_mismatched_dimensions(self):
        self.assertRaises(ValueError, Lattice, [{'tag': 'a', 'r0': (0., 0.)}], CUBIC)

    def test_geometry(self):
        lat = cubic(2, 2, 2)
        lat.shift_z(1.)
        self.assertAlmostEqual(lat.coor['z'].min(), 1.)
        lat.change_sign_z()
        self.assertAlmostEqual(lat.coor['z'].max(), -1.)
        lat.center()
        self.assertTrue(np.allclose([lat.coor[f].mean() for f in 'xyz'], 0.))
        lat = cubic(2, 2, 4)
        lat.slab(0.5, 2.5)
        self.assertEqual(lat.sites, 8)
        self.assertRaises(ValueError, lat.slab, 1., 0.)
        lat2d = Lattice(unit_cell=[{'tag': 'a', 'r0': (0., 0.)}], prim_vec=[(1., 0.)])
        lat2d.get_lattice(2)
        self.assertRaises(ValueError, lat2d.shift_z, 1.)
        self.assertRaises(ValueError, lat2d.change_sign_z)
        self.assertRaises(ValueError, lat2d.slab, 0., 1.)

    def test_clean_coor_add_sub(self):
        lat = cubic(2, 2, 2)
        coor = np.array([(0., 0., 0., 'a'), (0., 0., 5., 'b')], dtype=COOR_DTYPE_3D)
        lat.add_sites(coor)
        self.assertEqual(lat.sites, 10)
        lat.clean_coor()  # the duplicate at the origin, but not (0, 0, 5)
        self.assertEqual(lat.sites, 9)
        self.assertIn('b', lat.tags)
        self.assertRaises(TypeError, lat.add_sites,
                                np.array([(0., 0., 'a')], dtype=[('x', 'f8'), ('y', 'f8'), ('tag', 'U1')]))
        top = cubic(2, 2, 1)
        top.shift_z(1.)
        diff = cubic(2, 2, 2) - top
        self.assertEqual(diff.sites, 4)
        self.assertTrue(np.allclose(diff.coor['z'], 0.))
        lat = cubic(2, 2, 2)
        lat -= top
        self.assertEqual(lat.sites, 4)

    def test_remove_dangling(self):
        lat = cubic(2, 2, 2)
        lat.add_sites(np.array([(0., 0., -1., 'a')], dtype=COOR_DTYPE_3D))
        lat.remove_dangling()
        self.assertEqual(lat.sites, 8)


class TestSystem3D(unittest.TestCase):

    def test_cubic_spectrum(self):
        sys = System(cubic(4, 4, 4))
        sys.set_hopping([{'n': 1, 't': 1.}])
        sys.get_ham()
        ham = sys.ham.toarray()
        self.assertTrue(np.allclose(ham, ham.T))
        self.assertEqual(len(sys.hop), 3 * 4 * 4 * 3)
        sys.get_eig()
        k = np.pi * np.arange(1, 5) / 5
        e1 = 2 * np.cos(k)
        exact = np.sort((e1[:, None, None] + e1[None, :, None] + e1[None, None, :]).ravel())
        self.assertTrue(np.allclose(sys.en, exact))

    def test_vertical_bonds_upper_and_lower(self):
        lat = Lattice(unit_cell=[{'tag': 'a', 'r0': (0., 0., 0.)}], prim_vec=[(0., 0., 1.)])
        lat.get_lattice(4)
        sys = System(lat)
        sys.set_hopping([{'n': 1, 't': 1.}])
        self.assertTrue(np.allclose(sys.hop['ang'], 0.))
        sys.set_hopping([{'n': 1, 't': 2.}], upper_part=False)
        self.assertTrue(np.allclose(sys.hop['ang'][sys.hop['t'] == 2.], -180.))
        sys.get_ham()
        ham = sys.ham.toarray()
        self.assertEqual(ham[0, 1], 1.)
        self.assertEqual(ham[1, 0], 2.)


class TestKSpace3D(unittest.TestCase):

    def test_reciprocal_vectors_3d(self):
        prim = [(1., 0., 0.), (0.5, 1., 0.), (0.2, 0.3, 2.)]
        rec = np.array(reciprocal_vectors(prim))
        self.assertTrue(np.allclose(np.array(prim) @ rec.T, 2*np.pi*np.eye(3)))
        # a 2D lattice in 3D space: b_i in the plane of the a_i
        prim2 = [(1., 0., 1.), (0., 1., 0.)]
        rec2 = np.array(reciprocal_vectors(prim2))
        self.assertTrue(np.allclose(np.array(prim2) @ rec2.T, 2*np.pi*np.eye(2)))
        self.assertTrue(np.allclose(rec2 @ np.cross(*prim2), 0.))

    def test_collinear_prim_vec_refused(self):
        lat = Lattice(unit_cell=[{'tag': 'a', 'r0': (0., 0.)}], prim_vec=[(1., 0.), (2., 0.)])
        self.assertRaises(ValueError, KSpace, lat)

    def test_cubic_bands(self):
        ks = KSpace(Lattice(unit_cell=[{'tag': 'a', 'r0': (0., 0., 0.)}], prim_vec=CUBIC))
        ks.set_hopping([{'i': 0, 'j': 0, 'R': R, 't': 1.}
                                 for R in [(1, 0, 0), (0, 1, 0), (0, 0, 1)]])
        k = np.array([0.3, -1.2, 2.])
        self.assertAlmostEqual(ks.get_ham(k)[0, 0].real, 2*np.cos(k).sum())
        en = ks.mesh_bands(nk=4)
        self.assertEqual(en.shape, (64, 1))
        self.assertAlmostEqual(en.max(), 6.)
        _, en_path = ks.k_path([(0., 0., 0.), (np.pi, np.pi, np.pi)], nk=5)
        self.assertAlmostEqual(en_path[-1, 0], -6.)

    def test_weyl_chern_slices(self):
        wey = weyl(k0=np.pi/2)
        # gapless exactly at the Weyl points
        en = np.linalg.eigvalsh(wey.get_ham([0., 0., np.pi/2]))
        self.assertTrue(np.allclose(en, 0.))
        inside = wey.chern_number(bands=[0], nk=20, k_fixed=0.)
        outside = wey.chern_number(bands=[0], nk=20, k_fixed=0.5)
        self.assertAlmostEqual(abs(inside), 1., places=6)
        self.assertAlmostEqual(outside, 0., places=6)
        # planes containing k_z see no net flux: sum over a (k_x, k_z) plane
        self.assertAlmostEqual(wey.chern_number(bands=[0], nk=20, plane=(0, 2), k_fixed=0.25),
                                        0., places=6)
        self.assertRaises(TypeError, wey.chern_number, 0, 10, [0, 1])
        self.assertRaises(ValueError, wey.chern_number, 0, 10, (0, 0))
        self.assertRaises(ValueError, wey.chern_number, 0, 10, (0, 3))
        self.assertRaises(TypeError, wey.chern_number, 0, 10, (0, 1), 'a')

    def test_chern_orientation_follows_a3(self):
        # the plane's flux is oriented along a3: reversing a3 (and kz)
        # reverses the Chern number of the k_z = 0 plane
        c_up = weyl().chern_number(bands=[0], nk=16)
        lat = Lattice(unit_cell=[{'tag': 'a', 'r0': (0., 0., 0.)}],
                             prim_vec=[(1., 0., 0.), (0., 1., 0.), (0., 0., -1.)])
        ks = KSpace(lat, spin=True)
        hop, onsite = weyl_hop()
        ks.set_hopping([dict(h, R=(h['R'][0], h['R'][1], -h['R'][2])) for h in hop])
        ks.set_onsite({'a': onsite})
        self.assertAlmostEqual(ks.chern_number(bands=[0], nk=16), -c_up, places=6)

    def test_slab_fermi_arc(self):
        # a slab finite along y: surface states at E = 0 for k_x = 0 only
        # between the projected Weyl points
        hop, onsite = weyl_hop(k0=np.pi/2)
        lat = Lattice(unit_cell=[{'tag': 'a', 'r0': (0., 0., 0.)}], prim_vec=CUBIC)
        slab = ribbon(lat, hop, width=20, direction=1, onsite={'a': onsite}, spin=True)
        self.assertEqual(slab.dim, 2)
        self.assertEqual(slab.space_dim, 3)
        # k coordinates: along a1 (x), then the orthonormal direction in the
        # (a1, a3) plane (z)
        self.assertTrue(np.allclose(slab.k_basis, [[1., 0.], [0., 0.], [0., 1.]]))
        gap_in = np.min(np.abs(slab.get_bands([[0., 0.3]])))
        gap_out = np.min(np.abs(slab.get_bands([[0., 2.5]])))
        self.assertLess(gap_in, 1e-3)
        self.assertGreater(gap_out, 0.3)
        self.assertRaises(ValueError, ribbon, lat, hop, 5, 3)

    def test_two_d_lattice_in_three_d_space(self):
        # a square lattice in the (x, z) plane behaves like the planar one
        lat = Lattice(unit_cell=[{'tag': 'a', 'r0': (0., 0., 0.)}],
                             prim_vec=[(1., 0., 0.), (0., 0., 1.)])
        ks = KSpace(lat)
        ks.set_hopping([{'i': 0, 'j': 0, 'R': (1, 0), 't': 1.},
                                 {'i': 0, 'j': 0, 'R': (0, 1), 't': 1.}])
        self.assertAlmostEqual(ks.get_ham([0.4, 1.])[0, 0].real, 2*np.cos(0.4) + 2*np.cos(1.))
        self.assertTrue(np.allclose(ks.rec_vec_k, 2*np.pi*np.eye(2)))
        self.assertEqual(ks.mesh_bands(nk=5).shape, (25, 1))


class TestChecks3D(unittest.TestCase):

    def test_dim_checks(self):
        import tbkit.error_handling as eh
        eh.dim_2(2)  # kept for backward compatibility
        self.assertRaises(ValueError, eh.dim_2, 3)
        eh.dim_min(3, 2)
        self.assertRaises(ValueError, eh.dim_min, 1, 2)


if __name__ == '__main__':
    unittest.main()
