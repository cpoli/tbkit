"""
Supercells, band unfolding, spectral functions and twisted bilayers.
"""
import unittest

import numpy as np
from scipy.optimize import brentq

import tbkit.lattices as lattices
from tbkit.kspace import KSpace, PAULI
from tbkit.moire import (supercell, unfold, spectral_function, pz_hopping, commensurate_angle,
                                          twisted_bilayer, magic_angle_parameter, SupercellKSpace, MoireKSpace)


def graphene(t=-1.):
    gra = KSpace(lattices.honeycomb())
    gra.set_hopping([{'i': 0, 'j': 1, 'R': (0, 0), 't': t}, {'i': 0, 'j': 1, 'R': (-1, 0), 't': t},
                              {'i': 0, 'j': 1, 'R': (0, -1), 't': t}, {'i': 0, 'j': 0, 'R': (1, 0), 't': 0.1}])
    gra.set_onsite({'a': 0.3, 'b': -0.2})
    return gra


def chain(t=1.):
    ch = KSpace(lattices.chain())
    ch.set_hopping([{'i': 0, 'j': 0, 'R': (1,), 't': t}])
    return ch


def moire_k(tbl):
    '''Corner K of the moire Brillouin zone.'''
    b1, b2 = tbl.rec_vec_k
    c = b2 if b1 @ b2 > 0 else b1 + b2
    return np.linalg.solve(2 * np.array([b1, c]), np.array([b1 @ b1, c @ c]))


def central_bands(tbl, ks):
    n = tbl.norb // 2
    return np.array([np.linalg.eigvalsh(tbl.get_ham(k))[n - 2:n + 2] for k in ks])


def dirac_velocity(tbl, dq=1e-2):
    k = moire_k(tbl)
    e0 = central_bands(tbl, [k])[0]
    e1 = central_bands(tbl, [k * (1 + dq)])[0]
    return np.mean(np.abs(e1 - e0.mean())) / (dq * np.linalg.norm(k))


class TestSupercell(unittest.TestCase):

    def test_structure(self):
        gra = graphene()
        sc = supercell(gra, [[2, 1], [-1, 3]])
        self.assertIsInstance(sc, SupercellKSpace)
        self.assertEqual(sc.n_cells, 7)
        self.assertEqual(sc.norb, 14)
        self.assertTrue(sc.is_hermitian())
        # the same torus, cut two ways: identical spectra
        prim = np.linalg.eigvalsh(gra.finite_ham(4, periodic=True))
        big = np.linalg.eigvalsh(supercell(gra, [[2, 0], [0, 2]]).finite_ham(2, periodic=True))
        np.testing.assert_allclose(big, prim, atol=1e-10)
        # 1D, with an integer matrix
        self.assertEqual(supercell(chain(), 3).norb, 3)

    def test_folding(self):
        # the supercell bands at K are the primitive bands at K + G, G in the
        # supercell reciprocal lattice
        gra = graphene()
        sc = supercell(gra, [[2, 0], [0, 2]])
        k = np.array([0.4, -0.7])
        folded = np.concatenate([np.linalg.eigvalsh(gra.get_ham(k + f @ sc.rec_vec_k))
                                          for f in np.array([[0, 0], [0, 1], [1, 0], [1, 1]])])
        np.testing.assert_allclose(np.linalg.eigvalsh(sc.get_ham(k)), np.sort(folded), atol=1e-10)

    def test_spinful_and_overlap(self):
        ks = KSpace(lattices.square(), spin=True)
        ks.set_hopping([{'i': 0, 'j': 0, 'R': (1, 0), 't': PAULI['0'] + 0.3j * PAULI['y']},
                                {'i': 0, 'j': 0, 'R': (0, 1), 't': PAULI['0'] - 0.3j * PAULI['x']}])
        ks.set_onsite({'a': 0.2 * PAULI['x']})
        sc = supercell(ks, [[1, 1], [-1, 1]])
        self.assertEqual(sc.norb, 4)
        pts = np.array([[0.3, 1.1], [2., -0.4]])
        en, w = unfold(sc, pts)
        for k, e, ww in zip(pts, en, w):
            np.testing.assert_allclose(np.sort(e[ww > 0.5]), np.linalg.eigvalsh(ks.get_ham(k)), atol=1e-10)
        over = chain()
        over.set_overlap([{'i': 0, 'j': 0, 'R': (1,), 't': 0.1}])
        sc_over = supercell(over, 2)
        np.testing.assert_allclose(sc_over.get_overlap([0.3]), sc_over.get_overlap([0.3]).conj().T)
        self.assertRaises(ValueError, unfold, sc_over, [[0.3]])


class TestUnfolding(unittest.TestCase):

    def test_pristine_recovers_primitive_bands(self):
        gra = graphene()
        sc = supercell(gra, [[3, 1], [-1, 2]])
        pts = np.random.default_rng(1).uniform(-4., 4., (12, 2))
        en, w = unfold(sc, pts)
        # weights are 0 or 1, add up to the primitive orbitals ...
        np.testing.assert_allclose(w.sum(axis=1), 2., atol=1e-10)
        np.testing.assert_allclose(w * (1 - w), 0., atol=1e-10)
        # ... and the states of weight 1 are exactly the primitive bands
        for k, e, ww in zip(pts, en, w):
            np.testing.assert_allclose(np.sort(e[ww > 0.5]), np.linalg.eigvalsh(gra.get_ham(k)), atol=1e-12)

    def test_charge_density_wave_coherence_factors(self):
        # chain with a staggered potential +-V (2-site supercell): E = +-sqrt(eps^2 + V^2),
        # eps = 2t cos k, and the upper state has weight (1 + eps/E)/2 at k
        t, V = 1., 0.4
        sc = supercell(chain(t), 2)
        sc.onsite = np.array([V, -V], dtype='c16')
        ks = np.linspace(-np.pi, np.pi, 9)[:, None]
        en, w = unfold(sc, ks)
        eps = 2 * t * np.cos(ks[:, 0])
        e = np.sqrt(eps ** 2 + V ** 2)
        np.testing.assert_allclose(en[:, 1], e, atol=1e-12)
        np.testing.assert_allclose(w[:, 1], (1 + eps / e) / 2, atol=1e-12)
        np.testing.assert_allclose(w[:, 0], (1 - eps / e) / 2, atol=1e-12)

    def test_spectral_function(self):
        gra = graphene()
        sc = supercell(gra, [[2, 0], [0, 2]])
        pts = np.array([[0.2, 0.5], [1.5, -0.3]])
        omega = np.linspace(-8., 8., 4001)
        a = spectral_function(sc, pts, omega, broadening=0.05, kernel='gaussian')
        self.assertEqual(a.shape, (2, 4001))
        np.testing.assert_allclose(np.trapezoid(a, omega, axis=1), 2., atol=1e-6)
        # peaks at the primitive bands
        for k, row in zip(pts, a):
            e = np.linalg.eigvalsh(gra.get_ham(k))
            peaks = omega[np.argsort(row)[-1]]
            self.assertLess(np.min(np.abs(e - peaks)), 5e-3)
        lor = spectral_function(sc, pts[0], [0.1])
        self.assertEqual(lor.shape, (1, 1))
        # a uniform shift of the supercell shifts the unfolded bands
        sc.onsite = sc.onsite + 0.5
        shifted = spectral_function(sc, pts, omega + 0.5, 0.05, 'gaussian')
        np.testing.assert_allclose(shifted, a, atol=1e-10)

    def test_disorder_smears_weights(self):
        sc = supercell(graphene(), [[4, 0], [0, 4]])
        rng = np.random.default_rng(0)
        sc.onsite = sc.onsite + rng.uniform(-1., 1., sc.norb)
        en, w = unfold(sc, [[0.3, 0.2]])
        np.testing.assert_allclose(w.sum(), 2., atol=1e-10)
        self.assertGreater(np.sum((w > 0.01) & (w < 0.99)), 4)

    def test_errors(self):
        gra = graphene()
        self.assertRaises(TypeError, supercell, 'x', [[1, 0], [0, 1]])
        self.assertRaises(ValueError, supercell, gra, [[1, 0]])
        self.assertRaises(TypeError, supercell, gra, [[1.5, 0], [0, 1]])
        self.assertRaises(ValueError, supercell, gra, [[1, 2], [2, 4]])
        self.assertRaises(TypeError, unfold, gra, [[0., 0.]])
        sc = supercell(gra, [[2, 0], [0, 1]])
        self.assertRaises(ValueError, unfold, sc, [[0., 0., 0.]])
        self.assertRaises(ValueError, spectral_function, sc, [[0., 0.]], [])
        self.assertRaises(ValueError, spectral_function, sc, [[0., 0.]], [0.], -1.)
        self.assertRaises(ValueError, spectral_function, sc, [[0., 0.]], [0.], 0.1, 'box')


class TestTwistedBilayer(unittest.TestCase):

    def test_commensurate_angles(self):
        for m in (1, 2, 5, 31):
            n = 3 * m * m + 3 * m + 1
            self.assertAlmostEqual(np.cos(commensurate_angle(m)), (n - 0.5) / n, places=12)
            ns = 2 * m * m + 2 * m + 1
            self.assertAlmostEqual(np.cos(commensurate_angle(m, 'square')), 2 * m * (m + 1) / ns, places=12)
        self.assertAlmostEqual(np.degrees(commensurate_angle(1)), 21.7868, places=4)
        # graphene's first magic angle
        self.assertAlmostEqual(np.degrees(commensurate_angle(31)), 1.0501, places=4)

    def test_structure(self):
        for m in (1, 2):
            tbl = twisted_bilayer(m)
            self.assertIsInstance(tbl, MoireKSpace)
            self.assertEqual(tbl.norb, 4 * (3 * m * m + 3 * m + 1))
            self.assertEqual(tbl.n_layer_cells, 3 * m * m + 3 * m + 1)
            self.assertEqual(int(tbl.layer.sum()), tbl.norb // 2)
            ham = tbl.get_ham([0.13, -0.4])
            np.testing.assert_allclose(ham, ham.conj().T, atol=1e-14)
            # the moire cell: |T|^2 = (3m^2 + 3m + 1) a_lat^2
            a_lat = np.sqrt(3) * 0.142
            self.assertAlmostEqual(np.linalg.norm(tbl.lat.prim_vec[0]) ** 2,
                                              (3 * m * m + 3 * m + 1) * a_lat ** 2, places=10)
        sq = twisted_bilayer(2, 'square', a=1., d=1., cutoff=1.5)
        self.assertEqual(sq.norb, 2 * 13)
        self.assertAlmostEqual(sq.theta, commensurate_angle(2, 'square'))

    def test_decoupled_layers(self):
        # without interlayer hopping: two folded monolayers, whose Dirac
        # points both land on the moire K point (four degenerate states)
        def intra(d):
            return np.where(np.abs(d[:, 2]) > 0, 0., pz_hopping(d))
        tbl = twisted_bilayer(2, hopping=intra)
        e = central_bands(tbl, [moire_k(tbl)])[0]
        self.assertLess(e.max() - e.min(), 1e-10)
        # a complex hopping function is accepted
        tbl_c = twisted_bilayer(1, hopping=lambda d: (1 + 0j) * intra(d))
        np.testing.assert_allclose(np.linalg.eigvalsh(tbl_c.get_ham([0.1, 0.2])),
                                            np.linalg.eigvalsh(twisted_bilayer(1, hopping=intra).get_ham([0.1, 0.2])),
                                            atol=1e-10)

    def test_pz_hopping(self):
        d = np.array([[0.142, 0., 0.], [0., 0., 0.335], [0.246, 0., 0.]])
        t = pz_hopping(d)
        self.assertAlmostEqual(t[0], -2.7)  # nearest neighbours
        self.assertAlmostEqual(t[1], 0.48)  # AA interlayer
        self.assertAlmostEqual(t[2], -2.7 * np.exp(-(0.246 - 0.142) / 0.0453))
        self.assertRaises(ValueError, pz_hopping, [[0., 0., 0.]])
        self.assertRaises(ValueError, pz_hopping, [[1., 0.]])
        self.assertRaises(ValueError, pz_hopping, d, -2.7, 0.48, -1.)

    def test_bistritzer_macdonald_parameters(self):
        # w ~ 110 meV for the Moon-Koshino hoppings, and the first magic
        # angle alpha = 1/sqrt(3) close to 1.1 degrees
        w, alpha, ratio = magic_angle_parameter(np.radians(1.05))
        self.assertAlmostEqual(w, 0.11, delta=0.005)
        magic = brentq(lambda th: magic_angle_parameter(np.radians(th))[2], 0.8, 2.)
        self.assertAlmostEqual(magic, 1.1, delta=0.1)
        _, alpha2, _ = magic_angle_parameter(np.radians(1.05), interlayer_scale=2.)
        self.assertAlmostEqual(alpha2, 2 * alpha)
        self.assertRaises(TypeError, magic_angle_parameter, 0.1, interlayer='x')

    def test_fermi_velocity_follows_bistritzer_macdonald(self):
        # the Dirac velocity of the tight-binding moire bands, v*/v, against
        # (1 - 3 alpha^2) / (1 + 6 alpha^2) at large angles
        for m, tol in ((3, 0.01), (5, 0.015)):
            tbl = twisted_bilayer(m)
            v = dirac_velocity(tbl)
            v0 = dirac_velocity(twisted_bilayer(m, interlayer_scale=1e-9))
            _, _, ratio = magic_angle_parameter(tbl.theta, t=-v0 / (1.5 * 0.142))
            self.assertAlmostEqual(v / v0, ratio, delta=tol)
            self.assertLess(v / v0, 0.96)

    def test_flat_bands_near_the_magic_angle(self):
        # the physics depends on alpha only: scaling the interlayer hopping
        # brings the magic angle to theta = 6.0 degrees (m = 5); the four
        # central bands flatten by more than an order of magnitude
        widths = []
        for s in (1., 4.6):
            tbl = twisted_bilayer(5, interlayer_scale=s)
            b1 = tbl.rec_vec_k[0]
            k = moire_k(tbl)
            path = np.concatenate([np.linspace(k, 0 * k, 6), np.linspace(0 * k, b1 / 2, 5)])
            bands = central_bands(tbl, path)
            widths.append(bands.max() - bands.min())
        self.assertGreater(widths[0] / widths[1], 15.)

    def test_errors(self):
        self.assertRaises(ValueError, twisted_bilayer, 0)
        self.assertRaises(ValueError, twisted_bilayer, 1, 'kagome')
        self.assertRaises(TypeError, twisted_bilayer, 1, hopping=3)
        self.assertRaises(ValueError, twisted_bilayer, 1, hopping=lambda d: np.zeros(2))
        self.assertRaises(ValueError, twisted_bilayer, 1, interlayer_scale=0.)
        self.assertRaises(ValueError, twisted_bilayer, 1, cutoff=-1.)
        self.assertRaises(ValueError, commensurate_angle, 1, 'kagome')


if __name__ == '__main__':
    unittest.main()
