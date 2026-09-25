"""
Exceptional and diabolical points of 2D Bloch bands: eigenvalue tracking,
vorticity, the discriminant and its winding, the EP finder (points, rings,
Fermi arcs), encircling, and biorthogonal Chern numbers.

Expected values, derived by hand:

* ``pair_model``: H = (sin kx + i g) sx + sin ky sy, so
  E^2 = sin^2 kx + sin^2 ky - g^2 + 2 i g sin kx. EPs where sin kx = 0 and
  sin ky = +-g. Near (0, asin g): Delta = 4E^2 ~ 8g(cos(ky0) dky + i dkx),
  which winds -1 counterclockwise, so nu = +1/2 (W = -2 nu); opposite at
  (0, -asin g). At kx = pi, sin kx ~ -(kx - pi) flips both charges. The
  Fermi arc Re(E1 - E2) = 0 is sin kx = 0, |sin ky| < g.
* ``ring_model``: H = sin kx sx + sin ky sy + i g sz, E^2 = sin^2 kx +
  sin^2 ky - g^2 is real: EPs on the rings sin^2 kx + sin^2 ky = g^2.
* ``ep3_model``: H = [[0, 1, 0], [0, 0, 1], [z, 0, 0]], z = sin kx + i sin ky:
  E^3 = z, Delta = -27 z^2 (winding +2 at k = 0), every pair difference
  ~ z^(1/3) (nu = -1/3), an order-3 EP at each of the four TRIM.
"""
import unittest

import numpy as np

import tbkit.lattices as lattices
import tbkit.exceptional as ex
from tbkit.kspace import KSpace, PAULI
from tbkit.lattice import Lattice

G = 0.3
KY0 = np.arcsin(G)
HONEYCOMB_NN = [{'i': 0, 'j': 1, 'R': (0, 0), 't': 1.}, {'i': 0, 'j': 1, 'R': (-1, 0), 't': 1.},
                         {'i': 0, 'j': 1, 'R': (0, -1), 't': 1.}]


def dirac_square(onsite):
    '''sin kx sx + sin ky sy, plus a 2x2 onsite block.'''
    m = KSpace(lattices.square(), spin=True)
    m.set_onsite({'a': onsite})
    m.set_hopping([{'i': 0, 'j': 0, 'R': (1, 0), 't': -0.5j * PAULI['x']},
                             {'i': 0, 'j': 0, 'R': (0, 1), 't': -0.5j * PAULI['y']}])
    return m


def pair_model(g=G):
    return dirac_square(1j * g * PAULI['x'])


def ring_model(g=G):
    return dirac_square(1j * g * PAULI['z'])


def ep3_model():
    lat = Lattice(unit_cell=[{'tag': 'a', 'r0': (0., 0.)}, {'tag': 'b', 'r0': (0.3, 0.)},
                                        {'tag': 'c', 'r0': (0.6, 0.)}], prim_vec=[(1., 0.), (0., 1.)])
    m = KSpace(lat)
    m.set_hopping([{'i': 0, 'j': 1, 'R': (0, 0), 't': 1.}, {'i': 1, 'j': 2, 'R': (0, 0), 't': 1.},
                             {'i': 2, 'j': 0, 'R': (1, 0), 't': -0.5j}, {'i': 2, 'j': 0, 'R': (-1, 0), 't': 0.5j},
                             {'i': 2, 'j': 0, 'R': (0, 1), 't': 0.5}, {'i': 2, 'j': 0, 'R': (0, -1), 't': -0.5}],
                            hermitian=False)
    return m


def graphene():
    gra = KSpace(lattices.honeycomb())
    gra.set_hopping(HONEYCOMB_NN)
    return gra


def haldane(gamma=0., t2=0.2):
    '''Haldane model (phi = pi/2, M = 0) with staggered gain/loss +-i gamma.'''
    hal = graphene()
    nnn = [(0, 1), (-1, 0), (1, -1)]
    hal.set_hopping([{'i': 0, 'j': 0, 'R': R, 't': 1j * t2} for R in nnn]
                             + [{'i': 1, 'j': 1, 'R': R, 't': -1j * t2} for R in nnn])
    hal.set_onsite({'a': 1j * gamma, 'b': -1j * gamma})
    return hal


def k_point(gra):
    b1, b2 = (np.array(b) for b in gra.rec_vec_k)
    return (b1 - b2) / 3


def wrap(k):
    return (k + np.pi) % (2 * np.pi) - np.pi


class TestTracking(unittest.TestCase):

    def test_eigenvalues_swap_around_an_ep(self):
        s, k, en = ex.track_eigenvalues(pair_model(), ex.circle((0., KY0), 0.1))
        self.assertEqual(s[0], 0.)
        self.assertEqual(s[-1], 1.)
        self.assertTrue(np.allclose(k[0], k[-1]))
        # continuous: the two labels come back exchanged
        self.assertTrue(np.allclose(en[-1], en[0][::-1]))
        self.assertLess(np.max(np.abs(np.diff(en, axis=0))), 0.1)
        # two turns bring them back
        _, _, en2 = ex.track_eigenvalues(pair_model(), ex.circle((0., KY0), 0.1), n_loops=2)
        self.assertTrue(np.allclose(en2[-1], en2[0]))

    def test_independent_of_solver_order(self):
        # a callable model whose eigenvalue order is scrambled by a basis permutation
        base = pair_model()
        perm = np.array([1, 0])

        def scrambled(p):
            return base.get_ham(p)[np.ix_(perm, perm)]
        lp = ex.circle((0., KY0), 0.1)
        self.assertAlmostEqual(ex.vorticity(scrambled, lp), ex.vorticity(base, lp), places=12)

    def test_polygon_loop_and_crossing(self):
        # a square polygon around the EP; a degenerate model is tracked as a cluster
        square = [(-0.1, KY0 - 0.1), (0.1, KY0 - 0.1), (0.1, KY0 + 0.1), (-0.1, KY0 + 0.1)]
        self.assertAlmostEqual(ex.vorticity(pair_model(), square), 0.5, places=10)
        s, _, en = ex.track_eigenvalues(lambda p: np.eye(2), square, nk=4)
        self.assertTrue(np.allclose(en, 1.))
        self.assertRaises(ValueError, ex.vorticity, lambda p: np.zeros((2, 2)), square)

    def test_loop_through_an_ep(self):
        lp = ex.circle((-0.1, KY0), 0.1)  # passes through (0, asin g)
        self.assertRaises(ValueError, ex.vorticity, pair_model(), lp)
        self.assertRaises(ValueError, ex.discriminant_winding, pair_model(), lp)

    def test_checks(self):
        lp = ex.circle((0., 0.), 0.1)
        self.assertRaises(TypeError, ex.vorticity, 'model', lp)
        self.assertRaises(ValueError, ex.vorticity, KSpace(lattices.chain()), lp)
        self.assertRaises(TypeError, ex.vorticity, pair_model(), 3)
        self.assertRaises(ValueError, ex.vorticity, pair_model(), [(0., 0.), (1., 0.)])
        self.assertRaises(ValueError, ex.vorticity, pair_model(), lambda s: (s, s, s))
        self.assertRaises(ValueError, ex.vorticity, pair_model(), lambda s: (s, 0.))
        self.assertRaises(ValueError, ex.vorticity, lambda p: np.ones(3), lp)
        self.assertRaises(ValueError, ex.vorticity, pair_model(), lp, (0, 0))
        self.assertRaises(TypeError, ex.vorticity, pair_model(), lp, 0)
        self.assertRaises(TypeError, ex.vorticity, pair_model(), lp, (0, 1), 1.5)
        self.assertRaises(TypeError, ex.track_eigenvalues, pair_model(), lp, 1.)
        self.assertRaises(ValueError, ex.circle, (0., 0.), 0.)
        self.assertRaises(ValueError, ex.circle, (0.,), 1.)
        overlap = graphene()
        overlap.set_overlap([{'i': 0, 'j': 1, 'R': (0, 0), 't': 0.1}])
        self.assertRaises(ValueError, ex.vorticity, overlap, lp)


class TestVorticityAndDiscriminant(unittest.TestCase):

    def test_discriminant(self):
        rng = np.random.default_rng(1)
        for n in (2, 3, 4):
            h = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
            e = np.linalg.eigvals(h)
            prod = np.prod([(e[i] - e[j]) ** 2 for i in range(n) for j in range(i + 1, n)])
            self.assertTrue(np.isclose(ex.discriminant(lambda p, h=h: h, (0., 0.)), prod, rtol=1e-10))
        # two bands: 4 d.d
        k = np.array([0.4, -0.7])
        d = np.array([np.sin(k[0]) + 1j * G, np.sin(k[1])])
        self.assertAlmostEqual(ex.discriminant(pair_model(), k), 4 * d @ d, places=12)
        self.assertEqual(ex.discriminant(lambda p: [[2.]], (0., 0.)), 1.)
        self.assertEqual(ex.discriminant(lambda p: np.zeros((3, 3)), (0., 0.)), 0.)
        self.assertRaises(TypeError, ex.discriminant, pair_model(), 0.)

    def test_second_order_eps(self):
        m = pair_model()
        for center, nu in (((0., KY0), 0.5), ((0., -KY0), -0.5),
                                  ((np.pi, KY0), -0.5), ((np.pi, -KY0), 0.5)):
            lp = ex.circle(center, 0.1)
            self.assertAlmostEqual(ex.vorticity(m, lp), nu, places=10)
            self.assertAlmostEqual(ex.vorticity(m, lp, (1, 0)), nu, places=10)
            self.assertAlmostEqual(ex.discriminant_winding(m, lp), -2 * nu, places=10)
        # around the pair, and around nothing
        for lp in (ex.circle((0., 0.), 0.6), ex.circle((1.5, 1.5), 0.3)):
            self.assertAlmostEqual(ex.vorticity(m, lp), 0., places=10)
            self.assertAlmostEqual(ex.discriminant_winding(m, lp), 0., places=10)

    def test_conservation_of_vorticity(self):
        # the Hermitian DP (g = 0) and the EP pair it splits into: same total
        big = ex.circle((0., 0.), 0.8)
        for g in (0., 0.1, 0.3, 0.5):
            self.assertAlmostEqual(ex.discriminant_winding(pair_model(g), big), 0., places=10)
            self.assertAlmostEqual(ex.vorticity(pair_model(g), big), 0., places=10)

    def test_graphene_diabolical_points(self):
        gra = graphene()
        for k0 in (k_point(gra), -k_point(gra)):
            lp = ex.circle(k0, 0.2)
            self.assertAlmostEqual(ex.vorticity(gra, lp), 0., places=12)
            self.assertAlmostEqual(ex.discriminant_winding(gra, lp), 0., places=12)
            # Delta = 4|f|^2 >= 0: real and positive
            d = ex.discriminant(gra, k0 + 0.1)
            self.assertGreater(d.real, 0.)
            self.assertAlmostEqual(d.imag, 0., places=12)

    def test_third_order_ep(self):
        m = ep3_model()
        lp = ex.circle((0., 0.), 0.3)
        for pair in ((0, 1), (0, 2), (1, 2)):
            self.assertAlmostEqual(ex.vorticity(m, lp, pair), -1. / 3, places=10)
        self.assertAlmostEqual(ex.discriminant_winding(m, lp), 2., places=10)
        # the three eigenvalues permute cyclically
        _, _, en = ex.track_eigenvalues(m, lp)
        self.assertFalse(np.allclose(en[-1], en[0]))
        # same set of values; not via sort_complex, which orders two eigenvalues
        # with equal real parts by rounding noise
        dist = np.abs(en[-1][:, None] - en[0][None, :])
        self.assertTrue(np.allclose(dist.min(axis=1), 0.))
        self.assertEqual(sorted(dist.argmin(axis=1)), [0, 1, 2])
        _, _, en3 = ex.track_eigenvalues(m, lp, n_loops=3)
        self.assertTrue(np.allclose(en3[-1], en3[0]))

    def test_winding_checks(self):
        self.assertRaises(TypeError, ex.discriminant_winding, pair_model(), ex.circle((0., 0.), 1.), 0.5)
        self.assertRaises(ValueError, ex.discriminant_winding, pair_model(),
                                lambda s: (s, 0.))


class TestFinder(unittest.TestCase):

    def test_ep_pairs(self):
        res = ex.find_exceptional_points(pair_model(), nk=20)
        exact = np.array([[a, b] for a in (0., np.pi) for b in (KY0, -KY0, np.pi - KY0, KY0 - np.pi)])
        self.assertEqual(len(res.k), 8)
        for p, q in zip(res.k, res.charge):
            dist = np.linalg.norm(wrap(p - exact), axis=1)
            self.assertLess(dist.min(), 1e-8)
            # charge -1 at (0, asin g) and (pi, -asin g), see the module docstring
            ref = exact[np.argmin(dist)]
            expected = -1 if np.sign(np.cos(ref[0])) * np.sign(np.sin(ref[1])) * np.cos(ref[1]) > 0 else 1
            self.assertEqual(q, expected)
        self.assertTrue(np.all(res.order == 2))
        self.assertTrue(np.all(res.petermann > 1e8))
        self.assertTrue(np.allclose(res.energy, 0., atol=1e-6))
        self.assertEqual(res.total_charge, 0)
        self.assertEqual(res.lines.shape, (0, 2))

    def test_exceptional_rings(self):
        res = ex.find_exceptional_points(ring_model(), nk=20)
        self.assertEqual(len(res.k), 0)
        self.assertEqual(res.total_charge, 0)
        self.assertGreater(len(res.lines), 20)
        radius2 = np.sin(res.lines[:, 0]) ** 2 + np.sin(res.lines[:, 1]) ** 2
        self.assertTrue(np.allclose(radius2, G ** 2, atol=1e-12))
        # staggered gain/loss on graphene: rings |f(k)| = gamma around K and K'
        gra = graphene()
        gra.set_onsite({'a': 0.2j, 'b': -0.2j})
        res = ex.find_exceptional_points(gra, nk=30)
        for k in res.lines:
            self.assertAlmostEqual(abs(gra.get_ham(k)[0, 1]), 0.2, places=10)

    def test_hermitian_has_no_ep(self):
        res = ex.find_exceptional_points(graphene(), nk=20)
        self.assertEqual((len(res.k), len(res.lines)), (0, 0))
        self.assertEqual(len(ex.fermi_arcs(graphene(), nk=10)), 0)

    def test_third_order(self):
        res = ex.find_exceptional_points(ep3_model(), nk=10)
        exact = np.array([[a, b] for a in (0., np.pi) for b in (0., np.pi)])
        self.assertEqual(len(res.k), 4)
        for p, q in zip(res.k, res.charge):
            dist = np.linalg.norm(wrap(p - exact), axis=1)
            self.assertLess(dist.min(), 1e-8)
            ref = exact[np.argmin(dist)]
            self.assertEqual(q, 2 * int(np.cos(ref[0]) * np.cos(ref[1])))
        self.assertTrue(np.all(res.order == 3))
        self.assertEqual(res.total_charge, 0)

    def test_order_and_petermann(self):
        self.assertEqual(ex._order(np.diag([1., 2.]))[0], 1)
        en, kp = ex.petermann_factors(pair_model(), (0.5, 0.))
        self.assertEqual(len(en), 2)
        self.assertTrue(np.all(kp >= 1.))
        # K ~ 1/|dk| near the EP
        _, k1 = ex.petermann_factors(pair_model(), (1e-4, KY0))
        _, k2 = ex.petermann_factors(pair_model(), (1e-6, KY0))
        self.assertAlmostEqual(np.log10(k2[0] / k1[0]), 2., places=2)
        _, kh = ex.petermann_factors(graphene(), (0.3, 0.1))
        self.assertTrue(np.allclose(kh, 1.))

    def test_fermi_arcs(self):
        arcs = ex.fermi_arcs(pair_model(), nk=30)
        self.assertGreater(len(arcs), 4)
        self.assertLess(np.max(np.abs(np.sin(arcs[:, 0]))), 1e-10)
        self.assertLess(np.max(np.abs(np.sin(arcs[:, 1]))), G)
        # none inside an exceptional ring (the real parts coincide on an area)
        self.assertEqual(len(ex.fermi_arcs(ring_model(), nk=10)), 0)

    def test_checks(self):
        self.assertRaises(TypeError, ex.find_exceptional_points, pair_model, 10)
        self.assertRaises(ValueError, ex.find_exceptional_points, KSpace(lattices.chain()))
        self.assertRaises(TypeError, ex.find_exceptional_points, pair_model(), 1.5)
        self.assertRaises(ValueError, ex.find_exceptional_points, pair_model(), 10, 0.)
        self.assertRaises(ValueError, ex.find_exceptional_points, pair_model(), 10, 1e-12, 0)
        empty = KSpace(lattices.square(), spin=True)  # degenerate everywhere
        self.assertRaises(ValueError, ex.find_exceptional_points, empty)
        self.assertRaises(TypeError, ex.fermi_arcs, 0)
        self.assertRaises(TypeError, ex.fermi_arcs, pair_model(), 1.5)


class TestEncircling(unittest.TestCase):

    def test_ep_four_loops(self):
        lp = ex.circle((0., KY0), 0.1)
        one = ex.encircle(pair_model(), lp, band=0, n_loops=1, nk=400)
        self.assertEqual(one.final_band, 1)  # the states swap
        # the transported vector is the other eigenvector
        h = pair_model().get_ham(lp(0.))
        v = one.vector[-1]
        self.assertTrue(np.allclose(h @ v, one.en[0, 1] * v, atol=1e-10))
        two = ex.encircle(pair_model(), lp, band=0, n_loops=2, nk=400)
        self.assertEqual(two.final_band, 0)
        self.assertAlmostEqual(abs(two.phase), np.pi, places=10)
        self.assertAlmostEqual(abs(two.overlap), 1., places=10)
        four = ex.encircle(pair_model(), lp, band=1, n_loops=4, nk=400)
        self.assertEqual(four.final_band, 1)
        self.assertAlmostEqual(four.phase, 0., places=10)
        self.assertEqual(four.vector.shape, (len(four.s), 2))
        self.assertTrue(np.allclose(four.k[0], lp(0.)))

    def test_dp_berry_phase(self):
        gra = graphene()
        for k0 in (k_point(gra), -k_point(gra)):
            for band in (0, 1):
                res = ex.encircle(gra, ex.circle(k0, 0.2), band=band, nk=400)
                self.assertEqual(res.final_band, band)  # no swap at a DP
                self.assertAlmostEqual(abs(res.phase), np.pi, places=10)
                self.assertAlmostEqual(abs(res.overlap), 1., places=10)  # norm kept
        # a loop enclosing no degeneracy: phase 0
        res = ex.encircle(gra, ex.circle((0., 0.), 0.3), nk=200)
        self.assertAlmostEqual(res.phase, 0., places=10)
        # agrees with the KSpace Berry phase convention: same sign as a
        # Wilson loop along the same small circle
        lp = ex.circle(k_point(gra), 0.2)
        vs = [np.linalg.eigh(gra.get_ham(lp(s)))[1][:, 0] for s in np.arange(400) / 400]
        wilson = np.prod([vs[m].conj() @ vs[(m + 1) % 400] for m in range(400)])
        self.assertAlmostEqual(np.cos(-np.angle(wilson) - ex.encircle(gra, lp, nk=400).phase), 1.,
                                         places=6)

    def test_steps_follow_the_eigenvectors(self):
        # constant eigenvalues +-1, eigenvectors turning by half the polar
        # angle: only the eigenvector check refines the two coarse steps
        def cone(p):
            th = np.arctan2(p[1], p[0])
            return np.array([[np.cos(th), np.sin(th)], [np.sin(th), -np.cos(th)]])
        res = ex.encircle(cone, ex.circle((0., 0.), 1.), nk=2)
        self.assertGreater(len(res.s), 3)  # 3 points without refinement
        self.assertEqual(res.final_band, 0)
        self.assertAlmostEqual(abs(res.phase), np.pi, places=10)

    def test_checks(self):
        lp = ex.circle((0., KY0), 0.1)
        self.assertRaises(ValueError, ex.encircle, pair_model(), lp, 2)
        self.assertRaises(TypeError, ex.encircle, pair_model(), lp, 0.)
        self.assertRaises(ValueError, ex.encircle, pair_model(), lp, 0, 0)


class TestBiorthogonalChern(unittest.TestCase):

    def test_hermitian_limit(self):
        hal = haldane()
        c = hal.chern_number(0, 20)
        self.assertAlmostEqual(c, 1., places=8)
        for kind in ('LR', 'RL', 'RR', 'LL'):
            self.assertAlmostEqual(hal.biorthogonal_chern_number(0, 20, kind=kind), c, places=8)
        self.assertTrue(np.allclose(hal.biorthogonal_berry_curvature([0], 20), hal.berry_curvature(0, 20)))

    def test_four_definitions_agree_until_the_gap_closes(self):
        for gamma in (0.3, 0.8):
            hal = haldane(gamma)
            for kind in ('LR', 'RL', 'RR', 'LL'):
                self.assertAlmostEqual(hal.biorthogonal_chern_number([0], (20, 21), kind=kind), 1.,
                                                 places=8)
            # the right-eigenvector result of chern_number is unchanged
            self.assertAlmostEqual(hal.chern_number(0, 20), 1., places=8)
            self.assertEqual(len(ex.find_exceptional_points(hal, nk=30).k), 0)
        # past the closing of the real line gap: EPs, and no Chern number
        hal = haldane(1.5)
        self.assertGreater(len(ex.find_exceptional_points(hal, nk=30).k), 0)
        self.assertRaises(ValueError, hal.biorthogonal_chern_number, 0, 20)
        self.assertRaises(ValueError, hal.chern_number, 0, 20)

    def test_imaginary_line_gap(self):
        # i H: same eigenvectors as H, an imaginary line gap, all Re E = 0.
        # Sorting by Re E scrambled the bands (chern_number gave -3, -7 or 5
        # depending on nk); it now refuses, and the imaginary gap gives C = 1.
        hal = haldane()
        ih = KSpace(lattices.honeycomb())
        ih.set_hopping([{'i': i, 'j': j, 'R': R, 't': 1j * t} for i, j, R, t in hal._hop_cells()],
                               hermitian=False)
        self.assertRaises(ValueError, ih.chern_number, 0, 20)
        self.assertAlmostEqual(ih.biorthogonal_chern_number(0, 20, gap='imaginary'), 1., places=8)
        self.assertRaises(ValueError, ih.biorthogonal_chern_number, 0, 20)

    def test_checks(self):
        hal = haldane(0.3)
        self.assertRaises(ValueError, hal.biorthogonal_chern_number, 0, 10, 'XY')
        self.assertRaises(ValueError, hal.biorthogonal_chern_number, 0, 10, 'LR', 'complex')
        self.assertRaises(ValueError, KSpace(lattices.chain()).biorthogonal_chern_number, 0)
        self.assertRaises(ValueError, hal.biorthogonal_chern_number, [0, 0])
        self.assertRaises(ValueError, hal.biorthogonal_chern_number, 0, 10, 'LR', 'real', (0, 0))
        self.assertRaises(TypeError, hal.biorthogonal_chern_number, 0, 10, 'LR', 'real', (0, 1), 'a')
        hal.set_overlap([{'i': 0, 'j': 1, 'R': (0, 0), 't': 0.1}])
        self.assertRaises(ValueError, hal.biorthogonal_chern_number, 0, 10)


if __name__ == '__main__':
    unittest.main()
