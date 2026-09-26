'''
Neighbour-order hoppings in k-space: KSpace.set_hopping (and set_overlap)
accept the ('n', 'ang', 'tag', 't') selectors of System.set_hopping.
'''
import unittest
from math import sqrt

import numpy as np

from tbkit import lattices
from tbkit.kspace import KSpace, PAULI
from tbkit.lattice import Lattice
from tbkit.system import System
from tbkit.neighbours import neighbour_bonds, neighbour_shells, neighbour_hoppings


RNG = np.random.default_rng(3)
KS2 = RNG.uniform(-4, 4, (7, 2))


def graphene_f(k):
    '''f(k) = sum over the three nearest-neighbour bond vectors of exp(i k.d), periodic gauge.'''
    a1 = np.array([sqrt(3), 0.])
    a2 = np.array([0.5 * sqrt(3), 1.5])
    return 1 + np.exp(-1j * k @ a1) + np.exp(-1j * k @ a2)


class TestExplicitFormUnchanged(unittest.TestCase):
    '''The explicit {'i', 'j', 'R', 't'} form keeps its exact behaviour.'''

    def test_stored_hoppings_pinned(self):
        lat = lattices.honeycomb()
        gra = KSpace(lat)
        gra.set_hopping([{'i': 0, 'j': 1, 'R': (0, 0), 't': 1.},
                                {'i': 0, 'j': 1, 'R': (-1, 0), 't': 2j}])
        self.assertEqual(len(gra._hop), 4)
        a1 = np.array(lat.prim_vec[0])
        expected = [(0, 1, np.zeros(2), 1.), (1, 0, np.zeros(2), 1.),
                          (0, 1, -a1, 2j), (1, 0, a1, -2j)]
        for (i, j, R, t), (ei, ej, eR, et) in zip(gra._hop, expected):
            self.assertEqual((i, j), (ei, ej))
            np.testing.assert_array_equal(R, eR)
            self.assertEqual(t, et)
        self.assertFalse(gra._nonreciprocal)
        gra.set_hopping([{'i': 0, 'j': 0, 'R': (1, 0), 't': 0.5}], hermitian=False)
        self.assertEqual(len(gra._hop), 5)
        self.assertTrue(gra._nonreciprocal)

    def test_explicit_graphene_bands(self):
        gra = KSpace(lattices.honeycomb())
        gra.set_hopping([{'i': 0, 'j': 1, 'R': (0, 0), 't': 1.},
                                {'i': 0, 'j': 1, 'R': (-1, 0), 't': 1.},
                                {'i': 0, 'j': 1, 'R': (0, -1), 't': 1.}])
        for k in KS2:
            np.testing.assert_allclose(gra.get_ham(k)[0, 1], graphene_f(k), atol=1e-12)

    def test_empty_list_is_explicit(self):
        gra = KSpace(lattices.honeycomb())
        gra.set_hopping([])
        self.assertEqual(gra._hop, [])


class TestShells(unittest.TestCase):

    def test_honeycomb_shells(self):
        np.testing.assert_allclose(neighbour_shells(lattices.honeycomb(), 4),
                                               [1., sqrt(3), 2., sqrt(7)], atol=1e-4)

    def test_coordination_numbers(self):
        # honeycomb: 3, 6, 3, 6 neighbours per site; square: 4, 4, 4, 8
        bonds = neighbour_bonds(lattices.honeycomb(), 4)
        per_site = [2 * np.sum(bonds['n'] == n) / 2 for n in range(1, 5)]
        self.assertEqual(per_site, [3, 6, 3, 6])
        bonds = neighbour_bonds(lattices.square(), 4)
        self.assertEqual([2 * np.sum(bonds['n'] == n) for n in range(1, 5)], [4, 4, 4, 8])
        # fcc (3D): 12 nearest and 6 next-nearest neighbours
        fcc = Lattice([{'tag': 'a', 'r0': (0., 0., 0.)}],
                             [(0., .5, .5), (.5, 0., .5), (.5, .5, 0.)])
        bonds = neighbour_bonds(fcc, 2)
        self.assertEqual([2 * np.sum(bonds['n'] == n) for n in (1, 2)], [12, 6])

    def test_bonds_match_system(self):
        '''The bulk bonds of a finite System and the periodic bonds agree:
        same angles and tags per shell.'''
        lat = lattices.honeycomb()
        bonds = neighbour_bonds(lat, 2)
        lat_f = lattices.honeycomb()
        lat_f.get_lattice(8, 8)
        sys = System(lat_f)
        sys.set_hopping([{'n': 1, 't': 1.}, {'n': 2, 't': 1.}])
        for n in (1, 2):
            per = {(round(float(a), 3), str(t)) for a, t in zip(bonds['ang'][bonds['n'] == n],
                                                                                    bonds['tag'][bonds['n'] == n])}
            fin = {(round(float(a), 3), str(t)) for a, t in zip(sys.hop['ang'][sys.hop['n'] == n],
                                                                                    sys.hop['tag'][sys.hop['n'] == n])}
            self.assertEqual(per, fin)


class TestNeighbourHoppings(unittest.TestCase):

    def test_graphene_nearest_equals_explicit(self):
        explicit = KSpace(lattices.honeycomb())
        explicit.set_hopping([{'i': 0, 'j': 1, 'R': (0, 0), 't': -2.7},
                                     {'i': 0, 'j': 1, 'R': (-1, 0), 't': -2.7},
                                     {'i': 0, 'j': 1, 'R': (0, -1), 't': -2.7}])
        by_n = KSpace(lattices.honeycomb())
        by_n.set_hopping([{'n': 1, 't': -2.7}])
        for k in KS2:
            np.testing.assert_allclose(by_n.get_ham(k), explicit.get_ham(k), atol=1e-12)

    def test_graphene_second_neighbours_analytic(self):
        # E = t2 (|f|^2 - 3) +- t |f|  (Castro Neto et al., RMP 81, 109, eq. 6)
        t, t2 = -2.7, 0.2
        gra = KSpace(lattices.honeycomb())
        gra.set_hopping([{'n': 1, 't': t}, {'n': 2, 't': t2}])
        for k in KS2:
            f = abs(graphene_f(k))
            exact = np.sort([t2 * (f**2 - 3) + t * f, t2 * (f**2 - 3) - t * f])
            np.testing.assert_allclose(gra.get_bands(k), [exact], atol=1e-12)

    def test_square_three_shells_analytic(self):
        t1, t2, t3 = 1., 0.3, -0.1
        sq = KSpace(lattices.square())
        sq.set_hopping([{'n': 1, 't': t1}, {'n': 2, 't': t2}, {'n': 3, 't': t3}])
        for kx, ky in KS2:
            exact = (2 * t1 * (np.cos(kx) + np.cos(ky)) + 4 * t2 * np.cos(kx) * np.cos(ky)
                        + 2 * t3 * (np.cos(2 * kx) + np.cos(2 * ky)))
            self.assertAlmostEqual(sq.get_ham((kx, ky))[0, 0].real, exact, places=12)

    def test_kagome_flat_band(self):
        t = 1.
        kag = KSpace(lattices.kagome())
        kag.set_hopping([{'n': 1, 't': t}])
        en = kag.mesh_bands(12)
        np.testing.assert_allclose(en[:, 0], -2 * t, atol=1e-12)
        self.assertAlmostEqual(en[:, 2].max(), 4 * t, places=12)

    def test_angle_selection_and_last_wins(self):
        tx, ty = 1., 0.4
        sq = KSpace(lattices.square())
        sq.set_hopping([{'n': 1, 't': ty}, {'n': 1, 'ang': 0., 't': tx}])
        for kx, ky in KS2:
            exact = 2 * tx * np.cos(kx) + 2 * ty * np.cos(ky)
            self.assertAlmostEqual(sq.get_ham((kx, ky))[0, 0].real, exact, places=12)

    def test_tag_selection_as_system(self):
        ''''tag': 'ab' picks the two bonds oriented from a to b (30 and 150
        degrees), as System.set_hopping does; 'ba' the vertical one.'''
        hops = neighbour_hoppings(lattices.honeycomb(), [{'n': 1, 'tag': 'ab', 't': 1.}])
        self.assertEqual(sorted(h['R'] for h in hops), [(-1, 0), (0, 0)])
        hops = neighbour_hoppings(lattices.honeycomb(), [{'n': 1, 'tag': 'ba', 't': 2.}])
        self.assertEqual(hops, [{'i': 1, 'j': 0, 'R': (0, 1), 't': 2.}])
        # angle and tag together
        hops = neighbour_hoppings(lattices.honeycomb(), [{'n': 1, 'ang': 30., 'tag': 'ab', 't': 3.}])
        self.assertEqual(hops, [{'i': 0, 'j': 1, 'R': (0, 0), 't': 3.}])

    def test_negative_angle(self):
        '''Hatano-Nelson chain: H(k) = tR e^{ik} + tL e^{-ik}.'''
        tr, tl = 1.2, 0.5
        hn = KSpace(lattices.chain())
        hn.set_hopping([{'n': 1, 'ang': 0., 't': tr}, {'n': 1, 'ang': -180., 't': tl}], hermitian=False)
        self.assertFalse(hn.is_hermitian())
        for k in (0.3, 1.7, -2.2):
            self.assertAlmostEqual(complex(hn.get_ham((k,))[0, 0]),
                                          tr * np.exp(1j * k) + tl * np.exp(-1j * k), places=12)
        # Hermitian: a hopping along the reversed bond is the conjugate
        ch = KSpace(lattices.chain())
        ch.set_hopping([{'n': 1, 'ang': -180., 't': 1j}])
        # H(k) = 1j e^{-ik} + (-1j) e^{ik} = 2 sin k
        self.assertAlmostEqual(complex(ch.get_ham((0.4,))[0, 0]), 2 * np.sin(0.4), places=12)
        # reversed tag: 'ba' along -150 degrees is the 'ab' bond at 30 degrees
        hops = neighbour_hoppings(lattices.honeycomb(), [{'n': 1, 'ang': -150., 'tag': 'ba', 't': 1j}])
        self.assertEqual(hops, [{'i': 0, 'j': 1, 'R': (0, 0), 't': -1j}])
        hops = neighbour_hoppings(lattices.honeycomb(), [{'n': 1, 'ang': -150., 'tag': 'ba', 't': 1j}],
                                               hermitian=False)
        self.assertEqual(hops, [{'i': 1, 'j': 0, 'R': (0, 0), 't': 1j}])

    def test_chain_second_neighbours(self):
        ch = KSpace(lattices.chain())
        ch.set_hopping([{'n': 1, 't': 1.}, {'n': 2, 't': 0.25}])
        for k in (0.1, 2.):
            self.assertAlmostEqual(ch.get_ham((k,))[0, 0].real, 2 * np.cos(k) + 0.5 * np.cos(2 * k), places=12)

    def test_cubic_3d(self):
        cub = KSpace(Lattice([{'tag': 'a', 'r0': (0., 0., 0.)}],
                                        [(1., 0., 0.), (0., 1., 0.), (0., 0., 1.)]))
        cub.set_hopping([{'n': 1, 't': 1.}])
        k = np.array([0.3, -1.2, 2.1])
        self.assertAlmostEqual(cub.get_ham(k)[0, 0].real, 2 * np.cos(k).sum(), places=12)
        # 3D angles: the +x and the vertical (+z) bonds both have angle 0
        cub.clear_hopping()
        cub.set_hopping([{'n': 1, 'ang': 0., 't': 1.}])
        self.assertAlmostEqual(cub.get_ham(k)[0, 0].real, 2 * (np.cos(k[0]) + np.cos(k[2])), places=12)

    def test_spin(self):
        lam = 0.3
        by_n = KSpace(lattices.chain(), spin=True)
        by_n.set_hopping([{'n': 1, 't': 1.}, {'n': 1, 'ang': 0., 't': PAULI['0'] + 1j * lam * PAULI['y']}])
        explicit = KSpace(lattices.chain(), spin=True)
        explicit.set_hopping([{'i': 0, 'j': 0, 'R': (1,), 't': PAULI['0'] + 1j * lam * PAULI['y']}])
        for k in (0.2, 1.3):
            np.testing.assert_allclose(by_n.get_ham((k,)), explicit.get_ham((k,)), atol=1e-12)
        # spinful reversed bond: the conjugate transpose of the 2x2 block
        rev = KSpace(lattices.chain(), spin=True)
        rev.set_hopping([{'n': 1, 'ang': -180., 't': (PAULI['0'] + 1j * lam * PAULI['y']).conj().T}])
        np.testing.assert_allclose(rev.get_ham((0.7,)), explicit.get_ham((0.7,)), atol=1e-12)

    def test_overlap_by_neighbour_order(self):
        s = 0.1
        by_n = KSpace(lattices.honeycomb())
        by_n.set_hopping([{'n': 1, 't': -2.7}])
        by_n.set_overlap([{'n': 1, 't': s}])
        explicit = KSpace(lattices.honeycomb())
        explicit.set_overlap([{'i': 0, 'j': 1, 'R': R, 't': s} for R in [(0, 0), (-1, 0), (0, -1)]])
        for k in KS2:
            np.testing.assert_allclose(by_n.get_overlap(k), explicit.get_overlap(k), atol=1e-12)
        # E = -t|f| / (1 - s|f|) ... (1 + s|f|): the known asymmetric pi bands
        k = KS2[0]
        f = abs(graphene_f(k))
        np.testing.assert_allclose(by_n.get_bands(k)[0], np.sort([-2.7 * f / (1 + s * f), 2.7 * f / (1 - s * f)]),
                                               atol=1e-12)

    def test_ribbon_accepts_explicit_list_from_neighbours(self):
        from tbkit.kspace import ribbon
        lat = lattices.honeycomb()
        rib = ribbon(lat, neighbour_hoppings(lat, [{'n': 1, 't': 1.}]), width=6)
        # zigzag ribbon: flat zero-energy edge band at k = pi / |a1|
        en = rib.get_bands((np.pi / sqrt(3),))[0]
        self.assertEqual(int(np.sum(np.abs(en) < 1e-9)), 2)


if __name__ == '__main__':
    unittest.main()
