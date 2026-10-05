"""
Sector, spin and mirror Chern numbers, the four 3D Z2 indices, the
entanglement spectrum, the Pfaffian and Kitaev's Majorana number, the Bott
index, and the Weyl point finder.
"""
import unittest

import numpy as np

import tbkit.error_handling as error_handling
import tbkit.lattices as lattices
from tbkit.bdg import bdg_kspace, majorana_number, pfaffian
from tbkit.higher_order import flake_positions
from tbkit.kspace import KSpace, PAULI
from tbkit.lattice import Lattice
from tbkit.moire import supercell
from tbkit.topology import WeylPoints, bott_index, entanglement_spectrum, find_weyl_points
from tests.test_three_d import CUBIC, weyl
from tests.test_topology import haldane, kane_mele, ssh

S_Z = np.kron(np.eye(2), PAULI['z'])
SIGMA = [PAULI['x'], PAULI['y'], PAULI['z']]
# z -> -z on the s, p model below: s even, p_z odd, i sigma_z on spin
MIRROR = 1j * np.kron(np.diag([1., -1.]), PAULI['z'])


def cubic_ti(M):
    '''
    H = m(k) tau_z + sum_i sin k_i tau_x sigma_i, m = M - 2 sum_i (1 - cos k_i):
    (0; 000) for M < 0, (1; 000) for 0 < M < 4, (0; 111) for 4 < M < 8.
    '''
    lat = Lattice(unit_cell=[{'tag': 's', 'r0': (0., 0., 0.)}, {'tag': 'p', 'r0': (0., 0., 0.)}],
                         prim_vec=CUBIC)
    ks = KSpace(lat, spin=True)
    hops = []
    for d in range(3):
        R = tuple(int(x) for x in np.eye(3, dtype=int)[d])
        hops += [{'i': 0, 'j': 0, 'R': R, 't': PAULI['0']}, {'i': 1, 'j': 1, 'R': R, 't': -PAULI['0']},
                     {'i': 0, 'j': 1, 'R': R, 't': -0.5j * SIGMA[d]},
                     {'i': 1, 'j': 0, 'R': R, 't': -0.5j * SIGMA[d]}]
    ks.set_hopping(hops)
    ks.set_onsite({'s': (M - 6) * PAULI['0'], 'p': -(M - 6) * PAULI['0']})
    return ks


def kitaev(mu, t=1., delta=0.6):
    chain = KSpace(lattices.chain())
    chain.set_hopping([{'i': 0, 'j': 0, 'R': (1,), 't': t}])
    return bdg_kspace(chain, [{'i': 0, 'j': 0, 'R': (1,), 'delta': delta}], mu=mu)


def p_wave_square(mu):
    '''Spinless p + ip superconductor: C = +-1 for 0 < |mu| < 4.'''
    sq = KSpace(lattices.square())
    sq.set_hopping([{'i': 0, 'j': 0, 'R': (1, 0), 't': -1.}, {'i': 0, 'j': 0, 'R': (0, 1), 't': -1.}])
    return bdg_kspace(sq, [{'i': 0, 'j': 0, 'R': (1, 0), 'delta': 0.5},
                                  {'i': 0, 'j': 0, 'R': (0, 1), 'delta': 0.5j}], mu=mu)


class TestSectorChernNumbers(unittest.TestCase):

    def test_kane_mele_spin_sectors(self):
        # s_z conserved: the sectors are the two spins, C = +-1
        c_plus, c_minus = kane_mele().sector_chern_numbers(S_Z, [0, 1])
        self.assertAlmostEqual(c_plus, 1.)
        self.assertAlmostEqual(c_minus, -1.)
        # the same operator as a function of k
        self.assertEqual(kane_mele().sector_chern_numbers(lambda k: S_Z, [0, 1], nk=12),
                              kane_mele().sector_chern_numbers(S_Z, [0, 1], nk=12))
        # the identity puts every band in the + sector
        c_plus, c_minus = haldane().sector_chern_numbers(np.eye(2), 0)
        self.assertAlmostEqual(c_plus, 1.)
        self.assertAlmostEqual(c_minus, 0.)

    def test_spin_chern_number(self):
        self.assertAlmostEqual(kane_mele().spin_chern_number([0, 1]), 1.)
        self.assertAlmostEqual(kane_mele(M=0.5).spin_chern_number([0, 1]), 0.)
        # Rashba coupling breaks s_z conservation; C_s stays 1 while the
        # bulk gap is open, and agrees with Z2
        self.assertAlmostEqual(kane_mele(M=0.1, rashba=0.1).spin_chern_number([0, 1]), 1.)
        self.assertAlmostEqual(kane_mele(M=0.1, rashba=0.3).spin_chern_number([0, 1]), 0., places=6)
        # an explicit operator: -s_z flips the sign
        self.assertAlmostEqual(kane_mele().spin_chern_number([0, 1], s_z=-S_Z), -1.)

    def test_mirror_chern_number(self):
        # nu = n_M mod 2 on the mirror planes k_z = 0, pi
        self.assertAlmostEqual(cubic_ti(2.).mirror_chern_number(MIRROR, [0, 1]), 1.)
        self.assertAlmostEqual(cubic_ti(2.).mirror_chern_number(MIRROR, [0, 1], k_fixed=0.5), 0., places=8)
        self.assertAlmostEqual(cubic_ti(-2.).mirror_chern_number(MIRROR, [0, 1]), 0., places=8)
        # M^2 = +1 (the Hermitian form), and a callable
        self.assertAlmostEqual(cubic_ti(2.).mirror_chern_number(-1j * MIRROR, [0, 1], nk=12),
                                        cubic_ti(2.).mirror_chern_number(lambda k: MIRROR, [0, 1], nk=12))
        # in 2D: Kane-Mele is invariant under z -> -z, M = i sigma_z on spin
        self.assertAlmostEqual(kane_mele().mirror_chern_number(1j * S_Z, [0, 1]), 1.)

    def test_checks(self):
        km = kane_mele()
        self.assertRaises(ValueError, km.sector_chern_numbers, np.zeros((4, 4)), [0, 1])  # no gap
        self.assertRaises(ValueError, km.sector_chern_numbers, np.triu(np.ones((4, 4))), [0, 1])
        self.assertRaises(ValueError, km.sector_chern_numbers, np.eye(3), [0, 1])
        self.assertRaises(ValueError, ssh(1., 0.5).sector_chern_numbers, np.eye(2), 0)  # 1D
        self.assertRaises(ValueError, haldane().spin_chern_number, [0])  # no spin
        ti = cubic_ti(2.)
        self.assertRaises(ValueError, ti.mirror_chern_number, MIRROR, [0, 1], k_fixed=0.3)  # not invariant
        self.assertRaises(ValueError, ti.mirror_chern_number, 2 * MIRROR, [0, 1])  # M^2 = -4
        self.assertRaises(ValueError, ti.mirror_chern_number, np.zeros((4, 4)), [0, 1])
        self.assertRaises(ValueError, ti.mirror_chern_number, MIRROR, [0, 1], tol=0.)
        nh = haldane()
        nh.set_onsite({'a': 0.1j, 'b': -0.1j})
        self.assertRaises(ValueError, nh.sector_chern_numbers, np.eye(2), 0)


class TestZ2Indices3D(unittest.TestCase):

    def test_phases(self):
        for M, indices in ((-2., (0, 0, 0, 0)), (2., (1, 0, 0, 0)), (6., (0, 1, 1, 1)),
                                  (10., (1, 1, 1, 1))):
            self.assertEqual(cubic_ti(M).z2_indices_3d([0, 1], nk=40, nk_perp=21), indices)

    def test_checks(self):
        self.assertRaises(ValueError, kane_mele().z2_indices_3d, [0, 1])
        self.assertRaises(ValueError, cubic_ti(2.).z2_indices_3d, [0])
        self.assertRaises(ValueError, error_handling.strong_index, np.array([0, 1, 0]))
        error_handling.strong_index(np.array([1, 1, 1]))


class TestEntanglementSpectrum(unittest.TestCase):

    def test_ssh_ring(self):
        n = 30
        topo = entanglement_spectrum(ssh(0.4, 1.).finite_ham(n, periodic=True), list(range(n)))
        triv = entanglement_spectrum(ssh(1., 0.4).finite_ham(n, periodic=True), list(range(n)))
        self.assertEqual(topo.shape, (n,))
        self.assertTrue(np.all((topo > -1e-12) & (topo < 1 + 1e-12)))
        self.assertEqual(int(np.sum(np.abs(topo - 0.5) < 1e-8)), 2)  # one mode per cut
        self.assertGreater(np.min(np.abs(triv - 0.5)), 0.4)
        # a sparse Hamiltonian, and the whole sample (pure state: xi = 0 or 1)
        whole = entanglement_spectrum(ssh(0.4, 1.).finite_ham(n, periodic=True, sparse=True),
                                                    list(range(2 * n)))
        self.assertTrue(np.allclose(np.sort(whole), [0.] * n + [1.] * n, atol=1e-10))

    def test_k_resolved(self):
        sc = supercell(haldane(), [[1, 0], [0, 8]])
        k = np.linspace(-0.5, 0.5, 81)[:, None] * sc.rec_vec_k[0][None]
        xi = sc.entanglement_spectrum(k, list(range(8)), list(range(8)))
        self.assertEqual(xi.shape, (81, 8))
        self.assertTrue(np.all(np.diff(xi, axis=1) >= 0.))
        # spectral flow: some mode crosses 1/2
        self.assertLess(np.min(np.abs(xi - 0.5)), 1e-8)
        sc_triv = supercell(haldane(M=2.), [[1, 0], [0, 8]])
        xi_triv = sc_triv.entanglement_spectrum(k, list(range(8)), list(range(8)))
        self.assertGreater(np.min(np.abs(xi_triv - 0.5)), 0.3)
        # a single k-point and a single band index
        self.assertEqual(haldane().entanglement_spectrum([0.1, 0.2], [0], 0).shape, (1, 1))

    def test_checks(self):
        ham = ssh(1., 0.5).finite_ham(4)
        self.assertRaises(TypeError, entanglement_spectrum, ham, 0)
        self.assertRaises(TypeError, entanglement_spectrum, ham, [])
        self.assertRaises(TypeError, entanglement_spectrum, ham, [0.5])
        self.assertRaises(ValueError, entanglement_spectrum, ham, [0, 0])
        self.assertRaises(ValueError, entanglement_spectrum, ham, [8])
        self.assertRaises(ValueError, entanglement_spectrum, np.ones((2, 3)), [0])
        self.assertRaises(ValueError, entanglement_spectrum, np.triu(np.ones((3, 3))), [0])
        self.assertRaises(TypeError, entanglement_spectrum, ham, [0], 'a')
        self.assertRaises(ValueError, haldane().entanglement_spectrum, [[0.1, 0.2]], [2], [0])
        self.assertRaises(ValueError, haldane().entanglement_spectrum, [[0.1]], [0], [0])


class TestBdGTopology(unittest.TestCase):

    def test_pfaffian(self):
        rng = np.random.default_rng(3)
        for n in (2, 4, 6, 8):
            a = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
            a = a - a.T
            det = np.linalg.det(a)
            self.assertAlmostEqual(abs(pfaffian(a) ** 2 - det) / abs(det), 0., places=10)
        self.assertEqual(pfaffian([[0., 2.], [-2., 0.]]), 2.)
        self.assertIsInstance(pfaffian(np.array([[0., 2.], [-2., 0.]])), float)
        self.assertIsInstance(pfaffian(np.array([[0., 2j], [-2j, 0.]])), complex)
        # Pf of a block-diagonal matrix: the product of the blocks
        block = np.array([[0., 3.], [-3., 0.]])
        self.assertAlmostEqual(pfaffian(np.kron(np.eye(2), block)), 9.)
        # pivoting: the first column's largest entry is not on the subdiagonal
        a = np.zeros((4, 4))
        a[0, 1], a[0, 2], a[1, 3], a[2, 3] = 1., 5., 2., 7.
        a = a - a.T
        self.assertAlmostEqual(pfaffian(a), 1. * 7. - 5. * 2.)
        self.assertEqual(pfaffian(np.zeros((3, 3))), 0.)  # odd dimension
        self.assertEqual(pfaffian(np.zeros((4, 4))), 0.)  # singular
        self.assertRaises(ValueError, pfaffian, np.ones((2, 2)))
        self.assertRaises(ValueError, pfaffian, np.ones((2, 3)))

    def test_majorana_number(self):
        for mu in (-3., -1., 0.5, 1.5, 3.):
            self.assertEqual(majorana_number(kitaev(mu)), -1 if abs(mu) < 2. else 1)
        # 2D: the parity of the Chern number
        for mu in (-5., -2., 2., 5.):
            ks = p_wave_square(mu)
            chern = int(np.round(ks.chern_number(0, 30)))
            self.assertEqual(majorana_number(ks), (-1) ** chern)
            self.assertEqual(abs(chern), int(abs(mu) < 4.))

    def test_checks(self):
        self.assertRaises(TypeError, majorana_number, 'kitaev')
        lat = Lattice(unit_cell=[{'tag': t, 'r0': (0., 0.)} for t in 'abc'], prim_vec=[(1., 0.)])
        self.assertRaises(ValueError, majorana_number, KSpace(lat))  # odd number of orbitals
        chain = KSpace(Lattice(unit_cell=[{'tag': t, 'r0': (0., 0.)} for t in 'ab'], prim_vec=[(1., 0.)]))
        chain.set_onsite({'a': 1., 'b': 1.})  # not particle-hole symmetric
        self.assertRaises(ValueError, majorana_number, chain)
        self.assertRaises(ValueError, majorana_number, kitaev(2.))  # the gap closes at k = 0


class TestBottIndex(unittest.TestCase):

    def _torus(self, ks, n=8):
        ham = ks.finite_ham((n, n), periodic=True)
        cell = [n * np.array(a) for a in ks.lat.prim_vec]
        return ham, flake_positions(ks, (n, n)), cell

    def test_chern_number(self):
        for t2, M, chern in ((0.2, 0., 1), (-0.2, 0., -1), (0.2, 1.5, 0)):
            ham, pos, cell = self._torus(haldane(t2, M))
            self.assertAlmostEqual(bott_index(ham, pos, cell), chern, places=8)
        # with weak disorder, still 1; positions in 3D (z = 0) are fine
        ham, pos, cell = self._torus(haldane())
        rng = np.random.default_rng(0)
        dis = ham + np.diag(rng.uniform(-1., 1., len(ham)))
        pos3 = np.column_stack([pos, np.zeros(len(pos))])
        cell3 = np.column_stack([cell, np.zeros(2)])
        self.assertAlmostEqual(bott_index(dis, pos3, cell3), 1., places=8)
        # strong disorder localizes everything
        dis = ham + np.diag(rng.uniform(-10., 10., len(ham)))
        self.assertAlmostEqual(bott_index(dis, pos, cell), 0., places=8)

    def test_checks(self):
        ham, pos, cell = self._torus(haldane(), n=3)
        self.assertRaises(ValueError, bott_index, ham, pos[:-1], cell)
        self.assertRaises(ValueError, bott_index, ham, pos, [cell[0], cell[0]])
        self.assertRaises(ValueError, bott_index, ham, pos, cell[:1])
        self.assertRaises(TypeError, bott_index, ham, pos, cell, 'a')


class TestWeylPoints(unittest.TestCase):

    def test_two_weyl_points(self):
        k0 = np.pi / 3
        points = find_weyl_points(weyl(k0), 0)
        self.assertIsInstance(points, WeylPoints)
        order = np.argsort(points.k[:, 2])
        self.assertTrue(np.allclose(points.k[order], [[0., 0., -k0], [0., 0., k0]], atol=1e-8))
        self.assertEqual(points.chirality[order].tolist(), [-1, 1])
        self.assertEqual(points.total_chirality, 0)
        self.assertTrue(np.all(points.gap < 1e-8))
        self.assertTrue(np.allclose(points.energy, 0., atol=1e-8))
        # the chirality is the jump of the Chern number of the k_z planes
        c_below, c_between, c_above = [weyl(k0).chern_number(0, 20, k_fixed=f) for f in (-0.4, 0., 0.4)]
        self.assertAlmostEqual(c_between - c_below, points.chirality[order][0])
        self.assertAlmostEqual(c_above - c_between, points.chirality[order][1])
        # a coarse mesh and an explicit sphere give the same points
        coarse = find_weyl_points(weyl(k0), [0], nk=6, radius=0.1)
        self.assertEqual(sorted(coarse.chirality.tolist()), [-1, 1])
        # far apart Weyl points
        wide = find_weyl_points(weyl(2.), 0, nk=6)
        order = np.argsort(wide.k[:, 2])
        self.assertTrue(np.allclose(wide.k[order], [[0., 0., -2.], [0., 0., 2.]], atol=1e-8))
        self.assertEqual(wide.chirality[order].tolist(), [-1, 1])

    def test_merged_candidates(self):
        # longer-range terms move the Weyl points off the k_z axis and add gap
        # minima, three of which refine onto the same two points
        ks = weyl(1.5)
        ks.set_hopping([{'i': 0, 'j': 0, 'R': (0, 0, 2), 't': 0.2 * PAULI['z']},
                               {'i': 0, 'j': 0, 'R': (1, 1, 0), 't': 0.1 * PAULI['x']}])
        coarse, fine = find_weyl_points(ks, 0, nk=8), find_weyl_points(ks, 0, nk=20)
        for points in (coarse, fine):
            order = np.argsort(points.k[:, 2])
            self.assertEqual(points.chirality[order].tolist(), [-1, 1])
            self.assertTrue(np.all(points.gap < 1e-8))
        self.assertTrue(np.allclose(np.sort(coarse.k, axis=0), np.sort(fine.k, axis=0), atol=1e-8))
        self.assertGreater(abs(coarse.k[0, 0]), 0.1)

    def test_no_weyl_point(self):
        # the gapped topological insulator: candidates exist, none closes
        points = find_weyl_points(cubic_ti(2.), [0, 1], nk=6)
        self.assertEqual(points.k.shape, (0, 3))
        self.assertEqual(points.total_chirality, 0)

    def test_checks(self):
        self.assertRaises(TypeError, find_weyl_points, 'weyl', 0)
        self.assertRaises(ValueError, find_weyl_points, haldane(), 0)
        self.assertRaises(ValueError, find_weyl_points, weyl(), 1)  # no band above
        self.assertRaises(ValueError, find_weyl_points, weyl(), 0, radius=0.)
        self.assertRaises(ValueError, find_weyl_points, weyl(), 0, nk=0)
        self.assertRaises(ValueError, find_weyl_points, weyl(), 0, tol=0.)
        self.assertRaises(ValueError, find_weyl_points, weyl(), 0, max_iter=0)
        nh = weyl()
        nh.set_onsite({'a': 0.1j * PAULI['0']})
        self.assertRaises(ValueError, find_weyl_points, nh, 0)


class TestValidators(unittest.TestCase):

    def test_validators(self):
        self.assertRaises(ValueError, error_handling.mirror_square, np.zeros((1, 2, 2)), 1e-8)
        self.assertEqual(error_handling.mirror_square(-np.eye(2)[None], 1e-8), -1)
        self.assertRaises(ValueError, error_handling.commutes, 1., 1e-8)
        self.assertRaises(ValueError, error_handling.torus_cell, np.ones((2, 2)), 2)
        self.assertRaises(ValueError, error_handling.band_below_top, 1, 2)
        self.assertRaises(ValueError, error_handling.antisymmetric, np.eye(2))
        self.assertRaises(ValueError, error_handling.bdg_orbitals, 3)
        self.assertRaises(ValueError, error_handling.majorana_form, 1j * np.eye(2))
        self.assertRaises(ValueError, error_handling.gapped_trim, 0.)
        self.assertRaises(ValueError, error_handling.projected_gap, np.array([[-1., 1.], [1., 2.]]))
        self.assertEqual(error_handling.projected_gap(np.array([[-1., 1.], [-2., 2.]])), 1)
