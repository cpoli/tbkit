"""
Slater-Koster integrals, multi-orbital and spinful real-space models
(OrbitalSystem), and non-orthogonal bases (overlap matrices).
"""
import unittest

import numpy as np
import scipy.linalg as LA

import tbkit.lattices as lattices
from tbkit.kspace import KSpace
from tbkit.lattice import Lattice
from tbkit.orbital import OrbitalSystem, SIGMA
from tbkit.plot import Plot
from tbkit.slater_koster import sk_block, orbital_angular_momentum, _rep, _rotation
from tbkit.system import System
from tests.test_topology import kane_mele

S3 = np.sqrt(3.)
ORBS = ['s', 'px', 'py', 'pz', 'dxy', 'dyz', 'dzx', 'dx2-y2', 'dz2']
PARAMS = {'ss_sigma': -1.1, 'sp_sigma': 0.7, 'pp_sigma': 1.3, 'pp_pi': -0.4, 'sd_sigma': 0.3,
                'pd_sigma': -0.8, 'pd_pi': 0.5, 'dd_sigma': -0.9, 'dd_pi': 0.6, 'dd_delta': -0.2}
SP3 = {'ss_sigma': -6.77, 'sp_sigma': 5.58, 'pp_sigma': 5.04, 'pp_pi': -3.03}


class TestSlaterKoster(unittest.TestCase):

    def test_table(self):
        # entries of the Slater-Koster (1954) table
        d = np.array([0.3, -0.5, 0.8])
        l, m, n = d / np.linalg.norm(d)
        E = sk_block(ORBS, ORBS, d, PARAMS)
        V = PARAMS
        table = {
            ('s', 's'): V['ss_sigma'],
            ('s', 'px'): l * V['sp_sigma'],
            ('px', 'px'): l*l*V['pp_sigma'] + (1 - l*l)*V['pp_pi'],
            ('px', 'py'): l*m*(V['pp_sigma'] - V['pp_pi']),
            ('s', 'dxy'): S3*l*m*V['sd_sigma'],
            ('s', 'dz2'): (n*n - (l*l + m*m)/2)*V['sd_sigma'],
            ('px', 'dxy'): S3*l*l*m*V['pd_sigma'] + m*(1 - 2*l*l)*V['pd_pi'],
            ('px', 'dyz'): S3*l*m*n*V['pd_sigma'] - 2*l*m*n*V['pd_pi'],
            ('py', 'dx2-y2'): S3/2*m*(l*l - m*m)*V['pd_sigma'] - m*(1 + l*l - m*m)*V['pd_pi'],
            ('pz', 'dz2'): n*(n*n - (l*l + m*m)/2)*V['pd_sigma'] + S3*n*(l*l + m*m)*V['pd_pi'],
            ('dxy', 'dxy'): 3*l*l*m*m*V['dd_sigma'] + (l*l + m*m - 4*l*l*m*m)*V['dd_pi']
                                   + (n*n + l*l*m*m)*V['dd_delta'],
            ('dxy', 'dyz'): 3*l*m*m*n*V['dd_sigma'] + l*n*(1 - 4*m*m)*V['dd_pi'] + l*n*(m*m - 1)*V['dd_delta'],
            ('dx2-y2', 'dz2'): S3/2*(l*l - m*m)*(n*n - (l*l + m*m)/2)*V['dd_sigma']
                                      + S3*n*n*(m*m - l*l)*V['dd_pi'] + S3/4*(1 + n*n)*(l*l - m*m)*V['dd_delta'],
            ('dz2', 'dz2'): (n*n - (l*l + m*m)/2)**2*V['dd_sigma'] + 3*n*n*(l*l + m*m)*V['dd_pi']
                                   + 0.75*(l*l + m*m)**2*V['dd_delta'],
        }
        for (a, b), val in table.items():
            self.assertAlmostEqual(E[ORBS.index(a), ORBS.index(b)], val, msg=(a, b))
        # the reverse bond is the transpose (homonuclear)
        self.assertTrue(np.allclose(sk_block(ORBS, ORBS, -d, PARAMS), E.T))

    def test_special_directions(self):
        # along +-z, and in the plane (2D vector)
        for d in ([0., 0., 1.], [0., 0., -2.]):
            E = sk_block(['pz', 'dz2'], ['pz', 'dz2'], d, PARAMS)
            self.assertAlmostEqual(E[0, 0], PARAMS['pp_sigma'])
            self.assertAlmostEqual(E[1, 1], PARAMS['dd_sigma'])
            self.assertAlmostEqual(E[0, 1], np.sign(d[2]) * PARAMS['pd_sigma'])
        E = sk_block(['pz'], ['pz'], (1., 1.), PARAMS)
        self.assertAlmostEqual(E[0, 0], PARAMS['pp_pi'])
        self.assertTrue(np.allclose(sk_block(['s'], ['pz'], (1., 0.), PARAMS), 0.))

    def test_heteronuclear(self):
        # V_ps differs from V_sp
        d = (1., 0., 0.)
        self.assertAlmostEqual(sk_block(['px'], ['s'], d, {'sp_sigma': 1., 'ps_sigma': 2.})[0, 0], -2.)
        self.assertAlmostEqual(sk_block(['px'], ['s'], d, {'sp_sigma': 1.})[0, 0], -1.)
        self.assertAlmostEqual(sk_block(['dz2'], ['s'], (0., 0., 1.), {'ds_sigma': 3.})[0, 0], 3.)
        self.assertAlmostEqual(sk_block(['dz2'], ['pz'], (0., 0., 1.), {'dp_sigma': 3.})[0, 0], -3.)

    def test_rotation_representations(self):
        for axis in ([0.1, 0.9, -0.3], [0., 0., -1.], [1., 0., 0.]):
            rot = _rotation(np.array(axis) / np.linalg.norm(axis))
            self.assertTrue(np.allclose(rot @ [0., 0., 1.], np.array(axis) / np.linalg.norm(axis)))
            d2 = _rep(2, rot)
            self.assertTrue(np.allclose(d2 @ d2.T, np.eye(5)))

    def test_angular_momentum(self):
        L = orbital_angular_momentum()
        self.assertTrue(np.allclose(L[0] @ L[1] - L[1] @ L[0], 1j * L[2]))
        self.assertTrue(np.allclose(sum(l @ l for l in L), 2 * np.eye(3)))  # l(l+1) = 2

    def test_checks(self):
        self.assertRaises(ValueError, sk_block, ['f'], ['s'], (1., 0.), {})
        self.assertRaises(ValueError, sk_block, ['s', 's'], ['s'], (1., 0.), {})
        self.assertRaises(TypeError, sk_block, 's', ['s'], (1., 0.), {})
        self.assertRaises(ValueError, sk_block, ['s'], ['s'], (0., 0.), {})
        self.assertRaises(ValueError, sk_block, ['s'], ['s'], (1., 0.), {'xx_sigma': 1.})
        self.assertRaises(TypeError, sk_block, ['s'], ['s'], (1., 0.), {'ss_sigma': 'a'})
        self.assertRaises(TypeError, sk_block, ['s'], ['s'], (1., 0.), [1.])


def honeycomb(n1=4, n2=3):
    lat = lattices.honeycomb()
    lat.get_lattice(n1, n2)
    return lat


class TestOrbitalSystem(unittest.TestCase):

    def test_kane_mele_flake(self):
        # real-space Kane-Mele + Rashba + mass, against the same open
        # parallelogram built from the Bloch model
        sys = OrbitalSystem(honeycomb(5, 4), spin=True)
        sys.set_hopping([{'n': 1, 't': 1.}])
        sys.set_spin_orbit(0.06)
        sys.set_rashba(0.05)
        sys.set_onsite({'a': 0.1, 'b': -0.1})
        sys.get_ham()
        sys.get_eig(eigenvec=True)
        flake = kane_mele(lam=0.06, M=0.1, rashba=0.05).finite_ham((5, 4))
        self.assertTrue(np.allclose(sys.en, np.linalg.eigvalsh(flake)))
        self.assertEqual(sys.intensity.shape, (40, 80))
        self.assertTrue(np.allclose(sys.intensity.sum(axis=0), 1.))
        self.assertEqual(sys.pola.shape, (80, 2))

    def test_graphene_sp3(self):
        # in-plane bonds: pz decouples, and its band is Wallace's with t = V_pp_pi
        orbs = {'a': ['s', 'px', 'py', 'pz'], 'b': ['s', 'px', 'py', 'pz']}
        sys = OrbitalSystem(honeycomb(), orbitals=orbs)
        sys.set_onsite({'a': {'s': -8.87}, 'b': {'s': -8.87}})
        sys.set_slater_koster(1, SP3)
        sys.get_ham()
        ham = sys.ham.toarray()
        pz = [sys.row(i, 'pz') for i in range(sys.lat.sites)]
        rest = [r for r in range(sys.n_rows) if r not in pz]
        self.assertTrue(np.allclose(ham[np.ix_(pz, rest)], 0.))
        ref = System(honeycomb())
        ref.set_hopping([{'n': 1, 't': SP3['pp_pi']}])
        ref.get_ham()
        self.assertTrue(np.allclose(ham[np.ix_(pz, pz)], ref.ham.toarray()))
        self.assertAlmostEqual(ham[sys.row(0, 's'), sys.row(0, 's')], -8.87)

    def test_per_pair_params(self):
        orbs = {'a': ['s'], 'b': ['px']}
        sys = OrbitalSystem(honeycomb(2, 2), orbitals=orbs)
        sys.set_slater_koster(1, {'ab': {'sp_sigma': 1.}, 'ba': {'ps_sigma': 1.}})
        sys.get_ham()
        self.assertTrue(np.allclose(sys.ham.toarray(), sys.ham.toarray().conj().T))
        self.assertRaises(ValueError, sys.set_slater_koster, 1, {'ab': {'sp_sigma': 1.}})

    def test_overlap(self):
        # H = t A and S = 1 + s A share their eigenvectors: E = t a / (1 + s a)
        sys = OrbitalSystem(honeycomb(3, 3), orbitals={'a': ['pz'], 'b': ['pz']})
        sys.set_slater_koster(1, {'pp_pi': -2.7}, overlap={'pp_pi': 0.1})
        sys.get_ham()
        adj = System(honeycomb(3, 3))
        adj.set_hopping([{'n': 1, 't': 1.}])
        adj.get_ham()
        eig_a = LA.eigvalsh(adj.ham.toarray())
        sys.get_eig(eigenvec=True)
        self.assertTrue(np.allclose(np.sort(sys.en), np.sort(-2.7 * eig_a / (1 + 0.1 * eig_a))))
        # Mulliken populations sum to one per state
        self.assertTrue(np.allclose(sys.intensity.sum(axis=0), 1.))
        self.assertRaises(ValueError, sys.get_eig, True, True)
        self.assertRaises(ValueError, sys.get_eig_sparse, 2)
        self.assertRaises(ValueError, sys.get_local_chern_marker, 0., 1.)
        # non-Hermitian with an overlap
        sys.set_onsite({'a': 0.2j, 'b': 0.})
        sys.get_ham()
        sys.get_eig(eigenvec=True)
        ham, s = sys.ham.toarray(), sys.overlap.toarray()
        self.assertTrue(np.allclose(ham @ sys.rn, s @ sys.rn * sys.en[None, :]))
        # removing the overlap
        sys.set_slater_koster(1, {'pp_pi': -2.7})
        sys.get_ham()
        self.assertIsNone(sys.overlap)

    def test_spin_terms(self):
        one = Lattice(unit_cell=[{'tag': 'a', 'r0': (0., 0.)}], prim_vec=[(1., 0.)])
        one.get_lattice(3)
        sys = OrbitalSystem(one, orbitals={'a': ['s', 'px', 'py', 'pz']}, spin=True)
        sys.set_hopping([{'n': 1, 't': 0.}])
        sys.set_atomic_soc(0.4)
        sys.get_ham()
        sys.get_eig()
        # s: 0 (x 2 spins); p: j = 1/2 at -lam (x 2), j = 3/2 at +lam/2 (x 4)
        levels, counts = np.unique(sys.en.round(8), return_counts=True)
        self.assertTrue(np.allclose(levels, [-0.4, 0., 0.2]))
        self.assertTrue(np.array_equal(counts, [6, 6, 12]))
        sys.set_zeeman((0., 0., 0.3))
        sys.get_ham()
        s_up, s_dn = sys.row(1, 's', 0), sys.row(1, 's', 1)
        ham = sys.ham.toarray()
        self.assertAlmostEqual(ham[s_up, s_up] - ham[s_dn, s_dn], 0.6)
        self.assertRaises(ValueError, sys.set_zeeman, (0., 1.))
        self.assertRaises(ValueError, sys.set_atomic_soc, 0.1, ['x'])
        spinless = OrbitalSystem(one)
        self.assertRaises(ValueError, spinless.set_zeeman, (0., 0., 1.))
        self.assertRaises(ValueError, spinless.set_spin_orbit, 0.1)
        self.assertRaises(ValueError, spinless.set_rashba, 0.1)
        self.assertRaises(ValueError, spinless.set_atomic_soc, 0.1)
        s_only = OrbitalSystem(one, spin=True)
        self.assertRaises(ValueError, s_only.set_atomic_soc, 0.1)
        # a square lattice's diagonal hop sees two opposite turns: no SOC
        sq = lattices.square()
        sq.get_lattice(3, 3)
        sqs = OrbitalSystem(sq, spin=True)
        sqs.set_spin_orbit(0.1)
        self.assertEqual(len(sqs.blocks[('soc', 2)]), 0)

    def test_blocks_and_peierls(self):
        # a manual block equal to t reproduces set_hopping, also in a field
        lat = lattices.square()
        lat.get_lattice(3, 3)
        ref = System(lat)
        ref.set_hopping([{'n': 1, 't': 1.}])
        ref.set_magnetic_field(0.1)
        ref.get_ham()
        sys = OrbitalSystem(lat)
        sys.set_hopping_block({(int(i), int(j)): [[1.]] for i, j in zip(ref.hop['i'], ref.hop['j'])})
        sys.set_magnetic_field(0.1)
        sys.get_ham()
        self.assertTrue(np.allclose(sys.ham.toarray(), ref.ham.toarray()))
        sys2 = OrbitalSystem(lat)
        sys2.set_hopping([{'n': 1, 't': 1.}])
        sys2.set_magnetic_field(0.1)
        sys2.get_ham()
        self.assertTrue(np.allclose(sys2.ham.toarray(), ref.ham.toarray()))
        self.assertRaises(ValueError, sys.set_hopping_block, {(0, 1): np.ones((2, 2))})
        sys.clear_hopping()
        self.assertRaises(RuntimeError, sys.get_ham)

    def test_system_features(self):
        # onsite by orbital, disorder, charge, plots, marker, sparse solver
        lat = honeycomb(3, 3)
        sys = OrbitalSystem(lat, orbitals={'a': ['s', 'pz'], 'b': ['pz']}, spin=True)
        sys.set_hopping([{'n': 1, 't': 1.}])
        sys.set_onsite({'a': {'s': 2.}, 'b': 0.1})
        sys.set_onsite_dis(0.05)
        sys.get_ham()
        self.assertEqual(sys.n_rows, 2 * (2 * 9 + 9))
        sys.get_eig(eigenvec=True)
        mu = sys.get_fermi_level(20)
        self.assertAlmostEqual(sys.get_charge_density(mu).sum(), 20.)
        fig = Plot(sys).spectrum(tag_pola='a')
        self.assertEqual(len(fig.axes[0].lines[0].get_xdata()), sys.n_rows)
        sys.get_eig_sparse(4, sigma=0.)
        self.assertEqual(len(sys.en), 4)
        self.assertTrue(np.allclose(sys.get_local_chern_marker(0.).sum(), 0.))
        self.assertRaises(ValueError, sys.row, 0, 'dxy')
        self.assertRaises(ValueError, sys.row, 0, 's', 2)
        self.assertRaises(ValueError, OrbitalSystem, lat, {'a': ['s']})
        self.assertRaises(TypeError, OrbitalSystem, lat, ['s'])
        self.assertRaises(ValueError, sys.set_onsite, {'a': {'dxy': 1.}})
        self.assertRaises(TypeError, sys.set_onsite, {'a': {'s': 'x'}})
        self.assertRaises(TypeError, sys.set_onsite, {'a': [1.]})
        self.assertRaises(ValueError, sys.set_onsite, {'c': 1.})
        self.assertRaises(TypeError, sys.set_onsite, 1.)
        self.assertRaises(TypeError, sys.set_slater_koster, 1, [1.])
        self.assertRaises(ValueError, sys.set_slater_koster, 500, SP3)

    def test_spinful_slater_koster_and_partial_soc(self):
        # spin-independent SK blocks: two identical spin copies
        orbs = {'a': ['s', 'px', 'py', 'pz'], 'b': ['s']}
        spin = OrbitalSystem(honeycomb(2, 2), orbitals=orbs, spin=True)
        spin.set_slater_koster(1, SP3)
        spin.get_ham()
        plain = OrbitalSystem(honeycomb(2, 2), orbitals=orbs)
        plain.set_slater_koster(1, SP3)
        plain.get_ham()
        e_spin = np.sort(LA.eigvalsh(spin.ham.toarray()))
        e_plain = np.sort(LA.eigvalsh(plain.ham.toarray()))
        self.assertTrue(np.allclose(e_spin, np.repeat(e_plain, 2)))
        # atomic L.S on the 'a' sites only
        spin.set_atomic_soc(0.3, tags=['a'])
        self.assertEqual(len(spin.onsite_blocks['atomic_soc']),
                                np.sum(spin.lat.coor['tag'] == 'a'))

    def test_marker_matches_system(self):
        # the Haldane flake of test_topology, one orbital per site
        from tests.test_topology import TestLocalChernMarker
        ref = TestLocalChernMarker()._flake(n=6)
        sys = OrbitalSystem(ref.lat)
        sys.hop = ref.hop.copy()
        sys.get_ham()
        self.assertTrue(np.allclose(sys.get_local_chern_marker(0.), ref.get_local_chern_marker(0.)))


class TestSKKSpace(unittest.TestCase):

    def test_matches_orbital_system(self):
        from tbkit.slater_koster import sk_kspace
        orbs = {'a': ['s', 'px', 'py', 'pz'], 'b': ['s', 'pz']}
        onsite = {'a': {'s': -8.87}, 'b': 0.5}
        ks = sk_kspace(lattices.honeycomb(), orbs, {1: SP3, 2: {'ss_sigma': 0.1}}, onsite=onsite)
        self.assertEqual(ks.norb, 6)
        self.assertEqual(ks.sk_orbitals[4], (1, 's'))
        sys = OrbitalSystem(honeycomb(4, 3), orbitals=orbs)
        sys.set_onsite(onsite)
        sys.set_slater_koster(1, SP3)
        sys.set_slater_koster(2, {'ss_sigma': 0.1})
        sys.get_ham()
        sys.get_eig()
        self.assertTrue(np.allclose(sys.en, np.linalg.eigvalsh(ks.finite_ham((4, 3)))))

    def test_overlap_and_checks(self):
        from tbkit.slater_koster import sk_kspace
        orbs = {'a': ['pz'], 'b': ['pz']}
        ks = sk_kspace(lattices.honeycomb(), orbs, {1: {'pp_pi': -2.7}}, overlap={1: {'pp_pi': 0.1}})
        self.assertTrue(np.allclose(ks.get_bands([[0., 0.]]).ravel(),
                                             sorted([-8.1 / 1.3, 8.1 / 0.7])))
        self.assertRaises(TypeError, sk_kspace, lattices.honeycomb(), orbs, {0: {}})
        self.assertRaises(TypeError, sk_kspace, lattices.honeycomb(), orbs, [{}])
        self.assertRaises(ValueError, sk_kspace, lattices.honeycomb(), orbs, {99: {}})
        self.assertRaises(ValueError, sk_kspace, lattices.honeycomb(), orbs, {1: {}}, None, {1: {'x': 1.}})


class TestKSpaceOverlap(unittest.TestCase):

    def graphene(self, s=0.1):
        g = KSpace(lattices.honeycomb())
        hop = [{'i': 0, 'j': 1, 'R': R, 't': -2.7} for R in [(0, 0), (-1, 0), (0, -1)]]
        g.set_hopping(hop)
        g.set_overlap([dict(h, t=s) for h in hop])
        return g

    def test_saito(self):
        g = self.graphene()
        lat = g.lat
        for k in (np.array([0.4, 1.1]), np.array([-1.3, 0.2])):
            f = sum(np.exp(1j * np.dot(k, n1 * np.array(lat.prim_vec[0]) + n2 * np.array(lat.prim_vec[1])))
                       for n1, n2 in [(0, 0), (-1, 0), (0, -1)])
            exact = sorted([-2.7 * abs(f) / (1 + 0.1 * abs(f)), 2.7 * abs(f) / (1 - 0.1 * abs(f))])
            self.assertTrue(np.allclose(g.get_bands([k]).ravel(), exact))
        s = g.get_overlap([0., 0.])
        self.assertTrue(np.allclose(s, [[1., 0.3], [0.3, 1.]]))
        en, vn = g.get_bands([[0.3, 0.2]], eigenvec=True)
        s = g.get_overlap([0.3, 0.2])
        self.assertTrue(np.allclose(vn[0].conj().T @ s @ vn[0], np.eye(2)))

    def test_topology_with_overlap(self):
        from tests.test_topology import haldane
        hal = haldane()
        hal.set_overlap([{'i': 0, 'j': 1, 'R': R, 't': 0.05} for R in [(0, 0), (-1, 0), (0, -1)]])
        self.assertAlmostEqual(hal.chern_number(0, 16), 1., places=6)
        hal.clear_hopping()
        self.assertEqual(len(hal._overlap_hop), 0)

    def test_non_hermitian_with_overlap(self):
        g = self.graphene()
        g.set_onsite({'a': 0.1j, 'b': 0.})
        en = g.get_bands([[0.3, 0.2]]).ravel()
        k = [0.3, 0.2]
        self.assertTrue(np.allclose(np.sort_complex(en),
                                             np.sort_complex(LA.eigvals(g.get_ham(k), g.get_overlap(k)))))


if __name__ == '__main__':
    unittest.main()
