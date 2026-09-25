"""
Mean-field Hubbard model, and Bogoliubov-de Gennes superconductivity.
"""
import unittest

import numpy as np
import scipy.linalg as LA

import tbkit.lattices as lattices
from tbkit.bdg import bdg_ham, bdg_kspace, pairing_bonds, pairing_s_wave, particle_hole
from tbkit.graphene import GrapheneLattice
from tbkit.kspace import KSpace, PAULI
from tbkit.meanfield import hubbard_mean_field
from tbkit.orbital import OrbitalSystem
from tbkit.system import System


def triangle(n):
    lat = GrapheneLattice()
    lat.triangle_zigzag(n)
    sys = System(lat)
    sys.set_hopping([{'n': 1, 't': -1.}])
    sys.get_ham()
    return sys


def staggered(lat, m=0.2):
    s = np.where(lat.coor['tag'] == 'a', m, -m)
    return 0.5 + s, 0.5 - s


class TestHubbard(unittest.TestCase):

    def test_lieb_theorem(self):
        # half filling on a bipartite flake: S = ||A| - |B|| / 2
        for n in (2, 3, 4):
            sys = triangle(n)
            lat = sys.lat
            n_a = np.sum(lat.coor['tag'] == 'a')
            up, dn = staggered(lat)
            res = hubbard_mean_field(sys.ham, 2., lat.sites, n_up=up, n_dn=dn)
            self.assertAlmostEqual(abs(res.total_spin), abs(2*n_a - lat.sites) / 2, places=6)
            self.assertAlmostEqual(np.sum(res.n_up + res.n_dn), lat.sites, places=8)

    def test_lowest_energy_solution(self):
        # random starts may converge to metastable states; the lowest energy
        # one is Lieb's
        sys = triangle(3)
        results = [hubbard_mean_field(sys.ham, 2., sys.lat.sites, seed=s) for s in range(8)]
        self.assertGreater(len({round(r.energy, 6) for r in results}), 1)
        best = min(results, key=lambda r: r.energy)
        self.assertAlmostEqual(abs(best.total_spin), 1.5, places=6)

    def test_dimer(self):
        # two sites, t = 1: mean-field antiferromagnetism only above U = 2
        dimer = np.array([[0., -1.], [-1., 0.]])
        para = hubbard_mean_field(dimer, 1.5, 2, n_up=[0.6, 0.4], n_dn=[0.4, 0.6])
        self.assertLess(np.max(np.abs(para.magnetization)), 1e-5)
        afm = hubbard_mean_field(dimer, 4., 2, n_up=[0.6, 0.4], n_dn=[0.4, 0.6])
        # m = sqrt(1 - (2/U)^2) / 2 (MF solution of the dimer)
        self.assertAlmostEqual(abs(afm.magnetization[0]), np.sqrt(1 - 0.25) / 2, places=6)
        self.assertAlmostEqual(afm.total_spin, 0., places=8)
        # at the same U, below the paramagnetic solution (E = 0 there)
        para4 = hubbard_mean_field(dimer, 4., 2, n_up=[0.5, 0.5], n_dn=[0.5, 0.5])
        self.assertAlmostEqual(para4.energy, 0., places=8)
        self.assertLess(afm.energy, para4.energy)

    def test_temperature_and_options(self):
        sys = triangle(2)
        res = hubbard_mean_field(sys.ham, np.full(sys.lat.sites, 1.), sys.lat.sites - 1,
                                             temperature=0.05, seed=0)
        self.assertAlmostEqual(np.sum(res.n_up + res.n_dn), sys.lat.sites - 1, places=6)
        self.assertEqual(len(res.en_up), sys.lat.sites)
        self.assertRaises(ValueError, hubbard_mean_field, sys.ham, 1., 2.5)
        self.assertRaises(ValueError, hubbard_mean_field, sys.ham, 1., 3, 0., None, None, 1.5)
        self.assertRaises(ValueError, hubbard_mean_field, sys.ham, [1., 2.], 3)
        self.assertRaises(ValueError, hubbard_mean_field, 1j * sys.ham, 1., 3)
        self.assertRaises(RuntimeError, hubbard_mean_field, sys.ham, 3., sys.lat.sites,
                                0., None, None, 0.5, 1e-14, 2)


class TestBdG(unittest.TestCase):

    def kitaev(self, mu, t=1., delta=0.6):
        chain = KSpace(lattices.chain())
        chain.set_hopping([{'i': 0, 'j': 0, 'R': (1,), 't': t}])
        return bdg_kspace(chain, [{'i': 0, 'j': 0, 'R': (1,), 'delta': delta}], mu=mu)

    def test_kitaev_bands(self):
        kit = self.kitaev(0.5)
        for k in (0.3, 1.2, 2.9):
            e = np.sqrt((2*np.cos(k) - 0.5)**2 + 4*0.36*np.sin(k)**2)
            self.assertTrue(np.allclose(np.linalg.eigvalsh(kit.get_ham([k])), [-e, e]))
        self.assertLess(kit.symmetry_error(particle_hole(1), antiunitary=True, anti=True), 1e-12)
        self.assertEqual(kit.tenfold_class(time_reversal=np.eye(2), particle_hole=particle_hole(1)), 'BDI')

    def test_kitaev_topology(self):
        # Berry phase pi for |mu| < 2t, with Majorana end modes; 0 otherwise
        for mu, topo in ((0.5, True), (3., False)):
            kit = self.kitaev(mu)
            self.assertAlmostEqual(abs(kit.berry_phase(0, 200)), np.pi if topo else 0., places=6)
            e = np.sort(np.abs(np.linalg.eigvalsh(kit.finite_ham(40))))
            self.assertEqual(e[1] < 1e-8, topo)

    def test_real_space_matches_kspace(self):
        lat = lattices.chain()
        lat.get_lattice(30)
        sys = System(lat)
        sys.set_hopping([{'n': 1, 't': 1.}])
        sys.get_ham()
        ham = bdg_ham(sys.ham, pairing_bonds(sys, 0.6), mu=0.5)
        self.assertTrue(np.allclose(np.linalg.eigvalsh(ham.toarray()),
                                             np.linalg.eigvalsh(self.kitaev(0.5).finite_ham(30))))
        # particle-hole symmetric spectrum
        e = np.linalg.eigvalsh(ham.toarray())
        self.assertTrue(np.allclose(e, -e[::-1]))

    def test_s_wave(self):
        # spinful chain with s-wave pairing: E = +-sqrt((2 cos k - mu)^2 + Delta^2)
        lat = lattices.chain()
        lat.get_lattice(8)
        sys = OrbitalSystem(lat, spin=True)
        sys.set_hopping([{'n': 1, 't': 1.}])
        sys.get_ham()
        ham = bdg_ham(sys.ham, pairing_s_wave(8, 0.4), mu=0.3)
        e = np.linalg.eigvalsh(ham.toarray())
        self.assertGreater(np.min(np.abs(e)), 0.4 - 1e-9)
        chain = KSpace(lattices.chain(), spin=True)
        chain.set_hopping([{'i': 0, 'j': 0, 'R': (1,), 't': 1.}])
        chain.set_onsite({'a': 0.1 * PAULI['x']})
        sc = bdg_kspace(chain, [{'i': 0, 'j': 1, 'R': (0,), 'delta': 0.4}], mu=0.3)
        k = 0.8
        eps = 2*np.cos(k) - 0.3
        e_k = np.linalg.eigvalsh(sc.get_ham([k]))
        # a Zeeman field shifts the quasiparticles: +-sqrt(eps^2 + Delta^2) +- b
        exact = np.sort([s1 * np.sqrt(eps**2 + 0.16) + s2 * 0.1 for s1 in (-1, 1) for s2 in (-1, 1)])
        self.assertTrue(np.allclose(e_k, exact))

    def test_checks(self):
        lat = lattices.chain()
        lat.get_lattice(4)
        sys = System(lat)
        sys.set_hopping([{'n': 1, 't': 1.}])
        sys.get_ham()
        self.assertRaises(ValueError, bdg_ham, sys.ham, np.eye(4))  # not antisymmetric
        self.assertRaises(ValueError, bdg_ham, sys.ham, np.zeros((3, 3)))
        chain = KSpace(lattices.chain())
        self.assertRaises(KeyError, bdg_kspace, chain, [{'i': 0, 'j': 0, 'R': (1,)}])
        self.assertRaises(ValueError, bdg_kspace, chain, [{'i': 1, 'j': 0, 'R': (1,), 'delta': 1.}])
        self.assertRaises(ValueError, bdg_kspace, chain, [{'i': 0, 'j': 0, 'R': (1, 0), 'delta': 1.}])
        self.assertRaises(TypeError, bdg_kspace, chain, {'i': 0})
        self.assertRaises(TypeError, bdg_kspace, chain, [0])
        chain.set_onsite({'a': 1j})
        self.assertRaises(ValueError, bdg_kspace, chain, [])


if __name__ == '__main__':
    unittest.main()
