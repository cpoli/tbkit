"""
Sparse eigensolver, Green's function and local density of states, and the
occupation of states (Fermi-Dirac, Fermi level, charge, band energy).
"""
import unittest

import numpy as np

import tbkit.lattices as lattices
import tbkit.occupation as occupation
from tbkit.kspace import KSpace
from tbkit.lattice import Lattice
from tbkit.system import System


def square_sys(n=6, onsite=None):
    lat = lattices.square()
    lat.get_lattice(n, n)
    sys = System(lat)
    sys.set_hopping([{'n': 1, 't': 1.}])
    if onsite is not None:
        sys.set_onsite({'a': 0.})
        sys.set_onsite_def(onsite)
    sys.get_ham()
    return sys


class TestSparse(unittest.TestCase):

    def test_matches_dense(self):
        sys = square_sys(8, onsite={3: 0.7})
        sys.get_eig()
        dense = sys.en.copy()
        sys.get_eig_sparse(n_eig=6, sigma=0.2718, eigenvec=True)
        # the six dense eigenvalues closest to 0.3
        closest = np.sort(dense[np.argsort(np.abs(dense - 0.2718))[:6]])
        self.assertTrue(np.allclose(sys.en, closest))
        ham = sys.ham.toarray()
        self.assertTrue(np.allclose(ham @ sys.rn, sys.rn * sys.en[None, :]))
        self.assertEqual(sys.pola.shape, (6, 1))
        self.assertTrue(np.allclose(sys.intensity.sum(axis=0), 1.))
        # sigma = 0 is exactly an eigenvalue here (a zero mode): still works
        self.assertLess(np.min(np.abs(dense)), 1e-12)
        sys.get_eig_sparse(n_eig=3)
        self.assertEqual(len(sys.en), 3)
        self.assertLess(np.min(np.abs(sys.en)), 1e-12)

    def test_non_hermitian(self):
        sys = square_sys(6, onsite={0: 0.5j, 7: -0.3j})
        sys.get_eig()
        dense = sys.en.copy()
        sys.get_eig_sparse(n_eig=4, sigma=0.2 + 0.1j, eigenvec=True)
        for e in sys.en:
            self.assertLess(np.min(np.abs(dense - e)), 1e-8)
        self.assertTrue(np.allclose(sys.ham.toarray() @ sys.rn, sys.rn * sys.en[None, :]))

    def test_checks(self):
        sys = square_sys(3)
        self.assertRaises(ValueError, sys.get_eig_sparse, 8)
        self.assertRaises(TypeError, sys.get_eig_sparse, 2.)
        self.assertRaises(TypeError, sys.get_eig_sparse, 2, 'a')
        self.assertRaises(TypeError, sys.get_eig_sparse, 2, 0., 1)


class TestGreen(unittest.TestCase):

    def test_green_and_ldos(self):
        sys = square_sys(5, onsite={2: 0.4})
        g = sys.get_green(0.3, eta=0.)
        self.assertTrue(np.allclose(g @ (0.3 * np.eye(25) - sys.ham.toarray()), np.eye(25)))
        es = np.array([-1., 0.2, 1.5])
        ldos = sys.get_ldos(es, eta=0.1)
        self.assertEqual(ldos.shape, (3, 25))
        for e, row in zip(es, ldos):
            self.assertTrue(np.allclose(row, -np.diag(sys.get_green(e, 0.1)).imag / np.pi))
        # summed over sites and energies: the Lorentzian DOS integrates to N
        grid = np.linspace(-40., 40., 40001)
        total = np.trapezoid(sys.get_ldos(grid, eta=0.1).sum(axis=1), grid)
        self.assertAlmostEqual(total / 25, 1., places=2)

    def test_non_hermitian_ldos(self):
        sys = square_sys(4, onsite={0: 0.3j})
        ldos = sys.get_ldos([0.5], eta=0.2)
        self.assertTrue(np.allclose(ldos[0], -np.diag(sys.get_green(0.5, 0.2)).imag / np.pi))

    def test_checks(self):
        sys = square_sys(3)
        self.assertRaises(TypeError, sys.get_green, 'a')
        self.assertRaises(ValueError, sys.get_green, 0., -1.)
        self.assertRaises(ValueError, sys.get_ldos, [0.], 0.)
        self.assertRaises(ValueError, sys.get_ldos, [])


class TestOccupation(unittest.TestCase):

    def test_fermi_dirac(self):
        e = np.array([-1., 0., 1.])
        self.assertTrue(np.allclose(occupation.fermi_dirac(e, 0.), [1., 0.5, 0.]))
        f = occupation.fermi_dirac(e, 0., 0.5)
        self.assertTrue(np.allclose(f, 1. / (np.exp(e / 0.5) + 1.)))
        self.assertTrue(np.allclose(occupation.fermi_dirac([1e6], 0., 1e-3), 0.))  # no overflow

    def test_fermi_level_zero_temperature(self):
        e = np.array([-2., -1., 1., 3.])
        self.assertAlmostEqual(occupation.fermi_level(e, 2), 0.)  # mid-gap
        self.assertAlmostEqual(occupation.fermi_level(e, 0), -3.)  # below everything
        self.assertAlmostEqual(occupation.fermi_level(e, 4), 4.)  # all filled
        self.assertAlmostEqual(occupation.fermi_level(e, 1.5), -1.)  # a partly filled level
        # degenerate last level
        self.assertAlmostEqual(occupation.fermi_level([-1., 0., 0., 2.], 2), 0.)
        # weights (a k-mesh)
        self.assertAlmostEqual(occupation.fermi_level(e, 0.5, weights=np.full(4, 0.25)), 0.)

    def test_fermi_level_temperature(self):
        e = np.linspace(-2., 2., 11)
        mu = occupation.fermi_level(e, 4.3, temperature=0.2)
        self.assertAlmostEqual(occupation.fermi_dirac(e, mu, 0.2).sum(), 4.3)

    def test_checks(self):
        self.assertRaises(ValueError, occupation.fermi_level, [0., 1.], 3)
        self.assertRaises(ValueError, occupation.fermi_level, [0., 1.], -1)
        self.assertRaises(ValueError, occupation.fermi_level, [0., 1.], 1, 0., [1.])
        self.assertRaises(ValueError, occupation.fermi_level, [0., 1.], 1, 0., [1., 0.])
        self.assertRaises(ValueError, occupation.fermi_level, [], 0)
        self.assertRaises(ValueError, occupation.fermi_dirac, [0.], 0., -1.)

    def test_system(self):
        sys = square_sys(4, onsite={0: 0.5})
        sys.get_eig(eigenvec=True)
        # a filling with the Fermi level in a gap
        n_e = next(n for n in range(4, 12) if sys.en[n] - sys.en[n - 1] > 1e-6)
        mu = sys.get_fermi_level(n_e)
        f = sys.get_occupations(mu)
        self.assertAlmostEqual(f.sum(), n_e)
        dens = sys.get_charge_density(mu)
        self.assertAlmostEqual(dens.sum(), n_e)
        self.assertAlmostEqual(sys.get_total_energy(mu), np.sort(sys.en)[:n_e].sum())
        # a degenerate level at the Fermi level is half filled at T = 0
        deg = next(n for n in range(2, 14) if abs(sys.en[n] - sys.en[n - 1]) < 1e-9)
        mu_deg = sys.get_fermi_level(deg)
        self.assertAlmostEqual(mu_deg, sys.en[deg])
        # a repulsive site holds less charge
        self.assertLess(dens[0], np.mean(dens[1:]))
        mu_t = sys.get_fermi_level(n_e, temperature=0.3)
        self.assertAlmostEqual(sys.get_occupations(mu_t, 0.3).sum(), n_e)
        self.assertAlmostEqual(sys.get_charge_density(mu_t, 0.3).sum(), n_e)

    def test_kspace(self):
        # half-filled square lattice: E_F = 0; quarter filling: below 0
        sq = KSpace(lattices.square())
        sq.set_hopping([{'i': 0, 'j': 0, 'R': (1, 0), 't': 1.}, {'i': 0, 'j': 0, 'R': (0, 1), 't': 1.}])
        self.assertAlmostEqual(sq.get_fermi_level(0.5, nk=20), 0., places=8)
        self.assertLess(sq.get_fermi_level(0.25, nk=20), -0.5)
        mu = sq.get_fermi_level(0.5, nk=20, temperature=0.1)
        self.assertAlmostEqual(mu, 0., places=8)


if __name__ == '__main__':
    unittest.main()
