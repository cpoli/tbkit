"""
Continuum discretization (tbkit.continuum): a k·p Hamiltonian, polynomial
in k_x, k_y, k_z, turned into a Tight-Binding KSpace on a square or cubic grid.
"""
import sys
import unittest

import numpy as np
import sympy

from tbkit.bridges import finite_system
from tbkit.continuum import discretize, discretize_symbolic
from tbkit.kspace import KSpace

BHZ = 'A*(k_x*sigma_x - k_y*sigma_y) + (M - B*(k_x**2 + k_y**2))*sigma_z'


class TestSymbolic(unittest.TestCase):

    def test_second_derivative(self):
        a = sympy.Symbol('a')
        hops = discretize_symbolic('k_x**2')
        self.assertEqual(set(hops), {(-1,), (0,), (1,)})
        self.assertEqual(hops[(1,)][0, 0], -1 / a**2)
        self.assertEqual(hops[(0,)][0, 0], 2 / a**2)

    def test_odd_power_uses_whole_sites(self):
        # k_x -> -i (psi_{n+1} - psi_{n-1}) / 2a; k_x**3 reaches 3 sites
        a = sympy.Symbol('a')
        self.assertEqual(discretize_symbolic('k_x')[(1,)][0, 0], -sympy.I / (2*a))
        self.assertEqual(set(discretize_symbolic('k_x**3')), {(-3,), (-1,), (1,), (3,)})

    def test_dimension(self):
        self.assertEqual(set(discretize_symbolic('M')), {(0,)})
        self.assertEqual(set(discretize_symbolic('k_y')), {(0, -1), (0, 1)})
        self.assertEqual(set(discretize_symbolic('k_x', dim=3)), {(-1, 0, 0), (1, 0, 0)})
        self.assertEqual(discretize_symbolic('k_x - k_x'), {})


class TestDiscretize(unittest.TestCase):

    def test_lattice_symbols(self):
        # k^n -> (2/a sin(ka/2))^n for even n, (sin(ka)/a)^n for odd n
        a, k = 0.7, 0.3
        for n in range(1, 5):
            ks = discretize('k_x**{}'.format(n), a=a)
            ref = (2 / a * np.sin(k * a / 2))**n if n % 2 == 0 else (np.sin(k * a) / a)**n
            np.testing.assert_allclose(ks.get_ham((k,))[0, 0], ref, atol=1e-12)

    def test_mixed_and_3d(self):
        a, k = 0.5, np.array([0.3, -0.4, 0.2])
        ks = discretize('k_x*k_y', a=a)
        np.testing.assert_allclose(ks.get_ham(k[:2])[0, 0], np.sin(k[0]*a) * np.sin(k[1]*a) / a**2)
        ks = discretize('k_x**2 + k_y**2 + k_z**2', a=a)
        self.assertEqual((ks.dim, ks.space_dim), (3, 3))
        np.testing.assert_allclose(ks.get_ham(k)[0, 0], np.sum((2 / a * np.sin(k * a / 2))**2))

    def test_converges_as_a_squared(self):
        k = 0.5
        err = [abs(discretize('k_x**2 + k_x', a=a).get_ham((k,))[0, 0] - (k**2 + k))
               for a in (0.8, 0.4)]
        self.assertAlmostEqual(err[0] / err[1], 4., delta=0.1)

    def test_bhz_parameters(self):
        ks = discretize(BHZ, a=0.5)
        self.assertIsInstance(ks, KSpace)
        self.assertEqual((ks.norb, list(ks.tags)), (2, ['a', 'b']))
        self.assertTrue(ks.is_hermitian())
        np.testing.assert_allclose(ks.get_ham((0., 0.), A=1., B=1., M=0.3), np.diag([0.3, -0.3]))
        # M/B > 0 is the inverted (topological) regime
        ks.set_params(A=1., B=1., M=1.)
        self.assertAlmostEqual(abs(ks.chern_number([0])), 1., places=6)
        ks.set_params(M=-1.)
        self.assertAlmostEqual(ks.chern_number([0]), 0., places=6)
        with self.assertRaises(TypeError):
            discretize(BHZ).get_ham((0., 0.), A=1.)

    def test_string_and_matrix_agree(self):
        kx, ky, A, B, M = sympy.symbols('k_x k_y A B M')
        d = M - B * (kx**2 + ky**2)
        mat = sympy.Matrix([[d, A * (kx + sympy.I * ky)], [A * (kx - sympy.I * ky), -d]])
        k, p = (0.3, 0.2), {'A': 1., 'B': 0.5, 'M': 0.7}
        np.testing.assert_allclose(discretize(mat, a=0.5).get_ham(k, **p),
                                   discretize(BHZ, a=0.5).get_ham(k, **p))
        # kron, and E read as a parameter (not Euler's number)
        two = discretize('kron(sigma_0, k_x*sigma_x + E*sigma_z)', tags='abcd')
        self.assertEqual(two.norb, 4)
        np.testing.assert_allclose(two.get_ham((0.,), E=0.4).diagonal(), [0.4, -0.4, 0.4, -0.4])

    def test_constant_terms(self):
        ks = discretize(sympy.Matrix([[1, 2j], [-2j, -1]]))
        np.testing.assert_allclose(ks.onsite, [1, -1])
        np.testing.assert_allclose(ks.get_ham((0.4,)), [[1, 2j], [-2j, -1]])

    def test_non_hermitian(self):
        ks = discretize('k_x**2 + I*g + I*k_x')
        self.assertTrue(ks._nonreciprocal)
        self.assertFalse(ks.is_hermitian())
        k = 0.3
        np.testing.assert_allclose(ks.get_ham((k,), g=0.2)[0, 0],
                                   2 * (1 - np.cos(k)) + 0.2j + 1j * np.sin(k))
        self.assertTrue(discretize('I*k_x**2')._nonreciprocal)

    def test_finite_system_matches_mesh(self):
        ks = discretize(BHZ, a=0.5)
        ks.set_params(A=1., B=1., M=1.)
        sys_ = finite_system(ks, (5, 5), periodic=True)
        sys_.get_ham()
        sys_.get_eig()
        np.testing.assert_allclose(np.sort(sys_.en.real), np.sort(ks.mesh_bands((5, 5)).ravel()),
                                   atol=1e-10)

    def test_errors(self):
        self.assertRaises(TypeError, discretize, 1.)
        self.assertRaises(ValueError, discretize, 'k_x +')
        self.assertRaises(TypeError, discretize, '(k_x, k_y)')
        self.assertRaises(ValueError, discretize, sympy.Matrix([[1, 2]]))
        self.assertRaises(ValueError, discretize, 'k_x**2 + V*x')
        self.assertRaises(ValueError, discretize, 'sin(k_x)')
        self.assertRaises(ValueError, discretize, 'k_y', dim=1)
        self.assertRaises(ValueError, discretize, 'k_x', a=0.2)
        self.assertRaises(ValueError, discretize, 'k_x', tags='ab')

    def test_sympy_missing(self):
        saved = sys.modules.get('sympy')
        sys.modules['sympy'] = None
        try:
            self.assertRaises(ImportError, discretize, 'k_x')
        finally:
            sys.modules['sympy'] = saved


if __name__ == '__main__':
    unittest.main()
