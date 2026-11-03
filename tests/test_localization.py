"""
Transfer-matrix Lyapunov exponents and localization lengths of strips.
"""
import unittest

import numpy as np

import tbkit.lattices as lattices
from tbkit.kspace import ribbon
from tbkit.localization import lyapunov_exponents
from tbkit.transport import lead_from_kspace

SQUARE = [{'i': 0, 'j': 0, 'R': (1, 0), 't': 1.}, {'i': 0, 'j': 0, 'R': (0, 1), 't': 1.}]


class TestCleanStrips(unittest.TestCase):
    def test_chain_outside_the_band(self):
        res = lyapunov_exponents([[0.]], [[1.]], 3., n_slices=2000, seed=0)
        self.assertLess(abs(res.exponents[0, 0] - np.arccosh(1.5)), 1e-10)
        self.assertLess(abs(res.localization_length[0] - 1 / np.arccosh(1.5)), 1e-9)

    def test_second_neighbours(self):
        # roots of t2 z^4 + t1 z^3 - E z^2 + t1 z + t2 = 0
        t1, t2, e = 1., 0.3, 3.5
        logs = np.log(np.abs(np.roots([t2, t1, -e, t1, t2])))
        exact = np.sort(logs[logs > 0])[::-1]
        res = lyapunov_exponents(0., [[[t1]], [[t2]]], e, n_slices=2000, seed=0)
        self.assertTrue(np.allclose(res.exponents[0], exact, atol=1e-9))

    def test_ribbon_channels(self):
        # open ribbon of 3 cells: channels 2 cos(n pi / 4), each with arccosh(|E - eps_n| / 2)
        h0, v = lead_from_kspace(ribbon(lattices.square(), SQUARE, 3, direction=1))
        eps = 2 * np.cos(np.arange(1, 4) * np.pi / 4)
        res = lyapunov_exponents(h0, v, 5., n_slices=1000, seed=0)
        self.assertTrue(np.allclose(res.exponents[0], np.sort(np.arccosh((5. - eps) / 2))[::-1], atol=1e-9))


class TestDisorderedStrips(unittest.TestCase):
    def test_thouless_weak_disorder(self):
        # 1 / lambda = W^2 / (24 (4 - E^2)) to second order in W
        energies = np.array([0.5, 1.])
        res = lyapunov_exponents([[0.]], [[1.]], energies, disorder=1., n_slices=10 ** 5, seed=1)
        ref = 1 / (24 * (4 - energies ** 2))
        self.assertTrue(np.all(np.abs(res.exponents[:, 0] - ref) < 3 * res.error + 0.02 * ref))
        self.assertTrue(np.all(res.error < 0.05 * ref))
        self.assertTrue(np.allclose(res.length_error, res.error * res.localization_length ** 2))

    def test_energies_share_the_strip(self):
        together = lyapunov_exponents(0., [[[1.]], [[0.3]]], [0.1, 1.3], disorder=2., n_slices=4000, seed=3)
        for n, e in enumerate((0.1, 1.3)):
            alone = lyapunov_exponents(0., [[[1.]], [[0.3]]], e, disorder=2., n_slices=4000, seed=3)
            self.assertTrue(np.allclose(together.exponents[n], alone.exponents[0], rtol=1e-12))
            self.assertAlmostEqual(together.error[n], alone.error[0], places=12)

    def test_qr_every(self):
        once = lyapunov_exponents(0., 1., 0.3, disorder=2., n_slices=4000, seed=4)
        every4 = lyapunov_exponents(0., 1., 0.3, disorder=2., n_slices=4000, seed=4, qr_every=4, n_warmup=0)
        self.assertGreater(once.exponents[0, 0], 0.)
        self.assertLess(abs(once.exponents[0, 0] - every4.exponents[0, 0]), 4 * (once.error[0] + every4.error[0]))


class TestChecks(unittest.TestCase):
    def test_arguments(self):
        self.assertRaises(ValueError, lyapunov_exponents, [[0., 1.], [0., 0.]], np.eye(2), 0.)
        self.assertRaises(ValueError, lyapunov_exponents, [[0.]], np.eye(2), 0.)
        self.assertRaises(ValueError, lyapunov_exponents, np.zeros((2, 2)), [[1., 1.], [1., 1.]], 0.)
        self.assertRaises(TypeError, lyapunov_exponents, [[0.]], [[1.]], 1j)
        self.assertRaises(ValueError, lyapunov_exponents, [[0.]], [[1.]], 0., disorder=-1.)
        self.assertRaises(TypeError, lyapunov_exponents, [[0.]], [[1.]], 0., n_chunks=2.)
        self.assertRaises(ValueError, lyapunov_exponents, [[0.]], [[1.]], 0., n_chunks=1)
        self.assertRaises(ValueError, lyapunov_exponents, [[0.]], [[1.]], 0., n_slices=10, qr_every=2, n_chunks=6)


if __name__ == '__main__':
    unittest.main()
