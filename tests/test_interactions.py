"""
Non-collinear Hubbard mean field, and the self-consistent s-wave BdG gap equation.
"""
import unittest

import numpy as np
from scipy.optimize import brentq

import tbkit.lattices as lattices
from tbkit.bdg import s_wave_gap, GapResult
from tbkit.kspace import KSpace, PAULI
from tbkit.meanfield import hubbard_mean_field, hubbard_mean_field_noncollinear, NonCollinearResult


def honeycomb_flake(n=3):
    ks = KSpace(lattices.honeycomb())
    ks.set_hopping([{'i': 0, 'j': 1, 'R': (0, 0), 't': -1.}, {'i': 0, 'j': 1, 'R': (-1, 0), 't': -1.},
                            {'i': 0, 'j': 1, 'R': (0, -1), 't': -1.}])
    return ks.finite_ham((n, n)).real


def triangular_torus(n=3):
    tri = KSpace(lattices.triangular())
    tri.set_hopping([{'i': 0, 'j': 0, 'R': (1, 0), 't': 1.}, {'i': 0, 'j': 0, 'R': (0, 1), 't': 1.},
                            {'i': 0, 'j': 0, 'R': (1, -1), 't': 1.}])
    return tri.finite_ham((n, n), periodic=True).real


def square_torus(n):
    sq = KSpace(lattices.square())
    sq.set_hopping([{'i': 0, 'j': 0, 'R': (1, 0), 't': -1.}, {'i': 0, 'j': 0, 'R': (0, 1), 't': -1.}])
    return sq.finite_ham(n, periodic=True).real, sq.mesh_bands(n)[:, 0]


class TestNonCollinear(unittest.TestCase):

    def test_collinear_limit(self):
        # moments along z stay along z and reproduce hubbard_mean_field
        h = honeycomb_flake()
        n = len(h)
        stag = np.where(np.arange(n) % 2 == 0, 0.2, -0.2)
        col = hubbard_mean_field(h, 3., n, n_up=0.5 + stag, n_dn=0.5 - stag)
        mag = np.zeros((n, 3))
        mag[:, 2] = stag
        nc = hubbard_mean_field_noncollinear(h, 3., n, magnetization=mag)
        self.assertIsInstance(nc, NonCollinearResult)
        self.assertAlmostEqual(nc.energy, col.energy, places=8)
        np.testing.assert_allclose(nc.magnetization[:, 2], col.magnetization, atol=1e-7)
        np.testing.assert_allclose(nc.magnetization[:, :2], 0., atol=1e-12)
        np.testing.assert_allclose(nc.density, col.n_up + col.n_dn, atol=1e-7)
        self.assertAlmostEqual(nc.e_fermi, col.e_fermi, places=6)
        # spin-rotation invariance: the same solution along x
        mag_x = np.roll(mag, -2, axis=1)
        ncx = hubbard_mean_field_noncollinear(h, 3., n, magnetization=mag_x)
        self.assertAlmostEqual(ncx.energy, col.energy, places=8)
        np.testing.assert_allclose(ncx.magnetization[:, 0], col.magnetization, atol=1e-7)
        # a spinful (2N, 2N) input gives the same
        ncs = hubbard_mean_field_noncollinear(np.kron(h, np.eye(2)), 3., n, magnetization=mag,
                                                               spinful=True)
        self.assertAlmostEqual(ncs.energy, col.energy, places=8)

    def test_120_degree_order(self):
        # half-filled triangular lattice: the three-sublattice 120-degree
        # state, reached from random starts, below the collinear up-up-down
        # state; at large U both approach the Heisenberg limit
        # E = -(t^2/U) sum_bonds (1 - m_i.m_j) (1.5 and 4/3 per bond)
        h = triangular_torus()
        U = 40.
        best = min((hubbard_mean_field_noncollinear(h, U, 9, seed=s) for s in range(3)),
                       key=lambda r: r.energy)
        m = best.magnetization
        u = m / np.linalg.norm(m, axis=1)[:, None]
        cos = (u @ u.T)[np.abs(h) > 0]
        np.testing.assert_allclose(cos, -0.5, atol=1e-4)
        np.testing.assert_allclose(best.total_spin, 0., atol=1e-6)
        np.testing.assert_allclose(np.linalg.norm(m, axis=1), np.linalg.norm(m[0]), atol=1e-8)
        cells = np.array([(i, j) for j in range(3) for i in range(3)])
        mag = np.zeros((9, 3))
        mag[:, 2] = np.where((cells[:, 0] - cells[:, 1]) % 3 == 0, -0.3, 0.3)
        uud = hubbard_mean_field_noncollinear(h, U, 9, magnetization=mag)
        self.assertLess(best.energy, uud.energy)
        self.assertAlmostEqual(best.energy * U / 27, -1.5, delta=0.03)
        self.assertAlmostEqual(uud.energy * U / 27, -4 / 3, delta=0.03)

    def test_spin_orbit_and_options(self):
        # a Rashba-like spin-flip hopping on a spinful chain, finite temperature
        n = 6
        h = np.zeros((2 * n, 2 * n), 'c16')
        for i in range(n - 1):
            block = -PAULI['0'] + 0.3j * PAULI['y']
            h[2 * i:2 * i + 2, 2 * i + 2:2 * i + 4] = block
            h[2 * i + 2:2 * i + 4, 2 * i:2 * i + 2] = block.conj().T
        res = hubbard_mean_field_noncollinear(h, 2., 5.5, temperature=0.05, spinful=True, seed=1)
        self.assertAlmostEqual(res.density.sum(), 5.5, places=6)
        self.assertEqual(res.states.shape, (2 * n, 2 * n))
        self.assertEqual(res.total_spin.shape, (3,))
        np.testing.assert_allclose(res.rho, res.rho.conj().transpose(0, 2, 1), atol=1e-12)
        res_u = hubbard_mean_field_noncollinear(h, np.full(n, 2.), 6, spinful=True, seed=1)
        self.assertAlmostEqual(res_u.density.sum(), 6., places=8)

    def test_errors(self):
        h = honeycomb_flake(2)
        n = len(h)
        f = hubbard_mean_field_noncollinear
        self.assertRaises(ValueError, f, 1j * h, 1., n)
        self.assertRaises(ValueError, f, np.eye(3), 1., 3, spinful=True)
        self.assertRaises(TypeError, f, h, 1., n, spinful=1)
        self.assertRaises(ValueError, f, h, [1., 2.], n)
        self.assertRaises(ValueError, f, h, 1., 2.5)
        self.assertRaises(ValueError, f, h, 1., n, magnetization=np.zeros((n, 2)))
        self.assertRaises(ValueError, f, h, 1., n, mixing=2.)
        self.assertRaises(RuntimeError, f, triangular_torus(), 12., 9, tol=1e-14, max_iter=3)


class TestGapEquation(unittest.TestCase):

    @staticmethod
    def bcs(xi, V, T):
        '''Uniform BCS gap on the same k-mesh (brentq on the gap equation).'''
        def eq(d):
            e = np.sqrt(xi ** 2 + d ** 2)
            return V / len(xi) * np.sum((np.tanh(e / (2 * T)) if T > 0 else 1.) / (2 * e)) - 1
        return brentq(eq, 1e-9, 5.)

    def test_clean_lattice_is_bcs(self):
        # on a clean torus, the real-space solution is uniform and solves
        # the k-space BCS equation on the same mesh
        h, eps = square_torus(8)
        mu, V = -0.5, 2.5
        for T in (0., 0.15):
            res = s_wave_gap(h, V, mu, temperature=T)
            self.assertIsInstance(res, GapResult)
            np.testing.assert_allclose(np.abs(res.delta), res.mean_gap, atol=1e-8)
            self.assertAlmostEqual(res.mean_gap, self.bcs(eps - mu, V, T), places=6)
        # quasiparticle gap = Delta (mu inside the band)
        self.assertGreater(np.min(np.abs(res.energies)), 0.)
        np.testing.assert_allclose(np.sort(res.energies), -np.sort(res.energies)[::-1], atol=1e-10)

    def test_bcs_ratio(self):
        # Delta(T)^2 vanishes linearly at T_c; the extrapolated T_c matches the
        # linearized gap equation, and Delta(0) / T_c = 1.764 (weak coupling)
        h, eps = square_torus(12)
        mu, V = -0.5, 2.5
        xi = eps - mu
        tc = brentq(lambda T: V / len(xi) * np.sum(np.tanh(xi / (2 * T)) / (2 * xi)) - 1, 1e-3, 3.)
        d0 = s_wave_gap(h, V, mu).mean_gap
        ts = np.array([0.85, 0.95]) * d0 / 1.764
        d2 = np.array([s_wave_gap(h, V, mu, temperature=T, delta0=d0).mean_gap ** 2 for T in ts])
        tc_fit = ts[0] - d2[0] * (ts[1] - ts[0]) / (d2[1] - d2[0])
        self.assertAlmostEqual(tc_fit / tc, 1., delta=0.01)
        self.assertAlmostEqual(d0 / tc_fit, 1.764, delta=0.02 * 1.764)
        above = s_wave_gap(h, V, mu, temperature=1.2 * tc, delta0=d0, tol=1e-6)
        self.assertLess(above.mean_gap, 1e-4)

    def test_inhomogeneous(self):
        # attraction on half of a chain only: the gap leaks into the normal
        # half (proximity effect) and decays there
        n = 40
        h = -(np.eye(n, k=1) + np.eye(n, k=-1))
        V = np.where(np.arange(n) < n // 2, 2.5, 0.)
        res = s_wave_gap(h, V, 0.3, mixing=0.7, delta0=np.full(n, 0.2 + 0j))
        self.assertGreater(abs(res.delta[5]), 0.1)
        np.testing.assert_allclose(res.delta[n // 2:], 0., atol=1e-12)  # Delta_i = V_i F_i

    def test_errors(self):
        h, _ = square_torus(4)
        self.assertRaises(ValueError, s_wave_gap, 1j * h, 1.)
        self.assertRaises(ValueError, s_wave_gap, h, -1.)
        self.assertRaises(ValueError, s_wave_gap, h, np.ones(3))
        self.assertRaises(ValueError, s_wave_gap, h, np.full(16, -1.))
        self.assertRaises(TypeError, s_wave_gap, h, 1., 'a')
        self.assertRaises(ValueError, s_wave_gap, h, 1., 0., -1.)
        self.assertRaises(ValueError, s_wave_gap, h, 1., 0., 0., np.ones(3))
        self.assertRaises(TypeError, s_wave_gap, h, 1., 0., 0., 'x')
        self.assertRaises(RuntimeError, s_wave_gap, h, 2., -0.5, 0., 0.1, 1., 1e-14, 2)


if __name__ == '__main__':
    unittest.main()
