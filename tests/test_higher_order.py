"""
Higher-order topology: nested Wilson loops, the BBH quadrupole moment and
corner charges (Benalcazar, Bernevig and Hughes, Science 357, 61 (2017)).
"""
import unittest

import numpy as np
import scipy.linalg as LA

import tbkit.lattices as lattices
from tbkit.higher_order import (bbh_model, wannier_bands, wannier_sector_polarization,
                                                    quadrupole_moment, flake_positions, corner_charges,
                                                    CornerCharges)
from tbkit.kspace import KSpace, PAULI
from tbkit.lattice import Lattice


def circ(a, b):
    '''Distance modulo 1.'''
    d = abs(a - b) % 1.
    return min(d, 1. - d)


def helical_model():
    '''Two Qi-Wu-Zhang layers with opposite Chern numbers.'''
    lat = Lattice(unit_cell=[{'tag': 'a', 'r0': (0., 0.)}, {'tag': 'b', 'r0': (0.5, 0.5)}],
                        prim_vec=[(1., 0.), (0., 1.)])
    ks = KSpace(lat, spin=True)
    hops = []
    for site, s in ((0, 1), (1, -1)):
        hops.append({'i': site, 'j': site, 'R': (1, 0), 't': PAULI['x'] / 2j + PAULI['z'] / 2})
        hops.append({'i': site, 'j': site, 'R': (0, 1), 't': s * PAULI['y'] / 2j + PAULI['z'] / 2})
    ks.set_hopping(hops)
    ks.set_onsite({'a': -1. * PAULI['z'], 'b': -1. * PAULI['z']})
    return ks


class TestBBHModel(unittest.TestCase):

    def test_bloch_hamiltonian(self):
        # H(k) read back from the hoppings equals the Gamma-matrix formula
        g, lam, delta = 0.3, 1.1, 0.2
        m = bbh_model(g, lam, delta)
        s0, sx = np.eye(2), np.array([[0., 1.], [1., 0.]])
        sy, sz = np.array([[0., -1j], [1j, 0.]]), np.diag([1., -1.])
        g0 = np.kron(sz, s0)
        g1, g2, g3 = (-np.kron(sy, s) for s in (sx, sy, sz))
        g4 = np.kron(sx, s0)
        for kx, ky in [(0.3, -1.2), (2.5, 0.7), (0., 0.)]:
            ref = ((g + lam * np.cos(kx)) * g4 + lam * np.sin(kx) * g3
                     + (g + lam * np.cos(ky)) * g2 + lam * np.sin(ky) * g1 + delta * g0)
            np.testing.assert_allclose(m.get_ham([kx, ky]), ref, atol=1e-13)

    def test_bulk_spectrum(self):
        # E = +-sqrt((g + l cos kx)^2 + l^2 sin^2 kx + (g + l cos ky)^2 + l^2 sin^2 ky),
        # each twice degenerate; the gap closes at |g| = |l|
        g, lam = 0.5, 1.
        m = bbh_model(g, lam)
        for k in [(0.4, 1.3), (np.pi, 0.), (0., 0.)]:
            e = np.sqrt(g**2 + lam**2 + 2*g*lam*np.cos(k[0]) + g**2 + lam**2 + 2*g*lam*np.cos(k[1]))
            np.testing.assert_allclose(np.linalg.eigvalsh(m.get_ham(list(k))), [-e, -e, e, e], atol=1e-12)


class TestNestedWilsonLoop(unittest.TestCase):

    def test_quadrupole_phase(self):
        # q_xy = 1/2 and p_y^{nu_x-} = p_x^{nu_y-} = 1/2 for |gamma| < |lambda|
        for g in (0., 0.5, -0.5, 0.9):
            m = bbh_model(g, 1.)
            self.assertLess(circ(quadrupole_moment(m, [0, 1], nk=20), 0.5), 1e-8)
            for d in (0, 1):
                for sector in (0, 1):
                    p = wannier_sector_polarization(m, [0, 1], sector, nk=20, direction=d)
                    self.assertLess(circ(p, 0.5), 1e-8)

    def test_trivial_phase(self):
        for g in (1.5, -2.):
            m = bbh_model(g, 1.)
            self.assertLess(circ(quadrupole_moment(m, [0, 1], nk=16), 0.), 1e-8)
            self.assertLess(circ(wannier_sector_polarization(m, [0, 1], nk=16), 0.), 1e-8)

    def test_positions_gauge(self):
        # the quantized values do not depend on the orbital positions
        for g, q in ((0.5, 0.5), (1.5, 0.)):
            m = bbh_model(g, 1.)
            self.assertLess(circ(quadrupole_moment(m, [0, 1], nk=16, positions=False), q), 1e-8)
            self.assertLess(circ(wannier_sector_polarization(m, [0, 1], nk=(16, 12), positions=False), q), 1e-8)

    def test_wannier_bands(self):
        m = bbh_model(0.5, 1.)
        nu, w = wannier_bands(m, [0, 1], nk=(12, 10))
        self.assertEqual(nu.shape, (12, 10, 2))
        self.assertEqual(w.shape, (12, 10, 4, 2))
        # independent of the base point along the loop, and equal to the
        # hybrid Wannier centres of KSpace.wannier_centers
        np.testing.assert_allclose(nu, np.broadcast_to(nu[:1], nu.shape), atol=1e-10)
        for j in range(10):
            ref = m.wannier_centers([0, 1], nk=12, direction=0, k_perp=j / 10)
            ref = np.sort(np.where(ref > 0.5, ref - 1., ref))
            np.testing.assert_allclose(nu[0, j], ref, atol=1e-10)
        # symmetric Wannier bands +-nu, gapped at 0 and 1/2
        np.testing.assert_allclose(nu[..., 0], -nu[..., 1], atol=1e-10)
        self.assertGreater(nu[..., 1].min(), 0.2)
        self.assertLess(nu[..., 1].max(), 0.4)
        # sector states: orthonormal, in the occupied subspace
        overlap = w[3, 4].conj().T @ w[3, 4]
        np.testing.assert_allclose(overlap, np.eye(2), atol=1e-10)
        nu_y, w_y = wannier_bands(m, [0, 1], nk=(12, 10), direction=1)
        self.assertEqual(nu_y.shape, (12, 10, 2))
        np.testing.assert_allclose(nu_y, np.broadcast_to(nu_y[:, :1], nu_y.shape), atol=1e-10)

    def test_errors(self):
        m = bbh_model()
        self.assertRaises(TypeError, wannier_bands, 'model', [0, 1])
        self.assertRaises(ValueError, wannier_bands, KSpace(lattices.chain()), [0])
        lossy = bbh_model()
        lossy.set_onsite({'a': 1j})
        self.assertRaises(ValueError, wannier_bands, lossy, [0, 1])
        self.assertRaises(ValueError, wannier_bands, m, [0, 4])
        self.assertRaises(ValueError, wannier_bands, m, [0, 1], nk=(10,))
        self.assertRaises(ValueError, wannier_bands, m, [0, 1], direction=2)
        self.assertRaises(TypeError, wannier_bands, m, [0, 1], positions=1)
        self.assertRaises(ValueError, wannier_sector_polarization, m, [0, 1], [2])
        # two Chern layers of opposite C: their Wannier bands wind in
        # opposite directions and cross, so there is no Wannier gap
        self.assertRaises(ValueError, wannier_sector_polarization, helical_model(), [0, 1], None, 16)
        self.assertRaises(TypeError, bbh_model, 'a')
        # one band: the sector is the whole group, no gap to check
        chain2d = KSpace(lattices.square())
        chain2d.set_hopping([{'i': 0, 'j': 0, 'R': (1, 0), 't': 1.}])
        self.assertLess(circ(wannier_sector_polarization(chain2d, 0, nk=8), 0.), 1e-10)


class TestCornerCharges(unittest.TestCase):

    def test_corner_states_dimerized_limit(self):
        # gamma = 0: isolated pi-flux plaquettes (E = +-sqrt(2) lambda, twice),
        # edge dimers (+-lambda) and four free corner sites (E = 0)
        n = 6
        en = np.linalg.eigvalsh(bbh_model(0., 1.).finite_ham(n))
        expected = np.sort(np.concatenate([np.zeros(4), np.repeat([-1., 1.], 4 * (n - 1)),
                                                              np.repeat([-np.sqrt(2), np.sqrt(2)], 2 * (n - 1) ** 2)]))
        np.testing.assert_allclose(en, expected, atol=1e-12)

    def test_four_corner_states(self):
        n = 14
        m = bbh_model(0.5, 1.)
        en, vec = LA.eigh(m.finite_ham(n))
        zero = np.abs(en) < 1e-3
        self.assertEqual(zero.sum(), 4)
        # in the gap of the bulk and of the edges
        self.assertGreater(np.sort(np.abs(en))[4], 0.3)
        # localized at the four corners of the flake, with amplitudes
        # decaying as (gamma/lambda)^(n1 + n2) cells away from the corner:
        # the r x r corner cells hold the weight (1 - (gamma/lambda)^(2r))^2
        pos = flake_positions(m, n)
        weight = (np.abs(vec[:, zero]) ** 2).sum(axis=1)
        center = pos.mean(axis=0)
        for r in (1, 2, 3):
            corner = np.all(np.abs(pos - center) > (n / 2 - r), axis=1)
            self.assertAlmostEqual(weight[corner].sum(), 4 * (1 - 0.25 ** r) ** 2, places=5)
        # trivial phase: no in-gap state
        en_triv = np.linalg.eigvalsh(bbh_model(1.5, 1.).finite_ham(n))
        self.assertGreater(np.abs(en_triv).min(), 0.4)

    def test_corner_charges(self):
        res = corner_charges(bbh_model(0.5, 1., 1e-3), 16)
        self.assertIsInstance(res, CornerCharges)
        # +-1/2 on alternating corners (delta fills two of the four corner states)
        np.testing.assert_allclose(np.abs(res.charges), 0.5, atol=2e-3)
        np.testing.assert_allclose(res.charges, -res.charges[::-1], atol=1e-10)
        self.assertAlmostEqual(res.charges[0, 0], -res.charges[0, 1], places=8)
        self.assertAlmostEqual(res.background, 0.5)
        self.assertAlmostEqual(res.density.sum(), 16 * 16 * 2, places=8)
        self.assertEqual(res.positions.shape, (16 * 16 * 4, 2))
        self.assertAlmostEqual(res.e_fermi, 0., places=6)
        # trivial: no corner charge
        triv = corner_charges(bbh_model(1.5, 1., 1e-3), 10)
        np.testing.assert_allclose(triv.charges, 0., atol=1e-4)
        # delta = 0: four degenerate zero modes, shared equally -> no charge
        sym = corner_charges(bbh_model(0., 1.), 6)
        np.testing.assert_allclose(sym.charges, 0., atol=1e-10)
        # finite temperature, explicit filling
        warm = corner_charges(bbh_model(0.5, 1., 0.01), (12, 12), n_electrons=288, temperature=1e-3)
        np.testing.assert_allclose(np.abs(warm.charges), 0.5, atol=0.01)
        empty = corner_charges(bbh_model(0.5, 1.), 4, n_electrons=0)
        np.testing.assert_allclose(empty.density, 0.)

    def test_flake_positions(self):
        # n1 runs fastest, then n2
        cubic = KSpace(lattices.square())
        pos = flake_positions(cubic, (3, 2))
        np.testing.assert_allclose(pos, [[0, 0], [1, 0], [2, 0], [0, 1], [1, 1], [2, 1]])
        m = bbh_model()
        pos = flake_positions(m, 2)
        np.testing.assert_allclose(pos[:4], m.orbital_positions())
        np.testing.assert_allclose(pos[4:8], m.orbital_positions() + [1., 0.])
        # consistent with finite_ham: every hopping of the flake has length 1/2
        ham = m.finite_ham(3)
        rows, cols = np.nonzero(np.abs(ham) > 1e-12)
        pos = flake_positions(m, 3)
        np.testing.assert_allclose(np.linalg.norm(pos[rows] - pos[cols], axis=1), 0.5)
        self.assertRaises(TypeError, flake_positions, 'x', 2)
        self.assertRaises(ValueError, flake_positions, m, (2,))

    def test_errors(self):
        m = bbh_model()
        self.assertRaises(ValueError, corner_charges, m, (2,))
        self.assertRaises(ValueError, corner_charges, m, 2, 100)
        self.assertRaises(ValueError, corner_charges, m, 2, 3.5)
        self.assertRaises(ValueError, corner_charges, m, 2, None, -1.)


if __name__ == '__main__':
    unittest.main()
