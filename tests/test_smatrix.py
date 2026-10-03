"""
Scattering-matrix transport: lead modes, the S-matrix, scattering states
and the local density of states (tbkit.transport).
"""
import unittest

import numpy as np
import scipy.sparse as sp

from tbkit.transport import SMatrix, Transport, lead_modes
from tests.test_transport import chain_device, ribbon_device, ONE

DIMER_H0 = np.array([[0., 1.], [1., 0.]])
DIMER_V = np.array([[0., 0.], [0.5, 0.]])  # singular: only the inter-cell bond


def dimer_device(n=6, impurity=0.7):
    '''A dimerized chain of n cells (intra 1, inter 0.5), with leads of the same chain.'''
    ham = (np.kron(np.eye(n), DIMER_H0) + np.kron(np.eye(n, k=1), DIMER_V)
           + np.kron(np.eye(n, k=-1), DIMER_V.T))
    ham[5, 5] = impurity
    tr = Transport(ham)
    tr.add_lead(DIMER_H0, DIMER_V.T, DIMER_V.T, [0, 1])
    tr.add_lead(DIMER_H0, DIMER_V, DIMER_V, [2 * n - 2, 2 * n - 1])
    return tr


def ns_junction(gap, n=300, barrier=0.):
    r'''
    A BdG chain (orbitals e, h per site; t = 1, half filling) that turns
    superconducting from site 5 on, with a normal lead: Q = tau_z.
    '''
    ham = np.kron(np.eye(n, k=1) + np.eye(n, k=-1), np.diag([-1., 1.]))
    ham += np.kron(np.diag((np.arange(n) >= 5) * gap), [[0., 1.], [1., 0.]])
    ham[0, 0], ham[1, 1] = barrier, -barrier
    tr = Transport(ham)
    tr.add_lead(np.zeros((2, 2)), np.diag([-1., 1.]), np.diag([-1., 1.]), [0, 1],
                conservation_law=np.diag([1., -1.]))
    return tr


def spin_device():
    '''A spin-degenerate chain (degenerate lead modes) with a spin-mixing scatterer.'''
    ham = np.kron(np.eye(8, k=1) + np.eye(8, k=-1), np.eye(2))
    ham[4, 4], ham[5, 5], ham[4, 5], ham[5, 4] = 0.5, -0.3, 0.2, 0.2
    tr = Transport(ham)
    tr.add_lead(np.zeros((2, 2)), np.eye(2), np.eye(2), [0, 1])
    tr.add_lead(np.zeros((2, 2)), np.eye(2), np.eye(2), [14, 15])
    return tr


class TestLeadModes(unittest.TestCase):

    def test_chain(self):
        # E = 2 cos k: modes e^{+-ik}, velocities -+2 sin k, unit current
        e = 0.5
        k = np.arccos(e / 2)
        modes = lead_modes([[0.]], [[1.]], e)
        self.assertTrue(np.allclose(modes.momenta, [k, -k]))
        self.assertTrue(np.allclose(modes.velocities, [-2 * np.sin(k), 2 * np.sin(k)]))
        self.assertTrue(np.allclose(np.abs(modes.wave_functions[0]) ** 2 * np.abs(modes.velocities), 1.))
        # outside the band: no propagating mode
        self.assertEqual(lead_modes([[0.]], [[1.]], 2.5).wave_functions.shape, (1, 0))

    def test_singular_coupling(self):
        # inside a band of the dimerized chain, outside it (in the gap)
        modes = lead_modes(DIMER_H0, DIMER_V, 1.)
        self.assertEqual(len(modes.momenta), 2)
        self.assertAlmostEqual(modes.velocities[0], -modes.velocities[1])
        self.assertEqual(len(lead_modes(DIMER_H0, DIMER_V, 0.3).momenta), 0)

    def test_degenerate(self):
        # two spin copies: the velocity is diagonal in the degenerate modes
        modes = lead_modes(np.zeros((2, 2)), np.eye(2), 0.4)
        self.assertEqual(len(modes.momenta), 4)
        self.assertTrue(np.allclose(np.abs(modes.velocities), 2 * np.sin(np.arccos(0.2))))

    def test_checks(self):
        self.assertRaises(ValueError, lead_modes, [[0.]], [[1.]], 2.)  # band edge
        self.assertRaises(ValueError, lead_modes, [[0., 1.], [0., 0.]], np.eye(2), 0.)
        self.assertRaises(ValueError, lead_modes, [[0.]], np.eye(2), 0.)
        self.assertRaises(TypeError, lead_modes, [[0.]], [[1.]], 'a')


class TestSMatrix(unittest.TestCase):

    def test_impurity_and_band_edge(self):
        # T = v^2 / (v^2 + eps^2), exact up to the band edge (no eta)
        tr = chain_device(impurity={5: 0.8})
        es = np.array([-1.5, 0.3, 1.2, 1.999999])
        v = 2 * np.sin(np.arccos(es / 2))
        self.assertTrue(np.allclose(tr.transmission(es), v**2 / (v**2 + 0.64), rtol=1e-8, atol=0))
        self.assertEqual(tr.transmission([3.])[0], 0.)

    def test_unitary_and_caroli(self):
        for tr, e in ((ribbon_device(6, 8, constriction=(4., (2., 3.))), 0.1),
                      (dimer_device(), 1.0), (spin_device(), 0.4)):
            s = tr.smatrix(e)
            self.assertIsInstance(s, SMatrix)
            n = s.data.shape[1]
            self.assertTrue(np.allclose(s.data.conj().T @ s.data, np.eye(n), atol=1e-12))
            t = s.transmission(1, 0)
            self.assertAlmostEqual(t + s.transmission(0, 0), s.num_propagating(0), places=12)
            self.assertAlmostEqual(t, tr.transmission([e])[0], places=12)
            self.assertAlmostEqual(t, tr.transmission([e], eta=1e-9)[0], places=6)
            self.assertEqual(s.submatrix(1, 0).shape, (s.num_propagating(1), s.num_propagating(0)))

    def test_multi_terminal(self):
        from tests.test_transport_response import hall_bar
        bar = hall_bar()
        s = bar.smatrix(-3.2)
        t = np.array([[s.transmission(p, q) if p != q else 0. for q in range(6)] for p in range(6)])
        self.assertTrue(np.allclose(t, bar.transmission_matrix(-3.2), atol=1e-6))

    def test_large_sparse_device(self):
        # a 20 000-site strip: never densified
        from tests.test_transport_response import strip, strip_leads
        lat, sys, rib = strip(400, 50, disorder=1.)
        cols, (h_l, v_l), (h_r, v_r) = strip_leads(lat, 400, rib)
        tr = Transport(sys.ham)
        tr.add_lead(h_l, v_l, np.eye(50), cols[0])
        tr.add_lead(h_r, v_r, np.eye(50), cols[1])
        t = tr.transmission([-0.9])[0]
        self.assertTrue(0. < t < 50.)
        self.assertIsNone(tr._dense)

    def test_fallbacks(self):
        tr = chain_device(impurity={5: 0.8})
        # the same lead: the Caroli quantity, as before
        self.assertAlmostEqual(tr.transmission([0.3], 0, 0)[0], tr.transmission([0.3], 0, 0, 1e-9)[0])
        # a non-Hermitian lead: Caroli with eta = 1e-9
        lossy = Transport(np.zeros((1, 1)))
        lossy.add_lead([[-0.1j]], [[1.]], ONE, [0])
        lossy.add_lead([[0.]], [[1.]], ONE, [0])
        self.assertAlmostEqual(lossy.transmission([0.5])[0], lossy.transmission([0.5], eta=1e-9)[0])
        self.assertRaises(ValueError, lossy.smatrix, 0.5)
        # no leads
        self.assertEqual(Transport(np.zeros((2, 2))).smatrix(0.).data.shape, (0, 0))

    def test_ham_property(self):
        ham = sp.random(5, 5, density=0.4, random_state=1) + 0j
        tr = Transport(ham)
        self.assertTrue(np.allclose(tr.ham, ham.toarray()))
        self.assertIs(tr.ham, tr.ham)

    def test_checks(self):
        tr = chain_device()
        s = tr.smatrix(0.5)
        self.assertRaises(ValueError, s.submatrix, 2, 0)
        self.assertRaises(TypeError, s.transmission, 0, 1.)
        self.assertRaises(ValueError, tr.smatrix, 2.)
        self.assertRaises(TypeError, tr.smatrix, 'a')
        self.assertRaises(ValueError, tr.wave_function, 0.5, 2)


class TestConservationLaw(unittest.TestCase):

    def test_spin_blocks(self):
        # sigma_z: the spin-flip blocks are those of the spin-mixing bond only
        tr = spin_device()
        sz = np.diag([1., -1.])
        modes = lead_modes(np.zeros((2, 2)), np.eye(2), 0.4, sz)
        self.assertEqual(list(modes.blocks), [0, 1, 0, 1])
        law = Transport(tr._ham)
        for h0, v, tau, sites in tr.leads:
            law.add_lead(h0, v, tau, sites, conservation_law=sz)
        s, plain = law.smatrix(0.4), tr.smatrix(0.4)
        self.assertEqual(s.num_propagating((0, 0)), 1)
        flips = s.transmission((1, 0), (0, 1)) + s.transmission((1, 1), (0, 0))
        self.assertGreater(flips, 1e-3)
        total = sum(s.transmission((1, a), (0, b)) for a in (0, 1) for b in (0, 1))
        self.assertAlmostEqual(total, plain.transmission(1, 0), places=12)
        self.assertAlmostEqual(total, law.transmission([0.4])[0], places=12)
        # a spin-conserving scatterer: no flips, and each spin its own impurity formula
        ham = np.kron(np.eye(8, k=1) + np.eye(8, k=-1), np.eye(2))
        ham[8, 8], ham[9, 9] = 0.8, -0.5
        conserving = Transport(ham)
        conserving.add_lead(np.zeros((2, 2)), np.eye(2), np.eye(2), [0, 1], conservation_law=sz)
        conserving.add_lead(np.zeros((2, 2)), np.eye(2), np.eye(2), [14, 15], conservation_law=sz)
        s = conserving.smatrix(0.4)
        v2 = 4 - 0.4**2
        self.assertLess(s.transmission((1, 0), (0, 1)), 1e-24)
        self.assertAlmostEqual(s.transmission((1, 1), (0, 1)), v2 / (v2 + 0.64), places=12)
        self.assertAlmostEqual(s.transmission((1, 0), (0, 0)), v2 / (v2 + 0.25), places=12)

    def test_andreev_reflection(self):
        # perfect NS interface at half filling: R_he(0) = 1 / (1 + (Delta / 2t)^2),
        # and every electron comes back, as an electron or as a hole
        for gap in (0.1, 0.4):
            s = ns_junction(gap).smatrix(0.)
            self.assertAlmostEqual(s.transmission((0, 0), (0, 1)), 1 / (1 + gap**2 / 4), places=10)
            self.assertAlmostEqual(s.transmission((0, 0), (0, 1)) + s.transmission((0, 1), (0, 1)), 1.,
                                   places=12)
            self.assertEqual(s.submatrix((0, 0), (0, 1)).shape, (1, 1))
        # a barrier suppresses it in the gap but not at its edge (BTK)
        s0, edge = ns_junction(0.05, barrier=1.).smatrix(0.), ns_junction(0.05, barrier=1.).smatrix(0.049)
        self.assertLess(s0.transmission((0, 0), (0, 1)), 0.5)
        self.assertGreater(edge.transmission((0, 0), (0, 1)), 0.99)

    def test_checks(self):
        tr = Transport(np.zeros((2, 2)))
        self.assertRaises(ValueError, tr.add_lead, np.zeros((2, 2)), np.eye(2), np.eye(2), [0, 1],
                          np.eye(3))
        self.assertRaises(ValueError, tr.add_lead, np.zeros((2, 2)), np.eye(2), np.eye(2), [0, 1],
                          [[0., 1.], [0., 0.]])
        self.assertRaises(ValueError, tr.add_lead, [[0., 1.], [1., 0.]], np.eye(2), np.eye(2), [0, 1],
                          np.diag([1., -1.]))
        self.assertRaises(ValueError, lead_modes, [[0., 1.], [1., 0.]], np.eye(2), 0., np.diag([1., -1.]))
        s = ns_junction(0.1).smatrix(0.)
        self.assertRaises(ValueError, s.submatrix, (0, 2), 0)
        self.assertRaises(ValueError, s.submatrix, (1, 0), 0)
        self.assertRaises(TypeError, s.submatrix, (0, 0, 1), 0)
        self.assertRaises(TypeError, s.submatrix, (0, 0.), 0)
        self.assertRaises(ValueError, chain_device().smatrix(0.5).submatrix, (0, 1), 0)


class TestTemperature(unittest.TestCase):

    def test_wiedemann_franz_and_mott(self):
        t = 0.01
        g, s, kappa = chain_device().thermoelectric(0.3, t)
        self.assertAlmostEqual(g, 1., places=12)
        self.assertAlmostEqual(s, 0., places=12)
        self.assertAlmostEqual(kappa / (t * g), np.pi**2 / 3, places=8)
        # an impurity: the Mott formula S = -pi^2/3 T d ln T / dE
        tr = chain_device(impurity={5: 0.8})
        mu, h = 0.3, 1e-4
        g, s, kappa = tr.thermoelectric(mu, t)
        dlog = np.diff(np.log(tr.transmission([mu - h, mu + h])))[0] / (2 * h)
        self.assertAlmostEqual(s / (-np.pi**2 / 3 * t * dlog), 1., delta=2e-3)  # O(T^2)
        self.assertAlmostEqual(kappa / (t * g), np.pi**2 / 3, places=3)
        self.assertAlmostEqual(g, tr.conductance(mu, t), places=14)
        # nothing transmitted: no thermopower
        g, s, kappa = tr.thermoelectric(6., 0.1)  # the Fermi window misses the band
        self.assertEqual(g, 0.)
        self.assertTrue(np.isnan(s))

    def test_conductance(self):
        tr = chain_device(impurity={5: 0.8})
        self.assertEqual(tr.conductance(0.3), tr.transmission([0.3])[0])
        # a smooth T(E): the thermal average approaches T(mu) as T -> 0
        self.assertAlmostEqual(tr.conductance(0.3, 1e-4), tr.transmission([0.3])[0], places=8)
        es = np.linspace(-1, 1.6, 2001)
        f = 1 / (4 * 0.05 * np.cosh((es - 0.3) / 0.1) ** 2)
        self.assertAlmostEqual(tr.conductance(0.3, 0.05), np.sum(tr.transmission(es) * f) * (es[1] - es[0]),
                               places=5)
        g = tr.conductance_matrix(0.3, temperature=0.05)
        self.assertTrue(np.allclose(g, tr.conductance(0.3, 0.05, eta=1e-9) * np.array([[1, -1], [-1, 1]])))
        self.assertAlmostEqual(tr.four_terminal_resistance(0.3, (0, 1), (0, 1), temperature=0.05),
                               1 / g[0, 0], places=10)
        self.assertAlmostEqual(tr.conductance(0.3, 0.05, n_points=1), tr.transmission([0.3])[0])

    def test_checks(self):
        tr = chain_device()
        self.assertRaises(ValueError, tr.conductance, 0.3, -1.)
        self.assertRaises(ValueError, tr.thermoelectric, 0.3, 0.)
        self.assertRaises(ValueError, tr.conductance, 0.3, 0.1, n_points=0)
        self.assertRaises(TypeError, tr.conductance, 'a')
        self.assertRaises(ValueError, tr.conductance_matrix, 0.3, temperature=-1.)


class TestScatteringStates(unittest.TestCase):

    def test_perfect_chain(self):
        # a unit-current plane wave: |psi|^2 = 1 / |v| everywhere, and the LDOS
        # of the infinite chain, 1 / (pi sqrt(4 - E^2))
        tr = chain_device()
        e = 0.5
        psi = tr.wave_function(e, 0)
        self.assertEqual(psi.shape, (1, 10))
        self.assertTrue(np.allclose(np.abs(psi) ** 2, 1 / np.sqrt(4 - e**2)))
        self.assertTrue(np.allclose(tr.ldos(e), 1 / (np.pi * np.sqrt(4 - e**2))))

    def test_current_conservation(self):
        # the bond current of a scattering state is the transmission, on every bond
        tr = chain_device(impurity={5: 0.8})
        psi = tr.wave_function(0.3, 0)[0]
        current = -2 * np.imag(np.conj(psi[:-1]) * psi[1:])
        self.assertTrue(np.allclose(current, tr.transmission([0.3])[0]))

    def test_ldos_matches_green(self):
        # without bound states, the scattering states carry all of -Im G / pi
        tr = ribbon_device(4, 6)
        ldos = tr.ldos(0.7)
        self.assertTrue(np.allclose(ldos, -np.imag(np.diag(tr.get_green(0.7))) / np.pi, atol=1e-7))


if __name__ == '__main__':
    unittest.main()
