"""
Multi-terminal Landauer-Buttiker transport, bond currents, shot noise, and
the recursive Green's function (tbkit.transport).
"""
import unittest

import numpy as np

import tbkit.error_handling as error_handling
import tbkit.lattices as lattices
from tbkit.kspace import ribbon
from tbkit.system import System
from tbkit.transport import (Transport, RecursiveTransport, lead_from_kspace,
                                           slices_from_positions)
from tests.test_transport import chain_device, SQUARE, ONE


def strip(length, width, disorder=0., seed=0, t=1.):
    '''A square-lattice strip with random onsite energies in [-disorder/2, disorder/2].'''
    lat = lattices.square()
    lat.get_lattice(length, width)
    sys = System(lat)
    sys.set_hopping([{'n': 1, 't': t}])
    sys.set_onsite({'a': 0.})
    sys.onsite[:] = disorder * (np.random.default_rng(seed).random(lat.sites) - 0.5)
    sys.get_ham()
    rib = ribbon(lattices.square(), [dict(h, t=t) for h in SQUARE], width=width, direction=1)
    return lat, sys, rib


def strip_leads(lat, length, rib):
    '''Edge columns (sorted by y, the order of the ribbon's orbitals) and the two leads.'''
    cols = [sorted(np.flatnonzero(np.isclose(lat.coor['x'], x)), key=lambda i: lat.coor['y'][i])
                for x in (0., length - 1.)]
    (h_l, v_l), (h_r, v_r) = lead_from_kspace(rib, -1), lead_from_kspace(rib, 1)
    return [list(map(int, c)) for c in cols], (h_l, v_l), (h_r, v_r)


def hall_bar(length=40, width=14, flux=0.05, t=-1., probe=6):
    r'''
    A square-lattice Hall bar in the Landau gauge A = (-B y, 0): current
    leads 0 (left) and 5 (right) carry the field, the voltage probes 1, 2
    (top, left to right) and 3, 4 (bottom) are field-free strips.
    '''
    lat = lattices.square()
    lat.get_lattice(length, width)
    x, y = lat.coor['x'], lat.coor['y']
    sys = System(lat)
    sys.set_hopping([{'n': 1, 't': t}])
    sys.set_onsite({'a': 0.})
    sys.set_peierls_phase(lambda xi, yi, xj, yj: -2 * np.pi * flux * (xj - xi) * (yi + yj) / 2)
    sys.get_ham()
    ham = sys.ham.toarray()
    cols = [sorted(np.flatnonzero(np.isclose(x, c)), key=lambda i: y[i])
                for c in (0, 1, length - 2, length - 1)]
    tr = Transport(ham)
    # the lead continues the bar: the same columns, the same gauge
    v_left = ham[np.ix_(cols[1], cols[0])]
    tr.add_lead(ham[np.ix_(cols[0], cols[0])], v_left, v_left, [int(i) for i in cols[0]])
    h_probe = t * (np.eye(probe, k=1) + np.eye(probe, k=-1))
    for yy in (width - 1, 0):
        for x0 in (length // 4 - probe // 2, 3 * length // 4 - probe // 2):
            sites = [int(np.flatnonzero(np.isclose(x, x0 + k) & np.isclose(y, yy))[0])
                        for k in range(probe)]
            tr.add_lead(h_probe, t * np.eye(probe), t * np.eye(probe), sites)
    v_right = ham[np.ix_(cols[2], cols[3])]
    tr.add_lead(ham[np.ix_(cols[3], cols[3])], v_right, v_right, [int(i) for i in cols[3]])
    return tr


class TestConductanceMatrix(unittest.TestCase):

    def test_two_terminal(self):
        tr = chain_device(10, impurity={4: 0.7})
        e = 0.3
        t = tr.transmission([e], eta=1e-9)[0]  # the Caroli path, as conductance_matrix
        g = tr.conductance_matrix(e)
        self.assertTrue(np.allclose(g, [[t, -t], [-t, t]]))
        self.assertTrue(np.allclose(tr.transmission_matrix(e), [[0, t], [t, 0]]))
        self.assertAlmostEqual(tr.four_terminal_resistance(e, (0, 1), (0, 1)), 1 / t, places=8)

    def test_hall_bar_plateaus(self):
        # Buttiker (1988): nu edge channels give R_xy = h/(nu e^2) and R_xx = 0
        # (Landau levels -4|t| + 4 pi flux |t| (n + 1/2), up to lattice corrections)
        tr = hall_bar()
        for e, nu in ((-3.4, 1), (-2.8, 2)):
            g = tr.conductance_matrix(e)
            # (to the precision of the lead Green's functions at eta = 1e-9)
            self.assertTrue(np.allclose(g.sum(axis=0), 0., atol=1e-6))
            self.assertTrue(np.allclose(g.sum(axis=1), 0., atol=1e-6))
            r_xy = tr.four_terminal_resistance(e, (0, 5), (1, 3))
            r_xx = tr.four_terminal_resistance(e, (0, 5), (1, 2))
            r_2t = tr.four_terminal_resistance(e, (0, 5), (0, 5))
            self.assertAlmostEqual(r_xy, 1 / nu, places=5)
            self.assertAlmostEqual(r_xx, 0., places=5)
            self.assertAlmostEqual(r_2t, 1 / nu, places=5)
            # the other pair of probes, and the opposite field
            self.assertAlmostEqual(tr.four_terminal_resistance(e, (0, 5), (2, 4)), 1 / nu, places=5)
        self.assertAlmostEqual(hall_bar(flux=-0.05).four_terminal_resistance(-3.4, (0, 5), (1, 3)),
                                        -1., places=5)
        # no field: no Hall voltage (mirror y -> -y), a finite R_xx only from the probes
        self.assertAlmostEqual(hall_bar(flux=0.).four_terminal_resistance(-3.4, (0, 5), (1, 3)),
                                        0., places=8)

    def test_errors(self):
        tr = chain_device(10)
        self.assertRaises(TypeError, tr.four_terminal_resistance, 0.3, 0, (0, 1))
        self.assertRaises(ValueError, tr.four_terminal_resistance, 0.3, (0, 0), (0, 1))
        self.assertRaises(ValueError, tr.four_terminal_resistance, 0.3, (0, 1), (0, 2))
        self.assertRaises(TypeError, tr.conductance_matrix, 'a')


class TestCurrents(unittest.TestCase):

    def test_chain(self):
        tr = chain_device(10, impurity={4: 0.7})
        e = 0.3
        t = tr.transmission([e])[0]
        bonds = tr.bond_currents(e, 0)
        self.assertTrue(np.allclose(bonds, -bonds.T))
        self.assertTrue(np.allclose([bonds[i, i + 1] for i in range(9)], t))
        # injected from the right, the current flows to the left
        self.assertTrue(np.allclose([tr.bond_currents(e, 1)[i + 1, i] for i in range(9)], t))
        positions = np.arange(10.)[:, None]
        self.assertTrue(np.allclose(tr.local_currents(e, 0, positions)[1:-1, 0], t))
        self.assertRaises(ValueError, tr.local_currents, e, 0, np.arange(9.)[:, None])
        self.assertRaises(ValueError, tr.bond_currents, e, 2)

    def test_disordered_strip(self):
        length, width = 12, 5
        lat, sys, rib = strip(length, width, disorder=1.5, seed=3)
        cols, (h_l, v_l), (h_r, v_r) = strip_leads(lat, length, rib)
        tr = Transport(sys.ham)
        tr.add_lead(h_l, v_l, np.eye(width), cols[0])
        tr.add_lead(h_r, v_r, np.eye(width), cols[1])
        e = -1.1
        t = tr.transmission([e])[0]
        bonds = tr.bond_currents(e, 0)
        x = lat.coor['x']
        # the current through every cross-section is the transmission
        for x0 in range(length - 1):
            cut = np.ix_(np.flatnonzero(np.isclose(x, x0)), np.flatnonzero(np.isclose(x, x0 + 1)))
            self.assertAlmostEqual(bonds[cut].sum(), t, places=6)
        # and it is conserved on every site away from the leads
        inner = (x > 0.5) & (x < length - 1.5)
        self.assertTrue(np.allclose(bonds.sum(axis=1)[inner], 0., atol=1e-7))  # up to 2 eta |G|^2


class TestShotNoise(unittest.TestCase):

    def test_ballistic_and_single_channel(self):
        # a perfect chain: T = 1, no shot noise (Fano factor 0)
        perfect = chain_device(10)
        energies = [-1.5, -0.3, 0.8]
        self.assertTrue(np.allclose(perfect.fano_factor(energies), 0., atol=1e-6))
        self.assertTrue(np.allclose(perfect.shot_noise(energies), 0., atol=1e-6))
        # one channel: S ~ T(1 - T), F = 1 - T
        tr = chain_device(10, impurity={4: 0.7})
        t = tr.transmission(energies)
        self.assertTrue(np.allclose(tr.fano_factor(energies), 1 - t))
        self.assertTrue(np.allclose(tr.shot_noise(energies), t * (1 - t)))
        # outside the band nothing is transmitted: no Fano factor
        self.assertTrue(np.isnan(tr.fano_factor(2.5)[0]))
        # a clean strip is ballistic too, with several open channels
        length, width = 8, 4
        lat, sys, rib = strip(length, width)
        cols, (h_l, v_l), (h_r, v_r) = strip_leads(lat, length, rib)
        clean = Transport(sys.ham)
        clean.add_lead(h_l, v_l, np.eye(width), cols[0])
        clean.add_lead(h_r, v_r, np.eye(width), cols[1])
        eig = clean.transmission_eigenvalues(-1.)
        self.assertEqual(len(eig), width)
        self.assertAlmostEqual(eig.sum(), clean.transmission([-1.], eta=1e-9)[0], places=8)
        self.assertTrue(np.allclose(clean.fano_factor([-1., 0.5]), 0., atol=1e-6))

    def test_tunnel_barrier(self):
        # a weak link: T << 1, Poissonian noise, F -> 1
        lat = lattices.chain()
        lat.get_lattice(10)
        sys = System(lat)
        sys.set_hopping([{'n': 1, 't': 1.}])
        sys.set_onsite({'a': 0.})
        sys.get_ham()
        ham = sys.ham.toarray()
        ham[4, 5] = ham[5, 4] = 0.05
        tr = Transport(ham)
        tr.add_lead([[0.]], [[1.]], ONE, [0])
        tr.add_lead([[0.]], [[1.]], ONE, [9])
        self.assertGreater(tr.fano_factor(0.1)[0], 0.99)
        self.assertRaises(ValueError, tr.fano_factor, [])


class TestRecursiveGreenFunction(unittest.TestCase):

    def test_matches_dense_transport(self):
        length, width = 15, 4
        lat, sys, rib = strip(length, width, disorder=2., seed=1)
        cols, (h_l, v_l), (h_r, v_r) = strip_leads(lat, length, rib)
        dense = Transport(sys.ham)
        dense.add_lead(h_l, v_l, np.eye(width), cols[0])
        dense.add_lead(h_r, v_r, np.eye(width), cols[1])
        slices = slices_from_positions(lat.coor['x'])
        self.assertEqual(len(slices), length)
        # slices sorted by x, sites within each by index (the lead columns
        # are sorted by y, which is the same order here)
        self.assertEqual(slices[0], cols[0])
        rgf = RecursiveTransport(sys.ham, slices, (h_l, v_l, np.eye(width)), (h_r, v_r, np.eye(width)))
        energies = np.linspace(-3.5, 3.5, 15)
        self.assertTrue(np.allclose(rgf.transmission(energies), dense.transmission(energies), atol=1e-9))
        self.assertTrue(np.allclose(rgf.transmission_eigenvalues(-0.7),
                                              dense.transmission_eigenvalues(-0.7), atol=1e-9))
        self.assertTrue(np.allclose(rgf.fano_factor([-0.7, 1.3]), dense.fano_factor([-0.7, 1.3]), atol=1e-9))
        # a dense Hamiltonian works too
        again = RecursiveTransport(sys.ham.toarray(), slices, (h_l, v_l, np.eye(width)),
                                              (h_r, v_r, np.eye(width)))
        self.assertAlmostEqual(again.transmission(-0.7)[0], dense.transmission([-0.7], eta=1e-9)[0], places=9)

    def test_single_slice_and_long_chain(self):
        # one slice holds both leads
        rgf = RecursiveTransport(np.zeros((1, 1)), [[0]], ([[0.]], [[1.]], ONE), ([[0.]], [[1.]], ONE))
        self.assertTrue(np.allclose(rgf.transmission([-1., 0.5]), 1.))
        # a long perfect chain transmits fully, without a dense inversion
        n = 3000
        lat = lattices.chain()
        lat.get_lattice(n)
        sys = System(lat)
        sys.set_hopping([{'n': 1, 't': 1.}])
        sys.set_onsite({'a': 0.})
        sys.get_ham()
        rgf = RecursiveTransport(sys.ham, [[i] for i in range(n)], ([[0.]], [[1.]], ONE),
                                             ([[0.]], [[1.]], ONE))
        # (eta absorbs a fraction ~ n eta / v of the current: keep it tiny)
        self.assertAlmostEqual(rgf.transmission(0.4, eta=1e-12)[0], 1., places=7)
        self.assertAlmostEqual(rgf.fano_factor(0.4, eta=1e-12)[0], 0., places=7)

    def test_errors(self):
        lead = ([[0.]], [[1.]], ONE)
        chain = chain_device(4).ham
        slices = [[0], [1], [2], [3]]
        self.assertRaises(TypeError, RecursiveTransport, chain, [0, 1, 2, 3], lead, lead)
        self.assertRaises(ValueError, RecursiveTransport, chain, [[0], [1], [2]], lead, lead)
        self.assertRaises(ValueError, RecursiveTransport, chain, [[0], [2], [1], [3]], lead, lead)
        self.assertRaises(TypeError, RecursiveTransport, chain, slices, lead[:2], lead)
        self.assertRaises(ValueError, RecursiveTransport, chain, slices, ([[0.]], [[1.]], np.ones((2, 1))), lead)
        self.assertRaises(ValueError, RecursiveTransport, np.zeros((2, 3)), slices, lead, lead)
        rgf = RecursiveTransport(chain, slices, lead, lead)
        self.assertRaises(TypeError, rgf.transmission, ['a'])
        self.assertRaises(ValueError, rgf.transmission, 0., 0.)
        self.assertRaises(TypeError, rgf.transmission_eigenvalues, '0')
        self.assertRaises(ValueError, rgf.fano_factor, [np.nan])
        self.assertRaises(ValueError, slices_from_positions, [])
        self.assertEqual(slices_from_positions([0., 1., 0., 1. + 1e-9]), [[0, 2], [1, 3]])

    def test_validators(self):
        self.assertRaises(ValueError, error_handling.block_tridiagonal, 2)
        self.assertRaises(ValueError, error_handling.site_positions, np.zeros(3), 3)
        self.assertRaises(TypeError, error_handling.lead_pair, (0, 1, 2), 3, 'current', True)
