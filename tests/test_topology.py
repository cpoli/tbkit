"""
Berry and Zak phases, Wannier centres and their flow, Z2 invariants (Wilson
loop and Fu-Kane parity), symmetry checks and the tenfold way, the quantum
geometric tensor, and the local Chern marker.
"""
import unittest

import numpy as np

import tbkit.lattices as lattices
from tbkit.kspace import KSpace, PAULI
from tbkit.lattice import Lattice
from tbkit.system import System
from tests.test_three_d import weyl


def ssh(v, w):
    lat = Lattice(unit_cell=[{'tag': 'a', 'r0': (0., 0.)}, {'tag': 'b', 'r0': (0.5, 0.)}],
                         prim_vec=[(1., 0.)])
    ks = KSpace(lat)
    ks.set_hopping([{'i': 0, 'j': 1, 'R': (0,), 't': v}, {'i': 1, 'j': 0, 'R': (1,), 't': w}])
    return ks


def haldane(t2=0.2, M=0.):
    ks = KSpace(lattices.honeycomb())
    ks.set_hopping([{'i': 0, 'j': 1, 'R': R, 't': 1.} for R in [(0, 0), (-1, 0), (0, -1)]])
    for R in [(0, 1), (-1, 0), (1, -1)]:
        ks.set_hopping([{'i': 0, 'j': 0, 'R': R, 't': 1j*t2}, {'i': 1, 'j': 1, 'R': R, 't': -1j*t2}])
    ks.set_onsite({'a': M, 'b': -M})
    return ks


def kane_mele(lam=0.06, M=0., rashba=0.):
    lat = lattices.honeycomb()
    ks = KSpace(lat, spin=True)
    ks.set_hopping([{'i': 0, 'j': 1, 'R': R, 't': 1.} for R in [(0, 0), (-1, 0), (0, -1)]])
    for R in [(0, 1), (-1, 0), (1, -1)]:
        ks.set_hopping([{'i': 0, 'j': 0, 'R': R, 't': 1j*lam*PAULI['z']},
                                 {'i': 1, 'j': 1, 'R': R, 't': -1j*lam*PAULI['z']}])
    if rashba:
        tau = np.array([d['r0'] for d in lat.unit_cell])
        a = np.array(lat.prim_vec)
        for R in [(0, 0), (-1, 0), (0, -1)]:
            d = tau[1] + R[0]*a[0] + R[1]*a[1] - tau[0]
            d = d / np.linalg.norm(d)
            ks.set_hopping([{'i': 0, 'j': 1, 'R': R,
                                      't': 1j*rashba*(PAULI['x']*d[1] - PAULI['y']*d[0])}])
    ks.set_onsite({'a': M, 'b': -M})
    return ks


# the inversion of graphene about a bond centre swaps the sublattices, and
# is the identity on spin
INVERSION_KM = np.kron(PAULI['x'], np.eye(2))
TIME_REVERSAL_KM = np.kron(np.eye(2), 1j*PAULI['y'])


class TestBerryPhase(unittest.TestCase):

    def test_ssh_zak_phase_and_wannier_centre(self):
        # the Wannier centre sits on the strong bond: the intracell bond
        # midpoint (1/4) when v > w, the intercell one (3/4) when v < w
        trivial, topo = ssh(1., 0.4), ssh(0.4, 1.)
        self.assertTrue(np.allclose(trivial.wannier_centers(0), [0.25]))
        self.assertTrue(np.allclose(topo.wannier_centers(0), [0.75]))
        self.assertAlmostEqual(abs(trivial.berry_phase(0)), np.pi/2)
        # the Zak phases differ by pi, in either gauge
        diff = (topo.berry_phase(0) - trivial.berry_phase(0)) % (2*np.pi)
        self.assertAlmostEqual(diff, np.pi)
        diff_periodic = (topo.berry_phase(0, positions=False)
                                - trivial.berry_phase(0, positions=False)) % (2*np.pi)
        self.assertAlmostEqual(diff_periodic, np.pi)
        # both bands together: the centres sum to the orbital positions, 0 + 1/2
        self.assertAlmostEqual(trivial.wannier_centers([0, 1]).sum() % 1., 0.5)

    def test_checks(self):
        ks = ssh(1., 0.5)
        self.assertRaises(ValueError, ks.berry_phase, 0, 10, 1)
        self.assertRaises(ValueError, ks.berry_phase, 0, 10, 0, (0.1,))
        self.assertRaises(TypeError, ks.berry_phase, 0, 10, 0, None, 1)
        self.assertRaises(ValueError, ks.berry_phase, 5)
        self.assertRaises(ValueError, ks.wannier_flow, 0)  # needs 2D


class TestWannierFlow(unittest.TestCase):

    def test_chern_winding(self):
        # the Haldane band's hybrid Wannier centre winds by C = 1 cell
        fracs, centers = haldane().wannier_flow(0, nk=40, nk_perp=41)
        self.assertEqual(centers.shape, (41, 1))
        steps = np.diff(centers[:, 0])
        winding = np.sum((steps + 0.5) % 1. - 0.5)
        self.assertAlmostEqual(abs(winding), 1., places=6)
        self.assertAlmostEqual(fracs[-1], 1.)
        # a loop at a given k_perp (a plain number in 2D)
        self.assertAlmostEqual(haldane().wannier_centers(0, nk=40, k_perp=0.3)[0],
                                        centers[np.argmin(np.abs(fracs - 0.3)), 0], places=1)
        # and loops along a2, moved along a1, see the same winding
        _, centers2 = haldane().wannier_flow(0, nk=40, nk_perp=41, direction=1)
        steps2 = np.diff(centers2[:, 0])
        self.assertAlmostEqual(abs(np.sum((steps2 + 0.5) % 1. - 0.5)), 1., places=6)

    def test_three_d(self):
        wey = weyl()
        _, centers = wey.wannier_flow(0, nk=30, nk_perp=5, direction=0, flow=1, k_fixed=0.)
        self.assertEqual(centers.shape, (5, 1))
        self.assertTrue(np.isfinite(wey.berry_phase(0, nk=20, k_perp=(0., 0.25))))
        self.assertRaises(ValueError, wey.wannier_flow, 0, 10, 5, 0, 0)
        self.assertRaises(TypeError, wey.wannier_flow, 0, 10, 5, 0, 1.)


class TestZ2(unittest.TestCase):

    def test_kane_mele(self):
        # topological below the critical mass 3 sqrt(3) lambda (~0.31 here),
        # also with Rashba coupling, which breaks S_z conservation
        self.assertEqual(kane_mele().z2_invariant([0, 1], nk=60, nk_perp=41), 1)
        self.assertEqual(kane_mele(M=0.1).z2_invariant([0, 1], nk=60, nk_perp=41), 1)
        self.assertEqual(kane_mele(M=0.5).z2_invariant([0, 1], nk=60, nk_perp=41), 0)
        self.assertEqual(kane_mele(rashba=0.05).z2_invariant([0, 1], nk=60, nk_perp=41), 1)
        self.assertRaises(ValueError, kane_mele().z2_invariant, [0])
        self.assertRaises(TypeError, kane_mele().z2_invariant, 0)

    def test_fu_kane_parity(self):
        self.assertEqual(kane_mele().parity_z2(INVERSION_KM, [0, 1]), 1)
        # the same answer with the inversion given as a function of k
        self.assertEqual(kane_mele().parity_z2(lambda k: INVERSION_KM, [0, 1]), 1)
        # a spinless-like trivial model: two copies of plain graphene with a
        # gap from an inversion-symmetric (third-neighbour-free) term is out of
        # reach, so check the trivial answer on a trivially gapped chain pair
        lat = Lattice(unit_cell=[{'tag': 'a', 'r0': (0., 0.)}], prim_vec=[(1., 0.), (0., 1.)])
        triv = KSpace(lat, spin=True)
        triv.set_hopping([{'i': 0, 'j': 0, 'R': (1, 0), 't': 0.1}])
        triv.set_onsite({'a': (-1., -1.)})
        self.assertEqual(triv.parity_z2(np.eye(2), [0, 1]), 0)

    def test_parity_checks(self):
        km = kane_mele(M=0.1)  # the mass breaks inversion
        self.assertRaises(ValueError, km.parity_z2, INVERSION_KM, [0, 1])
        self.assertRaises(ValueError, kane_mele().parity_z2, np.eye(3), [0, 1])
        # a spinless band pair of opposite parities is not a Kramers pair
        lat = Lattice(unit_cell=[{'tag': 'a', 'r0': (0., 0.)}, {'tag': 'b', 'r0': (0.5, 0.)}],
                             prim_vec=[(1., 0.), (0., 1.)])
        ks = KSpace(lat)
        ks.set_onsite({'a': -1., 'b': 1.})
        self.assertRaises(ValueError, ks.parity_z2, np.diag([1., -1.]), [0, 1])

    def test_helpers(self):
        self.assertEqual(KSpace._between(0.1, 0.3, np.array([0.2, 0.5])), 1)
        self.assertEqual(KSpace._between(0.9, 0.1, np.array([0.95, 0.5])), 1)  # wraps
        self.assertAlmostEqual(KSpace._largest_gap(np.array([0.1, 0.2])), 0.65)


class TestSymmetry(unittest.TestCase):

    def test_symmetry_error(self):
        km = kane_mele()
        self.assertLess(km.symmetry_error(TIME_REVERSAL_KM, antiunitary=True), 1e-12)
        self.assertLess(km.symmetry_error(INVERSION_KM), 1e-12)
        self.assertGreater(haldane().symmetry_error(np.eye(2), antiunitary=True), 0.1)
        self.assertLess(ssh(1., 0.5).symmetry_error(PAULI['z'], 'identity', anti=True), 1e-12)
        self.assertRaises(ValueError, km.symmetry_error, INVERSION_KM, 'plus')

    def test_tenfold(self):
        self.assertEqual(haldane().tenfold_class(), 'A')
        chain = ssh(1., 0.5)
        self.assertEqual(chain.tenfold_class(chiral=PAULI['z']), 'AIII')
        self.assertEqual(chain.tenfold_class(time_reversal=np.eye(2)), 'AI')
        self.assertEqual(chain.tenfold_class(time_reversal=np.eye(2), chiral=PAULI['z']), 'BDI')
        self.assertEqual(chain.tenfold_class(particle_hole=PAULI['z'], chiral=PAULI['z']), 'BDI')
        self.assertEqual(chain.tenfold_class(time_reversal=np.eye(2), particle_hole=PAULI['z']), 'BDI')
        self.assertEqual(kane_mele().tenfold_class(time_reversal=TIME_REVERSAL_KM), 'AII')
        # not a symmetry
        self.assertRaises(ValueError, haldane().tenfold_class, np.eye(2))
        # a symmetry (of a model with H = 0) whose square is not +-1
        lat = Lattice(unit_cell=[{'tag': 'a', 'r0': (0., 0.)}, {'tag': 'b', 'r0': (0.5, 0.)}],
                             prim_vec=[(1., 0.)])
        self.assertRaises(ValueError, KSpace(lat).tenfold_class, np.array([[0., 1.], [1j, 0.]]))
        self.assertRaises(ValueError, chain.tenfold_class, None, None, None, 5, -1.)  # tol


class TestQuantumGeometry(unittest.TestCase):

    def test_curvature_and_metric(self):
        hal = haldane()
        fr = (np.arange(24) + 0.5) / 24
        rec = hal.rec_vec_k
        dA = abs(np.linalg.det(rec)) / 24**2
        total = 0.
        for f1 in fr:
            for f2 in fr:
                q = hal.quantum_geometric_tensor(0, f1*rec[0] + f2*rec[1])
                omega = -2 * q[0, 1].imag
                g = q.real
                total += omega * dA
                self.assertTrue(np.allclose(q, q.conj().T))
                self.assertGreaterEqual(np.sqrt(np.linalg.det(g)), abs(omega)/2 - 1e-8)
        self.assertAlmostEqual(total / (2*np.pi), 1., places=3)
        # the local values depend on the orbital positions (only integrals,
        # like the Chern number, do not)
        k = 0.3*rec[0] + 0.1*rec[1]
        q1 = hal.quantum_geometric_tensor([0], k)
        q2 = hal.quantum_geometric_tensor([0], k, positions=False)
        self.assertFalse(np.allclose(q1.real, q2.real))
        self.assertRaises(ValueError, hal.quantum_geometric_tensor, 0, [0.])
        self.assertRaises(ValueError, hal.quantum_geometric_tensor, 0, [0., 0.], -1.)


class TestLocalChernMarker(unittest.TestCase):

    def _flake(self, t2=0.2, n=12):
        lat = lattices.honeycomb()
        lat.get_lattice(n, n)
        sys = System(lat)
        sys.set_hopping([{'n': 1, 't': 1.}])
        # the Haldane model of haldane(), in real space: +i t2 along a2, -a1,
        # a1 - a2 on 'a' (angles 60, 180, -60), the opposite on 'b'
        for tag, sign in (('aa', 1), ('bb', -1)):
            sys.set_hopping([{'n': 2, 'ang': 60., 'tag': tag, 't': 1j*t2*sign},
                                      {'n': 2, 'ang': 0., 'tag': tag, 't': -1j*t2*sign},
                                      {'n': 2, 'ang': 120., 'tag': tag, 't': -1j*t2*sign}])
        sys.get_ham()
        return sys

    def test_bulk_marker_is_chern_number(self):
        sys = self._flake()
        marker = sys.get_local_chern_marker(0.)
        x, y = sys.lat.coor['x'], sys.lat.coor['y']
        bulk = np.hypot(x - x.mean(), y - y.mean()) < 3.
        self.assertAlmostEqual(marker[bulk].mean(), haldane().chern_number(0, 20), places=2)
        self.assertAlmostEqual(marker.sum(), 0., places=8)
        # an explicit area per site scales it
        honey = abs(np.linalg.det(np.array(sys.lat.prim_vec))) / 2
        self.assertTrue(np.allclose(sys.get_local_chern_marker(0., area=2*honey), marker/2))

    def test_trivial_and_checks(self):
        sys = self._flake(t2=0.)
        self.assertLess(np.max(np.abs(sys.get_local_chern_marker(0.))), 1e-8)
        chain = Lattice(unit_cell=[{'tag': 'a', 'r0': (0., 0.)}], prim_vec=[(1., 0.)])
        chain.get_lattice(4)
        csys = System(chain)
        csys.set_hopping([{'n': 1, 't': 1.}])
        csys.get_ham()
        self.assertRaises(ValueError, csys.get_local_chern_marker, 0.)
        csys.get_local_chern_marker(0., area=1.)
        csys.set_onsite({'a': 0.})
        csys.set_onsite_def({0: 1j})
        csys.get_ham()
        self.assertRaises(ValueError, csys.get_local_chern_marker, 0., 1.)


if __name__ == '__main__':
    unittest.main()
