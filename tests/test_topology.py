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


class TestNonHermitianClass(unittest.TestCase):
    """The 38-fold way (Kawabata, Shiozaki, Ueda and Sato 2019)."""

    @staticmethod
    def chain(blocks):
        """A chain with hopping matrices blocks[R] (R = -1, 0, 1)."""
        norb = len(blocks[0])
        ks = KSpace(Lattice(unit_cell=[{'tag': 'a', 'r0': (0., 0.)}] * norb, prim_vec=[(1., 0.)]))
        hop = []
        for R, b in blocks.items():
            for i in range(norb):
                for j in range(norb):
                    if R == 0 and i == j:
                        ks.onsite[i] = b[i, i]
                    else:
                        hop.append({'i': i, 'j': j, 'R': (R,), 't': b[i, j]})
        ks.set_hopping(hop, hermitian=False)
        return ks

    def symmetric_chain(self, symmetries, norb=2, seed=0):
        """A random chain with the given symmetries {kind: U}, kind as in kspace._NH_KINDS."""
        rng = np.random.default_rng(seed)
        blocks = {R: rng.normal(size=(norb, norb)) + 1j * rng.normal(size=(norb, norb)) for R in (-1, 0, 1)}
        for (conj, transpose, minus), u in symmetries.items():
            image = {}
            for R in blocks:
                t = blocks[-R if transpose else R]
                t = t.conj() if conj else t
                image[R] = (-1) ** minus * u @ (t.T if transpose else t) @ u.conj().T
            blocks = {R: (blocks[R] + image[R]) / 2 for R in blocks}
        return self.chain(blocks)

    def test_symmetry_error_transpose(self):
        hatano_nelson = self.chain({1: np.eye(1), -1: 0.5 * np.eye(1), 0: np.zeros((1, 1))})
        lossy = self.chain({1: np.eye(1), -1: np.eye(1), 0: -0.5j * np.eye(1)})
        self.assertLess(hatano_nelson.symmetry_error(np.eye(1), antiunitary=True), 1e-12)
        self.assertGreater(hatano_nelson.symmetry_error(np.eye(1), transpose=True), 0.1)
        self.assertLess(lossy.symmetry_error(np.eye(1), transpose=True), 1e-12)
        self.assertGreater(lossy.symmetry_error(np.eye(1), antiunitary=True), 0.1)
        # H^dagger: the chiral symmetry of the Hermitian SSH chain
        self.assertLess(ssh(1., 0.5).symmetry_error(PAULI['z'], 'identity', True, True, transpose=True), 1e-12)
        self.assertRaises(TypeError, lossy.symmetry_error, np.eye(1), transpose=1)
        self.assertEqual(hatano_nelson.non_hermitian_class(time_reversal=np.eye(1)), 'AI')
        self.assertEqual(hatano_nelson.non_hermitian_class(), 'A')
        self.assertEqual(lossy.non_hermitian_class(time_reversal_dagger=np.eye(1)), 'AI†')
        self.assertEqual(lossy.non_hermitian_class(particle_hole_dagger=None, time_reversal_dagger=np.eye(1)), 'AI†')

    def test_hermitian_ssh(self):
        # a Hermitian model has every symmetry twice: TRS = TRS^dagger, CS = SLS, eta = 1
        chain, sz, i2 = ssh(1., 0.5), PAULI['z'], np.eye(2)
        self.assertEqual(chain.non_hermitian_class(chiral=sz), 'AIII')
        self.assertEqual(chain.non_hermitian_class(time_reversal=i2, chiral=sz), 'BDI')
        self.assertEqual(chain.non_hermitian_class(time_reversal_dagger=i2, particle_hole_dagger=sz), 'BDI†')
        self.assertEqual(chain.non_hermitian_class(particle_hole_dagger=sz), 'D†')
        self.assertEqual(chain.non_hermitian_class(particle_hole_dagger=sz, canonical=True), 'AI')
        self.assertEqual(chain.non_hermitian_class(sublattice=sz), 'A + S')
        self.assertEqual(chain.non_hermitian_class(chiral=sz, sublattice=sz), 'AIII + S+')
        self.assertEqual(chain.non_hermitian_class(time_reversal=i2, chiral=sz, pseudo_hermiticity=i2), 'BDI + S++')
        self.assertEqual(chain.non_hermitian_class(pseudo_hermiticity=i2), 'A + η')
        self.assertEqual(chain.non_hermitian_class(pseudo_hermiticity=i2, canonical=True), 'AIII')
        self.assertEqual(chain.non_hermitian_class(time_reversal=i2, pseudo_hermiticity=i2), 'AI + η+')
        self.assertEqual(chain.non_hermitian_class(particle_hole=sz, pseudo_hermiticity=i2), 'D + η+')
        self.assertEqual(chain.non_hermitian_class(particle_hole=sz, pseudo_hermiticity=i2, canonical=True), 'BDI')

    def test_sublattice_subscripts(self):
        # S U = eps U S^*, Gamma S = eps S Gamma (Kawabata et al., Tables VI, VII, XII and XIII)
        sx, sy, sz = PAULI['x'], PAULI['y'], PAULI['z']
        trs, phs_d, trs_d, cs, sls = (1, 0, 0), (1, 0, 1), (0, 1, 0), (1, 1, 1), (0, 0, 1)
        cases = [
            ({trs: np.eye(2), sls: sz}, dict(time_reversal=np.eye(2), sublattice=sz), 'AI + S+'),
            ({trs: np.eye(2), sls: sy}, dict(time_reversal=np.eye(2), sublattice=sy), 'AI + S-'),
            ({trs_d: np.eye(2), sls: sy}, dict(time_reversal_dagger=np.eye(2), sublattice=sy), 'C + S-'),
            ({cs: sx, sls: sz}, dict(chiral=sx, sublattice=sz), 'AIII + S-'),
            # BDI^dagger + S_{+-} is DIII + S_{-+} (Table XIII)
            ({trs_d: np.eye(2), phs_d: sx, sls: sz},
             dict(time_reversal_dagger=np.eye(2), particle_hole_dagger=sx, sublattice=sz), 'DIII + S-+'),
        ]
        for symmetries, kwargs, expected in cases:
            ks = self.symmetric_chain(symmetries)
            self.assertEqual(ks.non_hermitian_class(**kwargs), expected)
        ks = self.symmetric_chain({trs: np.eye(2), sls: sy})
        self.assertEqual(ks.non_hermitian_class(time_reversal=np.eye(2), sublattice=sy, canonical=True), 'AI + S-')
        # its partner under H -> iH: TRS becomes PHS^dagger
        ks = self.symmetric_chain({phs_d: np.eye(2), sls: sy})
        self.assertEqual(ks.non_hermitian_class(particle_hole_dagger=np.eye(2), sublattice=sy), 'AII + S-')
        self.assertEqual(ks.non_hermitian_class(particle_hole_dagger=np.eye(2), sublattice=sy, canonical=True),
                              'AI + S-')

    def test_errors(self):
        chain = ssh(1., 0.5)
        self.assertRaises(ValueError, chain.non_hermitian_class, None, None, None, np.eye(2), tol=-1.)
        self.assertRaises(TypeError, chain.non_hermitian_class, canonical=1)
        # not a symmetry
        self.assertRaises(ValueError, haldane().non_hermitian_class, time_reversal_dagger=np.eye(2))
        lat = Lattice(unit_cell=[{'tag': 'a', 'r0': (0., 0.)}, {'tag': 'b', 'r0': (0.5, 0.)}],
                             prim_vec=[(1., 0.)])
        zero = KSpace(lat)  # H = 0: every operator is a symmetry
        self.assertRaises(ValueError, zero.non_hermitian_class, np.array([[0., 1.], [1j, 0.]]))  # square
        # TRS and PHS imply the chiral symmetry U_C U_T^* = 1, not sigma_z: a unitary symmetry is left
        self.assertRaises(ValueError, zero.non_hermitian_class, np.eye(2), np.eye(2), PAULI['z'])


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
        # many k-points at once, one tensor per k-point
        ks = np.array([k, 0.1*rec[0] - 0.4*rec[1]])
        q = hal.quantum_geometric_tensor([0], ks, positions=False)
        self.assertEqual(q.shape, (2, 2, 2))
        self.assertTrue(np.allclose(q[0], q2))
        self.assertTrue(np.allclose(q[1], hal.quantum_geometric_tensor(0, ks[1], positions=False)))
        self.assertRaises(ValueError, hal.quantum_geometric_tensor, 0, [0.])
        self.assertRaises(ValueError, hal.quantum_geometric_tensor, 0, np.zeros((3, 1)))
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
