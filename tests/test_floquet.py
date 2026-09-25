"""
Floquet theory: evolution operator, quasienergies, Sambe space, and driven
band structures (graphene under circularly polarized light).
"""
import unittest

import numpy as np
import scipy.linalg as LA
import scipy.sparse as sparse

import tbkit.lattices as lattices
from tbkit.floquet import (FloquetKSpace, effective_hamiltonian, evolution_operator,
                                        harmonics, quasienergies, sambe_hamiltonian,
                                        DrivenKSpace, StepDrive, step_drive)
from tbkit.kspace import KSpace, reciprocal_vectors, ribbon, PAULI
from tbkit.lattice import Lattice
from tbkit.system import System


def two_level(omega=3., amp=0.4):
    return lambda t: np.array([[1., amp*np.cos(omega*t)], [amp*np.cos(omega*t), -1.]])


def graphene():
    g = KSpace(lattices.honeycomb())
    g.set_hopping([{'i': 0, 'j': 1, 'R': R, 't': 1.} for R in [(0, 0), (-1, 0), (0, -1)]])
    return g


def circular(a0, omega):
    return lambda t: (a0 * np.cos(omega * t), a0 * np.sin(omega * t))


class TestFloquet(unittest.TestCase):

    def test_static(self):
        # a static H: quasienergies = energies folded into [-w/2, w/2)
        h = np.diag([0.5, 2.5])
        period = 2 * np.pi / 3.
        eps = quasienergies(lambda t: h, period, 10)
        self.assertTrue(np.allclose(eps, [-0.5, 0.5]))  # 2.5 - 3
        u = evolution_operator(lambda t: h, period, 7)
        self.assertTrue(np.allclose(u, np.diag(np.exp(-1j * np.array([0.5, 2.5]) * period))))

    def test_sambe_matches_time_evolution(self):
        omega = 3.
        ham_t = two_level(omega)
        period = 2 * np.pi / omega
        eps = quasienergies(ham_t, period, 800)
        harm = harmonics(ham_t, period, 2)
        self.assertTrue(np.allclose(harm[1], [[0., 0.2], [0.2, 0.]]))
        self.assertTrue(np.allclose(harm[2], 0.))
        sambe = np.linalg.eigvalsh(sambe_hamiltonian(harm, omega, 12))
        central = np.sort(sambe[np.abs(sambe) < omega / 2])
        self.assertTrue(np.allclose(central, eps, atol=1e-5))
        h_f = effective_hamiltonian(ham_t, period, 800)
        self.assertTrue(np.allclose(h_f, h_f.conj().T))
        self.assertTrue(np.allclose(np.linalg.eigvalsh(h_f), eps))

    def test_checks(self):
        self.assertRaises(TypeError, quasienergies, 1., 1.)
        self.assertRaises(ValueError, quasienergies, two_level(), -1.)
        self.assertRaises(TypeError, harmonics, two_level(), 1., 1.5)
        self.assertRaises(ValueError, harmonics, two_level(), 1., -1)
        self.assertRaises(TypeError, sambe_hamiltonian, [np.eye(2)], 1., 1)
        self.assertRaises(ValueError, sambe_hamiltonian, {1: np.eye(2)}, 1., 1)
        self.assertRaises(ValueError, sambe_hamiltonian, {0: np.eye(2), 1: np.eye(3)}, 1., 1)


class TestFloquetKSpace(unittest.TestCase):

    def test_undriven(self):
        g = graphene()
        fl = FloquetKSpace(g, lambda t: (0., 0.), 0.5, 10)
        k = np.array([0.4, -0.9])
        self.assertTrue(np.allclose(fl.get_ham(k), g.get_ham(k)))
        # a static vector potential is a shift of k (with the orbital positions)
        a = np.array([0.3, 0.1])
        tau = g.orbital_positions()
        d = np.diag(np.exp(1j * tau @ a))
        self.assertTrue(np.allclose(g.get_ham_peierls(k, a), d.conj().T @ g.get_ham(k + a) @ d))

    def test_circularly_polarized_graphene(self):
        # light opens equal gaps at K and K' of opposite mass: a Chern insulator
        omega, a0 = 12., 0.6
        fl = FloquetKSpace(graphene(), circular(a0, omega), 2*np.pi/omega, 60)
        b1, b2 = (np.array(v) for v in reciprocal_vectors(fl.lat.prim_vec))
        K = (b1 - b2) / 3
        gap_k = np.diff(np.linalg.eigvalsh(fl.get_ham(K)))[0]
        gap_kp = np.diff(np.linalg.eigvalsh(fl.get_ham(-K)))[0]
        self.assertGreater(gap_k, 0.05)
        self.assertAlmostEqual(gap_k, gap_kp, places=8)
        self.assertAlmostEqual(abs(fl.chern_number(0, 16)), 1., places=6)
        # reversing the helicity reverses the Chern number
        rev = FloquetKSpace(graphene(), lambda t: (a0*np.cos(omega*t), -a0*np.sin(omega*t)),
                                    2*np.pi/omega, 60)
        self.assertAlmostEqual(rev.chern_number(0, 16), -fl.chern_number(0, 16), places=6)
        # linear polarization leaves the Dirac points gapless (time reversal)
        lin = FloquetKSpace(graphene(), lambda t: (a0*np.cos(omega*t), 0.), 2*np.pi/omega, 60)
        self.assertLess(np.min(np.abs(np.diff(lin.mesh_bands(30), axis=1))), 0.2)
        self.assertTrue(fl.is_hermitian())

    def test_high_frequency_limit(self):
        # van Vleck: H_eff = H_0 + sum_m [H_-m, H_m] / (m w) + O(1/w^2)
        g = graphene()
        omega, a0 = 40., 0.8
        drive = circular(a0, omega)
        period = 2 * np.pi / omega
        fl = FloquetKSpace(g, drive, period, 80)
        k = np.array([0.7, 0.3])
        harm = harmonics(lambda t: g.get_ham_peierls(k, drive(t)), period, 3)
        h_eff = harm[0] + sum((harm[-m] @ harm[m] - harm[m] @ harm[-m]) / (m * omega) for m in (1, 2, 3))
        self.assertTrue(np.allclose(np.linalg.eigvalsh(fl.get_ham(k)), np.linalg.eigvalsh(h_eff), atol=2e-3))

    def test_checks(self):
        fl = FloquetKSpace(graphene(), lambda t: (0., 0.), 1., 5)
        self.assertRaises(ValueError, fl.finite_ham, 2)
        self.assertRaises(TypeError, FloquetKSpace, 1., lambda t: (0., 0.), 1.)
        self.assertRaises(TypeError, FloquetKSpace, graphene(), 1., 1.)
        self.assertRaises(ValueError, graphene().get_ham_peierls, [0., 0.], [0.])


# ----------------------------------------------------------------------
# Anomalous Floquet phases: Rudner, Lindner, Berg and Levin, PRX 3, 031005 (2013)
# ----------------------------------------------------------------------

# bipartite square lattice, A at (0, 0), B at (1, 0), a1 = (2, 0), a2 = (1, 1)
RUDNER_LAT = Lattice(unit_cell=[{'tag': 'a', 'r0': (0., 0.)}, {'tag': 'b', 'r0': (1., 0.)}],
                              prim_vec=[(2., 0.), (1., 1.)])
# the bond b_s from A to its B neighbour, as the cell offset R of B:
# b1 = (1, 0), b2 = (0, 1), b3 = (-1, 0), b4 = (0, -1)
RUDNER_BONDS = [(0, 0), (-1, 1), (-1, 0), (0, -1)]


def rudner_hops(s, J):
    return [] if s == 4 else [{'i': 0, 'j': 1, 'R': RUDNER_BONDS[s], 't': -J}]


def rudner(J, delta_ab=0., period=1., width=None, order=(0, 1, 2, 3, 4)):
    '''
    The five-step model: hopping J along b_s in step s = 1..4, none in step 5,
    and the sublattice potential delta_AB = e_A - e_B throughout.
    '''
    onsite = {'a': delta_ab / 2, 'b': -delta_ab / 2}
    models = []
    for s in order:
        if width is None:
            ks = KSpace(RUDNER_LAT)
            ks.set_hopping(rudner_hops(s, J))
            ks.set_onsite(onsite)
        else:
            ks = ribbon(RUDNER_LAT, rudner_hops(s, J), width, onsite=onsite)
        models.append(ks)
    return step_drive(models, [period / 5] * 5)


def rudner_flake(n, J, period=1.):
    '''
    A (2n x 2n)-site square flake of the five-step model (delta_AB = 0), as
    System instances: the bonds b_s picked by angle and sublattice tags.
    '''
    lat = Lattice(unit_cell=[{'tag': 'a', 'r0': (0., 0.)}, {'tag': 'b', 'r0': (1., 0.)},
                                        {'tag': 'b', 'r0': (0., 1.)}, {'tag': 'a', 'r0': (1., 1.)}],
                      prim_vec=[(2., 0.), (0., 2.)])
    lat.get_lattice(n1=n, n2=n)
    models = []
    for ang, tag in [(0., 'ab'), (90., 'ab'), (0., 'ba'), (90., 'ba')]:
        sys = System(lat)
        sys.set_hopping([{'n': 1, 'ang': ang, 'tag': tag, 't': -J}])
        models.append(sys)
    models.append(np.zeros((lat.sites, lat.sites)))
    return lat, step_drive(models, [period / 5] * 5)


def old_effective_hamiltonian(ham_t, period, n_steps):
    # verbatim tbkit 0.3.0 + FloquetKSpace (before the branch cut option)
    u = evolution_operator(ham_t, period, n_steps)
    tri, z = LA.schur(u, output='complex')
    eps = -np.angle(np.diag(tri)) / period
    order = np.argsort(eps, kind='stable')
    eps, z = eps[order], z[:, order]
    h_f = z @ np.diag(eps) @ z.conj().T
    return (h_f + h_f.conj().T) / 2


class TestBranchCut(unittest.TestCase):

    def test_default_unchanged(self):
        # the default reproduces the pre-epsilon behaviour bit for bit
        ham_t = two_level(3., 0.8)
        period = 2 * np.pi / 3.
        old = old_effective_hamiltonian(ham_t, period, 50)
        self.assertTrue(np.array_equal(effective_hamiltonian(ham_t, period, 50), old))
        self.assertTrue(np.array_equal(quasienergies(ham_t, period, 50), np.linalg.eigvalsh(old)) or
                              np.allclose(quasienergies(ham_t, period, 50), np.linalg.eigvalsh(old), atol=1e-14))
        omega, a0 = 12., 0.6
        fl = FloquetKSpace(graphene(), circular(a0, omega), 2*np.pi/omega, 60)
        k = np.array([0.4, -0.9])
        ref = old_effective_hamiltonian(lambda t: graphene().get_ham_peierls(k, circular(a0, omega)(t)),
                                                    2*np.pi/omega, 60)
        self.assertTrue(np.array_equal(fl.get_ham(k), ref))
        self.assertIsNone(fl.epsilon)

    def test_epsilon(self):
        h = np.diag([0.5, 2.5])
        period = 2 * np.pi / 3.  # omega = 3
        self.assertTrue(np.allclose(quasienergies(lambda t: h, period, 4, epsilon=3.), [0.5, 2.5]))
        self.assertTrue(np.allclose(quasienergies(lambda t: h, period, 4, epsilon=1.), [-0.5, 0.5]))
        self.assertTrue(np.allclose(quasienergies(lambda t: h, period, 4, epsilon=0.), [-2.5, -0.5]))
        # epsilon = omega/2 is the default zone; the cut moved by omega shifts H_F by omega
        ham_t = two_level(3., 0.8)
        h_f = effective_hamiltonian(ham_t, period, 50)
        self.assertTrue(np.allclose(effective_hamiltonian(ham_t, period, 50, epsilon=1.5), h_f))
        self.assertTrue(np.allclose(effective_hamiltonian(ham_t, period, 50, epsilon=4.5), h_f + 3 * np.eye(2)))
        # the quasienergies lie in [epsilon - omega, epsilon), also for eigenvalues on the cut
        eps = quasienergies(lambda t: h, period, 4, epsilon=2.5)
        self.assertTrue(np.all((eps >= 2.5 - 3.) & (eps < 2.5)))
        fl = FloquetKSpace(graphene(), circular(0.6, 12.), 2*np.pi/12., 30, epsilon=0.)
        self.assertTrue(np.all(np.linalg.eigvalsh(fl.get_ham([0.3, 0.2])) < 0.))


class TestStepDrive(unittest.TestCase):

    def test_exact_evolution(self):
        rng = np.random.default_rng(1)
        hams = [(m + m.conj().T) / 2 for m in rng.normal(size=(3, 4, 4)) + 1j * rng.normal(size=(3, 4, 4))]
        durations = [0.3, 0.5, 0.2]
        drive = step_drive(hams, durations)
        self.assertIsInstance(drive, StepDrive)
        self.assertAlmostEqual(drive.period, 1.)
        u = LA.expm(-0.2j * hams[2]) @ LA.expm(-0.5j * hams[1]) @ LA.expm(-0.3j * hams[0])
        self.assertTrue(np.allclose(drive.evolution_operator(), u))
        self.assertTrue(np.allclose(drive.evolution_operator(0.5),
                                            LA.expm(-0.2j * hams[1]) @ LA.expm(-0.3j * hams[0])))
        self.assertTrue(np.allclose(drive.evolution_operator(0.), np.eye(4)))
        self.assertTrue(np.array_equal(drive.get_ham(0.4), hams[1]))
        self.assertTrue(np.array_equal(drive.get_ham(1.95), hams[2]))
        # quasienergies and H_F, with and without a branch cut
        eps = drive.quasienergies()
        self.assertTrue(np.allclose(LA.expm(-1j * drive.effective_hamiltonian()), u))
        self.assertTrue(np.allclose(np.linalg.eigvalsh(drive.effective_hamiltonian()), eps))
        self.assertTrue(np.all(drive.quasienergies(epsilon=0.) < 0.))
        cut = StepDrive(hams, durations, epsilon=0.)
        self.assertTrue(np.allclose(cut.effective_hamiltonian(), drive.effective_hamiltonian(epsilon=0.)))
        # stroboscopic evolution
        psi = np.zeros(4, 'c16')
        psi[0] = 1.
        out = drive.evolve(psi, 3)
        self.assertEqual(out.shape, (4, 4))
        self.assertTrue(np.allclose(out[3], np.linalg.matrix_power(u, 3) @ psi))
        # sparse matrices are accepted
        sp = step_drive([sparse.csr_matrix(h) for h in hams], durations)
        self.assertTrue(np.allclose(sp.evolution_operator(), u))

    def test_step_kspace_matches_time_grid(self):
        # a step drive is the midpoint time grid when the grid fits in the steps
        drive = rudner(1.7 * np.pi, 0.4)
        self.assertIsInstance(drive, DrivenKSpace)
        grid = DrivenKSpace(drive.steps[0], drive.ham_kt, 1., n_steps=100)
        k = np.array([0.3, -1.1])
        self.assertTrue(np.allclose(grid.get_evolution(k), drive.get_evolution(k), atol=1e-12))
        self.assertTrue(np.allclose(grid.get_ham(k), drive.get_ham(k), atol=1e-12))
        self.assertTrue(np.allclose(grid.get_evolution(k, 0.3), drive.get_evolution(k, 0.3), atol=1e-12))
        self.assertTrue(np.allclose(drive.get_evolution(k, 0.1),
                                            LA.expm(-0.1j * drive.steps[0].get_ham(k))))
        self.assertTrue(drive.is_hermitian())
        self.assertRaises(ValueError, drive.finite_ham, 2)

    def test_general_drive(self):
        # a sublattice mass switched on and off: no hopping in step 2, so
        # the drive is diagonal and H_F is the time average
        base = KSpace(RUDNER_LAT)
        base.set_hopping(rudner_hops(0, 0.3))
        sz = np.diag([1., -1.])
        drive = DrivenKSpace(base, lambda k, t: 0.5 * np.cos(2 * np.pi * t) * sz, 1., n_steps=40)
        self.assertTrue(np.allclose(drive.get_ham([0.2, 0.1]), 0., atol=1e-12))
        self.assertAlmostEqual(drive.winding_number(np.pi, nk=8, n_t=40), 0., places=10)


class TestRudnerModel(unittest.TestCase):

    def test_perfect_transfer(self):
        period = 1.
        drive = rudner(2.5 * np.pi / period)
        # U(T) = 1: both bands at quasienergy 0, zero Chern numbers
        for k in [(0.3, 0.7), (1.2, -2.)]:
            self.assertTrue(np.allclose(drive.get_evolution(k), np.eye(2), atol=1e-12))
        self.assertTrue(np.allclose(drive.mesh_bands(8), 0., atol=1e-12))
        self.assertAlmostEqual(drive.chern_number([0, 1], 8), 0., places=10)
        # yet one chiral mode per edge in the gap at pi/T
        self.assertAlmostEqual(drive.winding_number(np.pi / period), 1., places=10)
        n_lower, n_upper = rudner(2.5 * np.pi, width=8).edge_state_count(np.pi / period)
        self.assertAlmostEqual(n_lower, 1., places=10)
        self.assertAlmostEqual(n_upper, -1., places=10)
        # the reversed cycle (4-3-2-1-5) circulates the other way
        reverse = rudner(2.5 * np.pi, order=(3, 2, 1, 0, 4))
        self.assertAlmostEqual(reverse.winding_number(np.pi, nk=16), -1., places=10)

    def test_phases(self):
        # Fig. 3d of the paper, at delta_AB = 0.5 pi/T: the gaps close at
        # J = 1.21 and 2.03 pi/T (the paper reads 1.3 and 2.1 off its figure)
        expected = {1.0: (0, 0, 0), 1.6: (0, 1, -1), 2.4: (1, 1, 0)}
        for jn, (w0, wpi, c_up) in expected.items():
            drive = rudner(jn * np.pi, 0.5 * np.pi)
            self.assertAlmostEqual(drive.winding_number(0.), w0, delta=1e-3)
            self.assertAlmostEqual(drive.winding_number(np.pi), wpi, delta=1e-3)
            self.assertAlmostEqual(drive.chern_number(1, 16), c_up, places=6)
            # W(pi) - W(0) = -C of the upper band
            self.assertAlmostEqual(wpi - w0, -c_up)
            # independent count on a ribbon, gap by gap (the edge states' tail
            # on the other half, 5e-3 at width 12, falls exponentially with the width)
            rib = rudner(jn * np.pi, 0.5 * np.pi, width=20)
            for eps, w in [(0., w0), (np.pi, wpi)]:
                n_lower, n_upper = rib.edge_state_count(eps)
                self.assertAlmostEqual(n_lower, w, delta=1e-3)
                self.assertAlmostEqual(n_upper, -w, delta=1e-3)

    def test_convergence(self):
        # near a gap closing the raw value converges as nk grows
        drive = rudner(1.9 * np.pi, 0.5 * np.pi)
        errors = [abs(drive.winding_number(0., nk)) for nk in (24, 48, 96)]
        self.assertGreater(errors[0], 0.05)
        self.assertLess(errors[2], 1e-3)
        self.assertTrue(errors[0] > errors[1] > errors[2])
        # the time quadrature is already converged
        self.assertAlmostEqual(drive.winding_number(0., (96, 96), n_t=80), drive.winding_number(0., 96), places=8)

    def test_real_space_flake(self):
        lat, drive = rudner_flake(4, 2.5 * np.pi)
        x, y = lat.coor['x'], lat.coor['y']
        u = drive.evolution_operator()
        # every site goes to exactly one site after a period
        self.assertTrue(np.allclose(np.sort(np.abs(u), axis=0)[-1], 1.))
        # bulk sites return
        bulk = np.where((x > 0.5) & (x < 6.5) & (y > 0.5) & (y < 6.5))[0]
        self.assertTrue(np.allclose(np.abs(u[bulk, bulk]), 1.))
        # a particle on the bottom edge moves right, one unit cell (2 sites) per period
        start = np.where((x == 0.) & (y == 0.))[0][0]
        psi = np.zeros(lat.sites, 'c16')
        psi[start] = 1.
        states = drive.evolve(psi, 3)
        for n, state in enumerate(states):
            site = np.argmax(np.abs(state))
            self.assertAlmostEqual(abs(state[site]), 1.)
            self.assertEqual((x[site], y[site]), (2. * n, 0.))
        # and on the top edge it moves left
        start = np.where((x == 6.) & (y == 7.))[0][0]
        psi = np.zeros(lat.sites, 'c16')
        psi[start] = 1.
        site = np.argmax(np.abs(drive.evolve(psi, 2)[2]))
        self.assertEqual((x[site], y[site]), (2., 7.))


class TestWindingChern(unittest.TestCase):

    def test_static_chern_insulator(self):
        # a static model: W = 0 at the zone edge, W = -C(lower) in its gap
        hops = [{'i': 0, 'j': 0, 'R': (1, 0), 't': 0.5 * (PAULI['z'] - 1j * PAULI['x'])},
                    {'i': 0, 'j': 0, 'R': (0, 1), 't': 0.5 * (PAULI['z'] - 1j * PAULI['y'])}]
        qwz = KSpace(lattices.square(), spin=True)
        qwz.set_hopping(hops)
        qwz.set_onsite({'a': PAULI['z']})
        chern = qwz.chern_number(0, 20)
        self.assertAlmostEqual(chern, -1., places=8)
        drive = step_drive([qwz], [1.])
        self.assertAlmostEqual(drive.winding_number(np.pi), 0., places=8)
        self.assertAlmostEqual(drive.winding_number(0.), -chern, places=6)
        rib = step_drive([ribbon(lattices.square(), hops, 12, onsite={'a': PAULI['z']}, spin=True)], [1.])
        n_lower, n_upper = rib.edge_state_count(0., window=1.)
        self.assertAlmostEqual(n_lower, -chern, places=6)
        self.assertAlmostEqual(n_upper, chern, places=6)

    def test_circularly_driven_graphene(self):
        # the Floquet Chern insulator at omega = 12: the bands fit in the
        # zone (no anomalous edge states, W = 0 at pi/T) and W = -C in the gap
        omega, a0 = 12., 0.6
        period = 2 * np.pi / omega
        fl = FloquetKSpace(graphene(), circular(a0, omega), period, 60)
        self.assertLess(np.max(np.abs(fl.mesh_bands(20))), 0.5 * omega - 3.)
        chern = fl.chern_number(0, 16)
        self.assertAlmostEqual(fl.winding_number(np.pi / period, 24), 0., places=8)
        self.assertAlmostEqual(fl.winding_number(0., 96), -chern, delta=5e-3)

    def test_grid_fallback(self):
        # models whose class redefines get_ham/get_ham_peierls are evaluated point by point
        class Shifted(KSpace):
            def get_ham(self, k):
                return super().get_ham(k)

            def get_ham_peierls(self, k, A):
                return super().get_ham_peierls(k, A)
        g = Shifted(lattices.honeycomb())
        g.set_hopping([{'i': 0, 'j': 1, 'R': R, 't': 1.} for R in [(0, 0), (-1, 0), (0, -1)]])
        omega, a0 = 12., 0.6
        fl = FloquetKSpace(g, circular(a0, omega), 2 * np.pi / omega, 20)
        ref = FloquetKSpace(graphene(), circular(a0, omega), 2 * np.pi / omega, 20)
        self.assertAlmostEqual(fl.winding_number(np.pi * omega, 8), ref.winding_number(np.pi * omega, 8), places=10)
        drive = step_drive([g, g], [0.5, 0.5])
        self.assertAlmostEqual(drive.winding_number(np.pi, 8), 0., places=10)


class TestAnomalousChecks(unittest.TestCase):

    def test_checks(self):
        drive = rudner(2.5 * np.pi)
        rib = rudner(2.5 * np.pi, width=4)
        self.assertRaises(TypeError, effective_hamiltonian, two_level(), 1., 10, 'a')
        self.assertRaises(TypeError, FloquetKSpace, graphene(), lambda t: (0., 0.), 1., 5, [0.])
        self.assertRaises(TypeError, DrivenKSpace, 1., lambda k, t: np.eye(2), 1.)
        self.assertRaises(ValueError, DrivenKSpace, graphene(), lambda k, t: np.eye(3), 1.)
        self.assertRaises(ValueError, DrivenKSpace, graphene(), lambda k, t: np.array([[0., 1.], [0., 0.]]), 1.)
        self.assertRaises(TypeError, step_drive, [], [1.])
        self.assertRaises(TypeError, step_drive, [graphene(), 1.], [1., 1.])
        self.assertRaises(ValueError, step_drive, [graphene()], [1., 1.])
        self.assertRaises(ValueError, step_drive, [graphene()], [-1.])
        self.assertRaises(TypeError, step_drive, [np.eye(2)], 1.)
        self.assertRaises(TypeError, step_drive, [np.eye(2)], ['a'])
        self.assertRaises(ValueError, step_drive, [np.eye(2), np.eye(3)], [1., 1.])
        self.assertRaises(ValueError, step_drive, [np.array([[0., 1.], [0., 0.]])], [1.])
        self.assertRaises(ValueError, drive.get_evolution, [0., 0.], 2.)
        self.assertRaises(ValueError, drive.winding_number, 0., 2)
        self.assertRaises(ValueError, rib.winding_number, 0.)
        self.assertRaises(ValueError, drive.edge_state_count, 0.)
        self.assertRaises(ValueError, rib.edge_state_count, 0., 2)
        self.assertRaises(ValueError, rib.edge_state_count, 0., 10, -1.)
        sd = step_drive([np.eye(2)], [1.])
        self.assertRaises(ValueError, sd.evolve, np.zeros(3), 1)
        self.assertRaises(ValueError, sd.evolution_operator, -0.5)
        self.assertRaises(TypeError, sd.get_ham, 'a')


if __name__ == '__main__':
    unittest.main()
