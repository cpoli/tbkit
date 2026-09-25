"""
Intrinsic anomalous and spin Hall conductivities: the Kubo formula over the
Brillouin zone (KSpace.hall_conductivity, KSpace.spin_hall_conductivity).
The real-space Kubo-Bastin version (tbkit.kpm.hall_conductivity) is tested
in test_kpm.py.
"""
import unittest

import numpy as np
import scipy.linalg as LA

import tbkit.lattices as lattices
from tbkit.floquet import FloquetKSpace
from tbkit.kspace import KSpace, PAULI
from tbkit.lattice import Lattice
from tests.test_three_d import weyl
from tests.test_topology import kane_mele as kane_mele_rashba


DX, DY = 0.5 * np.sqrt(3), 0.5
# the three second-neighbour vectors a2, -a1, a1 - a2, 120 degrees apart
NNN = [(0, 1), (-1, 0), (1, -1)]


def haldane(t2=0.2, phi=np.pi/2, M=0.):
    ks = KSpace(lattices.honeycomb())
    ks.set_hopping([{'i': 0, 'j': 1, 'R': R, 't': 1.} for R in [(0, 0), (-1, 0), (0, -1)]])
    for R in NNN:
        ks.set_hopping([{'i': 0, 'j': 0, 'R': R, 't': t2*np.exp(1j*phi)},
                                 {'i': 1, 'j': 1, 'R': R, 't': t2*np.exp(-1j*phi)}])
    ks.set_onsite({'a': M, 'b': -M})
    return ks


def kane_mele(lam):
    '''Kane-Mele without Rashba, and its two decoupled spin sectors (Haldane models).'''
    km = kane_mele_rashba(lam=lam)
    return km, haldane(t2=lam, phi=np.pi/2), haldane(t2=lam, phi=-np.pi/2)


def qwz(m):
    '''
    Qi-Wu-Zhang model, H = sin kx sx + sin ky sy + (m + 2 - cos kx - cos ky) sz:
    a single Dirac cone v(kx sx + ky sy) + m sz at Gamma, with v = 1.
    '''
    lat = Lattice(unit_cell=[{'tag': 'a', 'r0': (0., 0.)}, {'tag': 'b', 'r0': (0., 0.)}],
                         prim_vec=[(1., 0.), (0., 1.)])
    ks = KSpace(lat)
    hop = []
    for R, s in (((1, 0), 'x'), ((0, 1), 'y')):
        block = -0.5j * PAULI[s] - 0.5 * PAULI['z']
        hop += [{'i': i, 'j': j, 'R': R, 't': complex(block[i, j])}
                     for i in range(2) for j in range(2) if block[i, j] != 0]
    ks.set_hopping(hop)
    ks.set_onsite({'a': m + 2., 'b': -(m + 2.)})
    return ks


def anisotropic_haldane(M=0.1):
    '''
    Haldane model with three different nearest-neighbour hoppings: no C3
    symmetry left, so the two position conventions differ in a band.
    '''
    hal = haldane(t2=0.15, M=M)
    hal.clear_hopping()
    hal.set_hopping([{'i': 0, 'j': 1, 'R': (0, 0), 't': 1.},
                            {'i': 0, 'j': 1, 'R': (-1, 0), 't': 0.6},
                            {'i': 0, 'j': 1, 'R': (0, -1), 't': 1.3}])
    for R in NNN:
        hal.set_hopping([{'i': 0, 'j': 0, 'R': R, 't': 0.15j}, {'i': 1, 'j': 1, 'R': R, 't': -0.15j}])
    return hal


def haldane_stack(tz=0.1, c=2.):
    '''Haldane layers stacked along z, c apart, coupled by tz.'''
    lat = Lattice(unit_cell=[{'tag': 'a', 'r0': (0., 0., 0.)}, {'tag': 'b', 'r0': (DX, DY, 0.)}],
                         prim_vec=[(2*DX, 0., 0.), (DX, 1.5, 0.), (0., 0., c)])
    ks = KSpace(lat)
    ks.set_hopping([{'i': 0, 'j': 1, 'R': R, 't': 1.} for R in [(0, 0, 0), (-1, 0, 0), (0, -1, 0)]])
    for R in [(0, 1, 0), (-1, 0, 0), (1, -1, 0)]:
        ks.set_hopping([{'i': 0, 'j': 0, 'R': R, 't': 0.2j}, {'i': 1, 'j': 1, 'R': R, 't': -0.2j}])
    ks.set_hopping([{'i': o, 'j': o, 'R': (0, 0, 1), 't': tz} for o in (0, 1)])
    return ks


def torus_kubo(ks, n, e_fermi):
    '''Exact Kubo formula on an n x n torus, from its eigenstates and finite_velocity.'''
    ham = ks.finite_ham(n, periodic=True)
    vx, vy = ks.finite_velocity(n, periodic=True)
    en, vec = LA.eigh(ham)
    x, y = vec.conj().T @ vx @ vec, vec.conj().T @ vy @ vec
    de = en[:, None] - en[None, :]
    deg = np.abs(de) < 1e-9
    w = np.where(deg, 0., np.imag(x * y.T) / np.where(deg, 1., de) ** 2)
    area = n * n * abs(np.linalg.det(np.array(ks.lat.prim_vec)))
    f = (en[None, :] < np.asarray(e_fermi)[:, None]).astype(float)
    return -2 * np.pi / area * np.einsum('en,nm->e', f, 2 * w)


class TestHallConductivity(unittest.TestCase):

    def test_gap_equals_chern_number(self):
        # both phases of the Haldane model, for several parameters: in the
        # gap, sigma_xy / (e^2/h) is the Chern number of the lower band
        cherns = []
        for t2, phi, M in [(0.2, np.pi/2, 0.), (0.2, np.pi/2, 0.8), (0.2, np.pi/2, 1.5),
                                  (0.15, np.pi/3, 0.3), (0.2, -np.pi/2, 0.4), (0.3, 2.5, -0.5)]:
            hal = haldane(t2=t2, phi=phi, M=M)
            en = hal.mesh_bands(60)
            e_mid = (en[:, 0].max() + en[:, 1].min()) / 2
            chern = hal.chern_number([0], 60)
            self.assertAlmostEqual(hal.hall_conductivity(e_mid, nk=60), chern, places=6)
            self.assertAlmostEqual(hal.hall_conductivity(e_mid, nk=60, positions=False), chern, places=6)
            cherns.append(round(chern))
        self.assertEqual(set(cherns), {-1, 0, 1})

    def test_empty_full_and_array_shape(self):
        hal = haldane(M=0.3)
        self.assertEqual(hal.hall_conductivity(-5.), 0.)
        self.assertAlmostEqual(hal.hall_conductivity(5.), 0., places=12)
        sigma = hal.hall_conductivity(np.zeros((2, 3)), nk=20)
        self.assertEqual(sigma.shape, (2, 3))
        self.assertIsInstance(hal.hall_conductivity(0., nk=20), float)

    def test_massive_dirac_limit(self):
        # Near the band edge of a small gap 2|m|, sigma follows the Dirac
        # cone at Gamma: its lower band has Omega = m / (2 E^3) (the sign
        # fixed by the Kubo formula at k = 0), so it contributes sgn(m)/2 in
        # the gap and m / (2|E_F|) outside it. The rest of the Brillouin
        # zone does not change near the edge, hence
        #   sigma(E_F) - sigma(gap) = m / (2|E_F|) - sgn(m)/2,
        # within the lattice corrections, of relative order (E_F/m)^2 m:
        # here below 0.006 (E_F/m)^2. (Ohm's-law sign: minus this, the
        # familiar -(e^2/2h) m/|E_F|.)
        for m in (0.05, -0.05):
            ks = qwz(m)
            gap = ks.hall_conductivity(0., nk=400, refine=4, refine_fraction=0.03)
            self.assertAlmostEqual(gap, ks.chern_number([0], 40), places=3)
            ratio = np.array([-2., -1.5, 1.5, 2.])
            sigma = ks.hall_conductivity(ratio * abs(m), nk=400, refine=4, refine_fraction=0.03)
            dirac = m / (2 * np.abs(ratio * m)) - np.sign(m) / 2
            self.assertTrue(np.all(np.abs(sigma - gap - dirac) < 0.006 * ratio ** 2))

    def test_continuous_in_e_fermi_and_converges_in_nk(self):
        hal = haldane(M=0.3)
        # continuous: halving the step in E_F halves the largest jump
        jumps = [np.max(np.abs(np.diff(hal.hall_conductivity(np.linspace(-3.5, 3.5, n), temperature=0.02,
                                                                                         nk=120))))
                      for n in (351, 701, 1401)]
        self.assertLess(jumps[0], 0.04)
        self.assertTrue(np.allclose(np.array(jumps[1:]) / jumps[:-1], 0.5, atol=0.05))
        in_band = np.array([-2.2, -1.4, 1.2])
        values = [hal.hall_conductivity(in_band, temperature=0.05, nk=nk) for nk in (30, 60, 120, 240)]
        errors = [np.max(np.abs(v - values[-1])) for v in values[:-1]]
        self.assertTrue(errors[0] > errors[1] > errors[2])
        self.assertLess(errors[2], 1e-4)
        # at T = 0, the mesh converges too, if more slowly (the Fermi surface)
        ref = hal.hall_conductivity(in_band, nk=600)
        self.assertLess(np.max(np.abs(hal.hall_conductivity(in_band, nk=300) - ref)), 3e-3)

    def test_adaptive_refinement(self):
        # refine = r on every cell of an n-mesh is exactly the (r n)-mesh
        hal = haldane(M=0.9)
        e_f = np.array([-1.3, 0.2, 0.4])
        full = hal.hall_conductivity(e_f, nk=40, refine=3, refine_fraction=1.)
        self.assertTrue(np.allclose(full, hal.hall_conductivity(e_f, nk=120), atol=1e-12))
        # just above a small gap (hot spots at K'), refining 5% of the cells
        # beats the plain mesh by an order of magnitude
        e_f = np.array([0.2, 0.25, 0.4])
        ref = hal.hall_conductivity(e_f, nk=1200)
        plain = np.max(np.abs(hal.hall_conductivity(e_f, nk=120) - ref))
        refined = np.max(np.abs(hal.hall_conductivity(e_f, nk=120, refine=5) - ref))
        self.assertLess(refined, plain / 5)

    def test_time_reversal_symmetric_models_give_zero(self):
        e_f = np.linspace(-3.5, 3.5, 29)
        rng = np.random.default_rng(0)
        # graphene with a mass; Kane-Mele with Rashba (spinful, T^2 = -1);
        # random real hoppings on the kagome lattice (T = complex conjugation)
        gra = haldane(t2=0., M=0.3)
        km = kane_mele_rashba(lam=0.06, M=0.1, rashba=0.05)
        kag = KSpace(lattices.kagome())
        kag.set_hopping([{'i': i, 'j': j, 'R': R, 't': rng.normal()}
                                  for i, j, R in [(0, 1, (0, 0)), (0, 2, (0, 0)), (1, 2, (0, 0)),
                                                         (0, 1, (-1, 0)), (0, 2, (0, -1)), (1, 2, (1, -1))]])
        kag.set_onsite({'a': 0.3, 'b': -0.2, 'c': 0.1})
        for ks in (gra, km, kag):
            for kw in ({}, {'temperature': 0.1}, {'positions': False}):
                self.assertLess(np.max(np.abs(ks.hall_conductivity(e_f, nk=30, **kw))), 1e-12)

    def test_position_conventions(self):
        # full bands: both conventions give C; partial filling: they differ
        # by the circulation of <n|tau|n> around the Fermi surface, which
        # C3 symmetry cancels (Haldane), but not an anisotropic model
        hal = anisotropic_haldane()
        en = hal.mesh_bands(60)
        e_gap = (en[:, 0].max() + en[:, 1].min()) / 2
        e_band = np.array([-2., -1.2, 1.5])
        phys = hal.hall_conductivity(e_band, nk=200)
        periodic = hal.hall_conductivity(e_band, nk=200, positions=False)
        self.assertGreater(np.max(np.abs(phys - periodic)), 0.01)
        for positions in (True, False):
            self.assertAlmostEqual(hal.hall_conductivity(e_gap, nk=60, positions=positions), 1., places=7)
        sym = haldane(M=0.3)
        self.assertTrue(np.allclose(sym.hall_conductivity(e_band, nk=100),
                                             sym.hall_conductivity(e_band, nk=100, positions=False), atol=1e-12))
        # the physical one is positions=True: exact diagonalization of a
        # torus, with the velocity i[H, r] of its bonds, gives the same
        # numbers on the torus's own k-mesh, at every filling
        e_f = np.linspace(-3., 3., 25)
        self.assertTrue(np.allclose(torus_kubo(hal, 6, e_f), hal.hall_conductivity(e_f, nk=6), atol=1e-10))
        self.assertFalse(np.allclose(torus_kubo(hal, 6, e_f),
                                               hal.hall_conductivity(e_f, nk=6, positions=False), atol=1e-3))

    def test_velocity_is_the_derivative_of_the_bloch_hamiltonian(self):
        hal = anisotropic_haldane()
        k = np.array([[0.3, -0.7]])
        tau = hal.orbital_positions()
        dk = 1e-6
        for positions in (False, True):
            def ham(kk):
                h = hal.get_ham(kk)
                if positions:
                    u = np.exp(1j * tau @ kk)
                    h = u.conj()[:, None] * h * u[None, :]
                return h
            ham0, dham = hal._bloch_derivatives(k, list(np.eye(2)), positions)
            self.assertTrue(np.allclose(ham0[0], ham(k[0])))
            for a in range(2):
                e = np.eye(2)[a] * dk
                self.assertTrue(np.allclose(dham[a, 0], (ham(k[0] + e) - ham(k[0] - e)) / (2 * dk), atol=1e-8))

    def test_ohm_sign(self):
        # The docstring's claim: under a field E_y (hbar k_y' = q E_y), a
        # band's velocity picks up -Omega q E_y along x, so Ohm's law gives
        # j_x = -(q^2/h) C E_y, minus the value returned. Checked by
        # evolving the lower band of the QWZ model in real time at one k,
        # with the field switched on smoothly (q = 1).
        ks = qwz(-1.)
        k0, e0, t_on, dt = np.array([0.4, -0.3]), 1e-3, 200., 0.05
        en, vec = LA.eigh(ks.get_ham(k0))
        psi = vec[:, 0]
        field = lambda t: e0 * np.sin(np.pi * min(t, t_on) / (2 * t_on)) ** 2
        k, t = k0.copy(), 0.
        for _ in range(int(t_on / dt)):
            k_mid = k + np.array([0., field(t + dt / 2) * dt / 2])
            psi = LA.expm(-1j * dt * ks.get_ham(k_mid)) @ psi
            k = k + np.array([0., (field(t) + 4 * field(t + dt / 2) + field(t + dt)) * dt / 6])
            t += dt
        _, dham = ks._bloch_derivatives(k[None, :], list(np.eye(2)), True)
        v_x = np.vdot(psi, dham[0, 0] @ psi).real
        en_k, vec_k = LA.eigh(ks.get_ham(k))
        group = np.vdot(vec_k[:, 0], dham[0, 0] @ vec_k[:, 0]).real
        omega = ks._kubo(k[None, :], np.array([0.]), 0., list(np.eye(2)), [(0, 1)], True, None)[0][0, 0, 0]
        self.assertAlmostEqual((v_x - group) / (-omega * e0), 1., places=2)

    def test_three_d(self):
        # a stack of Chern layers: every plane (0, 1) has C = 1; the planes
        # containing the stacking direction have 0; the Hall vector is C/c
        # along z, in e^2/(h length)
        c = 2.
        ks = haldane_stack(c=c)
        for k_fixed in (0., 0.3):
            self.assertAlmostEqual(ks.hall_conductivity(0., nk=40, k_fixed=k_fixed),
                                           ks.chern_number([0], 40, k_fixed=k_fixed), places=6)
        self.assertAlmostEqual(ks.hall_conductivity(0., nk=40, plane=(0, 2), k_fixed=0.1), 0., places=8)
        sigma = ks.hall_conductivity(0., nk=(30, 30, 6))
        self.assertTrue(np.allclose(sigma, [0., 0., 1. / c], atol=1e-6))
        self.assertEqual(ks.hall_conductivity([0., 5.], nk=(10, 10, 4)).shape, (2, 3))
        # a Weyl semimetal with nodes at kz = +-k0: sigma_xy = C k0 / pi
        # (Burkov and Balents), C the Chern number of the planes between them
        wsm = weyl(np.pi / 2)
        chern = wsm.chern_number([0], 30, k_fixed=0.)
        sigma = wsm.hall_conductivity(0., nk=(30, 30, 40), refine=2, refine_fraction=0.1)
        self.assertTrue(np.allclose(sigma, [0., 0., chern / 2], atol=0.01))

    def test_overlap(self):
        # Lowdin-orthogonalized, as chern_number: same Chern number in the gap
        hal = haldane(M=0.3)
        hal.set_overlap([{'i': 0, 'j': 1, 'R': R, 't': 0.1} for R in [(0, 0), (-1, 0), (0, -1)]])
        en = hal.mesh_bands(40)
        e_mid = (en[:, 0].max() + en[:, 1].min()) / 2
        self.assertAlmostEqual(hal.hall_conductivity(e_mid, nk=60), hal.chern_number([0], 60), places=6)
        # the derivative of S^-1/2 H S^-1/2 is exact
        k = np.array([[0.2, 0.5]])
        dk = 1e-6
        for positions in (False, True):
            ham0, dham = hal._bloch_derivatives(k, list(np.eye(2)), positions)
            for a in range(2):
                e = np.eye(2)[a] * dk
                plus = hal._bloch_derivatives(k + e, list(np.eye(2)), positions)[0]
                minus = hal._bloch_derivatives(k - e, list(np.eye(2)), positions)[0]
                self.assertTrue(np.allclose(dham[a, 0], (plus[0] - minus[0]) / (2 * dk), atol=1e-7))
        # a degenerate overlap spectrum (on-site identity blocks) is handled
        sq = KSpace(lattices.square(), spin=True)
        sq.set_hopping([{'i': 0, 'j': 0, 'R': (1, 0), 't': 1.}, {'i': 0, 'j': 0, 'R': (0, 1), 't': 1.}])
        sq.set_overlap([{'i': 0, 'j': 0, 'R': (1, 0), 't': 0.1}])
        self.assertAlmostEqual(sq.hall_conductivity(0., nk=10), 0., places=12)
        bad = haldane(M=0.3)
        bad.set_overlap([{'i': 0, 'j': 1, 'R': R, 't': 0.9} for R in [(0, 0), (-1, 0), (0, -1)]])
        self.assertRaises(ValueError, bad.hall_conductivity, 0.)

    def test_errors(self):
        hal = haldane(M=0.3)
        self.assertRaises(TypeError, hal.hall_conductivity, 'a')
        self.assertRaises(ValueError, hal.hall_conductivity, [])
        self.assertRaises(ValueError, hal.hall_conductivity, np.nan)
        self.assertRaises(ValueError, hal.hall_conductivity, 0., -1.)
        self.assertRaises(ValueError, hal.hall_conductivity, 0., 0., (10, 10, 10))
        self.assertRaises(TypeError, hal.hall_conductivity, 0., 0., 10, [0, 1])
        self.assertRaises(TypeError, hal.hall_conductivity, 0., 0., 10, (0, 1), 'a')
        self.assertRaises(TypeError, hal.hall_conductivity, 0., 0., 10, (0, 1), None, 1)
        self.assertRaises(ValueError, hal.hall_conductivity, 0., 0., 10, (0, 1), None, True, 0)
        self.assertRaises(ValueError, hal.hall_conductivity, 0., 0., 10, (0, 1), None, True, 2, 1.5)
        self.assertRaises(ValueError, hal.spin_hall_conductivity, 0.)  # spinless
        chain = KSpace(lattices.chain())
        chain.set_hopping([{'i': 0, 'j': 0, 'R': (1,), 't': 1.}])
        self.assertRaises(ValueError, chain.hall_conductivity, 0.)
        lossy = haldane(M=0.3)
        lossy.set_onsite({'a': 0.3 + 0.1j})
        self.assertRaises(ValueError, lossy.hall_conductivity, 0.)
        driven = FloquetKSpace(hal, lambda t: (0.1 * np.cos(t), 0.1 * np.sin(t)), 2 * np.pi)
        self.assertRaises(ValueError, driven.hall_conductivity, 0.)
        km, _, _ = kane_mele(0.06)
        self.assertRaises(ValueError, km.spin_hall_conductivity, 0., spin_axis='w')
        self.assertRaises(ValueError, weyl().spin_hall_conductivity, 0.)  # 3D: a plane is needed


class TestSpinHall(unittest.TestCase):

    def test_kane_mele_quantized(self):
        # S_z conserved: in the gap, sigma_s = (C_up - C_dn)/2 in units of
        # e/2pi, with C_up, C_dn the Chern numbers of the decoupled sectors
        for lam in (0.06, -0.1):
            km, up, down = kane_mele(lam)
            c_up, c_dn = up.chern_number([0], 40), down.chern_number([0], 40)
            self.assertAlmostEqual(abs(c_up - c_dn), 2., places=6)
            sigma_s = km.spin_hall_conductivity(0., nk=60)
            self.assertAlmostEqual(sigma_s, (c_up - c_dn) / 2, places=6)
            self.assertAlmostEqual(km.hall_conductivity(0., nk=30), 0., places=12)
            # and, in a band, the sum of the two sectors' Hall conductivities
            e_f = np.array([-2., -0.8, 1.1])
            sectors = (up.hall_conductivity(e_f, nk=60) - down.hall_conductivity(e_f, nk=60)) / 2
            self.assertTrue(np.allclose(km.spin_hall_conductivity(e_f, nk=60), sectors, atol=1e-10))
        # the in-plane spin components carry no spin Hall current here
        self.assertAlmostEqual(km.spin_hall_conductivity(0., nk=30, spin_axis='x'), 0., places=10)

    def test_rashba_breaks_quantization(self):
        km = kane_mele_rashba(lam=0.06, rashba=0.05)
        sigma_s = km.spin_hall_conductivity(0., nk=60)
        self.assertTrue(0.005 < abs(sigma_s) - 1. < 0.1)
        self.assertEqual(km.z2_invariant([0, 1], nk=40, nk_perp=21), 1)

    def test_three_d_plane(self):
        wsm = weyl(np.pi / 2)
        self.assertLess(abs(wsm.spin_hall_conductivity(-5., nk=10, k_fixed=0.)), 1e-12)


if __name__ == '__main__':
    unittest.main()
