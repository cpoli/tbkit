'''
Tests of tbkit.models: each named model gives, in each phase, the
invariant its gallery example asserts.
'''
import unittest

import numpy as np

import tbkit.error_handling as eh
from tbkit import models
from tbkit.bdg import majorana_number
from tbkit.bridges import finite_model, finite_system
from tbkit.kspace import PAULI, reciprocal_vectors
from tbkit.topology import find_weyl_points

INVERSION = np.kron(np.diag([1., -1.]), np.eye(2))  # s even, p odd (with spin)


def valleys(ks):
    b1, b2 = (np.array(v) for v in reciprocal_vectors(ks.lat.prim_vec))
    return (b1 - b2) / 3, (b2 - b1) / 3


class TestOneDimension(unittest.TestCase):
    def test_ssh(self):
        for v, w, zak in [(0.5, 1., np.pi), (1., 0.5, 0.)]:
            ks = models.ssh(v, w)
            self.assertLess(abs(abs(ks.berry_phase(0, positions=False)) - zak), 1e-10)
            gap = np.min(np.diff(ks.mesh_bands(200), axis=1))
            self.assertLess(abs(gap - 2 * abs(v - w)), 1e-10)

    def test_rice_mele(self):
        ks = models.rice_mele(t=1., delta=0.2, Delta=0.3)
        self.assertTrue(np.allclose(np.linalg.eigvalsh(ks.get_ham([np.pi])),
                                    [-np.hypot(0.3, 0.4), np.hypot(0.3, 0.4)]))
        self.assertTrue(np.allclose(models.rice_mele(1., 0.2).get_ham([0.7]),
                                    models.ssh(1.2, 0.8).get_ham([0.7])))

    def test_pump(self):
        self.assertLess(abs(abs(models.rice_mele_pump().chern_number([0], nk=40)) - 1.), 1e-6)
        self.assertLess(abs(models.rice_mele_pump(offset=1.2).chern_number([0], nk=40)), 1e-6)

    def test_kitaev_chain(self):
        for mu, number in [(0., -1), (1.5, -1), (-1.5, -1), (2.5, 1), (-3., 1)]:
            self.assertEqual(majorana_number(models.kitaev_chain(t=1., delta=0.6, mu=mu)), number)
        k, t, delta, mu = 0.7, 1., 0.6, 0.5
        e = np.sqrt((2 * t * np.cos(k) - mu) ** 2 + 4 * delta ** 2 * np.sin(k) ** 2)
        self.assertTrue(np.allclose(np.linalg.eigvalsh(models.kitaev_chain(t, delta, mu).get_ham([k])),
                                    [-e, e]))


class TestTwoDimensions(unittest.TestCase):
    def test_haldane(self):
        m_c = 3 * np.sqrt(3) * 0.1
        for phi, M, chern in [(np.pi / 2, 0., 1), (-np.pi / 2, 0., -1), (np.pi / 2, 1.2 * m_c, 0),
                              (np.pi / 2, -0.8 * m_c, 1), (np.pi / 3, 0.2, 1)]:
            ks = models.haldane(t2=0.1, phi=phi, M=M)
            self.assertLess(abs(ks.chern_number([0], nk=30) - chern), 1e-6)
        ks = models.haldane(t2=0.2, M=0.3)
        K, Kp = valleys(ks)
        for k, gap in [(K, 2 * abs(0.3 + 3 * np.sqrt(3) * 0.2)), (Kp, 2 * abs(0.3 - 3 * np.sqrt(3) * 0.2))]:
            self.assertLess(abs(np.ptp(np.linalg.eigvalsh(ks.get_ham(k))) - gap), 1e-10)

    def test_qi_wu_zhang(self):
        for m, chern in [(-3., 0), (-1., 1), (1., -1), (3., 0)]:
            self.assertLess(abs(models.qi_wu_zhang(m).chern_number([0], nk=30) - chern), 1e-6)
        k = np.array([0.3, -1.1])
        h = (np.sin(k[0]) * PAULI['x'] + np.sin(k[1]) * PAULI['y']
             + (0.5 + np.cos(k).sum()) * PAULI['z'])
        self.assertTrue(np.allclose(models.qi_wu_zhang(0.5).get_ham(k), h))

    def test_kane_mele(self):
        m_c = 3 * np.sqrt(3) * 0.06
        for lambda_v, nu in [(0., 1), (0.5 * m_c, 1), (1.5 * m_c, 0)]:
            self.assertEqual(models.kane_mele(lambda_v=lambda_v).z2_invariant([0, 1], nk=40, nk_perp=31), nu)
        rashba = models.kane_mele(lambda_v=0.1, lambda_r=0.05)
        self.assertEqual(rashba.z2_invariant([0, 1], nk=40, nk_perp=31), 1)
        self.assertLess(abs(rashba.chern_number([0, 1], nk=20)), 1e-6)
        self.assertGreater(np.abs(rashba.get_ham((0.3, 0.2)) - models.kane_mele(lambda_v=0.1).get_ham((0.3, 0.2))).max(), 0.01)

    def test_bhz(self):
        for M in (-1., 1., 3., 5., 7., 9.):
            nu = int(0 < M < 4 or 4 < M < 8)
            ks = models.bhz(M=M)
            self.assertEqual(ks.parity_z2(INVERSION, [0, 1]), nu)
            self.assertEqual(ks.z2_invariant([0, 1], nk=40, nk_perp=31), nu)
        # epsilon(k) shifts both orbitals equally: the invariant is unchanged
        self.assertEqual(models.bhz(M=1., C=0.3, D=0.5).z2_invariant([0, 1], nk=40, nk_perp=31), 1)
        k, A, B, C, D, M = np.array([0.4, -0.9]), 0.7, 1.2, 0.3, 0.2, 1.5
        s = 2 - np.cos(k).sum()
        dz, eps = M - 2 * B * s, C - 2 * D * s
        up = eps * np.eye(2) + A * np.sin(k[0]) * PAULI['x'] + A * np.sin(k[1]) * PAULI['y'] + dz * PAULI['z']
        down = eps * np.eye(2) - A * np.sin(k[0]) * PAULI['x'] + A * np.sin(k[1]) * PAULI['y'] + dz * PAULI['z']
        h = models.bhz(A, B, C, D, M).get_ham(k)  # (s up, s down, p up, p down)
        self.assertTrue(np.allclose(h[np.ix_([0, 2], [0, 2])], up))
        self.assertTrue(np.allclose(h[np.ix_([1, 3], [1, 3])], down))
        self.assertTrue(np.allclose(h[np.ix_([0, 2], [1, 3])], 0.))

    def test_tmd_three_band(self):
        # Liu et al., PRB 88, 085433 (2013), Eq. (4)-(11)
        p = models.TMD_PARAMETERS['MoS2']
        ks = models.tmd_three_band()
        a = p['a']
        r3 = np.sqrt(3)
        for k in (np.array([0.31, -0.57]), np.array([-0.9, 0.2])):
            al, be = k[0] * a / 2, r3 * k[1] * a / 2
            c, s = np.cos, np.sin
            h0 = 2 * p['t0'] * (c(2 * al) + 2 * c(al) * c(be)) + p['eps1']
            h1 = -2 * r3 * p['t2'] * s(al) * s(be) + 2j * p['t1'] * (s(2 * al) + s(al) * c(be))
            h2 = 2 * p['t2'] * (c(2 * al) - c(al) * c(be)) + 2j * r3 * p['t1'] * c(al) * s(be)
            h11 = 2 * p['t11'] * c(2 * al) + (p['t11'] + 3 * p['t22']) * c(al) * c(be) + p['eps2']
            h22 = 2 * p['t22'] * c(2 * al) + (3 * p['t11'] + p['t22']) * c(al) * c(be) + p['eps2']
            h12 = r3 * (p['t22'] - p['t11']) * s(al) * s(be) + 4j * p['t12'] * s(al) * (c(al) - c(be))
            h = np.array([[h0, h1, h2], [np.conj(h1), h11, h12], [np.conj(h2), np.conj(h12), h22]])
            self.assertTrue(np.allclose(ks.get_ham(k), h, atol=1e-12))
        K, Kp = valleys(ks)
        en = np.linalg.eigvalsh(ks.get_ham(K))
        self.assertLess(abs(en[1] - (p['eps1'] - 3 * p['t0'])), 1e-10)  # pure d_z2
        self.assertLess(abs(en[1] - en[0] - 1.663), 1e-3)
        self.assertLess(abs(np.min(np.diff(ks.mesh_bands(60), axis=1)[:, 0]) - (en[1] - en[0])), 1e-10)
        self.assertLess(abs(ks.chern_number([0], nk=30)), 1e-6)
        # valley Hall: the curvature peaks at K (fractions (1/3, -1/3), mesh
        # point (10, 20)) and K' (20, 10), with opposite signs. The plaquettes
        # (9, 19) / (19, 9) on the other side of each valley tie to rounding,
        # so compare values rather than argmin/argmax indices.
        curv = ks.berry_curvature([0], nk=30)
        self.assertLess(abs(curv[10, 20] - curv.min()), 1e-12)
        self.assertLess(abs(curv[20, 10] - curv.max()), 1e-12)
        self.assertLess(curv[10, 20], 0.)
        self.assertLess(abs(curv[10, 20] + curv[20, 10]), 1e-12)
        # spin-orbit: valence band split by 2 lam at K, spin-valley locked
        for q in models.TMD_PARAMETERS.values():
            spinful = models.tmd_three_band(**q, spin=True)
            sz = spinful.spin_operator('z')
            for k, sign in [(K * p['a'] / q['a'], -1), (Kp * p['a'] / q['a'], 1)]:
                e, v = np.linalg.eigh(spinful.get_ham(k))
                self.assertLess(abs(e[1] - e[0] - 2 * q['lam']), 1e-10)
                self.assertLess(abs(np.vdot(v[:, 1], sz @ v[:, 1]).real - sign), 1e-10)
        self.assertEqual(models.tmd_three_band(spin=True).z2_invariant([0, 1], nk=30, nk_perp=21), 0)

    def test_finite_system(self):
        # one definition for both pipelines
        hal = models.haldane(M=0.1)
        system = finite_system(hal, (6, 6), periodic=True)
        system.get_ham()
        self.assertTrue(np.allclose(np.linalg.eigvalsh(system.ham.toarray()), np.sort(hal.mesh_bands(6).ravel())))
        ham, pos, tags = finite_model(models.kane_mele(), (4, 4))
        self.assertEqual(ham.shape, (64, 64))


class TestThreeDimensions(unittest.TestCase):
    def test_topological_insulator_3d(self):
        phases = {-2.: (0, 0, 0, 0), 2.: (1, 0, 0, 0), 6.: (0, 1, 1, 1), 10.: (1, 1, 1, 1), 14.: (0, 0, 0, 0)}
        for M, indices in phases.items():
            ks = models.topological_insulator_3d(M)
            self.assertEqual(ks.parity_z2(INVERSION, [0, 1]), indices[0])
            self.assertEqual(ks.z2_indices_3d([0, 1], nk=30, nk_perp=21), indices)
        # B rescales the phase boundaries
        self.assertEqual(models.topological_insulator_3d(M=3., B=0.5, A=0.8).parity_z2(INVERSION, [0, 1]), 0)

    def test_weyl_semimetal(self):
        for k0 in (np.pi / 2, 1.):
            points = find_weyl_points(models.weyl_semimetal(k0), bands=0)
            order = np.argsort(points.k[:, 2])
            self.assertTrue(np.allclose(points.k[order], [[0., 0., -k0], [0., 0., k0]], atol=1e-8))
            self.assertEqual(points.chirality[order].tolist(), [-1, 1])
        ks = models.weyl_semimetal()
        self.assertLess(abs(ks.chern_number([0], nk=20, k_fixed=0.) + 1.), 1e-6)
        self.assertLess(abs(ks.chern_number([0], nk=20, k_fixed=0.4)), 1e-6)


class TestErrors(unittest.TestCase):
    def test_parameters(self):
        for fn, name in [(models.ssh, 'v'), (models.rice_mele, 'Delta'), (models.rice_mele_pump, 'offset'),
                         (models.kitaev_chain, 'mu'), (models.haldane, 'phi'), (models.kane_mele, 'lambda_r'),
                         (models.bhz, 'D'), (models.tmd_three_band, 't12'),
                         (models.topological_insulator_3d, 'B')]:
            self.assertRaises(TypeError, fn, **{name: 1j})
        self.assertRaises(TypeError, models.qi_wu_zhang, 'm')
        self.assertRaises(TypeError, models.weyl_semimetal, None)
        self.assertRaises(ValueError, models.tmd_three_band, a=-1.)
        self.assertRaises(TypeError, models.tmd_three_band, spin=1)

    def test_real_numbers(self):
        eh.real_numbers({'x': 1, 'y': 2.})
        self.assertRaises(TypeError, eh.real_numbers, {'x': 1, 'y': 'b'})


if __name__ == '__main__':
    unittest.main()
