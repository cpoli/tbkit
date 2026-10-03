"""
Berry-phase response beyond the Hall conductivity: the orbital
magnetization (KSpace.orbital_magnetization), the anomalous Nernst and
thermal Hall conductivities (KSpace.anomalous_nernst_conductivity,
KSpace.thermal_hall_conductivity) and the axion angle (KSpace.axion_angle).
"""
import unittest

import numpy as np
import scipy.linalg as LA
from scipy.integrate import trapezoid

import tbkit.lattices as lattices
from tbkit.bridges import finite_system
from tbkit.floquet import FloquetKSpace
from tbkit.kspace import KSpace, PAULI
from tbkit.lattice import Lattice
from tests.test_hall import anisotropic_haldane, haldane, haldane_stack
from tests.test_topology import kane_mele


def axion_model(m=3.):
    r'''
    Lattice Dirac model on the cubic lattice, pumped by phi:
    H = sum_i sin k_i G_i + (m + sum_i cos k_i + cos phi) G_0 + sin phi G_4,
    G_i = sz x s_i, G_0 = sx x 1, G_4 = sy x 1 (five anticommuting matrices).
    For 2 < m < 4 the cycle has C_2 = 1: trivial at phi = 0 (mass m + 1 > 3),
    a strong topological insulator at phi = pi (1 < m - 1 < 3).
    '''
    lat = Lattice(unit_cell=[{'tag': t, 'r0': (0., 0., 0.)} for t in 'abcd'],
                         prim_vec=[(1., 0., 0.), (0., 1., 0.), (0., 0., 1.)])
    ks = KSpace(lat)
    g0 = np.kron(PAULI['x'], PAULI['0'])
    hop = []
    for ax, s in enumerate('xyz'):
        block = np.kron(PAULI['z'], PAULI[s]) / 2j + g0 / 2
        hop += [{'i': i, 'j': j, 'R': tuple(int(a == ax) for a in range(3)), 't': complex(block[i, j])}
                     for i in range(4) for j in range(4) if block[i, j] != 0]
    ks.set_hopping(hop)
    ks.set_hopping([{'i': o, 'j': o + 2, 'R': (0, 0, 0), 't': lambda si, sj, phi: m + np.exp(-1j * phi)}
                             for o in (0, 1)])
    ks.set_params(phi=0.)
    return ks


def magnetization_by_differences(ks, k, mu, temperature, dk=1e-5):
    '''
    Integrand of the orbital magnetization at one k-point, straight from
    sum_n [f_n Im<d_x u_n|(H - E_n)|d_y u_n> + g_n Omega_n], with |d u_n>
    from finite differences of parallel-transported eigenvectors (bands
    non-degenerate), and the velocities' gauge of positions=True.
    '''
    def states(kk):
        ham = ks._bloch_derivatives(np.array([kk]), [], True)[0][0]
        return LA.eigh(ham)
    en, vec = states(k)
    ham = vec @ np.diag(en) @ vec.conj().T
    d = []
    for a in range(2):
        step = np.eye(2)[a] * dk
        pair = []
        for kk in (k + step, k - step):
            v = states(kk)[1]
            v = v * np.exp(-1j * np.angle(np.einsum('in,in->n', vec.conj(), v)))[None, :]
            pair.append(v)
        d.append((pair[0] - pair[1]) / (2 * dk))
    x = (en - mu) / temperature if temperature else None
    f = (en < mu).astype(float) if not temperature else 1. / (np.exp(x) + 1.)
    g = np.maximum(mu - en, 0.) if not temperature else temperature * np.log1p(np.exp(-x))
    total = 0.
    for n in range(len(en)):
        dx, dy = d[0][:, n], d[1][:, n]
        m_n = np.imag(dx.conj() @ (ham - en[n] * np.eye(len(en))) @ dy)
        omega = -2 * np.imag(dx.conj() @ dy)
        total += f[n] * m_n + g[n] * omega
    return total


class TestOrbitalMagnetization(unittest.TestCase):

    def test_streda_slope_in_the_gap(self):
        # in the gap, M is linear in mu with slope C (both signs and 0)
        for M_s, phi in ((0.3, np.pi / 2), (0.3, -np.pi / 2), (1.5, np.pi / 2)):
            hal = haldane(M=M_s, phi=phi)
            en = hal.mesh_bands(60)
            lo, hi = en[:, 0].max(), en[:, 1].min()
            mu = np.linspace(lo + 0.05, hi - 0.05, 5)
            mag = hal.orbital_magnetization(mu, nk=60)
            slope = np.diff(mag) / np.diff(mu)
            self.assertTrue(np.allclose(slope, hal.hall_conductivity(mu[0], nk=60), atol=1e-8))
        # empty and full bands carry no magnetization
        self.assertEqual(hal.orbital_magnetization(-5.), 0.)
        self.assertAlmostEqual(hal.orbital_magnetization(5., nk=30), 0., places=12)

    def test_streda_formula_on_a_flake(self):
        # dM/dmu (k-space) = dn/dB at fixed mu (real space): the electron
        # density deep inside a Haldane flake grows by C per flux quantum
        hal = haldane(M=0.3)
        slope = (hal.orbital_magnetization(0.05, nk=60) - hal.orbital_magnetization(-0.05, nk=60)) / 0.1
        dens = []
        for alpha in (0., 0.004):
            sys = finite_system(hal, (20, 20))
            if alpha:
                sys.set_magnetic_field(alpha)
            sys.get_ham()
            sys.get_eig(eigenvec=True)
            x, y = sys.lat.coor['x'], sys.lat.coor['y']
            inside = np.hypot(x - x.mean(), y - y.mean()) < 6.
            dens.append(sys.get_charge_density(0.)[inside].sum())
        cell = abs(np.linalg.det(np.array(hal.lat.prim_vec)))
        dn_dalpha = (dens[1] - dens[0]) / (inside.sum() / 2 * cell) / 0.004
        self.assertAlmostEqual(slope, 1., places=6)
        self.assertAlmostEqual(dn_dalpha, slope, places=3)

    def test_integrand_against_finite_differences(self):
        # the pair form equals the textbook formula at T = 0 and T > 0, at
        # Fermi levels in the gap and in a band, with the orbital positions
        hal = anisotropic_haldane(M=0.4)
        k = np.array([0.37, -0.81])
        for mu, temperature in ((0., 0.), (-1.1, 0.), (-1.1, 0.2), (0.7, 0.05)):
            pair = hal._kubo(k[None], np.array([mu]), temperature, list(np.eye(2)), [(0, 1)], True, None,
                                    'magnetization')[0][0, 0, 0]
            self.assertAlmostEqual(pair, magnetization_by_differences(hal, k, mu, temperature), places=6)

    def test_temperature_and_metal(self):
        hal = anisotropic_haldane(M=0.4)
        cold = hal.orbital_magnetization(-1.2, nk=40)
        self.assertAlmostEqual(hal.orbital_magnetization(-1.2, 1e-5, nk=40), cold, places=6)
        # in a metal, dM/dmu is not the Hall conductivity: the self-rotation
        # of the states at the Fermi level adds sum (-f') m_n
        h = 1e-4
        slope = (hal.orbital_magnetization(-1.2 + h, 0.05, nk=40)
                    - hal.orbital_magnetization(-1.2 - h, 0.05, nk=40)) / (2 * h)
        self.assertGreater(abs(slope - hal.hall_conductivity(-1.2, 0.05, nk=40)), 0.1)
        # time-reversal symmetric: no magnetization
        self.assertLess(abs(haldane(t2=0., M=0.3).orbital_magnetization(-1., 0.1, nk=20)), 1e-12)

    def test_three_d(self):
        # a stack of Chern layers c apart: the magnetization per length is
        # that of one plane divided by c, along z
        c = 2.
        ks = haldane_stack(tz=0., c=c)
        plane = ks.orbital_magnetization(0.2, nk=30, k_fixed=0.)
        self.assertAlmostEqual(plane, 0.2, places=8)
        mag = ks.orbital_magnetization(0.2, nk=(30, 30, 2))
        self.assertTrue(np.allclose(mag, [0., 0., plane / c], atol=1e-8))


class TestThermalResponse(unittest.TestCase):

    def mott(self, hal, mu, temperature, power):
        '''The Mott-type integral of the T = 0 Hall conductivity.'''
        e = np.linspace(-4., 4., 2001)
        sigma = hal.hall_conductivity(e, nk=60)
        x = np.abs(e[:, None] - mu) / temperature
        window = np.exp(-x) / (1 + np.exp(-x)) ** 2 / temperature
        return trapezoid((e[:, None] - mu) ** power * window * sigma[:, None], e, axis=0) / temperature

    def test_mott_relation(self):
        hal = anisotropic_haldane(M=0.2)
        mu, temperature = np.array([-1.5, -0.4, 0.9]), 0.08
        alpha = hal.anomalous_nernst_conductivity(mu, temperature, nk=60)
        self.assertTrue(np.allclose(alpha, self.mott(hal, mu, temperature, 1), atol=2e-4))
        self.assertGreater(np.min(np.abs(alpha)), 0.01)

    def test_wiedemann_franz(self):
        hal = haldane(M=0.3)
        mu, temperature = np.array([-1.5, 0., 0.9]), 0.08
        kappa = hal.thermal_hall_conductivity(mu, temperature, nk=60)
        self.assertTrue(np.allclose(kappa, self.mott(hal, mu, temperature, 2), atol=2e-4))
        # in the gap at low T: kappa / T = (pi^2 / 3) C, quantized; alpha = 0
        temperature = 0.01
        self.assertAlmostEqual(hal.thermal_hall_conductivity(0., temperature, nk=60) / temperature,
                                       np.pi ** 2 / 3, places=8)
        self.assertAlmostEqual(hal.anomalous_nernst_conductivity(0., temperature, nk=60), 0., places=10)

    def test_time_reversal_and_errors(self):
        km = kane_mele(lam=0.06, M=0.1, rashba=0.05)
        self.assertLess(abs(km.anomalous_nernst_conductivity(-1., 0.1, nk=20)), 1e-12)
        self.assertLess(abs(km.thermal_hall_conductivity(-1., 0.1, nk=20)), 1e-12)
        hal = haldane(M=0.3)
        for method in (hal.anomalous_nernst_conductivity, hal.thermal_hall_conductivity):
            self.assertRaises(ValueError, method, 0., 0.)
            self.assertRaises(TypeError, method, 0., 'a')
        self.assertRaises(ValueError, hal.orbital_magnetization, 0., -1.)
        driven = FloquetKSpace(hal, lambda t: (0.1 * np.cos(t), 0.1 * np.sin(t)), 2 * np.pi)
        self.assertRaises(ValueError, driven.orbital_magnetization, 0.)


class TestAxionAngle(unittest.TestCase):

    def test_pumping_cycle(self):
        # theta = 0 at the trivial point, pi at the topological one, and a
        # full cycle pumps 2 pi C_2 = 2 pi; the integrand is smooth, so the
        # k-mesh converges exponentially
        ks = axion_model(3.)
        phi = np.linspace(0., 2 * np.pi, 41)
        theta = ks.axion_angle([0, 1], 'phi', phi, nk=12)
        self.assertEqual(theta[0], 0.)
        self.assertAlmostEqual(theta[20] / np.pi, 1., delta=0.01)
        self.assertAlmostEqual(theta[-1] / np.pi, 2., delta=0.01)
        errors = [abs(ks.axion_angle([0, 1], 'phi', np.linspace(0., np.pi, 21), nk=nk)[-1] - np.pi)
                      for nk in (8, 12, 16)]
        self.assertTrue(errors[0] > 10 * errors[1] > 100 * errors[2])
        self.assertLess(errors[2], 1e-3)
        # the end points agree with the Z2 indices of the model
        ks.set_params(phi=np.pi)
        self.assertEqual(ks.z2_indices_3d([0, 1], nk=10)[0], 1)
        ks.set_params(phi=0.)
        self.assertEqual(ks.z2_indices_3d([0, 1], nk=10)[0], 0)
        # theta0, a decreasing path, the periodic gauge (all orbitals at
        # the origin: the same numbers)
        forward = ks.axion_angle([0, 1], 'phi', phi, nk=8)
        back = ks.axion_angle([0, 1], 'phi', phi[::-1], nk=8, theta0=forward[-1], positions=False)
        self.assertTrue(np.allclose(back[::-1], forward, atol=1e-10))

    def test_trivial_cycle(self):
        # deep in the trivial phase, the cycle pumps nothing, and theta
        # comes back to 0 at the time-reversal symmetric phi = pi
        ks = axion_model(6.)
        theta = ks.axion_angle([0, 1], 'phi', np.linspace(0., 2 * np.pi, 21), nk=8)
        self.assertLess(abs(theta[10]), 1e-3)
        self.assertLess(abs(theta[-1]), 1e-3)

    def test_errors(self):
        ks = axion_model(3.)
        self.assertRaises(ValueError, haldane().axion_angle, [0], 'phi', [0., 1.])
        self.assertRaises(TypeError, ks.axion_angle, [0, 1], 1, [0., 1.])
        self.assertRaises(ValueError, ks.axion_angle, [0, 1], 'phi', [0.])
        self.assertRaises(ValueError, ks.axion_angle, [0, 1], 'phi', [0., 1., 0.5])
        self.assertRaises(TypeError, ks.axion_angle, [0, 1], 'phi', [[0., 1.]])
        self.assertRaises(TypeError, ks.axion_angle, 0, 'phi', [0., 1.])
        self.assertRaises(TypeError, ks.axion_angle, [0, 1], 'phi', [0., 1.], theta0='a')
        self.assertRaises(ValueError, ks.axion_angle, [0, 1], 'phi', [0., 1.], nk=(4, 4))
        # m = 2: the gap closes at phi = pi (mass 1)
        self.assertRaises(ValueError, axion_model(2.).axion_angle, [0, 1], 'phi', [3., np.pi], nk=4)


if __name__ == '__main__':
    unittest.main()
