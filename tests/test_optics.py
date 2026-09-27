"""
Linear and nonlinear optical response of 2D Bloch models (tbkit.optics):
Kubo-Greenwood conductivity, joint density of states, Berry curvature
dipole and shift current.
"""
import unittest

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq

import tbkit.error_handling as error_handling
import tbkit.lattices as lattices
from tbkit import optics
from tbkit.floquet import FloquetKSpace
from tbkit.kspace import KSpace, PAULI
from tbkit.lattice import Lattice
from tests.test_hall import haldane


def graphene(t=-1., mass=0., spin=False):
    ks = KSpace(lattices.honeycomb(), spin=spin)
    ks.set_hopping([{'i': 0, 'j': 1, 'R': R, 't': t} for R in [(0, 0), (-1, 0), (0, -1)]])
    ks.set_onsite({'a': mass, 'b': -mass})
    return ks


def tilted_dirac(m, tilt, B, v=1.):
    r'''
    H = tilt sin kx + v (sin kx sx + sin ky sy) + (m + B(2 - cos kx - cos ky)) sz:
    the tilted massive Dirac cone t kx + v(kx sx + ky sy) + m sz at Gamma
    (Sodemann and Fu 2015), its doublers gapped by the Wilson mass B.
    '''
    lat = Lattice(unit_cell=[{'tag': 'a', 'r0': (0., 0.)}, {'tag': 'b', 'r0': (0., 0.)}],
                         prim_vec=[(1., 0.), (0., 1.)])
    ks = KSpace(lat)
    hop = []
    for R, s, tl in (((1, 0), 'x', tilt), ((0, 1), 'y', 0.)):
        block = -0.5j * v * PAULI[s] - 0.5 * B * PAULI['z'] - 0.5j * tl * PAULI['0']
        hop += [{'i': i, 'j': j, 'R': R, 't': complex(block[i, j])}
                     for i in range(2) for j in range(2) if block[i, j] != 0]
    ks.set_hopping(hop)
    ks.set_onsite({'a': m + 2 * B, 'b': -(m + 2 * B)})
    return ks


def breathing_kagome():
    '''Three bands, no inversion, orbitals off the cell origin.'''
    ks = KSpace(lattices.kagome())
    ta, tb = -1., -0.4
    ks.set_hopping([{'i': 0, 'j': 1, 'R': (0, 0), 't': ta}, {'i': 0, 'j': 2, 'R': (0, 0), 't': ta},
                             {'i': 1, 'j': 2, 'R': (0, 0), 't': ta}, {'i': 1, 'j': 0, 'R': (1, 0), 't': tb},
                             {'i': 2, 'j': 0, 'R': (0, 1), 't': tb}, {'i': 2, 'j': 1, 'R': (-1, 1), 't': tb}])
    ks.set_onsite({'a': 0.3, 'b': -0.2, 'c': 0.1})
    return ks


def f_sum(ks, e_fermi, temperature, nk, axis=0):
    r'''
    (pi/2) int d^2k/(2pi)^2 sum_n f_n <n|d^2H/dk_a^2|n>, in units of e^2/h,
    with the second derivative from finite differences of get_ham (periodic gauge).
    '''
    from tbkit.occupation import fermi_dirac
    _, kpts = ks.mesh_grid(nk)
    h = 1e-3
    e = np.eye(2)[axis] * h
    total = 0.
    for k in kpts:
        en, vec = np.linalg.eigh(ks.get_ham(k))
        d2 = (ks.get_ham(k + e) - 2 * ks.get_ham(k) + ks.get_ham(k - e)) / h ** 2
        f = fermi_dirac(en, e_fermi, temperature)
        total += np.sum(f * np.real(np.diag(vec.conj().T @ d2 @ vec)))
    return np.pi / 2 * total / (len(kpts) * optics.cell_area(ks)) * 2 * np.pi


class TestOpticalConductivity(unittest.TestCase):

    def test_graphene_universal_conductivity(self):
        # Re sigma_xx -> sigma_0/2 = e^2/(8 hbar) = pi/4 e^2/h per spin for
        # eta << hbar omega << t, with a correction quadratic in omega
        gra = graphene()
        omega = np.array([0.2, 0.3, 0.4])
        sigma = optics.optical_conductivity(gra, omega, e_fermi=0., eta=0.03, nk=400)
        c, a = np.polyfit(omega ** 2, sigma.real, 1)
        self.assertAlmostEqual(a / (np.pi / 4), 1., delta=5e-3)
        self.assertTrue(np.all(np.abs(sigma.real / (np.pi / 4) - 1) < 0.02))
        # isotropic: sigma_yy = sigma_xx; no Hall response with time reversal
        yy = optics.optical_conductivity(gra, omega, eta=0.03, nk=400, component='yy')
        self.assertTrue(np.allclose(yy, sigma, atol=1e-3))
        xy = optics.optical_conductivity(gra, omega, eta=0.03, nk=100, component='xy')
        self.assertTrue(np.allclose(xy, 0., atol=1e-10))
        # spinful: e^2/(4 hbar) = pi/2 e^2/h, twice the spinless value
        spin = optics.optical_conductivity(graphene(spin=True), 0.3, eta=0.05, nk=60)
        self.assertAlmostEqual(spin, 2 * optics.optical_conductivity(gra, 0.3, eta=0.05, nk=60), places=10)
        self.assertIsInstance(spin, complex)

    def test_pauli_blocking(self):
        # doped graphene absorbs only above hbar omega = 2 |E_F|
        gra = graphene()
        sigma = optics.optical_conductivity(gra, [0.2, 1.0], e_fermi=0.3, eta=0.02, nk=300)
        self.assertLess(sigma[0].real, 0.02)
        self.assertAlmostEqual(sigma[1].real / (np.pi / 4), 1.13, delta=0.02)

    def test_dc_hall_limit(self):
        # sigma_yx(omega -> 0) = hall_conductivity (TKNN sign); sigma_xy is its negative
        hal = haldane(M=0.3)
        for e_f, nk in ((0., 60), (-1.2, 120)):
            yx = optics.optical_conductivity(hal, 0., e_fermi=e_f, eta=1e-4, nk=nk, component='yx')
            xy = optics.optical_conductivity(hal, 0., e_fermi=e_f, eta=1e-4, nk=nk, component='xy')
            ref = hal.hall_conductivity(e_f, nk=nk)
            self.assertAlmostEqual((yx - xy).real / 2, ref, places=5)
            if e_f == 0.:
                self.assertAlmostEqual(yx.real, 1., places=5)
        # the periodic gauge also agrees in a gap
        per = optics.optical_conductivity(hal, [0.], eta=1e-4, nk=60, component='yx', positions=False)
        self.assertAlmostEqual(per[0].real, 1., places=4)

    def test_f_sum_rule(self):
        # int_{-inf}^{inf} Re sigma_xx d omega = pi <-d^2H> ... : interband only in an
        # insulator (T = 0), interband + Drude in a metal (T > 0)
        grid = np.concatenate([np.linspace(-400., -12., 4000), np.linspace(-12., 12., 24001)[1:-1],
                                         np.linspace(12., 400., 4000)])
        for ks, e_f, temp in ((haldane(M=0.3), 0., 0.), (graphene(), 0.8, 0.1)):
            sigma = optics.optical_conductivity(ks, grid, e_fermi=e_f, temperature=temp, eta=0.05,
                                                              nk=24, positions=False)
            weight = np.trapezoid(sigma.real, grid) / 2
            self.assertAlmostEqual(weight / f_sum(ks, e_f, temp, 24), 1., delta=2e-3)
        # the Drude peak holds the intraband part of the weight: absent at T = 0
        gra = graphene()
        cold = optics.optical_conductivity(gra, 0., e_fermi=0.8, eta=0.05, nk=24)
        hot = optics.optical_conductivity(gra, 0., e_fermi=0.8, temperature=0.1, eta=0.05, nk=24)
        self.assertLess(cold.real, 0.05)  # only the tails of the interband peaks
        self.assertGreater(hot.real, 1.)

    def test_errors(self):
        gra = graphene()
        self.assertRaises(TypeError, optics.optical_conductivity, gra, 'a')
        self.assertRaises(ValueError, optics.optical_conductivity, gra, [])
        self.assertRaises(ValueError, optics.optical_conductivity, gra, [np.inf])
        self.assertRaises(TypeError, optics.optical_conductivity, gra, 1., '0')
        self.assertRaises(ValueError, optics.optical_conductivity, gra, 1., 0., -1.)
        self.assertRaises(ValueError, optics.optical_conductivity, gra, 1., 0., 0., 0.)
        self.assertRaises(ValueError, optics.optical_conductivity, gra, 1., component='xz')
        self.assertRaises(ValueError, optics.optical_conductivity, gra, 1., component='xxx')
        self.assertRaises(TypeError, optics.optical_conductivity, gra, 1., component=0)
        self.assertRaises(TypeError, optics.optical_conductivity, gra, 1., positions=1)
        self.assertRaises(TypeError, optics.optical_conductivity, 'gra', 1.)
        chain = KSpace(lattices.chain())
        chain.set_hopping([{'i': 0, 'j': 0, 'R': (1,), 't': 1.}])
        self.assertRaises(ValueError, optics.optical_conductivity, chain, 1.)
        self.assertRaises(ValueError, optics.cell_area, chain)
        lossy = graphene()
        lossy.set_onsite({'a': 0.1j})
        self.assertRaises(ValueError, optics.joint_dos, lossy, 1.)
        driven = FloquetKSpace(gra, lambda t: (0.1 * np.cos(t), 0.1 * np.sin(t)), 2 * np.pi)
        self.assertRaises(ValueError, optics.optical_conductivity, driven, 1.)


class TestJointDos(unittest.TestCase):

    def test_graphene(self):
        gra = graphene()
        # J = A_c omega / (4 pi v^2) near the Dirac points, v = 3|t|a/2
        omega = np.array([0.1, 0.2, 0.3])
        jdos = optics.joint_dos(gra, omega, eta=0.02, nk=500)
        slope = optics.cell_area(gra) / (4 * np.pi * 1.5 ** 2)
        self.assertTrue(np.allclose(jdos / omega / slope, 1., atol=0.02))
        # one occupied and one empty band: unit total weight
        grid = np.linspace(-1., 7., 1601)
        self.assertAlmostEqual(np.trapezoid(optics.joint_dos(gra, grid, eta=0.05, nk=40), grid), 1., places=6)
        # a gapped model absorbs nothing below its gap 2 m
        self.assertLess(optics.joint_dos(graphene(mass=0.5), 0.5, eta=0.05, nk=60), 1e-12)
        # a filled band has nothing to absorb
        self.assertEqual(optics.joint_dos(gra, [1.], e_fermi=10., nk=10)[0], 0.)
        self.assertRaises(ValueError, optics.joint_dos, gra, 1., eta=0.)


def dipole_reference(mu, T, m, tilt):
    r'''
    D_x at temperature T: the T = 0 dipole -3 t m (mu^2 - m^2)/(8 pi mu^4)
    of the tilted massive Dirac cone (first order in the tilt, mu > |m|)
    averaged over the thermal window -f'.
    '''
    def d0(x):
        return -3 * tilt * m * (x ** 2 - m ** 2) / (8 * np.pi * x ** 4) if x > abs(m) else 0.
    def window(x):
        return 0.25 / T / np.cosh((x - mu) / (2 * T)) ** 2
    return quad(lambda x: window(x) * d0(x), mu - 40 * T, mu + 40 * T, points=[abs(m)], limit=200)[0]


class TestBerryCurvatureDipole(unittest.TestCase):

    def test_tilted_massive_dirac(self):
        # Sodemann and Fu (2015): the dipole of a tilted massive Dirac cone
        m, tilt, T = 0.07, 0.02, 0.01
        ks = tilted_dirac(m, tilt, B=0.12)
        mus = np.array([0.1, 0.12, 0.15])
        dip = optics.berry_curvature_dipole(ks, mus, T, nk=400)
        self.assertEqual(dip.shape, (3, 2))
        ref = [dipole_reference(mu, T, m, tilt) for mu in mus]
        self.assertTrue(np.allclose(dip[:, 0] / ref, 1., atol=0.015))
        # the mirror y -> -y forbids D_y
        self.assertTrue(np.allclose(dip[:, 1], 0., atol=1e-12))
        # odd in the tilt, and zero without it (the cone is then symmetric)
        flipped = optics.berry_curvature_dipole(tilted_dirac(m, -tilt, 0.12), 0.12, T, nk=400)
        self.assertAlmostEqual(flipped[0], -dip[1, 0], places=10)
        straight = optics.berry_curvature_dipole(tilted_dirac(m, 0., 0.12), 0.12, T, nk=100)
        self.assertTrue(np.allclose(straight, 0., atol=1e-12))
        # in the gap: no Fermi surface, no dipole
        self.assertTrue(np.allclose(optics.berry_curvature_dipole(ks, 0., 0.002, nk=100), 0., atol=1e-12))

    def test_errors(self):
        ks = tilted_dirac(0.1, 0.02, 0.1)
        self.assertRaises(ValueError, optics.berry_curvature_dipole, ks, 0.1, 0.)
        self.assertRaises(TypeError, optics.berry_curvature_dipole, ks, 'a', 0.1)


def rice_mele(t, delta, mass):
    r'''
    Rice-Mele chains (Fregoso, Morimoto and Moore, Phys. Rev. B 96, 075421
    (2017), Eq. (D2)): A (+mass) at 0 and B (-mass) at a/2, hoppings
    (t + delta)/2 and (t - delta)/2 alternating, stacked along y without
    coupling (a = 1, so a 1D quantity equals the 2D one per unit length).
    '''
    lat = Lattice(unit_cell=[{'tag': 'a', 'r0': (0., 0.)}, {'tag': 'b', 'r0': (0.5, 0.)}],
                  prim_vec=[(1., 0.), (0., 1.)])
    ks = KSpace(lat)
    ks.set_hopping([{'i': 0, 'j': 1, 'R': (0, 0), 't': (t + delta) / 2},
                    {'i': 1, 'j': 0, 'R': (1, 0), 't': (t - delta) / 2}])
    ks.set_onsite({'a': mass, 'b': -mass})
    return ks


class TestShiftCurrent(unittest.TestCase):

    def test_generalized_derivative_is_the_shift_vector(self):
        # Im[r^b_mn r^b_nm;a] = |r^b_nm|^2 (d_a phi_nm - A_nn + A_mm), from
        # finite differences of gauge-fixed link products
        ks = breathing_kagome()
        directions = [ks.k_basis[:, 0], ks.k_basis[:, 1]]
        k0 = np.array([[0.31, 0.17]])
        for positions in (True, False):
            self.check_shift_vector(ks, directions, k0, positions)

    def check_shift_vector(self, ks, directions, k0, positions):
        en, vel, w = next(optics._bands(ks, k0, positions, second=True))
        r, r_der = optics.generalized_derivative(en, vel, w)

        def position_matrix(k):
            ham, dham = ks._bloch_derivatives(np.array([k]), directions, positions)
            e, v = np.linalg.eigh(ham[0])
            om = e[:, None] - e[None, :]
            return np.array([-1j * (v.conj().T @ d[0] @ v) / np.where(om == 0, 1., om)
                                     * (om != 0) for d in dham]), v
        h = 1e-4
        for a in range(2):
            step = np.eye(2)[a] * h / 2
            r_minus, v_minus = position_matrix(k0[0] - step)
            r_plus, v_plus = position_matrix(k0[0] + step)
            link = np.diag(v_minus.conj().T @ v_plus)
            for b in range(2):
                x = np.conj(r_minus[b]) * r_plus[b] * link[:, None] / link[None, :]
                fd = np.abs(r[b, 0]) ** 2 * np.angle(x) / h
                self.assertTrue(np.allclose(np.imag(r[b, 0].T * r_der[b, a, 0]), fd, atol=1e-6))
        self.assertRaises(ValueError, optics.generalized_derivative, en, vel, w[0])

    def test_symmetries(self):
        omega = np.linspace(0.5, 4., 15)
        # inversion symmetry (gapless graphene): no shift current
        zero = optics.shift_current(graphene(), omega, eta=0.1, nk=40, component='yyy')
        self.assertTrue(np.allclose(zero, 0., atol=1e-12))
        # gapped graphene (C3v, mirror x -> -x): sigma^yyy = -sigma^yxx = -sigma^xxy,
        # sigma^xxx = 0, nothing below the gap 2m
        hbn = graphene(mass=0.4)
        yyy = optics.shift_current(hbn, omega, eta=0.1, nk=60, component='yyy')
        yxx = optics.shift_current(hbn, omega, eta=0.1, nk=60, component='yxx')
        xxy = optics.shift_current(hbn, omega, eta=0.1, nk=60, component='xxy')
        xxx = optics.shift_current(hbn, omega, eta=0.1, nk=60, component='xxx')
        self.assertGreater(np.abs(yyy).max(), 1e-3)
        self.assertTrue(np.allclose(yxx, -yyy, atol=1e-8))
        self.assertTrue(np.allclose(xxy, -yyy, atol=1e-8))
        self.assertTrue(np.allclose(xxx, 0., atol=1e-10))
        self.assertLess(abs(optics.shift_current(hbn, 0.3, eta=0.05, nk=40, component='yyy')), 1e-12)
        # m -> -m is the inversion image of the crystal: the current reverses
        flipped = optics.shift_current(graphene(mass=-0.4), omega, eta=0.1, nk=60, component='yyy')
        self.assertTrue(np.allclose(flipped, -yyy, atol=1e-8))
        # unlike a Hall conductance, the shift current depends on where the
        # orbitals sit in the cell: all at the origin, it changes
        per = optics.shift_current(hbn, omega, eta=0.1, nk=60, component='yyy', positions=False)
        self.assertGreater(np.abs(per - yyy).max(), 1e-2)

    def test_rice_mele(self):
        # sigma(omega) = pi/2 a^3 t delta Delta/(8 omega^3) sum_i 1/|dE/dk(k_i)|,
        # 2E(k_i) = omega: half the magnitude of Fregoso, Morimoto and Moore's
        # (D16), which integrates their (D12) as int dk where Sipe and
        # Shkrebtii's formula (their (A2)) has pi int dk / 2pi; the sign is
        # that of a charge -e
        t, dl, m = 1., 0.4, 0.3
        ks = rice_mele(t, dl, m)
        energy = lambda q: np.sqrt(t**2 * np.cos(q / 2)**2 + dl**2 * np.sin(q / 2)**2 + m**2)
        for omega in (1.2, 1.5, 1.9):
            k1 = brentq(lambda q: 2 * energy(q) - omega, 0., np.pi)
            slope = abs((dl**2 - t**2) * np.sin(k1) / (4 * energy(k1)))
            d16 = -t * dl * m / (8 * omega**3) * 2 / slope
            sigma = optics.shift_current(ks, omega, eta=0.002, nk=(20000, 1))
            self.assertAlmostEqual(sigma / d16, -0.5, delta=1e-3)

    def test_errors(self):
        hbn = graphene(mass=0.4)
        self.assertRaises(ValueError, optics.shift_current, hbn, 1., component='xx')
        hbn.set_overlap([{'i': 0, 'j': 1, 'R': (0, 0), 't': 0.1}])
        self.assertRaises(ValueError, optics.shift_current, hbn, 1.)


class TestOpticsChecks(unittest.TestCase):

    def test_validators(self):
        self.assertEqual(error_handling.tensor_component('yx', 2), (1, 0))
        self.assertRaises(TypeError, error_handling.frequencies, 'a', 'omega')
        self.assertRaises(ValueError, error_handling.frequencies, [np.nan], 'omega')
        self.assertRaises(ValueError, error_handling.band_derivatives,
                                np.zeros(3), np.zeros((2, 3, 3)), np.zeros((2, 2, 3, 3)))


class TestValidators(unittest.TestCase):

    def test_unique_names(self):
        # error_handling is extended by appending sections: a new check with
        # an existing name would silently replace the old one for every caller
        import ast
        import collections
        with open(error_handling.__file__) as source:
            tree = ast.parse(source.read())
        names = collections.Counter(n.name for n in tree.body if isinstance(n, ast.FunctionDef))
        self.assertEqual([k for k, v in names.items() if v > 1], [])
