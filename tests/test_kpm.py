"""
Kernel polynomial method: exact identities against diagonalization, and
stochastic traces; the Kubo-Bastin Hall conductivity on a torus; Chebyshev
time evolution, the spectral function A(k, E) and the optical conductivity.
"""
import unittest

import numpy as np
import scipy.linalg as LA
import scipy.sparse as sparse
from scipy.integrate import trapezoid

import tbkit.kpm as kpm
import tbkit.lattices as lattices
import tbkit.moire as moire
import tbkit.optics as optics
from tbkit.kspace import KSpace
from tbkit.propagation import Propagation
from tbkit.system import System
from tests.test_hall import anisotropic_haldane, haldane, torus_kubo


def disordered_square(n=8, w=1., seed=3):
    np.random.seed(seed)
    lat = lattices.square()
    lat.get_lattice(n, n)
    sys = System(lat)
    sys.set_hopping([{'n': 1, 't': 1.}])
    sys.set_onsite({'a': 0.})
    sys.set_onsite_dis(w)
    sys.get_ham()
    return sys


def smoothed_deltas(sys, n_moments, e_grid, kernel='jackson'):
    '''delta_K(E - E_a) for every eigenvalue, from the same truncated series.'''
    en, vec = LA.eigh(sys.ham.toarray())
    _, _, coef, a, b = kpm._setup(sys.ham, n_moments, kernel, None, e_grid)
    tm = np.cos(np.arange(n_moments)[:, None] * np.arccos((en - b) / a)[None, :])
    return coef.T @ tm, en, vec


class TestKernels(unittest.TestCase):

    def test_kernels(self):
        g = kpm.jackson_kernel(64)
        self.assertAlmostEqual(g[0], 1.)
        self.assertTrue(np.all(np.diff(g) < 0) and g[-1] > 0)
        gl = kpm.lorentz_kernel(64, lam=4.)
        self.assertAlmostEqual(gl[0], 1.)
        self.assertRaises(ValueError, kpm.jackson_kernel, 0)
        self.assertRaises(ValueError, kpm.lorentz_kernel, 8, -1.)

    def test_bounds(self):
        sys = disordered_square()
        lo, hi = kpm.spectral_bounds(sys.ham)
        en = LA.eigvalsh(sys.ham.toarray())
        self.assertLessEqual(lo, en.min())
        self.assertGreaterEqual(hi, en.max())


class TestDOS(unittest.TestCase):

    def test_exact_trace_identity(self):
        # with the exact trace, the KPM DOS is exactly the sum of the
        # kernel-broadened deltas of the eigenvalues
        sys = disordered_square()
        for kernel in ('jackson', 'lorentz'):
            e_grid, rho = kpm.dos(sys.ham, 64, None, kernel=kernel)
            deltas, _, _ = smoothed_deltas(sys, 64, e_grid, kernel)
            self.assertTrue(np.allclose(rho, deltas.sum(axis=1)))
        self.assertAlmostEqual(trapezoid(rho, e_grid) / sys.lat.sites, 1., places=2)

    def test_dos_from_levels(self):
        sys = disordered_square()
        e_grid, rho = kpm.dos(sys.ham, 64, None)
        levels = LA.eigvalsh(sys.ham.toarray())
        ref = kpm.dos_from_levels(levels, 64, kpm.spectral_bounds(sys.ham), e_grid)
        self.assertTrue(np.allclose(rho, ref))
        self.assertRaises(ValueError, kpm.dos_from_levels, [100.], 64, (-1., 1.), [0.])
        self.assertRaises(ValueError, kpm.dos_from_levels, [], 64, (-1., 1.), [0.])

    def test_stochastic_trace(self):
        sys = disordered_square(16)
        e_grid = np.linspace(-3., 3., 31)
        _, exact = kpm.dos(sys.ham, 64, None, e_grid=e_grid)
        _, est = kpm.dos(sys.ham, 64, 50, e_grid=e_grid, seed=0)
        self.assertLess(np.max(np.abs(est - exact)) / exact.max(), 0.05)
        _, est2 = kpm.dos(sys.ham, 64, 50, e_grid=e_grid, seed=0)
        self.assertTrue(np.array_equal(est, est2))  # seeded

    def test_chain_dos(self):
        # a long chain: rho(E)/N -> 1/(pi sqrt(4 - E^2))
        lat = lattices.chain()
        lat.get_lattice(20000)
        sys = System(lat)
        sys.set_hopping([{'n': 1, 't': 1.}])
        sys.get_ham()
        e_grid = np.array([-1., 0., 1.2])
        _, rho = kpm.dos(sys.ham, 512, 20, e_grid=e_grid, seed=1)
        self.assertTrue(np.allclose(rho / 20000, 1. / (np.pi * np.sqrt(4 - e_grid**2)), rtol=0.02))

    def test_ldos(self):
        sys = disordered_square()
        e_grid, rho = kpm.ldos(sys.ham, 5, 64)
        deltas, _, vec = smoothed_deltas(sys, 64, e_grid)
        self.assertTrue(np.allclose(rho, deltas @ np.abs(vec[5]) ** 2))
        self.assertRaises(ValueError, kpm.ldos, sys.ham, 64)
        self.assertRaises(TypeError, kpm.ldos, sys.ham, 1.)

    def test_checks(self):
        sys = disordered_square(3)
        self.assertRaises(ValueError, kpm.dos, sys.ham, 8, None, None, 'gauss')
        self.assertRaises(ValueError, kpm.dos, sys.ham, 8, None, [100.])
        self.assertRaises(ValueError, kpm.dos, sys.ham, 8, 0)
        sys.set_onsite_def({0: 1j})
        sys.get_ham()
        self.assertRaises(ValueError, kpm.dos, sys.ham)


class TestConductivity(unittest.TestCase):

    def test_exact_trace_identity(self):
        # Kubo-Greenwood with the exact trace equals
        # pi/N sum_ab |v_ab|^2 delta_K(E - E_a) delta_K(E - E_b)
        sys = disordered_square()
        x = sys.lat.coor['x']
        e_grid = np.linspace(-3., 3., 13)
        _, sigma = kpm.conductivity(sys.ham, x, 48, None, e_grid=e_grid)
        deltas, _, vec = smoothed_deltas(sys, 48, e_grid)
        ham = sys.ham.toarray()
        vel = 1j * (ham * x[None, :] - x[:, None] * ham)
        v_ab = vec.conj().T @ vel @ vec
        exact = np.pi / sys.lat.sites * np.einsum('ea,ab,eb->e', deltas, np.abs(v_ab)**2, deltas)
        self.assertTrue(np.allclose(sigma, exact))
        self.assertTrue(np.all(sigma > -1e-12))
        _, sigma_r = kpm.conductivity(sys.ham, x, 48, 20, e_grid=e_grid, seed=2, area=64.)
        self.assertLess(np.max(np.abs(sigma_r - sigma)) / sigma.max(), 0.2)
        self.assertRaises(ValueError, kpm.conductivity, sys.ham, x[:3])
        self.assertRaises(ValueError, kpm.conductivity, sys.ham, x, 8, None, None, 'jackson', None,
                                None, -1.)

    def test_disorder_lowers_conductivity(self):
        x = disordered_square(12, 0.5).lat.coor['x']
        e = [0.5]
        weak = kpm.conductivity(disordered_square(12, 0.5).ham, x, 64, None, e_grid=e)[1][0]
        strong = kpm.conductivity(disordered_square(12, 6.).ham, x, 64, None, e_grid=e)[1][0]
        self.assertGreater(weak, strong)



class TestKPMHall(unittest.TestCase):

    def test_matches_exact_diagonalization_on_a_torus(self):
        # n x n torus: the exact Kubo formula (eigenstates of finite_ham,
        # velocities of finite_velocity) is hall_conductivity on the n x n
        # k-mesh; KPM with the exact trace converges to it (up to the
        # kernel's energy resolution), at T = 0 and T > 0
        hal = haldane(M=0.3)
        n = 6
        ham = hal.finite_ham(n, periodic=True, sparse=True)
        vx, vy = hal.finite_velocity(n, periodic=True, sparse=True)
        area = n * n * abs(np.linalg.det(np.array(hal.lat.prim_vec)))
        e_f = np.array([-3., -1.2, -0.5, 0., 0.4, 1.2, 2.5])
        exact = torus_kubo(hal, n, e_f)
        self.assertTrue(np.allclose(exact, hal.hall_conductivity(e_f, nk=n), atol=1e-10))
        _, sigma = kpm.hall_conductivity(ham, vx, vy, 400, None, e_grid=e_f, area=area)
        self.assertLess(np.max(np.abs(sigma - exact)), 3e-3)
        _, sigma_t = kpm.hall_conductivity(ham, vx, vy, 400, None, e_grid=e_f, area=area, temperature=0.05)
        exact_t = hal.hall_conductivity(e_f, temperature=0.05, nk=n)
        self.assertLess(np.max(np.abs(sigma_t - exact_t)), 6e-3)
        # random-phase trace, dense velocities, default grid and area
        e_grid, sigma_r = kpm.hall_conductivity(ham, vx.toarray(), vy.toarray(), 128, 30, seed=1)
        self.assertEqual(len(e_grid), 1001)
        self.assertTrue(np.all(np.isfinite(sigma_r)))

    def test_hall_part_without_c3_symmetry(self):
        # Without a C3 axis, Bastin's sigma_xy has a large symmetric
        # Fermi-surface part in the bands (about -10 here, with the exact
        # trace); the antisymmetric part returned converges to the Berry
        # curvature Kubo formula, between the levels of the torus
        hal = anisotropic_haldane()
        n = 6
        ham = hal.finite_ham(n, periodic=True, sparse=True)
        vx, vy = hal.finite_velocity(n, periodic=True, sparse=True)
        area = n * n * abs(np.linalg.det(np.array(hal.lat.prim_vec)))
        en = LA.eigvalsh(ham.toarray())
        e_f = ((en[1:] + en[:-1]) / 2)[np.diff(en) > 0.15][::3]
        _, sigma = kpm.hall_conductivity(ham, vx, vy, 800, None, e_grid=e_f, area=area)
        self.assertTrue(np.allclose(sigma, torus_kubo(hal, n, e_f), atol=1e-3))
        self.assertGreater(np.max(np.abs(sigma)), 0.05)  # a nontrivial check

    def test_haldane_torus_plateau(self):
        # a Haldane torus of 20 x 20 cells: the Chern number in the gap,
        # both phases, from a few random vectors
        for M, chern in ((0.3, 1.), (1.5, 0.)):
            hal = haldane(M=M)
            n = 20
            ham = hal.finite_ham(n, periodic=True, sparse=True)
            vx, vy = hal.finite_velocity(n, periodic=True, sparse=True)
            area = n * n * abs(np.linalg.det(np.array(hal.lat.prim_vec)))
            _, sigma = kpm.hall_conductivity(ham, vx, vy, 256, 5, e_grid=[0.], area=area, seed=0)
            self.assertAlmostEqual(sigma[0], chern, delta=0.05)

    def test_finite_velocity(self):
        hal = haldane(M=0.3)
        ham = hal.finite_ham(3)
        vx, vy = hal.finite_velocity(3)
        # open sample: v = i[H, r] with the site coordinates
        tau = hal.orbital_positions()
        a = np.array(hal.lat.prim_vec)
        cells = np.array([(n1, n2) for n2 in range(3) for n1 in range(3)])
        r = np.repeat(cells @ a, 2, axis=0) + np.tile(tau, (9, 1))
        self.assertTrue(np.allclose(vx, 1j * (ham @ np.diag(r[:, 0]) - np.diag(r[:, 0]) @ ham)))
        self.assertTrue(np.allclose(vy, 1j * (ham @ np.diag(r[:, 1]) - np.diag(r[:, 1]) @ ham)))
        # sparse and dense agree, with and without periodic boundaries
        for periodic in (False, True):
            self.assertTrue(np.allclose(hal.finite_ham(4, periodic, sparse=True).toarray(),
                                                 hal.finite_ham(4, periodic)))
            for v_s, v_d in zip(hal.finite_velocity(4, periodic, True), hal.finite_velocity(4, periodic)):
                self.assertTrue(np.allclose(v_s.toarray(), v_d))
        self.assertRaises(TypeError, hal.finite_ham, 3, False, 1)
        self.assertRaises(TypeError, hal.finite_velocity, 3, False, 1)
        self.assertRaises(TypeError, hal.finite_velocity, 3, 1)
        self.assertRaises(ValueError, hal.finite_velocity, (3,))

    def test_checks(self):
        hal = haldane(M=0.3)
        ham = hal.finite_ham(3, periodic=True, sparse=True)
        vx, vy = hal.finite_velocity(3, periodic=True, sparse=True)
        self.assertRaises(ValueError, kpm.hall_conductivity, ham, vx[:4, :4], vy)
        self.assertRaises(ValueError, kpm.hall_conductivity, ham, vx, np.eye(2))
        self.assertRaises(ValueError, kpm.hall_conductivity, ham, 1j * vx, vy)
        self.assertRaises(ValueError, kpm.hall_conductivity, ham, vx, vy, 8, None, None, 'jackson', None,
                                None, -1.)
        self.assertRaises(ValueError, kpm.hall_conductivity, ham, vx, vy, 8, None, [0.], 'jackson', None,
                                None, None, -0.1)
        self.assertRaises(ValueError, kpm.hall_conductivity, ham, vx, vy, 8, None, [0.], 'lorentzian')
        # the Lorentz kernel works too
        _, sigma = kpm.hall_conductivity(ham, vx, vy, 64, None, e_grid=[0.], kernel='lorentz')
        self.assertTrue(np.isfinite(sigma[0]))


def graphene(t=1.):
    gra = KSpace(lattices.honeycomb())
    gra.set_hopping([{'i': 0, 'j': 1, 'R': R, 't': t} for R in [(0, 0), (-1, 0), (0, -1)]])
    return gra


def reciprocal(ks):
    a = np.array(ks.lat.prim_vec)[:, :2]
    return 2 * np.pi * np.linalg.inv(a).T


class TestTimeEvolution(unittest.TestCase):

    def test_matches_exponential(self):
        sys = disordered_square()
        psi0 = np.zeros(sys.lat.sites, 'c16')
        psi0[27] = 1.
        times = np.array([0., 0.3, 2., 7.5, 7.5, 20.])
        psi = kpm.time_evolution(sys.ham, psi0, times)
        ham = sys.ham.toarray()
        exact = np.array([LA.expm(-1j * ham * t) @ psi0 for t in times]).T
        self.assertTrue(np.allclose(psi, exact, atol=1e-12))
        self.assertTrue(np.allclose(np.linalg.norm(psi, axis=0), 1.))
        # backwards in time too
        back = kpm.time_evolution(sys.ham, psi[:, -1], [-20.])
        self.assertTrue(np.allclose(back[:, 0], psi0, atol=1e-12))

    def test_ballistic_spread(self):
        # a site state on a chain: psi_n(t) = (-i)^n J_n(2 t tau), <x^2> = 2 t^2 tau^2
        n, t = 401, 1.
        ham = sparse.diags([np.full(n - 1, t), np.full(n - 1, t)], [-1, 1], format='csr')
        psi0 = np.zeros(n, 'c16')
        psi0[n // 2] = 1.
        tau = np.linspace(0., 60., 7)
        psi = kpm.time_evolution(ham, psi0, tau)
        x = np.arange(n) - n // 2
        x2 = (np.abs(psi) ** 2 * x[:, None] ** 2).sum(axis=0)
        self.assertTrue(np.allclose(x2, 2 * t ** 2 * tau ** 2))

    def test_crank_nicolson(self):
        sys = disordered_square(6)
        psi0 = np.zeros(sys.lat.sites, 'c16')
        psi0[14] = 1.
        cheb, crank = Propagation(sys.lat), Propagation(sys.lat)
        cheb.get_propagation(sys.ham, psi0, steps=50, dz=0.01, method='chebyshev')
        crank.get_propagation(sys.ham, psi0, steps=50, dz=0.01)
        self.assertLess(np.abs(cheb.prop - crank.prop).max(), 1e-4)
        normed = Propagation(sys.lat)
        normed.get_propagation(sys.ham, psi0, steps=50, dz=0.01, norm=True, method='chebyshev')
        self.assertTrue(np.allclose(normed.prop, cheb.prop))
        self.assertRaises(ValueError, cheb.get_propagation, sys.ham, psi0, 5, 0.1, False, 'euler')

    def test_checks(self):
        sys = disordered_square(3)
        psi0 = np.ones(9, 'c16')
        self.assertRaises(ValueError, kpm.time_evolution, sys.ham, psi0[:4], [1.])
        self.assertRaises(TypeError, kpm.time_evolution, sys.ham, list(psi0), [1.])
        self.assertRaises(ValueError, kpm.time_evolution, sys.ham, psi0, [2., 1.])
        self.assertRaises(ValueError, kpm.time_evolution, sys.ham, psi0, [])
        self.assertRaises(ValueError, kpm.time_evolution, sys.ham, psi0, [np.inf])
        self.assertRaises(ValueError, kpm.time_evolution, sys.ham, psi0, [1.], None, 0.)
        sys.set_onsite_def({0: 1j})
        sys.get_ham()
        self.assertRaises(ValueError, kpm.time_evolution, sys.ham, psi0, [1.])


class TestSpectralFunction(unittest.TestCase):

    def test_matches_unfolding(self):
        # a disordered 6 x 6 supercell of graphene at Gamma is the 6 x 6
        # torus: on its k-mesh, A(k, E) is the unfolded spectrum broadened
        # by the same Chebyshev series
        gra = graphene()
        n = 6
        sc = moire.supercell(gra, [[n, 0], [0, n]])
        sc.onsite = np.random.default_rng(1).uniform(-0.5, 0.5, sc.norb)
        ham = sparse.csr_matrix(sc._bloch_ham(np.zeros((1, 2)))[0])
        pos = sc.cells.repeat(2, axis=0)[:, :2] + np.tile(gra.orbital_positions()[:, :2], (n * n, 1))
        k = np.array([[0., 0.], [1 / 3, 1 / 3], [0.5, 0.], [1 / 3, 1 / 6]]) @ reciprocal(gra)
        # an even and an odd number of moments (the moments come in pairs)
        for tags, m in ((np.tile([0, 1], n * n), 96), (np.tile(['a', 'b'], n * n), 95)):
            e_grid, spec = kpm.spectral_function(ham, pos, k, tags, m)
            en, weights = moire.unfold(sc, k)
            bounds = kpm.spectral_bounds(ham)
            exact = np.array([weights[i] @ np.array([kpm.dos_from_levels([e], m, bounds, e_grid)
                                                                for e in en[i]]) for i in range(len(k))])
            self.assertTrue(np.allclose(spec, exact, atol=1e-10))
        self.assertTrue(np.allclose(trapezoid(spec, e_grid, axis=1), 2., atol=1e-3))

    def test_chain(self):
        # one orbital per cell, 1D k: peaks at E = 2 cos k
        n = 60
        ham = sparse.diags([np.ones(n - 1), np.ones(n - 1), [1.], [1.]], [-1, 1, n - 1, 1 - n], format='csr')
        k = 2 * np.pi * np.array([5, 12]) / n
        e_grid, spec = kpm.spectral_function(ham, np.arange(n), k, n_moments=256)
        self.assertTrue(np.allclose(e_grid[np.argmax(spec, axis=1)], 2 * np.cos(k), atol=0.02))
        self.assertRaises(ValueError, kpm.spectral_function, ham, np.arange(n), np.zeros((2, 2)))
        self.assertRaises(ValueError, kpm.spectral_function, ham, np.arange(3), k)
        self.assertRaises(ValueError, kpm.spectral_function, ham, np.arange(n), k, [0, 1])


class TestOpticalConductivity(unittest.TestCase):

    def test_exact_trace_identity(self):
        # with the exact trace, the Chebyshev-broadened Kubo-Greenwood
        # formula from the eigenstates: pi/(Omega w) int [f(E) - f(E+w)]
        # sum_ab |v_ab|^2 delta_K(E - E_a) delta_K(E + w - E_b)
        gra = graphene()
        n, m = 6, 64
        ham = gra.finite_ham(n, periodic=True)
        ham = ham + np.diag(np.random.default_rng(0).uniform(-0.5, 0.5, len(ham)))
        vx, _ = gra.finite_velocity(n, periodic=True)
        en, vec = LA.eigh(ham)
        v2 = np.abs(vec.conj().T @ vx @ vec) ** 2
        lo, hi = kpm.spectral_bounds(ham)
        a, b = (hi - lo) / 1.98, (hi + lo) / 2

        def deltas(e):
            tm = np.cos(np.arange(m)[:, None] * np.arccos((en - b) / a)[None, :])
            return kpm._coefficients((e - b) / a, m, 'jackson', a).T @ tm

        def fermi(e, temp):
            return 1. / (np.exp(e / temp) + 1.)

        for temp in (0., 0.1):
            w = np.array([0.4, 1.3, 2.9])
            sigma = kpm.optical_conductivity(sparse.csr_matrix(ham), vx, w, temperature=temp,
                                                     n_moments=m, n_random=None, area=float(n * n))
            for wi, si in zip(w, sigma):
                if temp == 0:
                    e = np.linspace(max(b - 0.995 * a, -wi), min(0., b + 0.995 * a - wi), 20001)
                    occ = np.ones_like(e)
                else:
                    e = np.linspace(b - 0.995 * a, b + 0.995 * a - wi, 20001)
                    occ = fermi(e, temp) - fermi(e + wi, temp)
                joint = np.einsum('ea,ab,eb->e', deltas(e), v2, deltas(e + wi))
                exact = 2 * np.pi ** 2 / (n * n * wi) * trapezoid(occ * joint, e)
                self.assertLess(abs(si / exact - 1.), 1e-3)
        # no transition above the bandwidth
        self.assertEqual(kpm.optical_conductivity(sparse.csr_matrix(ham), vx, 20., n_moments=m,
                                                          n_random=None)[0], 0.)

    def test_universal_absorption(self):
        # clean graphene torus of 45,000 sites: pi/4 e^2/h for kernel width << w << t,
        # and the Brillouin-zone Kubo formula of tbkit.optics
        gra = graphene()
        n = 150
        ham = gra.finite_ham(n, periodic=True, sparse=True)
        vx, _ = gra.finite_velocity(n, periodic=True, sparse=True)
        w = np.array([0.3, 0.5, 1., 1.5, 2.5])
        sigma = kpm.optical_conductivity(ham, vx, w, n_moments=192, n_random=16, seed=0,
                                                 area=n * n * optics.cell_area(gra))
        self.assertLess(abs(sigma[:2].mean() / (np.pi / 4) - 1.), 0.03)
        bz = optics.optical_conductivity(gra, w, eta=0.04, nk=400).real
        self.assertTrue(np.allclose(sigma, bz, rtol=0.08))

    def test_hall_and_reactive_parts(self):
        # Haldane torus (C = 1) with the exact trace: below the gap the
        # Hall component Re sigma_yx -> C and the reactive Im sigma_xx < 0,
        # both the Brillouin-zone Kubo formula of tbkit.optics on the same
        # k-mesh (insensitive there to the broadening: Lorentzian or kernel)
        hal = haldane(M=0.3)
        n = 12
        ham = hal.finite_ham(n, periodic=True, sparse=True)
        vx, vy = hal.finite_velocity(n, periodic=True, sparse=True)
        area = n * n * optics.cell_area(hal)
        w = np.array([0.1, 0.4, 0.8])
        for temp in (0., 0.05):
            yx = kpm.optical_conductivity(ham, vy, w, temperature=temp, n_moments=256, n_random=None,
                                          area=area, vel_b=vx, part='complex')
            xx = kpm.optical_conductivity(ham, vx, w, temperature=temp, n_moments=256, n_random=None,
                                          area=area, part='imag')
            bz_yx = optics.optical_conductivity(hal, w, temperature=temp, eta=0.01, nk=n, component='yx')
            bz_xx = optics.optical_conductivity(hal, w, temperature=temp, eta=0.01, nk=n).imag
            self.assertTrue(np.allclose(yx.real, bz_yx.real, atol=0.01))
            self.assertTrue(np.allclose(np.abs(yx.imag), 0., atol=1e-3))
            self.assertTrue(np.allclose(xx, bz_xx, rtol=0.02, atol=0.))  # the kernel broadening, ~1%
            self.assertTrue(np.all(xx < 0))
        self.assertLess(abs(yx[0].real - hal.chern_number([0])), 0.01)
        # the real part alone, with or without the principal value
        self.assertTrue(np.allclose(
            kpm.optical_conductivity(ham, vy, w, temperature=temp, n_moments=256, n_random=None,
                                     area=area, vel_b=vx), yx.real))
        xx_c = kpm.optical_conductivity(ham, vx, [0.4, 30.], n_moments=64, n_random=None, part='complex')
        self.assertTrue(np.allclose(
            xx_c.real, kpm.optical_conductivity(ham, vx, [0.4, 30.], n_moments=64, n_random=None)))
        # above every transition: no absorption, an inductive (positive) Im
        self.assertEqual(xx_c[1].real, 0.)
        self.assertGreater(xx_c[1].imag, 0.)

    def test_checks(self):
        gra = graphene()
        ham = gra.finite_ham(3, periodic=True, sparse=True)
        vx, _ = gra.finite_velocity(3, periodic=True, sparse=True)
        self.assertRaises(ValueError, kpm.optical_conductivity, ham, vx, 1., vel_b=vx[:4, :4])
        self.assertRaises(ValueError, kpm.optical_conductivity, ham, vx, 1., vel_b=1j * vx)
        self.assertRaises(ValueError, kpm.optical_conductivity, ham, vx, 1., part='abs')
        self.assertRaises(ValueError, kpm.optical_conductivity, ham, vx[:4, :4], 1.)
        self.assertRaises(ValueError, kpm.optical_conductivity, ham, 1j * vx, 1.)
        self.assertRaises(ValueError, kpm.optical_conductivity, ham, vx, [0., 1.])
        self.assertRaises(TypeError, kpm.optical_conductivity, ham, vx, 'a')
        self.assertRaises(ValueError, kpm.optical_conductivity, ham, vx, 1., 10.)
        self.assertRaises(TypeError, kpm.optical_conductivity, ham, vx, 1., 'a')
        self.assertRaises(ValueError, kpm.optical_conductivity, ham, vx, 1., 0., -1.)
        self.assertRaises(ValueError, kpm.optical_conductivity, ham, vx, 1., 0., 0., 8, None,
                                'jackson', None, None, -1.)


if __name__ == '__main__':
    unittest.main()
