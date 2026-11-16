"""
Exact diagonalization of small interacting clusters (tbkit.manybody), and
the twisted boundary conditions of KSpace.finite_ham.
"""
import unittest

import numpy as np
import scipy.sparse as sp

import tbkit.lattices as lattices
from tbkit.kspace import KSpace, reciprocal_vectors
from tbkit.manybody import (ExactResult, FockSector, _pair_coefficients, _popcount, exact_diagonalization,
                            project_interaction)


def jordan_wigner(n_modes):
    '''Annihilation operators of n_modes fermionic modes, dense.'''
    z, a = np.diag([1., -1.]), np.array([[0., 1.], [0., 0.]])
    ops = []
    for m in range(n_modes):
        out = np.ones((1, 1))
        for o in [z] * m + [a] + [np.eye(2)] * (n_modes - m - 1):
            out = np.kron(out, o)
        ops.append(out)
    return ops


def random_model(n, seed):
    rng = np.random.default_rng(seed)
    h = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    v = rng.uniform(0., 1., (n, n))
    v = v + v.T
    np.fill_diagonal(v, 0.)
    return h + h.conj().T, rng.uniform(0., 3., n), v


class TestAgainstJordanWigner(unittest.TestCase):
    '''Every quantity against the full Fock space of 2N modes.'''

    @classmethod
    def setUpClass(cls):
        n = cls.n = 4
        cls.h, cls.u, cls.v = random_model(n, 1)
        c = jordan_wigner(2 * n)
        num = [x.T @ x for x in c]
        cls.dens = [num[i] + num[n + i] for i in range(n)]
        ham = sum(cls.h[i, j] * (c[i].T @ c[j] + c[n + i].T @ c[n + j]) for i in range(n) for j in range(n))
        ham = ham + sum(cls.u[i] * num[i] @ num[n + i] for i in range(n))
        ham = ham + sum(cls.v[i, j] * cls.dens[i] @ cls.dens[j] for i in range(n) for j in range(i + 1, n))
        keep = np.isclose(np.diag(sum(num[:n])), 2) & np.isclose(np.diag(sum(num[n:])), 1)
        e, vec = np.linalg.eigh(ham[np.ix_(keep, keep)])
        cls.energies = e
        cls.ground = np.zeros(len(ham), complex)
        cls.ground[keep] = vec[:, 0]
        cls.num = num
        splus = [c[i].T @ c[n + i] for i in range(n)]
        sz = [(num[i] - num[n + i]) / 2 for i in range(n)]
        cls.ss = [[sz[i] @ sz[j] + (splus[i] @ splus[j].T + splus[i].T @ splus[j]) / 2 for j in range(n)]
                  for i in range(n)]
        spinless = sum(cls.h[i, j] * c[i].T @ c[j] for i in range(n) for j in range(n))
        spinless = spinless + sum(cls.v[i, j] * num[i] @ num[j] for i in range(n) for j in range(i + 1, n))
        keep = np.isclose(np.diag(sum(num[:n])), 2) & np.isclose(np.diag(sum(num[n:])), 0)
        cls.spinless = np.linalg.eigvalsh(spinless[np.ix_(keep, keep)])
        cls.res = exact_diagonalization(cls.h, 2, 1, U=cls.u, V=cls.v, n_states=5)

    def expect(self, op):
        return np.real(np.vdot(self.ground, op @ self.ground))

    def test_energies(self):
        self.assertTrue(np.allclose(self.res.energies, self.energies[:5]))
        self.assertIsInstance(self.res, ExactResult)
        res = exact_diagonalization(self.h, 2, V=self.v, n_states=3)
        self.assertTrue(np.allclose(res.energies, self.spinless[:3]))

    def test_observables(self):
        n = self.n
        n_up, n_dn = self.res.densities()
        self.assertTrue(np.allclose(n_up, [self.expect(self.num[i]) for i in range(n)]))
        self.assertTrue(np.allclose(n_dn, [self.expect(self.num[n + i]) for i in range(n)]))
        self.assertTrue(np.isclose(n_up.sum(), 2) and np.isclose(n_dn.sum(), 1))
        self.assertTrue(np.allclose(self.res.double_occupancy(),
                                    [self.expect(self.num[i] @ self.num[n + i]) for i in range(n)]))
        dd = [[self.expect(self.dens[i] @ self.dens[j]) for j in range(n)] for i in range(n)]
        self.assertTrue(np.allclose(self.res.density_correlation(), dd))
        ss = [[self.expect(self.ss[i][j]) for j in range(n)] for i in range(n)]
        corr = self.res.spin_correlation()
        self.assertTrue(np.allclose(corr, ss))
        s = self.res.total_spin()
        self.assertTrue(np.isclose(s * (s + 1), corr.sum()))

    def test_excited_state(self):
        self.assertTrue(np.isclose(np.sum(self.res.densities(3)[0]), 2.))


class TestClusters(unittest.TestCase):
    def test_dimer(self):
        dimer = [[0., -1.], [-1., 0.]]
        for U in (0., 1., 4., 20.):
            res = exact_diagonalization(dimer, 1, 1, U=U, n_states=4)
            self.assertTrue(abs(res.energies[0] - (U - np.sqrt(U ** 2 + 16)) / 2) < 1e-10)
            self.assertTrue(abs(res.total_spin()) < 1e-6)
            corr = res.spin_correlation()
            self.assertTrue(abs(corr.sum()) < 1e-10 and corr[0, 1] < 0)

    def test_free_ring_lanczos(self):
        ring = lattices.chain()
        ks = KSpace(ring)
        ks.set_hopping([{'n': 1, 't': -1.}])
        h = ks.finite_ham(8, periodic=True, twist=[np.pi], sparse=True)
        res = exact_diagonalization(h, 4, 4)       # 4900 states: eigsh
        self.assertGreater(res.sector.dim, 1500)
        levels = np.linalg.eigvalsh(h.toarray())
        self.assertTrue(abs(res.energies[0] - 2 * levels[:4].sum()) < 1e-9)

    def test_polarized_sector(self):
        h = -(np.eye(3, k=1) + np.eye(3, k=-1))
        res = exact_diagonalization(h, 2, 0, U=5.)
        self.assertTrue(abs(res.total_spin() - 1.) < 1e-10)
        self.assertTrue(abs(res.spin_correlation().sum() - 2.) < 1e-10)
        self.assertTrue(np.allclose(res.double_occupancy(), 0.))

    def test_sector(self):
        sec = FockSector(4, 2, 1)
        self.assertEqual(sec.dim, 24)
        self.assertEqual(sec.sz, 0.5)
        occ_up, occ_dn = sec.occupations()
        self.assertEqual(occ_up.shape, (6, 4))
        self.assertTrue(np.allclose(occ_dn.sum(axis=1), 1))
        spinless = FockSector(4, 0)
        self.assertTrue(spinless.spinless and spinless.dim == 1 and spinless.sz == 0.)
        self.assertEqual(FockSector(5, np.int64(2)).dim, 10)
        numpy_counts = FockSector(4, np.int64(2), np.int64(1))
        self.assertTrue(numpy_counts.dim == 24 and type(numpy_counts.n_dn) is int)
        mat = sec.hamiltonian(np.eye(4), U=1.)
        self.assertTrue(sp.issparse(mat))
        self.assertTrue(np.allclose(mat.diagonal(), 3. + np.ravel(occ_up @ occ_dn.T)))
        self.assertTrue(np.array_equal(_popcount(np.array([0, 7, 2 ** 40 + 1])), [0, 3, 2]))


class TestTwoBody(unittest.TestCase):
    def test_projection_onto_every_state(self):
        h, _, v = random_model(5, 2)
        e, phi = np.linalg.eigh(h)
        direct = exact_diagonalization(h, 2, V=v, n_states=4).energies
        projected = exact_diagonalization(np.diag(e), 2, W=project_interaction(phi, v), n_states=4).energies
        self.assertTrue(np.allclose(direct, projected))
        sites = exact_diagonalization(h, 2, W=project_interaction(np.eye(5), sp.csr_matrix(v)), n_states=4)
        self.assertTrue(np.allclose(direct, sites.energies))
        real = exact_diagonalization(np.diag(e), 2, W=project_interaction(np.eye(5), v), n_states=2)
        self.assertTrue(np.isrealobj(real.states))

    def test_empty_pairs(self):
        self.assertEqual(_pair_coefficients(np.zeros((1, 1, 1, 1))), [])
        w = project_interaction(np.eye(3), np.array([[0., 1., 0.], [1., 0., 0.], [0., 0., 0.]]))
        res = exact_diagonalization(np.zeros((3, 3)), 1, W=w, n_states=3)   # one fermion: no pair
        self.assertTrue(np.allclose(res.energies, 0.))
        _, phi = np.linalg.eigh(random_model(3, 4)[0])
        w = project_interaction(phi, np.array([[0., 1., 0.], [1., 0., 0.], [0., 0., 0.]]))
        res = exact_diagonalization(np.zeros((3, 3)), 3, W=w)               # full: no empty pair
        self.assertTrue(np.allclose(res.energies, 1.))

    def test_momentum_sectors(self):
        # spinless fermions on a ring of 6 sites with a neighbour repulsion, in the Bloch basis
        L = 6
        ks = KSpace(lattices.chain())
        ks.set_hopping([{'n': 1, 't': -1.}])
        h = ks.finite_ham(L, periodic=True)
        v = np.abs(h.real)
        full = exact_diagonalization(h, 3, V=v, n_states=20).energies
        m = np.arange(L)
        phi = np.exp(2j * np.pi * np.outer(np.arange(L), m) / L) / np.sqrt(L)
        e = -2 * np.cos(2 * np.pi * m / L)
        w = project_interaction(phi, v)
        sectors = np.concatenate([exact_diagonalization(np.diag(e), 3, W=w, n_states=FockSector(L, 3, momentum=(m, L, K)).dim,
                                                        momentum=(m, L, K)).energies for K in range(L)])
        self.assertEqual(len(sectors), 20)
        self.assertTrue(np.allclose(np.sort(sectors), full))
        sec = FockSector(L, 3, momentum=(m[:, None], (L,), (7,)))
        self.assertEqual(sec.momentum[2][0], 1)


class TestTwist(unittest.TestCase):
    def test_shifted_mesh(self):
        lat = lattices.honeycomb()
        ks = KSpace(lat)
        ks.set_hopping([{'n': 1, 't': -1.}])
        b = np.array(reciprocal_vectors(lat.prim_vec))
        theta, n = np.array([0.7, -1.3]), (3, 4)
        h = ks.finite_ham(n, periodic=True, twist=theta)
        self.assertTrue(np.allclose(h, ks.finite_ham(n, periodic=True, twist=theta, sparse=True).toarray()))
        self.assertTrue(np.allclose(h, h.conj().T))
        ks_ = [((m1 + theta[0] / (2 * np.pi)) / n[0]) * b[0] + ((m2 + theta[1] / (2 * np.pi)) / n[1]) * b[1]
               for m1 in range(n[0]) for m2 in range(n[1])]
        ref = np.sort(np.concatenate([np.linalg.eigvalsh(ks.get_ham(k)) for k in ks_]))
        self.assertTrue(np.allclose(ref, np.linalg.eigvalsh(h)))
        self.assertTrue(np.allclose(ks.finite_ham(n, periodic=True, twist=[2 * np.pi, 0.]),
                                    ks.finite_ham(n, periodic=True)))

    def test_checks(self):
        ks = KSpace(lattices.chain())
        ks.set_hopping([{'n': 1, 't': -1.}])
        self.assertRaises(ValueError, ks.finite_ham, 4, twist=[1.])
        self.assertRaises(ValueError, ks.finite_ham, 4, periodic=True, twist=[1., 2.])
        self.assertRaises(ValueError, ks.finite_ham, 4, periodic=True, twist=[np.inf])


class TestChecks(unittest.TestCase):
    def test_sector(self):
        self.assertRaises(ValueError, FockSector, 63, 1)
        self.assertRaises(TypeError, FockSector, 4., 1)
        self.assertRaises(TypeError, FockSector, 4, 1.)
        self.assertRaises(ValueError, FockSector, 4, 5)
        self.assertRaises(ValueError, FockSector, 4, 1, -1)

    def test_hamiltonian(self):
        h = np.zeros((3, 3))
        self.assertRaises(ValueError, exact_diagonalization, np.zeros((3, 2)), 1, 1)
        self.assertRaises(ValueError, exact_diagonalization, [[0., 1.], [0., 0.]], 1, 1)
        self.assertRaises(ValueError, exact_diagonalization, h, 1, U=1.)
        self.assertRaises(ValueError, exact_diagonalization, h, 1, 1, U=[1., 1.])
        self.assertRaises(ValueError, exact_diagonalization, h, 1, V=np.ones((3, 3)))
        self.assertRaises(ValueError, exact_diagonalization, h, 1, V=np.triu(np.ones((3, 3)), 1))
        self.assertRaises(ValueError, exact_diagonalization, h, 1, V=np.zeros((2, 2)))
        self.assertRaises(ValueError, exact_diagonalization, h, 1, V=1j * np.zeros((3, 3)))
        self.assertRaises(ValueError, exact_diagonalization, h, 1, n_states=4)
        self.assertRaises(TypeError, exact_diagonalization, h, 1, n_states=1.)
        self.assertRaises(ValueError, FockSector(3, 1).hamiltonian, np.zeros((4, 4)))

    def test_observables(self):
        res = exact_diagonalization(np.zeros((3, 3)), 1, n_states=2)
        self.assertRaises(ValueError, res.densities, 2)
        self.assertRaises(TypeError, res.densities, 0.)
        for f in (res.double_occupancy, res.spin_correlation, res.total_spin):
            self.assertRaises(ValueError, f)

    def test_two_body(self):
        h = np.zeros((3, 3))
        w = np.zeros((3, 3, 3, 3))
        self.assertRaises(ValueError, exact_diagonalization, h, 1, 1, W=w)
        self.assertRaises(ValueError, exact_diagonalization, h, 1, W=np.zeros((2, 2, 2, 2)))
        bad = w.copy()
        bad[0, 1, 1, 2] = 1.
        self.assertRaises(ValueError, exact_diagonalization, h, 1, W=bad)
        self.assertRaises(ValueError, project_interaction, np.ones((3, 2)), np.zeros((3, 3)))
        self.assertRaises(ValueError, project_interaction, np.ones(3), np.zeros((3, 3)))
        self.assertRaises(ValueError, project_interaction, np.eye(3), np.ones((3, 3)))

    def test_momentum(self):
        m = np.arange(3)
        self.assertRaises(ValueError, FockSector, 3, 1, 1, momentum=(m, 3, 0))
        self.assertRaises(ValueError, FockSector, 3, 1, momentum=(m, 3))
        self.assertRaises(ValueError, FockSector, 3, 1, momentum=[m, 3, 0])
        self.assertRaises(ValueError, FockSector, 3, 1, momentum=(m * 1., 3, 0))
        self.assertRaises(ValueError, FockSector, 3, 1, momentum=(m[:2], 3, 0))
        self.assertRaises(ValueError, FockSector, 3, 1, momentum=(m, 0, 0))
        self.assertRaises(ValueError, FockSector, 3, 1, momentum=(m, 3., 0))
        self.assertRaises(ValueError, FockSector, 3, 1, momentum=(m, (3, 3), 0))
        hop = -(np.eye(3, k=1) + np.eye(3, k=-1))
        self.assertRaises(ValueError, exact_diagonalization, hop, 1, momentum=(m, 3, 0))
        _, phi = np.linalg.eigh(random_model(3, 5)[0])
        w = project_interaction(phi, np.abs(hop))           # not momentum states
        self.assertRaises(ValueError, exact_diagonalization, np.zeros((3, 3)), 2, W=w, momentum=(m, 3, 0))


if __name__ == '__main__':
    unittest.main()
