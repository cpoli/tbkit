"""
Wannier functions (tbkit.wannier): projection onto trial orbitals and
Marzari-Vanderbilt maximal localization, checked against the Wilson-loop
centres, the band projector of a torus, and real-space expectation values;
Souza-Marzari-Vanderbilt disentanglement of entangled bands, and the
Wannier-interpolated KSpace.
"""
import unittest

import numpy as np

import tbkit.lattices as lattices
from tbkit.kspace import KSpace
from tbkit.lattice import Lattice
from tbkit.slater_koster import sk_kspace
from tbkit.system import System
from tbkit.wannier import WannierFunctions, wannierize, _shells
from tests.test_topology import ssh, haldane, kane_mele


def translates_projector(ks, wf):
    '''
    Sum of |W><W| over every Wannier function and all its lattice
    translates on the torus of wf.n_cells cells.
    '''
    nk, norb = wf.n_cells, ks.norb
    axes = tuple(range(len(nk)))
    proj = 0.
    for f in wf.functions:
        amp = f.reshape(tuple(nk)[::-1] + (norb,))
        for shift in np.ndindex(*tuple(nk)[::-1]):
            t = np.roll(amp, shift, axis=axes).ravel()
            proj = proj + np.outer(t, t.conj())
    return proj


def band_projector(ks, n_bands, nk):
    '''
    Projector on the n_bands lowest bands of the torus of nk cells.
    '''
    _, vec = np.linalg.eigh(ks.finite_ham(nk, periodic=True))
    occ = vec[:, :n_bands * int(np.prod(nk))]
    return occ @ occ.conj().T


def entangled_chain():
    '''
    Two narrow bands (orbitals a, b) crossed by a wide one (orbital c),
    hybridized: no band index follows the a-b bands.
    '''
    lat = Lattice(unit_cell=[{'tag': 'a', 'r0': (0., 0.)}, {'tag': 'b', 'r0': (0.5, 0.)},
                                     {'tag': 'c', 'r0': (0.25, 0.)}], prim_vec=[(1., 0.)])
    ks = KSpace(lat)
    ks.set_onsite({'a': -1., 'b': 1., 'c': 0.5})
    ks.set_hopping([{'i': 0, 'j': 1, 'R': (0,), 't': 0.6}, {'i': 0, 'j': 1, 'R': (-1,), 't': 0.4},
                          {'i': 2, 'j': 2, 'R': (1,), 't': 1.2}, {'i': 0, 'j': 2, 'R': (0,), 't': 0.3},
                          {'i': 1, 'j': 2, 'R': (0,), 't': 0.2}])
    return ks


def frozen_error(e, e_wann, frozen):
    '''
    Largest distance from an exact energy inside *frozen* to the nearest
    Wannier band, over the k-points (rows).
    '''
    inside = (e >= frozen[0]) & (e <= frozen[1])
    return max(np.abs(e_wann[k][:, None] - e[k][inside[k]][None]).min(axis=0).max()
                     for k in range(len(e)) if inside[k].any())


class TestShells(unittest.TestCase):

    def test_finite_difference_condition(self):
        # square, hexagonal (one shell of 6), oblique and anisotropic meshes, 1D and 3D
        cases = [(np.eye(2), (6, 6), 4), (np.array(KSpace(lattices.honeycomb()).rec_vec_k), (6, 6), 6),
                     (np.array([[1., 0.], [0.3, 1.]]), (5, 9), None), (np.eye(1), (7,), 2),
                     (np.eye(3), (4, 5, 6), 6), (np.eye(3), (6, 6, 3), 6)]  # skips the (1, 1, 0) shell
        for rec, nk, n_b in cases:
            _, b, w = _shells(rec, nk)
            self.assertTrue(np.allclose(np.einsum('b,ba,bc->ac', w, b, b), np.eye(len(nk))))
            if n_b is not None:
                self.assertEqual(len(b), n_b)


class TestWannierize(unittest.TestCase):

    def test_ssh_centre_and_spread(self):
        # one band in 1D: the minimum has Omega = Omega_I, and the centre is
        # the Wilson-loop (Berry phase) centre, on the bond between the cells
        ks = ssh(0.5, 1.)
        wf = wannierize(ks, 0, [0], nk=20)
        self.assertIsInstance(wf, WannierFunctions)
        self.assertTrue(wf.converged)
        self.assertLess(wf.omega_d, 1e-8)
        self.assertLess(wf.omega_od, 1e-12)
        self.assertAlmostEqual(wf.omega, wf.omega_i, places=7)
        self.assertTrue(np.all(np.diff(wf.history) <= 0.))
        self.assertGreater(wf.history[0] - wf.history[-1], 0.03)
        centre = ks.wannier_centers(0, nk=20)[0]
        self.assertAlmostEqual(wf.centers[0, 0] % 1., centre, places=10)
        self.assertEqual(wf.centers[0, 0] // 1., 10.)  # moved to the middle cell
        # amplitudes: normalized, centred where the spread functional says
        p = np.abs(wf.functions[0]) ** 2
        self.assertAlmostEqual(p.sum(), 1., places=12)
        self.assertAlmostEqual(p @ wf.positions[:, 0], wf.centers[0, 0], delta=1e-6)

    def test_completeness_and_real_space(self):
        # the translates of the Wannier functions span the band: they
        # rebuild the band projector of the torus exactly
        ks = haldane(0.2, 2.)  # trivial
        wf = wannierize(ks, 0, [1], nk=6)
        self.assertTrue(np.allclose(translates_projector(ks, wf), band_projector(ks, 1, (6, 6)), atol=1e-12))
        self.assertTrue(np.allclose(np.einsum('kmn,kpn->kmp', wf.gauge, wf.gauge.conj()), np.eye(1)))
        wf = wannierize(ks, 0, [1], nk=24)
        self.assertAlmostEqual(wf.omega, wf.omega_i + wf.omega_d + wf.omega_od, places=10)
        p = np.abs(wf.functions[0]) ** 2
        r = p @ wf.positions
        self.assertTrue(np.allclose(r, wf.centers[0], atol=1e-8))
        self.assertAlmostEqual(p @ np.sum((wf.positions - r) ** 2, axis=1), wf.spreads[0], delta=2e-3)
        # on the System of the sample
        sys = wf.system
        self.assertIsInstance(sys, System)
        self.assertIs(wf.system, sys)
        coords = np.array([sys.lat.coor['x'], sys.lat.coor['y']]).T
        self.assertTrue(np.allclose(coords, wf.positions))

    def test_chern_obstruction(self):
        # trivial phase: the spread converges with the mesh; Chern phase: it
        # keeps growing, and the projection gets close to singular
        triv = [wannierize(haldane(0.2, 2.), 0, [1], nk=nk).omega for nk in (8, 16)]
        chern = [wannierize(haldane(0.2, 0.), 0, [1], nk=nk) for nk in (8, 16)]
        self.assertLess(triv[1] - triv[0], 0.01)
        self.assertGreater(chern[1].omega - chern[0].omega, 0.25)
        self.assertLess(chern[1].min_singular_value, chern[0].min_singular_value / 1.5)
        # a mesh through K, where the lower band lies on sublattice a only
        self.assertRaises(ValueError, wannierize, haldane(0.2, 0.), 0, [1], 12)

    def test_spinful_multiband(self):
        km = kane_mele(0.06, M=1.)  # trivial Z2
        wf = wannierize(km, [0, 1], [2, 3], nk=8)
        self.assertTrue(np.allclose(translates_projector(km, wf), band_projector(km, 2, (8, 8)), atol=1e-12))
        self.assertAlmostEqual(wf.spreads[0], wf.spreads[1], places=8)  # Kramers partners
        self.assertRaises(ValueError, lambda: wf.system)

    def test_trial_array_and_iterations(self):
        gra = KSpace(lattices.honeycomb())
        gra.set_hopping([{'i': 0, 'j': 1, 'R': R, 't': -1.} for R in [(0, 0), (-1, 0), (0, -1)]])
        # both bands together are the atomic orbitals: zero spread
        wf = wannierize(gra, [0, 1], np.eye(2), nk=6)
        self.assertLess(wf.omega, 1e-12)
        self.assertTrue(wf.converged)
        # max_iter=0 keeps the projection, max_iter=1 stops before convergence
        ks = ssh(0.5, 1.)
        wf0 = wannierize(ks, 0, [0], nk=12, max_iter=0)
        self.assertEqual(len(wf0.history), 1)
        self.assertTrue(wf0.converged)
        wf1 = wannierize(ks, 0, [0], nk=12, max_iter=1)
        self.assertFalse(wf1.converged)
        self.assertLess(wf1.omega, wf0.omega)

    def test_three_dimensions(self):
        lat = Lattice(unit_cell=[{'tag': 'a', 'r0': (0., 0., 0.)}, {'tag': 'b', 'r0': (0.5, 0.5, 0.5)}],
                           prim_vec=[(1., 0., 0.), (0., 1., 0.), (0., 0., 1.)])
        ks = KSpace(lat)
        ks.set_onsite({'a': -2., 'b': 2.})
        ks.set_hopping([{'i': 0, 'j': 1, 'R': R, 't': 0.5}
                              for R in [(0, 0, 0), (-1, 0, 0), (0, -1, 0), (0, 0, -1),
                                        (-1, -1, 0), (-1, 0, -1), (0, -1, -1), (-1, -1, -1)]])
        wf = wannierize(ks, 0, [0], nk=(4, 4, 5))
        self.assertTrue(np.allclose(translates_projector(ks, wf), band_projector(ks, 1, (4, 4, 5)), atol=1e-12))
        self.assertTrue(np.allclose(wf.centers[0], [2., 2., 2.], atol=1e-10))  # on site a, by symmetry

    def test_planar_lattice_in_space(self):
        # a 2D lattice in 3D: the centre out of the plane is <z>
        lat = Lattice(unit_cell=[{'tag': 'a', 'r0': (0., 0., 0.)}, {'tag': 'b', 'r0': (0.5, 0., 0.3)}],
                           prim_vec=[(1., 0., 0.), (0.3, 1., 0.)])
        ks = KSpace(lat)
        ks.set_onsite({'a': -1., 'b': 1.})
        ks.set_hopping([{'i': 0, 'j': 1, 'R': (0, 0), 't': 0.4}, {'i': 0, 'j': 1, 'R': (-1, 0), 't': 0.3},
                              {'i': 0, 'j': 0, 'R': (0, 1), 't': 0.2}])
        wf = wannierize(ks, 0, [0], nk=(5, 7))
        p = np.abs(wf.functions[0]) ** 2
        self.assertAlmostEqual(wf.centers[0, 2], p @ wf.positions[:, 2], places=12)
        self.assertGreater(wf.centers[0, 2], 0.)
        self.assertTrue(np.allclose(translates_projector(ks, wf), band_projector(ks, 1, (5, 7)), atol=1e-12))

    def test_errors(self):
        ks = ssh(0.5, 1.)
        self.assertRaises(TypeError, wannierize, 'ks', 0, [0])
        self.assertRaises(ValueError, wannierize, ks, 0, [0], nk=2)
        self.assertRaises(ValueError, wannierize, ks, 0, [0, 1])
        self.assertRaises(ValueError, wannierize, ks, 0, np.ones((2, 2)))
        self.assertRaises(ValueError, wannierize, ks, 0, [0], max_iter=-1)
        self.assertRaises(ValueError, wannierize, ks, 0, [0], tol=0.)
        self.assertRaises(ValueError, wannierize, ks, 0, [0], step=0.)
        nh = ssh(0.5, 1.)
        nh.set_onsite({'a': 0.1j})
        self.assertRaises(ValueError, wannierize, nh, 0, [0])
        gapless = ssh(1., 1.)  # bands touch at k = pi
        self.assertRaises(ValueError, wannierize, gapless, 0, [0], nk=10)
        # isolated dimers (t = 1): the lower band is (a - b)/sqrt(2) at every
        # k, which the symmetric bond orbital misses
        dimer = ssh(1., 0.)
        self.assertRaises(ValueError, wannierize, dimer, 0, np.array([[1.], [1.]]))


class TestDisentangle(unittest.TestCase):

    def test_frozen_bands_and_omega_i(self):
        ks = entangled_chain()
        frozen = (-5., -0.5)
        wf = wannierize(ks, [0, 1, 2], [0, 1], nk=10, frozen=frozen)
        self.assertTrue(wf.dis_converged)
        self.assertTrue(np.all(np.diff(wf.dis_history) <= 1e-12))  # every step lowers Omega_I
        self.assertLess(wf.dis_history[-1], wf.dis_history[0] / 10)
        self.assertAlmostEqual(wf.omega_i, wf.dis_history[-1], places=10)
        # the subspace: orthonormal columns in the basis of the three bands
        self.assertEqual(wf.gauge.shape, (10, 3, 2))
        self.assertTrue(np.allclose(wf.gauge.conj().transpose(0, 2, 1) @ wf.gauge, np.eye(2)))
        # the Wannier bands hold the frozen ones exactly on the mesh
        _, mesh = ks.mesh_grid(10)
        self.assertLess(frozen_error(ks._eigs(mesh), np.linalg.eigvalsh(wf.ham_k), frozen), 1e-12)
        self.assertLess(frozen_error(ks._eigs(mesh), wf.kspace()._eigs(mesh), frozen), 1e-12)
        # and close to them between its points, closer on a finer mesh
        path = np.linspace(0., 2 * np.pi, 201)[:, None]
        err = [frozen_error(ks._eigs(path), wannierize(ks, [0, 1, 2], [0, 1], nk=nk, frozen=frozen)
                                  .kspace()._eigs(path), frozen) for nk in (10, 40)]
        self.assertLess(err[1], err[0] / 20)
        # mixing 1 converges to the same subspace
        wf1 = wannierize(ks, [0, 1, 2], [0, 1], nk=10, frozen=frozen, mixing=1.)
        self.assertAlmostEqual(wf1.omega_i, wf.omega_i, places=8)

    def test_outer_window(self):
        # the states of the top band above 1.6 are left out of the subspace
        ks = entangled_chain()
        window = (-5., 1.6)
        wf = wannierize(ks, [0, 1, 2], [0, 1], nk=12, window=window, frozen=(-5., -0.5))
        _, mesh = ks.mesh_grid(12)
        outside = ks._eigs(mesh) > window[1]
        self.assertTrue(outside.any())
        self.assertTrue(np.all(np.abs(wf.gauge[outside]) == 0.))
        # too narrow: one state at some k
        self.assertRaises(ValueError, wannierize, ks, [0, 1, 2], [0, 1], nk=12, window=(-5., -0.95))
        # frozen window holding the three bands
        self.assertRaises(ValueError, wannierize, ks, [0, 1, 2], [0, 1], nk=12, frozen=(-5., 5.))

    def test_isolated_group_through_window(self):
        # an isolated band given through the disentanglement path: the same functions
        ks = haldane(0.2, 2.)
        ref = wannierize(ks, 0, [1], nk=6)
        wf = wannierize(ks, [0], [1], nk=6, window=(-10., 10.))
        self.assertEqual(len(wf.dis_history), 2)  # nothing to choose: converged at once
        self.assertAlmostEqual(wf.omega, ref.omega, places=10)
        self.assertTrue(np.allclose(wf.centers, ref.centers))
        self.assertEqual(len(ref.dis_history), 0)
        self.assertTrue(ref.dis_converged)

    def test_flat_graphene_pz(self):
        # pi bands crossing the sigma bands: the p_z subspace exactly, Wallace's model
        orbitals = {'a': ['s', 'px', 'py', 'pz'], 'b': ['s', 'px', 'py', 'pz']}
        gr = sk_kspace(lattices.honeycomb(), orbitals,
                             {1: {'ss_sigma': -6.77, 'sp_sigma': 5.58, 'pp_sigma': 5.04, 'pp_pi': -3.03}},
                             onsite={'a': {'s': -8.87}, 'b': {'s': -8.87}})
        pz = [n for n, (_, o) in enumerate(gr.sk_orbitals) if o == 'pz']
        self.assertRaises(ValueError, wannierize, gr, [3, 4], pz, nk=6)
        wf = wannierize(gr, list(range(8)), pz, nk=6, frozen=(-2., 2.))
        self.assertLess(abs(wf.omega_i), 1e-10)
        weight = np.abs(wf.functions.reshape(2, -1, gr.norb)) ** 2
        self.assertTrue(np.allclose(weight[:, :, pz].sum(axis=(1, 2)), 1.))
        wk = wf.kspace()
        self.assertEqual(wk.norb, 2)
        self.assertTrue(np.allclose(np.abs([h[3] for h in wk._hop]), 3.03))
        k = np.random.default_rng(0).random((20, 2)) @ np.array(gr.rec_vec_k)
        pi = np.linalg.eigvalsh(np.array([gr.get_ham(q)[np.ix_(pz, pz)] for q in k]))
        self.assertTrue(np.allclose(wk._eigs(k), pi, atol=1e-10))
        # orbitals on the carbon sites (folded into the home cell)
        self.assertTrue(np.allclose(np.sort(wk.orbital_positions(), axis=0),
                                             np.sort(gr.orbital_positions()[pz], axis=0), atol=1e-8))

    def test_iterations(self):
        ks = entangled_chain()
        wf0 = wannierize(ks, [0, 1, 2], [0, 1], nk=10, frozen=(-5., -0.5), dis_iter=0)
        self.assertEqual(len(wf0.dis_history), 1)
        self.assertTrue(wf0.dis_converged)
        wf1 = wannierize(ks, [0, 1, 2], [0, 1], nk=10, frozen=(-5., -0.5), dis_iter=1)
        self.assertFalse(wf1.dis_converged)
        self.assertLess(wf1.omega_i, wf0.omega_i)

    def test_errors(self):
        ks = entangled_chain()
        self.assertRaises(ValueError, wannierize, ks, [0, 1], [0, 1, 2])
        self.assertRaises(TypeError, wannierize, ks, [0, 1, 2], [0, 1], window=[-1., 1.])
        self.assertRaises(ValueError, wannierize, ks, [0, 1, 2], [0, 1], frozen=(1., 0.))
        self.assertRaises(ValueError, wannierize, ks, [0, 1, 2], [0, 1], window=(-2., 1.), frozen=(-3., 0.))
        self.assertRaises(ValueError, wannierize, ks, [0, 1, 2], [0, 1], dis_iter=-1)
        self.assertRaises(ValueError, wannierize, ks, [0, 1, 2], [0, 1], dis_tol=0.)
        self.assertRaises(ValueError, wannierize, ks, [0, 1, 2], [0, 1], mixing=1.5)
        wf = wannierize(ks, [0, 1, 2], [0, 1], nk=6, dis_iter=0, max_iter=0)
        self.assertRaises(ValueError, wf.kspace, -1.)


class TestWannierInterpolation(unittest.TestCase):

    def test_isolated_bands(self):
        # exact on the mesh, exponentially convergent between its points
        ks = haldane(0.2, 2.)
        k = np.random.default_rng(0).random((50, 2)) @ np.array(ks.rec_vec_k)
        err = []
        for nk in (6, 12):
            wk = wannierize(ks, 0, [1], nk=nk).kspace()
            _, mesh = ks.mesh_grid(nk)
            self.assertTrue(np.allclose(wk._eigs(mesh)[:, 0], ks._eigs(mesh)[:, 0], atol=1e-12))
            err.append(np.abs(wk._eigs(k)[:, 0] - ks._eigs(k)[:, 0]).max())
        self.assertLess(err[1], err[0] / 10)
        self.assertTrue(wk.is_hermitian())

    def test_odd_mesh_spinful_and_three_dimensions(self):
        # the whole spectrum of a model in one group: its own hoppings back
        ssh_ks = ssh(0.5, 1.)
        wk = wannierize(ssh_ks, [0, 1], [0, 1], nk=5).kspace()
        k = np.linspace(-np.pi, np.pi, 31)[:, None]
        self.assertTrue(np.allclose(wk._eigs(k), ssh_ks._eigs(k), atol=1e-10))
        km = kane_mele(0.06, M=1.)
        wf = wannierize(km, [0, 1], [2, 3], nk=6)
        _, mesh = km.mesh_grid(6)
        self.assertTrue(np.allclose(wf.kspace()._eigs(mesh), km._eigs(mesh)[:, :2], atol=1e-12))
        lat = Lattice(unit_cell=[{'tag': 'a', 'r0': (0., 0., 0.)}, {'tag': 'b', 'r0': (0.5, 0.5, 0.5)}],
                           prim_vec=[(1., 0., 0.), (0., 1., 0.), (0., 0., 1.)])
        ks = KSpace(lat)
        ks.set_onsite({'a': -2., 'b': 2.})
        ks.set_hopping([{'i': 0, 'j': 1, 'R': R, 't': 0.5}
                              for R in [(0, 0, 0), (-1, 0, 0), (0, -1, 0), (0, 0, -1),
                                        (-1, -1, 0), (-1, 0, -1), (0, -1, -1), (-1, -1, -1)]])
        wf = wannierize(ks, 0, [0], nk=(4, 4, 5))
        _, mesh = ks.mesh_grid((4, 4, 5))
        self.assertTrue(np.allclose(wf.kspace()._eigs(mesh)[:, 0], ks._eigs(mesh)[:, 0], atol=1e-12))
        self.assertTrue(np.allclose(wf.kspace().orbital_positions(), [[0., 0., 0.]], atol=1e-10))


if __name__ == '__main__':
    unittest.main()
