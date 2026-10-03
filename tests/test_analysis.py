"""
Analysis and plotting: tetrahedron density of states, constant-energy
contours, fat bands and spin textures, high-symmetry paths, plot helpers.
"""
import unittest
from math import factorial

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import ellipk

import tbkit.dos as dos
import tbkit.lattices as lattices
from tbkit.lattice import Lattice
from tbkit.kspace import KSpace, PAULI, high_symmetry_path

S3 = np.sqrt(3)


def lattice(prim_vec, n_sites=1):
    dim = max(2, len(prim_vec[0]))
    cell = [{'tag': 'ab'[n], 'r0': tuple([0.5 * n] + [0.] * (dim - 1))} for n in range(n_sites)]
    return Lattice(unit_cell=cell, prim_vec=prim_vec)


def rhombohedral(angle):
    c, s = np.cos(np.radians(angle)), np.sin(np.radians(angle))
    y = (c - c * c) / s
    return [(1., 0., 0.), (c, s, 0.), (c, y, np.sqrt(1 - c * c - y * y))]


def square(t=-1.):
    ks = KSpace(lattice([(1., 0.), (0., 1.)]))
    ks.set_hopping([{'i': 0, 'j': 0, 'R': (1, 0), 't': t}, {'i': 0, 'j': 0, 'R': (0, 1), 't': t}])
    return ks


def cubic():
    ks = KSpace(lattice([(1., 0., 0.), (0., 1., 0.), (0., 0., 1.)]))
    ks.set_hopping([{'i': 0, 'j': 0, 'R': R, 't': -1.} for R in [(1, 0, 0), (0, 1, 0), (0, 0, 1)]])
    return ks


def graphene(mass=0.3):
    lat = Lattice(unit_cell=[{'tag': 'a', 'r0': (0., 0.)}, {'tag': 'b', 'r0': (S3 / 2, 0.5)}],
                      prim_vec=[(S3, 0.), (S3 / 2, 1.5)])
    ks = KSpace(lat)
    ks.set_hopping([{'i': 0, 'j': 1, 'R': R, 't': -1.} for R in [(0, 0), (-1, 0), (0, -1)]])
    ks.set_onsite({'a': mass, 'b': -mass})
    return ks


def haldane(t2=0.2j):
    ks = graphene(mass=0.)
    ks.set_hopping([{'i': i, 'j': i, 'R': R, 't': s * t2}
                          for i, s in ((0, 1), (1, -1)) for R in [(1, 0), (0, -1), (-1, 1)]])
    return ks


def rashba(alpha=0.5):
    # H = -2(cos kx + cos ky) + 2 alpha (sin ky sigma_x - sin kx sigma_y)
    ks = KSpace(lattice([(1., 0.), (0., 1.)]), spin=True)
    ks.set_hopping([{'i': 0, 'j': 0, 'R': (1, 0), 't': -np.eye(2) + 1j * alpha * PAULI['y']},
                          {'i': 0, 'j': 0, 'R': (0, 1), 't': -np.eye(2) - 1j * alpha * PAULI['x']}])
    return ks


class TestTetrahedronDos(unittest.TestCase):

    def tearDown(self):
        plt.close('all')

    def test_simplices_tile_the_cell(self):
        for dim in (1, 2, 3):
            offsets = dos._mesh_simplices(dim)
            self.assertEqual(len(offsets), factorial(dim))
            vols = [abs(np.linalg.det(np.diff(s, axis=0).astype(float))) for s in offsets]
            self.assertTrue(np.allclose(vols, 1.))
            # every simplex runs from the lowest to the highest corner
            self.assertTrue(np.all(offsets[:, 0] == 0) and np.all(offsets[:, -1] == 1))

    def test_chain_matches_exact_bin_average(self):
        # E = -2 cos k: N(E) = arccos(-E/2)/pi states per site below E
        n = 2000
        en = (-2 * np.cos(2 * np.pi * np.arange(n) / n))[:, None]
        e_grid = np.linspace(-1.5, 1.5, 31)
        _, rho = dos.tetrahedron_dos(en, e_grid)
        edges = np.concatenate([[-1.55], (e_grid[1:] + e_grid[:-1]) / 2, [1.55]])
        exact = np.diff(np.arccos(-edges / 2) / np.pi) / np.diff(edges)
        self.assertTrue(np.allclose(rho / n, exact, atol=1e-5))

    def test_square_lattice_van_hove(self):
        # exact: rho(E) = K(1 - E^2/16) / (2 pi^2) per site; log divergence at E = 0
        nk = 200
        en = square().mesh_bands(nk).reshape(nk, nk, 1)
        e_grid = np.linspace(-3.5, 3.5, 701)
        _, rho = dos.tetrahedron_dos(en, e_grid)
        sel = [50, 150, 250, 300, 330, 370, 450]  # E = -3, -2, -1, -0.5, -0.2, 0.2, 1
        exact = ellipk(1 - e_grid[sel] ** 2 / 16) / (2 * np.pi ** 2)
        self.assertTrue(np.allclose(rho[sel] / nk ** 2, exact, rtol=1e-3))
        # a Gaussian of the same mesh spacing is far off at the van Hove singularity
        _, rho_g = dos.density_of_states(en, np.array([0.]), broadening=0.1)
        _, rho_t = dos.tetrahedron_dos(en, np.linspace(-0.01, 0.01, 3))
        self.assertGreater(rho_t[1] / rho_g[0], 1.5)

    def test_normalization_and_flat_band(self):
        en = cubic().mesh_bands(16).reshape(16, 16, 16, 1)
        e_grid, rho = dos.tetrahedron_dos(en)
        self.assertEqual(len(e_grid), 401)
        self.assertAlmostEqual(np.sum(rho * np.gradient(e_grid)) / en.size, 1., places=6)
        # a flat band is a delta function: its full weight lands in one bin
        flat = np.zeros((8, 8, 1))
        e_grid, rho = dos.tetrahedron_dos(flat, np.linspace(-1., 1., 21))
        self.assertAlmostEqual(np.sum(rho) * 0.1, 64.)
        self.assertEqual(np.count_nonzero(rho), 1)
        # all levels equal, default grid
        e_grid, _ = dos.tetrahedron_dos(np.ones((4, 4, 1)))
        self.assertAlmostEqual(e_grid[-1] - e_grid[0], 0.04)

    def test_complex_energies_use_real_part(self):
        en = square().mesh_bands(20).reshape(20, 20, 1)
        e_grid = np.linspace(-4.5, 4.5, 50)
        self.assertTrue(np.allclose(dos.tetrahedron_dos(en + 0.1j, e_grid)[1],
                                            dos.tetrahedron_dos(en, e_grid)[1]))

    def test_plot_dos_tetrahedron(self):
        fig = graphene().plot_dos(nk=30, kernel='tetrahedron')
        x, y = fig.axes[0].lines[0].get_data()
        self.assertAlmostEqual(np.sum(y * np.gradient(x)) / (30 * 30 * 2), 1., places=6)
        fig = cubic().plot_dos(nk=(6, 6, 6), kernel='tetrahedron', e_grid=np.linspace(-6, 6, 13))
        self.assertEqual(len(fig.axes[0].lines[0].get_xdata()), 13)

    def test_errors(self):
        self.assertRaises(ValueError, dos.tetrahedron_dos, np.zeros(4))
        self.assertRaises(ValueError, dos.tetrahedron_dos, np.zeros((2, 2, 2, 2, 1)))
        self.assertRaises(ValueError, dos.tetrahedron_dos, np.zeros((4, 1)), np.array([1., 0.]))
        self.assertRaises(ValueError, square().plot_dos, kernel='box')


class TestFermiSurface(unittest.TestCase):

    def tearDown(self):
        plt.close('all')

    def test_square_lattice_half_filling(self):
        # E = 0: the perfectly nested diamond |kx| + |ky| = pi, of length 4 sqrt(2) pi
        segs = square().fermi_surface(0., nk=64)[0]
        self.assertEqual(segs.shape[1:], (2, 2))
        pts = segs.reshape(-1, 2)
        self.assertTrue(np.allclose(np.abs(pts).sum(axis=1), np.pi, atol=1e-10))
        length = np.linalg.norm(segs[:, 1] - segs[:, 0], axis=1).sum()
        self.assertAlmostEqual(length, 4 * np.sqrt(2) * np.pi, places=8)

    def test_contour_on_level_set_and_converges(self):
        sq = square()
        lengths = []
        for nk in (50, 200, 400):
            segs = sq.fermi_surface(-1., nk=nk)[0]
            pts = segs.reshape(-1, 2)
            err = np.abs(-2 * np.cos(pts[:, 0]) - 2 * np.cos(pts[:, 1]) + 1).max()
            self.assertLess(err, 40. / nk ** 2)
            lengths.append(np.linalg.norm(segs[:, 1] - segs[:, 0], axis=1).sum())
        # the length converges (with the mesh, oscillating)
        self.assertLess(abs(lengths[0] - lengths[2]), 5e-4)
        self.assertLess(abs(lengths[1] - lengths[2]), 5e-5)

    def test_folded_into_first_zone(self):
        # graphene near the Dirac energy: small circles around the six K corners
        gra = graphene(mass=0.)
        segs = gra.fermi_surface(0.3, nk=90, bands=1)
        self.assertEqual(len(segs), 1)
        centres = segs[0].mean(axis=1)
        k_corner = 4 * np.pi / (3 * S3)
        self.assertTrue(np.all(np.linalg.norm(centres, axis=1) <= k_corner + 1e-9))
        # every piece sits next to a zone corner (a K or K' point)
        angles = np.arctan2(centres[:, 1], centres[:, 0])
        k_point = high_symmetry_path(gra.lat)[0][2]
        corners = (k_point[0] + 1j * k_point[1]) * np.exp(1j * np.arange(6) * np.pi / 3)
        dist = np.abs((centres[:, 0] + 1j * centres[:, 1])[:, None] - corners[None]).min(axis=1)
        self.assertLess(dist.max(), 0.3)
        self.assertGreater(np.ptp(angles), 5.)

    def test_3d_cubic(self):
        tris = cubic().fermi_surface(-4., nk=24)[0]
        self.assertEqual(tris.shape[1:], (3, 3))
        pts = tris.reshape(-1, 3)
        self.assertLess(np.abs(-2 * np.cos(pts).sum(axis=1) + 4).max(), 0.05)
        self.assertTrue(np.all(np.abs(tris.mean(axis=1)) <= np.pi + 1e-9))
        # a closed surface around Gamma: sum of the vectors area * normal vanishes
        normals = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
        outward = np.sign(np.sum(normals * tris.mean(axis=1), axis=1))[:, None]
        self.assertLess(np.abs((normals * outward).sum(axis=0)).max(), 1e-8)

    def test_default_mesh(self):
        # nk = 100 in 2D
        segs = square().fermi_surface()[0]
        self.assertAlmostEqual(np.linalg.norm(segs[:, 1] - segs[:, 0], axis=1).sum(),
                                    4 * np.sqrt(2) * np.pi)
        self.assertEqual(segs.shape, square().fermi_surface(0., nk=100)[0].shape)

    def test_axis_labels(self):
        self.assertEqual(square()._k_labels(), ['$k_x$', '$k_y$'])
        self.assertEqual(cubic()._k_labels(), ['$k_x$', '$k_y$', '$k_z$'])
        # a slab in the x-z plane, and a lattice in 3D space along no axis
        slab = KSpace(lattice([(1., 0., 0.), (0., 0., 1.)]))
        self.assertEqual(slab._k_labels(), ['$k_x$', '$k_z$'])
        tilted = KSpace(lattice([(1., 1., 0.), (0., 1., 1.)]))
        self.assertEqual(tilted._k_labels(), ['$k_1$', '$k_2$'])
        slab.set_hopping([{'i': 0, 'j': 0, 'R': (1, 0), 't': -1.}, {'i': 0, 'j': 0, 'R': (0, 1), 't': -1.}])
        ax = slab.plot_fermi_surface(-1., nk=20).axes[0]
        self.assertEqual((ax.get_xlabel(), ax.get_ylabel()), ('$k_x$', '$k_z$'))
        plt.close('all')

    def test_band_outside_energy(self):
        self.assertEqual(square().fermi_surface(10., nk=20)[0].shape, (0, 2, 2))

    def test_plots(self):
        fig = graphene(mass=0.).plot_fermi_surface(0.5, nk=40)
        ax = fig.axes[0]
        self.assertEqual(len(ax.collections), 1)  # only the upper band crosses E = 0.5
        self.assertIsNone(ax.get_legend())
        # two decoupled square-lattice bands, both crossing E = 0
        two = KSpace(lattice([(1., 0.), (0., 1.)], n_sites=2))
        two.set_hopping([{'i': i, 'j': i, 'R': R, 't': -1.} for i in (0, 1) for R in ((1, 0), (0, 1))])
        two.set_onsite({'a': 0.5, 'b': -0.5})
        ax = two.plot_fermi_surface(0., nk=20).axes[0]
        self.assertEqual(len(ax.collections), 2)
        self.assertEqual([t.get_text() for t in ax.get_legend().get_texts()], ['band 0', 'band 1'])
        fig = cubic().plot_fermi_surface(-4., nk=12, bands=0)
        self.assertEqual(fig.axes[0].name, '3d')
        fig = square().plot_fermi_surface(10., nk=10)  # empty
        self.assertIsNone(fig.axes[0].get_legend())
        self.assertRaises(ValueError, square().plot_fermi_surface, 0., 10, None, 0.)

    def test_errors(self):
        self.assertRaises(ValueError, KSpace(lattices.chain()).fermi_surface, 0.)
        self.assertRaises(TypeError, square().fermi_surface, 'a')
        self.assertRaises(ValueError, square().fermi_surface, 0., 10, [3])


class TestProjections(unittest.TestCase):

    def tearDown(self):
        plt.close('all')

    def test_sublattice_weights(self):
        gra = graphene()
        points, labels = high_symmetry_path(gra.lat)
        gra.k_path(points, nk=20)
        w_a, w_b = gra.band_weights('a'), gra.band_weights('b')
        self.assertEqual(w_a.shape, gra.en.shape)
        self.assertTrue(np.allclose(w_a + w_b, 1.))
        # at K, the lower band (onsite -0.3) sits on sublattice b only
        k_index = 40
        self.assertAlmostEqual(w_b[k_index, 0], 1., places=10)
        self.assertTrue(np.allclose(gra.band_weights(0), w_a))
        self.assertTrue(np.allclose(gra.band_weights([0, 1]), 1.))
        sz = np.diag([1., -1.]).astype(complex)
        self.assertTrue(np.allclose(gra.band_weights(sz), w_a - w_b))
        self.assertTrue(np.allclose(gra.band_weights('a', gra.ks[:5]), w_a[:5]))

    def test_spin_weights_and_texture(self):
        ras = rashba()
        self.assertTrue(np.allclose(ras.spin_operator('z'), np.diag([1., -1.])))
        self.assertTrue(np.allclose(ras.band_weights('a', [[0.3, 0.2]]), 1.))
        segs = ras.fermi_surface(-3., nk=80, bands=0)[0]
        mids = segs.mean(axis=1)
        spin = ras.spin_texture(mids, 0)
        # in-plane, unit length, perpendicular to (sin kx, sin ky): spin-momentum locking
        self.assertTrue(np.allclose(np.linalg.norm(spin, axis=1), 1.))
        self.assertTrue(np.allclose(spin[:, 2], 0.))
        self.assertTrue(np.allclose(np.sum(spin[:, :2] * np.sin(mids), axis=1), 0., atol=1e-10))
        # opposite chirality in the other band
        self.assertTrue(np.allclose(ras.spin_texture(mids, 1)[:, :2], -spin[:, :2], atol=1e-10))
        fig = ras.plot_spin_texture(0, -3., nk=60, n_arrows=20)
        arrows = fig.axes[0].collections[1]
        self.assertTrue(10 <= len(arrows.get_offsets()) <= 30)

    def test_degenerate_bands_are_averaged(self):
        # a spinful square lattice without spin-orbit coupling: every band is a
        # Kramers pair, whose spin depends on the eigenvectors; the average does not
        sq = KSpace(lattice([(1., 0.), (0., 1.)]), spin=True)
        sq.set_hopping([{'i': 0, 'j': 0, 'R': (1, 0), 't': -1.}, {'i': 0, 'j': 0, 'R': (0, 1), 't': -1.}])
        ks = np.random.default_rng(0).uniform(-np.pi, np.pi, (20, 2))
        for band in (0, 1):
            self.assertTrue(np.allclose(sq.spin_texture(ks, band), 0., atol=1e-12))
        # rotate the spin quantization axis: a weight that was +-1 per eigenvector is now 0
        n = np.array([1., 2., 2.]) / 3
        op = sum(c * sq.spin_operator(a) for c, a in zip(n, 'xyz'))
        self.assertTrue(np.allclose(sq.band_weights(op, ks), 0., atol=1e-12))
        self.assertTrue(np.allclose(sq.band_weights('a', ks), 1.))
        # the Rashba bands are split away from the time-reversal invariant
        # momenta, and degenerate (zero spin) only at them
        ras = rashba()
        spin = ras.spin_texture([[0., 0.], [np.pi, 0.], [0.4, 0.3]], 0)
        self.assertTrue(np.allclose(spin[:2], 0., atol=1e-12))
        self.assertAlmostEqual(np.linalg.norm(spin[2]), 1.)

    def test_plot_fat_bands(self):
        gra = graphene()
        points, labels = high_symmetry_path(gra.lat)
        gra.k_path(points, nk=20)
        fig = gra.plot_bands(node_labels=labels, weights=gra.band_weights('a'))
        self.assertEqual(len(fig.axes), 2)  # with a colorbar
        self.assertEqual(fig.axes[0].collections[0].get_cmap().name, 'viridis')
        fig = gra.plot_bands(weights=gra.band_weights(np.diag([1., -1.]).astype(complex)))
        self.assertEqual(fig.axes[0].collections[0].get_cmap().name, 'RdBu_r')
        self.assertAlmostEqual(fig.axes[0].collections[0].norm.vmin, -fig.axes[0].collections[0].norm.vmax)
        fig = gra.plot_bands(weights=gra.band_weights('a'), style='size', ms=5.)
        self.assertEqual(len(fig.axes), 1)
        self.assertAlmostEqual(max(c.get_sizes().max() for c in fig.axes[0].collections), 25.)
        fig = gra.plot_bands(weights=np.zeros(gra.en.shape), style='size')
        self.assertEqual(fig.axes[0].collections[0].get_sizes().max(), 0.)

    def test_errors(self):
        gra = graphene()
        self.assertRaises(RuntimeError, gra.band_weights, 'a')  # no bands yet
        self.assertRaises(ValueError, gra.spin_operator, 'z')
        self.assertRaises(ValueError, rashba().spin_operator, 'w')
        self.assertRaises(ValueError, gra.spin_texture, [[0., 0.]], 0)
        self.assertRaises(ValueError, gra.plot_spin_texture, 0)
        self.assertRaises(ValueError, rashba().plot_spin_texture, 0, 10.)
        gra.get_bands([[0., 0.], [0.1, 0.]])
        self.assertRaises(ValueError, gra.plot_bands, weights=np.zeros((3, 2)))
        self.assertRaises(ValueError, gra.plot_bands, weights=np.zeros((2, 2)), style='fat')


class TestHighSymmetryPath(unittest.TestCase):

    def check(self, prim_vec, labels, points=None):
        p, l = high_symmetry_path(lattice(prim_vec))
        self.assertEqual(l, labels)
        if points is not None:
            self.assertTrue(np.allclose(np.array(p), points))
        return p

    def test_1d_and_2d(self):
        self.check([(2., 0.)], ['$-X$', r'$\Gamma$', 'X'], [[-np.pi / 2], [0.], [np.pi / 2]])
        G = r'$\Gamma$'
        self.check([(1., 0.), (0., 1.)], [G, 'X', 'M', G],
                      [[0, 0], [np.pi, 0], [np.pi, np.pi], [0, 0]])
        self.check([(1., 0.), (0., 2.)], [G, 'X', 'S', 'Y', G],
                      [[0, 0], [0, np.pi / 2], [np.pi, np.pi / 2], [np.pi, 0], [0, 0]])
        for a2 in [(0.5, S3 / 2), (-0.5, S3 / 2), (S3 / 2, 1.5)]:
            a1 = (1., 0.) if a2[0] != S3 / 2 else (S3, 0.)
            p = self.check([a1, a2], [G, 'M', 'K', G])
            a = np.linalg.norm(a1)
            self.assertAlmostEqual(np.linalg.norm(p[1]), 2 * np.pi / (S3 * a))
            self.assertAlmostEqual(np.linalg.norm(p[2]), 4 * np.pi / (3 * a))
        # oblique and centred rectangular: X, C, Y edge midpoints, H, H1 corners of the zone
        for prim_vec in ([(1., 0.), (0.3, 1.7)], [(1., 0.3), (1., -0.3)]):
            p = self.check(prim_vec, [G, 'X', 'H', 'C', '$H_1$', 'Y', G])
            for corner, mids in ((p[2], (p[1], p[3])), (p[4], (p[3], p[5]))):
                for m in mids:  # the corner lies on the bisector of both edges
                    self.assertAlmostEqual(corner @ m, m @ m)
        # a 2D lattice in 3D space: k in the coordinates of get_ham
        self.check([(1., 0., 0.), (0., 0., 1.)], [G, 'X', 'M', G])

    def test_3d(self):
        G = r'$\Gamma$'
        p = self.check([(1., 0., 0.), (0., 1., 0.), (0., 0., 1.)], [G, 'X', 'M', G, 'R', 'X'])
        self.assertTrue(np.allclose(p[4], np.pi))
        p = self.check([(0., .5, .5), (.5, 0., .5), (.5, .5, 0.)],
                          [G, 'X', 'W', 'K', G, 'L', 'U', 'W', 'L', 'K'])
        self.assertTrue(np.allclose(np.sort(np.abs(p[1])), [0, 0, 2 * np.pi]))
        self.assertTrue(np.allclose(np.abs(p[5]), np.pi))  # L = (pi, pi, pi)
        p = self.check([(-.5, .5, .5), (.5, -.5, .5), (.5, .5, -.5)], [G, 'H', 'N', G, 'P', 'H'])
        self.assertTrue(np.allclose(np.sort(np.abs(p[1])), [0, 0, 2 * np.pi]))
        for c in (1.6, 0.6):
            p = self.check([(1., 0., 0.), (0., 1., 0.), (0., 0., c)],
                              [G, 'X', 'M', G, 'Z', 'R', 'A', 'Z'])
            self.assertTrue(np.allclose(p[4], [0., 0., np.pi / c]))
        for a2, c in (((.5, S3 / 2, 0.), 1.6), ((-.5, S3 / 2, 0.), 1.)):
            p = self.check([(1., 0., 0.), a2, (0., 0., c)], [G, 'M', 'K', G, 'A', 'L', 'H', 'A'])
            self.assertAlmostEqual(np.linalg.norm(p[2]), 4 * np.pi / 3)
            self.assertTrue(np.allclose(p[4], [0., 0., np.pi / c]))
        # a rotated simple cubic lattice
        rot = np.linalg.qr(np.array([[1., 2., 0.], [0., 1., 3.], [1., 0., 1.]]))[0]
        p, _ = high_symmetry_path(lattice([tuple(r) for r in rot]))
        self.assertAlmostEqual(np.linalg.norm(p[4]), np.sqrt(3) * np.pi)

    def test_k_path_and_unsupported(self):
        gra = graphene()
        points, labels = high_symmetry_path(gra.lat)
        dist, en = gra.k_path(points, nk=10)
        self.assertEqual(len(gra.nodes), len(labels))
        gra.plot_bands(node_labels=labels)
        plt.close('all')
        for prim_vec in ([(1., 0., 0.), (0., 1.3, 0.), (0., 0., 1.6)],  # orthorhombic
                              [(-.5, .5, 1.), (.5, -.5, 1.), (.5, .5, -1.)],  # body-centred tetragonal
                              [(1., 0., 0.), (0.3, 1.1, 0.), (0.2, 0.1, 1.4)],  # triclinic
                              rhombohedral(70.)):
            self.assertRaises(ValueError, high_symmetry_path, lattice(prim_vec))


class TestPlotHelpers(unittest.TestCase):

    def tearDown(self):
        plt.close('all')

    def test_plot_berry_curvature(self):
        hal = haldane()
        fig = hal.plot_berry_curvature(0, nk=24)
        ax = fig.axes[0]
        self.assertIn('C = -1.000', ax.get_title())
        mesh = ax.collections[0]
        omega = mesh.get_array().reshape(24, 24)
        # density times plaquette area gives back the flux: total 2 pi C
        area = abs(np.linalg.det(hal.rec_vec_k)) / 24 ** 2
        self.assertAlmostEqual(omega.sum() * area, 2 * np.pi * hal.chern_number([0], nk=24))
        # 3D: a stack of Haldane layers
        lat3 = Lattice(unit_cell=[{'tag': 'a', 'r0': (0., 0., 0.)}, {'tag': 'b', 'r0': (S3 / 2, .5, 0.)}],
                           prim_vec=[(S3, 0., 0.), (S3 / 2, 1.5, 0.), (0., 0., 1.)])
        hal3 = KSpace(lat3)
        hal3.set_hopping([{'i': 0, 'j': 1, 'R': R + (0,), 't': -1.} for R in [(0, 0), (-1, 0), (0, -1)]])
        hal3.set_hopping([{'i': i, 'j': i, 'R': R + (0,), 't': s * 0.2j}
                                 for i, s in ((0, 1), (1, -1)) for R in [(1, 0), (0, -1), (-1, 1)]])
        fig = hal3.plot_berry_curvature(0, nk=12)
        self.assertEqual(fig.axes[0].get_xlabel(), '$k_1$')
        self.assertIn('C = -1.000', fig.axes[0].get_title())
        fig = square().plot_berry_curvature(0, nk=6)  # zero curvature
        self.assertIn('C = 0.000', fig.axes[0].get_title())

    def test_plot_wannier_flow(self):
        fig = haldane().plot_wannier_flow(0, nk=30, nk_perp=11)
        ax = fig.axes[0]
        self.assertEqual(ax.get_xlabel(), r'$k_2/|\mathbf{b}_2|$')
        self.assertEqual(len(ax.lines[0].get_xdata()), 11)
        fig = haldane().plot_wannier_flow(0, nk=30, nk_perp=11, direction=1, flow=0)
        self.assertEqual(fig.axes[0].get_xlabel(), r'$k_1/|\mathbf{b}_1|$')

    def test_plot_surface_spectral_function(self):
        ssh = KSpace(lattices.chain())
        ssh.set_hopping([{'i': 0, 'j': 0, 'R': (1,), 't': 1.}])
        energies = np.linspace(-3., 3., 31)
        hal = haldane()
        fig = hal.plot_surface_spectral_function([np.zeros(2), np.array([1., 0.]), np.array([2., 0.])],
                                                                   energies, direction=1, nk=5,
                                                                   node_labels=['a', 'b', 'c'])
        ax = fig.axes[0]
        self.assertEqual([t.get_text() for t in ax.get_xticklabels()], ['a', 'b', 'c'])
        self.assertEqual(ax.collections[0].get_array().size, 31 * 11)  # flattened before matplotlib 3.8
        fig = ssh.plot_surface_spectral_function([(0.,), (np.pi,)], energies, direction=0,
                                                                   nk=4, bulk=True, log=False)
        self.assertEqual(fig.axes[0].get_title(), 'Bulk')
        self.assertRaises(ValueError, hal.plot_surface_spectral_function, [np.zeros(2), np.ones(2)],
                              energies, 1, 5, node_labels=['a'])


if __name__ == '__main__':
    unittest.main()
