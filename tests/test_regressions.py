"""
Regression tests for bugs fixed after 0.2.1.

Each test pins down a defect that produced silently wrong numbers (or broke a
documented call) rather than raising, so a reappearance cannot slip through.
"""
import os
import unittest
from math import sqrt

import numpy as np
import scipy.sparse as sparse

import tbkit.lattices as lattices
from tbkit.graphene import GrapheneLattice, GrapheneSystem
from tbkit.kspace import KSpace, PAULI, ribbon
from tbkit.lattice import Lattice
from tbkit.plot import Plot
from tbkit.propagation import Propagation
from tbkit.save import Save
from tbkit.system import System


def square(n1=3, n2=3):
    lat = Lattice(unit_cell=[{'tag': 'a', 'r0': (0., 0.)}], prim_vec=[(1., 0.), (0., 1.)])
    lat.get_lattice(n1=n1, n2=n2)
    return lat


def pt_chain(gamma, n_cells=6):
    lat = Lattice(unit_cell=[{'tag': 'a', 'r0': (0., 0.)},
                                        {'tag': 'b', 'r0': (0.5, 0.)}], prim_vec=[(1., 0.)])
    lat.get_lattice(n1=n_cells)
    sys = System(lat)
    sys.set_onsite({'a': 1j*gamma, 'b': -1j*gamma})
    sys.set_hopping_manual({(2*n, 2*n + 1): 1. for n in range(n_cells)})
    sys.get_ham()
    return sys


class TestSystemRegressions(unittest.TestCase):

    def test_set_hopping_def_keeps_angles_and_tags(self):
        # set_hopping_def assigned a scalar to hop['ang'] and hop['tag']
        # without a mask, flattening every bond's angle onto one value.
        sys = System(square())
        sys.set_hopping([{'n': 1, 't': 1.}])
        ang_before = np.sort(np.unique(sys.hop['ang']))
        sys.set_hopping_def({(0, 1): 5.})
        self.assertTrue(np.allclose(np.sort(np.unique(sys.hop['ang'])), ang_before))
        self.assertTrue(np.allclose(ang_before, [0., 90.]))
        self.assertEqual(int((sys.hop['t'] == 5.).sum()), 1)

    def test_set_hopping_def_after_manual_hoppings(self):
        # it also read self.vec_hop, which set_hopping_manual never fills.
        lat = square(2, 2)
        sys = System(lat)
        sys.set_hopping_manual({(0, 1): 1., (1, 2): 1.})
        sys.set_hopping_def({(0, 1): 3.})
        self.assertEqual(sys.hop['t'][0], 3.)

    def test_set_hopping_by_tag_replaces_for_every_order(self):
        # '&' binds tighter than '==', so the mask that clears previously-set
        # hoppings misfired for every hopping order except n == 1.
        lat = Lattice(unit_cell=[{'tag': 'a', 'r0': (0., 0.)},
                                            {'tag': 'b', 'r0': (0.5, 0.)}],
                              prim_vec=[(1., 0.), (0., 1.)])
        lat.get_lattice(n1=5, n2=1)
        # which tags exist depends on the shell: 'ab' at odd n, 'aa' at even n
        for n, tag in ((1, 'ab'), (2, 'aa'), (3, 'ab'), (4, 'aa')):
            sys = System(lat)
            sys.set_hopping([{'n': n, 'tag': tag, 't': 1.}])
            size = len(sys.hop)
            sys.set_hopping([{'n': n, 'tag': tag, 't': 9.}])
            self.assertEqual(len(sys.hop), size, 'n={} duplicated hoppings'.format(n))
            self.assertTrue(np.allclose(sys.hop['t'], 9.), 'n={} kept stale values'.format(n))
            sys.get_ham()
            self.assertAlmostEqual(np.abs(sys.ham.toarray()).max(), 9., places=10,
                                                msg='n={} summed duplicates into the Hamiltonian'.format(n))

    def test_change_hopping_square_matches_angles_approximately(self):
        # angles come from arctan2 (29.999999999999996, ...), so exact
        # equality silently matched nothing.
        lat = lattices.honeycomb()
        lat.get_lattice(n1=4, n2=4)
        sys = System(lat)
        sys.set_hopping([{'n': 1, 't': 1.}])
        n30 = int(np.isclose(sys.hop['ang'], 30., atol=1e-3).sum())
        self.assertGreater(n30, 0)
        sys.change_hopping_square([{'n': 1, 'ang': 30., 't': 7.}],
                                                xlims=[-99., 99.], ylims=[-99., 99.])
        self.assertEqual(int((sys.hop['t'] == 7.).sum()), n30)

    def test_get_eig_clears_stale_left_eigenvectors(self):
        # left eigenvectors from an earlier left=True call were re-sorted with
        # the new ordering, so get_petermann silently returned inf.
        sys = pt_chain(0.3)
        sys.get_eig(eigenvec=True, left=True)
        sys.get_petermann()
        self.assertTrue(np.all(np.isfinite(sys.petermann)))
        sys.get_eig(eigenvec=True, left=False)
        self.assertEqual(sys.ln.size, 0)
        self.assertRaises(RuntimeError, sys.get_petermann)

    def test_get_eig_returns_right_eigenvectors_with_left(self):
        # scipy.linalg.eig returns (w, vl, vr), but the result was unpacked as
        # (en, rn, ln), so `rn` held the LEFT eigenvectors whenever left=True.
        # get_petermann survived it (|<L|R>| is symmetric), but `rn`,
        # `intensity`, `pola` and `ipr` described the wrong state. Only a
        # non-reciprocal model shows it: for a complex *symmetric* H (the PT
        # chain) the left vectors are the conjugates of the right ones, so
        # |rn|**2 is the same either way.
        lat = Lattice(unit_cell=[{'tag': 'a', 'r0': (0., 0.)}], prim_vec=[(1., 0.)])
        lat.get_lattice(n1=3)
        sys = System(lat)
        # Hatano-Nelson: hopping 1. to the right, 0.3 to the left.
        ham = np.array([[0., 1., 0.],
                                 [0.3, 0., 1.],
                                 [0., 0.3, 0.]], dtype='c16')
        sys.ham = sparse.csr_matrix(ham)
        sys.get_eig(eigenvec=True, left=True)
        # H|rn> = en|rn>
        self.assertTrue(np.allclose(ham @ sys.rn, sys.en * sys.rn))
        # <ln|H = en<ln|
        self.assertTrue(np.allclose(sys.ln.conj().T @ ham,
                                                 sys.en[:, None] * sys.ln.conj().T))
        # and left=True must not change the right eigenvectors it also returns
        rn_left = sys.rn.copy()
        sys.get_eig(eigenvec=True, left=False)
        self.assertTrue(np.allclose(np.abs(rn_left), np.abs(sys.rn)))

    def test_peierls_landau_gauge_matches_symmetric_gauge(self):
        # the documented Landau-gauge example integrated along x instead of y,
        # making it a pure gauge (zero field).
        alpha = 0.02
        sym = System(square(6, 6))
        sym.set_hopping([{'n': 1, 't': 1.}])
        sym.set_magnetic_field(alpha=alpha)
        sym.get_ham()
        lan = System(square(6, 6))
        lan.set_hopping([{'n': 1, 't': 1.}])
        B = 2 * np.pi * alpha
        lan.set_peierls_phase(lambda xi, yi, xj, yj: B * (yj - yi) * (xi + xj) / 2)
        lan.get_ham()
        # different gauges, same spectrum
        sym.get_eig()
        lan.get_eig()
        self.assertTrue(np.allclose(np.sort(sym.en.real), np.sort(lan.en.real), atol=1e-8))


class TestLatticeRegressions(unittest.TestCase):

    def test_rotation_uses_degrees(self):
        # theta was converted with PI/360, halving every rotation.
        lat = square(2, 1)
        lat.rotation(90.)
        far = np.argmax(np.hypot(lat.coor['x'], lat.coor['y']))
        self.assertTrue(np.allclose([lat.coor['x'][far], lat.coor['y'][far]], [0., 1.], atol=1e-12))

    def test_rotation_applied_once_for_multisite_unit_cell(self):
        # the rotation was re-applied once per unit-cell site, each about a
        # different centre.
        lat = lattices.kagome()
        lat.get_lattice(n1=2, n2=2)
        before = np.stack([lat.coor['x'], lat.coor['y']], axis=1).copy()
        lat.rotation(90.)
        after = np.stack([lat.coor['x'], lat.coor['y']], axis=1)
        rot = np.array([[0., -1.], [1., 0.]])
        self.assertTrue(np.allclose(np.sort(after, axis=0),
                                                np.sort(before @ rot.T, axis=0), atol=1e-12))


class TestKSpaceRegressions(unittest.TestCase):

    def test_1d_chain_any_orientation(self):
        # get_ham dropped the y-component of a 1D primitive vector, giving a
        # k-independent (flat) Hamiltonian for a chain not aligned with x.
        for prim_vec in ([(1., 0.)], [(0., 1.)], [(0.6, 0.8)], [(-1.2, 0.5)]):
            lat = Lattice(unit_cell=[{'tag': 'a', 'r0': (0., 0.)}], prim_vec=prim_vec)
            chain = KSpace(lat)
            chain.set_hopping([{'i': 0, 'j': 0, 'R': (1,), 't': 1.}])
            en = chain.mesh_bands(nk=64).ravel()
            self.assertAlmostEqual(en.max() - en.min(), 4., places=6,
                                                msg='prim_vec={}'.format(prim_vec))

    def test_mesh_grid_covers_full_brillouin_zone_in_1d(self):
        lat = Lattice(unit_cell=[{'tag': 'a', 'r0': (0., 0.)}], prim_vec=[(0.6, 0.8)])
        chain = KSpace(lat)
        chain.set_hopping([{'i': 0, 'j': 0, 'R': (1,), 't': 1.}])
        _, ks = chain.mesh_grid(nk=8)
        # one full period of the Bloch phase across the mesh
        a = np.array(lat.prim_vec[0])
        dk = 8 * (ks[1, 0] - ks[0, 0])
        self.assertAlmostEqual(dk * np.linalg.norm(a), 2*np.pi, places=10)

    def test_ribbon_direction_0_samples_full_zone(self):
        # a honeycomb ribbon cut with direction=0 sampled only 25% of the zone.
        lat = lattices.honeycomb()
        hop = [{'i': 0, 'j': 1, 'R': (0, 0), 't': 1.},
                  {'i': 0, 'j': 1, 'R': (-1, 0), 't': 1.},
                  {'i': 0, 'j': 1, 'R': (0, -1), 't': 1.}]
        for direction in (0, 1):
            rib = ribbon(lat, hop, width=4, direction=direction)
            a = np.array(rib.lat.prim_vec[0])
            _, ks = rib.mesh_grid(nk=16)
            dk = 16 * (ks[1, 0] - ks[0, 0])
            self.assertAlmostEqual(dk * np.linalg.norm(a), 2*np.pi, places=10,
                                                msg='direction={}'.format(direction))

    def test_spin_onsite_accepts_2x2_matrix(self):
        # an in-plane Zeeman field was inexpressible: set_hopping refused
        # i == j with R == 0 and pointed at set_onsite, which took only a
        # number or an (E_up, E_down) pair.
        chain = KSpace(lattices.chain(), spin=True)
        chain.set_onsite({'a': 0.5*PAULI['x']})
        ham = chain.get_ham([0.])
        self.assertTrue(np.allclose(ham, 0.5*PAULI['x']))
        self.assertTrue(np.allclose(ham, ham.conj().T))


class TestPropagationRegressions(unittest.TestCase):

    def test_norm_preserves_probability(self):
        # normalizing by sum|psi| (L1) destroyed sum|psi|^2 = 1.
        lat = Lattice(unit_cell=[{'tag': 'a', 'r0': (0., 0.)}], prim_vec=[(1., 0.)])
        lat.get_lattice(n1=20)
        sys = System(lat)
        sys.set_hopping([{'n': 1, 't': 1.}])
        sys.get_ham()
        psi = np.zeros(20, 'c16')
        psi[10] = 1.
        for norm in (False, True):
            prop = Propagation(lat)
            prop.get_propagation(sys.ham, psi, steps=30, dz=0.1, norm=norm)
            self.assertAlmostEqual(float((np.abs(prop.prop[:, -1])**2).sum()), 1., places=10)


class TestGrapheneRegressions(unittest.TestCase):

    def _flake(self, maker, n, center):
        lat = GrapheneLattice()
        getattr(lat, maker)(n=n)
        if center:
            lat.coor['x'] -= lat.coor['x'].mean()
            lat.coor['y'] -= lat.coor['y'].mean()
        return GrapheneSystem(lat)

    def test_beta_lims_keep_hoppings_positive(self):
        # the old formula used one sublattice's extreme y and ignored x, so it
        # returned limits well outside the positive-hopping range.
        for maker, n, center in [('triangle_zigzag', 10, True),
                                          ('triangle_zigzag', 10, False),
                                          ('hexagon_zigzag', 5, True),
                                          ('circle', 8, True)]:
            sys = self._flake(maker, n, center)
            lims = sys.get_beta_lims()
            self.assertLess(lims[0], lims[1], msg=maker)
            self.assertTrue(np.all(np.isfinite(lims)), msg=maker)
            for beta in lims:
                sys.set_hop_linear_strain(t=1., beta=float(beta))
                self.assertGreater(sys.hop['t'].real.min(), 0., msg=maker)
            # and the bounds are tight: just outside, a hopping turns negative
            for beta, step in ((lims[0], -0.02), (lims[1], 0.02)):
                sys.set_hop_linear_strain(t=1., beta=float(beta) + step)
                self.assertLess(sys.hop['t'].real.min(), 0., msg=maker)

    def test_get_butterfly_uses_its_t_argument(self):
        # get_butterfly accepted t but hard-coded t=1 in the sweep.
        sys = self._flake('triangle_zigzag', 6, True)
        sys.get_butterfly(t=1., N=3)
        one = sys.butterfly.copy()
        sys.get_butterfly(t=2., N=3)
        self.assertTrue(np.allclose(sys.butterfly, 2*one, atol=1e-8))


class TestPlotRegressions(unittest.TestCase):

    def test_petermann_plot_needs_only_get_petermann(self):
        # Plot.petermann() guarded on sys.ipr, raising AttributeError even
        # after a correct get_eig(left=True) + get_petermann().
        sys = pt_chain(0.3)
        sys.get_eig(eigenvec=True, left=True)
        sys.get_petermann()
        fig, _ = Plot(sys).petermann()
        self.assertEqual(fig.__class__.__name__, 'Figure')

    def test_spectrum_petermann_keyword(self):
        sys = pt_chain(0.3)
        sys.get_eig(eigenvec=True, left=True)
        sys.get_petermann()
        fig = Plot(sys).spectrum(petermann=True)
        self.assertEqual(fig.__class__.__name__, 'Figure')


class TestPlotAxParameter(unittest.TestCase):
    """Plot.lattice / lattice_hop can draw onto a caller-supplied axis."""

    def _system(self):
        lat = lattices.kagome()
        lat.get_lattice(n1=3, n2=3)
        sys = System(lat)
        sys.set_hopping([{'n': 1, 't': 1.}])
        sys.get_ham()
        return sys

    def test_lattice_draws_on_given_axis(self):
        import matplotlib.pyplot as plt
        sys = self._system()
        plt.close('all')
        fig, axes = plt.subplots(1, 2)
        for ax in axes:
            out = Plot(sys).lattice(plt_hop=True, ms=6, ax=ax)
            # the caller's figure is returned, not a new one
            self.assertIs(out, fig)
            self.assertGreater(len(ax.lines), 0)
        # no extra figures were created behind the caller's back
        self.assertEqual(len(plt.get_fignums()), 1)
        plt.close('all')

    def test_lattice_without_axis_still_makes_its_own_figure(self):
        import matplotlib.pyplot as plt
        sys = self._system()
        plt.close('all')
        fig = Plot(sys).lattice(plt_hop=True, ms=6)
        self.assertEqual(fig.__class__.__name__, 'Figure')
        self.assertEqual(len(plt.get_fignums()), 1)
        plt.close('all')

    def test_lattice_hop_accepts_axis(self):
        import matplotlib.pyplot as plt
        sys = self._system()
        sys.get_coor_hop()
        fig, ax = plt.subplots()
        out = Plot(sys).lattice_hop(ms=6, ax=ax)
        self.assertIs(out, fig)
        plt.close(fig)


class TestSaveRegressions(unittest.TestCase):

    def test_dir_main_is_joined_not_concatenated(self):
        import shutil
        import tempfile
        tmp = tempfile.mkdtemp()
        try:
            sav = Save(dir_name='run', dir_main=os.path.join(tmp, 'out'))
            self.assertEqual(sav.dir_name, os.path.join(tmp, 'out', 'run'))
            self.assertTrue(os.path.isdir(os.path.join(tmp, 'out', 'run')))
            self.assertFalse(os.path.exists(os.path.join(tmp, 'outrun')))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == '__main__':
    unittest.main()
