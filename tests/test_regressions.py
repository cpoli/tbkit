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

from tests.test_propagation import draw_animation


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


# --------------------------------------------------------------------------
# Bugs fixed after 0.3.0
# --------------------------------------------------------------------------


def hermitian(sys):
    ham = sys.ham.toarray()
    return np.allclose(ham, ham.conj().T)


def haldane_model(prim_swap=False, M=0.):
    lat = lattices.honeycomb()
    hops = [{'i': 0, 'j': 1, 'R': (0, 0), 't': 1.},
               {'i': 0, 'j': 1, 'R': (-1, 0), 't': 1.},
               {'i': 0, 'j': 1, 'R': (0, -1), 't': 1.}]
    for R in [(0, 1), (-1, 0), (1, -1)]:
        hops.append({'i': 0, 'j': 0, 'R': R, 't': 0.2j})
        hops.append({'i': 1, 'j': 1, 'R': R, 't': -0.2j})
    if prim_swap:
        # the same model, with a1 and a2 exchanged (a left-handed basis)
        lat = Lattice(unit_cell=lat.unit_cell, prim_vec=lat.prim_vec[::-1])
        hops = [dict(h, R=h['R'][::-1]) for h in hops]
    hal = KSpace(lat)
    hal.set_hopping(hops)
    hal.set_onsite({'a': M, 'b': -M})
    return hal


class TestSiteOrderRegressions(unittest.TestCase):

    def test_get_ham_hermitian_after_reordering_sites(self):
        # set_hopping took "i < j" as meaning "angle in [0, 180)", true only
        # for sites sorted by (y, x). After rotation / clean_coor / +=, bonds
        # came with mixed-sign angles, get_ham took them for a non-Hermitian
        # upper + lower fill and dropped the Hermitian conjugate: half the
        # bonds were missing (21 nonzeros instead of 42 for this flake).
        def rotated(theta):
            def f(lat):
                lat.rotation(theta)
            return f
        edits = {'rotation(90)': rotated(90.), 'rotation(-45)': rotated(-45.),
                    'rotation(150)': rotated(150.),
                    'clean_coor': lambda lat: lat.clean_coor(),
                    'change_sign_y': lambda lat: lat.change_sign_y()}
        ref = System(lattices.honeycomb())
        ref.lat.get_lattice(n1=3, n2=3)
        ref.set_hopping([{'n': 1, 't': 1.}])
        ref.get_ham()
        ref.get_eig()
        for name, edit in edits.items():
            lat = lattices.honeycomb()
            lat.get_lattice(n1=3, n2=3)
            edit(lat)
            sys = System(lat)
            sys.set_hopping([{'n': 1, 't': 1.}])
            self.assertTrue(np.all(sys.hop['ang'] >= 0), msg=name)
            sys.get_ham()
            self.assertTrue(hermitian(sys), msg=name)
            self.assertEqual(sys.ham.nnz, ref.ham.nnz, msg=name)
            sys.get_eig()
            self.assertTrue(np.allclose(sys.en, ref.en), msg=name)

    def test_get_ham_hermitian_after_iadd(self):
        a = square(2, 2)
        b = square(2, 2)
        b.shift_y(-2.)
        a += b  # appended below, without re-sorting
        sys = System(a)
        sys.set_hopping([{'n': 1, 't': 1.}])
        sys.get_ham()
        self.assertTrue(hermitian(sys))
        ref = System(square(2, 4))
        ref.set_hopping([{'n': 1, 't': 1.}])
        ref.get_ham()
        ref.get_eig()
        sys.get_eig()
        self.assertTrue(np.allclose(sys.en, ref.en))

    def test_angle_selection_on_rotated_lattice(self):
        # an 'ang' selection must see the rotated angles, all in [0, 180)
        lat = square(3, 3)
        lat.rotation(-30.)
        sys = System(lat)
        sys.set_hopping([{'n': 1, 'ang': 60., 't': 2.}, {'n': 1, 'ang': 150., 't': 1.}])
        self.assertEqual(len(sys.hop), 12)
        sys.get_ham()
        self.assertTrue(hermitian(sys))

    def test_strain_on_rotated_flake(self):
        # set_hop_linear_strain oriented the bonds from the sorted site order
        # (and the "outward" direction from the angles 30/150), so a rotated
        # flake got a non-Hermitian, wrongly strained Hamiltonian. Rotating
        # by 120 degrees maps the flake, sublattices and triaxial strain onto
        # themselves: the spectrum must not change.
        spectra = []
        for theta in (0., 120.):
            lat = GrapheneLattice()
            lat.circle(n=6)
            lat.rotation(theta)
            sys = GrapheneSystem(lat)
            sys.set_hop_linear_strain(t=1., beta=0.1)
            sys.get_ham()
            self.assertTrue(hermitian(sys))
            sys.get_eig()
            spectra.append(sys.en)
        self.assertTrue(np.allclose(spectra[0], spectra[1], atol=1e-8))


class TestSystemRegressions2(unittest.TestCase):

    def test_set_hopping_ang_and_tag_uses_atol(self):
        # the 'ang' + 'tag' branch matched angles within 1 degree for the
        # selection but within ATOL for its size: two bonds 0.5 degree apart
        # made set_given_hopping crash on a shape mismatch.
        c, s = np.cos(np.pi/360), np.sin(np.pi/360)
        coor = np.array([(0., 0., 'a'), (1., 0., 'a'), (0., 5., 'a'), (c, 5. + s, 'a')],
                               dtype=[('x', 'f8'), ('y', 'f8'), ('tag', 'U1')])
        lat = Lattice(unit_cell=[{'tag': 'a', 'r0': (0., 0.)}], prim_vec=[(1., 0.)])
        lat.add_sites(coor)
        sys = System(lat)
        sys.set_hopping([{'n': 1, 'ang': 0., 'tag': 'aa', 't': 1.}])
        self.assertEqual(len(sys.hop), 1)

    def test_get_ham_detects_stale_hoppings(self):
        # hop_sites compared max(i) with max(i) (never j), and sites < max
        # instead of sites <= max: stale hoppings reached scipy's cryptic
        # index error instead of the intended message.
        lat = square(3, 3)
        sys = System(lat)
        sys.set_hopping([{'n': 1, 't': 1.}])
        lat.remove_sites([8])
        with self.assertRaisesRegex(ValueError, 'clear_hopping'):
            sys.get_ham()

    def test_set_hopping_def_reversed_key(self):
        # a key given as (j, i) while sys.hop stores (i, j) was silently
        # ignored; it now sets H_ji = val, i.e. t_ij = conj(val).
        sys = System(square(2, 1))
        sys.set_hopping([{'n': 1, 't': 1.}])
        sys.set_hopping_def({(1, 0): 2j})
        sys.get_ham()
        self.assertEqual(sys.ham[1, 0], 2j)
        self.assertEqual(sys.ham[0, 1], -2j)

    def test_set_hopping_def_ignores_non_hoppings(self):
        # backward compatible: a pair that is not a hopping is still ignored
        sys = System(square(3, 1))
        sys.set_hopping([{'n': 1, 't': 1.}])
        before = sys.hop.copy()
        sys.set_hopping_def({(0, 2): 5.})
        self.assertTrue(np.array_equal(sys.hop, before))
    def test_set_hopping_manual_validates(self):
        # set_hopping_manual validated nothing: an out-of-range index built
        # a malformed hop array that only failed later, in get_ham. Only
        # inputs that could never have worked are refused.
        sys = System(square(2, 1))
        self.assertRaises(ValueError, sys.set_hopping_manual, {(0, 5): 1.})
        self.assertRaises(ValueError, sys.set_hopping_manual, {(-1, 0): 1.})
        self.assertRaises(TypeError, sys.set_hopping_manual, {(0, 1): 'a'})

    def test_set_hopping_manual_backward_compatible(self):
        # everything that worked before the validation still works:
        # NumPy indices and values, (i, i) keys, a non-bool upper_part.
        sys = System(square(3, 1))
        sys.set_hopping_manual({(i, i + 1): np.float64(1.) for i in np.arange(2)})
        sys.set_hopping_manual({(1, 1): 0.5}, upper_part=1)
        sys.get_ham()
        ref = np.array([[0., 1., 0.], [1., 1., 1.], [0., 1., 0.]])
        self.assertTrue(np.allclose(sys.ham.toarray(), ref))
    def test_set_hopping_four_keys_needs_ang_and_tag(self):
        sys = System(square(2, 2))
        self.assertRaises(KeyError, sys.set_hopping, [{'n': 1, 't': 1., 'tag': 'aa', 'x': 0}])

    def test_get_intensity_en_uses_real_part(self):
        sys = pt_chain(gamma=0.5)
        sys.get_eig(eigenvec=True)
        n_real = np.sum((sys.en.real > -0.5) & (sys.en.real < 0.5))
        intensity = sys.get_intensity_en([-0.5, 0.5])
        self.assertAlmostEqual(float(intensity.sum()), float(n_real), places=10)


class TestKSpaceRegressions2(unittest.TestCase):

    def test_chern_number_independent_of_prim_vec_handedness(self):
        # the plaquette loop k -> k+b1 -> k+b1+b2 -> k+b2 is clockwise for a
        # left-handed prim_vec, flipping the sign of every Berry flux.
        c_right = haldane_model().chern_number(bands=[0], nk=20)
        c_left = haldane_model(prim_swap=True).chern_number(bands=[0], nk=20)
        self.assertAlmostEqual(c_right, 1., places=6)
        self.assertAlmostEqual(c_left, c_right, places=6)

    def test_haldane_gap_formula(self):
        # with the C3-symmetric second-neighbor vectors the gaps at K, K' are
        # 2|M +- 3 sqrt(3) t2| (Haldane 1988); the gallery's previous vector
        # set broke C3 and gave a gap three times smaller.
        from tbkit.kspace import reciprocal_vectors
        hal = haldane_model()
        b1, b2 = (np.array(v) for v in reciprocal_vectors(hal.lat.prim_vec))
        K = (b1 - b2) / 3
        for k in (K, -K):
            en = np.linalg.eigvalsh(hal.get_ham(k))
            self.assertAlmostEqual(en[1] - en[0], 6*sqrt(3)*0.2, places=10)

    def test_plot_bands_after_plot_dos(self):
        # plot_dos (via mesh_bands) overwrote the band structure kept for
        # plot_bands, which then crashed on mismatched shapes.
        import matplotlib.pyplot as plt
        sq = KSpace(lattices.square())
        sq.set_hopping([{'i': 0, 'j': 0, 'R': (1, 0), 't': 1.},
                               {'i': 0, 'j': 0, 'R': (0, 1), 't': 1.}])
        _, en = sq.k_path([(0., 0.), (np.pi, 0.)], nk=10)
        plt.close(sq.plot_dos(nk=10))
        self.assertTrue(np.allclose(sq.en, en))
        plt.close(sq.plot_bands(node_labels=['G', 'X']))

    def test_plot_bands_after_get_bands(self):
        # documented, but get_bands never set ks_dist / nodes
        import matplotlib.pyplot as plt
        chain = KSpace(lattices.chain())
        chain.set_hopping([{'i': 0, 'j': 0, 'R': (1,), 't': 1.}])
        ks = np.linspace(-np.pi, np.pi, 21)[:, None]
        chain.get_bands(ks)
        self.assertEqual(len(chain.ks_dist), 21)
        self.assertAlmostEqual(chain.ks_dist[-1], 2*np.pi)
        plt.close(chain.plot_bands())

    def test_complex_onsite_non_hermitian_bands(self):
        # get_bands used eigvalsh, which silently dropped the imaginary part
        # of a gain/loss onsite energy (E = 2 instead of 2 + 1j). Complex
        # onsite energies are still accepted, and now diagonalized with the
        # general solver.
        import matplotlib.pyplot as plt
        chain = KSpace(lattices.chain())
        chain.set_hopping([{'i': 0, 'j': 0, 'R': (1,), 't': 1.}])
        chain.set_onsite({'a': 1j})
        self.assertFalse(chain.is_hermitian())
        en = chain.get_bands([[0.], [np.pi]])
        self.assertTrue(np.allclose(en.ravel(), [2. + 1j, -2. + 1j]))
        _, en_path = chain.k_path([(-np.pi,), (np.pi,)], nk=10)
        self.assertTrue(np.allclose(en_path.imag, 1.))
        plt.close(chain.plot_bands())
        plt.close(chain.plot_dos(nk=10))
        # PT-symmetric dimer chain: real bands below the exceptional point,
        # sorted by real part; spinful pairs of complex values are accepted.
        lat = Lattice(unit_cell=[{'tag': 'a', 'r0': (0., 0.)}, {'tag': 'b', 'r0': (0.5, 0.)}],
                             prim_vec=[(1., 0.)])
        dimer = KSpace(lat)
        dimer.set_hopping([{'i': 0, 'j': 1, 'R': (0,), 't': 1.}])
        dimer.set_onsite({'a': 0.5j, 'b': -0.5j})
        en, vn = dimer.get_bands([[0.]], eigenvec=True)
        self.assertTrue(np.allclose(en.ravel(), [-np.sqrt(0.75), np.sqrt(0.75)]))
        ham = dimer.get_ham([0.])
        self.assertTrue(np.allclose(ham @ vn[0], vn[0] * en[0]))
        KSpace(lattices.chain(), spin=True).set_onsite({'a': (1j, 0.)})
        # a Hermitian model keeps real energies and the Hermitian solver
        chain.set_onsite({'a': 1.})
        self.assertTrue(chain.is_hermitian())
        self.assertEqual(chain.get_bands([[0.]]).dtype, np.float64)

    def test_non_hermitian_chern_number(self):
        # a small uniform gain on the Haldane model keeps its Chern number
        hal = haldane_model()
        c_ref = hal.chern_number(bands=[0], nk=20)
        hal.set_onsite({'a': 0.05j, 'b': 0.05j})
        self.assertFalse(hal.is_hermitian())
        self.assertAlmostEqual(hal.chern_number(bands=[0], nk=20), c_ref, places=6)

    def test_non_hermitian_chern_number_without_real_line_gap(self):
        # Bands ordered by Re E are scrambled without a real line gap: for
        # i*H of the Qi-Wu-Zhang model (C = 1, all Re E = 0) chern_number gave
        # -3, -7 or 5 depending on nk; with 2.5i*sigma_z it gave 1 where the
        # imaginary-gap bands have C = 0. It now refuses.
        def qwz(scale=1., gz=0.):
            q = KSpace(lattices.square(), spin=True)
            q.set_onsite({'a': scale * -1. * PAULI['z'] + 1j * gz * PAULI['z']})
            hop = [((1, 0), 0.5 * PAULI['z'] - 0.5j * PAULI['x']),
                   ((0, 1), 0.5 * PAULI['z'] - 0.5j * PAULI['y'])]
            q.set_hopping([{'i': 0, 'j': 0, 'R': R, 't': scale * t} for R, t in hop]
                          + [{'i': 0, 'j': 0, 'R': (-R[0], -R[1]), 't': scale * t.conj().T}
                             for R, t in hop], hermitian=False)
            return q
        self.assertAlmostEqual(qwz().chern_number(0, 20), 1., places=8)
        for nk in (20, 30, 31):
            self.assertRaises(ValueError, qwz(1j).chern_number, 0, nk)
        self.assertAlmostEqual(qwz(1j).biorthogonal_chern_number(0, 30, gap='imaginary'), 1., places=8)
        self.assertRaises(ValueError, qwz(gz=2.5).chern_number, 0, 30)
        self.assertAlmostEqual(qwz(gz=2.5).biorthogonal_chern_number(0, 30, gap='imaginary'), 0.,
                               places=8)
    def test_ribbon_accepts_list_R(self):
        # ribbon() only indexes 'R', so a list worked and must keep working
        hop = [{'i': 0, 'j': 1, 'R': [0, 0], 't': 1.}, {'i': 0, 'j': 1, 'R': [-1, 0], 't': 1.},
                  {'i': 0, 'j': 1, 'R': [0, -1], 't': 1.}]
        rib = ribbon(lattices.honeycomb(), hop, width=3)
        self.assertEqual(rib.norb, 6)


class TestPropagationRegressions2(unittest.TestCase):

    def _chain(self, n=3):
        lat = Lattice(unit_cell=[{'tag': 'a', 'r0': (0., 0.)}], prim_vec=[(1., 0.)])
        lat.get_lattice(n1=n)
        sys = System(lat)
        sys.set_hopping([{'n': 1, 't': 1.}])
        sys.get_ham()
        return lat, sys

    def test_pumping_with_few_steps(self):
        # with steps < len(hams) + 1 each stage got 0 steps and the "after
        # pumping" loop overwrote prop[:, 0] with an uninitialized column
        # (nan with norm=True). It now keeps the initial state and evolves
        # under hams[-1].
        lat, sys = self._chain()
        _, sys2 = self._chain()
        sys2.set_onsite({'a': 1.})
        sys2.get_ham()
        psi = np.zeros(3, 'c16')
        psi[0] = 1.
        pump = Propagation(lat)
        pump.get_pumping([sys.ham, sys.ham, sys2.ham], psi, steps=3, dz=0.1)
        ref = Propagation(lat)
        ref.get_propagation(sys2.ham, psi, steps=3, dz=0.1)
        self.assertTrue(np.allclose(pump.prop, ref.prop))
        pump.get_pumping([sys.ham], psi, steps=1, dz=0.1)
        self.assertTrue(np.allclose(pump.prop[:, 0], psi))

    def test_pumping_unchanged_with_enough_steps(self):
        # the fix only touches the degenerate case: compare with the stages
        # written out by hand.
        lat, sys = self._chain()
        _, sys2 = self._chain()
        sys2.set_onsite({'a': 1.})
        sys2.get_ham()
        psi = np.zeros(3, 'c16')
        psi[0] = 1.
        pump = Propagation(lat)
        pump.get_pumping([sys.ham, sys2.ham], psi, steps=9, dz=0.1, norm=False)
        def step(ham, v):
            eye = np.eye(3)
            return np.linalg.solve(eye + 0.05j*ham.toarray(), (eye - 0.05j*ham.toarray()) @ v)
        v, out = psi.copy(), [psi.copy()]
        hams = [sys.ham]*2 + [(1-c)*sys.ham + c*sys2.ham for c in np.linspace(0, 1, 3)] + [sys2.ham]*3
        for ham in hams:
            v = step(ham, v)
            out.append(v)
        self.assertTrue(np.allclose(pump.prop, np.array(out).T))
    def test_get_animation_nb_imag(self):
        # prop_type='imag' animated the real part.
        import matplotlib.pyplot as plt
        lat, sys = self._chain(4)
        psi = np.zeros(4, 'c16')
        psi[0] = 1.
        prop = Propagation(lat)
        prop.get_propagation(sys.ham, psi, steps=5, dz=0.1)
        ani = prop.get_animation_nb(prop_type='imag')
        draw_animation(ani)
        scat = ani._fig.axes[0].collections[0]
        self.assertTrue(np.allclose(scat.get_array(), prop.prop.imag[:, 0]))
        plt.close(ani._fig)

    def test_get_animation_limits_cover_every_site(self):
        # the axis limits came from the first and last site, which for a
        # (y, x)-sorted flake are not the extreme x.
        import matplotlib.pyplot as plt
        lat = GrapheneLattice()
        lat.hexagon_zigzag(n=2)
        sys = System(lat)
        sys.set_hopping([{'n': 1, 't': 1.}])
        sys.get_ham()
        psi = np.zeros(lat.sites, 'c16')
        psi[0] = 1.
        prop = Propagation(lat)
        prop.get_propagation(sys.ham, psi, steps=3, dz=0.1)
        ani = prop.get_animation()
        draw_animation(ani)
        xlim = ani._fig.axes[0].get_xlim()
        self.assertLess(xlim[0], lat.coor['x'].min())
        self.assertGreater(xlim[1], lat.coor['x'].max())
        plt.close(ani._fig)


class TestPlotRegressions2(unittest.TestCase):

    def test_spectrum_limits_for_one_sided_spectrum(self):
        # the default y-range was [-max(E), max(E)]: an all-negative spectrum
        # gave an inverted range with every point off screen.
        import matplotlib.pyplot as plt
        sys = System(square(3, 3))
        sys.set_onsite({'a': -10.})
        sys.set_hopping([{'n': 1, 't': 1.}])
        sys.get_ham()
        sys.get_eig()
        for method in ('spectrum', 'spectrum_complex'):
            fig = getattr(Plot(sys), method)()
            ymin, ymax = fig.axes[0].get_ylim()
            self.assertLess(ymin, sys.en.real.min(), msg=method)
            self.assertGreater(ymax, sys.en.real.max(), msg=method)
            plt.close(fig)

    def test_butterfly_uses_its_betas_argument(self):
        # butterfly() read sys.betas, so it crashed for anything but a
        # GrapheneSystem that had run get_butterfly.
        import matplotlib.pyplot as plt
        sys = System(square(2, 2))
        betas = np.linspace(-1., 1., 5)
        butterfly = np.outer(betas, [1., 2., 3., 4.]) + np.arange(4)
        fig = Plot(sys).butterfly(betas, butterfly, lims=[-5., 10.])
        self.assertEqual(fig.axes[0].get_title(), 'Energies depending on strain')
        plt.close(fig)

    def test_intensity_area_figsize(self):
        import matplotlib.pyplot as plt
        sys = System(square(2, 2))
        fig = Plot(sys).intensity_area(np.ones(4) / 4, figsize=(3., 2.))
        self.assertTrue(np.allclose(fig.get_size_inches(), [3., 2.]))
        plt.close(fig)

    def test_negative_hoppings_drawn(self):
        # linewidths were c*Re(t): negative hoppings got negative widths.
        import matplotlib.pyplot as plt
        sys = System(square(2, 1))
        sys.set_hopping([{'n': 1, 't': -1.}])
        fig = Plot(sys).lattice(plt_hop=True)
        widths = [line.get_linewidth() for line in fig.axes[0].lines if len(line.get_xdata()) == 2]
        self.assertTrue(widths and all(w > 0 for w in widths))
        plt.close(fig)


class TestCoverageGaps(unittest.TestCase):

    def test_get_bands_checks_ks_shape(self):
        chain = KSpace(lattices.chain())
        chain.set_hopping([{'i': 0, 'j': 0, 'R': (1,), 't': 1.}])
        self.assertRaises(ValueError, chain.get_bands, np.zeros((3, 2)))
        # an empty set of k-points, and a non-bool eigenvec, still work
        self.assertEqual(chain.get_bands(np.zeros((0, 1))).shape, (0, 1))
        en, vn = chain.get_bands(np.zeros((2, 1)), 1)
        self.assertEqual(vn.shape, (2, 1, 1))
    def test_spin_onsite_scalar_applies_to_both_spins(self):
        chain = KSpace(lattices.chain(), spin=True)
        chain.set_onsite({'a': 0.3})
        self.assertTrue(np.allclose(chain.get_ham([0.]), 0.3*PAULI['0']))

    def test_plt_hopping_on_current_axis(self):
        import matplotlib.pyplot as plt
        sys = System(square(2, 1))
        sys.set_hopping([{'n': 1, 't': 1.}])
        fig, ax = plt.subplots()
        Plot(sys).plt_hopping(sys.lat.coor, sys.hop, 1.)
        self.assertEqual(len(ax.lines), 1)
        plt.close(fig)


if __name__ == '__main__':
    unittest.main()
