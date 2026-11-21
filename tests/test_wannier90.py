'''
tbkit.io.read_wannier90: hand-written _hr.dat files whose bands are known
analytically (degeneracy weights included).
'''
import itertools
import os
import tempfile
import unittest

import numpy as np

import scipy.linalg as sla

from tbkit import models
from tbkit.io import (read_wannier90, read_hr, read_win_cell, read_centres, BOHR, write_hr, write_tb,
                      write_wannier90, read_tb, read_wannier90_tb, save_model, load_model, _hr_kspace)
from tbkit.kspace import KSpace
from tbkit.lattice import Lattice
from tbkit.optics import optical_conductivity, berry_curvature_dipole, shift_current


def hr_text(num_wann, entries, weights, header=' written on 26Sep2026 at 12:00:00'):
    '''entries: dict R -> (num_wann, num_wann) matrix, in order.'''
    lines = [header, '{:12d}'.format(num_wann), '{:12d}'.format(len(entries))]
    w = list(weights)
    lines += [''.join('{:5d}'.format(g) for g in w[k:k + 15]) for k in range(0, len(w), 15)]
    for R, h in entries.items():
        for n in range(num_wann):
            for m in range(num_wann):
                lines.append('{:5d}{:5d}{:5d}{:5d}{:5d}{:12.6f}{:12.6f}'.format(
                    *R, m + 1, n + 1, h[m][n].real, h[m][n].imag))
    return '\n'.join(lines) + '\n'


class TestWannier90(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, name, text):
        path = os.path.join(self.tmp.name, name)
        with open(path, 'w') as f:
            f.write(text)
        return path

    def cubic_file(self, t=-1., t2=0.25):
        '''Simple cubic, one orbital: nearest neighbours t, plus a hopping
        t2 to (+-2, 0, 0) that sits on the Wigner-Seitz boundary of a 4-point
        mesh along x: listed at R = +-2 with weight 2 and value 2 t2 each,
        H(k) = 2t (cos kx + cos ky + cos kz) + 2 t2 cos 2kx. Zero-valued
        points (+-1, +-1, 0) and (0, +-1, +-1) bring nrpts to 17 (two lines
        of weights).'''
        entries, weights = {}, []
        for R in itertools.product((-2, -1, 0, 1, 2), (-1, 0, 1), (-1, 0, 1)):
            steps = sum(abs(c) for c in R)
            if R == (0, 0, 0):
                val = 0.3
            elif abs(R[0]) == 2 and R[1:] == (0, 0):
                val = 2 * t2
            elif steps == 1:
                val = t
            elif steps == 2 and abs(R[0]) < 2 and (R[0] == 0 or R[2] == 0):
                val = 0.
            else:
                continue
            entries[R] = [[complex(val)]]
            weights.append(2 if abs(R[0]) == 2 else 1)
        self.assertEqual(len(entries), 17)
        return self.write('cubic_hr.dat', hr_text(1, entries, weights))

    def test_read_hr(self):
        num_wann, R, ndegen, ham = read_hr(self.cubic_file())
        self.assertEqual((num_wann, R.shape, ham.shape), (1, (17, 3), (17, 1, 1)))
        self.assertEqual(int(ndegen.sum()), 19)
        self.assertEqual(ham[(R == [2, 0, 0]).all(axis=1)][0, 0, 0], 0.5)

    def test_cubic_with_degeneracy_weights(self):
        t, t2 = -1., 0.25
        ks = read_wannier90(self.cubic_file(t, t2), prim_vec=[(1., 0., 0.), (0., 1., 0.), (0., 0., 1.)])
        self.assertTrue(ks.is_hermitian())
        self.assertEqual(len(ks._hop), 2 * 4)  # 3 nearest-neighbour bonds + the (2, 0, 0) one, x2
        self.assertEqual(ks.onsite[0], 0.3)
        for k in np.random.default_rng(2).uniform(-3, 3, (6, 3)):
            exact = 0.3 + 2 * t * np.cos(k).sum() + 2 * t2 * np.cos(2 * k[0])
            self.assertAlmostEqual(ks.get_bands(k)[0, 0], exact, places=12)
        # a cutoff drops the small hopping
        ks = read_wannier90(self.cubic_file(t, t2), prim_vec=[(1., 0., 0.), (0., 1., 0.), (0., 0., 1.)],
                                     cutoff=0.3)
        self.assertEqual(len(ks._hop), 2 * 3)

    def graphene_files(self, t=-2.8, bohr=True):
        '''Graphene p_z, nearest neighbours: E = +- t |f(k)|.'''
        a = 2.46
        cell = np.array([[a, 0., 0.], [a / 2, a * np.sqrt(3) / 2, 0.], [0., 0., 12.]])
        tau_b = (cell[0] + cell[1]) / 3
        bonds = {(0, 0, 0): [[0, t], [t, 0]], (-1, 0, 0): [[0, t], [0, 0]], (0, -1, 0): [[0, t], [0, 0]],
                     (1, 0, 0): [[0, 0], [t, 0]], (0, 1, 0): [[0, 0], [t, 0]]}
        hr = self.write('gra_hr.dat', hr_text(2, {R: np.array(h, complex) for R, h in bonds.items()},
                                                               [1] * 5))
        scale = 1 / BOHR if bohr else 1.
        win = self.write('gra.win', '! comment\nnum_wann = 2\n\nBegin Unit_Cell_Cart\n{}\n'.format(
            'bohr' if bohr else 'ang') + '\n'.join(' '.join('{:.10f}'.format(c * scale) for c in v) for v in cell)
                                  + '  # trailing comment\nEnd Unit_Cell_Cart\n')
        xyz = self.write('gra_centres.xyz', '4\ncomment\nX 0. 0. 0.\nX {} {} 0.\nC 0. 0. 0.\nC 1. 1. 0.\n'
                                   .format(*tau_b[:2]))
        return hr, win, xyz, cell, tau_b

    def test_graphene_layer(self):
        t = -2.8
        hr, win, xyz, cell, tau_b = self.graphene_files(t)
        np.testing.assert_allclose(read_win_cell(win), cell, atol=1e-9)
        ks = read_wannier90(hr, win=win, centres=xyz, tags=['a', 'b'], dim=2)
        self.assertEqual(ks.dim, 2)
        self.assertEqual(ks.space_dim, 2)
        np.testing.assert_allclose(ks.orbital_positions(), [[0, 0], tau_b[:2]], atol=1e-9)
        self.assertEqual(list(ks.tags), ['a', 'b'])
        a1, a2 = cell[0, :2], cell[1, :2]
        for k in np.random.default_rng(3).uniform(-2, 2, (6, 2)):
            f = abs(1 + np.exp(-1j * k @ a1) + np.exp(-1j * k @ a2))
            np.testing.assert_allclose(ks.get_bands(k)[0], np.sort([t * f, -t * f]),
                                                   atol=1e-10)
        # Dirac point at K, and the neighbour-order model is the same
        b1, b2 = np.array(ks.rec_vec)
        np.testing.assert_allclose(ks.get_bands((b1 - b2) / 3)[0], 0., atol=1e-10)
        # the same file as a 3D model (the layer repeated along z, no hopping)
        hr, win, xyz, _, _ = self.graphene_files(t, bohr=False)
        ks3 = read_wannier90(hr, win=win, positions=[[0, 0, 0], tau_b])
        self.assertEqual(ks3.dim, 3)
        np.testing.assert_allclose(ks3.get_bands(np.zeros(3))[0], [3 * t, -3 * t], atol=1e-10)
        # without positions, every orbital at the origin
        ks0 = read_wannier90(hr, prim_vec=[tuple(cell[0, :2]), tuple(cell[1, :2])], dim=2)
        np.testing.assert_array_equal(ks0.orbital_positions(), 0.)

    def test_non_hermitian_file(self):
        '''Hatano-Nelson chain written as an _hr.dat file.'''
        tr, tl = 1.2, 0.5
        entries = {(-1, 0, 0): [[tl]], (0, 0, 0): [[0.1j]], (1, 0, 0): [[tr]]}
        hr = self.write('hn_hr.dat', hr_text(1, {R: np.array(h, complex) for R, h in entries.items()}, [1] * 3))
        ks = read_wannier90(hr, prim_vec=[(1., 0., 0.), (0., 1., 0.), (0., 0., 1.)])
        self.assertFalse(ks.is_hermitian())
        self.assertTrue(ks._nonreciprocal)
        k = 0.7
        self.assertAlmostEqual(complex(ks.get_ham((k, 0., 0.))[0, 0]),
                                      0.1j + tr * np.exp(1j * k) + tl * np.exp(-1j * k), places=12)
        # a file missing the partner of an R is not Hermitian either
        entries = {(0, 0, 0): [[0.]], (1, 0, 0): [[1.]]}
        hr = self.write('one_hr.dat', hr_text(1, {R: np.array(h, complex) for R, h in entries.items()}, [1] * 2))
        self.assertFalse(read_wannier90(hr, prim_vec=[(1., 0., 0.), (0., 1., 0.), (0., 0., 1.)]).is_hermitian())

    def test_errors(self):
        pv = [(1., 0., 0.), (0., 1., 0.), (0., 0., 1.)]
        good = self.cubic_file()
        self.assertRaises(ValueError, read_hr, self.write('a', 'header\n'))
        self.assertRaises(ValueError, read_hr, self.write('b', 'header\nx 1\n1\n'))
        self.assertRaises(ValueError, read_hr, self.write('c', 'header\n1\n2\n1\n'))
        self.assertRaises(ValueError, read_hr, self.write('d', 'header\n1\n1\n1\n0 0 0 1 1 0.\n'))
        self.assertRaises(ValueError, read_hr, self.write(
            'e', 'h\n2\n1\n1\n0 0 0 1 1 0 0\n0 0 0 2 1 0 0\n1 0 0 1 2 0 0\n0 0 0 2 2 0 0\n'))
        self.assertRaises(TypeError, read_hr, 1)
        self.assertRaises(ValueError, read_win_cell, self.write('w1', 'num_wann = 1\n'))
        self.assertRaises(ValueError, read_win_cell,
                                  self.write('w2', 'begin unit_cell_cart\n1 0 0\n0 1 0\nend unit_cell_cart\n'))
        self.assertRaises(ValueError, read_win_cell,
                                  self.write('w3', 'begin unit_cell_cart\n1 0 0\n0 1 0\n0 0 x\nend unit_cell_cart\n'))
        self.assertRaises(ValueError, read_centres, self.write('x', '1\nc\nX 0 0 0\n'), 2)
        self.assertRaises(ValueError, read_centres, self.write('x', '1\nc\nX 0 0 0\n'), 0)
        self.assertRaises(TypeError, read_wannier90, 1, prim_vec=pv)
        self.assertRaises(ValueError, read_wannier90, good, prim_vec=pv, dim=1)
        self.assertRaises(ValueError, read_wannier90, good, prim_vec=pv, cutoff=-1.)
        self.assertRaises(ValueError, read_wannier90, good, prim_vec=pv, tol=0.)
        self.assertRaises(ValueError, read_wannier90, good)
        self.assertRaises(ValueError, read_wannier90, good, prim_vec=[(1., 0.), (0., 1.)])
        self.assertRaises(ValueError, read_wannier90, good, prim_vec=[(1., 0., 0.), (1., 0., 0.), (0., 0., 1.)])
        self.assertRaises(ValueError, read_wannier90, good, prim_vec=pv, positions=np.zeros((2, 3)))
        self.assertRaises(TypeError, read_wannier90, good, prim_vec=pv, tags=['abcdefghi'])
        self.assertRaises(ValueError, read_wannier90, good, prim_vec=pv, tags=['a', 'b'])
        # a 2D model must not hop along R3, and its cell must be planar
        self.assertRaises(ValueError, read_wannier90, good, prim_vec=[(1., 0.), (0., 1.)], dim=2)
        tilted = self.write('t.win', 'begin unit_cell_cart\n1 0 0.5\n0 1 0\n0 0 1\nend unit_cell_cart\n')
        self.assertRaises(ValueError, read_wannier90, good, win=tilted, dim=2)



def rotated(ks, eps=0.6, n=25):
    r'''
    The 2-orbital model *ks* (orbitals at their positions, a diagonal
    position operator) in the basis of the k-dependent unitary
    V(k) = exp(i eps S(k)), S(k) a periodic Hermitian matrix: the same
    physics, whose position operator, A'(k) = V^dagger tau V + i V^dagger dV,
    has elements off the diagonal and between cells, components that do not
    commute and a curl. H'(R) and A'(R) from an n x n mesh (odd n, every R
    with its opposite); the Fourier tails are below 1e-13.
    '''
    a, b, tau = np.array(ks.lat.prim_vec), np.array(ks.rec_vec), ks.orbital_positions()
    f = np.arange(n) / n
    kc = np.array(np.meshgrid(f, f, indexing='ij')).reshape(2, -1).T @ b
    sx, sy, sz = (np.array(m, dtype=complex) for m in ([[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]))
    v = np.array([sla.expm(1j * eps * (sx * np.cos(k @ a[0]) + sy * np.sin(k @ a[1])
                                         + 0.5 * sz * np.cos(k @ (a[0] - a[1])))) for k in kc])
    cells = np.array(np.meshgrid(np.arange(n), np.arange(n), indexing='ij')).reshape(2, -1).T
    cells = (cells + n // 2) % n - n // 2
    rc = cells @ a
    phase = np.exp(-1j * kc @ rc.T) / len(kc)
    v_r = np.einsum('kR,kij->Rij', phase, v)
    vh = v.conj().transpose(0, 2, 1)
    conn = [vh @ np.diag(tau[:, c]) @ v
            + 1j * vh @ np.einsum('kR,Rij->kij', 1 / phase / len(kc), 1j * rc[:, c, None, None] * v_r)
            for c in range(2)]
    ham_r = np.einsum('kR,kij->Rij', phase, vh @ ks._bloch_ham(kc) @ v)
    pos_r = np.stack([np.einsum('kR,kij->Rij', phase, c) for c in conn], axis=-1)
    origin = np.flatnonzero((cells == 0).all(axis=1))[0]
    centres = pos_r[origin].diagonal().T.real.copy()
    new = _hr_kspace(ks.lat.prim_vec, centres, ['a', 'b'], cells, ham_r, 1e-13, 1e-7)
    pos_r[origin, [0, 1], [0, 1]] = 0.
    new._position_hop = [(int(i), int(j), cells[p] @ a, pos_r[p, i, j])
                         for p, i, j in zip(*np.nonzero(np.abs(pos_r).max(axis=-1) > 1e-13))]
    return new


class TestWriteWannier90(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = lambda name: os.path.join(self.tmp.name, name)

    def tearDown(self):
        self.tmp.cleanup()

    def assert_same_bands(self, ks1, ks2, atol=1e-10):
        ks = np.random.default_rng(4).uniform(-3, 3, (12, ks1.dim))
        np.testing.assert_allclose(ks1.get_bands(ks), ks2.get_bands(ks), rtol=0, atol=atol)

    def test_round_trips(self):
        hal = models.haldane(t2=0.15, phi=0.7, M=0.2)
        hal.set_onsite({'a': 0.3})
        hr, tb, win, xyz = write_wannier90(hal, self.path('hal'))
        self.assertEqual([os.path.basename(p) for p in (hr, tb, win, xyz)],
                         ['hal_hr.dat', 'hal_tb.dat', 'hal.win', 'hal_centres.xyz'])
        back = read_wannier90(hr, win=win, centres=xyz, dim=2)
        self.assert_same_bands(hal, back)
        np.testing.assert_allclose(back.orbital_positions(), hal.orbital_positions(), atol=1e-11)
        self.assertEqual(len(back._hop), len(hal._hop))
        # the cell of a 2D model: a3 along z, as long as the longest primitive vector
        cell = read_win_cell(win)
        np.testing.assert_allclose(cell[2], [0., 0., np.sqrt(3.)], atol=1e-11)
        back = read_wannier90_tb(tb, dim=2)
        self.assert_same_bands(hal, back)
        self.assertEqual(back._position_hop, [])  # the centres only
        np.testing.assert_allclose(back.orbital_positions(), hal.orbital_positions(), atol=1e-11)
        # a spinful 3D model, orbital by orbital; its onsite spin blocks become R = 0 hoppings
        ti = models.topological_insulator_3d()
        ti._onsite_offdiag[0, 1] = ti._onsite_offdiag[1, 0] = 0.1
        hr = write_hr(ti, self.path('ti_hr.dat'))
        back = read_wannier90(hr, prim_vec=ti.lat.prim_vec, positions=ti.orbital_positions())
        self.assertFalse(back.spin)
        self.assert_same_bands(ti, back)
        num_wann, R, ndegen, ham = read_hr(hr)
        self.assertEqual(num_wann, ti.norb)
        self.assertTrue(np.all(ndegen == 1))
        self.assertEqual(len(R), len(np.unique(np.concatenate([R, -R]), axis=0)))  # every R with its opposite
        self.assert_same_bands(ti, read_wannier90_tb(write_tb(ti, self.path('ti_tb.dat'))))
        # a value function is written with the parameters of set_params
        chain = KSpace(Lattice([{'tag': 'a', 'r0': (0., 0.)}, {'tag': 'b', 'r0': (0.5, 0.)}],
                               [(1., 0.), (0., 1.)]))
        chain.set_hopping([{'i': 0, 'j': 1, 'R': (0, 0), 't': lambda si, sj, w: w},
                           {'i': 1, 'j': 0, 'R': (1, 0), 't': 1.}])
        chain.set_params(w=0.4)
        self.assert_same_bands(chain, read_wannier90(write_hr(chain, self.path('c_hr.dat')),
                                                     prim_vec=[(1., 0.), (0., 1.)], dim=2))

    def test_export_errors(self):
        line = KSpace(Lattice([{'tag': 'a', 'r0': (0., 0.)}], [(1., 0.)]))
        self.assertRaises(ValueError, write_hr, line, self.path('x'))
        slab = models.kane_mele()
        slab3 = KSpace(Lattice([{'tag': 'a', 'r0': (0., 0., 0.)}], [(1., 0., 0.), (0., 1., 0.)]))
        self.assertRaises(ValueError, write_tb, slab3, self.path('x'))
        gra = models.haldane(t2=0.)
        gra.set_overlap([{'n': 1, 't': 0.1}])
        self.assertRaises(ValueError, write_wannier90, gra, self.path('x'))
        self.assertRaises(TypeError, write_hr, slab, 1)
        self.assertRaises(TypeError, write_tb, slab, 1)
        self.assertRaises(TypeError, write_wannier90, slab, 1)
        self.assertRaises(TypeError, write_hr, 'model', self.path('x'))

    def test_position_elements(self):
        r'''
        A model whose position operator has elements beyond the centres
        (*rotated*): with them, the responses of positions=True are those of
        the original model, at every filling; with the centres alone, those
        of a metal differ.
        '''
        hal = models.haldane(t2=0.15, phi=np.pi / 2, M=0.1)
        new = rotated(hal)
        tb = write_tb(new, self.path('rot_tb.dat'))
        cell, R, ndegen, ham, pos = read_tb(tb)
        np.testing.assert_allclose(cell[:2, :2], np.array(hal.lat.prim_vec), atol=1e-11)
        back = read_wannier90_tb(tb, tags=['a', 'b'], dim=2)
        centres = read_wannier90_tb(tb, dim=2, elements=False)
        self.assertGreater(len(back._position_hop), 100)
        self.assertEqual(centres._position_hop, [])
        stored = load_model(save_model(back, self.path('rot.npz')))
        self.assertEqual(len(stored._position_hop), len(back._position_hop))
        for e_fermi in (1., -1.5):  # in the bands
            expected = hal.hall_conductivity(e_fermi, temperature=0.02, nk=30)
            for ks in (back, stored):
                self.assertAlmostEqual(ks.hall_conductivity(e_fermi, temperature=0.02, nk=30), expected, places=9)
            self.assertGreater(abs(centres.hall_conductivity(e_fermi, temperature=0.02, nk=30) - expected), 0.05)
            # the refinement resamples with the same integrand
            self.assertAlmostEqual(back.hall_conductivity(e_fermi, temperature=0.02, nk=16, refine=3),
                                   hal.hall_conductivity(e_fermi, temperature=0.02, nk=16, refine=3), places=9)
        self.assertAlmostEqual(back.anomalous_nernst_conductivity(0.8, 0.05, nk=24),
                               hal.anomalous_nernst_conductivity(0.8, 0.05, nk=24), places=9)
        self.assertAlmostEqual(back.thermal_hall_conductivity(0.8, 0.05, nk=24),
                               hal.thermal_hall_conductivity(0.8, 0.05, nk=24), places=9)
        omega = np.array([0.5, 2.])
        for component in ('xx', 'xy'):
            np.testing.assert_allclose(
                optical_conductivity(back, omega, e_fermi=1., temperature=0.05, nk=24, component=component),
                optical_conductivity(hal, omega, e_fermi=1., temperature=0.05, nk=24, component=component),
                rtol=0, atol=1e-9)
        # positions=False ignores every position (centres included)
        np.testing.assert_allclose(back.hall_conductivity(1., temperature=0.02, nk=12, positions=False),
                                   centres.hall_conductivity(1., temperature=0.02, nk=12, positions=False))
        # the responses that know the centres only refuse the other elements
        km = models.kane_mele()
        km._position_hop = [(0, 2, np.zeros(2), np.array([0.1, 0.]))]
        self.assertRaises(ValueError, km.spin_hall_conductivity, 0., nk=6)
        km.spin_hall_conductivity(0., nk=6, positions=False)
        for call in (lambda: back.orbital_magnetization(0.5, nk=6),
                     lambda: berry_curvature_dipole(back, 0.5, 0.1, nk=6),
                     lambda: shift_current(back, 1., nk=6)):
            self.assertRaises(ValueError, call)
        back.orbital_magnetization(0.5, nk=6, positions=False)
        berry_curvature_dipole(back, 0.5, 0.1, nk=6, positions=False)
        ti = models.topological_insulator_3d()
        ti._position_hop = [(0, 1, np.zeros(3), np.array([0.1, 0., 0.]))]
        self.assertRaises(ValueError, ti.axion_angle, [0, 1], 'M', [2., 2.1], nk=2)
        # with an overlap, the position elements are refused
        back.set_overlap([{'i': 0, 'j': 1, 'R': (0, 0), 't': 0.05}])
        self.assertRaises(ValueError, back.hall_conductivity, 1., nk=6)

    def test_tb_errors(self):
        hal = models.haldane()
        with open(write_tb(hal, self.path('g_tb.dat'))) as f:
            good = f.read().splitlines()

        def write(name, lines):
            path = self.path(name)
            with open(path, 'w') as f:
                f.write('\n'.join(lines) + '\n')
            return path
        self.assertRaises(ValueError, read_tb, write('a', good[:1] + ['1 0', '0 1 0', '0 0 1'] + good[4:]))
        self.assertRaises(ValueError, read_tb, write('b', good[:4] + ['x'] + good[5:]))
        self.assertRaises(ValueError, read_tb, write('c', good[:-1]))
        # the position blocks must follow the order of the Hamiltonian blocks
        blocks = [p for p, line in enumerate(good) if line == '']
        half = blocks[len(blocks) // 2]
        swapped = good[:half + 1] + ['   99    0    0'] + good[half + 2:]
        self.assertRaises(ValueError, read_tb, write('d', swapped))
        # no R = 0 (the vectors shifted by one cell along a3)
        shifted = [line if not (p > blocks[0] and p - 1 in blocks) else
                   line[:10] + '{:5d}'.format(int(line[10:15]) + 1) for p, line in enumerate(good)]
        self.assertRaises(ValueError, read_wannier90_tb, write('e', shifted), dim=3)
        # a 2D model must not reach along a3: H and the positions alike
        self.assertRaises(ValueError, read_wannier90_tb, write('f', shifted), dim=2)
        self.assertRaises(TypeError, read_tb, 1)
        tb = self.path('g_tb.dat')
        self.assertRaises(TypeError, read_wannier90_tb, 1)
        self.assertRaises(ValueError, read_wannier90_tb, tb, dim=1)
        self.assertRaises(ValueError, read_wannier90_tb, tb, cutoff=-1.)
        self.assertRaises(ValueError, read_wannier90_tb, tb, tol=0.)
        self.assertRaises(TypeError, read_wannier90_tb, tb, elements=1)
        self.assertRaises(ValueError, read_wannier90_tb, tb, tags=['a'])


if __name__ == '__main__':
    unittest.main()
