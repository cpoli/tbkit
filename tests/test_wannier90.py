'''
tbkit.io.read_wannier90: hand-written _hr.dat files whose bands are known
analytically (degeneracy weights included).
'''
import itertools
import os
import tempfile
import unittest

import numpy as np

from tbkit.io import read_wannier90, read_hr, read_win_cell, read_centres, BOHR


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
        self.assertRaises(TypeError, read_wannier90, good, prim_vec=pv, tags=['ab'])
        self.assertRaises(ValueError, read_wannier90, good, prim_vec=pv, tags=['a', 'b'])
        # a 2D model must not hop along R3, and its cell must be planar
        self.assertRaises(ValueError, read_wannier90, good, prim_vec=[(1., 0.), (0., 1.)], dim=2)
        tilted = self.write('t.win', 'begin unit_cell_cart\n1 0 0.5\n0 1 0\n0 0 1\nend unit_cell_cart\n')
        self.assertRaises(ValueError, read_wannier90, good, win=tilted, dim=2)


if __name__ == '__main__':
    unittest.main()
