"""
Magnetic unit cells: Hofstadter subbands in reciprocal space, and their
TKNN Chern numbers.
"""
import unittest
import numpy as np

import tbkit.lattices as lattices
from tbkit.kspace import magnetic_supercell, PAULI

SQUARE = [{'i': 0, 'j': 0, 'R': (1, 0), 't': 1.}, {'i': 0, 'j': 0, 'R': (0, 1), 't': 1.}]
HONEYCOMB = [{'i': 0, 'j': 1, 'R': R, 't': 1.} for R in [(0, 0), (-1, 0), (0, -1)]]


def tknn(p, q, r):
    '''The TKNN solution t_r of r = q s_r + p t_r, |t_r| <= q/2.'''
    for t in range(-q, q + 1):
        if (r - p * t) % q == 0 and abs(t) <= q / 2:
            return t


class TestMagneticSupercell(unittest.TestCase):

    def test_half_flux_analytic(self):
        # alpha = 1/2 on the square lattice: E = +-2 sqrt(cos^2 k1 + cos^2 k2)
        # in the magnetic zone, with k1 along the doubled cell
        mag = magnetic_supercell(lattices.square(), SQUARE, 1, 2)
        for k in ([0.3, 0.2], [1.1, -0.4], [0., 0.]):
            en = np.linalg.eigvalsh(mag.get_ham(k))
            exact = 2 * np.sqrt(np.cos(k[0]) ** 2 + np.cos(k[1]) ** 2)
            self.assertTrue(np.allclose(en, [-exact, exact]))

    def test_tknn_diophantine(self):
        # Chern numbers of the isolated Hofstadter bands of the square
        # lattice: C_r = t_r - t_{r-1}, from r = q s_r + p t_r
        for p, q in [(1, 3), (2, 5), (1, 5), (3, 7)]:
            mag = magnetic_supercell(lattices.square(), SQUARE, p, q)
            chern = [mag.chern_number([n], nk=20) for n in range(q)]
            expect = [tknn(p, q, r + 1) - tknn(p, q, r) for r in range(q)]
            self.assertTrue(np.allclose(chern, expect, atol=1e-6), msg='{}/{}'.format(p, q))
            # reversing the field reverses every Chern number
            rev = magnetic_supercell(lattices.square(), SQUARE, -p, q)
            self.assertAlmostEqual(rev.chern_number([0], nk=20), -chern[0], places=6)

    def test_gauge_consistency(self):
        # the same flux in a twice larger magnetic cell gives the same
        # spectrum, for sites off the cell origin too (honeycomb)
        for lat, hop in ((lattices.square(), SQUARE), (lattices.honeycomb(), HONEYCOMB)):
            a = magnetic_supercell(lat, hop, 1, 3).mesh_bands((6, 6)).ravel()
            b = magnetic_supercell(lat, hop, 2, 6).mesh_bands((3, 6)).ravel()
            self.assertTrue(np.allclose(np.sort(a), np.sort(b)))
            mag = magnetic_supercell(lat, hop, 1, 4)
            ham = mag.get_ham([0.2, 0.9])
            self.assertTrue(np.allclose(ham, ham.conj().T))
            self.assertAlmostEqual(mag.chern_number(list(range(mag.norb)), nk=10), 0., places=8)

    def test_zero_flux_and_options(self):
        mag = magnetic_supercell(lattices.honeycomb(), HONEYCOMB, 0, 2, onsite={'a': 0.1, 'b': -0.1})
        self.assertEqual(mag.norb, 4)
        # zero flux: the folded graphene bands, top of the band at Gamma
        self.assertTrue(np.isclose(np.linalg.eigvalsh(mag.get_ham([0., 0.])).max(),
                                             np.sqrt(9 + 0.01)))
        spin = magnetic_supercell(lattices.square(), [dict(h, t=h['t']*PAULI['0']) for h in SQUARE],
                                             1, 3, spin=True)
        self.assertEqual(spin.norb, 6)
        self.assertRaises(TypeError, magnetic_supercell, lattices.square(), SQUARE, 0.5, 3)
        self.assertRaises(ValueError, magnetic_supercell, lattices.square(), SQUARE, 1, 0)
        self.assertRaises(ValueError, magnetic_supercell, lattices.chain(),
                                [{'i': 0, 'j': 0, 'R': (1,), 't': 1.}], 1, 3)


if __name__ == '__main__':
    unittest.main()
