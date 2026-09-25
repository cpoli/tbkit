"""
Finite samples of KSpace models, non-reciprocal hoppings, the spectral
winding number, and the generalized Brillouin zone.
"""
import unittest

import numpy as np

import tbkit.lattices as lattices
from tbkit.kspace import KSpace, PAULI
from tbkit.lattice import Lattice
from tbkit.system import System


def hatano_nelson(t_right=1., t_left=0.5):
    '''(H psi)_n = t_right psi_{n+1} + t_left psi_{n-1}'''
    hn = KSpace(lattices.chain())
    hn.set_hopping([{'i': 0, 'j': 0, 'R': (1,), 't': t_right},
                             {'i': 0, 'j': 0, 'R': (-1,), 't': t_left}], hermitian=False)
    return hn


class TestFiniteHam(unittest.TestCase):

    def test_open_chain_matches_system(self):
        ssh = KSpace(Lattice(unit_cell=[{'tag': 'a', 'r0': (0., 0.)}, {'tag': 'b', 'r0': (0.5, 0.)}],
                                        prim_vec=[(1., 0.)]))
        ssh.set_hopping([{'i': 0, 'j': 1, 'R': (0,), 't': 0.4}, {'i': 1, 'j': 0, 'R': (1,), 't': 1.}])
        ssh.set_onsite({'a': 0.1, 'b': -0.1})
        ham = ssh.finite_ham(10)
        self.assertEqual(ham.shape, (20, 20))
        self.assertTrue(np.allclose(ham, ham.conj().T))
        # the same open chain from System
        lat = Lattice(unit_cell=[{'tag': 'a', 'r0': (0., 0.)}, {'tag': 'b', 'r0': (0.5, 0.)}],
                             prim_vec=[(1., 0.)])
        lat.get_lattice(10)
        sys = System(lat)
        sys.set_onsite({'a': 0.1, 'b': -0.1})
        sys.set_hopping_manual({**{(2*n, 2*n+1): 0.4 for n in range(10)},
                                              **{(2*n+1, 2*n+2): 1. for n in range(9)}})
        sys.get_ham()
        sys.get_eig()
        self.assertTrue(np.allclose(np.linalg.eigvalsh(ham), sys.en))

    def test_periodic_matches_bloch(self):
        sq = KSpace(lattices.square())
        sq.set_hopping([{'i': 0, 'j': 0, 'R': (1, 0), 't': 1.}, {'i': 0, 'j': 0, 'R': (0, 1), 't': 1.}])
        e_torus = np.sort(np.linalg.eigvalsh(sq.finite_ham((4, 3), periodic=True)))
        fracs = np.array(np.meshgrid(np.arange(4)/4, np.arange(3)/3, indexing='ij')).reshape(2, -1).T
        self.assertTrue(np.allclose(e_torus, np.sort(sq.get_bands(fracs @ sq.rec_vec_k).ravel())))
        # spinful blocks, and the off-diagonal onsite terms, per cell
        spin = KSpace(lattices.chain(), spin=True)
        spin.set_onsite({'a': 0.3*PAULI['x']})
        self.assertTrue(np.allclose(spin.finite_ham(3), np.kron(np.eye(3), 0.3*PAULI['x'])))
        self.assertRaises(TypeError, sq.finite_ham, 3, 1)
        self.assertRaises(ValueError, sq.finite_ham, (3,))


class TestNonReciprocal(unittest.TestCase):

    def test_hermitian_flag(self):
        hn = hatano_nelson()
        self.assertFalse(hn.is_hermitian())
        self.assertEqual(hn.get_ham([0.])[0, 0], 1.5)
        self.assertTrue(np.allclose(hn.get_ham([np.pi/2])[0, 0], 0.5j))
        hn.clear_hopping()
        self.assertTrue(hn.is_hermitian())
        self.assertRaises(TypeError, hn.set_hopping, [{'i': 0, 'j': 0, 'R': (1,), 't': 1.}], 1)
        spin = KSpace(lattices.chain(), spin=True)
        spin.set_hopping([{'i': 0, 'j': 0, 'R': (1,), 't': PAULI['x']}], hermitian=False)
        self.assertTrue(np.allclose(spin.get_ham([0.]), PAULI['x']))

    def test_skin_effect(self):
        hn = hatano_nelson(1., 0.5)
        # periodic chain: an ellipse winding once around E = 0 ...
        self.assertAlmostEqual(hn.spectral_winding(0.), 1., places=8)
        self.assertAlmostEqual(hatano_nelson(0.5, 1.).spectral_winding(0.), -1., places=8)
        self.assertAlmostEqual(hn.spectral_winding(3.), 0., places=8)  # outside the ellipse
        # ... while the open chain has a real spectrum inside [-2 sqrt(t t'), 2 sqrt(t t')],
        # with every eigenstate piled up at one end
        ham = hn.finite_ham(40)
        en, vec = np.linalg.eig(ham)
        self.assertLess(np.max(np.abs(en.imag)), 1e-8)
        self.assertLess(np.max(np.abs(en.real)), 2*np.sqrt(0.5))
        weight = np.abs(vec) ** 2 / np.sum(np.abs(vec) ** 2, axis=0)
        self.assertGreater(np.min(weight[:10].sum(axis=0)), 0.9)

    def test_gbz(self):
        hn = hatano_nelson(1., 0.5)
        en = np.linalg.eigvals(hn.finite_ham(40))
        beta = hn.gbz(en)
        self.assertEqual(beta.shape, (40, 2))
        # a circle of radius sqrt(t'/t), the middle roots of equal modulus
        self.assertTrue(np.allclose(np.abs(beta), np.sqrt(0.5)))
        # and the open-chain spectrum is H(beta) on it
        for e, b in zip(en[:5], beta[:5]):
            self.assertAlmostEqual(hn.get_ham_beta(b[0])[0, 0], e)
        # Hermitian: the unit circle
        chain = KSpace(lattices.chain())
        chain.set_hopping([{'i': 0, 'j': 0, 'R': (1,), 't': 1.}])
        self.assertTrue(np.allclose(np.abs(chain.gbz([0.5, -1.])), 1.))
        # two bands, next-nearest hopping: finite roots on either side
        ssh = KSpace(Lattice(unit_cell=[{'tag': 'a', 'r0': (0., 0.)}, {'tag': 'b', 'r0': (0.5, 0.)}],
                                        prim_vec=[(1., 0.)]))
        ssh.set_hopping([{'i': 0, 'j': 1, 'R': (0,), 't': 1.}, {'i': 1, 'j': 0, 'R': (0,), 't': 0.6},
                                  {'i': 1, 'j': 0, 'R': (1,), 't': 1.}, {'i': 0, 'j': 1, 'R': (-1,), 't': 1.}],
                                 hermitian=False)
        en = np.linalg.eigvals(ssh.finite_ham(60))
        beta = ssh.gbz(en[np.abs(en) > 0.2])
        self.assertTrue(np.allclose(np.abs(beta[:, 0]), np.abs(beta[:, 1]), rtol=0.05))

    def test_checks(self):
        sq = KSpace(lattices.square())
        self.assertRaises(ValueError, sq.get_ham_beta, 1.)
        self.assertRaises(ValueError, sq.spectral_winding)
        self.assertRaises(ValueError, sq.gbz, [0.])
        hn = hatano_nelson()
        self.assertRaises(ValueError, hn.get_ham_beta, 0.)
        self.assertRaises(TypeError, hn.get_ham_beta, 'a')
        self.assertRaises(ValueError, hn.gbz, [])
        one_way = KSpace(lattices.chain())
        one_way.set_hopping([{'i': 0, 'j': 0, 'R': (1,), 't': 1.}], hermitian=False)
        self.assertRaises(ValueError, one_way.gbz, [0.5])


if __name__ == '__main__':
    unittest.main()
