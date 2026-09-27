"""
Validators of the higher-order topology, moire/supercell and
self-consistent interaction modules (the last section of error_handling.py).
"""
import unittest

import numpy as np

import tbkit.error_handling as eh


class TestHigherOrder(unittest.TestCase):

    def test_wannier_gap(self):
        nu = np.zeros((4, 3, 2))
        nu[..., 0], nu[..., 1] = -0.3, 0.3
        eh.wannier_gap(nu, [0])
        eh.wannier_gap(nu, [0, 1])  # the whole group: nothing to check
        touching = nu.copy()
        touching[1, 1] = 0.
        self.assertRaises(ValueError, eh.wannier_gap, touching, [0])
        # a sector that jumps across the circle between neighbouring k
        wrapped = nu.copy()
        wrapped[2, :, 0] = 0.45
        wrapped[2, :, 1] = 0.3
        self.assertRaises(ValueError, eh.wannier_gap, wrapped, [0])


class TestMoire(unittest.TestCase):

    def test_supercell_matrix(self):
        eh.supercell_matrix(np.array([[2, 1], [0, 1]]), 2)
        self.assertRaises(ValueError, eh.supercell_matrix, np.array([[2]]), 2)
        self.assertRaises(TypeError, eh.supercell_matrix, np.array([[2., 0.], [0., 1.]]), 2)
        self.assertRaises(ValueError, eh.supercell_matrix, np.array([[1, 1], [1, 1]]), 2)

    def test_supercell_model(self):
        eh.supercell_model(1, int)
        self.assertRaises(TypeError, eh.supercell_model, 1., int)

    def test_bond_vectors(self):
        eh.bond_vectors(np.ones((2, 3)))
        self.assertRaises(ValueError, eh.bond_vectors, np.ones((2, 2)))
        self.assertRaises(ValueError, eh.bond_vectors, np.zeros((1, 3)))

    def test_twist(self):
        eh.twist_lattice('square')
        self.assertRaises(ValueError, eh.twist_lattice, 'kagome')
        eh.commensurate(np.eye(2), np.eye(2))
        self.assertRaises(ValueError, eh.commensurate, np.eye(2), 1.01 * np.eye(2))
        eh.hopping_values(np.zeros(3), 3)
        self.assertRaises(ValueError, eh.hopping_values, np.zeros(2), 3)


if __name__ == '__main__':
    unittest.main()
