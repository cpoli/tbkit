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


if __name__ == '__main__':
    unittest.main()
