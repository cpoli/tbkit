"""
Landauer transport: surface Green's functions, self-energies, transmission.
"""
import unittest

import numpy as np

import tbkit.lattices as lattices
from tbkit.kspace import KSpace, ribbon
from tbkit.system import System
from tbkit.transport import Transport, lead_from_kspace, surface_green

ONE = np.ones((1, 1))
SQUARE = [{'i': 0, 'j': 0, 'R': (1, 0), 't': 1.}, {'i': 0, 'j': 0, 'R': (0, 1), 't': 1.}]


def chain_device(n=10, impurity=None):
    lat = lattices.chain()
    lat.get_lattice(n)
    sys = System(lat)
    sys.set_hopping([{'n': 1, 't': 1.}])
    sys.set_onsite({'a': 0.})
    if impurity is not None:
        sys.set_onsite_def(impurity)
    sys.get_ham()
    tr = Transport(sys.ham)
    tr.add_lead([[0.]], [[1.]], ONE, [0])
    tr.add_lead([[0.]], [[1.]], ONE, [n - 1])
    return tr


def ribbon_device(width, length, constriction=None):
    '''A square-lattice strip between two leads of the same width.'''
    lat = lattices.square()
    lat.get_lattice(length, width)
    if constriction is not None:
        x0, keep = constriction
        drop = [i for i in range(lat.sites)
                   if abs(lat.coor['x'][i] - x0) < 1e-9 and not keep[0] <= lat.coor['y'][i] <= keep[1]]
        lat.remove_sites(drop)
    sys = System(lat)
    sys.set_hopping([{'n': 1, 't': 1.}])
    sys.get_ham()
    rib = ribbon(lattices.square(), SQUARE, width=width, direction=1)
    tr = Transport(sys.ham)
    for x, direction in ((0., -1), (length - 1., 1)):
        sites = [i for i in range(lat.sites) if abs(lat.coor['x'][i] - x) < 1e-9]
        h0, v = lead_from_kspace(rib, direction)
        tr.add_lead(h0, v, np.eye(width), sites)
    return tr


class TestSurfaceGreen(unittest.TestCase):

    def test_chain(self):
        # semi-infinite chain: g = (E - i sqrt(4 - E^2)) / 2 in the band
        for e in (-1.3, 0., 0.5, 1.9):
            g = surface_green([[0.]], [[1.]], e)[0, 0]
            self.assertAlmostEqual(g, (e - 1j*np.sqrt(4 - e**2)) / 2, places=6)
        # a two-orbital lead (a dimerized chain) with a singular coupling:
        # compare with a long finite chain's end Green's function
        h0 = [[0., 1.], [1., 0.]]
        v = [[0., 0.], [0.5, 0.]]
        e, eta = 0.3, 1e-2
        n = 400
        ham = np.kron(np.eye(n), h0) + np.kron(np.eye(n, k=1), v) + np.kron(np.eye(n, k=-1), np.array(v).T)
        g_end = np.linalg.inv((e + 1j*eta) * np.eye(2*n) - ham)[:2, :2]
        self.assertTrue(np.allclose(surface_green(h0, v, e, eta), g_end, atol=1e-8))
        # outside the band: real, the root that decays
        g = surface_green([[0.]], [[1.]], 3.)[0, 0]
        self.assertAlmostEqual(g, (3 - np.sqrt(5)) / 2, places=6)

    def test_checks(self):
        self.assertRaises(RuntimeError, surface_green, [[0.]], [[1.]], 0.5, 1e-9, 1e-12, 2)
        self.assertRaises(ValueError, surface_green, [[0., 1.]], [[1.]], 0.)
        self.assertRaises(ValueError, surface_green, [[0.]], [[1., 0.], [0., 1.]], 0.)


class TestTransmission(unittest.TestCase):

    def test_perfect_chain(self):
        es = np.linspace(-1.9, 1.9, 7)
        self.assertTrue(np.allclose(chain_device().transmission(es), 1., atol=1e-6))
        self.assertTrue(np.allclose(chain_device().transmission([2.5, -3.]), 0., atol=1e-6))

    def test_impurity(self):
        # T = v^2 / (v^2 + eps^2), v = 2 sin k the velocity at E = 2 cos k
        tr = chain_device(impurity={5: 0.8})
        es = np.array([-1.5, 0.3, 1.2])
        v = 2 * np.sin(np.arccos(es / 2))
        self.assertTrue(np.allclose(tr.transmission(es), v**2 / (v**2 + 0.64), atol=1e-6))
        self.assertTrue(np.allclose(tr.transmission(es, 1, 0), tr.transmission(es, 0, 1)))

    def test_quantized_channels(self):
        # a clean strip transmits one quantum per open transverse mode
        width = 6
        es = np.linspace(-3.8, 3.8, 9)
        modes = [sum(abs(e - 2*np.cos(n*np.pi/(width+1))) < 2 for n in range(1, width+1)) for e in es]
        self.assertTrue(np.allclose(ribbon_device(width, 8).transmission(es), modes, atol=1e-6))
        # a constriction can only lower it
        qpc = ribbon_device(width, 8, constriction=(4., (2., 3.)))
        self.assertTrue(np.all(qpc.transmission(es) <= np.array(modes) + 1e-6))
        self.assertLessEqual(qpc.transmission([0.1])[0], 2. + 1e-6)

    def test_green_and_self_energy(self):
        tr = chain_device(4)
        sigma = tr.self_energy(0, 0.5)
        self.assertAlmostEqual(sigma[0, 0], surface_green([[0.]], [[1.]], 0.5)[0, 0])
        self.assertEqual(np.count_nonzero(sigma), 1)
        # the device plus leads is an infinite perfect chain: G_00 = -i / v
        g = tr.get_green(0.5)
        self.assertAlmostEqual(g[0, 0], -1j / np.sqrt(4 - 0.25), places=6)

    def test_lead_from_kspace(self):
        chain = KSpace(lattices.chain())
        chain.set_hopping([{'i': 0, 'j': 0, 'R': (1,), 't': 2.}])
        chain.set_onsite({'a': 0.3})
        h0, v = lead_from_kspace(chain)
        self.assertTrue(np.allclose(h0, [[0.3]]) and np.allclose(v, [[2.]]))
        _, v_left = lead_from_kspace(chain, -1)
        self.assertTrue(np.allclose(v_left, [[2.]]))
        long = KSpace(lattices.chain())
        long.set_hopping([{'i': 0, 'j': 0, 'R': (2,), 't': 1.}])
        self.assertRaises(ValueError, lead_from_kspace, long)
        self.assertRaises(ValueError, lead_from_kspace, chain, 0)
        self.assertRaises(ValueError, lead_from_kspace, KSpace(lattices.square()))

    def test_checks(self):
        tr = Transport(np.zeros((3, 3)))
        self.assertRaises(ValueError, Transport, np.zeros((2, 3)))
        self.assertRaises(TypeError, tr.add_lead, ONE, ONE, ONE, 0)
        self.assertRaises(ValueError, tr.add_lead, ONE, ONE, ONE, [3])
        self.assertRaises(ValueError, tr.add_lead, ONE, ONE, ONE, [0, 0])
        self.assertRaises(ValueError, tr.add_lead, ONE, ONE, np.ones((2, 1)), [0])
        self.assertRaises(ValueError, tr.self_energy, 0, 0.)
        tr.add_lead(ONE, ONE, ONE, [0])
        self.assertRaises(TypeError, tr.self_energy, 0., 0.)


if __name__ == '__main__':
    unittest.main()
