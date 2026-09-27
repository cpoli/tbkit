"""
Surface spectral functions of semi-infinite crystals (KSpace.surface_spectral_function).
"""
import unittest

import numpy as np

import tbkit.lattices as lattices
from tbkit.lattice import Lattice
from tbkit.kspace import KSpace, PAULI, ribbon

CUBIC = [(1., 0., 0.), (0., 1., 0.), (0., 0., 1.)]
SIGMA = [PAULI['x'], PAULI['y'], PAULI['z']]
# Haldane-like model with a third-neighbour-cell bond along a1, so the
# principal layer normal to a2 is one cell and normal to a1 two cells
T2 = 0.2j
HONEY_HOPS = [{'i': 0, 'j': 1, 'R': (0, 0), 't': 1.}, {'i': 0, 'j': 1, 'R': (-1, 0), 't': 1.},
                        {'i': 0, 'j': 1, 'R': (0, -1), 't': 1.},
                        {'i': 0, 'j': 0, 'R': (1, 0), 't': T2}, {'i': 0, 'j': 0, 'R': (0, -1), 't': T2},
                        {'i': 0, 'j': 0, 'R': (-1, 1), 't': T2},
                        {'i': 1, 'j': 1, 'R': (1, 0), 't': -T2}, {'i': 1, 'j': 1, 'R': (0, -1), 't': -T2},
                        {'i': 1, 'j': 1, 'R': (-1, 1), 't': -T2},
                        {'i': 0, 'j': 0, 'R': (2, 0), 't': 0.1}]
HONEY_ONSITE = {'a': 0.3, 'b': -0.3}


def chain(t=1.):
    ks = KSpace(lattices.chain())
    ks.set_hopping([{'i': 0, 'j': 0, 'R': (1,), 't': t}])
    return ks


def ti_hops():
    '''Hoppings of the cubic 3D topological insulator m(k) tau_z + sum_i sin k_i tau_x sigma_i.'''
    hops = []
    for d in range(3):
        R = tuple(int(x) for x in np.eye(3, dtype=int)[d])
        hops += [{'i': 0, 'j': 0, 'R': R, 't': PAULI['0']}, {'i': 1, 'j': 1, 'R': R, 't': -PAULI['0']},
                     {'i': 0, 'j': 1, 'R': R, 't': -0.5j * SIGMA[d]},
                     {'i': 1, 'j': 0, 'R': R, 't': -0.5j * SIGMA[d]}]
    return hops


def ti_onsite(M):
    return {'s': (M - 6) * PAULI['0'], 'p': -(M - 6) * PAULI['0']}


TI_LAT = Lattice(unit_cell=[{'tag': 's', 'r0': (0., 0., 0.)}, {'tag': 'p', 'r0': (0., 0., 0.)}],
                         prim_vec=CUBIC)


def topological_insulator(M):
    ks = KSpace(TI_LAT, spin=True)
    ks.set_hopping(ti_hops())
    ks.set_onsite(ti_onsite(M))
    return ks


def slab_spectral(H, rows, energies, eta):
    '''-Im Tr G / pi over the orbitals *rows* of a finite slab Hamiltonian.'''
    n = len(H)
    return np.array([-np.trace(np.linalg.inv((e + 1j * eta) * np.eye(n) - H)[np.ix_(rows, rows)]).imag
                             for e in energies]) / np.pi


class TestSurfaceSpectralFunction(unittest.TestCase):

    def test_chain_end_and_bulk(self):
        # semi-infinite chain: A_s = sqrt(4 - E^2)/(2 pi), A_b = 1/(pi sqrt(4 - E^2)),
        # including E = 0, where the cell Hamiltonian is singular
        ch = chain()
        E = np.linspace(-1.9, 1.9, 9)
        for eta in (1e-6, 1e-9):
            A = ch.surface_spectral_function([[0.]], E, 0, eta=eta)[0]
            self.assertTrue(np.allclose(A, np.sqrt(4 - E ** 2) / (2 * np.pi), atol=1e-6))
            B = ch.surface_spectral_function([[0.]], E, 0, eta=eta, bulk=True)[0]
            self.assertTrue(np.allclose(B, 1 / (np.pi * np.sqrt(4 - E ** 2)), atol=1e-5))
        # both ends alike; nothing outside the band; one number for a scalar energy
        self.assertTrue(np.allclose(ch.surface_spectral_function([[0.]], E, 0, side=-1, eta=1e-6)[0],
                                                np.sqrt(4 - E ** 2) / (2 * np.pi), atol=1e-6))
        self.assertLess(ch.surface_spectral_function([[0.]], 2.5, 0, eta=1e-6)[0, 0], 1e-5)
        self.assertEqual(ch.surface_spectral_function([0.], 0.5, 0).shape, (1, 1))

    def test_spectral_sum_rule(self):
        # int A_s dE = number of orbitals of the surface cell
        ch = chain()
        E = np.linspace(-60., 60., 24001)
        A = ch.surface_spectral_function([[0.]], E, 0, eta=0.05)[0]
        self.assertAlmostEqual(np.sum(A) * (E[1] - E[0]), 1., delta=2e-3)

    def test_matches_thick_ribbon(self):
        # the outer cell of a wide ribbon (eta resolves ~ 1/eta cells) is the
        # semi-infinite surface, on both edges, normal to either primitive vector
        lat = lattices.honeycomb()
        ks = KSpace(lat)
        ks.set_hopping(HONEY_HOPS)
        ks.set_onsite(HONEY_ONSITE)
        E = np.linspace(-3., 3., 13)
        eta, width = 0.05, 200
        k = np.array([0.37, 0.61]) @ ks.rec_vec_k
        for direction in (0, 1):
            rib = ribbon(lat, HONEY_HOPS, width, direction=direction, onsite=HONEY_ONSITE)
            a_par = np.array(lat.prim_vec[1 - direction])
            H = rib.get_ham([k @ a_par / np.linalg.norm(a_par)])
            for side, rows in ((1, [0, 1]), (-1, [len(H) - 2, len(H) - 1])):
                A = ks.surface_spectral_function([k], E, direction, side, eta=eta)[0]
                self.assertTrue(np.allclose(A, slab_spectral(H, rows, E, eta), atol=1e-10))
                # only k along the surface matters
                A2 = ks.surface_spectral_function([k + 0.3 * ks.rec_vec_k[direction]], E, direction,
                                                                   side, eta=eta)[0]
                self.assertTrue(np.allclose(A, A2, atol=1e-12))

    def test_matches_thick_slab_3d(self):
        # a spinful 3D model against slabs cut normal to a2 (slab k = (k_x, k_z))
        # and to a3 (slab k = (k_x, k_y)): the four orbitals of the outer cell
        ti = topological_insulator(2.)
        E, eta = np.linspace(-3., 3., 13), 0.1
        for direction, k_slab, k in ((1, [0.4, -0.3], [0.4, 1.3, -0.3]), (2, [0.2, -0.1], [0.2, -0.1, 0.])):
            slab = ribbon(TI_LAT, ti_hops(), 120, direction=direction, onsite=ti_onsite(2.), spin=True)
            A = ti.surface_spectral_function([k], E, direction, eta=eta)[0]
            ref = slab_spectral(slab.get_ham(k_slab), [0, 1, 2, 3], E, eta)
            self.assertTrue(np.allclose(A, ref, atol=1e-10))

    def test_topological_insulator_surface_dirac_cone(self):
        kx = np.linspace(-0.6, 0.6, 121)
        ks = np.column_stack([kx, 0 * kx, 0 * kx])
        E = np.array([-0.4, 0., 0.2, 0.4])
        ti = topological_insulator(2.)
        A = ti.surface_spectral_function(ks, E, 2, eta=0.02)
        bulk = ti.surface_spectral_function(ks, E, 2, eta=0.02, bulk=True)
        self.assertLess(bulk.max(), 0.05)  # all four energies in the bulk gap
        self.assertGreater(A[60, 1], 10.)  # the Dirac point at the zone centre
        for n, e in ((2, 0.2), (3, 0.4), (0, -0.4)):
            # exactly two peaks, at k = +-|E| (velocity 1)
            col = A[:, n]
            peaks = [i for i in range(1, len(kx) - 1)
                         if col[i] > col[i - 1] and col[i] > col[i + 1] and col[i] > 1.]
            self.assertEqual(len(peaks), 2)
            self.assertTrue(np.allclose(np.abs(kx[peaks]), abs(e), atol=0.02))
        # no surface states in the gap of the trivial insulator
        trivial = topological_insulator(-2.)
        self.assertLess(trivial.surface_spectral_function(ks, E, 2, eta=0.02).max(), 0.05)

    def test_non_reciprocal_chain(self):
        # Hatano-Nelson: the surface Green's function obeys g = 1/(z - tR tL g),
        # that of a Hermitian chain with t = sqrt(tR tL)
        hn = KSpace(lattices.chain())
        hn.set_hopping([{'i': 0, 'j': 0, 'R': (1,), 't': 2.}, {'i': 0, 'j': 0, 'R': (-1,), 't': 0.5}],
                              hermitian=False)
        E = np.linspace(-1.5, 1.5, 7)
        self.assertTrue(np.allclose(hn.surface_spectral_function([[0.]], E, 0, eta=1e-3),
                                                chain().surface_spectral_function([[0.]], E, 0, eta=1e-3)))

    def test_decoupled_layers(self):
        # no hopping along the surface normal: each layer is an isolated chain
        lat = lattices.square()
        ks = KSpace(lat)
        ks.set_hopping([{'i': 0, 'j': 0, 'R': (1, 0), 't': 1.}])
        kx, E = 0.7, np.linspace(-3., 3., 7)
        A = ks.surface_spectral_function([[kx, 0.]], E, 1, eta=0.1)[0]
        e0 = 2 * np.cos(kx)
        self.assertTrue(np.allclose(A, 0.1 / np.pi / ((E - e0) ** 2 + 0.01)))

    def test_errors(self):
        ch = chain()
        f = ch.surface_spectral_function
        self.assertRaises(ValueError, f, [[0., 0.]], 0., 0)
        self.assertRaises(TypeError, f, [[0.]], [0.1j], 0)
        self.assertRaises(ValueError, f, [[0.]], [], 0)
        self.assertRaises(ValueError, f, [[0.]], 0., 1)
        self.assertRaises(TypeError, f, [[0.]], 0., 0.)
        self.assertRaises(ValueError, f, [[0.]], 0., 0, 0)
        self.assertRaises(ValueError, f, [[0.]], 0., 0, True)
        self.assertRaises(ValueError, f, [[0.]], 0., 0, 1, -1.)
        self.assertRaises(TypeError, f, [[0.]], 0., 0, 1, 0.1, 1)
        self.assertRaises(RuntimeError, f, [[0.]], 0.5, 0, 1, 1e-9, False, 2)
        ch.set_overlap([{'i': 0, 'j': 0, 'R': (1,), 't': 0.1}])
        self.assertRaises(ValueError, f, [[0.]], 0., 0)


if __name__ == '__main__':
    unittest.main()
