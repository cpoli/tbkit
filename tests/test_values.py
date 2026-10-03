"""
Parametrized Hamiltonians: hoppings and onsite energies given as value
functions ``t(site_i, site_j, **params)`` / ``onsite(site, **params)``,
evaluated by ``get_ham(**params)``.
"""
import os
import tempfile
import unittest

import numpy as np

from tbkit import lattices
from tbkit.bdg import bdg_kspace
from tbkit.graphene import GrapheneLattice, GrapheneSystem
from tbkit.io import save_model
from tbkit.kspace import KSpace, PAULI, ribbon, magnetic_supercell
from tbkit.orbital import OrbitalSystem
from tbkit.system import System


def square(n1=5, n2=4):
    lat = lattices.square()
    lat.get_lattice(n1=n1, n2=n2)
    return lat


def dense(sys):
    return sys.ham.toarray()


class TestSystemValues(unittest.TestCase):

    def test_matches_numbers(self):
        lat = square()
        ref = System(lat)
        ref.set_hopping([{'n': 1, 'ang': 0., 't': -1.}, {'n': 1, 'ang': 90., 't': -0.5}])
        ref.set_onsite({'a': 0.3})
        ref.get_ham()
        sys = System(lat)
        sys.set_hopping([{'n': 1, 'ang': 0., 't': lambda si, sj, tx: tx},
                                 {'n': 1, 'ang': 90., 't': lambda si, sj, ty=-0.5: ty}])
        sys.set_onsite({'a': lambda site, V: V})
        # each function gets only the parameters its signature names
        sys.get_ham(tx=-1., V=0.3, unused=7.)
        np.testing.assert_allclose(dense(sys), dense(ref))

    def test_sites_vectorized(self):
        lat = square()
        sys = System(lat)
        calls = []

        def t(si, sj, **params):
            calls.append(params)
            return -np.exp(-si['x'] / params['xi'])
        sys.set_hopping([{'n': 1, 't': t}])
        sys.set_onsite({'a': lambda site, V: V * site['x'] + site['index']})
        sys.get_ham(xi=2., V=0.1)
        self.assertEqual(calls, [{'xi': 2., 'V': 0.1}])  # one call for every bond, all params
        h = dense(sys)
        x = lat.coor['x']
        np.testing.assert_allclose(np.diag(h), 0.1 * x + np.arange(lat.sites))
        for i, j in zip(sys.hop['i'], sys.hop['j']):
            self.assertAlmostEqual(h[i, j], -np.exp(-x[i] / 2.))
            self.assertAlmostEqual(h[j, i], -np.exp(-x[i] / 2.))

    def test_sweep(self):
        sys = System(square())
        sys.set_hopping([{'n': 1, 't': -1.}])
        sys.set_onsite({'a': lambda site, V: V})
        shifts = []
        for V in (0., 0.5, 1.):
            sys.get_ham(V=V)
            sys.get_eig()
            shifts.append(sys.en[0])
        np.testing.assert_allclose(np.diff(shifts), [0.5, 0.5])

    def test_modifiers_compose(self):
        lat = square()
        ref = System(lat)
        ref.set_hopping([{'n': 1, 't': -1.}])
        ref.set_magnetic_field(0.1)
        ref.set_onsite({'a': 0.})
        ref.set_onsite_def({0: 2.})
        ref.get_ham()
        sys = System(lat)
        sys.set_hopping([{'n': 1, 't': lambda si, sj, t: t}])
        sys.set_magnetic_field(0.1)  # multiplies the value function
        sys.set_onsite({'a': lambda site, V: V * np.ones(len(site))})
        sys.set_onsite_def({0: 2.})  # replaces it on site 0
        sys.get_ham(t=-1., V=0.)
        np.testing.assert_allclose(dense(sys), dense(ref))
        sys.set_onsite_dis(0.1)  # adds to it
        sys.get_ham(t=-1., V=1.)
        diag = np.diag(dense(sys)).real
        self.assertAlmostEqual(diag[0], sys.onsite[0].real)
        np.testing.assert_allclose(diag[1:], 1. + sys.onsite[1:].real)

    def test_override(self):
        lat = square()
        ref = System(lat)
        ref.set_hopping([{'n': 1, 't': -1.}])
        ref.get_ham()
        sys = System(lat)
        sys.set_hopping([{'n': 1, 't': lambda si, sj: 5.}])
        sys.set_hopping([{'n': 1, 't': -1.}])  # a number replaces the function
        self.assertEqual(sys._hop_values, [])
        sys.get_ham()
        np.testing.assert_allclose(dense(sys), dense(ref))
        sys.set_hopping([{'n': 1, 't': lambda si, sj: 5.}])
        sys.set_hopping([{'n': 1, 'ang': 0., 't': lambda si, sj: -1.}])
        sys.set_hopping([{'n': 1, 'ang': 90., 't': lambda si, sj: -1.}])
        sys.get_ham()
        np.testing.assert_allclose(dense(sys), dense(ref))
        self.assertEqual(len(sys._hop_values), 2)
        # set_hopping_def, given either orientation of a bond, replaces it
        i, j = int(sys.hop['i'][0]), int(sys.hop['j'][0])
        sys.set_hopping_def({(j, i): 2j})
        sys.get_ham()
        self.assertAlmostEqual(dense(sys)[j, i], 2j)
        sys.set_hopping_manual({(i, j): 0.5})  # an extra (numeric) row
        sys.clear_hopping()
        self.assertEqual(sys._hop_values, [])

    def test_tag_lower_part_non_hermitian(self):
        lat = lattices.chain()
        lat.unit_cell = [{'tag': 'a', 'r0': (0., 0.)}, {'tag': 'b', 'r0': (0.5, 0.)}]
        lat.prim_vec = [(1., 0.)]
        lat.get_lattice(n1=4)
        ref = System(lat)
        ref.set_hopping([{'n': 1, 'tag': 'ab', 't': 1.}, {'n': 1, 'tag': 'ba', 't': 2.}])
        ref.set_hopping([{'n': 1, 'tag': 'ab', 't': 0.5}, {'n': 1, 'tag': 'ba', 't': 3.}], upper_part=False)
        ref.get_ham()
        sys = System(lat)
        sys.set_hopping([{'n': 1, 'tag': 'ab', 't': lambda si, sj, g: g},
                                 {'n': 1, 'tag': 'ba', 't': lambda si, sj, g: 2 * g}])
        sys.set_hopping([{'n': 1, 'ang': -180., 'tag': 'ab', 't': lambda si, sj, g: 0.5 * g},
                                 {'n': 1, 'tag': 'ba', 't': lambda si, sj, g: 3 * g}], upper_part=False)
        sys.get_ham(g=1.)
        np.testing.assert_allclose(dense(sys), dense(ref))

    def test_change_hopping_region(self):
        lat = square(6, 6)
        ref = System(lat)
        ref.set_hopping([{'n': 1, 't': -1.}])
        ref.change_hopping_square([{'n': 1, 't': -2.}], xlims=[0., 2.], ylims=[0., 2.])
        ref.get_ham()
        sys = System(lat)
        sys.set_hopping([{'n': 1, 't': -1.}])
        sys.change_hopping_square([{'n': 1, 't': lambda si, sj, t: t}], xlims=[0., 2.], ylims=[0., 2.])
        sys.get_ham(t=-2.)
        np.testing.assert_allclose(dense(sys), dense(ref))
        # the last dictionary selecting a bond wins, function or number
        sys.change_hopping_ellipse([{'n': 1, 'tag': 'aa', 't': -1.},
                                                  {'n': 1, 'ang': 0., 't': lambda si, sj: -3.},
                                                  {'n': 1, 'ang': 90., 'tag': 'aa', 't': -1.}], rx=1.5, ry=1.5)
        self.assertEqual(len(sys._hop_values), 2)
        sys.get_ham(t=-2.)
        h = dense(sys)
        self.assertAlmostEqual(h[0, 1], -3.)  # (0, 0)-(1, 0): angle 0, in the ellipse
        self.assertAlmostEqual(h[0, 6], -1.)  # (0, 0)-(0, 1): angle 90, in the ellipse
        self.assertAlmostEqual(h[1, 2], -2.)  # (1, 0)-(2, 0): outside, in the square
        self.assertAlmostEqual(h[4, 5], -1.)  # (4, 0)-(5, 0): outside both

    def test_orbital_system(self):
        lat = square()
        ref = OrbitalSystem(lat, spin=True)
        ref.set_hopping([{'n': 1, 't': -1.}])
        ref.get_ham()
        sys = OrbitalSystem(lat, spin=True)
        sys.set_hopping([{'n': 1, 't': lambda si, sj, t: t}])
        sys.get_ham(t=-1.)
        np.testing.assert_allclose(sys.ham.toarray(), ref.ham.toarray())

    def test_graphene_strain_replaces_values(self):
        lat = GrapheneLattice()
        lat.hexagon_zigzag(2)
        sys = GrapheneSystem(lat)
        sys.set_hopping([{'n': 1, 't': lambda si, sj, t: t}])
        sys.set_hop_linear_strain(t=1., beta=0.)
        self.assertEqual(sys._hop_values, [])
        sys.get_ham()

    def test_errors(self):
        sys = System(square())
        sys.set_hopping([{'n': 1, 't': lambda si, sj, t: t}])
        with self.assertRaisesRegex(TypeError, 't'):
            sys.get_ham()
        sys.set_hopping([{'n': 1, 't': lambda si, sj: np.ones(3)}])
        with self.assertRaises(ValueError):
            sys.get_ham()
        with self.assertRaises(ValueError):
            sys.set_onsite({'a': 'x'})
        with self.assertRaises(TypeError):
            sys.set_hopping([{'n': 1, 't': 'x'}])
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                save_model(sys, os.path.join(tmp, 'model'))


def chain_ks(t):
    ks = KSpace(lattices.chain())
    ks.set_hopping([{'i': 0, 'j': 0, 'R': (1,), 't': t}])
    return ks


class TestKSpaceValues(unittest.TestCase):

    def test_matches_numbers(self):
        ref = KSpace(lattices.honeycomb())
        ref.set_hopping([{'n': 1, 't': -1.}, {'n': 2, 't': 0.1j}])
        ks = KSpace(lattices.honeycomb())
        ks.set_hopping([{'n': 1, 't': lambda si, sj, t: t}, {'n': 2, 't': lambda si, sj, t2: t2}])
        k = (0.3, -0.7)
        np.testing.assert_allclose(ks.get_ham(k, t=-1., t2=0.1j), ref.get_ham(k))
        ks.set_params(t=-1., t2=0.1j)
        ks2 = ks.get_ham(k, t=-2.)  # per-call values override set_params
        np.testing.assert_allclose(ks2, ref.get_ham(k) + (ks2 - ks.get_ham(k)))
        self.assertFalse(np.allclose(ks2, ks.get_ham(k)))
        ks.set_params(t=-1.)
        ks.get_bands([(0., 0.), k])
        np.testing.assert_allclose(ks.en, ref.get_bands([(0., 0.), k]))
        np.testing.assert_allclose(ks.finite_ham(3), ref.finite_ham(3))

    def test_sites(self):
        lat = lattices.honeycomb()
        ks = KSpace(lat)
        seen = []

        def t(si, sj):
            seen.append((si, sj))
            return np.hypot(sj['x'] - si['x'], sj['y'] - si['y'])
        ks.set_hopping([{'n': 1, 't': t}])
        ks.get_ham((0., 0.))
        (si, sj), = seen
        self.assertEqual(len(si), 3)  # one call for the three bonds
        np.testing.assert_allclose(t(si, sj), np.linalg.norm(lat.prim_vec[0]) / np.sqrt(3))
        # each bond joins an 'a' site (index 0) and a 'b' site (index 1)
        np.testing.assert_array_equal(si['tag'] != sj['tag'], True)
        np.testing.assert_array_equal(si['index'], (si['tag'] == 'b').astype(int))

    def test_reversed_bond(self):
        # a hopping given along a negative angle is the conjugate of the stored one
        ref = KSpace(lattices.square())
        ref.set_hopping([{'n': 1, 'ang': -90., 't': 1j}, {'n': 1, 'ang': 0., 't': 1.}])
        ks = KSpace(lattices.square())
        ks.set_hopping([{'n': 1, 'ang': -90., 't': lambda si, sj, a: 1j * a},
                                {'n': 1, 'ang': 0., 't': lambda si, sj: 1.}])
        for k in [(0.2, 0.5), (1., -0.3)]:
            np.testing.assert_allclose(ks.get_ham(k, a=1.), ref.get_ham(k))

    def test_non_hermitian(self):
        ref = KSpace(lattices.chain())
        ref.set_hopping([{'i': 0, 'j': 0, 'R': (1,), 't': 1.}, {'i': 0, 'j': 0, 'R': (-1,), 't': 0.5}],
                              hermitian=False)
        ks = KSpace(lattices.chain())
        ks.set_hopping([{'i': 0, 'j': 0, 'R': (1,), 't': lambda si, sj, g: 1.},
                               {'i': 0, 'j': 0, 'R': (-1,), 't': lambda si, sj, g: g}], hermitian=False)
        np.testing.assert_allclose(ks.get_ham((0.4,), g=0.5), ref.get_ham((0.4,)))
        self.assertFalse(ks.is_hermitian())

    def test_spin(self):
        so = 0.3 * PAULI['y']
        ref = KSpace(lattices.chain(), spin=True)
        ref.set_hopping([{'i': 0, 'j': 0, 'R': (1,), 't': -PAULI['0'] + 1j * so}])
        k = (0.7,)
        for t in [lambda si, sj, a: -PAULI['0'] + 1j * a * PAULI['y'],
                     lambda si, sj, a: (-PAULI['0'] + 1j * a * PAULI['y'])[None] * np.ones((len(si), 1, 1))]:
            ks = KSpace(lattices.chain(), spin=True)
            ks.set_hopping([{'i': 0, 'j': 0, 'R': (1,), 't': t}])
            np.testing.assert_allclose(ks.get_ham(k, a=0.3), ref.get_ham(k))
        ref = KSpace(lattices.chain(), spin=True)
        ref.set_hopping([{'n': 1, 'ang': -180., 't': 1j * PAULI['x']}])
        ks = KSpace(lattices.chain(), spin=True)
        ks.set_hopping([{'n': 1, 'ang': -180., 't': lambda si, sj: 1j * PAULI['x']}])
        np.testing.assert_allclose(ks.get_ham(k), ref.get_ham(k))
        ks = KSpace(lattices.chain(), spin=True)
        ks.set_hopping([{'n': 1, 't': lambda si, sj: -1.}])  # a number: spin-independent
        np.testing.assert_allclose(ks.get_ham(k), -2 * np.cos(0.7) * np.eye(2))

    def test_derived_models(self):
        # every tool reading the hoppings sees the values at ks.params
        ks = chain_ks(lambda si, sj, t: t)
        ks.set_params(t=-1.)
        ref = chain_ks(-1.)
        np.testing.assert_allclose(bdg_kspace(ks, [], mu=0.2).get_ham((0.3,)),
                                          bdg_kspace(ref, [], mu=0.2).get_ham((0.3,)))
        lat = lattices.square()
        rib = ribbon(lat, [{'i': 0, 'j': 0, 'R': (1, 0), 't': lambda si, sj, t: t},
                                {'i': 0, 'j': 0, 'R': (0, 1), 't': -1.}], width=4)
        rib_ref = ribbon(lat, [{'i': 0, 'j': 0, 'R': (1, 0), 't': -1.},
                                     {'i': 0, 'j': 0, 'R': (0, 1), 't': -1.}], width=4)
        np.testing.assert_allclose(rib.get_ham((0.2,), t=-1.), rib_ref.get_ham((0.2,)))
        ks.clear_hopping()
        self.assertEqual(ks._hop_values, [])

    def test_errors(self):
        ks = chain_ks(lambda si, sj, t: t)
        with self.assertRaisesRegex(TypeError, 't'):
            ks.get_ham((0.,))
        ks = KSpace(lattices.chain(), spin=True)
        ks.set_hopping([{'i': 0, 'j': 0, 'R': (1,), 't': lambda si, sj: np.ones(3)}])
        with self.assertRaises(ValueError):
            ks.get_ham((0.,))
        ks = KSpace(lattices.chain())
        with self.assertRaises(TypeError):
            ks.set_overlap([{'i': 0, 'j': 0, 'R': (1,), 't': lambda si, sj: 0.1}])
        with self.assertRaises(TypeError):
            magnetic_supercell(lattices.square(), [{'i': 0, 'j': 0, 'R': (1, 0), 't': lambda si, sj: 1.}],
                                       p=1, q=3)
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                save_model(chain_ks(lambda si, sj: 1.), os.path.join(tmp, 'model'))


if __name__ == '__main__':
    unittest.main()
