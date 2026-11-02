"""
Sublattice tags longer than one character ('Mo', 'Se1').

Every model built with them must match the same model built with the
one-character tags 'a' and 'b', and a bond selector names its two tags as a
pair, so that ('M', 'oS') and ('Mo', 'S') stay distinct bonds although both
concatenate to 'MoS'.
"""
import os
import tempfile
import unittest

import numpy as np

import tbkit.error_handling as eh
import tbkit.lattices as lattices
from tbkit.io import load_model, read_wannier90, save_model
from tbkit.kspace import KSpace
from tbkit.lattice import Lattice, COOR_DTYPE
from tbkit.neighbours import neighbour_hoppings
from tbkit.orbital import OrbitalSystem
from tbkit.system import System, HOP_DTYPE


def honeycomb(tags=('a', 'b')):
    ref = lattices.honeycomb()
    cell = [{'tag': t, 'r0': dic['r0']} for t, dic in zip(tags, ref.unit_cell)]
    return Lattice(unit_cell=cell, prim_vec=list(ref.prim_vec))


def flake(tags, n=4):
    lat = honeycomb(tags)
    lat.get_lattice(n1=n, n2=n)
    return lat


class TestLongTags(unittest.TestCase):

    def assertSameSpectrum(self, sys1, sys2):
        for sys in (sys1, sys2):
            sys.get_ham()
        self.assertTrue(np.allclose(sys1.ham.toarray(), sys2.ham.toarray()))

    def test_lattice(self):
        lat = flake(('Mo', 'Se1'))
        self.assertEqual(lat.coor.dtype, np.dtype(COOR_DTYPE))
        self.assertEqual(sorted(lat.tags), ['Mo', 'Se1'])
        self.assertEqual(set(lat.coor['tag']), {'Mo', 'Se1'})
        # a 'U1' array of new sites is still accepted, and widened
        lat.add_sites(np.array([(-5., -5., 'c')], dtype=[('x', 'f8'), ('y', 'f8'), ('tag', 'U1')]))
        self.assertIn('c', lat.tags)
        self.assertEqual(lat.coor.dtype, np.dtype(COOR_DTYPE))
        self.assertRaises(TypeError, lat.add_sites,
                          np.array([(-6., -6., 'c')], dtype=[('x', 'f8'), ('y', 'f8'), ('tag', 'U9')]))

    def test_set_hopping_pairs(self):
        # stored honeycomb bonds: 'ab' at 30 and 150 degrees, 'ba' at 90
        both = [{'n': 1, 'tag': 'ab', 't': 1.}, {'n': 1, 'tag': 'ba', 't': 2.}]
        sets = [
            (True, both, [{'n': 1, 'tag': ('Mo', 'Se1'), 't': 1.}, {'n': 1, 'tag': ['Se1', 'Mo'], 't': 2.}]),
            (False, both, [{'n': 1, 'tag': ('Mo', 'Se1'), 't': 1.}, {'n': 1, 'tag': ['Se1', 'Mo'], 't': 2.}]),
            (True, [{'n': 1, 't': 1.}, {'n': 1, 'ang': 30., 'tag': 'ab', 't': 3.}],
             [{'n': 1, 't': 1.}, {'n': 1, 'ang': 30., 'tag': ('Mo', 'Se1'), 't': 3.}]),
            (False, [{'n': 1, 't': 1.}, {'n': 1, 'ang': -150., 'tag': 'ba', 't': 3.}],
             [{'n': 1, 't': 1.}, {'n': 1, 'ang': -150., 'tag': ('Se1', 'Mo'), 't': 3.}]),
        ]
        for upper, short, long in sets:
            ref, sys = System(flake('ab')), System(flake(('Mo', 'Se1')))
            ref.set_hopping(short, upper_part=upper)
            sys.set_hopping(long, upper_part=upper)
            self.assertEqual(np.sum(ref.hop['t'] == 3.), np.sum(sys.hop['t'] == 3.))
            self.assertSameSpectrum(ref, sys)
        self.assertEqual(set(sys.hop['tag']), {'MoSe1', 'Se1Mo'})

    def test_change_hopping_and_onsite(self):
        ref, sys = System(flake('ab')), System(flake(('Mo', 'Se1')))
        ref.set_hopping([{'n': 1, 't': 1.}])
        sys.set_hopping([{'n': 1, 't': 1.}])
        ref.set_onsite({'a': 0.5, 'b': -0.5})
        sys.set_onsite({'Mo': 0.5, 'Se1': lambda site: np.where(site['tag'] == 'Se1', -0.5, 0.)})
        ref.change_hopping_square([{'n': 1, 'tag': 'ab', 't': 2.}], xlims=(-1., 3.), ylims=(-1., 3.))
        sys.change_hopping_square([{'n': 1, 'tag': ('Mo', 'Se1'), 't': 2.}], xlims=(-1., 3.), ylims=(-1., 3.))
        ref.change_hopping_square([{'n': 1, 'ang': 30., 'tag': 'ab', 't': 4.}], xlims=(-1., 3.))
        sys.change_hopping_square([{'n': 1, 'ang': 30., 'tag': ('Mo', 'Se1'), 't': 4.}], xlims=(-1., 3.))
        self.assertSameSpectrum(ref, sys)

    def test_concatenation_is_not_ambiguous(self):
        # tags M, oS, Mo, S along a chain: the bonds M-oS and Mo-S both have hop['tag'] 'MoS'
        cell = [{'tag': t, 'r0': (0.25 * k, 0.)} for k, t in enumerate(('M', 'oS', 'Mo', 'S'))]
        lat = Lattice(unit_cell=cell, prim_vec=[(1., 0.)])
        lat.get_lattice(n1=3)
        sys = System(lat)
        sys.set_hopping([{'n': 1, 't': 1.}, {'n': 1, 'tag': ('Mo', 'S'), 't': 5.}])
        tags = lat.coor['tag']
        five = sys.hop[sys.hop['t'] == 5.]
        self.assertEqual(len(five), 3)
        self.assertTrue(np.all(tags[five['i']] == 'Mo') and np.all(tags[five['j']] == 'S'))
        self.assertEqual(np.sum(sys.hop['tag'] == 'MoS'), 6)

    def test_kspace(self):
        ref, ks = KSpace(honeycomb()), KSpace(honeycomb(('Mo', 'Se1')))
        ref.set_onsite({'a': 0.3})
        ks.set_onsite({'Mo': 0.3})
        ref.set_hopping([{'n': 1, 't': 1.}, {'n': 1, 'ang': -90., 'tag': 'ab', 't': 2.}])
        ks.set_hopping([{'n': 1, 't': 1.}, {'n': 1, 'ang': -90., 'tag': ('Mo', 'Se1'), 't': 2.}])
        for k in ((0.3, -0.7), (1.1, 0.2)):
            self.assertTrue(np.allclose(ref.get_ham(k), ks.get_ham(k)))
        self.assertEqual(len(neighbour_hoppings(honeycomb(('Mo', 'Se1')),
                                                [{'n': 1, 'tag': ('Mo', 'Se1'), 't': 1.}])), 2)

    def test_save_load(self):
        sys = System(flake(('Mo', 'Se1')))
        sys.set_hopping([{'n': 1, 'tag': ('Mo', 'Se1'), 't': 1.}, {'n': 1, 'tag': ('Se1', 'Mo'), 't': 2.}])
        sys.get_ham()
        with tempfile.TemporaryDirectory() as tmp:
            path = save_model(sys, os.path.join(tmp, 'mos'))
            new = load_model(path)
            self.assertEqual(sorted(new.lat.tags), ['Mo', 'Se1'])
            new.get_ham()
            self.assertTrue(np.allclose(sys.ham.toarray(), new.ham.toarray()))
            # an archive written before the tags widened ('U1' sites, 'U2' bonds)
            ref = System(flake('ab'))
            ref.set_hopping([{'n': 1, 't': 1.}])
            path = save_model(ref, os.path.join(tmp, 'old'))
            with np.load(path) as data:
                old = dict(data)
            old['coor'] = old['coor'].astype([('x', 'f8'), ('y', 'f8'), ('tag', 'U1')])
            old['hop'] = old['hop'].astype([(f, 'U2' if f == 'tag' else d) for f, d in HOP_DTYPE])
            old['uc_tag'] = old['uc_tag'].astype('U1')
            np.savez(path, **old)
            new = load_model(path)
            self.assertEqual(new.hop.dtype, np.dtype(HOP_DTYPE))
            new.set_hopping([{'n': 1, 'tag': 'ab', 't': 3.}])
            self.assertEqual(np.sum(new.hop['t'] == 3.), np.sum(new.hop['tag'] == 'ab'))

    def test_slater_koster_pair_keys(self):
        orbs = {'Mo': ['s'], 'Se1': ['px']}
        sys = OrbitalSystem(flake(('Mo', 'Se1'), 2), orbitals=orbs)
        sys.set_slater_koster(1, {('Mo', 'Se1'): {'sp_sigma': 1.}, ('Se1', 'Mo'): {'ps_sigma': 1.}})
        sys.get_ham()
        ref = OrbitalSystem(flake('ab', 2), orbitals={'a': ['s'], 'b': ['px']})
        ref.set_slater_koster(1, {'ab': {'sp_sigma': 1.}, 'ba': {'ps_sigma': 1.}})
        ref.get_ham()
        self.assertTrue(np.allclose(sys.ham.toarray(), ref.ham.toarray()))
        self.assertRaises(ValueError, sys.set_slater_koster, 1, {('Mo', 'Se1'): {'sp_sigma': 1.}})

    def test_wannier90_tags(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'w_hr.dat')
            with open(path, 'w') as f:
                f.write('header\n1\n1\n1\n0 0 0 1 1 0.5 0.\n')
            ks = read_wannier90(path, prim_vec=[(1., 0., 0.), (0., 1., 0.), (0., 0., 1.)], tags=['Mo'])
        self.assertEqual(list(ks.tags), ['Mo'])

    def test_errors(self):
        self.assertRaises(ValueError, Lattice, unit_cell=[{'tag': 'abcdefghi', 'r0': (0., 0.)}],
                          prim_vec=[(1., 0.)])
        eh.hop_tag('ab')
        eh.hop_tag(('Mo', 'S'))
        self.assertRaises(TypeError, eh.hop_tag, 1)
        self.assertRaises(ValueError, eh.hop_tag, 'abc')
        self.assertRaises(TypeError, eh.hop_tag, ('Mo',))
        self.assertRaises(TypeError, eh.hop_tag, ('Mo', 1))
        self.assertRaises(ValueError, eh.hop_tag, ('Mo', ''))
        sys = System(flake(('Mo', 'Se1')))
        self.assertRaises(ValueError, sys.set_hopping, [{'n': 1, 'tag': ('Mo', 'Mo'), 't': 1.}])
        self.assertRaises(ValueError, sys.set_hopping, [{'n': 1, 'ang': 90., 'tag': ('Mo', 'Mo'), 't': 1.}])
        self.assertRaises(TypeError, KSpace(honeycomb()).set_hopping, [{'n': 1, 'tag': ('a',), 't': 1.}])


if __name__ == '__main__':
    unittest.main()
