"""
tbkit.symmetry: space-group operations and their representation, the
symmetry finder, irreps at high-symmetry points, the rotation and inversion
indicators, the corner charge, and symmetrization.
"""
import os
import tempfile
import unittest

import numpy as np
import scipy.linalg as sla

import tbkit.error_handling as eh
from tbkit import symmetry
from tbkit.higher_order import corner_charges
from tbkit.io import read_wannier90
from tbkit.kspace import KSpace, PAULI, high_symmetry_path
from tbkit.lattice import Lattice
from tbkit.slater_koster import sk_kspace
from tbkit.symmetry import (Operation, corner_charge, find_symmetries, inversion_indicators, irreps,
                                     mirror, representation, rotation, rotation_indicator, symmetrize)
from tests.test_invariants import cubic_ti
from tests.test_topology import haldane, kane_mele
from tests.test_wannier90 import hr_text

CUBIC_P = Operation(-np.eye(3), orbitals=np.kron(np.diag([1., -1.]), np.eye(2)))  # s even, p odd


def graphene(M=0.):
    return haldane(t2=0., M=M)


def qwz(m):
    '''Qi-Wu-Zhang, two orbitals on one site: C = 1 for -2 < m < 0, -1 for 0 < m < 2.'''
    lat = Lattice(unit_cell=[{'tag': 'a', 'r0': (0., 0.)}, {'tag': 'b', 'r0': (0., 0.)}],
                         prim_vec=[(1., 0.), (0., 1.)])
    ks = KSpace(lat)
    hops = []
    for R, s in (((1, 0), PAULI['x']), ((0, 1), PAULI['y'])):
        t = s / 2j + PAULI['z'] / 2
        hops += [{'i': i, 'j': j, 'R': R, 't': t[i, j]} for i in range(2) for j in range(2) if t[i, j] != 0]
    ks.set_hopping(hops)
    ks.set_onsite({'a': m, 'b': -m})
    return ks


def ssh_2d(g, lam, copies=1, d=0.):
    '''
    2D SSH model: four sites around the centre of the cell, intracell
    hopping g, intercell lam; *copies* copies coupled on site by d.
    Obstructed (one Wannier centre per cell at its corner) for g < lam.
    '''
    pos = [(0.25, 0.25), (0.75, 0.25), (0.75, 0.75), (0.25, 0.75)]
    cell = [{'tag': t + str(c), 'r0': p} for t, p in zip('abcd', pos) for c in range(copies)]
    ks = KSpace(Lattice(unit_cell=cell, prim_vec=[(1., 0.), (0., 1.)]))
    hops = []
    for c in range(copies):
        def s(i):
            return copies * i + c
        hops += [{'i': s(i), 'j': s((i + 1) % 4), 'R': (0, 0), 't': -g} for i in range(4)]
        hops += [{'i': s(1), 'j': s(0), 'R': (1, 0), 't': -lam}, {'i': s(2), 'j': s(3), 'R': (1, 0), 't': -lam},
                     {'i': s(3), 'j': s(0), 'R': (0, 1), 't': -lam}, {'i': s(2), 'j': s(1), 'R': (0, 1), 't': -lam}]
        if c:
            hops += [{'i': copies * i, 'j': s(i), 'R': (0, 0), 't': d} for i in range(4)]
    ks.set_hopping(hops)
    return ks


def sp_cubic():
    '''Simple cubic, an s and three p orbitals per site.'''
    lat = Lattice(unit_cell=[{'tag': 'a', 'r0': (0., 0., 0.)}],
                         prim_vec=[(1., 0., 0.), (0., 1., 0.), (0., 0., 1.)])
    return sk_kspace(lat, {'a': ['s', 'px', 'py', 'pz']},
                            {1: {'ss_sigma': -1., 'sp_sigma': 0.8, 'pp_sigma': 1.2, 'pp_pi': -0.3}},
                            onsite={'a': {'s': -2., 'px': 3., 'py': 3., 'pz': 3.}})


def doubled_cubic_ti(M):
    '''Two decoupled copies of cubic_ti(M): z4 = 2 for a strong M.'''
    one = cubic_ti(M)
    lat = Lattice(unit_cell=[{'tag': t, 'r0': (0., 0., 0.)} for t in ('s', 'p', 's2', 'p2')],
                         prim_vec=one.lat.prim_vec)
    ks = KSpace(lat, spin=True)
    ks._hop = [(i + c, j + c, R, t) for c in (0, 4) for i, j, R, t in one._hop]
    ks.onsite = np.tile(one.onsite, 2)
    return ks


def names(ops, anti=False):
    return [op.name for op in ops if op.antiunitary == anti]


class TestOperations(unittest.TestCase):

    def test_matrices_and_names(self):
        np.testing.assert_allclose(rotation(np.pi / 2), [[0, -1], [1, 0]], atol=1e-15)
        np.testing.assert_allclose(rotation(np.pi / 2, (0, 0, 2.)), sla.block_diag([[0, -1], [1, 0]], 1),
                                            atol=1e-15)
        np.testing.assert_allclose(mirror((1., 0.)), np.diag([-1., 1.]))
        for rot, name in ((np.eye(2), 'E'), (rotation(np.pi), 'C2'), (rotation(2 * np.pi / 3), 'C3'),
                                  (rotation(-2 * np.pi / 3), 'C3^-1'), (mirror((0., 1.)), 'M(0)'),
                                  (np.eye(3), 'E'), (-np.eye(3), 'I'), (rotation(np.pi / 2, (0, 0, 1)), 'C4[0 0 1]'),
                                  (mirror((0, 0, 1.)), 'M[0 0 1]'), (-rotation(np.pi / 2, (0, 0, 1)), 'S4[0 0 1]'),
                                  (-rotation(2 * np.pi / 3, (1, 1, 1)), 'S6[1 1 1]'),
                                  (-rotation(np.pi / 3, (0, 0, 1)), 'S3[0 0 1]')):
            self.assertEqual(Operation(rot).name, name)
        self.assertEqual(Operation(np.eye(2), antiunitary=True).name, 'E T')
        self.assertEqual(Operation(np.eye(2), name='x').name, 'x')
        self.assertIn('C2', repr(Operation(rotation(np.pi))))

    def test_products(self):
        c = np.array([0.3, -0.2])
        c4 = Operation.about(rotation(np.pi / 2), c, orbitals=np.diag([1., 1j]))
        np.testing.assert_allclose(c4.translation, c - rotation(np.pi / 2) @ c)
        g = c4 ** 4
        np.testing.assert_allclose(g.rotation, np.eye(2), atol=1e-14)
        np.testing.assert_allclose(g.translation, 0., atol=1e-14)
        np.testing.assert_allclose(g.orbitals, np.eye(2), atol=1e-14)
        # an antiunitary factor conjugates the orbital matrix of the next one
        t = Operation(np.eye(2), orbitals=np.eye(2), antiunitary=True)
        tc4 = t @ c4
        self.assertTrue(tc4.antiunitary)
        np.testing.assert_allclose(tc4.orbitals, np.diag([1., -1j]))
        self.assertIsNone((Operation(rotation(np.pi)) @ Operation(mirror((1., 0.)))).orbitals)

    def test_representation(self):
        # H(gk) = U H(k) U^dagger, for every operation of graphene (with
        # time reversal) and of Kane-Mele (spin rotations)
        for ks in (graphene(), kane_mele(rashba=0.1)):
            for op in find_symmetries(ks):
                k = np.array([0.37, -0.81])
                u = representation(ks, op, k)
                gk = (-1 if op.antiunitary else 1) * op.rotation @ k
                h = ks.get_ham(k)
                h = h.conj() if op.antiunitary else h
                np.testing.assert_allclose(ks.get_ham(gk), u @ h @ u.conj().T, atol=1e-12)
        # the symmetry error of the operations, and of broken ones
        gra = graphene()
        c6 = [op for op in find_symmetries(gra) if op.name == 'C6'][0]
        self.assertLess(gra.symmetry_error(c6), 1e-12)
        self.assertGreater(graphene(M=0.3).symmetry_error(c6), 0.5)
        # explicit orbital matrices: the C4 of Qi-Wu-Zhang, spinless or as a spin 1/2
        for d in (np.diag([1., 1j]), np.diag([np.exp(-1j * np.pi / 4), np.exp(1j * np.pi / 4)])):
            self.assertLess(qwz(-1.).symmetry_error(Operation(rotation(np.pi / 2), orbitals=d)), 1e-12)
        self.assertGreater(qwz(-1.).symmetry_error(Operation(rotation(np.pi / 2))), 0.5)
        # anti: chiral symmetry of graphene as a k -> k "operation" with the minus sign
        identity = Operation(np.eye(2))
        self.assertLess(gra.symmetry_error(identity), 1e-14)
        self.assertGreater(gra.symmetry_error(identity, anti=True), 1.)
        # H^T(k) = H(-k) for real hoppings: not H(k)
        self.assertGreater(gra.symmetry_error(identity, transpose=True), 1.)


class TestFinder(unittest.TestCase):

    def test_graphene_and_its_subgroups(self):
        # graphene: C6v, 12 operations, each also times time reversal
        ops = find_symmetries(graphene())
        self.assertEqual(len(names(ops)), 12)
        self.assertEqual(len(names(ops, True)), 12)
        self.assertEqual(names(ops)[0], 'E')
        self.assertIn('E T', names(ops, True))
        # every operation is a rotation about the centre of a hexagon
        centre = 2 * (np.array(graphene().lat.prim_vec[0]) + graphene().lat.prim_vec[1]) / 3
        for op in ops:
            self.assertTrue(np.allclose(op.translation % 1e9, (centre - op.rotation @ centre) % 1e9))
        # sublattice mass: C3v, with time reversal
        ops = find_symmetries(graphene(M=0.3))
        self.assertEqual(sorted(names(ops)), sorted(['E', 'C3', 'C3^-1', 'M(30)', 'M(90)', 'M(150)']))
        self.assertEqual(len(names(ops, True)), 6)
        # Haldane flux: C6, and the mirrors only times time reversal
        ops = find_symmetries(haldane())
        self.assertEqual(sorted(names(ops)), sorted(['E', 'C6', 'C6^-1', 'C3', 'C3^-1', 'C2']))
        self.assertIn('M(0) T', names(ops, True))
        self.assertNotIn('E T', names(ops, True))
        # Kane-Mele (spin rotations on the spin): C6v and time reversal; a
        # mass leaves C3v; Rashba keeps the vertical mirrors of the plane
        self.assertEqual(len(find_symmetries(kane_mele())), 24)
        self.assertEqual(len(names(find_symmetries(kane_mele(M=0.1)))), 6)
        self.assertEqual(len(find_symmetries(kane_mele(rashba=0.1))), 24)
        self.assertEqual(len(find_symmetries(graphene(), antiunitary=False)), 12)

    def test_orbitals(self):
        # s and p orbitals: Oh (48), and time reversal
        ops = find_symmetries(sp_cubic(), orbitals=lambda R: sla.block_diag(1., R))
        self.assertEqual(len(ops), 96)
        self.assertTrue({'I', 'C4[1 0 0]', 'C3[1 1 1]', 'S6[1 1 1]', 'M[0 0 1]'} <= set(names(ops)))
        # with the identity on the orbitals, only E keeps every p orbital
        self.assertEqual(len(find_symmetries(sp_cubic(), antiunitary=False)), 1)
        # spinful: the orbital matrix times the spin rotation
        self.assertEqual(len(find_symmetries(cubic_ti(2.), orbitals=lambda R: np.diag([1., np.linalg.det(R)]))),
                              96)

    def test_nonsymmorphic(self):
        # two sites related by a glide (x + 1/2, -y): p2mg, no common centre
        lat = Lattice(unit_cell=[{'tag': 'a', 'r0': (0.25, 0.1)}, {'tag': 'b', 'r0': (0.75, -0.1)}],
                             prim_vec=[(1., 0.), (0., 1.3)])
        ks = KSpace(lat)
        ks.set_hopping([{'i': 0, 'j': 1, 'R': (0, 0), 't': 1.}, {'i': 1, 'j': 0, 'R': (1, 0), 't': 1.},
                                 {'i': 0, 'j': 0, 'R': (0, 1), 't': 0.2}, {'i': 1, 'j': 1, 'R': (0, 1), 't': 0.2}])
        ops = find_symmetries(ks, antiunitary=False)
        self.assertEqual(len(ops), 4)
        glide = [op for op in ops if op.name == 'M(0)'][0]
        self.assertAlmostEqual(abs(glide.translation[0]) % 1, 0.5)
        self.assertRaises(ValueError, irreps, ks, np.zeros(2), ops)


class TestIrreps(unittest.TestCase):

    def test_graphene(self):
        gra = graphene()
        points, _ = high_symmetry_path(gra.lat)
        gamma, m, k = points[:3]
        res = irreps(gra, k)
        # the Dirac point: the two-dimensional irrep of the group of K
        self.assertEqual((res.group, res.labels, res.bands), ('C3v', ['E'], [[0, 1]]))
        self.assertAlmostEqual(res.energies[0], 0.)
        self.assertEqual(len(res.operations), 6)
        np.testing.assert_allclose(res.characters[0, 0], 2.)
        self.assertEqual(irreps(gra, gamma).group, 'C6v')
        self.assertEqual(irreps(gra, m).group, 'C2v')
        self.assertEqual(irreps(gra, (0.13, 0.31)).labels, ['A', 'A'])
        # the mass removes the mirrors of K: two one-dimensional irreps
        res = irreps(graphene(M=0.3), k)
        self.assertEqual((res.group, set(res.labels)), ('C3', {'¹E', '²E'}))
        self.assertEqual(irreps(graphene(M=0.3), m).group, 'Cs')
        self.assertEqual(irreps(haldane(), m).group, 'C2')

    def test_compatibility(self):
        # along Gamma-K-M, the characters of the group of each line equal
        # those of the endpoints restricted to it
        gra = graphene()
        ops = find_symmetries(gra, antiunitary=False)
        points, _ = high_symmetry_path(gra.lat)
        for p, q in zip(points[:-1], points[1:]):
            line = irreps(gra, 0.6 * p + 0.4 * q, ops)
            for end in (p, q):
                res = irreps(gra, end, ops)
                idx = [[o.name for o in res.operations].index(o.name) for o in line.operations]
                for n in range(gra.norb):
                    lv = [i for i, b in enumerate(res.bands) if n in b][0]
                    lv_line = [i for i, b in enumerate(line.bands) if n in b][0]
                    dim = len(res.bands[lv])
                    # a level of the endpoint splits into levels of the line
                    sub = sum(line.characters[i] for i, b in enumerate(line.bands) if set(b) <= set(res.bands[lv]))
                    np.testing.assert_allclose(res.characters[lv, idx], sub, atol=1e-10)
                    self.assertLessEqual(len(line.bands[lv_line]), dim)

    def test_cubic(self):
        ks = sp_cubic()
        ops = find_symmetries(ks, orbitals=lambda R: sla.block_diag(1., R), antiunitary=False)
        res = irreps(ks, np.zeros(3), ops)
        self.assertEqual((res.group, res.labels), ('Oh', ['A1g', 'T1u']))
        self.assertEqual(irreps(ks, np.full(3, np.pi), ops).labels, ['T1u', 'A1g'])
        self.assertRaises(ValueError, irreps, ks, np.array([np.pi, 0., 0.]), ops)  # D4h: not tabulated

    def test_accidental_degeneracy(self):
        res = irreps(ssh_2d(0.3, 1., copies=2), np.zeros(2))
        self.assertEqual(res.group, 'C4v')
        self.assertTrue(all(label.startswith('2') for label in res.labels))
        # a level split by noise is not a representation
        noisy = graphene()
        noisy.onsite = np.array([1e-3, 0.])
        k = high_symmetry_path(noisy.lat)[0][2]
        ops = find_symmetries(graphene(), antiunitary=False)
        self.assertRaises(ValueError, irreps, noisy, k, ops)

    def test_tables(self):
        # every tabulated group: orthonormal characters, sum of dim^2 = |G|
        hexa = symmetry._lattice_rotations(np.array([[1., 0.], [0.5, np.sqrt(3) / 2]]))
        square = symmetry._lattice_rotations(np.eye(2))
        cube = symmetry._lattice_rotations(np.eye(3))
        self.assertEqual((len(hexa), len(square), len(cube)), (12, 8, 48))

        def sub(rots, keep):
            return [r for r in rots if keep(r)]

        def named(rots, keep):
            return [r for r in rots if symmetry._name(r, False) in keep]

        proper = sub(cube, lambda r: np.linalg.det(r) > 0)
        t_group = sub(proper, lambda r: symmetry._cubic_class(r) in ('E', 'C3', 'C3^2', 'C2'))
        groups = {
            'C6v': hexa, 'C6': sub(hexa, lambda r: np.linalg.det(r) > 0),
            'C3v': named(hexa, ('E', 'C3', 'C3^-1', 'M(0)', 'M(60)', 'M(120)')),
            'C3': named(hexa, ('E', 'C3', 'C3^-1')),
            'C4v': square, 'C4': sub(square, lambda r: np.linalg.det(r) > 0),
            'C2v': named(square, ('E', 'C2', 'M(0)', 'M(90)')), 'C2': [np.eye(2), -np.eye(2)],
            'Cs': [np.eye(2), mirror((1., 1.))], 'C1': [np.eye(2)],
            'Oh': cube, 'O': proper, 'T': t_group, 'Th': t_group + [-r for r in t_group],
            'Td': t_group + [-r for r in proper if not any(np.allclose(r, s) for s in t_group)],
        }
        for name, rots in groups.items():
            got, table = symmetry._point_group(rots, np.array([1., 0.] if len(rots[0]) == 2 else [1., 0., 0.]))
            self.assertEqual(got, name)
            chi = np.array([c for _, c in table])
            np.testing.assert_allclose(chi.conj() @ chi.T / len(rots), np.eye(len(chi)), atol=1e-12)
            e = [i for i, r in enumerate(rots) if np.allclose(r, np.eye(len(r)))][0]
            self.assertAlmostEqual(np.sum(chi[:, e].real ** 2), len(rots))
        # a non-cubic 3D group is not tabulated
        self.assertIsNone(symmetry._point_group([np.eye(3), -np.eye(3)], np.array([1., 0., 0.])))
        self.assertIsNone(symmetry._point_group(proper[:1] + [rotation(np.pi / 2, (1, 2, 3))], np.ones(3)))

    def test_plot(self):
        gra = graphene()
        points, labels = high_symmetry_path(gra.lat)
        gra.k_path(points, 20)
        fig = gra.plot_bands(node_labels=labels, irreps=True)
        texts = [t.get_text() for t in fig.axes[0].texts]
        self.assertIn('E', texts)
        self.assertEqual(len(texts), 2 + 2 + 1 + 2)  # Gamma, M, K, Gamma
        fig = gra.plot_bands(irreps=find_symmetries(gra, antiunitary=False))
        self.assertEqual(len(fig.axes[0].texts), 7)
        self.assertEqual(len(gra.plot_bands(irreps=False).axes[0].texts), 0)
        # nodes whose group is not tabulated are skipped
        ks = sp_cubic()
        ks.k_path([np.zeros(3), np.array([np.pi, 0., 0.])], 10)
        fig = ks.plot_bands(irreps=find_symmetries(ks, orbitals=lambda R: sla.block_diag(1., R),
                                                                    antiunitary=False))
        self.assertEqual(len(fig.axes[0].texts), 2)
        # needs a k-path, and no spin
        ks = graphene()
        ks.get_bands([np.zeros(2), np.ones(2)])
        self.assertRaises(RuntimeError, ks.plot_bands, irreps=True)
        km = kane_mele()
        km.k_path(points, 5)
        self.assertRaises(ValueError, km.plot_bands, irreps=True)


class TestIndicators(unittest.TestCase):

    def test_chern_from_rotations(self):
        # Qi-Wu-Zhang (C4, and C2), spinless or spin-1/2 representation, and a
        # clockwise rotation, against the Chern number
        for m in (-3., -1., 1., 3.):
            ks = qwz(m)
            c = int(round(ks.chern_number(0)))
            for d in (np.diag([1., 1j]), np.diag([np.exp(-1j * np.pi / 4), np.exp(1j * np.pi / 4)])):
                c4 = Operation(rotation(np.pi / 2), orbitals=d)
                self.assertEqual(rotation_indicator(ks, c4, [0]), c % 4)
                self.assertEqual(rotation_indicator(ks, c4 ** 2, [0]), c % 2)
            c4m = Operation(rotation(-np.pi / 2), orbitals=np.diag([1., -1j]))
            self.assertEqual(rotation_indicator(ks, c4m, [0]), c % 4)
        # Haldane (C3, and C6 without a mass)
        for t2, M in ((0.2, 0.), (-0.2, 0.), (0.2, 0.5), (0.2, 2.)):
            h = haldane(t2, M)
            c = int(round(h.chern_number(0)))
            ops = {op.name: op for op in find_symmetries(h, antiunitary=False)}
            self.assertEqual(rotation_indicator(h, ops['C3'], [0]), c % 3)
            if 'C6' in ops:
                self.assertEqual(rotation_indicator(h, ops['C6'], [0]), c % 6)
        # checks
        ks = qwz(-1.)
        self.assertRaises(ValueError, rotation_indicator, ks, Operation(rotation(np.pi / 2)), [0])  # no symmetry
        self.assertRaises(ValueError, rotation_indicator, ks, Operation(mirror((1., 0.))), [0])
        odd = Operation(rotation(np.pi / 2), orbitals=np.exp(1j * np.pi / 8) * np.diag([1., 1j]))
        self.assertRaises(ValueError, rotation_indicator, ks, odd, [0])  # C4^4 = i
        gra = graphene()
        c3 = {op.name: op for op in find_symmetries(gra)}['C3']
        self.assertRaises(ValueError, rotation_indicator, gra, c3, [0])  # the Dirac point at K
        self.assertRaises(ValueError, rotation_indicator, cubic_ti(2.), CUBIC_P, [0, 1])  # 3D

    def test_inversion_indicators(self):
        # z4 mod 2 and the weak indices are the Z2 indices of z2_indices_3d
        for M, z2 in ((-2., (0, 0, 0, 0)), (2., (1, 0, 0, 0)), (6., (0, 1, 1, 1)), (10., (1, 1, 1, 1))):
            z4, *weak = inversion_indicators(cubic_ti(M), CUBIC_P, [0, 1])
            self.assertEqual((z4 % 2, *weak), z2)
        self.assertEqual(inversion_indicators(cubic_ti(2.), CUBIC_P, [0, 1])[0], 3)
        # two copies of a strong TI: invisible to Z2, z4 = 2
        double = doubled_cubic_ti(2.)
        p2 = Operation(-np.eye(3), orbitals=np.kron(np.diag([1., -1., 1., -1.]), np.eye(2)))
        self.assertEqual(inversion_indicators(double, p2, [0, 1, 2, 3]), (2, 0, 0, 0))
        self.assertEqual(double.z2_indices_3d([0, 1, 2, 3], nk=20, nk_perp=11), (0, 0, 0, 0))
        self.assertRaises(ValueError, inversion_indicators, cubic_ti(2.), Operation(np.eye(3)), [0, 1])
        self.assertRaises(ValueError, inversion_indicators, haldane(), Operation(-np.eye(2)), [0])

    def test_corner_charge(self):
        # one Wannier centre per cell at its corner: Q = 1/4; two: 1/2; trivial: 0
        c4 = Operation.about(rotation(np.pi / 2), (0.5, 0.5))
        n = 8
        for ks, bands, q in ((ssh_2d(0.2, 1.), [0], 0.25), (ssh_2d(1., 0.2), [0], 0.),
                                     (ssh_2d(0.2, 1., copies=2, d=0.1), [0, 1], 0.5)):
            self.assertEqual(corner_charge(ks, c4, bands), q)
            # the flake: bulk bands and edge states below the gap at -0.6, on
            # the background of the filled bulk bands
            en = np.linalg.eigvalsh(ks.finite_ham(n))
            self.assertGreater(np.min(np.abs(en + 0.6)), 0.05)
            n_el = int(np.sum(en < -0.6))
            res = corner_charges(ks, n, n_electrons=n_el, background=len(bands) / ks.norb)
            np.testing.assert_allclose((res.charges - q + 0.5) % 1. - 0.5, 0., atol=1e-8)
        ks = ssh_2d(0.2, 1.)
        self.assertRaises(ValueError, corner_charge, ks, c4 ** 2, [0])  # C2: left out
        self.assertRaises(ValueError, corner_charge, ks, Operation(rotation(np.pi / 2)), [0])  # not the cell centre
        self.assertRaises(ValueError, corner_charge, cubic_ti(2.), CUBIC_P, [0])


class TestSymmetrize(unittest.TestCase):

    def test_symmetric_unchanged(self):
        k = np.array([0.31, -0.77])
        for ks in (graphene(), haldane(), kane_mele(rashba=0.1)):
            sym = symmetrize(ks, find_symmetries(ks))
            np.testing.assert_allclose(sym.get_ham(k), ks.get_ham(k), atol=1e-14)
            self.assertTrue(sym.is_hermitian())
        ti = cubic_ti(2.)
        k3 = np.array([0.1, 0.2, 0.3])
        np.testing.assert_allclose(symmetrize(ti, [CUBIC_P]).get_ham(k3), ti.get_ham(k3), atol=1e-14)
        q = qwz(-1.)
        c4 = Operation(rotation(np.pi / 2), orbitals=np.diag([1., 1j]))
        np.testing.assert_allclose(symmetrize(q, [c4]).get_ham(k), q.get_ham(k), atol=1e-14)

    def test_noisy_model(self):
        gra = graphene()
        ops = find_symmetries(gra)
        rng = np.random.default_rng(1)
        noisy = KSpace(gra.lat)  # noise on every bond, its conjugate added by set_hopping
        noisy.set_hopping([{'i': 0, 'j': 1, 'R': R, 't': 1. + 1e-3 * (rng.normal() + 1j * rng.normal())}
                                    for R in [(0, 0), (-1, 0), (0, -1)]])
        noisy.set_hopping([{'i': i, 'j': i, 'R': R, 't': 1e-3 * (rng.normal() + 1j * rng.normal())}
                                    for i in range(2) for R in [(1, 0), (0, 1), (1, -1)]])
        noisy.onsite = 1e-3 * rng.normal(size=2)
        k_point = high_symmetry_path(gra.lat)[0][2]
        self.assertGreater(np.ptp(noisy.get_bands([k_point])), 1e-4)
        self.assertGreater(noisy.symmetry_error(ops[1]), 1e-4)
        sym = symmetrize(noisy, ops)
        self.assertLess(np.ptp(sym.get_bands([k_point])), 1e-14)
        self.assertLess(max(sym.symmetry_error(op) for op in ops), 1e-14)
        # the generators give the same group: C6, a mirror and time reversal
        gens = [op for op in ops if op.name in ('C6', 'M(0)', 'E T')]
        np.testing.assert_allclose(symmetrize(noisy, gens).get_ham((0.3, 0.1)), sym.get_ham((0.3, 0.1)),
                                            atol=1e-14)
        # C6 alone keeps the noise that breaks time reversal (a Haldane flux)
        c6 = symmetrize(noisy, [op for op in gens if op.name == 'C6'])
        self.assertLess(c6.symmetry_error(gens[0]), 1e-14)
        self.assertGreater(c6.symmetry_error([op for op in ops if op.name == 'E T'][0]), 1e-5)

    def test_wannier90_file(self):
        # graphene p_z with second neighbours, written by Wannier90 with noise
        t, t2 = -2.8, 0.1
        rng = np.random.default_rng(0)
        hop = {}
        for R in [(0, 0), (-1, 0), (0, -1)]:
            hop.setdefault(R, np.zeros((2, 2), complex))[0, 1] += t
            hop.setdefault(tuple(-r for r in R), np.zeros((2, 2), complex))[1, 0] += t
        for R in [(1, 0), (0, 1), (1, -1), (-1, 0), (0, -1), (-1, 1)]:
            hop.setdefault(R, np.zeros((2, 2), complex))[[0, 1], [0, 1]] += t2
        for R in list(hop):
            noise = 2e-4 * (rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2)))
            hop[R] = hop[R] + noise
        for R in list(hop):  # keep the file Hermitian
            m = tuple(-r for r in R)
            if R < m:
                hop[m] = hop[R].conj().T
        hop[(0, 0)] = (hop[(0, 0)] + hop[(0, 0)].conj().T) / 2
        entries = {(R[0], R[1], 0): h for R, h in hop.items()}
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'gra_hr.dat')
            with open(path, 'w') as f:
                f.write(hr_text(2, entries, [1] * len(entries)))
            a = 1.42 * np.sqrt(3)
            prim = [(a, 0.), (a / 2, 1.5 * 1.42)]
            ks = read_wannier90(path, prim_vec=prim, positions=[[0., 0.], [a / 2, 0.5 * 1.42]], dim=2)
        ops = find_symmetries(ks, tol=1e-2)
        self.assertEqual(len(ops), 24)
        self.assertGreater(ks.symmetry_error(ops[1]), 1e-5)
        sym = symmetrize(ks, ops)
        k_point = high_symmetry_path(ks.lat)[0][2]
        self.assertLess(np.ptp(sym.get_bands([k_point])), 1e-13)
        self.assertGreater(np.ptp(ks.get_bands([k_point])), 1e-5)
        self.assertEqual(find_symmetries(sym, tol=1e-12)[0].name, 'E')
        self.assertEqual(len(find_symmetries(sym, tol=1e-12)), 24)
        self.assertEqual(irreps(sym, k_point).labels, ['E'])

    def test_checks(self):
        ks = graphene()
        self.assertRaises(ValueError, symmetrize, ks, [Operation(rotation(1.))])  # not a finite group
        ov = graphene()
        ov.set_overlap([{'i': 0, 'j': 1, 'R': (0, 0), 't': 0.1}])
        self.assertRaises(ValueError, symmetrize, ov, find_symmetries(ks))
        self.assertRaises(TypeError, symmetrize, ks, [])
        # same rotation, translations that differ by a non-lattice vector
        half = np.array(ks.lat.prim_vec[0]) / 2
        group = symmetry._closure(ks, [Operation(mirror((0., 1.))), Operation(mirror((0., 1.)), translation=half)])
        self.assertEqual(len(group), 4)


class TestSymmetryChecks(unittest.TestCase):

    def test_operation_checks(self):
        self.assertRaises(ValueError, Operation, np.ones((2, 2)))
        self.assertRaises(ValueError, Operation, np.eye(4))
        self.assertRaises(ValueError, Operation, np.eye(2) * 1j)
        self.assertRaises(ValueError, Operation, np.eye(2), translation=(1., 2., 3.))
        self.assertRaises(ValueError, Operation, np.eye(2), orbitals=np.ones((2, 2)))
        self.assertRaises(ValueError, Operation, np.eye(2), orbitals=np.ones(2))
        self.assertRaises(TypeError, Operation, np.eye(2), antiunitary=1)
        self.assertRaises(TypeError, Operation, np.eye(2), name=1)
        self.assertRaises(ValueError, Operation.about, np.eye(2), (1., 2., 3.))
        self.assertRaises(ValueError, rotation, 1., (0., 0., 0.))
        self.assertRaises(TypeError, rotation, 'a')
        self.assertRaises(ValueError, mirror, (0., 0.))
        self.assertRaises(ValueError, Operation(np.eye(2)).__matmul__, Operation(np.eye(3)))
        self.assertRaises(ValueError, Operation(np.eye(2)).__matmul__, Operation(np.eye(2), orbitals=np.eye(2)))
        self.assertRaises(ValueError, Operation(np.eye(2), orbitals=np.eye(1)).__matmul__,
                               Operation(np.eye(2), orbitals=np.eye(2)))
        self.assertRaises(TypeError, Operation(np.eye(2)).__matmul__, np.eye(2))
        self.assertRaises(ValueError, Operation(np.eye(2)).__pow__, 0)

    def test_model_checks(self):
        gra = graphene()
        self.assertRaises(ValueError, gra.symmetry_error, Operation(np.eye(3)))
        self.assertRaises(ValueError, gra.symmetry_error, Operation(rotation(np.pi / 2)))  # not the lattice
        self.assertRaises(ValueError, gra.symmetry_error, Operation(np.eye(2), orbitals=np.eye(2)))
        lat = Lattice(unit_cell=[{'tag': 'a', 'r0': (0., 0.)}, {'tag': 'b', 'r0': (0., 0.)},
                                         {'tag': 'c', 'r0': (0.5, 0.)}], prim_vec=[(1., 0.), (0., 1.)])
        self.assertRaises(ValueError, KSpace(lat).symmetry_error, Operation(np.eye(2), (0.5, 0.)))
        self.assertRaises(TypeError, irreps, gra, np.zeros(2), [])
        self.assertRaises(TypeError, find_symmetries, gra, orbitals=1)
        chain = KSpace(Lattice(unit_cell=[{'tag': 'a', 'r0': (0., 0.)}], prim_vec=[(1., 0.)]))
        self.assertRaises(ValueError, find_symmetries, chain)
        self.assertRaises(ValueError, irreps, kane_mele(), np.zeros(2))
        ov = graphene()
        ov.set_overlap([{'i': 0, 'j': 1, 'R': (0, 0), 't': 0.1}])
        self.assertRaises(ValueError, irreps, ov, np.zeros(2))

    def test_error_handling(self):
        eh.dim_irreps(2)
        self.assertRaises(ValueError, eh.dim_irreps, 1)
        eh.quantized(2.0001, 'x')
        self.assertRaises(ValueError, eh.quantized, 0.5, 'x')
        eh.invariant_bands(np.ones(2))
        self.assertRaises(ValueError, eh.invariant_bands, np.array([1., 0.5]))
        self.assertRaises(ValueError, eh.multiplicities, np.array([[0.5]]))
        self.assertRaises(ValueError, eh.multiplicities, np.array([[-1.]]))
        self.assertRaises(ValueError, eh.tabulated, None)
        self.assertRaises(ValueError, eh.symmorphic, 0.1)
        self.assertRaises(ValueError, eh.maps_lattice, np.array([[True, True], [False, False]]))
        self.assertRaises(ValueError, eh.block_sizes, [1, 2], [1, 0], None)
        self.assertRaises(ValueError, eh.group_size, 193)
        self.assertEqual(eh.rotation_power(-np.eye(2)), -1)


if __name__ == '__main__':
    unittest.main()
