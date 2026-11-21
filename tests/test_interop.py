'''
tbkit.interop: models exchanged with ASE, PythTB, TBmodels, Z2Pack, qsymm,
pybinding and Kwant.

CI has none of these packages. Each one is replaced by a stand-in module
(``fake_*``, injected into ``sys.modules``) that records what tbkit gives
it, and the tests rebuild the Bloch Hamiltonian, the bands or the
transmissions from those records. The classes ``TestWith*`` run the same
conversions against the packages themselves where they are installed
(PythTB 1.x and 2, TBmodels, Z2Pack, qsymm, pybinding, Kwant, ASE).
'''
import importlib.util
import types
import unittest
import warnings
from unittest import mock

import numpy as np
import scipy.sparse as sp

from tbkit import interop, models, lattices
from tbkit.kspace import KSpace
from tbkit.lattice import Lattice
from tbkit.system import System
from tbkit.kspace import ribbon
from tbkit.transport import Transport


def installed(name):
    return importlib.util.find_spec(name) is not None


def same_bands(test, ks1, ks2, sign=1, atol=1e-10):
    k = np.random.default_rng(5).uniform(-3, 3, (10, ks1.dim))
    np.testing.assert_allclose(ks1.get_bands(k), ks2.get_bands(sign * k), rtol=0, atol=atol)


def gauge(ks, k):
    '''H(k) with the orbital positions in the phases (convention 1).'''
    d = np.exp(1j * (ks.k_basis @ k) @ ks.orbital_positions().T)
    return d.conj()[:, None] * ks.get_ham(k) * d[None, :]


def sample_models():
    km = models.kane_mele(lambda_so=0.06, lambda_r=0.05)
    km._onsite_offdiag[0, 1] = km._onsite_offdiag[1, 0] = 0.07  # an onsite spin block
    return [models.haldane(t2=0.15, phi=0.7, M=0.2), km, models.topological_insulator_3d(),
            models.ssh(0.4, 1.)]


# ----------------------------------------------------------------------------
# stand-ins
# ----------------------------------------------------------------------------

def fake_ase():
    module = types.ModuleType('ase')

    class Atoms():
        def __init__(self, symbols, positions, cell, pbc):
            self.symbols, self.positions = list(symbols), np.array(positions, dtype=float)
            self.cell, self.pbc = np.array(cell, dtype=float), np.array(pbc, dtype=bool)

        def get_cell(self):
            return self.cell

        def get_pbc(self):
            return self.pbc

        def get_positions(self):
            return self.positions

        def get_chemical_symbols(self):
            return self.symbols

    module.Atoms = Atoms
    module.data = types.SimpleNamespace(chemical_symbols=['X', 'H', 'He', 'C', 'N', 'Mo', 'S', 'Si'])
    return module


def fake_pythtb(version=2):
    '''PythTB 2 (TBModel, Lattice) or 1.x (tb_model), storing what tbkit sets
    in the attributes that from_pythtb reads.'''
    module = types.ModuleType('pythtb')

    class Lattice():
        def __init__(self, lat_vecs, orb_vecs, periodic_dirs):
            self.lat_vecs, self.orb_vecs = np.array(lat_vecs), np.array(orb_vecs)
            self.periodic_dirs = list(periodic_dirs)

    class TBModel():
        def __init__(self, lattice, spinful=False):
            self.lattice, self.spinful, self.hoppings = lattice, spinful, []

        @property
        def lat_vecs(self):
            return self.lattice.lat_vecs

        def get_orb_vecs(self, cartesian=False):
            return self.lattice.orb_vecs @ self.lattice.lat_vecs if cartesian else self.lattice.orb_vecs

        def set_onsite(self, values):
            self.onsite = np.array(values)

        def set_hop(self, amp, i, j, R):
            hop = {'amplitude': amp, 'from_orbital': i, 'to_orbital': j}
            if any(R):  # PythTB 2 leaves the vector out within the cell
                hop['lattice_vector'] = list(R)
            self.hoppings.append(hop)

    class tb_model():
        def __init__(self, dim_k, dim_r, lat, orb, per, nspin):
            self._lat, self._orb, self._per, self._nspin = np.array(lat), np.array(orb), per, nspin
            self._hoppings = []

        def set_onsite(self, values):
            self._site_energies = list(values)

        def set_hop(self, amp, i, j, R):
            self._hoppings.append([amp, i, j, np.array(R)])

    if version == 2:
        module.Lattice, module.TBModel = Lattice, TBModel
    else:
        module.tb_model = tb_model
    return module


def fake_tbmodels():
    module = types.ModuleType('tbmodels')

    class Model():
        @classmethod
        def from_hop_list(cls, **kwargs):
            model = cls()
            model.kwargs, model.files = kwargs, []
            return model

        def to_hdf5_file(self, path):
            self.files.append(path)

        def hamilton(self, k):
            '''Convention 2, as TBmodels.'''
            kw = self.kwargs
            ham = np.diag(kw['on_site']).astype(complex)
            for t, i, j, R in kw['hop_list']:
                term = t * np.exp(2j * np.pi * np.dot(k, R))
                ham[i, j] += term
                ham[j, i] += np.conj(term)
            return ham

    module.Model = Model
    return module


def fake_z2pack():
    module = types.ModuleType('z2pack')

    class System():
        def __init__(self, hamilton, **kwargs):
            self.hamilton, self.kwargs = hamilton, kwargs

    module.hm = types.SimpleNamespace(System=System)
    return module


def fake_qsymm():
    module = types.ModuleType('qsymm')

    class BlochModel(dict):
        def __init__(self, ham, momenta):
            super().__init__(ham)
            self.momenta = momenta

    module.BlochModel = BlochModel
    module.model = types.SimpleNamespace(BlochCoeff=lambda d, coeff: (tuple(d), coeff))
    return module


def fake_pybinding():
    '''A Lattice that stores its sublattices and hoppings as pybinding
    exposes them (positions, energy matrices, unique ids, hopping families
    of terms).'''
    module = types.ModuleType('pybinding')

    class Lattice():
        def __init__(self, *vectors):
            self.vectors = [np.array(v, dtype=float) for v in vectors]
            self.ndim = len(vectors)
            self.sublattices, self.hoppings = {}, {}

        def add_sublattices(self, *subs):
            for name, pos, energy in subs:
                self.sublattices[name] = types.SimpleNamespace(
                    position=np.array(pos + [0.] * (3 - len(pos))), energy=energy,
                    unique_id=len(self.sublattices), alias_id=len(self.sublattices))

        def add_hoppings(self, *hops):
            for R, a, b, energy in hops:
                term = types.SimpleNamespace(relative_index=np.array(R + [0] * (3 - len(R))),
                                             from_id=self.sublattices[a].unique_id,
                                             to_id=self.sublattices[b].unique_id)
                self.hoppings['h{}'.format(len(self.hoppings))] = types.SimpleNamespace(
                    energy=energy, terms=[term])

    module.Lattice = Lattice
    return module


def fake_kwant():
    '''Builder, SiteFamily, ModesLead, physics.modes and DiscreteSymmetry, as
    far as to_kwant uses them; modes records its arguments.'''
    module = types.ModuleType('kwant')
    module.calls = []

    class Site(tuple):
        @property
        def pos(self):
            return self[0].pos(self[1])

    class SiteFamily():
        def __init__(self, canonical_repr, name, norbs):
            self.canonical_repr, self.name, self.norbs = canonical_repr, name, norbs

        def __call__(self, *tag):
            return Site((self, tuple(self.normalize_tag(tag))))

    class Builder(dict):
        def __init__(self):
            super().__init__()
            self.leads = []

    class ModesLead():
        def __init__(self, modes_func, interface, parameters):
            self.modes_func, self.interface, self.parameters = modes_func, interface, parameters

    def modes(h_cell, h_hop, discrete_symmetry=None):
        module.calls.append((h_cell, h_hop, discrete_symmetry))
        return 'modes'

    module.Builder = Builder
    module.builder = types.SimpleNamespace(SiteFamily=SiteFamily, ModesLead=ModesLead)
    module.physics = types.SimpleNamespace(modes=modes, DiscreteSymmetry=lambda projectors: projectors)
    return module


def fake_tinyarray():
    module = types.ModuleType('tinyarray')
    module.array = tuple
    return module


def patched(**modules):
    return mock.patch.dict('sys.modules', modules)


def builder_matrix(builder, sites):
    '''The Hamiltonian of a (fake) Builder on *sites*, conjugates added.'''
    index = {s: p for p, s in enumerate(sites)}
    ham = np.zeros((len(sites), len(sites)), dtype=complex)
    for key, value in builder.items():
        if isinstance(key[0], tuple):  # a hopping (site, site)
            if key[0] not in index or key[1] not in index:
                continue
            ham[index[key[0]], index[key[1]]] += value
            ham[index[key[1]], index[key[0]]] += np.conj(value)
        elif key in index:
            ham[index[key], index[key]] += value
    return ham


# ----------------------------------------------------------------------------
# tests with the stand-ins
# ----------------------------------------------------------------------------

class TestHelpers(unittest.TestCase):

    def test_import(self):
        with self.assertRaises(ImportError):
            interop._import('tbkit_no_such_package')

    def test_default_tags(self):
        self.assertEqual(interop._default_tags(28)[:2] + interop._default_tags(28)[-2:],
                         ['a', 'b', 's26', 's27'])

    def test_exports_refuse(self):
        hn = KSpace(Lattice([{'tag': 'a', 'r0': (0., 0.)}], [(1., 0.)]))
        hn.set_hopping([{'i': 0, 'j': 0, 'R': (1,), 't': 1.}], hermitian=False)
        gra = models.haldane(t2=0.)
        gra.set_overlap([{'n': 1, 't': 0.1}])
        for ks, error in ((hn, ValueError), (gra, ValueError), ('model', TypeError)):
            for export in (interop.to_pythtb, interop.to_tbmodels, interop.z2pack_hamiltonian,
                           interop.to_z2pack, interop.to_qsymm, interop.to_pybinding):
                self.assertRaises(error, export, ks)


class TestASE(unittest.TestCase):

    def test_round_trip(self):
        ase = fake_ase()
        lat = Lattice([{'tag': 'C', 'r0': (0., 0.)}, {'tag': 'N', 'r0': (0., 1.)}],
                      [(np.sqrt(3), 0.), (np.sqrt(3) / 2, 1.5)])
        with patched(ase=ase):
            atoms = interop.lattice_to_ase(lat)
            np.testing.assert_array_equal(atoms.pbc, [True, True, False])
            np.testing.assert_allclose(atoms.cell[2], 0.)
            back = interop.lattice_from_ase(atoms)
            self.assertEqual(back.prim_vec, lat.prim_vec)
            self.assertEqual(back.unit_cell, lat.unit_cell)
            # 3D space on request, then the z coordinates are kept
            self.assertEqual(len(interop.lattice_from_ase(atoms, space_dim=3).prim_vec[0]), 3)
            # tags that are no chemical symbols need a map
            self.assertRaises(ValueError, interop.lattice_to_ase, Lattice([{'tag': 'a', 'r0': (0., 0.)}], [(1., 0.)]))
            atoms = interop.lattice_to_ase(Lattice([{'tag': 'a', 'r0': (0., 0.)}], [(1., 0.)]),
                                           symbols={'a': 'Si'})
            self.assertEqual(atoms.symbols, ['Si'])
            # the sites of a finite sample, as a cluster
            lat.get_lattice(3, 2)
            cluster = interop.lattice_to_ase(lat, sites=True)
            self.assertEqual(len(cluster.symbols), 12)
            self.assertFalse(cluster.pbc.any())
            self.assertRaises(RuntimeError, interop.lattice_to_ase, Lattice(lat.unit_cell, lat.prim_vec), sites=True)
            self.assertRaises(TypeError, interop.lattice_to_ase, lat, sites=1)
            # a buckled layer stays in 3D space, a bulk crystal too
            buckled = ase.Atoms(['Si', 'Si'], [[0, 0, 0], [1, 0.5, 0.4]], [[2, 0, 0], [1, 1.7, 0], [0, 0, 0]],
                                [True, True, False])
            self.assertEqual(len(interop.lattice_from_ase(buckled).prim_vec[0]), 3)
            self.assertRaises(ValueError, interop.lattice_from_ase, buckled, space_dim=2)
            self.assertRaises(ValueError, interop.lattice_from_ase, buckled, space_dim=1)
            bulk = ase.Atoms(['Si'], [[0, 0, 0]], np.eye(3) * 2, [True] * 3)
            self.assertEqual(len(interop.lattice_from_ase(bulk).prim_vec), 3)
            self.assertEqual(interop.lattice_from_ase(bulk, tags=['x']).unit_cell[0]['tag'], 'x')
            self.assertRaises(ValueError, interop.lattice_from_ase, bulk, tags=['x', 'y'])
            self.assertRaises(TypeError, interop.lattice_from_ase, bulk, tags='x')
            molecule = ase.Atoms(['H'], [[0, 0, 0]], np.zeros((3, 3)), [False] * 3)
            self.assertRaises(ValueError, interop.lattice_from_ase, molecule)
        self.assertRaises(TypeError, interop.lattice_to_ase, 'lattice')


class TestPythTB(unittest.TestCase):

    def test_round_trips(self):
        for version in (2, 1):
            with patched(pythtb=fake_pythtb(version)):
                for ks in sample_models():
                    model = interop.to_pythtb(ks)
                    back = interop.from_pythtb(model)
                    self.assertEqual(back.spin, ks.spin)
                    same_bands(self, ks, back)
                    np.testing.assert_allclose(back.orbital_positions(), ks.orbital_positions(), atol=1e-12)

    def test_import(self):
        # a PythTB 1.x chain in 1D space, placed along x in the plane
        chain = types.SimpleNamespace(_lat=[[1.]], _orb=[[0.], [0.4]], _per=[0], _nspin=1,
                                      _site_energies=[0.2, -0.2],
                                      _hoppings=[[0.7, 0, 1, np.array([0])], [0.3j, 1, 0, np.array([1])]])
        ks = interop.from_pythtb(chain, tags=['A', 'B'])
        self.assertEqual((ks.dim, ks.space_dim), (1, 2))
        self.assertEqual(list(ks.tags), ['A', 'B'])
        k = 0.8
        h = np.array([[0.2, 0.7 + np.conj(0.3j) * np.exp(-1j * k)], [0., -0.2]])
        h[1, 0] = np.conj(h[0, 1])
        np.testing.assert_allclose(ks.get_ham((k,)), h, atol=1e-14)
        bad = types.SimpleNamespace(**vars(chain))
        bad._site_energies = ['m', 0.]
        self.assertRaises(ValueError, interop.from_pythtb, bad)
        bad._site_energies, bad._hoppings = [0., 0.], [[lambda m: m, 0, 1, np.array([0])]]
        self.assertRaises(ValueError, interop.from_pythtb, bad)
        bad._per = []
        self.assertRaises(ValueError, interop.from_pythtb, bad)
        self.assertRaises(ValueError, interop.from_pythtb, chain, tags=['a'])


class TestTBmodelsZ2PackQsymm(unittest.TestCase):

    def test_tbmodels(self):
        with patched(tbmodels=fake_tbmodels()):
            for ks in sample_models():
                model = interop.to_tbmodels(ks, path='model.hdf5')
                self.assertEqual(model.files, ['model.hdf5'])
                self.assertFalse(model.kwargs['contains_cc'])
                for k in np.random.default_rng(6).random((4, ks.dim)):
                    np.testing.assert_allclose(model.hamilton(k), ks.get_ham(k @ ks.rec_vec_k), atol=1e-12)
                pos = model.kwargs['pos'] @ model.kwargs['uc']
                np.testing.assert_allclose(pos, ks.orbital_positions() @ ks.k_basis, atol=1e-12)
            self.assertEqual(interop.to_tbmodels(models.ssh()).files, [])
            self.assertRaises(TypeError, interop.to_tbmodels, models.ssh(), path=1)

    def test_z2pack(self):
        ks = models.haldane()
        with patched(z2pack=fake_z2pack()):
            system = interop.to_z2pack(ks, bands=[0], hermitian_tol=1e-8)
        self.assertEqual((system.kwargs['dim'], system.kwargs['bands'], system.kwargs['convention']), (2, [0], 2))
        self.assertEqual(system.kwargs['hermitian_tol'], 1e-8)
        np.testing.assert_allclose(system.kwargs['pos'] @ np.array(ks.lat.prim_vec), ks.orbital_positions())
        np.testing.assert_allclose(system.hamilton([0.2, 0.3]), ks.get_ham(np.array([0.2, 0.3]) @ ks.rec_vec_k))

    def test_qsymm(self):
        with patched(qsymm=fake_qsymm()):
            for ks in sample_models():
                model = interop.to_qsymm(ks)
                self.assertEqual(model.momenta, ['k_x', 'k_y', 'k_z'][:ks.dim])
                for k in np.random.default_rng(7).uniform(-2, 2, (3, ks.dim)):
                    ham = sum(mat * np.exp(1j * np.dot(k, d)) for (d, _), mat in model.items())
                    np.testing.assert_allclose(ham, gauge(ks, k), atol=1e-12)


class TestPybinding(unittest.TestCase):

    def test_round_trips(self):
        with patched(pybinding=fake_pybinding()):
            for ks in sample_models():
                lattice = interop.to_pybinding(ks)
                same_bands(self, ks, interop.from_pybinding(lattice, spin=ks.spin))
            # repeated tags are numbered
            ssh = models.ssh()
            ssh.tags[:] = 'a'
            self.assertEqual(list(interop.to_pybinding(ssh).sublattices), ['a0', 'a1'])

    def test_import(self):
        pb = fake_pybinding()
        lattice = pb.Lattice([1., 0.], [0.5, 0.8])
        lattice.add_sublattices(('A', [0., 0.], 0.3), ('B', [0.3, 0.2], np.array([[1., 0.1], [0.1, -1.]])))
        lattice.add_hoppings(([1, 0], 'A', 'A', -1.), ([0, 0], 'A', 'B', np.array([[0.2, 0.3j]])),
                             ([0, 1], 'B', 'B', np.array([[0.1, 0.2], [0.3, 0.4]])))
        ks = interop.from_pybinding(lattice)
        self.assertEqual(list(ks.tags), ['A', 'B0', 'B1'])
        self.assertEqual(ks.space_dim, 2)
        k = np.array([0.4, -0.9])
        ham = np.array([[0.3 - 2 * np.cos(k[0]), 0.2, 0.3j], [0.2, 1., 0.1], [-0.3j, 0.1, -1.]], dtype=complex)
        hop = np.array([[0.1, 0.2], [0.3, 0.4]]) * np.exp(1j * k @ [0.5, 0.8])
        ham[1:, 1:] += hop + hop.conj().T
        np.testing.assert_allclose(ks.get_ham(k), ham, atol=1e-14)
        self.assertRaises(ValueError, interop.from_pybinding, lattice, spin=True)
        self.assertRaises(TypeError, interop.from_pybinding, lattice, spin=1)
        lattice.sublattices['A'].alias_id = 1
        self.assertRaises(ValueError, interop.from_pybinding, lattice)
        # out of the plane: 3D space
        layer = pb.Lattice([1., 0., 0.], [0., 1., 0.])
        layer.add_sublattices(('A', [0., 0., 0.5], 0.))
        self.assertEqual(interop.from_pybinding(layer).space_dim, 3)
        # a finite Model: its matrix and orbital positions
        xyz = types.SimpleNamespace(x=[0., 1.], y=[0., 0.], z=[0., 0.])
        model = types.SimpleNamespace(hamiltonian=sp.csr_matrix(np.ones((2, 2))),
                                      system=types.SimpleNamespace(expanded_positions=xyz))
        ham, pos = interop.from_pybinding(model)
        self.assertEqual(ham.shape, (2, 2))
        np.testing.assert_array_equal(pos, [[0., 0., 0.], [1., 0., 0.]])


def strip_transport(length=8, width=3, disorder=1.):
    '''A disordered square-lattice strip with two ribbon leads.'''
    lat = lattices.square()
    lat.get_lattice(length, width)
    sys = System(lat)
    sys.set_hopping([{'n': 1, 't': -1.}])
    sys.set_onsite({'a': 0.})
    sys.onsite[:] = disorder * (np.random.default_rng(2).random(lat.sites) - 0.5)
    sys.get_ham()
    rib = ribbon(lattices.square(), [{'i': 0, 'j': 0, 'R': (1, 0), 't': -1.},
                                     {'i': 0, 'j': 0, 'R': (0, 1), 't': -1.}], width=width, direction=1)
    tr = Transport(sys.ham)
    tr.attach_lead(sys, rib, -1)
    tr.attach_lead(sys, rib, 1)
    return sys, tr


class TestKwant(unittest.TestCase):

    def test_system(self):
        sys, _ = strip_transport()
        with patched(kwant=fake_kwant(), tinyarray=fake_tinyarray()):
            builder = interop.to_kwant(sys)
        sites = sorted((k for k in builder if not isinstance(k[0], tuple)), key=lambda s: s[1])
        np.testing.assert_allclose(builder_matrix(builder, sites), sys.ham.toarray(), atol=1e-14)
        np.testing.assert_allclose([s.pos for s in sites], sys._coords())
        self.assertEqual(builder.leads, [])
        empty = System(sys.lat)
        self.assertRaises(RuntimeError, interop.to_kwant, empty)
        gain = System(sys.lat)
        gain.set_hopping([{'n': 1, 't': -1.}])
        gain.set_onsite({'a': 0.1j})
        gain.get_ham()
        self.assertRaises(ValueError, interop.to_kwant, gain)
        self.assertRaises(TypeError, interop.to_kwant, 'system')

    def test_transport(self):
        sys, tr = strip_transport()
        tr._laws[1] = [np.eye(3)[:, [1]], np.eye(3)[:, [0, 2]]]  # blocks of a conservation law
        kwant = fake_kwant()
        with patched(kwant=kwant, tinyarray=fake_tinyarray()):
            builder = interop.to_kwant(tr, sys)
            no_positions = interop.to_kwant(tr)
        device = [k for k in builder if not isinstance(k[0], tuple) and k[0].name == 'device']
        device.sort(key=lambda s: s[1])
        np.testing.assert_allclose([s.pos for s in device], sys._coords())
        self.assertEqual(len(builder.leads), 2)
        for l, lead in enumerate(builder.leads):
            h0, v, tau, lead_sites = tr.leads[l]
            cell = list(lead.interface)
            ham = builder_matrix(builder, device + cell)
            n = len(device)
            np.testing.assert_allclose(ham[n:, n:], h0, atol=1e-14)
            np.testing.assert_allclose(ham[np.ix_(lead_sites, range(n, n + len(cell)))], tau, atol=1e-14)
            self.assertEqual(lead.modes_func(0.3), 'modes')
            h_cell, h_hop, symmetry = kwant.calls[-1]
            np.testing.assert_allclose(h_cell, h0 - 0.3 * np.eye(len(h0)))
            np.testing.assert_allclose(h_hop, v.conj().T)
            self.assertEqual(symmetry is None, l == 0)
        # without the System, site i sits at (i, 0) and a lead cell at the mean of its sites
        np.testing.assert_allclose(no_positions.leads[0].interface[0].pos, [np.mean(tr.leads[0][3]), 0.])
        self.assertRaises(ValueError, interop.to_kwant, tr, System(lattices.square()))

    def test_from_kwant(self):
        ham = sp.csr_matrix(np.arange(9.).reshape(3, 3))
        sites = [types.SimpleNamespace(pos=(0., 0.)), types.SimpleNamespace(pos=(1., 0.))]
        fsys = types.SimpleNamespace(hamiltonian_submatrix=lambda sparse, params: ham, sites=sites,
                                     site_ranges=[(0, 2, 0), (1, 1, 2), (2, 0, 3)])
        h, pos = interop.from_kwant(fsys)
        np.testing.assert_array_equal(h.toarray(), ham.toarray())
        np.testing.assert_array_equal(pos, [[0., 0.], [0., 0.], [1., 0.]])
        fsys.site_ranges = None
        self.assertRaises(ValueError, interop.from_kwant, fsys)
        fsys.sites = sites + [types.SimpleNamespace(pos=(2., 0.))]
        self.assertEqual(interop.from_kwant(fsys)[1].shape, (3, 2))
        fsys.cell_size = 1  # a lead
        self.assertRaises(TypeError, interop.from_kwant, fsys)
        self.assertRaises(TypeError, interop.from_kwant, 'system')


# ----------------------------------------------------------------------------
# the packages themselves, where installed
# ----------------------------------------------------------------------------

class Quiet(unittest.TestCase):
    '''Their own deprecation warnings are not ours.'''

    def setUp(self):
        catcher = warnings.catch_warnings()
        catcher.__enter__()
        self.addCleanup(catcher.__exit__, None, None, None)
        warnings.simplefilter('ignore')


@unittest.skipUnless(installed('pythtb'), 'PythTB is not installed')
class TestWithPythTB(Quiet):

    def test_models(self):
        import pythtb
        rng = np.random.default_rng(8)
        for ks in sample_models():
            model = interop.to_pythtb(ks)
            kr = rng.random((20, ks.dim))
            if hasattr(pythtb, 'TBModel'):
                e = model.solve_ham(kr, flatten_spin_axis=True)
            else:
                e = np.asarray(model.solve_all(kr)).reshape(ks.norb, -1).T
            np.testing.assert_allclose(np.sort(e, axis=1), ks.get_bands(kr @ ks.rec_vec_k), atol=1e-12)
            same_bands(self, ks, interop.from_pythtb(model))
        if hasattr(pythtb, 'TBModel'):
            from pythtb.models import haldane, kane_mele
            hal = haldane(0.2, -1., 0.15)
            ks = interop.from_pythtb(hal)
            self.assertAlmostEqual(ks.chern_number(0, nk=40),
                                   hal.chern_number((0, 1), (40, 40), occ_idxs=[0]), places=8)
            for delta, z2 in ((0.7, 1), (4.5, 0)):
                self.assertEqual(interop.from_pythtb(kane_mele(delta, -1., 0.24, 0.05)).z2_invariant([0, 1]), z2)


@unittest.skipUnless(installed('tbmodels'), 'TBmodels is not installed')
class TestWithTBmodels(Quiet):

    def test_bands(self):
        for ks in sample_models()[:3]:
            model = interop.to_tbmodels(ks)
            kr = np.random.default_rng(9).random((10, ks.dim))
            np.testing.assert_allclose(model.eigenval(kr), ks.get_bands(kr @ ks.rec_vec_k), atol=1e-12)


@unittest.skipUnless(installed('z2pack'), 'Z2Pack is not installed')
class TestWithZ2Pack(Quiet):

    def test_invariants(self):
        import logging
        import z2pack
        logging.disable(logging.CRITICAL)
        self.addCleanup(logging.disable, logging.NOTSET)
        hal = models.haldane(t2=0.15, phi=np.pi / 2, M=0.2)
        result = z2pack.surface.run(system=interop.to_z2pack(hal, bands=[0]),
                                    surface=lambda t1, t2: [t2, t1], min_neighbour_dist=1e-4)
        self.assertAlmostEqual(z2pack.invariant.chern(result), round(hal.chern_number(0, nk=40)), places=6)
        km = models.kane_mele()
        result = z2pack.surface.run(system=interop.to_z2pack(km, bands=[0, 1]),
                                    surface=lambda t1, t2: [t1 / 2, t2], min_neighbour_dist=1e-4)
        self.assertEqual(z2pack.invariant.z2(result), km.z2_invariant([0, 1]))


@unittest.skipUnless(installed('qsymm'), 'qsymm is not installed')
class TestWithQsymm(Quiet):

    def test_graphene(self):
        import qsymm
        from tbkit.symmetry import find_symmetries
        gra = models.haldane(t2=0., M=0.)
        group, _ = qsymm.symmetries(interop.to_qsymm(gra), qsymm.groups.hexagonal(dim=2, ph=False))
        self.assertEqual(len(group), len(find_symmetries(gra, antiunitary=True)))


@unittest.skipUnless(installed('pybinding'), 'pybinding is not installed')
class TestWithPybinding(Quiet):

    def test_bands(self):
        import pybinding as pb
        for ks in sample_models()[:2]:
            model = pb.Model(interop.to_pybinding(ks), pb.translational_symmetry())
            for k in np.random.default_rng(10).random((5, 2)) @ ks.rec_vec_k:
                model.set_wave_vector(list(-k) + [0.])
                np.testing.assert_allclose(np.linalg.eigvalsh(model.hamiltonian.toarray()),
                                           ks.get_bands(k)[0], atol=1e-5)  # single precision


@unittest.skipUnless(installed('kwant'), 'Kwant is not installed')
class TestWithKwant(Quiet):

    def test_transmissions(self):
        import kwant
        sys, tr = strip_transport(10, 4, 1.5)
        fsys = interop.to_kwant(tr, sys).finalized()
        for e in (-1.1, 0.4, 1.7):
            self.assertAlmostEqual(kwant.smatrix(fsys, e).transmission(1, 0), tr.transmission([e])[0], places=10)
        ham, pos = interop.from_kwant(interop.to_kwant(sys).finalized())
        self.assertAlmostEqual(abs(ham - sys.ham).max(), 0.)
        np.testing.assert_allclose(pos, sys._coords())

    def test_conservation_law(self):
        import kwant
        w = 3
        h0 = np.kron(-(np.eye(w, k=1) + np.eye(w, k=-1)), np.eye(2)) + np.kron(np.eye(w), np.diag([0.3, -0.3]))
        v = np.kron(-np.eye(w), np.eye(2)) * np.exp(0.3j)
        cells = [list(range(2 * w * c, 2 * w * (c + 1))) for c in range(6)]
        dev = np.zeros((12 * w, 12 * w), dtype=complex)
        flip = 0.4 * np.kron(np.eye(w), [[0, 1], [1, 0]])
        for c in range(6):
            dev[np.ix_(cells[c], cells[c])] = h0 + (flip if 1 < c < 4 else 0)
            if c < 5:
                dev[np.ix_(cells[c], cells[c + 1])] = v
                dev[np.ix_(cells[c + 1], cells[c])] = v.conj().T
        law = np.kron(np.eye(w), np.diag([1., -1.]))
        tr = Transport(dev)
        tr.add_lead(h0, v, v.conj().T, cells[0], law)
        tr.add_lead(h0, v, v, cells[5], law)
        s_kw = kwant.smatrix(interop.to_kwant(tr).finalized(), 0.5)
        s_tb = tr.smatrix(0.5)
        for a in range(2):
            for b in range(2):
                self.assertAlmostEqual(s_kw.transmission((1, a), (0, b)), s_tb.transmission((1, a), (0, b)),
                                       places=10)


@unittest.skipUnless(installed('ase'), 'ASE is not installed')
class TestWithASE(Quiet):

    def test_structures(self):
        from ase.build import bulk, graphene, mx2
        lat = interop.lattice_from_ase(graphene(a=2.46, vacuum=None))
        self.assertEqual(len(lat.prim_vec[0]), 2)
        self.assertEqual([d['tag'] for d in lat.unit_cell], ['C', 'C'])
        self.assertEqual(len(interop.lattice_from_ase(mx2('MoS2')).prim_vec[0]), 3)
        si = interop.lattice_from_ase(bulk('Si'))
        atoms = interop.lattice_to_ase(si)
        np.testing.assert_allclose(atoms.get_cell()[:], bulk('Si').get_cell()[:])


if __name__ == '__main__':
    unittest.main()
