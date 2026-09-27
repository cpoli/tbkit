r"""
The Tenfold Way: Symmetry Classes of Tight-Binding Models
===============================================================

A. Altland and M. Zirnbauer (1997) completed Dyson's classification of
random matrices: a Hamiltonian's non-spatial symmetries -- time reversal
:math:`T = U_T\mathcal{K}`, particle-hole :math:`C = U_C\mathcal{K}` (each
squaring to :math:`+1` or :math:`-1`, or absent) and their product, the
chiral symmetry :math:`S` -- sort it into exactly ten classes. Schnyder,
Ryu, Furusaki and Ludwig, and Kitaev, then showed (2008-2009) that the
class and the dimension alone decide which topological invariant a
gapped phase can carry: the *periodic table* of topological insulators
and superconductors. In 1D, for instance, classes AIII and BDI carry an
integer winding number, D a :math:`\mathbb{Z}_2`; in 2D, classes A and D
carry a Chern number, AII a :math:`\mathbb{Z}_2`.

:meth:`~tbkit.kspace.KSpace.symmetry_error` checks each symmetry on the
Bloch Hamiltonian, and :meth:`~tbkit.kspace.KSpace.tenfold_class` names
the class of the models of this gallery.
"""
import numpy as np

import tbkit.lattices as lattices
from tbkit.lattice import Lattice
from tbkit.kspace import KSpace, PAULI
from tbkit.bdg import bdg_kspace, particle_hole


def ssh(v, w):
    lat = Lattice(unit_cell=[{'tag': 'a', 'r0': (0., 0.)}, {'tag': 'b', 'r0': (0.5, 0.)}],
                         prim_vec=[(1., 0.)])
    model = KSpace(lat)
    model.set_hopping([{'i': 0, 'j': 1, 'R': (0,), 't': v}, {'i': 1, 'j': 0, 'R': (1,), 't': w}])
    return model


def honeycomb_model(t2=0., lam=0., spin=False):
    model = KSpace(lattices.honeycomb(), spin=spin)
    model.set_hopping([{'i': 0, 'j': 1, 'R': R, 't': 1.} for R in [(0, 0), (-1, 0), (0, -1)]])
    for R in [(0, 1), (-1, 0), (1, -1)]:
        if spin:
            model.set_hopping([{'i': 0, 'j': 0, 'R': R, 't': 1j*lam*PAULI['z']},
                                        {'i': 1, 'j': 1, 'R': R, 't': -1j*lam*PAULI['z']}])
        else:
            model.set_hopping([{'i': 0, 'j': 0, 'R': R, 't': 1j*t2}, {'i': 1, 'j': 1, 'R': R, 't': -1j*t2}])
    return model


def kitaev(delta):
    chain = KSpace(lattices.chain())
    chain.set_hopping([{'i': 0, 'j': 0, 'R': (1,), 't': 1.}])
    return bdg_kspace(chain, [{'i': 0, 'j': 0, 'R': (1,), 'delta': delta}], mu=0.5)


SZ = PAULI['z']
T_SPINLESS = np.eye(2)
T_SPINFUL = np.kron(np.eye(2), 1j * PAULI['y'])

# %%
# Time reversal, particle-hole, chiral: which hold?
# --------------------------------------------------------
# Complex hoppings break the (spinless) time reversal of the SSH chain,
# but not its sublattice (chiral) symmetry; a complex hopping phase breaks
# the Kitaev chain's time reversal, leaving particle-hole symmetry alone.

lat = Lattice(unit_cell=[{'tag': 'a', 'r0': (0., 0.)}, {'tag': 'b', 'r0': (0.5, 0.)}], prim_vec=[(1., 0.)])
ssh_complex = KSpace(lat)
ssh_complex.set_hopping([{'i': 0, 'j': 1, 'R': (0,), 't': 0.5j}, {'i': 1, 'j': 0, 'R': (1,), 't': 1.}])
chain = KSpace(lattices.chain())
chain.set_hopping([{'i': 0, 'j': 0, 'R': (1,), 't': np.exp(0.4j)}])
kitaev_complex = bdg_kspace(chain, [{'i': 0, 'j': 0, 'R': (1,), 'delta': 0.6j}], mu=0.5)

for model in (ssh_complex, kitaev_complex, honeycomb_model(t2=0.2)):
    assert model.symmetry_error(np.eye(model.norb), antiunitary=True) > 0.1

cases = [
    ('SSH chain', ssh(0.5, 1.), dict(time_reversal=T_SPINLESS, chiral=SZ), 'BDI'),
    ('SSH chain, complex hoppings', ssh_complex, dict(chiral=SZ), 'AIII'),
    ('Haldane model', honeycomb_model(t2=0.2), {}, 'A'),
    ('graphene (spinless)', honeycomb_model(), dict(time_reversal=np.eye(2), chiral=SZ), 'BDI'),
    ('Kane-Mele model', honeycomb_model(lam=0.06, spin=True), dict(time_reversal=T_SPINFUL), 'AII'),
    ('Kitaev chain', kitaev(0.6), dict(time_reversal=np.eye(2), particle_hole=particle_hole(1)), 'BDI'),
    ('Kitaev chain, complex hopping', kitaev_complex, dict(particle_hole=particle_hole(1)), 'D'),
]
for name, model, symmetries, expected in cases:
    cls = model.tenfold_class(**symmetries)
    print('{:<32s} class {}'.format(name, cls))
    assert cls == expected
