"""Translate a PythTB 2.0 ``TBModel`` (2D, periodic) into a tbkit ``KSpace``.

Only the model data (lattice, orbital positions, onsite terms and hoppings
with their lattice vectors) is carried over; every diagnostic is then
computed separately by each package.
"""
import numpy as np

from tbkit.kspace import KSpace
from tbkit.lattice import Lattice

TAGS = 'abcdefghijklmnopqrstuvwxyz'


def to_kspace(model):
    lat_vecs = np.asarray(model.lat_vecs, float)
    orbs = np.asarray(model.get_orb_vecs(cartesian=True), float)
    unit_cell = [{'tag': TAGS[o], 'r0': tuple(r)} for o, r in enumerate(orbs)]
    lat = Lattice(unit_cell=unit_cell, prim_vec=[tuple(a) for a in lat_vecs])
    ks = KSpace(lat, spin=model.spinful)
    ks.set_onsite({TAGS[o]: _value(v) for o, v in enumerate(model.onsite)})
    hops = []
    for h in model.hoppings:
        R = tuple(int(n) for n in h.get('lattice_vector', (0,) * len(lat_vecs)))
        hops.append({'i': h['from_orbital'], 'j': h['to_orbital'], 'R': R,
                     't': _value(h['amplitude'])})
    ks.set_hopping(hops)
    return ks


def _value(v):
    """A plain number, or a 2x2 spin matrix."""
    v = np.asarray(v)
    return complex(v) if v.ndim == 0 else v.astype(complex)


def reciprocal(model):
    """Rows b_i with a_i . b_j = 2 pi delta_ij: k_cart = k_red @ reciprocal(model)."""
    return 2 * np.pi * np.linalg.inv(np.asarray(model.lat_vecs, float)).T
