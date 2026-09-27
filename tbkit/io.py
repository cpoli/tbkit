r'''
Save and load models, and import models from other codes.

*save_model* / *load_model* store a **Lattice**, **System** or **KSpace**
in a versioned NumPy ``.npz`` archive (no pickle, readable with NumPy
alone): the lattice (*unit_cell*, *prim_vec*, and the sites of
*get_lattice* if any), and for a System its onsite energies and hopping
array *sys.hop*, for a KSpace its onsite terms (with the spin-off-diagonal
block), hoppings, overlaps and *spin*. Every number is stored in binary,
so a loaded model reproduces the Hamiltonians exactly.

Archive layout (format ``'tbkit-model'``, version :data:`VERSION`):

* ``format``, ``version``, ``kind`` (``'Lattice'``, ``'System'`` or
  ``'KSpace'``), ``tbkit_version``;
* ``uc_tag`` (n_sites,), ``uc_r0`` (n_sites, space_dim), ``prim_vec``
  (dim, space_dim), ``coor`` (the structured site array), ``n_cells`` (3,);
* System: ``onsite`` (sites,), ``hop`` (the structured *sys.hop*);
* KSpace: ``spin``, ``onsite`` (norb,), ``onsite_offdiag`` (norb, norb),
  ``nonreciprocal``, and ``hop_i``, ``hop_j``, ``hop_R`` (Cartesian
  lattice vectors), ``hop_t`` for the hoppings, ``ovl_i``, ``ovl_j``,
  ``ovl_R``, ``ovl_t`` for the overlaps.

*read_wannier90* imports a tight-binding model from Wannier90's
``seedname_hr.dat`` (see there).
'''
from __future__ import annotations

import os

import numpy as np

import tbkit
import tbkit.error_handling as error_handling
from tbkit.kspace import KSpace
from tbkit.lattice import Lattice
from tbkit.orbital import OrbitalSystem
from tbkit.system import System


#: Name of the archive format written by *save_model*.
FORMAT = 'tbkit-model'
#: Version of the archive format written by *save_model*; *load_model*
#: reads this version and every earlier one.
VERSION = 1


def _hop_arrays(hops: list, space_dim: int, prefix: str) -> dict:
    '''
    Private function. A KSpace hopping list (i, j, R_cart, t) as arrays.
    '''
    return {prefix + '_i': np.array([h[0] for h in hops], dtype='i8'),
            prefix + '_j': np.array([h[1] for h in hops], dtype='i8'),
            prefix + '_R': np.array([h[2] for h in hops], dtype='f8').reshape(len(hops), space_dim),
            prefix + '_t': np.array([h[3] for h in hops], dtype='c16')}


def _hop_list(data, prefix: str) -> list:
    '''
    Private function. Inverse of *_hop_arrays*.
    '''
    return [(int(i), int(j), R.copy(), complex(t))
               for i, j, R, t in zip(data[prefix + '_i'], data[prefix + '_j'],
                                             data[prefix + '_R'], data[prefix + '_t'])]


def save_model(model: Lattice | System | KSpace, path: str | os.PathLike) -> str:
    '''
    Save a **Lattice**, **System** or **KSpace** to a ``.npz`` archive (see
    the module docstring for its layout). Subclasses are saved as their
    base class (e.g. a *GrapheneSystem* as a *System*); an *OrbitalSystem*
    or a driven (Floquet) model cannot be saved.

    :param model: **Lattice**, **System** or **KSpace** instance.
    :param path: String or path. File name; ``.npz`` is appended if missing.

    :returns:
        * **path** -- String. The file written.

    Example usage::

        save_model(gra, 'graphene.npz')
        gra2 = load_model('graphene.npz')
    '''
    error_handling.saveable_model(model, Lattice, System, KSpace, OrbitalSystem)
    error_handling.file_path(path, 'path')
    path = os.fspath(path)
    if not path.endswith('.npz'):
        path += '.npz'
    lat = model.lat if isinstance(model, (System, KSpace)) else model
    kind = 'System' if isinstance(model, System) else 'KSpace' if isinstance(model, KSpace) else 'Lattice'
    data = {'format': np.array(FORMAT), 'version': np.array(VERSION), 'kind': np.array(kind),
                'tbkit_version': np.array(tbkit.__version__),
                'uc_tag': np.array([dic['tag'] for dic in lat.unit_cell], dtype='U1'),
                'uc_r0': np.array([dic['r0'] for dic in lat.unit_cell], dtype='f8'),
                'prim_vec': np.array(lat.prim_vec, dtype='f8'),
                'coor': lat.coor,
                'n_cells': np.array([lat.n1, lat.n2, lat.n3], dtype='i8')}
    if kind == 'System':
        data['onsite'] = np.asarray(model.onsite, dtype='c16')
        data['hop'] = model.hop
    elif kind == 'KSpace':
        data.update({'spin': np.array(model.spin), 'onsite': model.onsite,
                          'onsite_offdiag': model._onsite_offdiag,
                          'nonreciprocal': np.array(model._nonreciprocal)})
        data.update(_hop_arrays(model._hop, model.space_dim, 'hop'))
        data.update(_hop_arrays(model._overlap_hop, model.space_dim, 'ovl'))
    np.savez(path, **data)
    return path


def load_model(path: str | os.PathLike) -> Lattice | System | KSpace:
    '''
    Load a model saved by *save_model*.

    :param path: String or path. File name.

    :returns:
        * **model** -- **Lattice**, **System** or **KSpace** instance, whose
          Hamiltonians (*get_ham*) are exactly those of the saved model.
    '''
    error_handling.file_path(path, 'path')
    with np.load(path, allow_pickle=False) as data:
        error_handling.model_archive(data, FORMAT, VERSION)
        unit_cell = [{'tag': str(tag), 'r0': tuple(float(c) for c in r0)}
                          for tag, r0 in zip(data['uc_tag'], data['uc_r0'])]
        prim_vec = [tuple(float(c) for c in a) for a in data['prim_vec']]
        lat = Lattice(unit_cell=unit_cell, prim_vec=prim_vec)
        lat.coor = data['coor'].astype(lat.dtype)
        lat.sites = len(lat.coor)
        lat.n1, lat.n2, lat.n3 = (int(n) for n in data['n_cells'])
        if lat.sites:
            lat.tags = np.unique(np.concatenate([lat.tags, lat.coor['tag']]))
        kind = str(data['kind'])
        if kind == 'Lattice':
            return lat
        if kind == 'System':
            sys = System(lat)
            sys.onsite = data['onsite'].copy()
            sys.hop = data['hop'].copy()
            return sys
        ks = KSpace(lat, spin=bool(data['spin']))
        ks.onsite = data['onsite'].copy()
        ks._onsite_offdiag = data['onsite_offdiag'].copy()
        ks._nonreciprocal = bool(data['nonreciprocal'])
        ks._hop = _hop_list(data, 'hop')
        ks._overlap_hop = _hop_list(data, 'ovl')
        return ks
