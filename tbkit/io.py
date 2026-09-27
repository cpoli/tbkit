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
from numpy.typing import ArrayLike, NDArray

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


#################################
# WANNIER90
#################################

#: Bohr radius in Angstrom (CODATA 2018), for ``unit_cell_cart`` given in bohr.
BOHR = 0.529177210903


def read_hr(path: str | os.PathLike) -> tuple[int, NDArray[np.int64], NDArray[np.int64], NDArray[np.complex128]]:
    r'''
    Read a Wannier90 ``seedname_hr.dat`` file: a header line, the number of
    Wannier functions, the number of lattice vectors :math:`\mathbf{R}`,
    their degeneracy weights (15 per line), then one line
    ``R1 R2 R3 m n Re Im`` per matrix element
    :math:`H_{mn}(\mathbf{R}) = \langle m\mathbf{0}|H|n\mathbf{R}\rangle`
    (m, n counted from 1).

    :param path: String or path. The ``_hr.dat`` file.

    :returns:
        * **num_wann** -- Integer. Number of Wannier functions.
        * **R** -- Integer ndarray, shape (nrpts, 3). Lattice vectors (in
          units of the primitive vectors).
        * **ndegen** -- Integer ndarray, shape (nrpts,). Degeneracy weights.
        * **ham** -- Complex ndarray, shape (nrpts, num_wann, num_wann).
          :math:`H_{mn}(\mathbf{R})` as written in the file (not divided by
          the weights).
    '''
    error_handling.file_path(path, 'path')
    with open(path) as f:
        lines = f.read().splitlines()[1:]  # the first line is a free comment
    tokens = ' '.join(lines).split()
    error_handling.hr_header(tokens)
    num_wann, nrpts = int(tokens[0]), int(tokens[1])
    ndegen = np.array(tokens[2:2 + nrpts], dtype='i8')
    body = tokens[2 + nrpts:]
    error_handling.hr_body(len(body), nrpts, num_wann)
    rows = np.array(body, dtype='f8').reshape(nrpts * num_wann ** 2, 7)
    R = rows[::num_wann ** 2, :3].astype('i8')
    block = np.repeat(np.arange(nrpts), num_wann ** 2)
    error_handling.hr_blocks(np.array_equal(rows[:, :3].astype('i8'), R[block]))
    m, n = rows[:, 3].astype('i8') - 1, rows[:, 4].astype('i8') - 1
    ham = np.zeros((nrpts, num_wann, num_wann), 'c16')
    ham[block, m, n] = rows[:, 5] + 1j * rows[:, 6]
    return num_wann, R, ndegen, ham


def read_win_cell(path: str | os.PathLike) -> NDArray[np.float64]:
    r'''
    Read the lattice vectors of a Wannier90 ``seedname.win`` file (block
    ``unit_cell_cart``, in Angstrom, or in bohr if its first line says
    ``bohr``). Comments (``!``, ``#``) are ignored.

    :param path: String or path. The ``.win`` file.

    :returns:
        * **cell** -- Real ndarray, shape (3, 3). The rows are
          :math:`\mathbf{a}_1, \mathbf{a}_2, \mathbf{a}_3`, in Angstrom.
    '''
    error_handling.file_path(path, 'path')
    with open(path) as f:
        lines = [line.split('!')[0].split('#')[0].strip() for line in f]
    lines = [line for line in lines if line]
    low = [line.lower().replace(' ', '') for line in lines]
    error_handling.win_block('beginunit_cell_cart' in low and 'endunit_cell_cart' in low)
    block = lines[low.index('beginunit_cell_cart') + 1:low.index('endunit_cell_cart')]
    scale = 1.
    if block and block[0].lower() in ('bohr', 'ang'):
        scale = BOHR if block[0].lower() == 'bohr' else 1.
        block = block[1:]
    cell = [line.split() for line in block]
    error_handling.win_cell(cell)
    return scale * np.array(cell, dtype='f8')


def read_centres(path: str | os.PathLike, num_wann: int) -> NDArray[np.float64]:
    '''
    Read the Wannier centres of a Wannier90 ``seedname_centres.xyz`` file
    (the first *num_wann* atoms labelled ``X``, in Angstrom).

    :param path: String or path. The ``_centres.xyz`` file.
    :param num_wann: Positive integer. Number of Wannier functions.

    :returns:
        * **centres** -- Real ndarray, shape (num_wann, 3).
    '''
    error_handling.file_path(path, 'path')
    error_handling.positive_int(num_wann, 'num_wann')
    with open(path) as f:
        lines = f.read().splitlines()[2:]  # atom count and comment
    xs = [line.split()[1:4] for line in lines if line.split() and line.split()[0] == 'X']
    error_handling.centres_count(len(xs), num_wann)
    return np.array(xs[:num_wann], dtype='f8')


def read_wannier90(
    hr: str | os.PathLike,
    prim_vec: list[tuple[float, ...]] | None = None,
    win: str | os.PathLike | None = None,
    centres: str | os.PathLike | None = None,
    positions: ArrayLike | None = None,
    tags: list[str] | None = None,
    dim: int = 3,
    cutoff: float = 0.,
    tol: float = 1e-6,
) -> KSpace:
    r'''
    Import a tight-binding model from Wannier90's ``seedname_hr.dat``,
    the Hamiltonian in a basis of (maximally localized) Wannier functions:

    .. math::

        H_{mn}(\mathbf{k}) = \sum_{\mathbf{R}} \frac{H_{mn}(\mathbf{R})}{N_{\mathbf{R}}}\,
        e^{i\mathbf{k}\cdot\mathbf{R}}\, ,

    :math:`N_{\mathbf{R}}` being the degeneracy weight of :math:`\mathbf{R}`
    (lattice vectors on the boundary of the Wigner-Seitz supercell appear
    several times). The Wannier functions become the orbitals of a
    **KSpace**, in the file's order: :math:`H_{mm}(\mathbf{0})` the onsite
    energies, the other elements the hoppings
    :math:`t_{mn}(\mathbf{R})` of *KSpace.set_hopping*. If the file is
    Hermitian (within *tol*), one element of each conjugate pair is kept
    and the model is Hermitian (with real onsite energies); otherwise every
    element is kept, with ``hermitian=False``.

    :param hr: String or path. The ``seedname_hr.dat`` file.
    :param prim_vec: List of *dim* tuples of *dim* real numbers. Default
        value None. Primitive vectors, in the units of the positions
        (Angstrom for Wannier90). If None, read from *win*.
    :param win: String or path. Default value None. The ``seedname.win``
        file, for its ``unit_cell_cart`` block (if *prim_vec* is None).
    :param centres: String or path. Default value None. The
        ``seedname_centres.xyz`` file: the Wannier centres become the
        orbital positions (used by the Wilson-loop tools, the Peierls
        phases, *bridges.finite_system*, ...).
    :param positions: Array, shape (num_wann, dim) or (num_wann, 3).
        Default value None. Orbital positions, instead of *centres*. With
        neither, every orbital sits at the origin of the cell.
    :param tags: List of num_wann one-character strings. Default value
        None (every orbital tagged ``'a'``). Sublattice tags.
    :param dim: 2 or 3. Default value 3. A 2D model (a layer) keeps only
        :math:`(R_1, R_2)`: every element with :math:`R_3\neq 0` must vanish
        (within *tol*), and the lattice lies in the :math:`(x, y)` plane
        (the z components of the vectors read from *win* must vanish; those
        of the positions are dropped).
    :param cutoff: Positive real number or zero. Default value 0. Drop the
        hoppings with :math:`|t| \le` *cutoff* (the exact zeros by default).
    :param tol: Positive real number. Default value 1e-6. Tolerance of the
        Hermiticity and 2D checks (the file has 6 decimals).

    :returns:
        * **ks** -- **KSpace** instance, with num_wann orbitals.

    Example usage::

        ks = read_wannier90('graphene_hr.dat', win='graphene.win',
                                     centres='graphene_centres.xyz', dim=2)
    '''
    error_handling.file_path(hr, 'hr')
    error_handling.wannier_dim(dim)
    error_handling.positive_real_zero(cutoff, 'cutoff')
    error_handling.positive_real(tol, 'tol')
    num_wann, R, ndegen, ham = read_hr(hr)
    if prim_vec is None:
        error_handling.win_given(win)
        cell = read_win_cell(win)
        error_handling.layer_cell(cell, dim)
        prim_vec = [tuple(float(c) for c in a[:dim]) for a in cell[:dim]]
    error_handling.prim_vec(prim_vec)
    error_handling.wannier_prim_vec(prim_vec, dim)
    if positions is None and centres is not None:
        positions = read_centres(centres, num_wann)
    if positions is None:
        positions = np.zeros((num_wann, dim))
    positions = np.asarray(positions, dtype='f8')
    error_handling.wannier_positions(positions.shape, num_wann, dim)
    tags = ['a'] * num_wann if tags is None else tags
    error_handling.wannier_tags(tags, num_wann)
    if dim == 2:
        error_handling.layer_hoppings(float(np.abs(ham[R[:, 2] != 0]).max(initial=0.)), tol)
        keep = R[:, 2] == 0
        R, ndegen, ham = R[keep, :2], ndegen[keep], ham[keep]
    ham = ham / ndegen[:, None, None]
    unit_cell = [{'tag': tag, 'r0': tuple(float(c) for c in r[:dim])} for tag, r in zip(tags, positions)]
    ks = KSpace(Lattice(unit_cell=unit_cell, prim_vec=list(prim_vec)))
    # Hermitian: every H(R) has its partner H(-R) = H(R)^dagger
    index = {tuple(r): p for p, r in enumerate(R.tolist())}
    partner = [index.get(tuple(-c for c in r)) for r in R.tolist()]
    hermitian = None not in partner and bool(
        np.abs(ham - ham[partner].conj().transpose(0, 2, 1)).max() <= tol)
    list_hop = []
    for r, h in zip(R.tolist(), ham):
        nonzero = [c for c in r if c != 0]
        for m in range(num_wann):
            for n in range(num_wann):
                t = complex(h[m, n])
                if m == n and not nonzero:
                    ks.onsite[m] = t.real if hermitian else t
                elif abs(t) > cutoff and (not hermitian or (nonzero and nonzero[0] > 0)
                                                    or (not nonzero and m < n)):
                    list_hop.append({'i': m, 'j': n, 'R': tuple(r), 't': t})
    ks.set_hopping(list_hop, hermitian=hermitian)
    return ks
