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
* System: ``onsite`` (sites,), ``hop`` (the structured *sys.hop*), and,
  from version 2, ``periodic`` (one boolean per primitive vector) for a
  System with periodic boundaries (every other model is still written as
  version 1, which older tbkit versions read);
* KSpace: ``spin``, ``onsite`` (norb,), ``onsite_offdiag`` (norb, norb),
  ``nonreciprocal``, and ``hop_i``, ``hop_j``, ``hop_R`` (Cartesian
  lattice vectors), ``hop_t`` for the hoppings, ``ovl_i``, ``ovl_j``,
  ``ovl_R``, ``ovl_t`` for the overlaps.

A KSpace read from a Wannier90 ``_tb.dat`` file also stores its position
matrix elements: ``pos_i``, ``pos_j``, ``pos_R``, ``pos_r`` (the vectors
:math:`\langle i\mathbf{0}|\mathbf{r}|j\mathbf{R}\rangle`, complex, shape
(n, space_dim)). Versions of tbkit before 0.9 ignore them.

Wannier90 (`wannier.org <https://wannier.org>`_):

* *read_wannier90* imports a tight-binding model from Wannier90's
  ``seedname_hr.dat``, *read_wannier90_tb* from ``seedname_tb.dat``, with
  the matrix elements of the position operator (see there).
* *write_wannier90* writes a **KSpace** as ``seedname_hr.dat``,
  ``seedname_tb.dat``, ``seedname.win`` and ``seedname_centres.xyz``
  (*write_hr*, *write_tb*).

Other codes (ASE, PythTB, TBmodels, Z2Pack, qsymm, pybinding, Kwant): see
*tbkit.interop*.
'''
from __future__ import annotations

import os
import time
from typing import Any, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

import tbkit
import tbkit.error_handling as error_handling
from tbkit.kspace import KSpace
from tbkit.lattice import Lattice
from tbkit.orbital import OrbitalSystem
from tbkit.system import System, HOP_DTYPE


#: Name of the archive format written by *save_model*.
FORMAT = 'tbkit-model'
#: Version of the archive format written by *save_model*; *load_model*
#: reads this version and every earlier one.
VERSION = 2


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

    Parameters
    ----------
    model : **Lattice**, **System** or **KSpace** instance
    path : String or path
        File name; ``.npz`` is appended if missing.

    Returns
    -------
    path : String
        The file written.

    Examples
    --------
    Example usage::

        save_model(gra, 'graphene.npz')
        gra2 = load_model('graphene.npz')
    '''
    error_handling.saveable_model(model, Lattice, System, KSpace, OrbitalSystem)
    error_handling.no_value_functions(model, 'save_model')
    error_handling.file_path(path, 'path')
    path = os.fspath(path)
    if not path.endswith('.npz'):
        path += '.npz'
    lat = model.lat if isinstance(model, (System, KSpace)) else model
    kind = 'System' if isinstance(model, System) else 'KSpace' if isinstance(model, KSpace) else 'Lattice'
    # only what needs version 2 (a periodic System) is written as such, so
    # that older tbkit versions keep reading everything else
    data: dict[str, Any] = {'format': np.array(FORMAT), 'version': np.array(1), 'kind': np.array(kind),
                'tbkit_version': np.array(tbkit.__version__),
                'uc_tag': np.array([dic['tag'] for dic in lat.unit_cell], dtype=str),
                'uc_r0': np.array([dic['r0'] for dic in lat.unit_cell], dtype='f8'),
                'prim_vec': np.array(lat.prim_vec, dtype='f8'),
                'coor': lat.coor,
                'n_cells': np.array([lat.n1, lat.n2, lat.n3], dtype='i8')}
    if isinstance(model, System):
        data['onsite'] = np.asarray(model.onsite, dtype='c16')
        data['hop'] = model.hop
        if any(model.periodic):
            data['periodic'] = np.array(model.periodic, dtype=bool)
            data['version'] = np.array(VERSION)
    elif isinstance(model, KSpace):
        data.update({'spin': np.array(model.spin), 'onsite': model.onsite,
                          'onsite_offdiag': model._onsite_offdiag,
                          'nonreciprocal': np.array(model._nonreciprocal)})
        data.update(_hop_arrays(model._hop, model.space_dim, 'hop'))
        data.update(_hop_arrays(model._overlap_hop, model.space_dim, 'ovl'))
        if model._position_hop:
            elements = model._position_hop
            data['pos_i'] = np.array([e[0] for e in elements], dtype='i8')
            data['pos_j'] = np.array([e[1] for e in elements], dtype='i8')
            data['pos_R'] = np.array([e[2] for e in elements], dtype='f8')
            data['pos_r'] = np.array([e[3] for e in elements], dtype='c16')
    np.savez(path, **data)
    return path


def load_model(path: str | os.PathLike) -> Lattice | System | KSpace:
    '''
    Load a model saved by *save_model*.

    Parameters
    ----------
    path : String or path
        File name.

    Returns
    -------
    model
        **Lattice**, **System** or **KSpace** instance, whose
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
            periodic = tuple(bool(p) for p in data['periodic']) if 'periodic' in data.files else False
            sys = System(lat, periodic=periodic)
            sys.onsite = data['onsite'].copy()
            sys.hop = data['hop'].astype(HOP_DTYPE)  # archives before 0.8 have a 'U2' tag
            return sys
        ks = KSpace(lat, spin=bool(data['spin']))
        ks.onsite = data['onsite'].copy()
        ks._onsite_offdiag = data['onsite_offdiag'].copy()
        ks._nonreciprocal = bool(data['nonreciprocal'])
        ks._hop = _hop_list(data, 'hop')
        ks._overlap_hop = _hop_list(data, 'ovl')
        if 'pos_r' in data.files:
            ks._position_hop = [(int(i), int(j), R.copy(), r.copy()) for i, j, R, r in
                                    zip(data['pos_i'], data['pos_j'], data['pos_R'], data['pos_r'])]
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

    Parameters
    ----------
    path : String or path
        The ``_hr.dat`` file.

    Returns
    -------
    num_wann : Integer
        Number of Wannier functions.
    R : Integer ndarray, shape (nrpts, 3)
        Lattice vectors (in
        units of the primitive vectors).
    ndegen : Integer ndarray, shape (nrpts,)
        Degeneracy weights.
    ham : Complex ndarray, shape (nrpts, num_wann, num_wann)
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

    Parameters
    ----------
    path : String or path
        The ``.win`` file.

    Returns
    -------
    cell : Real ndarray, shape (3, 3)
        The rows are
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

    Parameters
    ----------
    path : String or path
        The ``_centres.xyz`` file.
    num_wann : Positive integer
        Number of Wannier functions.

    Returns
    -------
    centres : Real ndarray, shape (num_wann, 3)
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

    Parameters
    ----------
    hr : String or path
        The ``seedname_hr.dat`` file.
    prim_vec : List of *dim* tuples of *dim* real numbers
        Default
        value None. Primitive vectors, in the units of the positions
        (Angstrom for Wannier90). If None, read from *win*.
    win : String or path, default None
        The ``seedname.win``
        file, for its ``unit_cell_cart`` block (if *prim_vec* is None).
    centres : String or path, default None
        The
        ``seedname_centres.xyz`` file: the Wannier centres become the
        orbital positions (used by the Wilson-loop tools, the Peierls
        phases, *bridges.finite_system*, ...).
    positions : Array, shape (num_wann, dim) or (num_wann, 3), default None
        Orbital positions, instead of *centres*. With
        neither, every orbital sits at the origin of the cell.
    tags : List of num_wann strings (1 to 8 characters)
        Default value
        None (every orbital tagged ``'a'``). Sublattice tags.
    dim : 2 or 3, default 3
        A 2D model (a layer) keeps only
        :math:`(R_1, R_2)`: every element with :math:`R_3\neq 0` must vanish
        (within *tol*), and the lattice lies in the :math:`(x, y)` plane
        (the z components of the vectors read from *win* must vanish; those
        of the positions are dropped).
    cutoff : Positive real number or zero, default 0
        Drop the
        hoppings with :math:`|t| \le` *cutoff* (the exact zeros by default).
    tol : Positive real number, default 1e-6
        Tolerance of the
        Hermiticity and 2D checks (the file has 6 decimals).

    Returns
    -------
    ks : **KSpace** instance, with num_wann orbitals

    Examples
    --------
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
        cell = read_win_cell(cast('str | os.PathLike', win))
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
    return _hr_kspace(prim_vec, positions[:, :dim], tags, R, ham / ndegen[:, None, None], cutoff, tol)


def _hr_kspace(
    prim_vec: list[tuple[float, ...]], positions: NDArray[np.float64], tags: list[str],
    R: NDArray[np.int_], ham: NDArray[np.complex128], cutoff: float, tol: float,
    spin: bool = False,
) -> KSpace:
    r'''
    Private function. The **KSpace** of the matrices :math:`H(\mathbf{R})`
    (*ham*, shape (nR, num_wann, num_wann), degeneracy weights already
    divided out) at the lattice vectors *R* (integers, shape (nR, dim)),
    with orbitals at *positions*, for *read_wannier90*, *read_wannier90_tb*,
    *WannierFunctions.kspace* and the importers of *tbkit.interop*. It is
    Hermitian (with real onsite energies) when every :math:`H(\mathbf{R})`
    has its partner :math:`H(-\mathbf{R}) = H(\mathbf{R})^\dagger` within
    *tol*. With *spin*, *positions* and *tags* are those of the sites, and
    the orbitals their spin components (site-major, see *KSpace*).
    '''
    num_wann = ham.shape[1]
    unit_cell = [{'tag': tag, 'r0': tuple(float(c) for c in r)} for tag, r in zip(tags, positions)]
    lat = Lattice(unit_cell=unit_cell, prim_vec=list(prim_vec))
    ks = KSpace(lat, spin=spin)
    # Hermitian: every H(R) has its partner H(-R) = H(R)^dagger
    index = {tuple(r): p for p, r in enumerate(R.tolist())}
    partner = [index.get(tuple(-c for c in r)) for r in R.tolist()]
    hermitian = None not in partner and bool(
        np.abs(ham - ham[np.array(partner)].conj().transpose(0, 2, 1)).max() <= tol)
    a = np.array(lat.prim_vec, dtype='f8')
    hops = []
    for r, h in zip(R.tolist(), ham):
        nonzero = [c for c in r if c != 0]
        R_cart = np.array(r, dtype='f8') @ a
        for m in range(num_wann):
            for n in range(num_wann):
                t = complex(h[m, n])
                if m == n and not nonzero:
                    ks.onsite[m] = t.real if hermitian else t
                elif abs(t) > cutoff and (not hermitian or (nonzero and nonzero[0] > 0)
                                                    or (not nonzero and m < n)):
                    # as set_hopping stores them: the element, then its conjugate
                    hops.append((m, n, R_cart, t))
                    if hermitian:
                        hops.append((n, m, -R_cart, t.conjugate()))
    ks._hop = hops
    ks._nonreciprocal = not hermitian
    return ks


def _integer_vectors(ks: KSpace, R_cart: NDArray[np.float64]) -> NDArray[np.int64]:
    r'''
    Private function. Cartesian lattice vectors (shape (n, space_dim)) as
    integers :math:`(n_1, ..., n_{dim})`, as in *KSpace._hop_cells*.
    '''
    rec = np.array(ks.rec_vec, dtype='f8')
    return np.rint(np.reshape(R_cart, (-1, ks.space_dim)) @ rec.T / (2 * np.pi)).astype('i8')


def _wannier_cell(ks: KSpace) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    r'''
    Private function. The Wannier90 cell (rows :math:`\mathbf{a}_1,
    \mathbf{a}_2, \mathbf{a}_3`, shape (3, 3)) and the orbital positions
    (shape (norb, 3)) of a 2D (in the plane) or 3D **KSpace**. A 2D model
    gets :math:`\mathbf{a}_3 = (0, 0, c)`, *c* its longest primitive vector.
    '''
    a = np.array(ks.lat.prim_vec, dtype='f8')
    tau = ks.orbital_positions()
    if ks.dim == 3:
        return a, tau
    c = float(np.linalg.norm(a, axis=1).max())
    cell = np.array([[*a[0], 0.], [*a[1], 0.], [0., 0., c]])
    return cell, np.concatenate([tau, np.zeros((len(tau), 1))], axis=1)


def _wannier_blocks(
    ks: KSpace,
) -> tuple[NDArray[np.int64], NDArray[np.complex128], NDArray[np.complex128]]:
    r'''
    Private function. The matrices :math:`H_{mn}(\mathbf{R}) =
    \langle m\mathbf{0}|H|n\mathbf{R}\rangle` (onsite terms at
    :math:`\mathbf{R} = 0`) and :math:`\langle m\mathbf{0}|\mathbf{r}|n\mathbf{R}\rangle`
    (the orbital positions on the diagonal at :math:`\mathbf{R} = 0`, plus
    *_position_hop*) of a **KSpace**, at the lattice vectors *R* (shape
    (nR, 3), each with its opposite, sorted; zero for a 2D model's third
    component): shapes (nR, norb, norb) and (nR, norb, norb, 3).
    '''
    n = ks.norb
    _, tau = _wannier_cell(ks)
    hops, elements = ks._hop, ks._position_hop
    cells = np.zeros((len(hops) + len(elements), 3), dtype='i8')
    cells[:len(hops), :ks.dim] = _integer_vectors(ks, np.array([h[2] for h in hops]))
    cells[len(hops):, :ks.dim] = _integer_vectors(ks, np.array([e[2] for e in elements]))
    R = np.unique(np.concatenate([cells, -cells, np.zeros((1, 3), dtype='i8')]), axis=0)
    index = {tuple(r): p for p, r in enumerate(R.tolist())}
    ham = np.zeros((len(R), n, n), dtype='c16')
    pos = np.zeros((len(R), n, n, 3), dtype='c16')
    origin = index[(0, 0, 0)]
    ham[origin] += np.diag(ks.onsite) + ks._onsite_offdiag
    pos[origin, np.arange(n), np.arange(n)] = tau
    for (i, j, _, t), cell in zip(hops, cells[:len(hops)]):
        ham[index[tuple(cell.tolist())], i, j] += t
    for (i, j, _, r), cell in zip(elements, cells[len(hops):]):
        pos[index[tuple(cell.tolist())], i, j, :ks.space_dim] += r
    return R, ham, pos


def _header() -> str:
    '''
    Private function. The first line of the files written by tbkit.
    '''
    return ' written by tbkit {} on {}'.format(tbkit.__version__, time.strftime('%d%b%Y at %H:%M:%S'))


def _weights_lines(nrpts: int) -> list[str]:
    '''
    Private function. The degeneracy weights (all 1), 15 per line.
    '''
    return [''.join('{:5d}'.format(1) for _ in range(k, min(k + 15, nrpts))) for k in range(0, nrpts, 15)]


def _check_wannier_export(ks: KSpace) -> None:
    '''
    Private function. Validate a **KSpace** to be written in the Wannier90 formats.
    '''
    error_handling.kspace(ks, KSpace)
    ks._check_static()
    error_handling.no_overlap(bool(ks._overlap_hop))
    error_handling.wannier_export_dim(ks.dim, ks.space_dim)


def write_hr(ks: KSpace, path: str | os.PathLike) -> str:
    r'''
    Write a **KSpace** as a Wannier90 ``seedname_hr.dat`` file (see
    *read_hr*): every matrix :math:`H_{mn}(\mathbf{R}) =
    \langle m\mathbf{0}|H|n\mathbf{R}\rangle`, the onsite terms at
    :math:`\mathbf{R} = 0`, with degeneracy weights 1, at the lattice
    vectors of the hoppings and their opposites. The values are written
    with 12 decimals, so *read_wannier90* gives the model back to
    :math:`10^{-12}`. The value functions are written with the parameters
    of *set_params*.

    Parameters
    ----------
    ks : **KSpace** instance
        2D (in the plane) or 3D, without overlap (*set_overlap*). A spinful
        model is written orbital by orbital (site-major).
    path : String or path
        The file to write.

    Returns
    -------
    path : String
        The file written.
    '''
    _check_wannier_export(ks)
    error_handling.file_path(path, 'path')
    R, ham, _ = _wannier_blocks(ks)
    lines = [_header(), '{:12d}'.format(ks.norb), '{:12d}'.format(len(R))] + _weights_lines(len(R))
    for r, h in zip(R.tolist(), ham):
        for n in range(ks.norb):
            for m in range(ks.norb):
                lines.append('{:5d}{:5d}{:5d}{:5d}{:5d}{:20.12f}{:20.12f}'.format(
                    *r, m + 1, n + 1, h[m, n].real, h[m, n].imag))
    path = os.fspath(path)
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    return path


def write_tb(ks: KSpace, path: str | os.PathLike) -> str:
    r'''
    Write a **KSpace** as a Wannier90 ``seedname_tb.dat`` file (see
    *read_tb*): the cell, the matrices :math:`H_{mn}(\mathbf{R})` of
    *write_hr* and the position matrix elements
    :math:`\langle m\mathbf{0}|\mathbf{r}|n\mathbf{R}\rangle`: the orbital
    positions on the diagonal at :math:`\mathbf{R} = 0`, and the other
    elements of a model read by *read_wannier90_tb*. A 2D model gets a
    third cell vector :math:`(0, 0, c)`, *c* its longest primitive vector.

    Parameters
    ----------
    ks : **KSpace** instance
        See *write_hr*.
    path : String or path
        The file to write.

    Returns
    -------
    path : String
        The file written.
    '''
    _check_wannier_export(ks)
    error_handling.file_path(path, 'path')
    cell, _ = _wannier_cell(ks)
    R, ham, pos = _wannier_blocks(ks)
    lines = [_header()] + [''.join('{:20.12f}'.format(c) for c in a) for a in cell]
    lines += ['{:12d}'.format(ks.norb), '{:12d}'.format(len(R))] + _weights_lines(len(R))
    for r, h in zip(R.tolist(), ham):
        lines += ['', '{:5d}{:5d}{:5d}'.format(*r)]
        lines += ['{:5d}{:5d}{:20.12f}{:20.12f}'.format(m + 1, n + 1, h[m, n].real, h[m, n].imag)
                     for n in range(ks.norb) for m in range(ks.norb)]
    for r, p in zip(R.tolist(), pos):
        lines += ['', '{:5d}{:5d}{:5d}'.format(*r)]
        lines += ['{:5d}{:5d}'.format(m + 1, n + 1)
                     + ''.join('{:20.12f}{:20.12f}'.format(c.real, c.imag) for c in p[m, n])
                     for n in range(ks.norb) for m in range(ks.norb)]
    path = os.fspath(path)
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    return path


def write_wannier90(ks: KSpace, seedname: str | os.PathLike) -> list[str]:
    r'''
    Write a **KSpace** in the files of Wannier90 that *read_wannier90* and
    *read_wannier90_tb* read, and that other codes read too (PythTB's
    ``W90``, TBmodels, WannierTools, WannierBerri, Z2Pack through
    TBmodels):

    * ``seedname_hr.dat``, the Hamiltonian (*write_hr*);
    * ``seedname_tb.dat``, the cell, the Hamiltonian and the position
      matrix elements (*write_tb*);
    * ``seedname.win``, with ``num_wann`` and the ``unit_cell_cart`` block;
    * ``seedname_centres.xyz``, the orbital positions as Wannier centres.

    Wannier90's units are Angstrom and eV: the numbers are written as they
    are, in the units of the model.

    Parameters
    ----------
    ks : **KSpace** instance
        See *write_hr*.
    seedname : String or path
        The common prefix of the files.

    Returns
    -------
    paths : List of 4 strings
        The files written, in the order above.

    Examples
    --------
    Example usage::

        write_wannier90(haldane, 'haldane')
        ks = read_wannier90('haldane_hr.dat', win='haldane.win',
                                     centres='haldane_centres.xyz', dim=2)
    '''
    _check_wannier_export(ks)
    error_handling.file_path(seedname, 'seedname')
    seedname = os.fspath(seedname)
    cell, tau = _wannier_cell(ks)
    hr = write_hr(ks, seedname + '_hr.dat')
    tb = write_tb(ks, seedname + '_tb.dat')
    win = seedname + '.win'
    with open(win, 'w') as f:
        f.write('!' + _header() + '\nnum_wann = {}\n\nbegin unit_cell_cart\nang\n'.format(ks.norb)
                  + '\n'.join(''.join('{:20.12f}'.format(c) for c in a) for a in cell)
                  + '\nend unit_cell_cart\n')
    xyz = seedname + '_centres.xyz'
    with open(xyz, 'w') as f:
        f.write('{:6d}\n'.format(ks.norb) + _header() + '\n'
                  + ''.join('X' + ''.join('{:20.12f}'.format(c) for c in r) + '\n' for r in tau))
    return [hr, tb, win, xyz]


def read_tb(
    path: str | os.PathLike,
) -> tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.int64], NDArray[np.complex128],
           NDArray[np.complex128]]:
    r'''
    Read a Wannier90 ``seedname_tb.dat`` file: a header line, the three
    lattice vectors (Angstrom, one per line), the number of Wannier
    functions, the number of lattice vectors :math:`\mathbf{R}` and their
    degeneracy weights (15 per line), then, for each :math:`\mathbf{R}`, an
    empty line, ``R1 R2 R3`` and one line ``m n Re Im`` per element of
    :math:`H_{mn}(\mathbf{R})` (as in ``_hr.dat``), and again, for each
    :math:`\mathbf{R}`, an empty line, ``R1 R2 R3`` and one line
    ``m n Re(x) Im(x) Re(y) Im(y) Re(z) Im(z)`` per position matrix element
    :math:`\mathbf{r}_{mn}(\mathbf{R}) = \langle m\mathbf{0}|\mathbf{r}|n\mathbf{R}\rangle`
    (m, n counted from 1; Wannier90's ``hamiltonian_write_tb``).

    Parameters
    ----------
    path : String or path
        The ``_tb.dat`` file.

    Returns
    -------
    cell : Real ndarray, shape (3, 3)
        The rows are :math:`\mathbf{a}_1, \mathbf{a}_2, \mathbf{a}_3`.
    R : Integer ndarray, shape (nrpts, 3)
        Lattice vectors (in units of the primitive vectors).
    ndegen : Integer ndarray, shape (nrpts,)
        Degeneracy weights.
    ham : Complex ndarray, shape (nrpts, num_wann, num_wann)
        :math:`H_{mn}(\mathbf{R})` as written in the file (not divided by
        the weights).
    pos : Complex ndarray, shape (nrpts, num_wann, num_wann, 3)
        :math:`\mathbf{r}_{mn}(\mathbf{R})` as written (not divided by the weights).
    '''
    error_handling.file_path(path, 'path')
    with open(path) as f:
        lines = f.read().splitlines()[1:]  # the first line is a free comment
    cell_lines = [line.split()[:3] for line in lines[:3]]
    error_handling.win_cell(cell_lines, 'The first three lines after the header of a _tb.dat file')
    tokens = ' '.join(lines[3:]).split()
    error_handling.hr_header(tokens, '_tb.dat')
    num_wann, nrpts = int(tokens[0]), int(tokens[1])
    ndegen = np.array(tokens[2:2 + nrpts], dtype='i8')
    body = tokens[2 + nrpts:]
    size_h, size_r = 3 + 4 * num_wann ** 2, 3 + 8 * num_wann ** 2
    error_handling.tb_body(len(body), nrpts, num_wann)
    rows_h = np.array(body[:nrpts * size_h], dtype='f8').reshape(nrpts, size_h)
    rows_r = np.array(body[nrpts * size_h:], dtype='f8').reshape(nrpts, size_r)
    R = rows_h[:, :3].astype('i8')
    error_handling.tb_vectors(np.array_equal(rows_r[:, :3].astype('i8'), R))
    elem_h = rows_h[:, 3:].reshape(nrpts, num_wann ** 2, 4)
    elem_r = rows_r[:, 3:].reshape(nrpts, num_wann ** 2, 8)
    block = np.repeat(np.arange(nrpts), num_wann ** 2)
    ham = np.zeros((nrpts, num_wann, num_wann), 'c16')
    m, n = elem_h[..., 0].astype('i8').ravel() - 1, elem_h[..., 1].astype('i8').ravel() - 1
    ham[block, m, n] = (elem_h[..., 2] + 1j * elem_h[..., 3]).ravel()
    pos = np.zeros((nrpts, num_wann, num_wann, 3), 'c16')
    m, n = elem_r[..., 0].astype('i8').ravel() - 1, elem_r[..., 1].astype('i8').ravel() - 1
    pos[block, m, n] = (elem_r[..., 2::2] + 1j * elem_r[..., 3::2]).reshape(-1, 3)
    cell = np.array(cell_lines, dtype='f8')
    return cell, R, ndegen, ham, pos


def read_wannier90_tb(
    tb: str | os.PathLike,
    tags: list[str] | None = None,
    dim: int = 3,
    cutoff: float = 0.,
    tol: float = 1e-6,
    elements: bool = True,
) -> KSpace:
    r'''
    Import a tight-binding model from Wannier90's ``seedname_tb.dat``
    (see *read_tb*), which holds, besides the Hamiltonian of
    ``seedname_hr.dat``, the cell and the matrix elements of the position
    operator between the Wannier functions,
    :math:`\mathbf{r}_{mn}(\mathbf{R}) = \langle m\mathbf{0}|\mathbf{r}|n\mathbf{R}\rangle`.
    The Hamiltonian becomes a **KSpace** as in *read_wannier90*, and
    the Wannier centres :math:`\mathbf{r}_{mm}(\mathbf{0})` its orbital
    positions.

    With *elements*, the model also keeps the other position matrix
    elements, and the responses that take ``positions=True`` use the whole
    position operator instead of the centres alone: the velocity is
    :math:`i[H, \mathbf{r}]` (in *hall_conductivity* and the other
    Berry-curvature responses, *tbkit.optics.optical_conductivity*), and the
    band curvature includes that of the Wannier functions' own Berry
    connection (Wang, Yates, Souza and Vanderbilt, Phys. Rev. B 74, 195118
    (2006)). *KSpace.spin_hall_conductivity*, *orbital_magnetization*,
    *axion_angle* and the Berry curvature dipole and shift current of
    *tbkit.optics* refuse them; the Wilson loops and the quantum geometric
    tensor use the centres only. *tbkit.io.save_model* keeps them.

    Parameters
    ----------
    tb : String or path
        The ``seedname_tb.dat`` file.
    tags : List of num_wann strings (1 to 8 characters), default None
        See *read_wannier90*.
    dim : 2 or 3, default 3
        See *read_wannier90*. For a 2D model, the position elements with
        :math:`R_3 \neq 0` must vanish too, and their z components are dropped.
    cutoff : Positive real number or zero, default 0
        Drop the hoppings
        and the position elements (by the norm of the vector) of size at
        most *cutoff*.
    tol : Positive real number, default 1e-6
        See *read_wannier90*.
    elements : Boolean, default True
        Keep the position matrix elements other than the centres.

    Returns
    -------
    ks : **KSpace** instance, with num_wann orbitals

    Examples
    --------
    Example usage::

        ks = read_wannier90_tb('wannier90_tb.dat', dim=2)
        sigma = ks.hall_conductivity(e_fermi, nk=200)   # with i[H, r]
    '''
    error_handling.file_path(tb, 'tb')
    error_handling.wannier_dim(dim)
    error_handling.positive_real_zero(cutoff, 'cutoff')
    error_handling.positive_real(tol, 'tol')
    error_handling.boolean(elements, 'elements')
    cell, R, ndegen, ham, pos = read_tb(tb)
    num_wann = ham.shape[1]
    error_handling.layer_cell(cell, dim)
    prim_vec = [tuple(float(c) for c in a[:dim]) for a in cell[:dim]]
    error_handling.prim_vec(prim_vec)
    error_handling.wannier_prim_vec(prim_vec, dim)
    tags = ['a'] * num_wann if tags is None else tags
    error_handling.wannier_tags(tags, num_wann)
    ham = ham / ndegen[:, None, None]
    pos = pos / ndegen[:, None, None, None]
    origin = np.flatnonzero((R == 0).all(axis=1))
    error_handling.tb_origin(len(origin))
    diag = np.arange(num_wann)
    centres = pos[origin[0], diag, diag].real
    if dim == 2:
        out = R[:, 2] != 0
        error_handling.layer_hoppings(float(max(np.abs(ham[out]).max(initial=0.),
                                                np.abs(pos[out]).max(initial=0.))), tol)
        R, ham, pos = R[~out, :2], ham[~out], pos[~out, :, :, :2]
        origin = np.flatnonzero((R == 0).all(axis=1))
    ks = _hr_kspace(prim_vec, centres[:, :dim], tags, R, ham, cutoff, tol)
    if elements:
        pos[origin[0], diag, diag] = 0.
        a = np.array(prim_vec, dtype='f8')
        size = np.linalg.norm(pos, axis=-1)
        ks._position_hop = [(int(m), int(n), R[p].astype('f8') @ a, pos[p, m, n].copy())
                                for p, m, n in zip(*np.nonzero(size > cutoff))]
    return ks
