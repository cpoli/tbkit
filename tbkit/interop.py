r'''
Exchange models with other codes: ASE, PythTB, TBmodels, Z2Pack, qsymm,
pybinding and Kwant (for Wannier90's files, see *tbkit.io*).

None of these packages is a dependency of tbkit: each function imports
the one it needs when it is called, and ``import tbkit`` never does. The
importers only read attributes, so they also accept objects that look
like the package's.

==============================  ==========================================
Function                        Does
==============================  ==========================================
:func:`lattice_from_ase`        ASE ``Atoms`` -> **Lattice**
:func:`lattice_to_ase`          **Lattice** -> ASE ``Atoms`` (unit cell or sites)
:func:`from_pythtb`             PythTB ``TBModel`` (``tb_model`` in 1.x) -> **KSpace**
:func:`to_pythtb`               **KSpace** -> PythTB ``TBModel`` (``tb_model`` in 1.x)
:func:`to_tbmodels`             **KSpace** -> TBmodels ``Model`` (and its HDF5 file)
:func:`z2pack_hamiltonian`      **KSpace** -> the callable :math:`H(\mathbf{k})` of Z2Pack
:func:`to_z2pack`               **KSpace** -> ``z2pack.hm.System``
:func:`to_qsymm`                **KSpace** -> qsymm ``BlochModel``
:func:`from_pybinding`          pybinding ``Lattice`` -> **KSpace**; ``Model`` -> matrix, positions
:func:`to_pybinding`            **KSpace** -> pybinding ``Lattice``
:func:`to_kwant`                **System** or **Transport** (with its leads) -> Kwant ``Builder``
:func:`from_kwant`              finalized Kwant system -> matrix, positions
==============================  ==========================================

**Conventions.** All these codes, and tbkit, store a hopping as the matrix
element :math:`t_{ij}(\mathbf{R}) = \langle i,\mathbf{0}|H|j,\mathbf{R}\rangle`,
so the hopping lists translate one to one. They differ in the Bloch
phases: tbkit's *get_ham* uses the periodic gauge
:math:`e^{i\mathbf{k}\cdot\mathbf{R}}` (Wannier90's and TBmodels'
"convention 2", Z2Pack's default), PythTB (1.x and 2.0) and qsymm include
the orbital positions, :math:`e^{i\mathbf{k}\cdot(\mathbf{R}+\boldsymbol\tau_j-\boldsymbol\tau_i)}`
("convention 1"), and pybinding uses :math:`e^{-i\mathbf{k}\cdot\mathbf{R}}`.
The matrices :math:`H(\mathbf{k})` then differ by a diagonal unitary (or by
:math:`\mathbf{k}\to-\mathbf{k}`), and the bands, Chern numbers and
:math:`\mathbb{Z}_2` invariants are the same. PythTB counts k in units of
the reciprocal vectors, tbkit in Cartesian coordinates.

**What is not exchanged.** Value functions are written with the
parameters of *KSpace.set_params*; overlaps (*set_overlap*), driven models
and non-Hermitian models are refused by the exporters, which all need a
Hermitian, orthogonal tight-binding model.
'''
from __future__ import annotations

import importlib
from typing import Any, Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray
import scipy.linalg as sla
import scipy.sparse as sp

import tbkit.error_handling as error_handling
from tbkit.io import _hr_kspace
from tbkit.kspace import KSpace
from tbkit.lattice import Lattice
from tbkit.system import System
from tbkit.transport import Transport


def _import(name: str) -> Any:
    '''
    Private function. Import an optional package on first use.
    '''
    try:
        return importlib.import_module(name)
    except ImportError:
        raise ImportError('\n\nThis function needs the optional package {0} '
                                  '(pip install {0}).\n'.format(name)) from None


def _default_tags(n: int) -> list[str]:
    '''
    Private function. One tag per site: 'a', 'b', ..., 'z', then 's26', 's27', ...
    '''
    return [chr(ord('a') + s) if s < 26 else 's{}'.format(s) for s in range(n)]


def _check_export(ks: KSpace) -> None:
    '''
    Private function. Validate a **KSpace** to be exported: static,
    orthogonal and Hermitian.
    '''
    error_handling.kspace(ks, KSpace)
    ks._check_static()
    error_handling.no_overlap(bool(ks._overlap_hop))
    error_handling.hermitian_model(ks.is_hermitian())


def _site_positions(ks: KSpace) -> NDArray[np.float64]:
    '''
    Private function. The positions of the sites of the unit cell, shape (n_sites, space_dim).
    '''
    return np.array([dic['r0'] for dic in ks.lat.unit_cell], dtype='f8').reshape(ks.n_sites, -1)


def _forward(u: int, v: int, R: tuple[int, ...]) -> bool:
    '''
    Private function. Whether the hopping (u, v, R) is the representative
    of the pair it forms with its conjugate (v, u, -R): the first nonzero
    component of R positive, or R = 0 and u < v.
    '''
    nonzero = [c for c in R if c != 0]
    return nonzero[0] > 0 if nonzero else u < v


def _site_blocks(ks: KSpace, nb: int) -> tuple[NDArray[np.complex128], dict]:
    r'''
    Private function. The onsite blocks (shape (units, nb, nb)) and one
    hopping block :math:`\langle u,\mathbf{0}|H|v,\mathbf{R}\rangle` (nb x nb)
    per conjugate pair, keyed by (u, v, R), of a Hermitian **KSpace**, with
    units of *nb* orbitals: the sites (*nb* = 2 for a spinful model) or the
    orbitals (*nb* = 1).
    '''
    units = ks.norb // nb
    onsite = np.zeros((units, nb, nb), dtype='c16')
    blocks: dict = {}
    zero = (0,) * ks.dim
    full = np.diag(ks.onsite) + ks._onsite_offdiag
    terms = [(int(i), int(j), zero, full[i, j]) for i, j in zip(*np.nonzero(full))]
    for i, j, R, t in terms + ks._hop_cells():
        u, v = i // nb, j // nb
        if u == v and not any(R):
            onsite[u, i % nb, j % nb] += t
        else:
            blocks.setdefault((u, v, R), np.zeros((nb, nb), dtype='c16'))[i % nb, j % nb] += t
    return onsite, {key: b for key, b in blocks.items() if _forward(*key)}


def _full_basis(ks: KSpace) -> NDArray[np.float64]:
    '''
    Private function. The primitive vectors, completed by an orthonormal
    basis of their orthogonal complement (shape (space_dim, space_dim), rows).
    '''
    a = np.array(ks.lat.prim_vec, dtype='f8')
    if ks.dim == ks.space_dim:
        return a
    return np.concatenate([a, sla.null_space(a).T])


def _reduced_positions(ks: KSpace) -> NDArray[np.float64]:
    r'''
    Private function. The orbital positions in units of the primitive
    vectors, in the plane (or along the line) of the lattice: shape (norb, dim).
    '''
    a = np.array(ks.lat.prim_vec, dtype='f8') @ ks.k_basis
    return ks.orbital_positions() @ ks.k_basis @ np.linalg.inv(a)


#################################
# ASE
#################################


def lattice_from_ase(atoms: Any, tags: list[str] | None = None, space_dim: int | None = None) -> Lattice:
    r'''
    Get the **Lattice** of an ASE ``Atoms`` object (`ase-lib.org
    <https://ase-lib.org>`_): its periodic cell vectors (those with
    ``pbc`` True) are the primitive vectors, its atoms the sites of the
    unit cell, in the same order and units (Angstrom).

    Parameters
    ----------
    atoms : ``ase.Atoms``
        At least one periodic direction.
    tags : List of strings (1 to 8 characters), default None
        One per atom;
        the chemical symbols by default.
    space_dim : 2, 3 or None, default None
        2 keeps the x and y coordinates only, for a
        lattice in the plane: the periodic vectors must lie in the plane and
        the atoms at the same height (dropped). None takes 2 when this is the
        case (at most two periodic directions), 3 otherwise.

    Returns
    -------
    lat : **Lattice** instance

    Examples
    --------
    Example usage::

        from ase.build import graphene
        lat = lattice_from_ase(graphene(a=2.46, vacuum=None))  # 2 sites 'C'
    '''
    cell = np.asarray(atoms.get_cell(), dtype='f8').reshape(3, 3)
    pbc = np.asarray(atoms.get_pbc(), dtype=bool).reshape(3)
    pos = np.asarray(atoms.get_positions(), dtype='f8').reshape(-1, 3)
    error_handling.ase_periodic(pbc)
    tags = list(atoms.get_chemical_symbols()) if tags is None else tags
    error_handling.site_tags(tags, len(pos))
    a = cell[pbc]
    scale = max(1., float(np.abs(pos).max(initial=0.)), float(np.abs(a).max()))
    planar = len(a) <= 2 and bool(np.all(np.abs(a[:, 2]) <= 1e-9 * scale)) \
        and float(np.ptp(pos[:, 2])) <= 1e-9 * scale if len(pos) else False
    if space_dim is None:
        space_dim = 2 if planar else 3
    error_handling.ase_space_dim(space_dim, planar)
    prim_vec = [tuple(float(c) for c in v[:space_dim]) for v in a]
    unit_cell = [{'tag': tag, 'r0': tuple(float(c) for c in r[:space_dim])} for tag, r in zip(tags, pos)]
    return Lattice(unit_cell=unit_cell, prim_vec=prim_vec)


def lattice_to_ase(lat: Lattice, symbols: dict[str, str] | None = None, sites: bool = False) -> Any:
    r'''
    Get an ASE ``Atoms`` object of a **Lattice**: its unit cell, periodic
    along the primitive vectors (the other cell vectors zero and not
    periodic), or, with *sites*, the finite sample of *get_lattice* as a
    cluster (no periodic direction). A 2D lattice is placed at z = 0.

    Parameters
    ----------
    lat : **Lattice** instance
    symbols : Dictionary, default None
        Chemical symbol of each tag
        (``{'a': 'C', 'b': 'N'}``). By default, the tags themselves, which
        must then be chemical symbols.
    sites : Boolean, default False
        The sites of *lat.coor* instead of the unit cell.

    Returns
    -------
    atoms : ``ase.Atoms``
    '''
    error_handling.lat(lat)
    error_handling.boolean(sites, 'sites')
    ase = _import('ase')
    if sites:
        error_handling.empty_coor(lat.coor)
        tags = [str(t) for t in lat.coor['tag']]
        pos = np.stack([lat.coor[f] for f in ('x', 'y', 'z')[:lat.space_dim]], axis=1)
    else:
        tags = [dic['tag'] for dic in lat.unit_cell]
        pos = np.array([dic['r0'] for dic in lat.unit_cell], dtype='f8').reshape(len(tags), -1)
    names = [symbols.get(t, t) for t in tags] if symbols is not None else tags
    error_handling.chemical_symbols(names, ase.data.chemical_symbols)
    pos = np.concatenate([pos, np.zeros((len(pos), 3 - pos.shape[1]))], axis=1)
    cell = np.zeros((3, 3))
    pbc = [False] * 3
    if not sites:
        for c, v in enumerate(lat.prim_vec):
            cell[c, :len(v)] = v
            pbc[c] = True
    return ase.Atoms(symbols=names, positions=pos, cell=cell, pbc=pbc)


#################################
# PYTHTB
#################################


def from_pythtb(model: Any, tags: list[str] | None = None) -> KSpace:
    r'''
    Get the **KSpace** of a PythTB model (`pythtb.readthedocs.io
    <https://pythtb.readthedocs.io>`_): a ``TBModel`` of PythTB 2, or a
    ``tb_model`` of PythTB 1.x. The periodic lattice vectors become the
    primitive vectors, the orbitals the sites (spinful models give a
    spinful **KSpace**), and every hopping, with its conjugate, a hopping
    of tbkit: both codes store :math:`\langle i,\mathbf{0}|H|j,\mathbf{R}\rangle`.
    The bands and the topological invariants are those of PythTB; its
    :math:`H(\mathbf{k})` includes the orbital positions (convention 1) and
    tbkit's does not (see the module docstring). A model of 1D space is
    placed along x in the plane.

    Parameters
    ----------
    model : ``pythtb.TBModel`` or ``pythtb.tb_model``
        With at least one
        periodic direction and numbers only (no parametric terms).
    tags : List of strings (1 to 8 characters), default None
        One per orbital
        (per site for a spinful model); 'a', 'b', ... by default.

    Returns
    -------
    ks : **KSpace** instance
    '''
    if hasattr(model, 'hoppings'):  # PythTB 2
        lat_vecs = np.asarray(model.lat_vecs, dtype='f8')
        per = [int(p) for p in model.lattice.periodic_dirs]
        positions = np.asarray(model.get_orb_vecs(cartesian=True), dtype='f8')
        spin = bool(model.spinful)
        onsite = list(model.onsite)
        hops = [(h['amplitude'], h['from_orbital'], h['to_orbital'],  # no vector within the cell
                    h.get('lattice_vector', np.zeros(len(lat_vecs)))) for h in model.hoppings]
    else:  # PythTB 1.x
        lat_vecs = np.asarray(model._lat, dtype='f8')
        per = [int(p) for p in model._per]
        positions = np.asarray(model._orb, dtype='f8') @ lat_vecs
        spin = model._nspin == 2
        onsite = list(model._site_energies)
        hops = [(h[0], h[1], h[2], h[3]) for h in model._hoppings]
    error_handling.pythtb_periodic(len(per))
    error_handling.numeric_terms([e for e in onsite] + [h[0] for h in hops], 'PythTB')
    n_sites = len(positions)
    tags = _default_tags(n_sites) if tags is None else tags
    error_handling.site_tags(tags, n_sites)
    pad = max(0, 2 - lat_vecs.shape[1])  # a model of 1D space lies along x
    lat_vecs = np.pad(lat_vecs, ((0, 0), (0, pad)))
    positions = np.pad(positions.reshape(n_sites, -1), ((0, 0), (0, pad)))
    nb = 2 if spin else 1
    blocks: dict = {}

    def add(R, i, j, value):
        block = np.asarray(value, dtype='c16')
        block = block * np.eye(nb) if block.ndim == 0 else block.reshape(nb, nb)
        mat = blocks.setdefault(R, np.zeros((nb * n_sites, nb * n_sites), dtype='c16'))
        mat[nb * i:nb * (i + 1), nb * j:nb * (j + 1)] += block
        return block

    for s, e in enumerate(onsite):
        add((0,) * len(per), s, s, e)
    for amp, i, j, R in hops:
        R = tuple(int(round(float(np.ravel(R)[p]))) for p in per)
        block = add(R, i, j, amp)
        add(tuple(-c for c in R), j, i, block.conj().T)
    R_arr = np.array(sorted(blocks), dtype='i8').reshape(len(blocks), len(per))
    ham = np.array([blocks[tuple(r)] for r in R_arr.tolist()])
    prim_vec = [tuple(float(c) for c in lat_vecs[p]) for p in per]
    return _hr_kspace(prim_vec, positions, tags, R_arr, ham, 0., 1e-10, spin)


def to_pythtb(ks: KSpace) -> Any:
    r'''
    Get the PythTB model of a **KSpace**: a ``TBModel`` with PythTB 2, a
    ``tb_model`` with PythTB 1.x. The lattice vectors are the primitive
    vectors, completed by orthonormal non-periodic ones for a lattice of
    lower dimension than its space (a chain in the plane, a slab); the
    sites become the orbitals (spinful with ``spin=True``), and one
    hopping of each conjugate pair is set (PythTB adds the other). See
    *from_pythtb* for the conventions.

    Parameters
    ----------
    ks : **KSpace** instance
        Hermitian, without overlap.

    Returns
    -------
    model : ``pythtb.TBModel`` (or ``pythtb.tb_model``)

    Examples
    --------
    Example usage::

        model = to_pythtb(models.haldane())
        model.chern_number((0, 1), (60, 60), occ_idxs=[0])   # 1.0
    '''
    _check_export(ks)
    pythtb = _import('pythtb')
    nb = 2 if ks.spin else 1
    vecs = _full_basis(ks)
    reduced = _site_positions(ks) @ np.linalg.inv(vecs)
    onsite, hops = _site_blocks(ks, nb)
    per = list(range(ks.dim))
    if hasattr(pythtb, 'TBModel'):
        model = pythtb.TBModel(pythtb.Lattice(vecs, reduced, periodic_dirs=per), spinful=ks.spin)
    else:
        model = pythtb.tb_model(ks.dim, ks.space_dim, vecs.tolist(), reduced.tolist(), per=per, nspin=nb)
    model.set_onsite([float(b[0, 0].real) if nb == 1 else b for b in onsite])
    for (u, v, R), b in hops.items():
        model.set_hop(complex(b[0, 0]) if nb == 1 else b, u, v, list(R) + [0] * (ks.space_dim - ks.dim))
    return model


#################################
# TBMODELS, Z2PACK, QSYMM
#################################


def to_tbmodels(ks: KSpace, path: str | None = None) -> Any:
    r'''
    Get the TBmodels ``Model`` of a **KSpace** (`tbmodels.greschd.ch
    <https://tbmodels.greschd.ch>`_), and write its HDF5 file with *path*.
    The orbitals keep their order; the unit cell and the positions are
    those of the plane (or line) of the lattice, in its own coordinates
    (*KSpace.k_basis*): a slab's positions out of its plane are dropped.
    TBmodels' default (convention 2) is the Bloch Hamiltonian of tbkit.

    Parameters
    ----------
    ks : **KSpace** instance
        Hermitian, without overlap.
    path : String or None, default None
        HDF5 file to write (needs h5py).

    Returns
    -------
    model : ``tbmodels.Model``
    '''
    _check_export(ks)
    if path is not None:
        error_handling.file_path(path, 'path')
    tbmodels = _import('tbmodels')
    onsite, hops = _site_blocks(ks, 1)
    uc = np.array(ks.lat.prim_vec, dtype='f8') @ ks.k_basis
    model = tbmodels.Model.from_hop_list(
        hop_list=[(complex(b[0, 0]), u, v, R) for (u, v, R), b in hops.items()],
        size=ks.norb, dim=ks.dim, on_site=onsite[:, 0, 0].real, pos=_reduced_positions(ks), uc=uc,
        contains_cc=False)
    if path is not None:
        model.to_hdf5_file(path)
    return model


def z2pack_hamiltonian(ks: KSpace) -> Callable[[ArrayLike], NDArray[np.complex128]]:
    r'''
    Get the Bloch Hamiltonian of a **KSpace** as the function of
    ``z2pack.hm.System``: :math:`\mathbf{k}` in units of the reciprocal
    vectors (*dim* numbers) to the matrix of *get_ham* (the periodic gauge,
    Z2Pack's ``convention=2``). Z2Pack then needs the orbital positions
    (*to_z2pack* passes them).

    Parameters
    ----------
    ks : **KSpace** instance
        Hermitian, without overlap.

    Returns
    -------
    hamilton : Function
        ``hamilton(k)``, *k* a sequence of *dim* real numbers.
    '''
    _check_export(ks)
    rec = np.array(ks.rec_vec_k)

    def hamilton(k: ArrayLike) -> NDArray[np.complex128]:
        return ks.get_ham(np.asarray(k, dtype='f8') @ rec)
    return hamilton


def to_z2pack(ks: KSpace, bands: int | list[int] | None = None, **kwargs) -> Any:
    r'''
    Get the ``z2pack.hm.System`` of a **KSpace** (`z2pack.greschd.ch
    <https://z2pack.greschd.ch>`_), for its Wilson loops, Chern numbers and
    :math:`\mathbb{Z}_2` invariants: the function of *z2pack_hamiltonian*,
    the dimension *dim* and the reduced orbital positions, with
    ``convention=2``. Z2Pack's Chern number has the sign of *chern_number*
    on the surface ``lambda t1, t2: [t2, t1]`` (Wilson loops along
    :math:`\mathbf{b}_1`, moved along :math:`\mathbf{b}_2`), the opposite
    one on ``[t1, t2]``.

    Parameters
    ----------
    ks : **KSpace** instance
        Hermitian, without overlap.
    bands : Integer, list of integers or None, default None
        The occupied bands (Z2Pack's
        default, None, is the lower half).
    kwargs
        Passed to ``z2pack.hm.System`` (*hermitian_tol*, ...).

    Returns
    -------
    system : ``z2pack.hm.System``

    Examples
    --------
    Example usage::

        system = to_z2pack(models.haldane(), bands=[0])
        result = z2pack.surface.run(system=system, surface=lambda t1, t2: [t2, t1])
        z2pack.invariant.chern(result)   # 1, as chern_number(0)
    '''
    hamilton = z2pack_hamiltonian(ks)
    z2pack = _import('z2pack')
    return z2pack.hm.System(hamilton, dim=ks.dim, pos=_reduced_positions(ks), bands=bands,
                            convention=2, **kwargs)


def to_qsymm(ks: KSpace) -> Any:
    r'''
    Get the qsymm ``BlochModel`` of a **KSpace** (`qsymm
    <https://qsymm.readthedocs.io>`_), for qsymm's symmetry finder
    (``qsymm.symmetries``): :math:`H(\mathbf{k}) = \sum_{\mathbf{d}}
    H_{\mathbf{d}}\,e^{i\mathbf{k}\cdot\mathbf{d}}` over the bond vectors
    :math:`\mathbf{d} = \mathbf{R}+\boldsymbol\tau_j-\boldsymbol\tau_i`
    (orbital positions included, as qsymm expects), in the coordinates of
    *KSpace.k_basis* (Cartesian when the lattice fills its space), with the
    momenta ``k_x``, ``k_y`` (, ``k_z``). Needs sympy.

    Parameters
    ----------
    ks : **KSpace** instance
        Hermitian, without overlap.

    Returns
    -------
    model : ``qsymm.BlochModel``
    '''
    _check_export(ks)
    qsymm = _import('qsymm')
    sympy = _import('sympy')
    tau = ks.orbital_positions()
    full = np.diag(ks.onsite) + ks._onsite_offdiag
    terms = [(int(i), int(j), np.zeros(ks.space_dim), full[i, j]) for i, j in zip(*np.nonzero(full))]
    groups: dict = {}
    for i, j, R, t in terms + ks._hop:
        d = (R + tau[j] - tau[i]) @ ks.k_basis
        key = tuple(np.round(d, 10) + 0.)
        groups.setdefault(key, (d, np.zeros((ks.norb, ks.norb), dtype='c16')))[1][i, j] += t
    ham = {qsymm.model.BlochCoeff(d, sympy.Integer(1)): mat for d, mat in groups.values()}
    return qsymm.BlochModel(ham, momenta=['k_x', 'k_y', 'k_z'][:ks.dim])


#################################
# PYBINDING
#################################


def from_pybinding(obj: Any, spin: bool = False) -> KSpace | tuple[sp.csr_matrix, NDArray[np.float64]]:
    r'''
    Get the **KSpace** of a pybinding ``Lattice`` (`pybinding
    <https://docs.pybinding.site>`_), or the Hamiltonian and orbital
    positions of a pybinding ``Model`` (a finite sample, as pybinding
    builds it from a lattice, a shape and modifiers).

    The sublattices of a ``Lattice`` become the sites, in the order they
    were added, at their positions, with their names as tags; a sublattice
    of several orbitals gives one site per orbital, tagged by its name and
    the orbital index (``'A0'``, ``'A1'``), unless *spin* asks for one
    spinful site per sublattice (each must then have 2 orbitals). Both
    codes store :math:`\langle i,\mathbf{0}|H|j,\mathbf{R}\rangle`; pybinding's
    Bloch phases are :math:`e^{-i\mathbf{k}\cdot\mathbf{R}}`, so its
    :math:`H(\mathbf{k})` is tbkit's :math:`H(-\mathbf{k})` (to its single
    precision, pybinding's default).

    Parameters
    ----------
    obj : ``pybinding.Lattice`` or ``pybinding.Model``
        Without sublattice aliases.
    spin : Boolean, default False
        See above.

    Returns
    -------
    ks : **KSpace** instance, for a ``Lattice``
    ham, positions : Sparse CSR complex matrix and real ndarray of shape (n, 3), for a ``Model``
    '''
    error_handling.boolean(spin, 'spin')
    if hasattr(obj, 'hamiltonian') and hasattr(obj, 'system'):
        ham = sp.csr_matrix(obj.hamiltonian, dtype='c16')
        xyz = obj.system.expanded_positions  # one per orbital
        return ham, np.stack([np.asarray(xyz.x, dtype='f8'), np.asarray(xyz.y, dtype='f8'),
                                  np.asarray(xyz.z, dtype='f8')], axis=1)
    ndim = int(obj.ndim)
    vectors = np.array([np.asarray(v, dtype='f8') for v in obj.vectors]).reshape(ndim, -1)
    subs = sorted(obj.sublattices.items(), key=lambda item: item[1].unique_id)
    error_handling.pybinding_aliases([s.alias_id == s.unique_id for _, s in subs])
    energies = [np.atleast_2d(np.asarray(s.energy, dtype='c16')) for _, s in subs]
    sizes = [len(e) for e in energies]
    if spin:
        error_handling.pybinding_spin(sizes)
    offsets = np.concatenate([[0], np.cumsum(sizes)])
    index = {s.unique_id: p for p, (_, s) in enumerate(subs)}
    pos = np.array([np.asarray(s.position, dtype='f8') for _, s in subs]).reshape(len(subs), -1)
    pos = np.pad(pos, ((0, 0), (0, 3 - pos.shape[1])))
    vectors = np.pad(vectors, ((0, 0), (0, 3 - vectors.shape[1])))
    space_dim = 3 if ndim == 3 or np.any(vectors[:, 2] != 0) or np.any(pos[:, 2] != 0) else 2
    n = int(offsets[-1])
    blocks: dict = {}

    def add(R, p, q, e):
        mat = blocks.setdefault(R, np.zeros((n, n), dtype='c16'))
        mat[offsets[p]:offsets[p + 1], offsets[q]:offsets[q + 1]] += e

    for p, e in enumerate(energies):
        add((0,) * ndim, p, p, e)
    for family in obj.hoppings.values():
        e = np.atleast_2d(np.asarray(family.energy, dtype='c16'))
        for term in family.terms:
            R = tuple(int(c) for c in np.asarray(term.relative_index)[:ndim])
            p, q = index[term.from_id], index[term.to_id]
            add(R, p, q, e)
            add(tuple(-c for c in R), q, p, e.conj().T)
    R_arr = np.array(sorted(blocks), dtype='i8').reshape(len(blocks), ndim)
    ham = np.array([blocks[tuple(r)] for r in R_arr.tolist()])
    prim_vec = [tuple(float(c) for c in v[:space_dim]) for v in vectors]
    names = [str(name) for name, _ in subs]
    if spin:
        return _hr_kspace(prim_vec, pos[:, :space_dim], names, R_arr, ham, 0., 1e-10, True)
    tags = [name if size == 1 else '{}{}'.format(name, o) for name, size in zip(names, sizes)
            for o in range(size)]
    positions = np.repeat(pos[:, :space_dim], sizes, axis=0)
    return _hr_kspace(prim_vec, positions, tags, R_arr, ham, 0., 1e-10)


def to_pybinding(ks: KSpace) -> Any:
    r'''
    Get the pybinding ``Lattice`` of a **KSpace**: one sublattice per site
    (with a 2 x 2 onsite matrix for a spinful model), named by its tag (by
    its tag and index when tags repeat), and one hopping of each conjugate
    pair (pybinding adds the other). See *from_pybinding* for the conventions.
    pybinding then builds finite samples (``pb.Model(lattice, shape)``)
    and its KPM tools on it.

    Parameters
    ----------
    ks : **KSpace** instance
        Hermitian, without overlap.

    Returns
    -------
    lattice : ``pybinding.Lattice``
    '''
    _check_export(ks)
    pb = _import('pybinding')
    nb = 2 if ks.spin else 1
    onsite, hops = _site_blocks(ks, nb)
    tags = [str(t) for t in ks.tags]
    names = tags if len(set(tags)) == len(tags) else ['{}{}'.format(t, s) for s, t in enumerate(tags)]
    lattice = pb.Lattice(*[list(v) for v in ks.lat.prim_vec])
    lattice.add_sublattices(*[(name, list(r), float(b[0, 0].real) if nb == 1 else b)
                                 for name, r, b in zip(names, _site_positions(ks), onsite)])
    lattice.add_hoppings(*[(list(R), names[u], names[v], complex(b[0, 0]) if nb == 1 else b)
                              for (u, v, R), b in hops.items()])
    return lattice


#################################
# KWANT
#################################


def _site_family(kwant: Any, tinyarray: Any, positions: NDArray[np.float64], name: str) -> Any:
    '''
    Private function. A Kwant site family of one orbital per site, whose
    tags (i,) index *positions*.
    '''
    class Sites(kwant.builder.SiteFamily):

        def __init__(self) -> None:
            super().__init__('tbkit.interop.{}.{}'.format(name, id(self)), name, 1)

        def normalize_tag(self, tag):
            return tinyarray.array((int(tag[0]),))

        def pos(self, tag):
            return positions[tag[0]]

    return Sites()


def to_kwant(model: System | Transport, sys: System | None = None) -> Any:
    r'''
    Get a Kwant ``Builder`` (`kwant-project.org <https://kwant-project.org>`_)
    of a **System**, or of a **Transport** device with its leads.

    Every site (orbital) of the model becomes a Kwant site of one orbital
    at its position, with the onsite energies and hoppings of the
    Hamiltonian (*System.ham*, after *get_ham*, or the device of the
    *Transport*). Each lead of a *Transport* becomes a copy of its first
    cell, joined to the device by the lead's coupling, followed by a
    ``kwant.builder.ModesLead`` whose modes Kwant computes from the lead's
    :math:`h_0` and :math:`v` (and its conservation law, so that
    ``smatrix.submatrix((lead, block), ...)`` resolves its blocks). The
    added cell does not change the transmissions; the phases of the
    scattering amplitudes are those of Kwant's modes.

    Parameters
    ----------
    model : **System** or **Transport** instance
        Hermitian.
    sys : **System** instance or None, default None
        For a *Transport*: the
        system its device was built from, for the site positions (those
        added by *attach_lead* with ``extend=True`` included). Without it,
        site *i* sits at :math:`(i, 0)`.

    Returns
    -------
    builder : ``kwant.Builder``
        Finalize it (``builder.finalized()``) to compute.

    Examples
    --------
    Example usage::

        fsys = to_kwant(tr, sys).finalized()
        kwant.smatrix(fsys, 0.3).transmission(1, 0)   # = tr.transmission([0.3])[0]
    '''
    if isinstance(model, Transport):
        ham = model._ham
        if sys is None:
            positions = np.stack([np.arange(ham.shape[0], dtype='f8'), np.zeros(ham.shape[0])], axis=1)
        else:
            error_handling.sys(sys)
            error_handling.lead_device_size(sys.lat.sites + len(model._added_tags), ham.shape[0])
            positions = model._site_positions(sys)
        leads, laws = model.leads, model._laws
    else:
        error_handling.sys(model)
        error_handling.empty_hop(model.hop)
        error_handling.kwant_ham(model.ham.shape[0], model.lat.sites)
        ham = sp.csr_matrix(model.ham, dtype='c16')
        positions = model._coords()
        leads, laws = [], []
    error_handling.hermitian_matrix(ham, leads)
    kwant = _import('kwant')
    tinyarray = _import('tinyarray')
    family = _site_family(kwant, tinyarray, positions, 'device')
    builder = kwant.Builder()
    sites = [family(i) for i in range(ham.shape[0])]
    _fill(builder, sites, sp.csr_matrix(ham))
    for l, ((h0, v, tau, lead_sites), law) in enumerate(zip(leads, laws)):
        centre = positions[lead_sites].mean(axis=0)
        cell_family = _site_family(kwant, tinyarray, np.repeat(centre[None], len(h0), axis=0), 'lead{}'.format(l))
        cell = [cell_family(q) for q in range(len(h0))]
        _fill(builder, cell, sp.csr_matrix(h0))
        for p, s in enumerate(lead_sites):
            for q in np.flatnonzero(tau[p]):
                builder[sites[s], cell[q]] = complex(tau[p, q])
        builder.leads.append(kwant.builder.ModesLead(_modes(kwant, h0, v, law), cell, ()))
    return builder


def _fill(builder: Any, sites: list, ham: sp.csr_matrix) -> None:
    '''
    Private function. Set the onsite energies and the hoppings (one
    triangle; Kwant adds the conjugates) of the Hermitian *ham* on *sites*.
    '''
    diag = ham.diagonal()
    for i, site in enumerate(sites):
        builder[site] = complex(diag[i])
    upper = sp.triu(ham, 1).tocoo()
    for i, j, t in zip(upper.row, upper.col, upper.data):
        builder[sites[i], sites[j]] = complex(t)


def _modes(kwant: Any, h0: NDArray[np.complex128], v: NDArray[np.complex128],
           law: list | None) -> Callable:
    r'''
    Private function. The modes function of a Kwant ``ModesLead`` for the
    lead :math:`(h_0, v)`: Kwant's hopping towards the scattering region is
    :math:`\langle n|H|n-1\rangle = v^\dagger`.
    '''
    symmetry = None if law is None else kwant.physics.DiscreteSymmetry(
        projectors=[sp.csr_matrix(p) for p in law])

    def modes(energy, args=(), *, params=None):
        return kwant.physics.modes(h0 - energy * np.eye(len(h0)), v.conj().T, discrete_symmetry=symmetry)
    return modes


def from_kwant(fsys: Any, params: dict | None = None) -> tuple[sp.csr_matrix, NDArray[np.float64]]:
    r'''
    Get the Hamiltonian matrix and the orbital positions of a finalized
    Kwant system (``builder.finalized()``), with the parameters *params* of
    its value functions. Its leads are not imported (attach tbkit leads to
    the result with *Transport*). Build a **Transport** (``Transport(ham)``)
    or a *System* from them.

    Parameters
    ----------
    fsys : ``kwant.system.FiniteSystem``
        Finalized, finite.
    params : Dictionary or None, default None
        Passed to ``hamiltonian_submatrix``.

    Returns
    -------
    ham : Sparse CSR complex matrix
    positions : Real ndarray, shape (n_orbitals, space dimension)
        The position of the site of each orbital.
    '''
    error_handling.kwant_finite(fsys)
    ham = sp.csr_matrix(fsys.hamiltonian_submatrix(sparse=True, params=params), dtype='c16')
    pos = np.array([np.asarray(s.pos, dtype='f8') for s in fsys.sites])
    counts = np.ones(len(pos), dtype=int)
    if getattr(fsys, 'site_ranges', None) is not None:
        ranges = np.asarray(fsys.site_ranges)
        counts = np.repeat(ranges[:-1, 1], np.diff(ranges[:, 0]))
    error_handling.kwant_orbitals(int(counts.sum()), ham.shape[0])
    return ham, np.repeat(pos, counts, axis=0)
