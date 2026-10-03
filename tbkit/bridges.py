r'''
Bridges between the two pipelines of tbkit: the real-space *System* (a
finite lattice, its sparse Hamiltonian) and the reciprocal-space *KSpace*
(the Bloch Hamiltonian :math:`H(\mathbf{k})` of the infinite lattice).

* *kspace_from_system* reads the Bloch model off a translation-invariant
  *System* built the real-space way (*get_lattice*, *set_hopping* by
  neighbour order, *set_onsite*, ...).
* *finite_model* and *finite_system* cut a finite sample (open boundaries,
  or a torus) out of a *KSpace* model: its sparse Hamiltonian with the site
  positions, or a ready-to-use *System*.

Both use the convention of *KSpace.finite_ham*: the hopping
:math:`t_{ij}(\mathbf{R}) = \langle i,\mathbf{0}|H|j,\mathbf{R}\rangle`
joins orbital *i* of every cell to orbital *j* of the cell :math:`\mathbf{R}`
further. So, for the same model, the spectrum of a System on an
:math:`N_1\times N_2` torus is the set of KSpace bands on the
:math:`N_1\times N_2` Brillouin-zone mesh.
'''
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
import scipy.sparse as sp

import tbkit.error_handling as error_handling
import tbkit.neighbours as neighbours
from tbkit.kspace import KSpace
from tbkit.lattice import Lattice
from tbkit.orbital import OrbitalSystem
from tbkit.system import System, HOP_DTYPE, ATOL, _upper


def _coords(lat: Lattice) -> NDArray[np.float64]:
    '''
    Private function. Site coordinates of a finite lattice, shape (sites, space_dim).
    '''
    return np.stack([lat.coor[f] for f in ('x', 'y', 'z')[:lat.space_dim]], axis=1)


def cell_orbitals(lat: Lattice) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    r'''
    Get, for every site of a finite lattice (*lat.coor*), the unit-cell
    orbital it is a copy of and its cell: site *s* sits at
    :math:`\boldsymbol\tau_{o_s} + \sum_k n_{s,k}\mathbf{a}_k`, with the tag
    of orbital :math:`o_s` (within ``ATOL``).

    :param lat: **Lattice** instance, after *get_lattice* (sites may have
        been removed, but not moved).

    :returns:
        * **cells** -- Integer ndarray, shape (sites, dim). :math:`n_{s,k}`.
        * **orbitals** -- Integer ndarray, shape (sites,). :math:`o_s`.
    '''
    error_handling.lat(lat)
    error_handling.empty_coor(lat.coor)
    a, tau, tags = neighbours._geometry(lat)
    pos = _coords(lat)
    pinv = np.linalg.pinv(a)
    match = np.zeros((lat.sites, len(tau)), bool)
    cells = np.zeros((lat.sites, len(tau), len(a)), int)
    for o in range(len(tau)):
        rel = pos - tau[o]
        n = np.rint(rel @ pinv).astype(int)
        match[:, o] = (np.linalg.norm(n @ a - rel, axis=1) < ATOL) & (lat.coor['tag'] == tags[o])
        cells[:, o] = n
    error_handling.lattice_sites(match.sum(axis=1))
    orb = np.argmax(match, axis=1)
    return cells[np.arange(lat.sites), orb], orb


def kspace_from_system(sys: System, periodic: bool | None = None, tol: float = 1e-9) -> KSpace:
    r'''
    Get the Bloch Hamiltonian of a periodic model defined the real-space
    way: every matrix element :math:`H_{ab}` of the finite *System* (built
    by *get_ham*) joins orbital :math:`o_a` of cell :math:`\mathbf{n}_a` to
    orbital :math:`o_b` of cell :math:`\mathbf{n}_b` (see *cell_orbitals*),
    i.e. is the hopping :math:`t_{o_ao_b}(\mathbf{n}_b-\mathbf{n}_a)` of the
    infinite lattice. Each such hopping must take the same value in every
    cell where it appears (bonds cut by the boundary are simply missing).
    The lattice must be large enough to hold every bond of the model.

    A System on a torus (*periodic*, e.g. from *finite_system*) has bonds
    that wrap around: their lattice vectors are then taken modulo the
    :math:`N_1\times N_2\times\dots` cells of *get_lattice*, as the shortest
    image (so the model's range must stay below half the torus).

    A Hermitian hopping matrix gives a Hermitian model (one representative
    per bond, *set_hopping* adding the conjugate); otherwise every matrix
    element is kept, with ``hermitian=False``. Onsite energies are taken
    orbital by orbital.

    :param sys: **System** instance with its hoppings set (one orbital per site).
    :param periodic: Boolean. Default value None: True if *sys* has
        periodic boundaries along some primitive vector (``System(lat,
        periodic=...)``, or *finite_system* with ``periodic=True``). The
        System is a torus of the *lat.n1* x *lat.n2* (x *lat.n3*) cells of
        *get_lattice*.
    :param tol: Positive real number. Default value 1e-9. Largest allowed
        difference between equivalent matrix elements.

    :returns:
        * **ks** -- **KSpace** instance (spinless), on a new **Lattice**
          with the *unit_cell* and *prim_vec* of *sys.lat*.

    Example usage::

        lat = lattices.honeycomb()
        lat.get_lattice(6, 6)
        sys = System(lat)
        sys.set_hopping([{'n': 1, 't': -1.}, {'n': 2, 't': 0.1}])
        gra = kspace_from_system(sys)   # the Bloch model of the same graphene
    '''
    error_handling.sys(sys)
    error_handling.not_orbital_system(sys, OrbitalSystem)
    if periodic is None:
        periodic = any(sys.periodic)
    error_handling.boolean(periodic, 'periodic')
    error_handling.positive_real(tol, 'tol')
    sys.get_ham()
    cells, orb = cell_orbitals(sys.lat)
    ham = sys.ham.tocoo()
    rows, cols, vals = ham.row, ham.col, ham.data.astype('c16')
    R = cells[cols] - cells[rows]
    if periodic:
        sizes = np.array((sys.lat.n1, sys.lat.n2, sys.lat.n3)[:R.shape[1]])
        R = (R + sizes // 2) % sizes - sizes // 2  # shortest image, in [-N/2, N/2)
        error_handling.torus_range(R, sizes)
    keys = np.column_stack([orb[rows], orb[cols], R])
    uniq, first, inv = np.unique(keys, axis=0, return_index=True, return_inverse=True)
    inv = inv.ravel()
    values = vals[first]
    error_handling.translation_invariant(float(np.abs(vals - values[inv]).max()), tol)
    off = ham - sp.diags(ham.diagonal())
    hermitian = bool(abs(off - off.conj().T).max() <= tol)
    lat = Lattice(unit_cell=list(sys.lat.unit_cell), prim_vec=list(sys.lat.prim_vec))
    ks = KSpace(lat)
    list_hop = []
    for key, t in zip(uniq, values):
        i, j, R = int(key[0]), int(key[1]), tuple(int(n) for n in key[2:])
        if i == j and not any(R):
            ks.onsite[i] = t
            continue
        nonzero = [n for n in R if n != 0]
        if hermitian and not ((nonzero and nonzero[0] > 0) or (not nonzero and i < j)):
            continue  # the conjugate of a kept representative
        list_hop.append({'i': i, 'j': j, 'R': R, 't': complex(t)})
    ks.set_hopping(list_hop, hermitian=hermitian)
    return ks


def _finite_sites(ks: KSpace, n_cells: tuple[int, ...]) -> tuple[NDArray[np.float64], NDArray]:
    '''
    Private function. Positions and tags of the rows of *ks.finite_ham(n_cells)*.
    '''
    n_tot = int(np.prod(n_cells))
    cells = np.array(np.unravel_index(np.arange(n_tot), n_cells, order='F')).T
    a = np.array(ks.lat.prim_vec, dtype='f8')
    pos = (cells @ a)[:, None, :] + ks.orbital_positions()[None, :, :]
    tags = np.repeat(ks.tags, 2) if ks.spin else ks.tags
    return pos.reshape(n_tot * ks.norb, ks.space_dim), np.tile(tags, n_tot)


def finite_model(
    ks: KSpace, n_cells: int | tuple[int, ...], periodic: bool = False,
) -> tuple[sp.csr_matrix, NDArray[np.float64], NDArray]:
    r'''
    Get a finite sample of a *KSpace* model: its sparse Hamiltonian (that of
    *KSpace.finite_ham*, open or periodic boundaries), with the position
    and tag of every row. Spinful models are supported (the two spin
    components of a site share its position).

    :param ks: **KSpace** instance.
    :param n_cells: Positive integer, or tuple of *dim* positive integers.
        Number of unit cells along each primitive vector.
    :param periodic: Boolean. Default value False. Periodic boundaries (a torus).

    :returns:
        * **ham** -- Complex CSR matrix, shape (N*norb, N*norb).
        * **positions** -- Real ndarray, shape (N*norb, space_dim). Row
          :math:`c\,\mathrm{norb}+o` sits at :math:`\boldsymbol\tau_o + \sum_k n_k\mathbf{a}_k`
          (cell *c* has :math:`c = n_1 + N_1 n_2 + N_1N_2 n_3`).
        * **tags** -- String ndarray, shape (N*norb,). Sublattice tags.
    '''
    error_handling.kspace(ks, KSpace)
    error_handling.nk(n_cells, ks.dim)
    error_handling.boolean(periodic, 'periodic')
    n_cells = (n_cells,) * ks.dim if isinstance(n_cells, int) else n_cells
    pos, tags = _finite_sites(ks, n_cells)
    return ks.finite_ham(n_cells, periodic=periodic, sparse=True), pos, tags


def finite_system(ks: KSpace, n_cells: int | tuple[int, ...], periodic: bool = False) -> System:
    r'''
    Build a finite *System* out of a (spinless) *KSpace* model: *lat.coor*
    holds the sites of :math:`N_1\times N_2\times\dots` unit cells (in the
    order of *finite_model*), and *sys.hop* / *sys.onsite* the model's
    hoppings and onsite energies, so that *sys.get_ham()* gives
    *ks.finite_ham(n_cells, periodic)*. Every real-space tool (*get_eig*,
    *get_ldos*, *Plot*, disorder, ...) then applies.

    In *sys.hop*, each bond is stored with the conventions of
    *System.set_hopping*: a Hermitian model keeps one orientation per bond
    (angle in [0, 180), *get_ham* adds the conjugate); a non-reciprocal
    one keeps both, the missing direction with a zero amplitude. 'n' is the
    neighbour order of the bond (0 for coinciding orbitals), 'tag' the
    sublattice pair. On a torus, a bond that wraps around keeps its short
    bond vector for its angle, and a hopping that wraps onto its own site
    is added to the onsite energy; the System has periodic boundaries
    (``sys.periodic``), so that further *set_hopping* calls, *Plot* and
    *get_bott_index* treat it as a torus.

    :param ks: **KSpace** instance, with ``spin=False``.
    :param n_cells: Positive integer, or tuple of *dim* positive integers.
    :param periodic: Boolean. Default value False. Periodic boundaries (a torus).

    :returns:
        * **sys** -- **System** instance, with its hoppings set.

    Example usage::

        sys = finite_system(gra, (30, 30), periodic=True)
        sys.get_ham()
        sys.get_eig()   # = gra.mesh_bands((30, 30)), sorted
    '''
    error_handling.kspace(ks, KSpace)
    error_handling.nk(n_cells, ks.dim)
    error_handling.boolean(periodic, 'periodic')
    error_handling.spinless(ks.spin)
    n_cells = (n_cells,) * ks.dim if isinstance(n_cells, int) else n_cells
    pos, tags = _finite_sites(ks, n_cells)
    lat = Lattice(unit_cell=list(ks.lat.unit_cell), prim_vec=list(ks.lat.prim_vec))
    coor = np.zeros(len(pos), dtype=lat.dtype)
    for c, f in enumerate(('x', 'y', 'z')[:ks.space_dim]):
        coor[f] = pos[:, c]
    coor['tag'] = tags
    lat.coor, lat.sites = coor, len(coor)
    lat.n1, lat.n2, lat.n3 = (tuple(n_cells) + (1, 1))[:3]
    sys = System(lat, periodic=periodic)
    _, n_tot, entries = ks._finite_entries(n_cells, periodic)
    tau = ks.orbital_positions()
    rows = np.concatenate([np.zeros(0, int)] + [e[0] for e in entries])
    cols = np.concatenate([np.zeros(0, int)] + [e[1] for e in entries])
    t = np.concatenate([np.zeros(0, 'c16')] + [np.full(len(e[0]), e[6], 'c16') for e in entries])
    d = np.concatenate([np.zeros((0, ks.space_dim))]
                              + [np.repeat((e[5] + tau[e[3]] - tau[e[2]])[None], len(e[0]), axis=0)
                                 for e in entries])
    n_sites = len(pos)
    diag = rows == cols
    sys.onsite = np.tile(ks.onsite, n_tot) + (np.bincount(rows[diag], t[diag].real, n_sites)
                                                               + 1j * np.bincount(rows[diag], t[diag].imag, n_sites))
    rows, cols, t, d = rows[~diag], cols[~diag], t[~diag], d[~diag]
    dist = np.linalg.norm(d, axis=1)
    zero = dist <= ATOL
    ang = neighbours._bond_angles(d)
    up = np.where(zero, rows < cols, _upper(ang))
    ang_up = np.where(zero, 0., np.maximum(np.where(up, ang, neighbours._bond_angles(-d)), 0.))
    if ks._nonreciprocal:
        i = np.concatenate([rows, cols])
        j = np.concatenate([cols, rows])
        t = np.concatenate([t, np.zeros(len(t), 'c16')])
        ang = np.concatenate([np.where(up, ang_up, ang_up - 180.), np.where(up, ang_up - 180., ang_up)])
        dist = np.concatenate([dist, dist])
    else:
        i, j, t, ang, dist = rows[up], cols[up], t[up], ang_up[up], dist[up]
    hop = np.zeros(len(i), dtype=HOP_DTYPE)
    hop['n'] = neighbours.shell_index(ks.lat, dist)
    hop['i'], hop['j'], hop['t'], hop['ang'] = i, j, t, ang
    hop['tag'] = np.char.add(tags[i], tags[j])
    sys.hop = hop
    return sys
