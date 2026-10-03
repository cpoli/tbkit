r'''
Neighbour shells of a periodic lattice, and the neighbour-order hoppings of
*KSpace.set_hopping*.

*System.set_hopping* addresses the bonds of a finite lattice by neighbour
order ``'n'`` (1st, 2nd, ... shortest distance), optionally narrowed by
bond angle ``'ang'`` and sublattice-pair tag ``'tag'``. This module finds
the same bonds in a *periodic* lattice, from *unit_cell* and *prim_vec*
only: each bond is an intra-cell orbital pair :math:`(i, j)` plus a lattice
vector :math:`\mathbf{R} = n_1\mathbf{a}_1+\dots`, with bond vector
:math:`\mathbf{d} = \mathbf{R} + \boldsymbol\tau_j - \boldsymbol\tau_i`,
and turns a neighbour-order hopping list into the explicit
``{'i', 'j', 'R', 't'}`` list of *KSpace.set_hopping*.

The conventions are those of *System*: distances are grouped after rounding
to 4 decimals and matched within ``ATOL = 1e-3``; every bond is kept once,
oriented from *i* to *j* so that its angle (that of the projection of
:math:`\mathbf{d}` on the :math:`(x, y)` plane, in degrees; a vertical bond
has angle 0 pointing up, -180 pointing down) lies in :math:`[0, 180)`; its
tag is ``tag_i + tag_j``.
'''
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

import tbkit.error_handling as error_handling
import tbkit.values as values
from tbkit.lattice import Lattice
from tbkit.system import ATOL, _upper


PI = np.pi


def _geometry(lat: Lattice) -> tuple[NDArray, NDArray, NDArray]:
    '''
    Private function. Primitive vectors (dim, space_dim), orbital positions
    (n_sites, space_dim) and tags of *lat*.
    '''
    a = np.array(lat.prim_vec, dtype='f8')
    tau = np.array([dic['r0'] for dic in lat.unit_cell], dtype='f8')
    tags = np.array([dic['tag'] for dic in lat.unit_cell])
    return a, tau, tags


def _bond_angles(d: NDArray[np.float64]) -> NDArray[np.float64]:
    '''
    Private function. Angles (degrees) of the bond vectors *d* (shape
    (nb, space_dim)), with the vertical-bond rule of *System.get_distances*.
    '''
    ang = 180. / PI * np.arctan2(d[:, 1], d[:, 0])
    if d.shape[1] == 3:
        vertical = np.hypot(d[:, 0], d[:, 1]) < 1e-6 * np.maximum(np.linalg.norm(d, axis=1), 1.)
        ang[vertical & (d[:, 2] > 0)] = 0.
        ang[vertical & (d[:, 2] <= 0)] = -180.
    return ang


def _candidates(lat: Lattice, m: int) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    '''
    Private function. Every (i, j, R) with :math:`|n_k|\\le m`, and its bond
    vector: orbitals i, j (nb,), cells R (nb, dim), bonds d (nb, space_dim).
    '''
    a, tau, _ = _geometry(lat)
    dim, ns = len(a), len(tau)
    cells = np.array(np.meshgrid(*[np.arange(-m, m + 1)] * dim, indexing='ij')).reshape(dim, -1).T
    i, j = [m.ravel() for m in np.meshgrid(np.arange(ns), np.arange(ns), indexing='ij')]
    ii = np.repeat(i[None, :], len(cells), axis=0).ravel()
    jj = np.repeat(j[None, :], len(cells), axis=0).ravel()
    R = np.repeat(cells, len(i), axis=0)
    d = R @ a + tau[jj] - tau[ii]
    return ii, jj, R, d


def neighbour_shells(lat: Lattice, n_max: int) -> NDArray[np.float64]:
    r'''
    Get the distances of the first *n_max* neighbour shells of the infinite
    periodic lattice (the ``dist_uni[1:]`` of *System*, without its edges).

    :param lat: **Lattice** instance. Only *unit_cell* and *prim_vec* are used.
    :param n_max: Positive integer. Number of shells.

    :returns:
        * **shells** -- Real ndarray, shape (n_max,). Distances, ascending
          (rounded to 4 decimals, as in *System*).
    '''
    error_handling.lat(lat)
    error_handling.positive_int(n_max, 'n_max')
    return _shells(lat, n_max)[0]


def _shells(lat: Lattice, n_max: int, r_max: float | None = None) -> tuple[NDArray, int]:
    r'''
    Private function. The first *n_max* shell distances, and the cell range
    *m* whose candidates hold every bond of those shells: a bond of length
    :math:`|\mathbf{d}|` has :math:`|n_k| \le |\mathbf{b}_k|(|\mathbf{d}| + \max|\boldsymbol\tau_j-\boldsymbol\tau_i|)/2\pi`,
    so a box of half-width *m* is complete up to the radius
    :math:`2\pi m/\max_k|\mathbf{b}_k| - \max|\boldsymbol\tau_j-\boldsymbol\tau_i|`.
    With *r_max*, every shell up to (at least) that radius instead.
    '''
    a, tau, _ = _geometry(lat)
    b_norm = np.linalg.norm(np.linalg.pinv(a), axis=0).max()  # max |b_k| / 2 pi
    dtau = np.linalg.norm(tau[:, None, :] - tau[None, :, :], axis=2).max()
    m = 1
    while True:
        _, _, _, d = _candidates(lat, m)
        dist = np.linalg.norm(d, axis=1)
        radius = m / b_norm - dtau - 2 * ATOL
        uni = np.unique(dist[dist > ATOL].round(4))
        uni = uni[uni <= radius]
        if r_max is not None:
            if radius > r_max:
                return uni, m
        elif len(uni) >= n_max:
            return uni[:n_max], m
        m += 1


def shell_index(lat: Lattice, dist: NDArray[np.float64]) -> NDArray[np.int64]:
    '''
    Private function. Neighbour order of bond lengths *dist* (0 for a
    length below ATOL, as *System*'s ``dist_uni[0]``).
    '''
    dist = np.asarray(dist, dtype='f8')
    if not np.any(dist > ATOL):
        return np.zeros(len(dist), dtype=int)
    shells, _ = _shells(lat, 0, r_max=float(dist.max()) + ATOL)
    n = np.argmin(np.abs(dist[:, None] - shells[None, :]), axis=1) + 1
    return np.where(dist > ATOL, n, 0)


def neighbour_bonds(lat: Lattice, n_max: int) -> NDArray:
    r'''
    Get every bond of the first *n_max* neighbour shells of the infinite
    periodic lattice, once each, oriented so that its angle lies in
    :math:`[0, 180)` (see the module docstring).

    :param lat: **Lattice** instance. Only *unit_cell* and *prim_vec* are used.
    :param n_max: Positive integer. Number of shells.

    :returns:
        * **bonds** -- Structured ndarray with fields 'n' (shell), 'i', 'j'
          (orbitals), 'R' (lattice vector, *dim* integers), 'ang' (degrees,
          in [0, 180)), 'tag' (``tag_i + tag_j``) and 'dis' (length),
          sorted by (n, i, j, R).

    Example usage::

        bonds = neighbour_bonds(lattices.honeycomb(), 1)
        # three 'ab'/'ba' bonds of length 1, at angles 30, 90 and 150
    '''
    error_handling.lat(lat)
    error_handling.positive_int(n_max, 'n_max')
    return _bonds(lat, n_max)


def _bonds(lat: Lattice, n_max: int) -> NDArray:
    '''
    Private function. *neighbour_bonds* without the argument checks.
    '''
    shells, m = _shells(lat, n_max)
    i, j, R, d = _candidates(lat, m)
    dist = np.linalg.norm(d, axis=1)
    diff = np.abs(dist[:, None] - shells[None, :])
    n = np.argmin(diff, axis=1)
    ang = _bond_angles(d)
    keep = (diff[np.arange(len(dist)), n] <= ATOL) & (dist > ATOL) & _upper(ang)
    _, _, tags = _geometry(lat)
    dim = len(lat.prim_vec)
    bonds = np.zeros(int(keep.sum()), dtype=[('n', 'u2'), ('i', 'u4'), ('j', 'u4'), ('R', 'i8', (dim,)),
                                                          ('ang', 'f8'), ('tag', 'U2'), ('dis', 'f8')])
    bonds['n'] = n[keep] + 1
    bonds['i'], bonds['j'], bonds['R'] = i[keep], j[keep], R[keep]
    bonds['ang'] = np.maximum(ang[keep], 0.)
    bonds['tag'] = np.char.add(tags[i[keep]], tags[j[keep]])
    bonds['dis'] = dist[keep]
    order = np.lexsort(tuple(bonds['R'][:, k] for k in range(dim)[::-1]) + (bonds['j'], bonds['i'], bonds['n']))
    return bonds[order]


def neighbour_hoppings(
    lat: Lattice, list_hop: list[dict], spin: bool = False, hermitian: bool = True,
) -> list[dict]:
    r'''
    Turn hoppings given by neighbour order, as in *System.set_hopping*,
    into the explicit ``{'i', 'j', 'R', 't'}`` list of *KSpace.set_hopping*.

    :param lat: **Lattice** instance. Only *unit_cell* and *prim_vec* are used.
    :param list_hop: List of dictionaries with keys ('n', 't') and
        optionally 'ang' and/or 'tag' (see *System.set_hopping*):

        * 'n': Positive integer. Neighbour order (1 for nearest neighbours).
        * 'ang': Real number, in degrees. Only the bonds along this angle.
          An angle in :math:`[0, 180)` selects the bonds oriented as they
          are stored (see *neighbour_bonds*), an angle in :math:`[-180, 0)`
          the same bonds oriented the other way (from *j* to *i*).
        * 'tag': String of length 2. Only the bonds from a site of the
          first sublattice to one of the second, in the orientation above.
        * 't': Complex number (or, if *spin*, a 2x2 matrix): the matrix
          element :math:`\langle i|H|j\rangle` along the oriented bond. Or a
          value function (see *KSpace.set_hopping*), passed on as it is (as
          ``t(j, i)^\dagger`` for a reversed bond of a Hermitian model).

        When several dictionaries select the same bond, the last one wins,
        as in *System.set_hopping*.
    :param spin: Boolean. Default value False. See *KSpace*.
    :param hermitian: Boolean. Default value True. If True, each bond is
        returned once, in its stored orientation (a hopping given along a
        negative angle becomes its conjugate transpose); if False, the two
        orientations are independent hoppings (non-reciprocal models).

    :returns:
        * **list_hop** -- List of dictionaries ('i', 'j', 'R', 't').

    Example usage::

        # graphene: nearest (t) and next-nearest (t2) neighbours
        explicit = neighbour_hoppings(lattices.honeycomb(),
                                                [{'n': 1, 't': -1.}, {'n': 2, 't': 0.1}])
    '''
    error_handling.lat(lat)
    error_handling.set_hopping_neighbours(list_hop, spin)
    error_handling.boolean(spin, 'spin')
    error_handling.boolean(hermitian, 'hermitian')
    bonds = _bonds(lat, max(dic['n'] for dic in list_hop))
    chosen = {}
    for dic in list_hop:
        shell = bonds[bonds['n'] == dic['n']]
        reverse = 'ang' in dic and dic['ang'] < 0
        ind = np.ones(len(shell), bool)
        if 'ang' in dic:
            error_handling.shell_angle(dic['ang'], shell['ang'])
            ind &= np.isclose(dic['ang'] + 180. * reverse, shell['ang'], atol=ATOL)
        if 'tag' in dic:
            ind &= shell['tag'] == (dic['tag'][::-1] if reverse else dic['tag'])
        error_handling.index(ind, dic)
        t = dic['t']
        if reverse and hermitian:
            t_rev = values.reversed_conj(t) if callable(t) else                 np.conj(t) if np.ndim(t) == 0 else np.asarray(t).conj().T
        for bond in shell[ind]:
            i, j, R = int(bond['i']), int(bond['j']), tuple(int(c) for c in bond['R'])
            if not reverse:
                chosen[(i, j, R)] = t
            elif hermitian:
                chosen[(i, j, R)] = t_rev
            else:
                chosen[(j, i, tuple(-c for c in R))] = t
    return [{'i': i, 'j': j, 'R': R, 't': t} for (i, j, R), t in chosen.items()]
