r"""
Slater-Koster two-centre integrals (J. C. Slater and G. F. Koster, Phys.
Rev. 94, 1498 (1954)) between s, p and d orbitals.

The matrix element between orbital :math:`\alpha` on one atom and orbital
:math:`\beta` on a neighbour at :math:`\mathbf{d} = d\,(l, m, n)` depends
only on a handful of bond integrals :math:`V_{ll'\mu}`
(:math:`\mu = \sigma, \pi, \delta`: the angular momentum about the bond),
through the direction cosines :math:`(l, m, n)` -- e.g.

.. math::

    E_{s,x} = l\,V_{sp\sigma}\, ,\qquad
    E_{x,x} = l^2 V_{pp\sigma} + (1-l^2) V_{pp\pi}\, ,\qquad
    E_{x,y} = lm\,(V_{pp\sigma} - V_{pp\pi})\, .

Rather than tabulating all 81 such formulas, they are generated: in a
frame whose z axis lies along the bond, the matrix is diagonal in the
angular momentum about the bond, with entries :math:`V_{ll'\mu}`; rotating
the frame onto :math:`\hat{\mathbf{d}}` gives every entry,
:math:`E(\hat{\mathbf{d}}) = D_{l}(R)\,E(\hat z)\,D_{l'}(R)^T`, with
:math:`D_1 = R` for the p orbitals and :math:`D_2` its action on the
quadratic forms of the d orbitals. This is exactly the Slater-Koster table
(checked against it in the tests), in the usual real-orbital basis:

``'s'``, ``'px'``, ``'py'``, ``'pz'``, ``'dxy'``, ``'dyz'``, ``'dzx'``,
``'dx2-y2'``, ``'dz2'``.

Bond integrals are given as a dictionary, with keys ``'ss_sigma'``,
``'sp_sigma'``, ``'pp_sigma'``, ``'pp_pi'``, ``'sd_sigma'``,
``'pd_sigma'``, ``'pd_pi'``, ``'dd_sigma'``, ``'dd_pi'``, ``'dd_delta'``
(missing keys are zero). Between different atoms, :math:`V_{sp\sigma}`
(s on the first atom, p on the second) and :math:`V_{ps\sigma}` (p on the
first, s on the second) differ: give ``'ps_sigma'``, ``'ds_sigma'``,
``'dp_sigma'``, ``'dp_pi'`` for the latter (they default to their
homonuclear counterparts).
"""
from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

import tbkit.error_handling as error_handling


#: Real orbitals, with their angular momentum l.
ORBITALS = {'s': 0, 'px': 1, 'py': 1, 'pz': 1, 'dxy': 2, 'dyz': 2, 'dzx': 2,
                   'dx2-y2': 2, 'dz2': 2}

#: Bond-integral keys (the heteronuclear ps/ds/dp keys are optional).
PARAMS = ('ss_sigma', 'sp_sigma', 'ps_sigma', 'pp_sigma', 'pp_pi', 'sd_sigma',
                'ds_sigma', 'pd_sigma', 'pd_pi', 'dp_sigma', 'dp_pi', 'dd_sigma',
                'dd_pi', 'dd_delta')

_P = ['px', 'py', 'pz']
_D = ['dxy', 'dyz', 'dzx', 'dx2-y2', 'dz2']
_S3 = np.sqrt(3.)
# the d orbitals as quadratic forms r^T M r, all of Frobenius norm^2 = 3/2
_M = {'dxy': np.array([[0., _S3/2, 0.], [_S3/2, 0., 0.], [0., 0., 0.]]),
         'dyz': np.array([[0., 0., 0.], [0., 0., _S3/2], [0., _S3/2, 0.]]),
         'dzx': np.array([[0., 0., _S3/2], [0., 0., 0.], [_S3/2, 0., 0.]]),
         'dx2-y2': np.diag([_S3/2, -_S3/2, 0.]),
         'dz2': np.diag([-0.5, -0.5, 1.])}


def _rotation(d_hat: NDArray[np.float64]) -> NDArray[np.float64]:
    '''
    Private function. A rotation matrix taking z onto d_hat.
    '''
    z = np.array([0., 0., 1.])
    c = float(np.dot(z, d_hat))
    if c > 1. - 1e-14:
        return np.eye(3)
    if c < -1. + 1e-14:
        return np.diag([1., -1., -1.])  # a half turn about x
    axis = np.cross(z, d_hat)
    s = np.linalg.norm(axis)
    k = axis / s
    kx = np.array([[0., -k[2], k[1]], [k[2], 0., -k[0]], [-k[1], k[0], 0.]])
    return np.eye(3) + s * kx + (1. - c) * kx @ kx


def _rep(l: int, rot: NDArray[np.float64]) -> NDArray[np.float64]:
    '''
    Private function. Real representation D_l(R) of a rotation on the real
    orbitals of angular momentum l (columns: rotated orbitals).
    '''
    if l == 0:
        return np.ones((1, 1))
    if l == 1:
        return rot
    return np.array([[np.sum(_M[a] * (rot @ _M[b] @ rot.T)) / 1.5 for b in _D] for a in _D])


def _bond_frame(l1: int, l2: int, params: dict) -> NDArray[np.float64]:
    '''
    Private function. Matrix between the orbitals of angular momentum l1 (first
    atom) and l2 (second atom, at +z), in the bond frame.
    '''
    def get(key, fallback=None):
        return params.get(key, params.get(fallback, 0.) if fallback else 0.)

    if (l1, l2) == (0, 0):
        return np.array([[get('ss_sigma')]])
    if (l1, l2) == (0, 1):
        return np.array([[0., 0., get('sp_sigma')]])
    if (l1, l2) == (1, 0):
        return np.array([[0.], [0.], [-get('ps_sigma', 'sp_sigma')]])
    if (l1, l2) == (1, 1):
        return np.diag([get('pp_pi'), get('pp_pi'), get('pp_sigma')])
    if (l1, l2) == (0, 2):
        return np.array([[0., 0., 0., 0., get('sd_sigma')]])
    if (l1, l2) == (2, 0):
        return np.array([[0.], [0.], [0.], [0.], [get('ds_sigma', 'sd_sigma')]])
    if (l1, l2) == (1, 2):
        # (px, dzx), (py, dyz): pi; (pz, dz2): sigma
        out = np.zeros((3, 5))
        out[0, 2] = out[1, 1] = get('pd_pi')
        out[2, 4] = get('pd_sigma')
        return out
    if (l1, l2) == (2, 1):
        out = np.zeros((5, 3))
        out[2, 0] = out[1, 1] = -get('dp_pi', 'pd_pi')
        out[4, 2] = -get('dp_sigma', 'pd_sigma')
        return out
    return np.diag([get('dd_delta'), get('dd_pi'), get('dd_pi'), get('dd_delta'), get('dd_sigma')])


def sk_block(orbs1: list[str], orbs2: list[str], d: ArrayLike, params: dict) -> NDArray[np.float64]:
    r'''
    Get the Slater-Koster hopping matrix between the orbitals *orbs1* of an
    atom and the orbitals *orbs2* of a neighbour at displacement :math:`\mathbf{d}`.

    :param orbs1: List of orbital names (see :data:`ORBITALS`).
    :param orbs2: List of orbital names.
    :param d: Tuple of 2 or 3 reals. Displacement from the first atom to the
        second (a 2D vector lies in the (x, y) plane).
    :param params: Dictionary of bond integrals (see the module docstring).

    :returns:
        * **block** -- Real ndarray, shape (len(orbs1), len(orbs2)).
    '''
    error_handling.orbitals(orbs1)
    error_handling.orbitals(orbs2)
    error_handling.sk_params(params)
    d = np.asarray(d, dtype='f8')
    error_handling.bond_vector(d)
    d = np.concatenate([d, np.zeros(3 - len(d))])
    rot = _rotation(d / np.linalg.norm(d))
    names = {0: ['s'], 1: _P, 2: _D}
    block = np.zeros((len(orbs1), len(orbs2)))
    for l1 in {ORBITALS[o] for o in orbs1}:
        for l2 in {ORBITALS[o] for o in orbs2}:
            full = _rep(l1, rot) @ _bond_frame(l1, l2, params) @ _rep(l2, rot).T
            rows = [(i, names[l1].index(o)) for i, o in enumerate(orbs1) if ORBITALS[o] == l1]
            cols = [(j, names[l2].index(o)) for j, o in enumerate(orbs2) if ORBITALS[o] == l2]
            for i, a in rows:
                for j, b in cols:
                    block[i, j] = full[a, b]
    return block


def orbital_angular_momentum() -> list[NDArray[np.complex128]]:
    r'''
    Get the orbital angular momentum :math:`(L_x, L_y, L_z)` (:math:`\hbar = 1`)
    on the real p orbitals ``(px, py, pz)``: :math:`(L_k)_{ij} = -i\epsilon_{kij}`.

    :returns:
        * **L** -- List of three complex ndarrays, shape (3, 3).
    '''
    eps = np.zeros((3, 3, 3))
    eps[0, 1, 2] = eps[1, 2, 0] = eps[2, 0, 1] = 1.
    eps[0, 2, 1] = eps[2, 1, 0] = eps[1, 0, 2] = -1.
    return [-1j * eps[k] for k in range(3)]


def sk_kspace(lat, orbitals: dict[str, list[str]], bonds: dict[int, dict],
                   onsite: dict | None = None, overlap: dict[int, dict] | None = None):
    r'''
    Get the Slater-Koster Bloch Hamiltonian of a periodic lattice: a
    **KSpace** with one orbital per (site, orbital) of the unit cell, and
    the hoppings of *sk_block* between the *n*-th neighbours listed in
    *bonds* -- the reciprocal-space twin of
    :meth:`tbkit.orbital.OrbitalSystem.set_slater_koster`.

    :param lat: **Lattice** instance (unit cell and primitive vectors).
    :param orbitals: Dictionary {tag: list of orbital names}.
    :param bonds: Dictionary {n: bond integrals}: the neighbour orders to
        connect (1 for nearest neighbours), each with its Slater-Koster
        parameters (see the module docstring).
    :param onsite: Dictionary. Default value None. {tag: energy} (every
        orbital of the site) or {tag: {orbital: energy}}.
    :param overlap: Dictionary. Default value None. {n: bond integrals} of
        the overlap matrix (see *KSpace.set_overlap*).

    :returns:
        * **ks** -- **KSpace** instance; *ks.sk_orbitals* lists the (site,
          orbital) of each of its orbitals.
    '''
    from tbkit.kspace import KSpace
    from tbkit.lattice import Lattice
    error_handling.lat(lat)
    error_handling.orbital_dict(orbitals, lat.tags)
    error_handling.sk_bonds(bonds)
    if overlap is not None:
        error_handling.sk_bonds(overlap)
    cell = lat.unit_cell
    table = [(i, o) for i, dic in enumerate(cell) for o in orbitals[dic['tag']]]
    index = {key: n for n, key in enumerate(table)}
    new_cell = [{'tag': cell[i]['tag'], 'r0': cell[i]['r0']} for i, _ in table]
    ks = KSpace(Lattice(unit_cell=new_cell, prim_vec=lat.prim_vec))
    ks.sk_orbitals = table
    if onsite is not None:
        error_handling.set_onsite_orb(onsite, lat.tags, orbitals)
        for n, (i, o) in enumerate(table):
            val = onsite.get(cell[i]['tag'], 0.)
            ks.onsite[n] = val.get(o, 0.) if isinstance(val, dict) else val
    # candidate bonds: every site of the home cell to every site of the
    # neighbouring cells, each unordered bond once
    a = np.array(lat.prim_vec, dtype='f8')
    r0 = np.array([dic['r0'] for dic in cell], dtype='f8')
    shifts = np.array(np.meshgrid(*[range(-3, 4)] * len(a), indexing='ij')).reshape(len(a), -1).T
    cand = []
    for R in shifts:
        for i in range(len(cell)):
            for j in range(len(cell)):
                d = r0[j] + R @ a - r0[i]
                dist = np.linalg.norm(d)
                if dist < 1e-9:
                    continue
                if (i, j) > (j, i) or (i == j and tuple(-R) > tuple(R)):
                    continue
                cand.append((round(dist, 4), i, j, tuple(int(c) for c in R), d))
    distances = sorted({c[0] for c in cand})
    for store, spec in ((ks.set_hopping, bonds), (ks.set_overlap, overlap or {})):
        hops = []
        for n, params in spec.items():
            error_handling.positive_int_lim(n, 'n', len(distances))
            for dist, i, j, R, d in cand:
                if dist != distances[n - 1]:
                    continue
                orbs_i, orbs_j = orbitals[cell[i]['tag']], orbitals[cell[j]['tag']]
                block = sk_block(orbs_i, orbs_j, d, params)
                for p, oi in enumerate(orbs_i):
                    for q, oj in enumerate(orbs_j):
                        if block[p, q]:
                            hops.append({'i': index[(i, oi)], 'j': index[(j, oj)], 'R': R, 't': float(block[p, q])})
        if hops:
            store(hops)
    return ks
