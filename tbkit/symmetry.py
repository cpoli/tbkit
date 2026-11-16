# Copyright 2014 Charles Poli.
#
# This file is part of TBKIT.  It is subject to the license terms in the
# LICENSE file found in the top-level directory of this distribution and at
# https://github.com/cpoli/tbkit.

r'''
Crystal symmetry of a Bloch Hamiltonian (:class:`tbkit.kspace.KSpace`).

A space-group operation :math:`g = \{R|\boldsymbol\tau\}` (a rotation or
reflection :math:`R`, a translation :math:`\boldsymbol\tau`, a matrix
:math:`D` on the orbitals of a site, possibly times complex conjugation) is
an :class:`Operation`. It sends the orbital :math:`s` of the home cell to
the orbital :math:`\pi(s)` of the cell :math:`\mathbf{L}_s`,
:math:`R\boldsymbol\tau_s + \boldsymbol\tau = \boldsymbol\tau_{\pi(s)} +
\mathbf{L}_s`, so that in the periodic gauge of *KSpace*

.. math::

    U_g(\mathbf{k})_{\pi(s), s} = D\, e^{-i g\mathbf{k}\cdot\mathbf{L}_s},\qquad
    H(g\mathbf{k}) = U_g(\mathbf{k})\, H(\mathbf{k})\, U_g(\mathbf{k})^\dagger

(with :math:`H^*(\mathbf{k})` and :math:`g\mathbf{k} = -R\mathbf{k}` for an
antiunitary operation). On this representation are built:

* :func:`find_symmetries`: the operations of the Bravais lattice that are
  symmetries of the model, as ``qsymm.symmetries`` does;
* :func:`irreps`: the irreducible representations of the bands at a
  k-point (Bouckaert, Smoluchowski and Wigner, Phys. Rev. 50, 58 (1936)),
  from the character tables of the ten 2D point groups and of the five
  cubic ones; *KSpace.plot_bands* writes them on the bands;
* the symmetry indicators :func:`rotation_indicator` (the Chern number
  modulo :math:`n` from the :math:`C_n` eigenvalues, Fang, Gilbert and
  Bernevig, Phys. Rev. B 86, 115112 (2012)), :func:`inversion_indicators`
  (Khalaf, Po, Vishwanath and Watanabe, Phys. Rev. X 8, 031070 (2018)) and
  :func:`corner_charge` (of a :math:`C_4`-symmetric insulator, Benalcazar,
  Li and Hughes, Phys. Rev. B 99, 245151 (2019));
* :func:`symmetrize`: the group average of the hoppings, as
  ``TBmodels.Model.symmetrize``.
'''

from __future__ import annotations

import itertools
from typing import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.spatial.transform import Rotation

import tbkit.error_handling as error_handling
from tbkit.kspace import KSpace, PAULI

PI = np.pi
TOL = 1e-6  # positions, rotation matrices and fractional coordinates
_I_SIGMA_Y = 1j * PAULI['y']  # the spin part of time reversal, T = i sigma_y K


class Operation():
    r'''
    A space-group operation :math:`g = \{R|\boldsymbol\tau\}`:
    :math:`\mathbf{r} \to R\mathbf{r} + \boldsymbol\tau`, possibly
    antiunitary (times complex conjugation, e.g. time reversal).

    Parameters
    ----------
    rotation : Real ndarray, shape (space_dim, space_dim)
        Orthogonal matrix :math:`R`
        (a rotation, a reflection or the inversion), in the Cartesian frame
        of *prim_vec*. See :func:`rotation` and :func:`mirror`.
    translation : Real ndarray, shape (space_dim,), default None (zero)
        :math:`\boldsymbol\tau`. A rotation about a point
        :math:`\mathbf{c}` has :math:`\boldsymbol\tau = \mathbf{c} - R\mathbf{c}`
        (see :meth:`about`). *irreps* and the indicators need it so, not
        reduced modulo a lattice vector: :math:`\{R|\boldsymbol\tau +
        \mathbf{L}\}` is another element of the space group, whose
        characters differ by :math:`e^{-i\mathbf{k}\cdot\mathbf{L}}`.
    orbitals : Complex ndarray, default None
        Unitary matrix :math:`D` on the
        orbitals at one position of the unit cell (the entries of
        *unit_cell* that share a position, in their order, each twice,
        site-major, if ``spin=True``): orbital :math:`a` of a position goes
        to :math:`\sum_b D_{ba}\,b` at the image position. None: the
        identity on the orbitals and, if ``spin=True``, the spin-1/2
        rotation of :math:`\det(R)R` (times :math:`i\sigma_y` if
        *antiunitary*, so that the antiunitary identity is time reversal).
        Given, it must include the spin part.
    antiunitary : Boolean, default False
    name : String, default None (from the rotation)

    Examples
    --------
    Example usage::

        c3 = Operation.about(rotation(2 * np.pi / 3), center)   # C3 about a hexagon centre
        mx = Operation(mirror((1., 0.)))                         # x -> -x
        time_reversal = Operation(np.eye(2), antiunitary=True)
        ks.symmetry_error(c3)
    '''

    def __init__(
        self, rotation: ArrayLike, translation: ArrayLike | None = None, orbitals: ArrayLike | None = None,
        antiunitary: bool = False, name: str | None = None,
    ) -> None:
        error_handling.operation(rotation, translation, orbitals, antiunitary, name)
        self.rotation = np.array(rotation, dtype='f8')
        d = len(self.rotation)
        self.translation = np.zeros(d) if translation is None else np.array(translation, dtype='f8')
        self.orbitals = None if orbitals is None else np.array(orbitals, dtype='c16')
        self.antiunitary = antiunitary
        self.name = _name(self.rotation, antiunitary) if name is None else name

    @classmethod
    def about(
        cls, rotation: ArrayLike, center: ArrayLike, orbitals: ArrayLike | None = None,
        antiunitary: bool = False, name: str | None = None,
    ) -> Operation:
        r'''
        The operation :math:`R` about the point *center* :math:`\mathbf{c}`:
        :math:`\boldsymbol\tau = \mathbf{c} - R\mathbf{c}`. Other
        parameters: see :class:`Operation`.
        '''
        error_handling.operation(rotation, None, orbitals, antiunitary, name)
        rot = np.array(rotation, dtype='f8')
        c = np.array(center, dtype='f8')
        error_handling.ndarray(c, 'center', len(rot))
        return cls(rot, c - rot @ c, orbitals, antiunitary, name)

    def __matmul__(self, other: Operation) -> Operation:
        r'''
        The product :math:`g_1 g_2` (*other* first): :math:`\{R_1R_2|R_1\boldsymbol\tau_2 +
        \boldsymbol\tau_1\}`, :math:`D_1 D_2` (:math:`D_1 D_2^*` if :math:`g_1`
        is antiunitary). With ``orbitals=None`` and spin, the spin part of the
        product is fixed up to a sign (the double group is not followed).
        '''
        error_handling.compose(self, other)
        orb = None
        if self.orbitals is not None and other.orbitals is not None:
            orb = self.orbitals @ (other.orbitals.conj() if self.antiunitary else other.orbitals)
        return Operation(self.rotation @ other.rotation, self.rotation @ other.translation + self.translation,
                                 orb, self.antiunitary != other.antiunitary)

    def __pow__(self, n: int) -> Operation:
        '''
        :math:`g^n`, *n* a positive integer.
        '''
        error_handling.positive_int(n, 'n')
        out = self
        for _ in range(n - 1):
            out = self @ out
        return out

    def __repr__(self) -> str:
        return 'Operation({}, translation={})'.format(self.name, np.round(self.translation, 6).tolist())


def rotation(angle: float, axis: ArrayLike | None = None) -> NDArray[np.float64]:
    r'''
    The matrix of a counterclockwise rotation by *angle* (radians): 2x2 in
    the plane, or 3x3 about *axis*.

    Parameters
    ----------
    angle : Real number
    axis : Real ndarray of 3 numbers, default None (a 2D rotation)

    Returns
    -------
    R : Real ndarray, shape (2, 2) or (3, 3)
    '''
    error_handling.real_number(angle, 'angle')
    if axis is None:
        c, s = np.cos(angle), np.sin(angle)
        return np.array([[c, -s], [s, c]])
    n = np.array(axis, dtype='f8')
    error_handling.ndarray(n, 'axis', 3)
    error_handling.nonzero_vector(n, 'axis')
    return Rotation.from_rotvec(angle * n / np.linalg.norm(n)).as_matrix()


def mirror(normal: ArrayLike) -> NDArray[np.float64]:
    r'''
    The matrix :math:`1 - 2\hat{\mathbf{n}}\hat{\mathbf{n}}^T` of the
    reflection through the line (2D) or plane (3D) normal to *normal*.

    Parameters
    ----------
    normal : Real ndarray of 2 or 3 numbers

    Returns
    -------
    R : Real ndarray, shape (len(normal), len(normal))
    '''
    n = np.array(normal, dtype='f8')
    error_handling.nonzero_vector(n, 'normal')
    n = n / np.linalg.norm(n)
    return np.eye(len(n)) - 2 * np.outer(n, n)


def _proper(rot: NDArray[np.float64]) -> NDArray[np.float64]:
    '''
    Private function. The proper rotation det(R) R, embedded in 3D (a 2D
    operation acts on the plane z = 0 and leaves z unchanged: a 2D mirror
    is a vertical mirror plane).
    '''
    r3 = np.eye(3)
    d = len(rot)
    r3[:d, :d] = rot
    return np.linalg.det(r3) * r3


def _su2(rot: NDArray[np.float64]) -> NDArray[np.complex128]:
    r'''
    Private function. The spin-1/2 matrix :math:`e^{-i\theta\hat{\mathbf{n}}\cdot\boldsymbol\sigma/2}`
    of the proper part of *rot* (one of its two signs).
    '''
    x, y, z, w = Rotation.from_matrix(_proper(rot)).as_quat()
    return w * PAULI['0'] - 1j * (x * PAULI['x'] + y * PAULI['y'] + z * PAULI['z'])


def _angle_axis(rot: NDArray[np.float64]) -> tuple[float, NDArray[np.float64]]:
    '''
    Private function. Angle in [0, pi] and unit axis of the proper part of *rot*.
    '''
    vec = Rotation.from_matrix(_proper(rot)).as_rotvec()
    angle = float(np.linalg.norm(vec))
    if angle < TOL:
        return 0., np.array([0., 0., 1.])
    return angle, vec / angle


def _name(rot: NDArray[np.float64], antiunitary: bool) -> str:
    '''
    Private function. A name for an operation: 'E', 'C3', 'C3^-1', 'M(30)'
    (a mirror line at 30 degrees), 'I', 'S4[0 0 1]', ... and ' T' if antiunitary.
    '''
    det = np.linalg.det(rot)
    angle, axis = _angle_axis(rot)
    if len(rot) == 2:
        theta = float(np.arctan2(rot[1, 0], rot[0, 0]))
        if det < 0:
            name = 'M({:g})'.format(round(np.degrees(theta / 2) % 180., 3))
        elif abs(theta) < TOL:
            name = 'E'
        else:
            n = int(round(2 * PI / abs(theta)))
            name = 'C{}'.format(n) + ('^-1' if theta < 0 and n > 2 else '')
    else:
        big = axis[np.argmax(np.abs(axis))]
        ax = '[{}]'.format(' '.join('{:g}'.format(round(c / big, 3) + 0.) for c in axis))
        n = int(round(2 * PI / angle)) if angle > TOL else 1
        if det > 0:
            name = 'E' if n == 1 else 'C{}{}'.format(n, ax)
        elif n == 1:
            name = 'I'
        elif n == 2:
            name = 'M' + ax
        else:  # -R = C_n: S_2n for odd n, S_(n/2) for n = 2 mod 4, S_n otherwise
            name = 'S{}{}'.format(2 * n if n % 2 else (n // 2 if n % 4 == 2 else n), ax)
    return name + (' T' if antiunitary else '')


class _Action():
    '''
    Private class. How an operation acts on the orbitals of a KSpace: the
    position blocks (orbital indices sharing a position), the permutation
    of the blocks, the lattice vectors L_s (integers and Cartesian), and the
    matrix D of each block.
    '''

    def __init__(self, ks: KSpace, op: Operation) -> None:
        error_handling.is_operation(op)
        error_handling.operation_space(op, ks.space_dim)
        tau = np.array([dic['r0'] for dic in ks.lat.unit_cell], dtype='f8').reshape(ks.n_sites, -1)
        keys: list = []
        entries: list[list[int]] = []
        for e, r in enumerate(tau):
            for b, p in enumerate(keys):
                if np.allclose(r, p, atol=TOL):
                    entries[b].append(e)
                    break
            else:
                keys.append(r)
                entries.append([e])
        pos = np.array(keys)
        a = np.array(ks.lat.prim_vec, dtype='f8')
        diff = (pos @ op.rotation.T + op.translation)[:, None, :] - pos[None, :, :]
        frac = diff @ np.linalg.pinv(a)
        ints = np.round(frac)
        ok = np.all(np.abs(frac - ints) < TOL, axis=2) & np.all(np.abs(ints @ a - diff) < TOL, axis=2)
        error_handling.maps_lattice(ok)
        self.perm = np.argmax(ok, axis=1)
        self.L = ints[np.arange(len(pos)), self.perm].astype(int)
        self.L_cart = self.L @ a
        self.positions = pos
        self.idx = [np.array([2 * e + s for e in ent for s in range(2)] if ks.spin else ent) for ent in entries]
        self.tags = [tuple(ks.tags[ent]) for ent in entries]
        sizes = [len(i) for i in self.idx]
        error_handling.block_sizes(sizes, self.perm, None if op.orbitals is None else len(op.orbitals))
        if op.orbitals is not None:
            self.D = [op.orbitals] * len(pos)
        elif ks.spin:
            s = _su2(op.rotation) @ (_I_SIGMA_Y if op.antiunitary else PAULI['0'])
            self.D = [np.kron(np.eye(n // 2), s) for n in sizes]
        else:
            self.D = [np.eye(n) for n in sizes]


def _representations(
    ks: KSpace, op: Operation, k: NDArray[np.float64],
) -> tuple[NDArray[np.complex128], NDArray[np.float64]]:
    r'''
    Private function. The matrices :math:`U_g(\mathbf{k})` (shape (nk, norb,
    norb)) at the k-points *k* (shape (nk, dim), coordinates of *get_ham*),
    and the images :math:`g\mathbf{k}` (same coordinates).
    '''
    act = _Action(ks, op)
    k_cart = np.atleast_2d(k) @ ks.k_basis.T
    gk = (-1. if op.antiunitary else 1.) * k_cart @ op.rotation.T
    phase = np.exp(-1j * gk @ act.L_cart.T)
    u = np.zeros((len(k_cart), ks.norb, ks.norb), dtype='c16')
    for b, target in enumerate(act.perm):
        u[:, act.idx[target][:, None], act.idx[b][None, :]] = phase[:, b, None, None] * act.D[b][None]
    return u, gk @ ks.k_basis


def representation(ks: KSpace, op: Operation, k: ArrayLike) -> NDArray[np.complex128]:
    r'''
    Get the matrix :math:`U_g(\mathbf{k})` of an operation on the Bloch
    basis of *ks* (periodic gauge), which sends :math:`\mathbf{k}` to
    :math:`g\mathbf{k}` (see the module docstring).

    Parameters
    ----------
    ks : **KSpace** instance
    op : :class:`Operation`
    k : k-point (coordinates of *KSpace.get_ham*)

    Returns
    -------
    U : Complex ndarray, shape (norb, norb)
    '''
    error_handling.k_vector(k, 'k', ks.dim)
    return _representations(ks, op, np.array(k, dtype='f8')[None])[0][0]


def _lattice_rotations(a: NDArray[np.float64]) -> list[NDArray[np.float64]]:
    '''
    Private function. The orthogonal matrices that map the lattice of the
    rows of *a* (dim = space_dim) onto itself: the point group of the
    Bravais lattice.
    '''
    d = len(a)
    combos = np.array(list(itertools.product(range(-2, 3), repeat=d)))
    vecs = combos @ a
    norms = np.linalg.norm(vecs, axis=1)
    cands = [vecs[np.abs(norms - np.linalg.norm(v)) < TOL] for v in a]
    inv = np.linalg.inv(a.T)
    out: list[NDArray[np.float64]] = []
    for combo in itertools.product(*cands):
        rot = np.array(combo).T @ inv
        if np.allclose(rot @ rot.T, np.eye(d), atol=TOL) and not any(np.allclose(rot, r, atol=TOL) for r in out):
            out.append(rot)
    return out


def _structure_translations(act_data: tuple, rot: NDArray[np.float64]) -> list[NDArray[np.float64]]:
    '''
    Private function. The translations t (one per class modulo the lattice)
    for which {rot|t} maps the positions of the unit cell onto positions
    with as many orbitals (the tags are not compared: a sublattice mass is
    found broken by the Hamiltonian, not by its tags). *act_data* =
    (positions, orbitals per position, lattice rows).
    '''
    pos, tags, a = act_data
    pinv = np.linalg.pinv(a)
    out: list[NDArray[np.float64]] = []
    for target in range(len(pos)):
        t = pos[target] - rot @ pos[0]
        f = t @ pinv
        t = (f - np.floor(f + TOL)) @ a + (t - f @ a)  # reduced into the cell (component off the lattice kept)
        diff = (pos @ rot.T + t)[:, None, :] - pos[None, :, :]
        frac = diff @ pinv
        ok = np.all(np.abs(frac - np.round(frac)) < TOL, axis=2) & np.all(
            np.abs(np.round(frac) @ a - diff) < TOL, axis=2)
        match = [np.flatnonzero(row) for row in ok]
        if all(len(m) == 1 and len(tags[int(m[0])]) == len(tags[s]) for s, m in enumerate(match)) and not any(
                _lattice_vector(t - u, a) for u in out):
            out.append(t)
    return out


def _lattice_vector(v: NDArray[np.float64], a: NDArray[np.float64]) -> bool:
    '''
    Private function. Whether *v* is a lattice vector of the rows of *a*.
    '''
    f = v @ np.linalg.pinv(a)
    return bool(np.all(np.abs(f - np.round(f)) < TOL) and np.allclose(np.round(f) @ a, v, atol=TOL))


def _origin(cosets: list, a: NDArray[np.float64]) -> NDArray[np.float64] | None:
    r'''
    Private function. A point :math:`\mathbf{c}` about which every rotation
    of *cosets* ((R, t) pairs) is a symmetry of the structure, :math:`\{R|\mathbf{c} -
    R\mathbf{c}\}` in the class of one of its t: the group is symmorphic
    with origin :math:`\mathbf{c}`. Tried on the points of fractional
    coordinates n/12 (every Wyckoff position of highest symmetry), the
    closest to the origin first. None if there is none (a nonsymmorphic group).
    '''
    d = len(a)
    grid = np.array(list(itertools.product(np.arange(12) / 12, repeat=d))) @ a
    grid = grid[np.argsort(np.linalg.norm(grid, axis=1), kind='stable')]
    good = np.ones(len(grid), bool)
    rots: dict = {}
    for rot, t in cosets:
        rots.setdefault(_key(rot), []).append((rot, t))
    for group in rots.values():
        hit = np.zeros(len(grid), bool)
        for rot, t in group:
            diff = grid - grid @ rot.T - t
            f = diff @ np.linalg.inv(a)
            hit |= np.all(np.abs(f - np.round(f)) < TOL, axis=1)
        good &= hit
    return grid[np.argmax(good)] if good.any() else None


def _key(rot: NDArray[np.float64]) -> tuple:
    '''
    Private function. A hashable key of a rotation matrix.
    '''
    return tuple(np.round(rot, 6).ravel() + 0.)


def find_symmetries(
    ks: KSpace, orbitals: Callable | None = None, antiunitary: bool = True, nk: int = 5, tol: float = 1e-8,
) -> list[Operation]:
    r'''
    Find the space-group operations of a model: the operations of its
    Bravais lattice, combined with the translations that map every
    position of the unit cell onto one with as many orbitals, that leave :math:`H(\mathbf{k})`
    invariant (*KSpace.symmetry_error* below *tol*), as
    ``qsymm.symmetries`` does. For a symmorphic structure, every operation
    is a rotation about the same point (a site, a bond centre, the centre of
    a hexagon...), the one closest to the origin: the form *irreps* needs.
    Lattice translations are not listed: the result is one operation per
    element of the point group (times time reversal).

    Parameters
    ----------
    ks : **KSpace** instance, whose lattice fills its space (dim = space_dim)
    orbitals : Callable, default None
        ``orbitals(R)`` returns the matrix
        on the orbitals at one position (without spin) for the rotation
        :math:`R`, e.g. ``lambda R: scipy.linalg.block_diag(1., R)`` for an s
        and two p orbitals at each site; with ``spin=True``, the
        :math:`D` of :class:`Operation` is its Kronecker product with the
        spin-1/2 rotation. None: the identity on the orbitals (and the
        spin-1/2 rotation if ``spin=True``).
    antiunitary : Boolean, default True
        Also look for antiunitary
        operations (time reversal, alone or times a rotation): with spin,
        :math:`D\,i\sigma_y` on every site.
    nk : Positive integer, default 5
        Points per direction of the test mesh of *symmetry_error*.
    tol : Positive real, default 1e-8

    Returns
    -------
    ops : List of :class:`Operation`
        The unitary ones first, each group
        in the order of the lattice operations; the identity first.

    Examples
    --------
    Example usage::

        ops = find_symmetries(graphene)          # 24: C6v, and each times time reversal
        [op.name for op in ops if not op.antiunitary]
    '''
    error_handling.full_dim(ks.dim, ks.space_dim)
    error_handling.callable_or_none(orbitals, 'orbitals')
    error_handling.boolean(antiunitary, 'antiunitary')
    error_handling.positive_int(nk, 'nk')
    error_handling.positive_real(tol, 'tol')
    a = np.array(ks.lat.prim_vec, dtype='f8')
    act = _Action(ks, Operation(np.eye(ks.space_dim)))
    rots = _lattice_rotations(a)
    rots.sort(key=lambda r: (-round(np.linalg.det(r)), -np.trace(r)))
    data = (act.positions, act.tags, a)
    cosets = [(rot, t) for rot in rots for t in _structure_translations(data, rot)]
    origin = _origin(cosets, a)
    unitary, anti = [], []
    for rot, t in cosets:
        if origin is not None:
            t0 = origin - rot @ origin
            t = t0 if _lattice_vector(t - t0, a) else t
        orb = None if orbitals is None else np.asarray(orbitals(rot), dtype='c16')
        if orb is not None and ks.spin:
            orb = np.kron(orb, _su2(rot))
        op = Operation(rot, t, orb)
        if ks.symmetry_error(op, nk=nk) < tol:
            unitary.append(op)
        if antiunitary:
            if orb is not None and ks.spin:
                orb = orb @ np.kron(np.eye(len(orb) // 2), _I_SIGMA_Y)
            op = Operation(rot, t, orb, antiunitary=True)
            if ks.symmetry_error(op, nk=nk) < tol:
                anti.append(op)
    return unitary + anti


def _invariant(ks: KSpace, op: Operation, k: NDArray[np.float64]) -> bool:
    '''
    Private function. Whether the operation leaves the k-point invariant
    (modulo a reciprocal lattice vector).
    '''
    gk = _representations(ks, op, k[None])[1][0]
    vecs = np.array(ks.lat.prim_vec, dtype='f8') @ ks.k_basis
    f = (gk - k) @ vecs.T / (2 * PI)
    return bool(np.all(np.abs(f - np.round(f)) < TOL))


# Character tables. Each irrep is a function of an element's descriptor.
# 2D: ('rot', theta) with theta the counterclockwise angle, or ('mir', s)
# with s = +1 for the class sigma_v (the mirror lines closest to a_1) and
# -1 for sigma_d.
def _table_2d(n: int, mirrors: bool) -> list[tuple[str, Callable]]:
    '''
    Private function. Irreps of the 2D point groups C_n and C_nv.
    '''
    def rot(f):
        return lambda d: f(d[1]) if d[0] == 'rot' else None

    if not mirrors:
        table: list[tuple[str, Callable]] = [('A', rot(lambda t: 1.))]
        if n % 2 == 0:
            table.append(('B', rot(lambda t: np.exp(1j * n * t / 2))))
        orders = [m for m in range(1, (n - 1) // 2 + 1)]
        for m in orders:
            suffix = str(m) if len(orders) > 1 else ''
            table.append(('¹E' + suffix, rot(lambda t, m=m: np.exp(1j * m * t))))
            table.append(('²E' + suffix, rot(lambda t, m=m: np.exp(-1j * m * t))))
        return table
    if n == 1:
        return [("A'", lambda d: 1.), ("A''", lambda d: 1. if d[0] == 'rot' else -1.)]

    def irrep(fr, fm):
        return lambda d: fr(d[1]) if d[0] == 'rot' else fm(d[1])

    table = [('A1', irrep(lambda t: 1., lambda s: 1.)), ('A2', irrep(lambda t: 1., lambda s: -1.))]
    if n % 2 == 0:
        table.append(('B1', irrep(lambda t: np.cos(n * t / 2), lambda s: s)))
        table.append(('B2', irrep(lambda t: np.cos(n * t / 2), lambda s: -s)))
    orders = [m for m in range(1, (n - 1) // 2 + 1)]
    for m in orders:
        table.append(('E' + (str(m) if len(orders) > 1 else ''),
                          irrep(lambda t, m=m: 2 * np.cos(m * t), lambda s: 0.)))
    return table


# Cubic groups, by the class of the proper part P = det(R) R in O:
# E, C3 (8), C2' (6, about face diagonals), C4 (6), C2 (3, about the cube axes)
_O_TABLE = {'A1': (1, 1, 1, 1, 1), 'A2': (1, 1, -1, -1, 1), 'E': (2, -1, 0, 0, 2),
                  'T1': (3, 0, -1, 1, -1), 'T2': (3, 0, 1, -1, -1)}
_O_CLASSES = ('E', 'C3', "C2'", 'C4', 'C2')
_W = np.exp(2j * PI / 3)
# T: E, C3 (+120 degrees about [111], [1-1-1], ...), C3^2, C2
_T_TABLE = {'A': (1, 1, 1, 1), '¹E': (1, _W, _W.conjugate(), 1), '²E': (1, _W.conjugate(), _W, 1),
                  'T': (3, 0, 0, -1)}
_T_CLASSES = ('E', 'C3', 'C3^2', 'C2')


def _cubic_class(rot: NDArray[np.float64]) -> str | None:
    '''
    Private function. The class (in O, or T for 'C3'/'C3^2') of the proper
    part of a cubic operation, None if its axis is not a cube axis, face
    diagonal or body diagonal of the Cartesian frame.
    '''
    angle, axis = _angle_axis(rot)
    if angle < TOL:
        return 'E'
    mags = np.sort(np.round(np.abs(axis) * np.sqrt(3), 6))
    order = int(round(2 * PI / angle))
    if order == 3 and np.allclose(mags, 1.):
        return 'C3' if np.prod(np.sign(axis)) > 0 else 'C3^2'
    if order in (2, 4) and np.allclose(mags, [0., 0., np.sqrt(3)]):
        return 'C4' if order == 4 else 'C2'
    if order == 2 and np.allclose(mags, [0., np.sqrt(1.5), np.sqrt(1.5)]):
        return "C2'"
    return None


def _point_group(rots: list[NDArray[np.float64]], a1: NDArray[np.float64]):
    '''
    Private function. The name of the point group of the matrices *rots*
    and its irreps, as (name, [(label, characters over rots)]), or None if
    it is not tabulated (3D groups other than the cubic ones).
    '''
    if len(rots[0]) == 2:
        dets = [np.linalg.det(r) for r in rots]
        n = sum(d > 0 for d in dets)
        mirrors = n < len(rots)
        lines = [float(np.arctan2(r[1, 0], r[0, 0]) / 2) % PI for r, d in zip(rots, dets) if d < 0]
        alpha = float(np.arctan2(a1[1], a1[0])) % PI
        dist = [min(abs(phi - alpha) % PI, PI - abs(phi - alpha) % PI) for phi in lines]
        phi0 = lines[int(np.argmin(dist))] if mirrors else 0.
        desc = []
        for r, d in zip(rots, dets):
            if d > 0:
                desc.append(('rot', float(np.arctan2(r[1, 0], r[0, 0]))))
            else:
                phi = float(np.arctan2(r[1, 0], r[0, 0]) / 2)
                desc.append(('mir', 1. if int(round((phi - phi0) * n / PI)) % 2 == 0 else -1.))
        name = ('C{}v'.format(n) if n > 1 else 'Cs') if mirrors else 'C{}'.format(n)
        return name, [(label, np.array([f(x) for x in desc], dtype='c16'))
                             for label, f in _table_2d(n, mirrors)]
    found = [_cubic_class(r) for r in rots]
    det3 = np.array([np.linalg.det(r) for r in rots])
    if None in found or sum(c in ('C3', 'C3^2') for c, d in zip(found, det3) if d > 0) != 8:
        return None
    classes = [str(c) for c in found]
    parity = np.where(det3 > 0, 1., -1.)
    proper = bool(np.all(det3 > 0))
    has_inv = any(np.allclose(r, -np.eye(3), atol=TOL) for r in rots)
    out = []
    if not any(c in ('C4', "C2'") for c in classes):
        name = 'T' if proper else 'Th'
        idx = [_T_CLASSES.index(c) for c in classes]
        for label, row in _T_TABLE.items():
            chi = np.array([row[i] for i in idx], dtype='c16')
            if proper:
                out.append((label, chi))
            else:
                out += [(label + 'g', chi), (label + 'u', chi * parity)]
        return name, out
    name = 'O' if proper else ('Oh' if has_inv else 'Td')
    idx = [_O_CLASSES.index(c.replace('^2', '')) for c in classes]  # C3 and C3^2 are one class in O
    for label, row_o in _O_TABLE.items():
        chi = np.array([row_o[i] for i in idx], dtype='c16')
        if name == 'Oh':
            out += [(label + 'g', chi), (label + 'u', chi * parity)]
        else:
            out.append((label, chi))
    return name, out


class Irreps():
    r'''
    The irreducible representations of the bands at a k-point, see *irreps*.

    Attributes
    ----------
    k : Real ndarray, the k-point
    group : String
        Point group of the little group (Schoenflies: 'C6v', 'C3v', 'Oh', ...).
    operations : List of :class:`Operation`
        The little group of *k*.
    energies : Real ndarray
        Energy of each level (group of degenerate bands).
    bands : List of lists of band indices, one per level
    labels : List of strings, one per level
        The irrep (Mulliken label), or a sum
        such as 'A1+B2' for an accidental degeneracy.
    characters : Complex ndarray, shape (n_levels, n_operations)
    '''

    def __init__(self, k, group, operations, energies, bands, labels, characters) -> None:
        self.k = k
        self.group = group
        self.operations = operations
        self.energies = energies
        self.bands = bands
        self.labels = labels
        self.characters = characters


def _irreps(ks: KSpace, k: NDArray[np.float64], ops: list[Operation], deg_tol: float) -> Irreps | None:
    '''
    Private function. *irreps*, or None if the point group of the little
    group is not tabulated.
    '''
    little = [op for op in ops if not op.antiunitary and _invariant(ks, op, k)]
    rots = [op.rotation for op in little]
    a = np.array(ks.lat.prim_vec, dtype='f8')
    # symmorphic about a common centre c: t = c - R c for every operation
    stack = np.concatenate([np.eye(ks.space_dim) - r for r in rots])
    t = np.concatenate([op.translation for op in little])
    c = np.linalg.lstsq(stack, t, rcond=None)[0]
    error_handling.symmorphic(float(np.max(np.abs(stack @ c - t))))
    table = _point_group(rots, a[0])
    if table is None:
        return None
    name, irreps = table
    en, vec = ks._eigs(k[None], eigenvec=True)
    en, vec = en[0].real, vec[0]
    us = np.array([_representations(ks, op, k[None])[0][0] for op in little])
    levels = np.split(np.arange(ks.norb), np.flatnonzero(np.diff(en) > deg_tol) + 1)
    chars = np.array([np.einsum('ia,gij,ja->g', vec[:, lv].conj(), us, vec[:, lv]) for lv in levels])
    chi = np.array([c for _, c in irreps])
    mult = (chi.conj() @ chars.T).T / len(little)
    error_handling.multiplicities(mult)
    labels = []
    for row in np.round(mult.real).astype(int):
        labels.append('+'.join(('{}'.format(m) if m > 1 else '') + label
                                         for (label, _), m in zip(irreps, row) if m))
    return Irreps(k, name, little, np.array([en[lv].mean() for lv in levels]),
                       [lv.tolist() for lv in levels], labels, chars)


def irreps(
    ks: KSpace, k: ArrayLike, ops: list[Operation] | None = None, deg_tol: float = 1e-6,
) -> Irreps:
    r'''
    Get the irreducible representations of the bands at a k-point
    (Bouckaert, Smoluchowski and Wigner, Phys. Rev. 50, 58 (1936)): the
    little group of :math:`\mathbf{k}` (the unitary operations with
    :math:`g\mathbf{k} = \mathbf{k}` modulo a reciprocal lattice vector),
    the characters :math:`\chi(g) = \mathrm{Tr}\,V^\dagger U_g(\mathbf{k})V`
    on each level :math:`V` (degenerate bands within *deg_tol*), and their
    decomposition on the character table of the point group.

    Tabulated: the ten 2D point groups (C1, C2, C3, C4, C6, Cs, C2v, C3v,
    C4v, C6v) and the five cubic ones (T, Th, O, Td, Oh, with the cube
    axes along x, y, z); spinless models; symmorphic little groups (every
    operation a rotation about one point, see *Operation*). Conventions: in
    C4v and C6v (and C2v), :math:`\sigma_v` is the class of the mirror
    lines closest to :math:`\mathbf{a}_1` (B1 is even under it); the
    complex pairs of the cyclic groups are ¹E (:math:`\chi(C_n) =
    e^{2\pi i/n}`, :math:`C_n` counterclockwise) and ²E; in T, :math:`C_3`
    is the counterclockwise rotation about [111].

    Parameters
    ----------
    ks : **KSpace** instance, spinless, Hermitian, 2D or 3D (dim = space_dim)
    k : k-point (coordinates of *get_ham*)
    ops : List of :class:`Operation`, default None (*find_symmetries*)
    deg_tol : Positive real, default 1e-6
        Energies closer than this are one level.

    Returns
    -------
    result : :class:`Irreps`

    Examples
    --------
    Example usage::

        res = irreps(graphene, K)
        res.group, res.labels      # 'C3v', ['E'] (the Dirac point)
    '''
    error_handling.spinless_irreps(ks.spin)
    error_handling.full_dim(ks.dim, ks.space_dim)
    error_handling.dim_irreps(ks.dim)
    error_handling.hermitian_model(ks.is_hermitian())
    error_handling.no_overlap(ks._overlap_hop)
    error_handling.k_vector(k, 'k', ks.dim)
    error_handling.positive_real(deg_tol, 'deg_tol')
    ops = find_symmetries(ks, antiunitary=False) if ops is None else ops
    error_handling.operations(ops)
    res = _irreps(ks, np.array(k, dtype='f8'), ops, deg_tol)
    error_handling.tabulated(res)
    assert res is not None
    return res


def _rotation_data(ks: KSpace, op: Operation, bands: list[int]) -> tuple[int, float, list, Callable]:
    r'''
    Private function. For a rotation :math:`C_n` of a 2D model: :math:`n`,
    the signed angle, the points with a nontrivial stabilizer, one per
    orbit, as (k, s) with :math:`C_n^s` the rotation that keeps k, and the
    function ``eig(s, k)``: the eigenvalues of :math:`C_n^s` on *bands* at k.
    '''
    error_handling.dim_exact(ks.dim, 2)
    error_handling.full_dim(ks.dim, ks.space_dim)
    error_handling.is_operation(op)
    error_handling.proper_rotation(op)
    error_handling.band_indices(bands, ks.norb)
    error_handling.is_symmetry(ks.symmetry_error(op), 1e-8)
    theta = float(np.arctan2(op.rotation[1, 0], op.rotation[0, 0]))
    n = int(round(2 * PI / abs(theta)))
    rec = ks.rec_vec_k

    def eig(power, k):
        v = ks._subspaces(k[None], bands)[0]
        u = _representations(ks, op ** power, k[None])[0][0]
        m = v.conj().T @ u @ v
        error_handling.invariant_bands(np.linalg.svd(m, compute_uv=False))
        return np.linalg.eigvals(m)

    def reduced(k):
        f = np.linalg.solve(rec.T, k)
        return (f - np.floor(f + TOL)) @ rec

    seen: list = []
    points = []
    for k in np.array(list(itertools.product(np.arange(6) / 6, repeat=2))) @ rec:
        if any(np.allclose(k, p, atol=TOL) for p in seen):
            continue
        orbit = [k]
        while True:
            nxt = reduced(_representations(ks, op, orbit[-1][None])[1][0])
            if np.allclose(nxt, k, atol=TOL):
                break
            orbit.append(nxt)
        seen += orbit
        if len(orbit) < n:
            points.append((k, len(orbit)))
    return n, theta, points, eig


def rotation_indicator(ks: KSpace, op: Operation, bands: list[int]) -> int:
    r'''
    Get the Chern number modulo :math:`n` of a group of bands of a
    :math:`C_n`-symmetric 2D model from its rotation eigenvalues (Fang,
    Gilbert and Bernevig, Phys. Rev. B 86, 115112 (2012)):

    .. math::

        e^{2\pi i C/n} = (-1)^{FN}\prod_{\mathbf{k}_*}\ \prod_{m \in \mathrm{bands}}
        \lambda_m\big(\mathbf{k}_*, C_n^{s}\big)\, ,

    over one point :math:`\mathbf{k}_*` of each orbit of the points left
    invariant by some power of :math:`C_n` (:math:`s` the length of the
    orbit, :math:`C_n^s` the rotation that keeps it): :math:`\Gamma` and
    M, and X for :math:`C_4`; :math:`\Gamma`, K and K' for :math:`C_3`;
    :math:`\Gamma`, K and M for :math:`C_6`; the four time-reversal
    invariant momenta for :math:`C_2`. When :math:`U_{C_n}^n = -1` (spin
    1/2), the eigenvalues are those of :math:`e^{i\pi/n}C_n`, whose
    :math:`n`-th power is 1: this gives the factor :math:`(-1)^{FN} =
    (-1)^N`, :math:`N` = len(*bands*), for :math:`C_3`, :math:`C_4` and
    :math:`C_6`, and 1 for :math:`C_2`. No integral over the zone: only the
    eigenvalues at a few points.

    Parameters
    ----------
    ks : **KSpace** instance, 2D (dim = space_dim)
    op : :class:`Operation`
        A rotation :math:`C_n` (n = 2, 3, 4 or 6),
        unitary, a symmetry of the model, about its centre (see *Operation*).
        Its sense sets the sign: :math:`C` is that of *chern_number* for a
        counterclockwise rotation.
    bands : List of band indices (isolated from the others at the invariant points)

    Returns
    -------
    c : Integer, :math:`C \bmod n`, in [0, n)
    '''
    n, theta, points, eig = _rotation_data(ks, op, bands)
    u_n = _representations(ks, op ** n, np.zeros((1, 2)))[0][0]
    # with U^n = -1 (spin 1/2), e^(i pi/n) U is a rotation with U^n = +1
    shift = 0. if error_handling.rotation_power(u_n) == 1 else np.sign(theta) * PI / n
    product = 1. + 0j
    for k, s in points:
        product *= complex(np.prod(np.exp(1j * shift * s) * eig(s, k)))
    x = float(np.angle(product)) / theta
    error_handling.quantized(x, 'rotation eigenvalues')
    return int(round(x)) % n


def inversion_indicators(ks: KSpace, op: Operation, bands: list[int]) -> tuple[int, int, int, int]:
    r'''
    Get the symmetry indicators of a 3D inversion- and time-reversal
    symmetric insulator (Khalaf, Po, Vishwanath and Watanabe, Phys. Rev. X
    8, 031070 (2018)) from the numbers :math:`n^\pm_{\mathbf{K}}` of
    occupied states of parity :math:`\pm1` at the eight time-reversal
    invariant momenta :math:`\mathbf{K}`:

    .. math::

        z_4 = \sum_{\mathbf{K}} \frac{n^+_{\mathbf{K}} - n^-_{\mathbf{K}}}{4} \bmod 4\, ,\qquad
        z_{2,i} = \sum_{K_i = 1/2} \frac{n^+_{\mathbf{K}} - n^-_{\mathbf{K}}}{4} \bmod 2\, ,

    :math:`K_i` the fractional coordinate along :math:`\mathbf{b}_i`.
    :math:`z_4 \bmod 2` is the strong :math:`\mathbb{Z}_2` index
    :math:`\nu_0` and :math:`z_{2,i}` the weak ones :math:`\nu_i` (Fu and
    Kane 2007, see *z2_indices_3d*); :math:`z_4 = 2` (with
    :math:`\nu_0 = 0`) is an insulator that :math:`\mathbb{Z}_2` misses: a
    doubled strong topological insulator, with hinge states.

    Parameters
    ----------
    ks : **KSpace** instance, 3D
    op : :class:`Operation`
        The inversion (rotation :math:`-1`), unitary, a
        symmetry of the model, about its centre.
    bands : List of band indices (an even number: Kramers pairs)

    Returns
    -------
    indicators : Tuple of four integers :math:`(z_4, z_{2,1}, z_{2,2}, z_{2,3})`
    '''
    error_handling.dim_exact(ks.dim, 3)
    error_handling.is_operation(op)
    error_handling.inversion_operation(op)
    error_handling.even_bands(bands)
    error_handling.band_indices(bands, ks.norb)
    error_handling.is_symmetry(ks.symmetry_error(op), 1e-8)
    fracs = np.array(list(itertools.product((0., 0.5), repeat=3)))
    ks_trim = fracs @ ks.rec_vec_k
    v = ks._subspaces(ks_trim, bands)
    u = _representations(ks, op, ks_trim)[0]
    diff = np.zeros(8)
    for p in range(8):
        xi = np.linalg.eigvals(v[p].conj().T @ u[p] @ v[p])
        error_handling.parities(xi)
        diff[p] = np.sum(xi.real > 0) - np.sum(xi.real < 0)
    z4 = int(round(diff.sum() / 4)) % 4
    weak = [int(round(diff[fracs[:, i] == 0.5].sum() / 4)) % 2 for i in range(3)]
    return z4, weak[0], weak[1], weak[2]


def corner_charge(ks: KSpace, op: Operation, bands: list[int]) -> float:
    r'''
    Get the corner charge of a :math:`C_4`-symmetric 2D insulator from its
    rotation eigenvalues (Benalcazar, Li and Hughes, Phys. Rev. B 99,
    245151 (2019)):

    .. math::

        Q^{(4)} = \tfrac{1}{4}\left([X_1^{(2)}] + 2[M_1^{(4)}] + 3[M_2^{(4)}]\right) \bmod 1\, ,

    :math:`[\Pi_p^{(m)}] = \#\Pi_p^{(m)} - \#\Gamma_p^{(m)}` the number of
    bands of eigenvalue :math:`e^{2\pi i(p-1)/m}` of the rotation
    :math:`C_m` that keeps :math:`\Pi` (:math:`C_4^2` at
    X :math:`= \mathbf{b}_1/2`, :math:`C_4` at M :math:`=
    (\mathbf{b}_1+\mathbf{b}_2)/2`), minus that at :math:`\Gamma`. It is
    the excess number of electrons (charge in units of :math:`-e`, as
    *higher_order.corner_charges*) in a quadrant of a square flake of whole
    unit cells, with *bands* filled in the bulk (and the edge states below
    the bulk gap) on an ionic background of len(*bands*) electrons per
    cell. The flake is built from the unit cell, so the rotation must be
    about its centre: it maps the orbitals of the unit cell onto
    themselves. The formula assumes a spinless model with time reversal.

    (The :math:`C_2` formula of the same paper needs a vanishing
    polarization, and was not reproduced here by flakes: it is left out.)

    Parameters
    ----------
    ks : **KSpace** instance, 2D, spinless
    op : :class:`Operation`
        A rotation :math:`C_4` about the centre of the
        unit cell, unitary, a symmetry of the model.
    bands : List of band indices (the occupied bands)

    Returns
    -------
    q : Real number in [0, 1), a multiple of 1/4
    '''
    error_handling.spinless_irreps(ks.spin)
    n, theta, _, eig = _rotation_data(ks, op, bands)
    error_handling.corner_rotation(n)
    error_handling.cell_rotation(_Action(ks, op).L)
    b1, b2 = ks.rec_vec_k
    gamma = np.zeros(2)

    def count(k, s, p):
        '''[Pi_p] at k, whose stabilizer C_(n/s) is generated by g^s'''
        target = np.exp(1j * np.sign(theta) * 2 * PI * (p - 1) * s / n)
        return int(np.sum(np.abs(eig(s, k) - target) < 1e-4)) - int(np.sum(np.abs(eig(s, gamma) - target) < 1e-4))

    m = (b1 + b2) / 2
    q = (count(b1 / 2, 2, 1) + 2 * count(m, 1, 1) + 3 * count(m, 1, 2)) / 4
    return float(q % 1.)


def _closure(ks: KSpace, ops: list[Operation]) -> list[Operation]:
    '''
    Private function. The group generated by *ops* (modulo lattice
    translations, and the phase of the orbital matrices).
    '''
    a = np.array(ks.lat.prim_vec, dtype='f8')
    m = None if ops[0].orbitals is None else len(ops[0].orbitals)
    group = [Operation(np.eye(ks.space_dim), orbitals=None if m is None else np.eye(m))]

    def same(g, h):
        if g.antiunitary != h.antiunitary or not np.allclose(g.rotation, h.rotation, atol=TOL):
            return False
        if not _lattice_vector(g.translation - h.translation, a):
            return False
        if g.orbitals is None:
            return True
        overlap = np.trace(g.orbitals.conj().T @ h.orbitals) / len(g.orbitals)
        return bool(np.allclose(h.orbitals, overlap * g.orbitals, atol=TOL))

    todo = list(ops)
    while todo:
        g = todo.pop()
        if any(same(g, h) for h in group):
            continue
        group.append(g)
        error_handling.group_size(len(group))
        todo += [g @ h for h in group] + [h @ g for h in group]
    return group


def symmetrize(ks: KSpace, ops: list[Operation]) -> KSpace:
    r'''
    Symmetrize a model: the group average of its hoppings,

    .. math::

        H_{\mathrm{sym}} = \frac{1}{|G|}\sum_{g\in G} g\,H\,g^{-1},\qquad
        (gHg^{-1})_{\pi(a)\pi(b)}(R\mathbf{R} + \mathbf{L}_b - \mathbf{L}_a) =
        D\,H_{ab}(\mathbf{R})\,D^\dagger

        (D\,H_{ab}(\mathbf{R})^*D^\dagger \text{ if } g \text{ is antiunitary}),

    over the group :math:`G` generated by *ops*, as
    ``TBmodels.Model.symmetrize``. A model read from Wannier90, whose
    symmetries are broken by numerical noise, gets them back exactly: its
    degeneracies become exact, and *symmetry_error* zero. A symmetric model
    is unchanged; the result is the projection of the hoppings onto the
    symmetric ones (closest in the Frobenius norm).

    Parameters
    ----------
    ks : **KSpace** instance, without overlap
        Value functions are evaluated with *params*.
    ops : List of :class:`Operation`, the generators

    Returns
    -------
    ks_sym : **KSpace** instance, on the same lattice
    '''
    error_handling.no_overlap(ks._overlap_hop)
    error_handling.operations(ops)
    group = _closure(ks, ops)
    a = np.array(ks.lat.prim_vec, dtype='f8')
    pinv = np.linalg.pinv(a)
    # H(R) as a dictionary {integer R: (norb, norb) matrix}
    hams: dict = {(0,) * ks.dim: np.diag(ks.onsite) + ks._onsite_offdiag}
    for i, j, R, t in ks._hop:
        key = tuple(int(x) for x in np.round(np.asarray(R) @ pinv))
        hams.setdefault(key, np.zeros((ks.norb, ks.norb), dtype='c16'))[i, j] += t
    out: dict = {}
    for g in group:
        act = _Action(ks, g)
        m = np.round(a @ g.rotation.T @ pinv)  # integer rotation: R (n @ a) = (n @ m) @ a
        for key, h in hams.items():
            h = h.conj() if g.antiunitary else h
            for s, s_img in enumerate(act.perm):
                for s2, s2_img in enumerate(act.perm):
                    new = tuple(int(x) for x in np.array(key) @ m + act.L[s2] - act.L[s])
                    blk = act.D[s] @ h[np.ix_(act.idx[s], act.idx[s2])] @ act.D[s2].conj().T
                    mat = out.setdefault(new, np.zeros((ks.norb, ks.norb), dtype='c16'))
                    mat[np.ix_(act.idx[s_img], act.idx[s2_img])] += blk / len(group)
    new_ks = KSpace(ks.lat, spin=ks.spin)
    new_ks._nonreciprocal = ks._nonreciprocal
    hops = []
    for key, mat in out.items():
        R_cart = np.array(key, dtype='f8') @ a
        if not any(key):
            new_ks.onsite = np.diag(mat).copy()
            mat = mat - np.diag(np.diag(mat))
        for i, j in zip(*np.nonzero(np.abs(mat) > 1e-14)):
            hops.append((int(i), int(j), R_cart, complex(mat[i, j])))
    new_ks._hop = hops
    return new_ks
