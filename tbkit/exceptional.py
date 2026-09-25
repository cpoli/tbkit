r"""
Band touchings of two-dimensional (non-Hermitian) Bloch bands:
**diabolical points** (DPs), Hermitian degeneracies such as graphene's
Dirac points, where two eigenvalues meet but their eigenvectors stay
distinct; and **exceptional points** (EPs), where the eigenvalues *and*
the eigenvectors coalesce, so that :math:`H(\mathbf{k})` stops being
diagonalizable (a Jordan block).

The tools below tell them apart and compute their invariants. Each takes a
*model*: a 2D **KSpace** (the plane is the Brillouin zone, in the k
coordinates of *get_ham*), or any callable ``model(p)`` returning a square
matrix for a point ``p`` of a two-parameter plane.

* **Eigenvalue vorticity** of a pair of bands around a closed loop
  :math:`\Gamma` (Shen, Zhen and Fu, Phys. Rev. Lett. 120, 146402 (2018)),

  .. math::

      \nu_{mn}(\Gamma) = -\frac{1}{2\pi}\oint_\Gamma
      \nabla_{\mathbf{k}}\arg\left[E_m(\mathbf{k}) - E_n(\mathbf{k})\right]\cdot d\mathbf{k}\, .

  Band labels are not single-valued around an EP (the two eigenvalues swap
  after one loop), so *vorticity* follows the eigenvalues by continuation
  along the loop (*track_eigenvalues*), never by sorting. A second-order EP
  has :math:`\nu = \pm1/2`, a DP :math:`\nu = 0`.
* **Discriminant winding** (Yang, Schnyder, Hu and Chiu, Phys. Rev. Lett.
  126, 086401 (2021)): the discriminant
  :math:`\Delta(\mathbf{k}) = \prod_{m<n}(E_m - E_n)^2` is a polynomial in the
  matrix elements, computed from the characteristic polynomial without any
  eigenvalue, and its winding
  :math:`W(\Gamma) = \frac{1}{2\pi}\oint_\Gamma \nabla_{\mathbf{k}}\arg\Delta\cdot d\mathbf{k}`
  counts the EPs inside :math:`\Gamma` with their charges. With the two sign
  conventions above, :math:`W = -2\sum_{m<n}\nu_{mn}`: :math:`W = -2\nu = \mp1`
  for a second-order EP of a two-band model.
* **EP finder** (*find_exceptional_points*): zeros of :math:`\Delta` with
  :math:`W \neq 0` on a plaquette mesh of the Brillouin zone, refined by
  Newton's method, with their charge and their order (the size of the Jordan
  block, from the Petermann factors, which diverge at an EP). Since
  :math:`\Delta` is periodic, the charges sum to zero over the zone: EPs come in
  pairs of opposite charge (the doubling theorem). When a symmetry makes
  :math:`\Delta` real, EPs form lines instead (exceptional rings, Zhen et al.,
  Nature 525, 354 (2015)), which the finder also returns; *fermi_arcs* draws
  the bulk Fermi arcs :math:`\mathrm{Re}(E_m - E_n) = 0` that join a pair of
  EPs (Kozii and Fu, arXiv:1708.05841 (2017); Zhou et al., Science 359, 1009
  (2018)).
* **Encircling** (*encircle*): parallel transport of an eigenvector around
  a loop. Around a DP one loop returns the state with a Berry phase
  :math:`\pi`; around a second-order EP one loop swaps the two eigenstates,
  two loops return the state with a sign flip, and four loops restore it
  (Heiss, Phys. Rev. E 61, 929 (2000); Dembowski et al., Phys. Rev. Lett.
  86, 787 (2001)).
"""
from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray
import scipy.linalg as LA
from scipy.optimize import linear_sum_assignment

import tbkit.error_handling as error_handling
from tbkit.kspace import KSpace


PI = np.pi
_HALVINGS = 30  # a continuation step may be halved this many times
_ARG_STEP = PI / 4  # largest phase change of an eigenvalue difference per step
_MESH_SHIFT = (0.3183098861837907, 0.2718281828459045)  # keeps mesh edges off symmetry lines


#################################
# MODELS AND LOOPS
#################################


def _model(model) -> Callable[[NDArray[np.float64]], NDArray[np.complex128]]:
    '''
    Private function. The matrix-valued function p -> H(p) of a validated model.
    '''
    error_handling.ham_model(model, KSpace)
    if isinstance(model, KSpace):
        error_handling.no_overlap(model._overlap_hop)
        return model.get_ham

    def ham(p):
        h = np.asarray(model(np.asarray(p, dtype='f8')))
        error_handling.ham_matrix(h)
        return h.astype('c16')
    return ham


def circle(center: ArrayLike, radius: float) -> Callable[[float], NDArray[np.float64]]:
    r'''
    Get a circular loop, run counterclockwise:
    :math:`\mathbf{k}(s) = \mathbf{k}_0 + r(\cos 2\pi s, \sin 2\pi s)`,
    :math:`s\in[0, 1]`.

    :param center: Tuple/list/ndarray of 2 real numbers. :math:`\mathbf{k}_0`.
    :param radius: Positive real. :math:`r`.

    :returns:
        * **loop** -- Callable of s, returning a real ndarray of shape (2,).
    '''
    error_handling.k_vector(center, 'center', 2)
    error_handling.positive_real(radius, 'radius')
    k0 = np.asarray(center, dtype='f8')

    def loop(s):
        return k0 + radius * np.array([np.cos(2 * PI * s), np.sin(2 * PI * s)])
    return loop


def _loop(loop) -> Callable[[float], NDArray[np.float64]]:
    '''
    Private function. The callable s -> k(s), s in [0, 1], of a validated loop
    (a callable, or the vertices of a polygon, each side covering an equal
    range of s).
    '''
    error_handling.loop(loop)
    if callable(loop):
        def path(s):
            p = np.asarray(loop(float(s)), dtype='f8')
            error_handling.loop_point(p)
            return p
        return path
    vert = np.asarray(loop, dtype='f8')
    closed = np.vstack([vert, vert[:1]])
    m = len(vert)

    def path(s):
        x = (s % 1.) * m
        i = min(int(x), m - 1)
        return closed[i] + (x - i) * (closed[i + 1] - closed[i])
    return path


def _closed(ham, path) -> int:
    '''
    Private function. Check that the loop is closed (H at its end equals H at
    its start, which allows loops across the zone), and return norb.
    '''
    h0 = ham(path(0.))
    error_handling.closed_loop(h0, ham(path(1.)))
    return h0.shape[0]


#################################
# EIGENVALUE CONTINUATION
#################################


def _eig(h: NDArray[np.complex128], vectors: bool):
    '''
    Private function. Eigenvalues, right and left eigenvectors (as kets:
    columns of vl with vl^H H = E vl^H) of the general eigenproblem.
    '''
    if vectors:
        w, vl, vr = LA.eig(h, left=True, right=True)
        return w, vr, vl
    return LA.eigvals(h), None, None


def _match(pred, w_old, w_new, vr_old, vr_new):
    '''
    Private function. Permutation of *w_new* continuing *w_old* (predicted
    at *pred*), or None if the step is ambiguous: a matched eigenvalue must
    be four times closer to its prediction than to any other eigenvalue,
    no eigenvalue difference may turn by more than pi/4, and (with vectors)
    each eigenvector must stay close to its predecessor. Eigenvalues equal to
    rounding precision form a cluster and do not make the step ambiguous.
    '''
    _, perm = linear_sum_assignment(np.abs(pred[:, None] - w_new[None, :]))
    w = w_new[perm]
    deg = 1e-9 * max(1., float(np.max(np.abs(w))))
    dist = np.abs(w[:, None] - w[None, :])
    np.fill_diagonal(dist, np.inf)
    single = dist.min(axis=1) >= deg  # no partner equal to rounding precision
    dist[dist < deg] = np.inf
    if np.any(np.abs(w - pred) >= 0.25 * dist.min(axis=1)):
        return None
    d_old = w_old[:, None] - w_old[None, :]
    d_new = w[:, None] - w[None, :]
    keep = (np.abs(d_old) > deg) & (np.abs(d_new) > deg)
    if np.any(np.abs(np.angle(d_new[keep] / d_old[keep])) > _ARG_STEP):
        return None
    if vr_old is not None:
        # within a degenerate cluster the eigenvectors are arbitrary
        overlap = np.abs(np.sum(vr_old.conj() * vr_new[:, perm], axis=0))
        if np.any(overlap[single] < 0.9):
            return None
    return perm


def _track(ham, path, s_end: float, nk: int, vectors: bool = False):
    '''
    Private function. Follow the eigenvalues of H(path(s)) by continuation
    from s = 0 to *s_end* (loops are evaluated at s modulo 1): steps of at
    most 1/nk, halved while a step is ambiguous (see *_match*), with a
    linear prediction from the previous step. The labels are those of the
    eigenvalues at s = 0 sorted by real part.

    Returns s (n,), en (n, norb), and, with *vectors*, the right and left
    eigenvectors (n, norb, norb), column m following label m.
    '''
    ds_max = 1. / nk
    ds_min = ds_max / 2 ** _HALVINGS
    w, vr, vl = _eig(ham(path(0.)), vectors)
    order = np.argsort(w.real, kind='stable')
    w = w[order]
    if vectors:
        vr, vl = vr[:, order], vl[:, order]
    s_list, w_list, vr_list, vl_list = [0.], [w], [vr], [vl]
    slope = np.zeros_like(w)
    s, ds = 0., ds_max
    while s < s_end:
        last = ds >= s_end - s
        s_new = s_end if last else s + ds
        step = s_new - s
        k = path(s_new % 1. if s_new != s_end or s_end % 1. else 1.)
        w_new, vr_new, vl_new = _eig(ham(k), vectors)
        perm = _match(w + slope * step, w, w_new, vr, vr_new)
        if perm is None:
            if step <= ds_min:
                error_handling.tracking(k)
            ds = step / 2
            continue
        slope = (w_new[perm] - w) / step
        w, s = w_new[perm], s_new
        if vectors:
            vr, vl = vr_new[:, perm], vl_new[:, perm]
        s_list.append(s)
        w_list.append(w)
        vr_list.append(vr)
        vl_list.append(vl)
        ds = min(2 * step, ds_max)
    s_arr, en = np.array(s_list), np.array(w_list)
    if vectors:
        return s_arr, en, np.array(vr_list), np.array(vl_list)
    return s_arr, en, None, None


def track_eigenvalues(
    model, loop, n_loops: int = 1, nk: int = 100,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.complex128]]:
    r'''
    Follow the (complex) eigenvalues of :math:`H(\mathbf{k})` continuously
    around a closed loop, *n_loops* times. Each step (at most 1/nk of the
    loop) matches the new eigenvalues to a linear prediction from the
    previous ones, and is halved while that matching is ambiguous; nothing
    depends on how the eigensolver orders its output. Label :math:`m` is the
    :math:`m`-th eigenvalue at the start of the loop, sorted by real part.
    After one loop around a second-order exceptional point the labels come
    back swapped: ``en[-1]`` is then a permutation of ``en[0]``.

    :param model: 2D **KSpace**, or callable of a point of a plane returning
        a square matrix.
    :param loop: Closed loop: a callable of :math:`s\in[0, 1]` returning a
        point (e.g. *circle*), or an array of polygon vertices, shape (m, 2).
        For a **KSpace**, points are k in the coordinates of *get_ham*, and a
        loop may close up to a reciprocal lattice vector.
    :param n_loops: Positive integer. Default value 1. Number of turns.
    :param nk: Positive integer. Default value 100. Steps per turn before
        refinement.

    :returns:
        * **s** -- Real ndarray, shape (n,). Loop parameter, from 0 to *n_loops*.
        * **k** -- Real ndarray, shape (n, 2). Points of the loop.
        * **en** -- Complex ndarray, shape (n, norb). Eigenvalues, en[:, m]
          continuous in s.
    '''
    ham = _model(model)
    path = _loop(loop)
    error_handling.positive_int(n_loops, 'n_loops')
    error_handling.positive_int(nk, 'nk')
    _closed(ham, path)
    s, en, _, _ = _track(ham, path, float(n_loops), nk)
    return s, np.array([path(x % 1.) for x in s]), en


def vorticity(model, loop, bands: tuple[int, int] = (0, 1), nk: int = 100) -> float:
    r'''
    Get the eigenvalue vorticity of a pair of bands around a closed loop
    :math:`\Gamma` (Shen, Zhen and Fu, Phys. Rev. Lett. 120, 146402 (2018)),

    .. math::

        \nu_{mn}(\Gamma) = -\frac{1}{2\pi}\oint_\Gamma
        \nabla_{\mathbf{k}}\arg\left[E_m(\mathbf{k}) - E_n(\mathbf{k})\right]
        \cdot d\mathbf{k}\, ,

    with the eigenvalues followed by continuation (see *track_eigenvalues*).
    :math:`\nu_{mn} = \pm1/2` around a second-order exceptional point, where
    :math:`E_m - E_n \propto \sqrt{\mathbf{k}-\mathbf{k}_{EP}}` changes sign
    after one loop; :math:`\pm1/3` for each pair of a third-order one;
    0 around a diabolical point, or a loop enclosing no degeneracy.
    :math:`\nu_{mn} = \nu_{nm}`.

    :param model: See *track_eigenvalues*.
    :param loop: See *track_eigenvalues*. Counterclockwise for the sign above.
    :param bands: Pair of distinct integers. Default value (0, 1). Band labels
        :math:`(m, n)` at the start of the loop, sorted by real part.
    :param nk: Positive integer. Default value 100. See *track_eigenvalues*.

    :returns:
        * **nu** -- Real number, a multiple of 1/2 around second-order EPs.
    '''
    ham = _model(model)
    path = _loop(loop)
    error_handling.positive_int(nk, 'nk')
    norb = _closed(ham, path)
    error_handling.band_pair(bands, norb)
    s, en, _, _ = _track(ham, path, 1., nk)
    diff = en[:, bands[0]] - en[:, bands[1]]
    if np.any(diff == 0.):
        error_handling.tracking(path(s[np.argmin(np.abs(diff))] % 1.))
    return float(-np.sum(np.angle(diff[1:] / diff[:-1])) / (2 * PI))


#################################
# DISCRIMINANT
#################################


def _charpoly(h: NDArray[np.complex128]) -> NDArray[np.complex128]:
    r'''
    Private function. Coefficients, highest degree first, of the monic
    characteristic polynomial :math:`p(z) = \det(z - H)`, from its values
    at N+1 points of a circle of radius :math:`\|H\|_F` (a discrete Fourier
    transform; no eigenvalues).
    '''
    n = h.shape[0]
    r = LA.norm(h) or 1.
    z = r * np.exp(2j * PI * np.arange(n + 1) / (n + 1))
    vals = np.array([LA.det(zz * np.eye(n) - h) for zz in z])
    coef = np.fft.fft(vals) / (n + 1) / r ** np.arange(n + 1)
    coef[n] = 1.
    return coef[::-1]


def _discriminant(h: NDArray[np.complex128]) -> complex:
    r'''
    Private function. :math:`\Delta = (-1)^{N(N-1)/2}\,\mathrm{Res}(p, p')`
    of the monic characteristic polynomial, from the determinant of the
    Sylvester matrix, computed for :math:`H/\|H\|_F` and rescaled.
    '''
    n = h.shape[0]
    if n < 2:
        return 1. + 0j
    r = LA.norm(h)
    if r == 0.:
        return 0j
    c = _charpoly(h / r)
    dc = c[:-1] * np.arange(n, 0, -1)
    syl = np.zeros((2 * n - 1, 2 * n - 1), 'c16')
    for i in range(n - 1):
        syl[i, i:i + n + 1] = c
    for i in range(n):
        syl[n - 1 + i, i:i + n] = dc
    return complex((-1) ** (n * (n - 1) // 2) * LA.det(syl) * r ** (n * (n - 1)))


def discriminant(model, k: ArrayLike) -> complex:
    r'''
    Get the discriminant of :math:`H(\mathbf{k})`,

    .. math::

        \Delta(\mathbf{k}) = \prod_{m<n}\left[E_m(\mathbf{k}) - E_n(\mathbf{k})\right]^2
        = (-1)^{N(N-1)/2}\,\mathrm{Res}(p, p')\, ,

    computed as the resultant of the characteristic polynomial
    :math:`p(E) = \det[E - H(\mathbf{k})]` and its derivative (the determinant
    of their Sylvester matrix), without the eigenvalues. It is a polynomial
    in the matrix elements, hence smooth and single-valued, and it vanishes
    exactly where two eigenvalues meet. For a two-band model
    :math:`H = d_0 + \mathbf{d}\cdot\boldsymbol\sigma`,
    :math:`\Delta = 4\,\mathbf{d}\cdot\mathbf{d}`.

    :param model: See *track_eigenvalues*.
    :param k: Tuple/list/ndarray of 2 real numbers. Point of the plane.

    :returns:
        * **delta** -- Complex number, in units of energy^(N(N-1)) (1 if N = 1).
    '''
    ham = _model(model)
    error_handling.k_vector(k, 'k', 2)
    return _discriminant(ham(np.asarray(k, dtype='f8')))


def _arg_change(disc, path, s_a, s_b, d_a, d_b, depth=0) -> float:
    '''
    Private function. Change of arg(Delta) from path(s_a) to path(s_b),
    bisecting until each piece turns by at most pi/4.
    '''
    if d_a == 0. or d_b == 0. or depth > 40:
        error_handling.winding_path(path(s_b if d_b == 0. else s_a))
    step = float(np.angle(d_b / d_a))
    if abs(step) <= _ARG_STEP:
        return step
    s_m = (s_a + s_b) / 2
    d_m = disc(path(s_m))
    return (_arg_change(disc, path, s_a, s_m, d_a, d_m, depth + 1)
                + _arg_change(disc, path, s_m, s_b, d_m, d_b, depth + 1))


def _winding(ham, path, nk: int) -> float:
    '''
    Private function. Winding of the discriminant along path(s), s in [0, 1].
    '''
    def disc(p):
        return _discriminant(ham(p))
    s = np.linspace(0., 1., nk + 1)
    d = [disc(path(x)) for x in s]
    total = sum(_arg_change(disc, path, s[i], s[i + 1], d[i], d[i + 1]) for i in range(nk))
    return total / (2 * PI)


def discriminant_winding(model, loop, nk: int = 100) -> float:
    r'''
    Get the winding number of the discriminant (see *discriminant*) around a
    closed loop :math:`\Gamma` (Yang, Schnyder, Hu and Chiu, Phys. Rev. Lett.
    126, 086401 (2021)),

    .. math::

        W(\Gamma) = \frac{1}{2\pi}\oint_\Gamma \nabla_{\mathbf{k}}\arg\Delta(\mathbf{k})
        \cdot d\mathbf{k} = -2\sum_{m<n}\nu_{mn}(\Gamma)\, ,

    the total charge of the exceptional points inside :math:`\Gamma`
    (:math:`\pm1` for a second-order EP, where :math:`\Delta` vanishes
    linearly). It needs no eigenvalue tracking. A diabolical point has
    :math:`W = 0`: :math:`\Delta \geq 0` for a Hermitian model. The relation to
    the vorticities *vorticity* holds with the signs as written, the minus
    sign coming from Shen, Zhen and Fu's definition of :math:`\nu`.

    :param model: See *track_eigenvalues*.
    :param loop: See *track_eigenvalues*. Counterclockwise for the sign above.
    :param nk: Positive integer. Default value 100. Points on the loop; a
        step along which :math:`\arg\Delta` turns by more than :math:`\pi/4`
        is bisected.

    :returns:
        * **W** -- Real number, close to an integer.
    '''
    ham = _model(model)
    path = _loop(loop)
    error_handling.positive_int(nk, 'nk')
    _closed(ham, path)
    return float(_winding(ham, path, nk))


#################################
# EXCEPTIONAL POINTS
#################################


def _petermann(h: NDArray[np.complex128]):
    '''
    Private function. Eigenvalues (sorted by real part) and Petermann factors.
    '''
    w, vl, vr = LA.eig(h, left=True, right=True)
    order = np.argsort(w.real, kind='stable')
    w, vl, vr = w[order], vl[:, order], vr[:, order]
    norms = np.sum(np.abs(vl) ** 2, axis=0) * np.sum(np.abs(vr) ** 2, axis=0)
    pair = np.abs(np.sum(vl.conj() * vr, axis=0)) ** 2
    return w, norms / np.maximum(pair, np.finfo(float).tiny)


def petermann_factors(model, k: ArrayLike) -> tuple[NDArray[np.complex128], NDArray[np.float64]]:
    r'''
    Get the eigenvalues of :math:`H(\mathbf{k})` and their Petermann factors

    .. math::

        K_n = \frac{\langle L_n|L_n\rangle\langle R_n|R_n\rangle}{|\langle L_n|R_n\rangle|^2}
        \geq 1\, ,

    with :math:`|R_n\rangle`, :math:`|L_n\rangle` the right and left
    eigenvectors: 1 for a normal (e.g. Hermitian) matrix, diverging at an
    exceptional point, where :math:`\langle L_n|R_n\rangle \to 0`
    (as :math:`|\mathbf{k}-\mathbf{k}_{EP}|^{-1}` near a second-order one).

    :param model: See *track_eigenvalues*.
    :param k: Tuple/list/ndarray of 2 real numbers.

    :returns:
        * **en** -- Complex ndarray, shape (norb,). Eigenvalues, sorted by real part.
        * **K** -- Real ndarray, shape (norb,). Petermann factors.
    '''
    ham = _model(model)
    error_handling.k_vector(k, 'k', 2)
    return _petermann(ham(np.asarray(k, dtype='f8')))


def _order(h: NDArray[np.complex128], k_min: float = 1e4) -> tuple[int, float, complex]:
    '''
    Private function. Order of the exceptional point at H: the number of
    eigenvalues whose Petermann factor exceeds sqrt(max K) (the coalescing
    ones, all of order max K), 1 if max K < k_min (no EP); with max K and
    the mean coalescing eigenvalue.
    '''
    w, kp = _petermann(h)
    k_max = float(np.max(kp))
    if k_max < k_min:
        return 1, k_max, complex(w[np.argmax(kp)])
    big = kp > np.sqrt(k_max)
    return int(np.sum(big)), k_max, complex(np.mean(w[big]))


class ExceptionalPoints():
    r'''
    Exceptional points of a 2D model, see *find_exceptional_points*.

    :ivar k: Real ndarray, shape (n, 2). Isolated exceptional points.
    :ivar charge: Integer ndarray, shape (n,). Discriminant winding around each
        one (see *discriminant_winding*): :math:`\pm1` for a second-order EP.
    :ivar order: Integer ndarray, shape (n,). Number of coalescing eigenvalues
        (the size of the Jordan block): 2, 3, ...
    :ivar petermann: Real ndarray, shape (n,). Largest Petermann factor at
        each point (divergent at an EP: about :math:`10^{8}` or more once refined).
    :ivar energy: Complex ndarray, shape (n,). The coalescing eigenvalue.
    :ivar lines: Real ndarray, shape (m, 2). Points on exceptional lines
        (rings), when a symmetry makes the discriminant real; empty otherwise.
    '''

    def __init__(self, k, charge, order, petermann, energy, lines) -> None:
        self.k = k
        self.charge = charge
        self.order = order
        self.petermann = petermann
        self.energy = energy
        self.lines = lines

    @property
    def total_charge(self) -> int:
        '''
        Sum of the charges: zero over a whole Brillouin zone (doubling theorem).
        '''
        return int(np.sum(self.charge))


def _mesh(ks: KSpace, nk: int):
    '''
    Private function. Shifted nk x nk mesh of the zone: node k-points,
    shape (nk + 1, nk + 1, 2) (the last row/column repeat the first ones
    shifted by a reciprocal vector), and the plaquette size.
    '''
    fr = (np.arange(nk + 1)[:, None] + np.array(_MESH_SHIFT)[None, :]) / nk
    fracs = np.stack(np.meshgrid(fr[:, 0], fr[:, 1], indexing='ij'), axis=-1)
    size = min(np.linalg.norm(b) for b in ks.rec_vec_k) / nk
    return fracs @ ks.rec_vec_k, size


def _edges(nk: int):
    '''
    Private function. Mesh edges ((i, j), (i', j')) along b1 then along b2.
    '''
    return ([((i, j), (i + 1, j)) for i in range(nk) for j in range(nk)]
                + [((i, j), (i, j + 1)) for i in range(nk) for j in range(nk)])


def _bisect(func, k_a, k_b, f_a, n_iter=60):
    '''
    Private function. Root of the real function *func* on the segment
    [k_a, k_b], across which it changes sign (f_a = func(k_a)).
    '''
    lo, hi = 0., 1.
    for _ in range(n_iter):
        mid = (lo + hi) / 2
        f_m = func(k_a + mid * (k_b - k_a))
        if np.sign(f_m) == np.sign(f_a):
            lo, f_a = mid, f_m
        else:
            hi = mid
    return k_a + (lo + hi) / 2 * (k_b - k_a)


def _newton(disc, k0, mult, h, tol, max_iter):
    r'''
    Private function. Newton's method on (Re Delta, Im Delta) = 0, with a
    central-difference Jacobian; the step is multiplied by the multiplicity
    *mult* of the zero (Delta ~ L(k)^mult, L complex linear, as at a
    higher-order exceptional point).
    '''
    k = np.array(k0, dtype='f8')
    for _ in range(max_iter):
        f = disc(k)
        jac = np.zeros((2, 2))
        for mu in range(2):
            e = np.zeros(2)
            e[mu] = h
            d = (disc(k + e) - disc(k - e)) / (2 * h)
            jac[:, mu] = d.real, d.imag
        step = mult * np.linalg.lstsq(jac, np.array([f.real, f.imag]), rcond=None)[0]
        k = k - step
        if np.linalg.norm(step) < tol:
            break
    return k


def find_exceptional_points(
    ks: KSpace, nk: int = 40, tol: float = 1e-12, max_iter: int = 50,
) -> ExceptionalPoints:
    r'''
    Locate the exceptional points of a 2D model over its Brillouin zone.

    1. The discriminant :math:`\Delta(\mathbf{k})` (see *discriminant*) is
       computed on an :math:`nk\times nk` mesh of the zone (shifted off the
       high-symmetry lines), and its winding around every plaquette from the
       phase changes along the mesh edges (bisected where :math:`\arg\Delta`
       turns by more than :math:`\pi/4`). A plaquette with a nonzero winding
       :math:`w` holds an EP.
    2. Each one is refined by Newton's method on
       :math:`(\mathrm{Re}\,\Delta, \mathrm{Im}\,\Delta) = 0` from the plaquette
       centre (the step multiplied by :math:`|w|`, the multiplicity of the zero).
    3. Its charge is the winding of :math:`\Delta` around a small circle
       (a quarter of a plaquette), and its order the number of eigenvalues
       whose Petermann factor diverges there (see *petermann_factors*).

    If a symmetry makes :math:`\Delta` real everywhere (up to a constant
    phase), EPs are not isolated: they form lines, exceptional rings, where
    :math:`\Delta` changes sign. The lines are then sampled where they cross
    the mesh edges (bisection to machine precision), and no isolated point is
    returned. A Hermitian model has :math:`\Delta \geq 0`, and no EP.

    Since :math:`\Delta` is periodic, the charges add up to zero
    (*total_charge*): EPs come in pairs of opposite charge. The mesh must be
    fine enough that no plaquette holds two EPs of opposite charge.

    :param ks: **KSpace** instance with two primitive vectors.
    :param nk: Positive integer. Default value 40. Mesh points along each
        reciprocal vector.
    :param tol: Positive real. Default value 1e-12. Newton's method stops
        when a step is shorter than *tol* (units of k).
    :param max_iter: Positive integer. Default value 50. Newton iterations.

    :returns:
        * **eps** -- **ExceptionalPoints** instance (positions *k*, *charge*,
          *order*, *petermann*, *energy*, and the exceptional *lines*).
    '''
    error_handling.kspace(ks, KSpace)
    error_handling.dim_exact(ks.dim, 2)
    ham = _model(ks)
    error_handling.positive_int(nk, 'nk')
    error_handling.positive_real(tol, 'tol')
    error_handling.positive_int(max_iter, 'max_iter')

    def disc(k):
        return _discriminant(ham(k))

    kmesh, size = _mesh(ks, nk)
    d = np.array([[disc(kmesh[i, j]) for j in range(nk)] for i in range(nk)])
    d = np.pad(d, ((0, 1), (0, 1)), mode='wrap')
    scale = float(np.max(np.abs(d)))
    error_handling.discriminant_nonzero(scale)
    phase = np.exp(-0.5j * np.angle(np.sum(d ** 2)))
    empty = np.zeros((0, 2))
    if np.max(np.abs((phase * d).imag)) <= 1e-8 * scale:
        # a real discriminant: exceptional lines where it changes sign
        lines = [_bisect(lambda k: (phase * disc(k)).real, kmesh[a], kmesh[b], (phase * d[a]).real)
                     for a, b in _edges(nk) if (phase * d[a]).real * (phase * d[b]).real < 0.]
        return ExceptionalPoints(empty, np.zeros(0, int), np.zeros(0, int), np.zeros(0),
                                                np.zeros(0, 'c16'), np.array(lines).reshape(-1, 2))
    # phase change along the edge leaving node (i, j) along b1 (axis 0) or b2
    # (axis 1); Delta is periodic, so the last row/column repeat the first ones
    edge = np.zeros((2, nk, nk))
    for a, b in _edges(nk):
        edge[int(b[1] != a[1])][a] = _arg_change(
            disc, lambda s, a=a, b=b: kmesh[a] + s * (kmesh[b] - kmesh[a]), 0., 1., d[a], d[b])
    b1, b2 = ks.rec_vec_k
    orient = np.sign(b1[0] * b2[1] - b1[1] * b2[0])
    points = []
    for i in range(nk):
        for j in range(nk):
            w = (edge[0, i, j] + edge[1, (i + 1) % nk, j]
                   - edge[0, i, (j + 1) % nk] - edge[1, i, j]) / (2 * PI)
            w = int(np.rint(orient * w))
            if w == 0:
                continue
            center = (kmesh[i, j] + kmesh[i + 1, j + 1]) / 2
            k_ep = _newton(disc, center, abs(w), 1e-5 * size, tol, max_iter)
            if all(np.linalg.norm(k_ep - p) > 1e-6 * size for p in points):
                points.append(k_ep)
    charge, order, peter, energy = [], [], [], []
    for p in points:
        charge.append(int(np.rint(_winding(ham, circle(p, 0.25 * size), 16))))
        o, kp, e = _order(ham(p))
        order.append(o)
        peter.append(kp)
        energy.append(e)
    return ExceptionalPoints(np.array(points).reshape(-1, 2), np.array(charge, int),
                                            np.array(order, int), np.array(peter),
                                            np.array(energy, 'c16'), empty)


def fermi_arcs(ks: KSpace, nk: int = 60) -> NDArray[np.float64]:
    r'''
    Get the bulk Fermi arcs of a 2D non-Hermitian model: the curves where
    two eigenvalues have equal real parts but different imaginary parts,
    :math:`\mathrm{Re}(E_m - E_n) = 0`, :math:`\mathrm{Im}(E_m - E_n) \neq 0`.
    Such an arc joins two exceptional points of opposite charge (for a
    two-band model it is where :math:`\Delta` is real and negative, see
    *discriminant*); it is what a Hermitian Dirac point turns into when a
    non-Hermitian term splits it into an EP pair (Kozii and Fu 2017; Zhou et
    al., Science 359, 1009 (2018)).

    Along every edge of an :math:`nk\times nk` mesh of the zone the
    eigenvalues are followed by continuation (see *track_eigenvalues*), and
    where :math:`\mathrm{Re}(E_m - E_n)` changes sign, the crossing is found by
    bisection. Regions where the real parts coincide over an area (inside an
    exceptional ring) show no sign change and are not reported, and neither
    are edges that cross an exceptional point or line, along which the
    eigenvalues cannot be followed.

    :param ks: **KSpace** instance with two primitive vectors.
    :param nk: Positive integer. Default value 60. Mesh points along each
        reciprocal vector.

    :returns:
        * **points** -- Real ndarray, shape (m, 2). Points on the arcs.
    '''
    error_handling.kspace(ks, KSpace)
    error_handling.dim_exact(ks.dim, 2)
    ham = _model(ks)
    error_handling.positive_int(nk, 'nk')
    kmesh, _ = _mesh(ks, nk)
    points = []
    for a, b in _edges(nk):
        k_a, k_b = kmesh[a], kmesh[b]
        try:
            s, en, _, _ = _track(ham, lambda x: k_a + x * (k_b - k_a), 1., 1)
        except ValueError:
            continue  # the edge crosses an exceptional point (or line)
        noise = 1e-10 * max(1., float(np.max(np.abs(en))))
        norb = en.shape[1]
        for m in range(norb):
            for n in range(m + 1, norb):
                g = (en[:, m] - en[:, n]).real
                for i in np.nonzero((g[:-1] * g[1:] < 0.) & (np.abs(g[:-1]) > noise)
                                                & (np.abs(g[1:]) > noise))[0]:
                    k_lo = k_a + s[i] * (k_b - k_a)
                    k_hi = k_a + s[i + 1] * (k_b - k_a)
                    k_arc = _arc_point(ham, k_lo, k_hi, en[i], m, n)
                    # not a degeneracy (an EP at the end of the arc)
                    e = LA.eigvals(ham(k_arc))
                    sep = np.abs(e[:, None] - e[None, :])
                    np.fill_diagonal(sep, np.inf)
                    if np.min(sep) > noise:
                        points.append(k_arc)
    return np.array(points).reshape(-1, 2)


def _arc_point(ham, k_lo, k_hi, w_lo, m, n, n_iter=50):
    '''
    Private function. Bisection for Re(E_m - E_n) = 0 between k_lo and k_hi,
    the labels carried along by matching to the eigenvalues at the lower end.
    '''
    g_lo = (w_lo[m] - w_lo[n]).real
    for _ in range(n_iter):
        k_mid = (k_lo + k_hi) / 2
        w = LA.eigvals(ham(k_mid))
        _, perm = linear_sum_assignment(np.abs(w_lo[:, None] - w[None, :]))
        w = w[perm]
        g = (w[m] - w[n]).real
        if np.sign(g) == np.sign(g_lo):
            k_lo, w_lo, g_lo = k_mid, w, g
        else:
            k_hi = k_mid
    return (k_lo + k_hi) / 2


#################################
# ENCIRCLING
#################################


class Encircling():
    r'''
    An eigenstate carried around a loop by parallel transport, see *encircle*.

    :ivar s: Real ndarray, shape (n,). Loop parameter, from 0 to *n_loops*.
    :ivar k: Real ndarray, shape (n, 2). Points of the loop.
    :ivar en: Complex ndarray, shape (n, norb). All eigenvalues, followed by
        continuation (see *track_eigenvalues*).
    :ivar vector: Complex ndarray, shape (n, norb). The transported right
        eigenvector :math:`|R(s)\rangle` of the followed state.
    :ivar band: Integer. Label of the state at the start.
    :ivar final_band: Integer. Label, at the start, of the eigenvalue the state
        ends on (different from *band* after an odd number of loops around a
        second-order EP: the states swap).
    :ivar overlap: Complex number. :math:`c = \langle L_{f}(0)|R(\mathrm{end})\rangle`,
        with :math:`f` = *final_band* and :math:`\langle L_f(0)|R_f(0)\rangle = 1`:
        the transported state is :math:`c\,|R_f(0)\rangle`. If *final_band* is
        *band*, :math:`c = e^{i\gamma}` (times a modulus that is 1 for a
        Hermitian model), :math:`\gamma` the geometric phase; otherwise its
        phase depends on the (arbitrary) phase of :math:`|R_f(0)\rangle`.
    '''

    def __init__(self, s, k, en, vector, band, final_band, overlap) -> None:
        self.s, self.k, self.en = s, k, en
        self.vector = vector
        self.band, self.final_band = band, final_band
        self.overlap = overlap

    @property
    def phase(self) -> float:
        r'''
        Accumulated phase :math:`\arg c` in :math:`(-\pi, \pi]` (the Berry
        phase when the state returns to itself).
        '''
        return float(np.angle(self.overlap))


def encircle(model, loop, band: int = 0, n_loops: int = 1, nk: int = 100) -> Encircling:
    r'''
    Carry an eigenstate around a closed loop *n_loops* times, by parallel
    transport of its right eigenvector :math:`|R\rangle`, with the left one
    :math:`\langle L|` normalized by :math:`\langle L|R\rangle = 1`:

    .. math::

        \langle L(\mathbf{k})|\partial_s R(\mathbf{k})\rangle = 0\, ,

    discretized symmetrically between successive points,
    :math:`\langle L_j + L_{j+1}|R_{j+1} - R_j\rangle = 0` (second order in
    the step; the eigenvalue is followed by continuation, see
    *track_eigenvalues*). For a Hermitian model :math:`L = R`, the norm is
    kept exactly, and this is the usual parallel transport: the state comes
    back as :math:`e^{i\gamma}|R\rangle`,
    :math:`\gamma = \oint i\langle u|\nabla_{\mathbf{k}}u\rangle\cdot d\mathbf{k}` the
    Berry phase (the convention of *KSpace.berry_phase*), :math:`\pi` around
    a diabolical point. Around a second-order exceptional point:

    * one loop swaps the two eigenstates (*final_band* differs from *band*);
    * two loops bring the state back with a sign flip, :math:`\gamma = \pi`;
    * four loops restore it.

    :param model: See *track_eigenvalues*.
    :param loop: See *track_eigenvalues*.
    :param band: Integer. Default value 0. Label of the state at the start of
        the loop (by real part of the eigenvalue).
    :param n_loops: Positive integer. Default value 1. Number of turns.
    :param nk: Positive integer. Default value 100. Steps per turn before
        refinement.

    :returns:
        * **result** -- **Encircling** instance (*en*, the transported
          *vector*, *final_band*, *overlap* and *phase*).
    '''
    ham = _model(model)
    path = _loop(loop)
    error_handling.positive_int(n_loops, 'n_loops')
    error_handling.positive_int(nk, 'nk')
    norb = _closed(ham, path)
    error_handling.band_index(band, norb)
    s, en, vr, vl = _track(ham, path, float(n_loops), nk, vectors=True)
    # biorthonormal pairs at the start: <L_b|R_b> = 1
    left0 = vl[0].conj().T / np.sum(vl[0].conj() * vr[0], axis=0)[:, None]
    vector = [vr[0][:, band]]
    for j in range(1, len(s)):
        # <L_j + L_j+1|R_j+1 - R_j> = 0 with <L|R> = 1 at both ends, for
        # R_j+1 = c R_raw: c^2 = b/a (the root closest to the first-order 1/a).
        # Second order in the step; |c| = 1 for a Hermitian model.
        bra = vl[j - 1][:, band].conj()
        bra = bra / (bra @ vector[-1])
        raw, raw_bra = vr[j][:, band], vl[j][:, band].conj()
        a = bra @ raw
        c = np.sqrt((raw_bra @ vector[-1]) / (raw_bra @ raw) / a)
        if abs(c - 1 / a) > abs(c + 1 / a):
            c = -c
        vector.append(c * raw)
    final = int(np.argmin(np.abs(en[0] - en[-1, band])))
    return Encircling(s, np.array([path(x % 1.) for x in s]), en, np.array(vector), band, final,
                             complex(left0[final] @ vector[-1]))
