r"""
Topological diagnostics beyond the Brillouin-zone integrals of
:class:`tbkit.kspace.KSpace`:

* **Bott index** (*bott_index*): the Chern number of a finite sample on a
  torus, from its projector on the occupied states and the exponentiated
  position operators (Loring and Hastings, EPL 92, 67004 (2010)). It needs
  no Brillouin zone, so it applies to disordered samples and quasicrystals,
  and complements the local Chern marker of
  :meth:`tbkit.system.System.get_local_chern_marker`.
* **Entanglement spectrum** (*entanglement_spectrum*): the eigenvalues of
  the correlation matrix of the occupied states restricted to a region of
  a finite sample (Peschel 2003; Li and Haldane 2008). Its k-resolved
  version for ribbons and supercells is
  :meth:`tbkit.kspace.KSpace.entanglement_spectrum`.
* **Weyl points** (*find_weyl_points*): the band touchings of a 3D model
  and their chiralities, the Berry flux through a small sphere around each
  one (Wan, Turner, Vishwanath and Savrasov, Phys. Rev. B 83, 205101
  (2011)) -- the Hermitian counterpart of
  :func:`tbkit.exceptional.find_exceptional_points`.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
import scipy.linalg as LA
import scipy.sparse as sparse

import tbkit.error_handling as error_handling
from tbkit.kspace import KSpace


PI = np.pi
_MESH_SHIFT = (0.3183098861837907, 0.2718281828459045, 0.1414213562373095)  # off symmetry planes


def _occupied(ham, e_fermi: float) -> NDArray[np.complex128]:
    '''
    Private function. Validate a Hermitian Hamiltonian (dense or sparse)
    and return its eigenvectors below *e_fermi* (columns).
    '''
    h = sparse.csr_matrix(ham, dtype='c16')
    error_handling.square_matrix(h, 'ham')
    error_handling.hermitian(h)
    error_handling.real_number(e_fermi, 'e_fermi')
    en, vec = LA.eigh(h.toarray())
    return vec[:, en < e_fermi]


def bott_index(ham, positions: ArrayLike, cell: ArrayLike, e_fermi: float = 0.) -> float:
    r'''
    Get the Bott index of the states below *e_fermi* of a finite sample with
    periodic boundary conditions (a torus spanned by :math:`\mathbf{L}_1,
    \mathbf{L}_2`; Loring and Hastings, EPL 92, 67004 (2010)):

    .. math::

        B = \frac{1}{2\pi}\,\mathrm{Im}\,\mathrm{Tr}\log\left(
        \tilde V\tilde U\tilde V^\dagger\tilde U^\dagger\right)\, ,\qquad
        \tilde U = P e^{2\pi i X_1} P\, ,\quad \tilde V = P e^{2\pi i X_2} P\, ,

    with :math:`P` the projector on the occupied states and
    :math:`X_{1,2}` the fractional coordinates of the orbitals along
    :math:`\mathbf{L}_{1,2}`. The exponentials are well defined on the torus,
    where the position operator is not. :math:`B` is an integer whenever the
    occupied states are localized or gapped, equals the Chern number of a
    clean sample (in the sign convention of *KSpace.chern_number*), and
    keeps working with disorder until the mobility gap closes. The torus
    comes from *KSpace.finite_ham* with ``periodic=True`` and the positions
    from *tbkit.higher_order.flake_positions*.

    :param ham: Square Hermitian matrix (dense or sparse), shape (N, N).
    :param positions: Real array, shape (N, d). Position of the orbital of each row.
    :param cell: Real array, shape (2, d). The two vectors
        :math:`\mathbf{L}_1, \mathbf{L}_2` spanning the torus (e.g.
        :math:`N_1\mathbf{a}_1` and :math:`N_2\mathbf{a}_2`).
    :param e_fermi: Real number. Default value 0. Fermi energy.

    :returns:
        * **bott** -- Real number, close to an integer.

    Example usage::

        n = 12
        ham = ks.finite_ham((n, n), periodic=True)
        pos = flake_positions(ks, (n, n))
        cell = [n * np.array(a) for a in ks.lat.prim_vec]
        bott = bott_index(ham, pos, cell)
    '''
    occ = _occupied(ham, e_fermi)
    positions = np.asarray(positions, dtype='f8')
    cell = np.asarray(cell, dtype='f8')
    error_handling.site_positions(positions, len(occ))
    error_handling.torus_cell(cell, positions.shape[1])
    fracs = positions @ np.linalg.pinv(cell)
    u = occ.conj().T @ (np.exp(2j * PI * fracs[:, 0])[:, None] * occ)
    v = occ.conj().T @ (np.exp(2j * PI * fracs[:, 1])[:, None] * occ)
    w = v @ u @ v.conj().T @ u.conj().T
    return float(np.sum(np.angle(np.linalg.eigvals(w))) / (2 * PI))


def entanglement_spectrum(ham, region: list[int], e_fermi: float = 0.) -> NDArray[np.float64]:
    r'''
    Get the entanglement spectrum of the states below *e_fermi* of a finite
    sample: the eigenvalues :math:`\xi_n \in [0, 1]` of the correlation
    matrix :math:`C_{ij} = \langle c_i^\dagger c_j\rangle` restricted to the
    orbitals :math:`i, j` of *region* (Peschel, J. Phys. A 36, L205 (2003)).
    The entanglement Hamiltonian of the region is quadratic, with
    single-particle energies :math:`\epsilon_n = \ln[(1-\xi_n)/\xi_n]`, and
    the entanglement entropy is
    :math:`S = -\sum_n[\xi_n\ln\xi_n + (1-\xi_n)\ln(1-\xi_n)]`.
    Its low-lying part mirrors the edge spectrum at the cut (Li and Haldane,
    Phys. Rev. Lett. 101, 010504 (2008)): a topological phase leaves
    entanglement modes inside (0, 1), e.g. a mode at exactly 1/2 at each cut
    of a chiral-symmetric chain -- without any physical edge, so it is a
    property of the ground state alone. See
    *KSpace.entanglement_spectrum* for the k-resolved version.

    :param ham: Square Hermitian matrix (dense or sparse), shape (N, N).
    :param region: List of row indices (the region :math:`A`).
    :param e_fermi: Real number. Default value 0. Fermi energy.

    :returns:
        * **xi** -- Real ndarray, shape (len(region),), sorted ascending.
    '''
    occ = _occupied(ham, e_fermi)
    error_handling.region(region, len(occ))
    occ = occ[region]
    return np.linalg.eigvalsh(occ @ occ.conj().T)


class WeylPoints():
    r'''
    Weyl points of a 3D model, see *find_weyl_points*.

    :ivar k: Real ndarray, shape (n, 3). Positions, in the k coordinates of
        *KSpace.get_ham*, folded to fractional coordinates in [-1/2, 1/2).
    :ivar chirality: Integer ndarray, shape (n,). Berry flux of the bands
        below the touching out of a small sphere around each point, over
        :math:`2\pi`: :math:`\pm1` for a simple Weyl point.
    :ivar energy: Real ndarray, shape (n,). Energy of the touching.
    :ivar gap: Real ndarray, shape (n,). Remaining gap at the refined point
        (zero up to the tolerance).
    '''

    def __init__(self, k, chirality, energy, gap) -> None:
        self.k = k
        self.chirality = chirality
        self.energy = energy
        self.gap = gap

    @property
    def total_chirality(self) -> int:
        '''
        Sum of the chiralities: zero over a whole Brillouin zone
        (Nielsen-Ninomiya doubling theorem).
        '''
        return int(np.sum(self.chirality))


def _gap(ks: KSpace, kpts: NDArray[np.float64], n: int) -> NDArray[np.float64]:
    '''
    Private function. Gap E_{n+1} - E_n at the k-points *kpts* (shape (m, 3)).
    '''
    en = ks._eigs(kpts)
    return en[:, n + 1] - en[:, n]


def _newton(ks: KSpace, k0: NDArray[np.float64], n: int, h: float, tol: float, max_iter: int):
    r'''
    Private function. Newton's method on the touching of bands n and n+1:
    at the current point, the two bands span a 2D subspace :math:`W`, and
    :math:`W^\dagger H(\mathbf{k})W - \mathrm{tr}/2 = \mathbf{d}\cdot\boldsymbol\sigma`
    near it; the step solves :math:`\mathbf{d} + J\delta\mathbf{k} = 0`
    (central-difference Jacobian). It returns the point and its gap.
    '''
    sigma = np.array([[[0., 1.], [1., 0.]], [[0., -1j], [1j, 0.]], [[1., 0.], [0., -1.]]])
    k = np.array(k0, dtype='f8')
    for _ in range(max_iter):
        _, vec = ks._eigs(k[None], eigenvec=True)
        w = vec[0][:, n:n + 2]

        def d(kk):
            pts = np.atleast_2d(kk)
            heff = w.conj().T @ ks._hams(pts) @ w
            return np.einsum('aij,kji->ka', sigma, heff).real / 2
        shifts = h * np.eye(3)
        dd = d(np.concatenate([k + shifts, k - shifts]))
        jac = ((dd[:3] - dd[3:]) / (2 * h)).T
        step = np.linalg.lstsq(jac, -d(k)[0], rcond=None)[0]
        k = k + step
        if np.linalg.norm(step) < tol:
            break
    return k, float(_gap(ks, k[None], n)[0])


def _sphere_chirality(ks: KSpace, center: NDArray[np.float64], radius: float,
                                bands: list[int], n_theta: int = 16) -> int:
    r'''
    Private function. Berry flux of *bands* out of the sphere of *radius*
    around *center*, over :math:`2\pi`: the Fukui-Hatsugai-Suzuki
    plaquettes of a (theta, phi) mesh, poles included, oriented outward
    (the sign convention of *KSpace.berry_curvature*).
    '''
    n_phi = 2 * n_theta
    theta = np.linspace(0., PI, n_theta + 1)
    phi = 2 * PI * np.arange(n_phi) / n_phi
    tt, pp = np.meshgrid(theta, phi, indexing='ij')
    dirs = np.stack([np.sin(tt) * np.cos(pp), np.sin(tt) * np.sin(pp), np.cos(tt)], axis=-1)
    pts = center + radius * dirs.reshape(-1, 3)
    v1 = ks._subspaces(pts, bands).reshape(n_theta + 1, n_phi, ks.norb, len(bands))
    # each pole is one point: use the same basis for all its phi
    v1[0] = v1[0, 0]
    v1[-1] = v1[-1, 0]

    def overlap(a, b):
        return np.linalg.det(a.conj().swapaxes(-1, -2) @ b)
    a, b = v1[:-1], v1[1:]
    a_next, b_next = np.roll(a, -1, axis=1), np.roll(b, -1, axis=1)
    # corners (theta, phi) -> (theta + d, phi) -> (theta + d, phi + d) -> (theta, phi + d):
    # counterclockwise about e_theta x e_phi = e_r, the outward normal
    link = overlap(a, b) * overlap(b, b_next) * overlap(b_next, a_next) * overlap(a_next, a)
    return int(np.rint(-np.angle(link).sum() / (2 * PI)))


def find_weyl_points(
    ks: KSpace, bands: int | list[int], nk: int = 20, radius: float | None = None,
    tol: float = 1e-10, max_iter: int = 50,
) -> WeylPoints:
    r'''
    Locate the Weyl points of a 3D model: the points where the highest band
    of *bands* (band :math:`n`) touches the next one, and their chirality.

    1. The gap :math:`E_{n+1} - E_n` is computed on an :math:`nk^3` mesh of
       the Brillouin zone (shifted off the high-symmetry planes); its local
       minima (over the 26 neighbours, periodically) are the candidates.
    2. Each candidate is refined by Newton's method on the two-band
       Hamiltonian :math:`\mathbf{d}(\mathbf{k})\cdot\boldsymbol\sigma` of bands
       :math:`n, n+1`, whose zeros are the touchings; candidates whose gap
       does not close (avoided crossings) are dropped, and points equal up
       to a reciprocal lattice vector merged.
    3. The chirality of each point is the Berry flux of *bands* out of a
       small sphere around it, divided by :math:`2\pi` -- the monopole
       charge, :math:`\pm1` for a linear Weyl point. With the sign
       convention of *KSpace.berry_curvature*, the Chern number of the
       planes :math:`k_l = \mathrm{const}` (*KSpace.chern_number*) jumps by
       the chirality as the plane crosses the point along
       :math:`+\mathbf{b}_l`.

    The chiralities add up to zero (*total_chirality*): Weyl points come in
    pairs of opposite chirality (Nielsen-Ninomiya). The mesh must resolve
    the gap minima: on too coarse a mesh, two candidates may refine to the
    same point and miss another one, which a nonzero *total_chirality*
    reveals (increase *nk*).
    Nodal lines (degeneracies along curves) are not isolated points and are
    outside the scope of this finder.

    :param ks: **KSpace** instance with three primitive vectors (Hermitian).
    :param bands: Band index, or list of band indices (e.g. the occupied
        bands): the touching is between ``max(bands)`` and ``max(bands) + 1``.
    :param nk: Positive integer. Default value 20. Mesh points along each
        reciprocal vector.
    :param radius: Positive real. Default value None: a fifth of the mesh
        step. Radius of the sphere for the chirality (in units of k); it
        must enclose no other Weyl point.
    :param tol: Positive real. Default value 1e-10. Newton's method stops
        when a step is shorter than *tol*; a point is kept if its gap is
        below :math:`10^4` *tol* times the bandwidth.
    :param max_iter: Positive integer. Default value 50. Newton iterations.

    :returns:
        * **weyl** -- **WeylPoints** instance (*k*, *chirality*, *energy*, *gap*).
    '''
    error_handling.kspace(ks, KSpace)
    error_handling.dim_exact(ks.dim, 3)
    error_handling.hermitian_kspace(ks.is_hermitian(), ks._overlap_hop)
    if isinstance(bands, int):
        bands = [bands]
    error_handling.band_indices(bands, ks.norb)
    n = max(bands)
    error_handling.band_below_top(n, ks.norb)
    error_handling.positive_int(nk, 'nk')
    step = min(np.linalg.norm(b) for b in ks.rec_vec_k) / nk
    if radius is None:
        radius = step / 5
    error_handling.positive_real(radius, 'radius')
    error_handling.positive_real(tol, 'tol')
    error_handling.positive_int(max_iter, 'max_iter')
    fr = (np.arange(nk)[:, None] + np.array(_MESH_SHIFT)[None, :]) / nk
    fracs = np.stack(np.meshgrid(fr[:, 0], fr[:, 1], fr[:, 2], indexing='ij'), axis=-1).reshape(-1, 3)
    kmesh = fracs @ ks.rec_vec_k
    en = ks._eigs(kmesh)
    scale = float(en.max() - en.min()) or 1.
    gap = (en[:, n + 1] - en[:, n]).reshape(nk, nk, nk)
    is_min = np.ones_like(gap, bool)
    for shift in np.ndindex(3, 3, 3):
        if shift != (1, 1, 1):
            is_min &= gap <= np.roll(gap, np.array(shift) - 1, axis=(0, 1, 2))
    inv = np.linalg.inv(ks.rec_vec_k)
    points: list = []
    gaps: list = []
    for idx in np.argwhere(is_min):
        k, g = _newton(ks, kmesh[np.ravel_multi_index(idx, gap.shape)], n,
                                1e-5 * step, tol, max_iter)
        if not np.isfinite(g) or g > 1e4 * tol * scale:
            continue
        # fold into the zone centred on Gamma, and merge points equal up to
        # a reciprocal vector
        frac = (k @ inv + 0.5) % 1. - 0.5
        if any(np.linalg.norm((frac - p + 0.5) % 1. - 0.5) < 1e-6 for p in points):
            continue
        points.append(frac)
        gaps.append(g)
    k_w = np.array(points).reshape(-1, 3) @ ks.rec_vec_k
    chirality = np.array([_sphere_chirality(ks, k, radius, bands) for k in k_w], int)
    energy = np.array([ks._eigs(k[None])[0, n] for k in k_w])
    return WeylPoints(k_w, chirality, energy.reshape(-1), np.array(gaps))
