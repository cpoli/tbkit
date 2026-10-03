r"""
Wannier functions of a group of bands of a :class:`tbkit.kspace.KSpace`
model, built in two steps (three for entangled bands):

* **Disentanglement** (when there are more bands than Wannier functions):
  at each k, the :math:`n_W`-dimensional subspace of the states in an outer
  energy window that is smoothest across the mesh, i.e. that minimizes
  :math:`\Omega_I`, keeping the states of an optional frozen window exactly
  (Souza, Marzari and Vanderbilt, Phys. Rev. B 65, 035109 (2001)).
* **Projection** onto trial orbitals :math:`|g_n\rangle` localized in the
  home cell: :math:`A_{mn}(\mathbf{k}) = \langle u_{m\mathbf{k}}|g_n\rangle`,
  made unitary by Lowdin orthogonalization,
  :math:`U(\mathbf{k}) = A(A^\dagger A)^{-1/2}` (Marzari, Mostofi, Yates,
  Souza and Vanderbilt, Rev. Mod. Phys. 84, 1419 (2012), Sec. III.A).
* **Maximal localization**: conjugate-gradient descent of the spread
  :math:`\Omega = \sum_n \langle r^2\rangle_n - |\bar{\mathbf{r}}_n|^2`
  over the gauge :math:`U(\mathbf{k})`, from the overlaps
  :math:`M^{(\mathbf{k},\mathbf{b})}_{mn} = \langle u_{m\mathbf{k}}|u_{n\mathbf{k}+\mathbf{b}}\rangle`
  between neighbouring points of the k-mesh (Marzari and Vanderbilt, Phys.
  Rev. B 56, 12847 (1997)).

The Wannier functions are the Fourier transforms of the rotated Bloch
states, on the :math:`N_1\times N_2\times\dots` cells of the k-mesh (the
supercell on which they are periodic). Their Hamiltonian
:math:`H_{mn}(\mathbf{R}) = \langle W_{m\mathbf{0}}|H|W_{n\mathbf{R}}\rangle`
is a tight-binding model of its own (*WannierFunctions.kspace*), which
interpolates the bands between the mesh points (Wannier interpolation).

A band group with a nonzero Chern number has no exponentially localized
Wannier functions (Thouless, J. Phys. C 17, L325 (1984); Brouder et al.,
Phys. Rev. Lett. 98, 046402 (2007)): any projection vanishes somewhere in
the Brillouin zone, and the spread grows with the mesh instead of
converging.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

import tbkit.error_handling as error_handling
from tbkit.bridges import _finite_sites, finite_system
from tbkit.io import _hr_kspace
from tbkit.kspace import KSpace


class WannierFunctions():
    r'''
    Wannier functions of a group of bands, see *wannierize*.

    :ivar functions: Complex ndarray, shape (n_wann, N*norb). Amplitudes
        of each Wannier function on the orbitals of the
        :math:`N = N_1N_2\dots` cells of the k-mesh, indexed as the rows of
        *KSpace.finite_ham(nk)* (and the sites of *system*). Each one is a
        lattice translate chosen to sit in the middle of the sample.
    :ivar positions: Real ndarray, shape (N*norb, space_dim). Position of each row.
    :ivar centers: Real ndarray, shape (n_wann, space_dim). Centres
        :math:`\bar{\mathbf{r}}_n` of *functions* (Cartesian).
    :ivar spreads: Real ndarray, shape (n_wann,). Spreads
        :math:`\langle r^2\rangle_n - |\bar{\mathbf{r}}_n|^2` (squared
        length), along the primitive vectors only for a chain or a 2D
        lattice in 3D.
    :ivar omega_i: Real number. Gauge-invariant part :math:`\Omega_I` of the total spread.
    :ivar omega_d: Real number. Diagonal part :math:`\tilde\Omega_D`.
    :ivar omega_od: Real number. Off-diagonal part :math:`\tilde\Omega_{OD}`.
    :ivar gauge: Complex ndarray, shape (N, len(bands), n_wann).
        :math:`U(\mathbf{k})` over the k-mesh (flattened as *KSpace.mesh_grid*),
        acting on the eigenvectors of *bands*: unitary for an isolated
        group, otherwise the disentangled subspace (orthonormal columns,
        zero on the states outside the window) times the unitary gauge.
    :ivar history: Real ndarray. Total spread after the projection and after each step.
    :ivar converged: Boolean. Whether the spread changed by less than *tol*.
    :ivar min_singular_value: Real number. Smallest singular value of the
        projection :math:`A(\mathbf{k})` over the mesh: close to zero when the
        trial orbitals miss the bands somewhere (as they must for a Chern band).
    :ivar dis_history: Real ndarray. :math:`\Omega_I` after the projection
        and after each disentanglement step (empty for an isolated group).
    :ivar dis_converged: Boolean. Whether :math:`\Omega_I` changed by
        less than *dis_tol* (True for an isolated group).
    :ivar ham_k: Complex ndarray, shape (N, n_wann, n_wann). Hamiltonian
        :math:`U^\dagger(\mathbf{k})\,\mathrm{diag}(E)\,U(\mathbf{k})`
        in the gauge of the Wannier functions, over the k-mesh.
    '''

    def __init__(self, ks, n_cells, functions, positions, centers, spreads, omega_i,
                      omega_d, omega_od, gauge, history, converged, min_singular_value,
                      dis_history=None, dis_converged=True, ham_k=None, shift=None) -> None:
        self._ks = ks
        self.n_cells = n_cells
        self.functions = functions
        self.positions = positions
        self.centers = centers
        self.spreads = spreads
        self.omega_i = omega_i
        self.omega_d = omega_d
        self.omega_od = omega_od
        self.gauge = gauge
        self.history = history
        self.converged = converged
        self.min_singular_value = min_singular_value
        self.dis_history = np.array([]) if dis_history is None else dis_history
        self.dis_converged = dis_converged
        self.ham_k = ham_k
        self._shift = shift  # lattice vectors (integers) moving each function to the middle cell
        self._system = None

    @property
    def omega(self) -> float:
        r'''
        Total spread :math:`\Omega = \Omega_I + \tilde\Omega_D + \tilde\Omega_{OD}`,
        the sum of *spreads*.
        '''
        return float(np.sum(self.spreads))

    @property
    def system(self):
        '''
        The finite *System* (open boundaries) whose sites carry *functions*:
        *bridges.finite_system(ks, nk)*, built on first access. Spinless
        models only (for a spinful one, use *positions*).
        '''
        if self._system is None:
            self._system = finite_system(self._ks, self.n_cells)
        return self._system

    def kspace(self, cutoff: float = 1e-10) -> KSpace:
        r'''
        The tight-binding model of the Wannier functions (Wannier
        interpolation): one orbital per function, at its centre (folded
        into the home cell), and the matrix elements

        .. math::

            H_{mn}(\mathbf{R}) = \langle W_{m\mathbf{0}}|H|W_{n\mathbf{R}}\rangle
            = \frac{1}{N}\sum_{\mathbf{k}} e^{-i\mathbf{k}\cdot\mathbf{R}}\,
            [U^\dagger(\mathbf{k})\,\mathrm{diag}(E)\,U(\mathbf{k})]_{mn}

        as onsite energies and hoppings. Each :math:`\mathbf{R}` is the
        image on the :math:`N_1\times N_2\times\dots` supercell closest to
        the bond, :math:`|\mathbf{R} + \boldsymbol\tau_n - \boldsymbol\tau_m|`
        minimal (equidistant images share the element, as
        Wannier90's ``use_ws_distance``). Its bands equal the Wannierized
        ones on the k-mesh: all of them for an isolated group, those in the
        frozen window after disentanglement. Between the mesh points they
        converge exponentially with the mesh, as the functions decay. The
        model is the one *io.read_wannier90* would read from Wannier90's
        ``seedname_hr.dat``.

        :param cutoff: Positive real number or zero. Default value 1e-10.
            Drop the hoppings with :math:`|t| \le` *cutoff*.

        :returns:
            * **ks** -- **KSpace** instance, Hermitian, with n_wann orbitals.

        Example usage::

            wf = wannierize(ks, list(range(ks.norb)), trial=[0, 1], frozen=(-3., 0.))
            wk = wf.kspace()
            dist, en = wk.k_path(high_symmetry_path(ks.lat)[0], 50)
        '''
        error_handling.positive_real_zero(cutoff, 'cutoff')
        ks, nk = self._ks, tuple(self.n_cells)
        dim, n_k, n_w = ks.dim, int(np.prod(nk)), len(self.spreads)
        prim = np.array(ks.lat.prim_vec, dtype='f8')
        # orbital m is W_m(T_m), the translate of the Fourier transform W_m(0)
        # (the functions before their move to the middle cell) in the home cell
        centre = self.centers - self._shift @ prim
        cell = np.floor(centre @ np.array(ks.rec_vec, dtype='f8').T / (2 * np.pi) + 1e-8).astype(int)
        tau = centre - cell @ prim
        # H'_mn(R) = H_mn(R + T_n - T_m): a phase on the mesh, then the Fourier transform
        fracs = np.array(np.meshgrid(*[np.arange(n) / n for n in nk], indexing='ij')).reshape(dim, -1).T
        dt = cell[None, :, :] - cell[:, None, :]  # T_n - T_m, shape (n_w, n_w, dim)
        ham = self.ham_k * np.exp(-2j * np.pi * np.einsum('ka,mna->kmn', fracs, dt))
        ham_r = np.fft.fftn(ham.reshape(nk + (n_w, n_w)), axes=tuple(range(dim))) / n_k
        # the image R = j + N s closest to each bond, j centred on 0
        j = np.array(np.meshgrid(*[(np.arange(n) + n // 2) % n - n // 2 for n in nk],
                                       indexing='ij')).reshape(dim, -1).T
        s = np.array(np.meshgrid(*[np.arange(-1, 2)] * dim, indexing='ij')).reshape(dim, -1).T
        R = (j[:, None, :] + s[None, :, :] * np.array(nk)).reshape(-1, dim)  # (n_k * 3^dim, dim)
        bond = (R @ prim)[:, None, None, :] + tau[None, None, :, :] - tau[None, :, None, :]
        dist = np.linalg.norm(bond, axis=3).reshape(n_k, len(s), n_w, n_w)
        closest = dist <= dist.min(axis=1, keepdims=True) + 1e-6 * max(1., float(np.abs(prim).max()))
        weight = closest / np.sum(closest, axis=1, keepdims=True)
        ham_r = (ham_r.reshape(n_k, 1, n_w, n_w) * weight).reshape(-1, n_w, n_w)
        keep = np.any(weight.reshape(-1, n_w, n_w) > 0., axis=(1, 2))
        return _hr_kspace(ks.lat.prim_vec, tau, ['a'] * n_w, R[keep], ham_r[keep], cutoff, 1e-8)


def _shells(rec: NDArray[np.float64], nk: tuple[int, ...]) -> tuple[NDArray, NDArray, NDArray]:
    r'''
    Private function. Neighbours :math:`\mathbf{b}` of a k-point on the mesh
    and their weights :math:`w_b`, so that
    :math:`\sum_b w_b b_\alpha b_\beta = \delta_{\alpha\beta}` (the
    finite-difference condition of Marzari and Vanderbilt 1997, Eq. B1).
    Shells of equal :math:`|\mathbf{b}|` are added by increasing length,
    skipping those that add no independent direction, until the condition
    is solvable (as Wannier90 does).

    :returns: steps (integers, shape (nb, dim)), b (shape (nb, dim)), w (shape (nb,)).
    '''
    dim = len(nk)
    steps = np.array(np.meshgrid(*[np.arange(-1, 2)] * dim, indexing='ij')).reshape(dim, -1).T
    steps = steps[np.any(steps != 0, axis=1)]
    b = (steps / np.array(nk)[None, :]) @ rec
    length = np.linalg.norm(b, axis=1)
    radii = np.unique(np.round(length / length.min(), 8))
    upper = np.triu_indices(dim)
    target = np.eye(dim)[upper]
    chosen, columns = [], []
    for r in radii:
        shell = np.abs(length / length.min() - r) < 1e-6
        column = np.einsum('ba,bc->ac', b[shell], b[shell])[upper]
        trial = np.array(columns + [column]).T
        if np.linalg.matrix_rank(trial, tol=1e-10) < len(columns) + 1:
            continue
        chosen.append(shell)
        columns.append(column)
        w, *_ = np.linalg.lstsq(trial, target, rcond=None)
        # always reached: the steps e_i and e_i + e_j give dim(dim+1)/2
        # independent columns, and the system is then square and invertible
        if np.allclose(trial @ w, target, atol=1e-8):
            break
    keep = np.concatenate([np.flatnonzero(s) for s in chosen])
    weights = np.concatenate([np.full(np.sum(s), ws) for s, ws in zip(chosen, w)])
    return steps[keep], b[keep], weights


def _spread(m, b, w, n_k):
    r'''
    Private function. Centres (shape (n_wann, dim), k coordinates), spreads
    and the parts :math:`\Omega_I, \tilde\Omega_D, \tilde\Omega_{OD}` from
    the overlaps *m* (shape (nk, nb, n_wann, n_wann)).
    '''
    diag = np.diagonal(m, axis1=2, axis2=3)  # (nk, nb, n_wann)
    phase = np.angle(diag)
    centers = -np.einsum('b,ba,kbn->na', w, b, phase) / n_k
    r2 = np.einsum('b,kbn->n', w, 1. - np.abs(diag) ** 2 + phase ** 2) / n_k
    spreads = r2 - np.sum(centers ** 2, axis=1)
    total = np.einsum('b,kbmn->', w, np.abs(m) ** 2) / n_k
    on_diag = np.einsum('b,kbn->', w, np.abs(diag) ** 2) / n_k
    omega_i = float(np.sum(w) * m.shape[2] - total)
    omega_od = float(total - on_diag)
    omega_d = float(np.einsum('b,kbn->', w, (phase + (b @ centers.T)[None]) ** 2) / n_k)
    return centers, spreads, omega_i, omega_d, omega_od


def _gradient(m, b, w, centers):
    r'''
    Private function. Gradient :math:`G(\mathbf{k})` of the spread for a
    gauge change :math:`U \to U e^{dW}` (Marzari and Vanderbilt 1997,
    Eq. 52), :math:`G = 4\sum_b w_b(\mathcal{A}[R] - \mathcal{S}[T])`:
    :math:`\Omega` decreases along :math:`dW = \epsilon G`, :math:`\epsilon > 0`,
    with :math:`R_{mn} = M_{mn}M_{nn}^*`, :math:`T_{mn} = (M_{mn}/M_{nn})q_n`,
    :math:`q_n = \mathrm{Im}\ln M_{nn} + \mathbf{b}\cdot\bar{\mathbf{r}}_n`,
    :math:`\mathcal{A}[B] = (B - B^\dagger)/2`, :math:`\mathcal{S}[B] = (B + B^\dagger)/2i`.
    Shape (nk, n_wann, n_wann), anti-Hermitian.
    '''
    diag = np.diagonal(m, axis1=2, axis2=3)
    r = m * diag.conj()[:, :, None, :]
    q = np.angle(diag) + (b @ centers.T)[None]
    t = m / diag[:, :, None, :] * q[:, :, None, :]
    a_r = (r - r.conj().transpose(0, 1, 3, 2)) / 2
    s_t = (t + t.conj().transpose(0, 1, 3, 2)) / 2j
    return 4 * np.einsum('b,kbmn->kmn', w, a_r - s_t)


def _unitary_step(g: NDArray[np.complex128], eps: float) -> NDArray[np.complex128]:
    '''
    Private function. exp(eps G) for anti-Hermitian matrices G (stacked).
    '''
    h = 1j * g  # Hermitian
    lam, vec = np.linalg.eigh((h + h.conj().transpose(0, 2, 1)) / 2)
    return (vec * np.exp(-1j * eps * lam)[:, None, :]) @ vec.conj().transpose(0, 2, 1)


def _disentangle(m0, neighbour, w, inside, frozen, a, n_w, max_iter, tol, mixing):
    r'''
    Private function. Souza-Marzari-Vanderbilt subspace selection. At each
    k, the :math:`n_W` states :math:`U(\mathbf{k})` (columns, in the basis
    of the candidate bands) are the frozen ones plus the leading
    eigenvectors of

    .. math::

        Z(\mathbf{k}) = \sum_b w_b\, M^{(0)}_{\mathbf{k},\mathbf{b}}\,
        P(\mathbf{k}+\mathbf{b})\, M^{(0)\dagger}_{\mathbf{k},\mathbf{b}}

    restricted to the free states of the window (*inside* and not *frozen*),
    :math:`P = UU^\dagger` from the previous step, with linear mixing of
    :math:`Z` (Eqs. 21-25 of SMV 2001). Each step lowers
    :math:`\Omega_I = \sum_b w_b(n_W - \langle\|U_\mathbf{k}^\dagger M^{(0)}U_{\mathbf{k}+\mathbf{b}}\|^2\rangle_k)`.
    The first subspace is the projection of the trial orbitals *a*
    (overlaps :math:`\langle u_m|g_n\rangle`, shape (nk, N, n_w)).

    :returns: U (nk, N, n_w), the :math:`\Omega_I` history, converged.
    '''
    n_k, n = inside.shape
    free = inside & ~frozen
    # the eigenvalues of Z lie in [0, sum w]: frozen states above, excluded below
    big = 2 * np.sum(w) + 1
    level = np.where(frozen, big, np.where(inside, 0., -big))
    diag = np.arange(n)

    def select(z):
        z = z * (free[:, :, None] & free[:, None, :])
        z[:, diag, diag] += level
        return np.linalg.eigh(z)[1][:, :, -n_w:]

    def z_matrix(u):
        mu = m0 @ u[neighbour]  # (nk, nb, N, n_w)
        return np.einsum('b,kbmi,kbni->kmn', w, mu, mu.conj())

    u = select(a @ a.conj().transpose(0, 2, 1))
    z = z_matrix(u)
    history = [float(n_w * np.sum(w) - np.einsum('kmi,kmn,kni->', u.conj(), z, u).real / n_k)]
    converged = max_iter == 0
    z_in = z
    for _ in range(max_iter):
        u = select(z_in)
        z = z_matrix(u)
        history.append(float(n_w * np.sum(w) - np.einsum('kmi,kmn,kni->', u.conj(), z, u).real / n_k))
        if abs(history[-2] - history[-1]) < tol:
            converged = True
            break
        z_in = mixing * z + (1 - mixing) * z_in
    return u, np.array(history), converged


def wannierize(
    ks: KSpace, bands: int | list[int], trial: list[int] | ArrayLike,
    nk: int | tuple[int, ...] = 12, max_iter: int = 500, tol: float = 1e-10,
    step: float = 0.5, window: tuple[float, float] | None = None,
    frozen: tuple[float, float] | None = None, dis_iter: int = 2000, dis_tol: float = 1e-10,
    mixing: float = 0.5,
) -> WannierFunctions:
    r'''
    Build the maximally localized Wannier functions of a group of bands:
    project the Bloch states onto trial orbitals (Lowdin orthogonalized),
    then minimize the spread

    .. math::

        \Omega = \sum_n \left[\langle r^2\rangle_n - |\bar{\mathbf{r}}_n|^2\right]
        = \Omega_I + \tilde\Omega_D + \tilde\Omega_{OD}

    by conjugate gradients over the gauge :math:`U(\mathbf{k})` (Marzari
    and Vanderbilt, Phys. Rev. B 56, 12847 (1997); as in Wannier90, with a
    parabolic line search). The finite-difference
    formulas use the overlaps of the Bloch states at neighbouring mesh
    points, with the orbital positions included (as in the Wilson loops of
    *KSpace.wannier_centers*, so the centres agree with them). Only
    :math:`\tilde\Omega_D + \tilde\Omega_{OD}` depends on the gauge;
    :math:`\Omega_I` is fixed by the band group.

    In 1D, the minimum has :math:`\tilde\Omega_D = \tilde\Omega_{OD} = 0`
    for one band, and the centre is the Berry phase over :math:`2\pi`. For
    a Chern band, every projection fails somewhere (*min_singular_value*
    tends to zero), and the spread diverges as the mesh is refined. The
    descent finds the local minimum nearest the projected gauge: trial
    orbitals that miss the bands at isolated k-points (a small
    *min_singular_value*) leave phase vortices that it does not remove,
    even in a trivial band, so choose orbitals the bands actually live on.

    **Entangled bands.** With fewer trial orbitals :math:`n_W` than
    *bands* (or with a *window*), the bands need not be isolated: the
    Wannier functions span, at each k, the :math:`n_W`-dimensional subspace
    of the states of *bands* inside the outer *window* that minimizes
    :math:`\Omega_I` (Souza, Marzari and Vanderbilt, Phys. Rev. B 65,
    035109 (2001)), found by iterating from the projection of the trial
    orbitals, with linear *mixing*. The states inside the *frozen* window
    are kept exactly, so the Wannier bands (*WannierFunctions.kspace*)
    reproduce them. The maximal localization then runs within that
    subspace. Pass every band (``list(range(ks.norb))``) and select the
    states with the windows.

    :param ks: **KSpace** instance, Hermitian.
    :param bands: Band index, or list of band indices. With one trial
        orbital per band and no window, the group, which must be separated
        in energy from the other bands over the whole mesh; otherwise the
        candidate bands of the disentanglement.
    :param trial: Trial orbitals, one per Wannier function (at most one per
        band): a list of orbital indices (each a single orbital of the home
        cell), or a complex array of shape (norb, n_wann) whose columns are
        combinations of the orbitals of the home cell (e.g. a bond orbital).
    :param nk: Positive integer, or tuple of *dim* integers. Default value
        12. k-mesh, at least 3 along each direction; the Wannier functions
        live on the same number of cells.
    :param max_iter: Positive integer or zero. Default value 500. Maximal
        number of descent steps (0 keeps the projected gauge).
    :param tol: Positive real number. Default value 1e-10. Convergence
        threshold on the change of :math:`\Omega` between steps.
    :param step: Positive real number. Default value 0.5. Trial step
        :math:`\alpha` of the line search, :math:`dW = \alpha D/4\sum_b w_b`
        along the search direction :math:`D`; it is halved whenever the
        spread would grow.
    :param window: Tuple of two real numbers. Default value None (every
        state of *bands*). Outer energy window :math:`(E_{min}, E_{max})`:
        the states of *bands* inside it are the candidates, at least
        n_wann at every k-point.
    :param frozen: Tuple of two real numbers. Default value None (no
        frozen states). Inner energy window, inside *window*: its states,
        at most n_wann at every k-point, are kept in the subspace.
    :param dis_iter: Positive integer or zero. Default value 2000. Maximal
        number of disentanglement steps (0 keeps the projected subspace).
    :param dis_tol: Positive real number. Default value 1e-10. Convergence
        threshold on the change of :math:`\Omega_I` between steps.
    :param mixing: Real in (0, 1]. Default value 0.5. Fraction of the new
        :math:`Z(\mathbf{k})` mixed in at each disentanglement step.

    :returns:
        * **wf** -- **WannierFunctions** instance.

    Example usage::

        wf = wannierize(ssh, 0, trial=[0], nk=20)
        wf.centers, wf.spreads, wf.functions
        # two Wannier functions out of four bands, the lowest two kept exactly
        wf = wannierize(ks, [0, 1, 2, 3], trial=[0, 1], window=(-4., 3.), frozen=(-4., 0.))
        wf.kspace()  # the Wannier-interpolated model
    '''
    error_handling.kspace(ks, KSpace)
    error_handling.hermitian_model(ks.is_hermitian())
    if isinstance(bands, int):
        bands = [bands]
    error_handling.band_indices(bands, ks.norb)
    error_handling.trial_orbitals(trial, ks.norb, len(bands))
    error_handling.nk(nk, ks.dim)
    nk = (nk,) * ks.dim if isinstance(nk, int) else nk
    error_handling.nk_min(nk, 3)
    error_handling.positive_int_zero(max_iter, 'max_iter')
    error_handling.positive_real(tol, 'tol')
    error_handling.positive_real(step, 'step')
    if window is not None:
        error_handling.energy_window(window, 'window')
    if frozen is not None:
        error_handling.energy_window(frozen, 'frozen')
        if window is not None:
            error_handling.frozen_window(frozen, window)
    error_handling.positive_int_zero(dis_iter, 'dis_iter')
    error_handling.positive_real(dis_tol, 'dis_tol')
    error_handling.mixing(mixing)
    if isinstance(trial, list) and all(isinstance(o, int) for o in trial):
        g = np.eye(ks.norb, dtype='c16')[:, trial]
    else:
        g = np.asarray(trial, dtype='c16')

    n_k, n_w = int(np.prod(nk)), g.shape[1]
    entangled = n_w < len(bands) or window is not None or frozen is not None
    _, kpts = ks.mesh_grid(nk)
    en = ks._eigs(kpts)
    if not entangled:
        error_handling.isolated_bands(en, bands)
    v = ks._subspaces(kpts, bands)  # (n_k, norb, len(bands))

    # overlaps of the Bloch states with their mesh neighbours
    steps, b, w = _shells(ks.rec_vec_k, nk)
    index = np.arange(n_k).reshape(nk)
    neighbour = np.stack([np.roll(index, tuple(-st), axis=tuple(range(ks.dim))).ravel()
                                   for st in steps], axis=1)  # (n_k, nb)
    tau = ks.orbital_positions()
    phase = np.exp(-1j * tau @ (ks.k_basis @ b.T))  # (norb, nb)
    m0 = np.einsum('kom,ob,kbon->kbmn', v.conj(), phase, v[neighbour])

    # disentanglement: the n_w-dimensional subspace, then the same steps within it
    energies = en[:, bands]
    dis_history, dis_converged = np.array([]), True
    u_dis = np.broadcast_to(np.eye(n_w, dtype='c16'), (n_k, n_w, n_w))
    if entangled:
        lo, hi = window if window is not None else (-np.inf, np.inf)
        inside = (energies >= lo) & (energies <= hi)
        fixed = np.zeros_like(inside) if frozen is None else \
            inside & (energies >= frozen[0]) & (energies <= frozen[1])
        error_handling.window_states(int(inside.sum(axis=1).min()), int(fixed.sum(axis=1).max()), n_w)
        u_dis, dis_history, dis_converged = _disentangle(
            m0, neighbour, w, inside, fixed, v.conj().transpose(0, 2, 1) @ g, n_w,
            dis_iter, dis_tol, mixing)
        v = v @ u_dis  # (n_k, norb, n_w)
        m0 = u_dis.conj().transpose(0, 2, 1)[:, None] @ m0 @ u_dis[neighbour]

    # projection, made unitary: A = W s V^dag -> U = W V^dag
    a = v.conj().transpose(0, 2, 1) @ g
    w_svd, s, vh = np.linalg.svd(a)
    error_handling.projection(s.min())
    u = w_svd @ vh

    def overlaps(u):
        return u.conj().transpose(0, 2, 1)[:, None] @ m0 @ u[neighbour]

    def trial_gauge(u, d, x):
        u_x = u @ _unitary_step(d, x)
        m_x = overlaps(u_x)
        c_x, s_x, *_ = _spread(m_x, b, w, n_k)
        return float(np.sum(s_x)), u_x, m_x, c_x

    m = overlaps(u)
    centers, spreads, *_ = _spread(m, b, w, n_k)
    history = [float(np.sum(spreads))]
    eps = step / (4 * np.sum(w))
    converged = max_iter == 0
    d, g_norm = None, 0.
    for it in range(max_iter):
        # conjugate gradients (Fletcher-Reeves, restarted every 5 steps; a
        # direction without descent fails the line search, which restarts)
        grad = _gradient(m, b, w, centers)
        g_new = float(np.sum(np.abs(grad) ** 2))
        d = grad + g_new / g_norm * d if d is not None and it % 5 and g_norm > 0. else grad
        g_norm = g_new
        # parabola through steps 0, eps, 2 eps; keep the lowest spread found
        tries = [trial_gauge(u, d, eps), trial_gauge(u, d, 2 * eps)]
        o0, o1, o2 = history[-1], tries[0][0], tries[1][0]
        curv = o2 - 2 * o1 + o0
        if curv > 0.:
            x = (3 * o0 - 4 * o1 + o2) / (2 * curv)
            if 0. < x < 4.:
                tries.append(trial_gauge(u, d, x * eps))
        best = min(tries, key=lambda t: t[0])
        if best[0] >= o0:
            # no descent along d: restart from the gradient with a smaller step,
            # until the step is negligible (a minimum, within roundoff)
            eps /= 2
            d = None
            if eps < 1e-6 * step / (4 * np.sum(w)):
                converged = True
                break
            continue
        omega, u, m, centers = best
        history.append(omega)
        if o0 - omega < tol:
            converged = True
            break
    centers, spreads, omega_i, omega_d, omega_od = _spread(m, b, w, n_k)

    # Wannier functions: W_n(R, o) = (1/N) sum_k e^{ik.R} (v U)_{on}(k)
    amp = np.fft.ifftn((v @ u).reshape(nk + (ks.norb, n_w)), axes=tuple(range(ks.dim)))
    # move each one so that its centre sits in the middle cell
    rec = np.array(ks.rec_vec, dtype='f8')
    cart = centers @ ks.k_basis.T
    frac = cart @ rec.T / (2 * np.pi)
    shift = np.array(nk)[None, :] // 2 - np.floor(frac + 1e-8).astype(int)  # roundoff at a cell corner
    functions = np.zeros((n_w, n_k * ks.norb), 'c16')
    for n in range(n_w):
        moved = np.roll(amp[..., n], tuple(shift[n]), axis=tuple(range(ks.dim)))
        # rows of finite_ham: cell n1 + N1 n2 + N1 N2 n3, then the orbital
        functions[n] = np.transpose(moved, tuple(range(ks.dim))[::-1] + (ks.dim,)).ravel()
    cart = cart + shift @ np.array(ks.lat.prim_vec, dtype='f8')
    positions, _ = _finite_sites(ks, nk)
    # out of the plane of a 2D lattice in 3D (or off a chain), the position
    # operator is well defined: the centre is its expectation value
    perp = np.eye(ks.space_dim) - ks.k_basis @ ks.k_basis.T
    cart = cart + (np.abs(functions) ** 2 @ positions) @ perp
    gauge = u_dis @ u  # (n_k, len(bands), n_w)
    ham_k = gauge.conj().transpose(0, 2, 1) @ (energies[:, :, None] * gauge)
    return WannierFunctions(ks, nk, functions, positions, cart, spreads, omega_i, omega_d,
                                      omega_od, gauge, np.array(history), converged, float(s.min()),
                                      dis_history, dis_converged, ham_k, shift)
