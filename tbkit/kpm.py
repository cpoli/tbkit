r"""
The kernel polynomial method (KPM; Silver and Roder 1994, Weisse,
Wellein, Alvermann and Fehske, Rev. Mod. Phys. 78, 275 (2006)): spectral
quantities of a large sparse Hamiltonian from its Chebyshev moments,
without diagonalizing it. The spectrum is first mapped into
:math:`(-1, 1)`, :math:`\tilde H = (H - b)/a`; then

.. math::

    \delta(E - H) = \sum_{m=0}^{M-1} c_m(E)\, T_m(\tilde H)\, ,\qquad
    c_m(E) = \frac{(2-\delta_{m0})\,g_m\,T_m(\epsilon)}{a\pi\sqrt{1-\epsilon^2}}\, ,

with :math:`\epsilon = (E-b)/a`, :math:`T_m` the Chebyshev polynomials
(computed by the recursion :math:`T_{m+1}(\tilde H)|r\rangle =
2\tilde H\,T_m(\tilde H)|r\rangle - T_{m-1}(\tilde H)|r\rangle`, one sparse
matrix-vector product each), and :math:`g_m` a kernel that damps the Gibbs
oscillations of the truncated series (Jackson: an almost Gaussian
broadening of width about :math:`\pi a/M`). Traces are estimated with a
few random-phase vectors, :math:`\mathrm{Tr}\,A \approx
\frac1R\sum_r\langle r|A|r\rangle`, whose error falls as :math:`1/\sqrt{RN}`:
the cost is :math:`O(MRN)` for :math:`N` sites, so millions of sites are
within reach.

The Hamiltonian must be Hermitian (*System.ham*, a sparse matrix).
"""
from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
import scipy.sparse as sparse
from scipy.integrate import trapezoid
from scipy.special import jv

import tbkit.error_handling as error_handling
import tbkit.occupation as occupation


PI = np.pi


def spectral_bounds(ham) -> tuple[float, float]:
    r'''
    Get bounds on the spectrum of a Hermitian matrix from Gershgorin's
    circle theorem: every eigenvalue lies within
    :math:`H_{ii} \pm \sum_{j\neq i}|H_{ij}|` for some :math:`i`.

    Parameters
    ----------
    ham : Square sparse (or dense) Hermitian matrix

    Returns
    -------
    e_min, e_max : Real numbers
    '''
    ham = sparse.csr_matrix(ham)
    diag = ham.diagonal().real
    radii = np.asarray(abs(ham).sum(axis=1)).ravel() - np.abs(ham.diagonal())
    return float(np.min(diag - radii)), float(np.max(diag + radii))


def jackson_kernel(n_moments: int) -> NDArray[np.float64]:
    r'''
    Get the Jackson kernel

    .. math::

        g_m = \frac{(M-m+1)\cos\frac{\pi m}{M+1}
              + \sin\frac{\pi m}{M+1}\cot\frac{\pi}{M+1}}{M+1}\, ,

    the optimal choice for spectral densities (positive, and resolution
    about :math:`\pi/M` in the scaled energy).

    Parameters
    ----------
    n_moments : Positive integer :math:`M`

    Returns
    -------
    g : Real ndarray, shape (n_moments,)
    '''
    error_handling.positive_int(n_moments, 'n_moments')
    m = np.arange(n_moments)
    q = PI / (n_moments + 1)
    return ((n_moments - m + 1) * np.cos(q * m) + np.sin(q * m) / np.tan(q)) / (n_moments + 1)


def lorentz_kernel(n_moments: int, lam: float = 4.) -> NDArray[np.float64]:
    r'''
    Get the Lorentz kernel :math:`g_m = \sinh[\lambda(1 - m/M)]/\sinh\lambda`,
    which gives Lorentzian broadening (the better choice for Green's
    functions).

    Parameters
    ----------
    n_moments : Positive integer :math:`M`
    lam : Positive real, default 4

    Returns
    -------
    g : Real ndarray, shape (n_moments,)
    '''
    error_handling.positive_int(n_moments, 'n_moments')
    error_handling.positive_real(lam, 'lam')
    return np.sinh(lam * (1. - np.arange(n_moments) / n_moments)) / np.sinh(lam)


def _setup(
    ham, n_moments, kernel, bounds, e_grid, n_grid=1001,
) -> tuple[sparse.csr_matrix, NDArray[np.float64], NDArray[np.float64], float, float]:
    '''
    Private function. Validate, rescale the Hamiltonian, and build the
    coefficients c_m(E) (shape (n_moments, len(e_grid))).
    '''
    error_handling.positive_int(n_moments, 'n_moments')
    error_handling.kpm_kernel(kernel)
    ham_s, a, b = _rescale(ham, bounds)
    if e_grid is None:
        e_grid = b + a * np.cos(PI * (np.arange(n_grid) + 0.5) / n_grid)[::-1]
    e_grid = np.atleast_1d(np.asarray(e_grid, dtype='f8'))
    error_handling.ndarray_empty(e_grid, 'e_grid')
    eps = (e_grid - b) / a
    error_handling.kpm_grid(eps)
    return ham_s, e_grid, _coefficients(eps, n_moments, kernel, a), a, b


def _coefficients(eps, n_moments, kernel, a) -> NDArray[np.float64]:
    '''
    Private function. The coefficients c_m(E) of delta(E - H) at the scaled
    energies eps, shape (n_moments, len(eps)).
    '''
    g = jackson_kernel(n_moments) if kernel == 'jackson' else lorentz_kernel(n_moments)
    tm = np.cos(np.arange(n_moments)[:, None] * np.arccos(eps)[None, :])
    weight = np.where(np.arange(n_moments) == 0, 1., 2.)
    return (weight * g)[:, None] * tm / (a * PI * np.sqrt(1. - eps ** 2))[None, :]


def _rescale(ham, bounds) -> tuple[sparse.csr_matrix, float, float]:
    '''
    Private function. Validate a Hermitian Hamiltonian and map its spectrum
    into (-0.99, 0.99): H~ = (H - b)/a.
    '''
    ham = sparse.csr_matrix(ham, dtype='c16')
    error_handling.square_matrix(ham, 'ham')
    error_handling.hermitian(ham)
    if bounds is None:
        bounds = spectral_bounds(ham)
    error_handling.lims(bounds)
    e_min, e_max = bounds
    a = (e_max - e_min) / (2. - 0.02)  # keep the scaled spectrum inside (-0.99, 0.99)
    b = (e_max + e_min) / 2.
    return (ham - b * sparse.identity(ham.shape[0], format='csr')) / a, a, b


def _vectors(n: int, n_random: int | None, seed) -> NDArray[np.complex128]:
    '''
    Private function. Random-phase vectors (columns, |r_i| = 1), or, if
    n_random is None, the full basis (the exact trace).
    '''
    if n_random is None:
        return np.eye(n, dtype='c16')
    error_handling.positive_int(n_random, 'n_random')
    rng = np.random.default_rng(seed)
    return np.exp(2j * PI * rng.random((n, n_random)))


def _chebyshev(ham_s, vec: NDArray[np.complex128], n_moments: int):
    '''
    Private function. Yield T_m(H~) vec for m < n_moments, keeping only two
    vectors in memory.
    '''
    prev, cur = None, vec
    for m in range(n_moments):
        yield cur
        prev, cur = cur, (ham_s @ cur if m == 0 else 2 * (ham_s @ cur) - prev)


def _diagonal_moments(ham_s, vecs: NDArray[np.complex128], n_moments: int) -> NDArray[np.float64]:
    '''
    Private function. mu_m = <v|T_m(H~)|v> for every column v of vecs, shape
    (n_moments, n_columns), from half as many products: mu_2m = 2<t_m|t_m> - mu_0,
    mu_2m+1 = 2<t_m+1|t_m> - mu_1 (Weisse et al., Eq. (48)).
    '''
    def dot(u, w):
        return (u.real * w.real + u.imag * w.imag).sum(axis=0)  # Re <u|w>

    mu = np.empty((n_moments + 1, vecs.shape[1]))
    prev = None
    for m, t in enumerate(_chebyshev(ham_s, vecs, n_moments // 2 + 1)):
        if m == 0:
            mu[0] = dot(vecs, vecs)
        else:
            if m == 1:
                mu[1] = dot(vecs, t)
            mu[2 * m - 1] = 2 * dot(t, prev) - mu[1]
            if 2 * m <= n_moments:
                mu[2 * m] = 2 * dot(t, t) - mu[0]
        prev = t
    return mu[:n_moments]


def _two_index_moments(ham_s, va, vb, vec: NDArray[np.complex128], n_moments: int) -> NDArray[np.complex128]:
    '''
    Private function. mu_mn = sum_r <r| v_a T_m(H~) v_b T_n(H~) |r> over the
    columns r of vec (not divided by their number).
    '''
    mu = np.zeros((n_moments, n_moments), 'c16')
    block = 32
    for r in vec.T:
        right = np.array([vb @ t for t in _chebyshev(ham_s, r, n_moments)]).T  # v_b T_n |r>
        left = np.empty((block, len(r)), 'c16')  # T_m v_a |r>, a block of m at a time
        for m, t in enumerate(_chebyshev(ham_s, va @ r, n_moments)):
            left[m % block] = t
            if m % block == block - 1 or m == n_moments - 1:
                start = m - m % block
                mu[start:m + 1] += left[:m % block + 1].conj() @ right
    return mu


def dos(
    ham, n_moments: int = 256, n_random: int | None = 10, e_grid: ArrayLike | None = None,
    kernel: str = 'jackson', seed=None, bounds: tuple[float, float] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    r'''
    Get the density of states :math:`\rho(E) = \mathrm{Tr}\,\delta(E - H)`
    (normalized, like *tbkit.dos.density_of_states*, to integrate to the
    number of states).

    Parameters
    ----------
    ham : Square sparse Hermitian matrix (e.g. *System.ham*)
    n_moments : Positive integer, default 256
        Number of
        Chebyshev moments (the energy resolution is about
        :math:`\pi a/M`, :math:`2a` the spectral width).
    n_random : Positive integer, or None, default 10
        Number of
        random vectors for the trace; None for the exact trace (N vectors:
        small systems only).
    e_grid : Real array
        Default value None: 1001 Chebyshev nodes over
        the spectrum. Energies, inside the spectral bounds.
    kernel : String, default 'jackson'
        'jackson' or 'lorentz'.
    seed
        Default value None. Seed of the random vectors.
    bounds : Pair of reals
        Default value None (Gershgorin bounds, see
        *spectral_bounds*). Bounds enclosing the whole spectrum.

    Returns
    -------
    e_grid : Real ndarray
    rho : Real ndarray, same shape
    '''
    ham_s, e_grid, coef, _, _ = _setup(ham, n_moments, kernel, bounds, e_grid)
    vec = _vectors(ham_s.shape[0], n_random, seed)
    mu = np.array([np.vdot(vec, t).real for t in _chebyshev(ham_s, vec, n_moments)])
    if n_random is not None:
        mu = mu / n_random  # E<r|A|r> = Tr A for random phases
    return e_grid, mu @ coef


def dos_from_levels(
    levels: ArrayLike, n_moments: int, bounds: tuple[float, float], e_grid: ArrayLike,
    kernel: str = 'jackson',
) -> NDArray[np.float64]:
    r'''
    Get the KPM density of states of a *known* spectrum: every level
    broadened by exactly the truncated, kernel-damped Chebyshev series that
    *dos* uses. It is the reference *dos* converges to (up to the random
    trace), e.g. for the bands of a periodic model on a k-mesh.

    Parameters
    ----------
    levels : Real array
        Energy levels.
    n_moments : Positive integer
        Number of Chebyshev moments.
    bounds : Pair of reals
        Spectral bounds (those of *dos*, e.g.
        *spectral_bounds* of the Hamiltonian it is compared with).
    e_grid : Real array
        Energies.
    kernel : String, default 'jackson'

    Returns
    -------
    rho
        Real ndarray, same shape as *e_grid* (integrating to the number of levels).
    '''
    levels = np.asarray(levels, dtype='f8').ravel()
    error_handling.ndarray_empty(levels, 'levels')
    _, e_grid, coef, a, b = _setup(sparse.identity(1, format='csr'), n_moments, kernel, bounds, e_grid)
    error_handling.kpm_grid((levels - b) / a)
    tm = np.cos(np.arange(n_moments)[:, None] * np.arccos((levels - b) / a)[None, :])
    return coef.T @ tm.sum(axis=1)


def ldos(
    ham, site: int, n_moments: int = 256, e_grid: ArrayLike | None = None,
    kernel: str = 'jackson', bounds: tuple[float, float] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    r'''
    Get the local density of states on one site (or orbital),
    :math:`\rho_i(E) = \langle i|\delta(E-H)|i\rangle`, exactly (no random vectors).

    Parameters
    ----------
    ham : Square sparse Hermitian matrix
    site : Integer
        Site (row) index.
    n_moments : Positive integer, default 256
    e_grid
        See *dos*.
    kernel
        See *dos*.
    bounds
        See *dos*.

    Returns
    -------
    e_grid : Real ndarray
    rho : Real ndarray, same shape
    '''
    ham_s, e_grid, coef, _, _ = _setup(ham, n_moments, kernel, bounds, e_grid)
    error_handling.site_index(site, ham_s.shape[0])
    vec = np.zeros(ham_s.shape[0], 'c16')
    vec[site] = 1.
    mu = np.array([t[site].real for t in _chebyshev(ham_s, vec, n_moments)])
    return e_grid, mu @ coef


def conductivity(
    ham, x: ArrayLike, n_moments: int = 128, n_random: int | None = 10,
    e_grid: ArrayLike | None = None, kernel: str = 'jackson', seed=None,
    bounds: tuple[float, float] | None = None, area: float | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    r'''
    Get the zero-temperature DC conductivity along :math:`x` from the
    Kubo-Greenwood formula,

    .. math::

        \sigma_{xx}(E) = \frac{\pi}{\Omega}\,
        \mathrm{Tr}\left[v_x\,\delta(E-H)\,v_x\,\delta(E-H)\right]\, ,\qquad
        v_x = i[H, x]\, ,

    in units of :math:`e^2/\hbar` (:math:`\hbar = e = 1`), by the
    two-dimensional Chebyshev expansion of Weisse et al. (the moments
    :math:`\mu_{mn} = \mathrm{Tr}[v_x T_m(\tilde H) v_x T_n(\tilde H)]`,
    :math:`O(M N)` memory, the DOS needing only :math:`O(N)`).

    Parameters
    ----------
    ham : Square sparse Hermitian matrix
    x : Real array, one coordinate per site (e.g. *lat.coor['x']*)
    n_moments : Positive integer, default 128
    n_random : Positive integer, or None, default 10
        See *dos*.
    e_grid
        See *dos*.
    kernel
        See *dos*.
    seed
        See *dos*.
    bounds
        See *dos*.
    area : Positive real
        Default value None (the number of sites):
        the area (volume) :math:`\Omega` of the sample.

    Returns
    -------
    e_grid : Real ndarray
    sigma : Real ndarray, same shape
    '''
    ham_s, e_grid, coef, a, b = _setup(ham, n_moments, kernel, bounds, e_grid)
    n = ham_s.shape[0]
    x = np.asarray(x, dtype='f8')
    error_handling.ndarray(x, 'x', n)
    if area is None:
        area = float(n)
    error_handling.positive_real(area, 'area')
    # v = i [H, x]: v_ij = i H_ij (x_j - x_i); the scale a comes back below
    ham_c = sparse.csr_matrix(ham, dtype='c16')
    vel = 1j * (ham_c @ sparse.diags(x) - sparse.diags(x) @ ham_c)
    mu = _two_index_moments(ham_s, vel, vel, _vectors(n, n_random, seed), n_moments)
    if n_random is not None:
        mu = mu / n_random
    sigma = PI / area * np.einsum('me,mn,ne->e', coef, mu, coef).real
    return e_grid, sigma


def hall_conductivity(
    ham, vx, vy, n_moments: int = 256, n_random: int | None = 10,
    e_grid: ArrayLike | None = None, kernel: str = 'jackson', seed=None,
    bounds: tuple[float, float] | None = None, area: float | None = None,
    temperature: float = 0.,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    r'''
    Get the Hall conductivity :math:`\sigma_{xy}(\mu)` of a large sample, at
    any Fermi level and temperature, from the Kubo-Bastin formula expanded
    in Chebyshev polynomials (Garcia, Covaci and Rappoport, Phys. Rev. Lett.
    114, 116602 (2015), Eqs. (1)-(5)). Bastin's formula,

    .. math::

        \sigma^{B}_{\alpha\beta}(\mu) = \frac{ie^2\hbar}{\Omega}\int d\varepsilon\,f(\varepsilon)\,
        \mathrm{Tr}\left[v_\alpha\,\delta(\varepsilon-H)\,v_\beta\,\frac{dG^+}{d\varepsilon}
        - v_\alpha\,\frac{dG^-}{d\varepsilon}\,v_\beta\,\delta(\varepsilon-H)\right]\, ,

    :math:`G^\pm = (\varepsilon - H \pm i0)^{-1}`, becomes, with the spectrum
    rescaled into :math:`(-1, 1)`, :math:`\tilde H = (H-b)/a`,

    .. math::

        \sigma^{B}_{\alpha\beta}(\mu) = \frac{4e^2\hbar}{\pi\Omega a^2}\int_{-1}^{1}
        d\tilde\varepsilon\,\frac{f(\tilde\varepsilon)}{(1-\tilde\varepsilon^2)^2}
        \sum_{m,n}\Gamma_{nm}(\tilde\varepsilon)\,\mu^{\alpha\beta}_{nm}\, ,

    .. math::

        \mu^{\alpha\beta}_{mn} = \frac{g_mg_n}{(1+\delta_{m0})(1+\delta_{n0})}
        \mathrm{Tr}\left[v_\alpha T_m(\tilde H)\,v_\beta T_n(\tilde H)\right]\, ,

    .. math::

        \Gamma_{mn}(\tilde\varepsilon) = (\tilde\varepsilon - in\sqrt{1-\tilde\varepsilon^2})\,
        e^{in\theta}\,T_m(\tilde\varepsilon)
        + (\tilde\varepsilon + im\sqrt{1-\tilde\varepsilon^2})\,e^{-im\theta}\,T_n(\tilde\varepsilon)\, ,

    :math:`\theta = \arccos\tilde\varepsilon`, :math:`g_m` the kernel. The
    moments :math:`\mu_{mn}` (the expensive part: :math:`5MR` sparse
    products, :math:`O(MN)` memory) serve every Fermi level and temperature.
    The energy integral runs over :math:`|\tilde\varepsilon|\le 0.995`
    (the spectrum lies within :math:`\pm0.99`): at the ends,
    :math:`(1-\tilde\varepsilon^2)^{-2}` would only amplify rounding errors.

    **Hall part.** The value is computed from the antisymmetric moments
    :math:`(\mu^{xy}_{mn} - \mu^{yx}_{mn})/2`, i.e. it is the Hall
    conductivity proper, :math:`(\sigma^B_{xy} - \sigma^B_{yx})/2`, the part
    that the Berry curvature gives (*KSpace.hall_conductivity*). The
    symmetric part :math:`(\sigma^B_{xy} + \sigma^B_{yx})/2` is a
    Fermi-surface term, like :math:`\sigma_{xx}` (see *conductivity*): it
    vanishes by symmetry in many lattices (e.g. with a :math:`C_3` axis),
    but in a clean metal of lower symmetry it grows as the kernel's
    broadening :math:`\pi a/M` shrinks (the Drude weight), and with random
    vectors it is the main source of noise. Leaving it out removes that
    noise too. In a clean metal the random-vector estimate is still noisy
    inside the bands (increase *n_random*, or the size of the sample);
    disorder, which makes the Fermi-surface terms finite, and gaps, where
    they vanish, are much quieter.

    **Sign and units.** The value returned is :math:`-(\sigma^B_{xy}-\sigma^B_{yx})/2` in units
    of :math:`e^2/h` (:math:`\hbar = e = 1`), the sign of
    *KSpace.hall_conductivity*: :math:`C` in the gap of a Chern insulator
    of Chern number :math:`C` (*KSpace.chern_number*). Bastin's
    :math:`\sigma^B_{xy}` is the Ohm's-law tensor, :math:`j_x = \sigma^B_{xy}E_y`
    (see *KSpace.hall_conductivity*).

    **Sample.** The trace must run over a sample without edges -- a torus
    (*KSpace.finite_ham* and *KSpace.finite_velocity* with
    ``periodic=True``): over a whole open flake, the edge currents cancel
    the bulk and the trace vanishes in a gap.

    Parameters
    ----------
    ham : Square sparse Hermitian matrix
    vx : Square sparse (or dense) Hermitian matrix, like *ham*
        Velocity
        :math:`v_x = i[H, x]` (see *KSpace.finite_velocity*). For a spin
        Hall conductivity, pass the spin current :math:`\{s, v_x\}/2`
        instead (units :math:`e/2\pi`, see *KSpace.spin_hall_conductivity*).
    vy : Square sparse (or dense) Hermitian matrix
        Velocity :math:`v_y`.
    n_moments : Positive integer, default 256
        Number of
        Chebyshev moments :math:`M` (energy resolution about :math:`\pi a/M`).
    n_random : Positive integer, or None, default 10
        See *dos*.
    e_grid : Real array
        Default value None: 1001 Chebyshev nodes over
        the spectrum. Fermi energies :math:`\mu`.
    kernel : String, default 'jackson'
        See *dos*.
    seed
        See *dos*.
    bounds
        See *dos*.
    area : Positive real
        Default value None (the number of orbitals).
        Area :math:`\Omega` of the sample (e.g. number of cells times the
        cell area).
    temperature : Positive real or zero, default 0
        In energy units.

    Returns
    -------
    e_grid : Real ndarray
    sigma : Real ndarray, same shape, in units of :math:`e^2/h`
    '''
    ham_s, e_grid, _, a, b = _setup(ham, n_moments, kernel, bounds, e_grid)
    n = ham_s.shape[0]
    error_handling.velocity(vx, n, 'vx')
    error_handling.velocity(vy, n, 'vy')
    vx = sparse.csr_matrix(vx, dtype='c16')
    vy = sparse.csr_matrix(vy, dtype='c16')
    error_handling.hermitian(vx)
    error_handling.hermitian(vy)
    if area is None:
        area = float(n)
    error_handling.positive_real(area, 'area')
    error_handling.positive_real_zero(temperature, 'temperature')
    g = jackson_kernel(n_moments) if kernel == 'jackson' else lorentz_kernel(n_moments)
    w = g / np.where(np.arange(n_moments) == 0, 2., 1.)
    mu = np.zeros((n_moments, n_moments), 'c16')
    for r in _vectors(n, n_random, seed).T:
        # <r| v_x T_m = (T_m v_x |r>)^dagger and v_y T_n |r>, for mu^xy;
        # the same with x and y swapped, for mu^yx
        t_r = list(_chebyshev(ham_s, r, n_moments))
        left_x = np.array(list(_chebyshev(ham_s, vx @ r, n_moments)))
        left_y = np.array(list(_chebyshev(ham_s, vy @ r, n_moments)))
        right_y = np.array([vy @ t for t in t_r])
        right_x = np.array([vx @ t for t in t_r])
        mu += (left_x.conj() @ right_y.T - left_y.conj() @ right_x.T) / 2
    if n_random is not None:
        mu = mu / n_random
    mu = w[:, None] * mu * w[None, :]
    # the energy integrand on a grid uniform in theta = arccos(eps)
    n_int = max(2001, 8 * n_moments + 1)
    theta = np.linspace(np.arccos(-0.995), np.arccos(0.995), n_int)
    eps = np.cos(theta)
    root = np.sin(theta)
    m = np.arange(n_moments)[:, None]
    cheb = np.cos(m * theta[None, :])
    plus = (eps[None, :] - 1j * m * root[None, :]) * np.exp(1j * m * theta[None, :])
    minus = (eps[None, :] + 1j * m * root[None, :]) * np.exp(-1j * m * theta[None, :])
    # sum_{n,m} Gamma_nm mu_nm, Gamma_nm = plus_m T_n + minus_n T_m
    gamma_mu = np.sum(cheb * (mu @ plus), axis=0) + np.sum(minus * (mu @ cheb), axis=0)
    integrand = (gamma_mu / (1. - eps ** 2) ** 2).real
    prefactor = -2 * PI * 4. / (PI * area * a ** 2)  # e^2/hbar = 2 pi e^2/h, TKNN sign
    energies = a * eps + b
    if temperature == 0:
        cum = np.concatenate([[0.], np.cumsum(0.5 * (integrand[1:] + integrand[:-1]) * np.diff(eps))])
        sigma = prefactor * np.interp(e_grid, energies, cum)
    else:
        f = np.array([occupation.fermi_dirac(energies, float(e), temperature) for e in e_grid])
        sigma = prefactor * trapezoid(f * integrand[None, :], eps, axis=1)
    return e_grid, sigma



def time_evolution(
    ham, psi_init: NDArray, times: ArrayLike, bounds: tuple[float, float] | None = None,
    tol: float = 1e-14,
) -> NDArray[np.complex128]:
    r'''
    Get the time evolution :math:`|\psi(t)\rangle = e^{-iHt}|\psi(0)\rangle`
    (:math:`\hbar = 1`) of a large sparse Hamiltonian from its Chebyshev
    expansion (Tal-Ezer and Kosloff, J. Chem. Phys. 81, 3967 (1984)),

    .. math::

        e^{-iH\Delta t} = e^{-ib\Delta t}\sum_{m\ge0}(2-\delta_{m0})\,(-i)^m
        J_m(a\Delta t)\,T_m(\tilde H)\, ,\qquad \tilde H = (H-b)/a\, ,

    :math:`J_m` the Bessel functions of the first kind. The series converges
    exponentially once :math:`m > a\Delta t`: it is truncated where
    :math:`|J_m(a\Delta t)|` falls below *tol*, so the result is exact to
    machine precision and unitary, whatever the time step, at the cost of
    about :math:`a\Delta t` sparse matrix-vector products per interval
    (:math:`2a` the spectral width). Nothing is factorized or stored but a
    few vectors: unlike the Crank-Nicolson step of
    *tbkit.propagation.Propagation* (a sparse LU factorization, whose
    fill-in grows quickly in 2D and 3D, and an :math:`O(\Delta t^2)`
    error), it reaches millions of sites at machine precision.

    Parameters
    ----------
    ham : Square sparse Hermitian matrix (e.g. *System.ham*)
    psi_init : Complex ndarray, shape (N,)
        Initial state.
    times : Real array, non-decreasing
        Times at which the state is returned, counted from the initial
        state (at :math:`t = 0`). Each interval between two of them is one
        Chebyshev expansion.
    bounds : Pair of reals
        Default value None (Gershgorin bounds, see *spectral_bounds*).
        Bounds enclosing the whole spectrum.
    tol : Positive real, default 1e-14
        Truncation of the Bessel coefficients.

    Returns
    -------
    psi : Complex ndarray, shape (N, len(times))
        The states :math:`\psi(t)`, one column per time.
    '''
    ham_s, a, b = _rescale(ham, bounds)
    n = ham_s.shape[0]
    error_handling.ndarray(psi_init, 'psi_init', n)
    times = np.atleast_1d(np.asarray(times, dtype='f8'))
    error_handling.kpm_times(times)
    error_handling.positive_real(tol, 'tol')
    psi = np.empty((n, len(times)), 'c16')
    cur = np.asarray(psi_init, dtype='c16').ravel()
    for i, dt in enumerate(np.diff(np.concatenate([[0.], times]))):
        if dt != 0:
            x = a * dt
            n_terms = int(abs(x) + 10. * abs(x) ** (1. / 3.) + 20.)
            coef = jv(np.arange(n_terms), x) * (-1j) ** np.arange(n_terms)
            coef[1:] *= 2.
            n_terms = int(np.nonzero(np.abs(coef) > tol)[0][-1]) + 1
            nxt = np.zeros(n, 'c16')
            for c, t in zip(coef[:n_terms], _chebyshev(ham_s, cur, n_terms)):
                nxt += c * t
            cur = np.exp(-1j * b * dt) * nxt
        psi[:, i] = cur
    return psi


def spectral_function(
    ham, positions: ArrayLike, k: ArrayLike, orbitals: ArrayLike | None = None,
    n_moments: int = 256, e_grid: ArrayLike | None = None, kernel: str = 'jackson',
    bounds: tuple[float, float] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    r'''
    Get the spectral function :math:`A(\mathbf{k}, E)` of a large sample
    (disordered, with vacancies, ...), what angle-resolved photoemission
    measures, from the Chebyshev moments of plane waves,

    .. math::

        A(\mathbf{k}, E) = \sum_o\langle\mathbf{k}, o|\delta(E-H)|\mathbf{k}, o\rangle\, ,\qquad
        |\mathbf{k}, o\rangle = \frac{1}{\sqrt{N_o}}\sum_{i\in o}
        e^{i\mathbf{k}\cdot\mathbf{r}_i}|i\rangle\, ,

    the sum running over the :math:`N_o` sites of each sublattice (orbital)
    *o*: one recursion of *n_moments* products per k-point and sublattice,
    all of them done together. :math:`\int A\,dE` is the number of
    sublattices, and in a clean crystal :math:`A` is peaked on the bands
    :math:`E_n(\mathbf{k})`; disorder broadens and shifts the peaks. It is
    the spectral function that *tbkit.moire.spectral_function* unfolds from
    the eigenstates of a supercell (the same, for a torus of that
    supercell), without diagonalizing. On a torus, take :math:`\mathbf{k}`
    on its mesh (the plane waves are then periodic); in an open flake the
    edges add a broadening of the order of the inverse size.

    Parameters
    ----------
    ham : Square sparse Hermitian matrix, N rows
    positions : Real array, shape (N,) or (N, dim)
        Position of each site (row), e.g.
        ``np.column_stack([lat.coor['x'], lat.coor['y']])``. Cell origins
        give the same result.
    k : Real array, shape (nk, dim) (or (nk,) in 1D)
        Cartesian wavevectors.
    orbitals : Array of N labels, default None
        The sublattice (orbital) of each row, e.g. ``lat.coor['tag']``;
        None: a single one (one orbital per cell).
    n_moments : Positive integer, default 256
        See *dos*.
    e_grid
        See *dos*.
    kernel
        See *dos*.
    bounds
        See *dos*.

    Returns
    -------
    e_grid : Real ndarray
    A : Real ndarray, shape (nk, len(e_grid))
    '''
    ham_s, e_grid, coef, _, _ = _setup(ham, n_moments, kernel, bounds, e_grid)
    n = ham_s.shape[0]
    positions = np.asarray(positions, dtype='f8')
    if positions.ndim == 1:
        positions = positions[:, None]
    error_handling.ndarray(positions, 'positions', n)
    k = np.asarray(k, dtype='f8')
    if k.ndim == 1 and positions.shape[1] == 1:
        k = k[:, None]
    error_handling.ks(k, positions.shape[1])
    if orbitals is None:
        orbitals = np.zeros(n, int)
    orbitals = np.asarray(orbitals)
    error_handling.ndarray(orbitals, 'orbitals', n)
    _, label = np.unique(orbitals, return_inverse=True)
    label = label.ravel()
    n_orb = label.max() + 1
    # one plane wave per (k, orbital): column k*n_orb + o
    mask = (label[:, None] == np.arange(n_orb)[None, :]) / np.sqrt(np.bincount(label))[None, :]
    waves = (np.exp(1j * positions @ k.T)[:, :, None] * mask[:, None, :]).reshape(n, -1)
    mu = _diagonal_moments(ham_s, waves, n_moments).reshape(n_moments, len(k), n_orb).sum(axis=2)
    return e_grid, mu.T @ coef


def optical_conductivity(
    ham, vel, omega: float | ArrayLike, e_fermi: float = 0., temperature: float = 0.,
    n_moments: int = 256, n_random: int | None = 10, kernel: str = 'jackson', seed=None,
    bounds: tuple[float, float] | None = None, area: float | None = None,
    vel_b=None, part: str = 'real',
) -> NDArray[np.float64] | NDArray[np.complex128]:
    r'''
    Get the optical conductivity :math:`\sigma_{ab}(\omega)` of a large
    sample from the Kubo-Greenwood formula,

    .. math::

        \sigma_{ab}(\omega) = \frac{ie^2}{\hbar\,\Omega}\int\!\!\int dE_1\,dE_2\,
        j_{ab}(E_1, E_2)\,\frac{f(E_1) - f(E_2)}{E_2 - E_1}\,
        \frac{1}{\hbar\omega - (E_1 - E_2) + i0}\, ,\qquad
        j_{ab}(E_1, E_2) = \mathrm{Tr}\left[v_a\,\delta(E_1-H)\,v_b\,\delta(E_2-H)\right]\, ,

    the trace expanded in two-index Chebyshev moments
    :math:`\mu_{mn} = \mathrm{Tr}[v_b T_m(\tilde H) v_a T_n(\tilde H)]`
    (Weisse et al., Sec. IV; those of *conductivity*, which is the
    :math:`\omega\to0` limit of :math:`\mathrm{Re}\,\sigma_{xx}` at zero
    temperature). It is *tbkit.optics.optical_conductivity* (the same
    tensor and sign: :math:`j_a = \sigma_{ab}E_b`, both inter- and
    intraband, the Drude peak broadened by the kernel to a width
    :math:`\sim\pi a/M` instead of :math:`\eta`), for a sample of any size,
    clean or disordered. The moments (the expensive part, :math:`O(MR)`
    sparse products, :math:`O(MN)` memory) serve every frequency, Fermi
    level and temperature.

    With the spectral function of the transitions,
    :math:`S_{ab}(\nu) = \frac1\Omega\int dE\,j_{ab}(E+\nu, E)\,
    \frac{f(E) - f(E+\nu)}{\nu}`, it is
    :math:`\sigma_{ab}(\omega) = \frac{e^2}{\hbar}\left[\pi S_{ab}(\omega)
    + i\,\mathrm{P}\!\!\int d\nu\,\frac{S_{ab}(\nu)}{\omega - \nu}\right]`:
    the absorptive part :math:`\pi S_{ab}(\omega)` (the vertical
    transitions at :math:`\hbar\omega`, for ``part='real'`` and
    :math:`a = b`) and the reactive part, a principal value over all
    :math:`\nu`, positive and negative (Kramers-Kronig). The principal
    value needs :math:`S_{ab}` on a grid of :math:`8M` frequencies; it is
    skipped when only :math:`\mathrm{Re}\,\sigma_{aa}` is asked for.

    * :math:`\mathrm{Re}\,\sigma_{xx}`: for graphene at neutrality, the
      universal :math:`\pi/4\,e^2/h` (spinless, two valleys) for
      :math:`\pi a/M \ll \omega \ll t`.
    * :math:`\mathrm{Im}\,\sigma_{xx}`: the reactive response, negative
      below the gap of an insulator (in the sign convention of
      *tbkit.optics*).
    * Hall components (*vel* = :math:`v_y`, *vel_b* = :math:`v_x`): in the
      gap of a Chern insulator, :math:`\mathrm{Re}\,\sigma_{yx}(\omega\to0)`
      is its Chern number :math:`C`, the value of *hall_conductivity* and
      *KSpace.hall_conductivity*; at finite frequency, the Faraday and Kerr
      rotations.

    The random trace is noisier than that of *dos*: its relative error
    falls as :math:`1/\sqrt{RN}` but from a larger prefactor, about 10% for
    :math:`R = 4` random vectors on 45,000 orbitals of graphene at
    :math:`\omega \sim 0.3t`, 2% for :math:`R = 16`.

    The trace must run over a sample without edges, a torus
    (*KSpace.finite_ham* and *KSpace.finite_velocity* with
    ``periodic=True``), or over one large enough for its edges not to matter.

    Parameters
    ----------
    ham : Square sparse Hermitian matrix
    vel : Square sparse (or dense) Hermitian matrix, like *ham*
        Velocity :math:`v_a = i[H, r_a]` (*KSpace.finite_velocity*), or any
        current operator.
    omega : Positive real, or array of positive reals
        Photon energies :math:`\hbar\omega`.
    e_fermi : Real number, default 0
        Fermi energy, inside the spectral bounds.
    temperature : Positive real or zero, default 0
        In energy units.
    n_moments : Positive integer, default 256
        Number of Chebyshev moments :math:`M` (energy resolution about
        :math:`\pi a/M`).
    n_random : Positive integer, or None, default 10
        See *dos*.
    kernel
        See *dos*.
    seed
        See *dos*.
    bounds
        See *dos*.
    area : Positive real
        Default value None (the number of orbitals). Area (volume)
        :math:`\Omega` of the sample.
    vel_b : Square sparse (or dense) Hermitian matrix, like *ham*
        Default value None (*vel*: :math:`\sigma_{aa}`). Velocity
        :math:`v_b`, for an off-diagonal (Hall) component :math:`\sigma_{ab}`.
    part : String, default 'real'
        'real' (:math:`\mathrm{Re}\,\sigma_{ab}`), 'imag'
        (:math:`\mathrm{Im}\,\sigma_{ab}`) or 'complex' (both).

    Returns
    -------
    sigma
        Real ndarray (complex for ``part='complex'``), shape
        (len(omega),), in units of :math:`e^2/h`.
    '''
    ham_s, _, _, a, b = _setup(ham, n_moments, kernel, bounds, [0.])
    n = ham_s.shape[0]
    error_handling.velocity(vel, n, 'vel')
    vel = sparse.csr_matrix(vel, dtype='c16')
    error_handling.hermitian(vel)
    diagonal = vel_b is None
    if not diagonal:
        error_handling.velocity(vel_b, n, 'vel_b')
        vel_b = sparse.csr_matrix(vel_b, dtype='c16')
        error_handling.hermitian(vel_b)
    error_handling.frequencies(omega, 'omega')
    w_arr = np.atleast_1d(np.asarray(omega, dtype='f8')).ravel()
    error_handling.positive_frequencies(w_arr)
    error_handling.real_number(e_fermi, 'e_fermi')
    error_handling.kpm_grid(np.array([(e_fermi - b) / a]), 'e_fermi')
    error_handling.positive_real_zero(temperature, 'temperature')
    if area is None:
        area = float(n)
    error_handling.positive_real(area, 'area')
    error_handling.optical_part(part)
    # mu_mn = Tr[v_b T_m v_a T_n]: k(E1, E2) = Tr[v_b delta(E1 - H) v_a delta(E2 - H)]
    # = j_ab(E2, E1), so that j_ab(E + nu, E) = k(E, E + nu)
    mu = _two_index_moments(ham_s, vel if diagonal else vel_b, vel, _vectors(n, n_random, seed), n_moments)
    if n_random is not None:
        mu = mu / n_random
    # on a fine energy grid; j_aa is real (sum_nm |v_nm|^2 delta delta)
    n_grid = max(1001, 8 * n_moments + 1)
    eps = np.linspace(-0.995, 0.995, n_grid)
    energies = a * eps + b
    coef = _coefficients(eps, n_moments, kernel, a)
    joint = coef.T @ (mu.real if diagonal else mu) @ coef
    lo, hi = energies[0], energies[-1]
    step = (hi - lo) / (n_grid - 1)

    def bilinear(x, y):
        # joint at (x, y), linear between the grid points in both directions
        fx = np.clip((x - lo) / step, 0., n_grid - 1.)
        fy = np.clip((y - lo) / step, 0., n_grid - 1.)
        i = np.minimum(fx.astype(int), n_grid - 2)
        j = np.minimum(fy.astype(int), n_grid - 2)
        tx, ty = fx - i, fy - j
        return ((1 - tx) * ((1 - ty) * joint[i, j] + ty * joint[i, j + 1])
                + tx * ((1 - ty) * joint[i + 1, j] + ty * joint[i + 1, j + 1]))

    def spectral(nu):
        # S(nu) = 1/Omega int dE j_ab(E + nu, E) [f(E) - f(E + nu)] / nu, for nu != 0
        if temperature == 0:
            e_lo = np.maximum(np.maximum(lo, lo - nu), e_fermi - np.maximum(nu, 0.))
            e_hi = np.minimum(np.minimum(hi, hi - nu), e_fermi + np.maximum(-nu, 0.))
        else:
            e_lo, e_hi = np.maximum(lo, lo - nu), np.minimum(hi, hi - nu)
        width = np.maximum(e_hi - e_lo, 0.)
        out = np.zeros(len(nu), 'c16')
        frac = np.linspace(0., 1., n_grid)
        rows = max(1, 2 ** 20 // n_grid)
        for c0 in range(0, len(nu), rows):
            sl = slice(c0, c0 + rows)
            e = e_lo[sl, None] + width[sl, None] * frac[None, :]
            if temperature == 0:
                occ = np.sign(nu[sl])[:, None] * np.ones_like(e)
            else:
                occ = (occupation.fermi_dirac(e, e_fermi, temperature)
                       - occupation.fermi_dirac(e + nu[sl, None], e_fermi, temperature))
            val = bilinear(e, e + nu[sl, None])
            out[sl] = trapezoid(occ * val, dx=1., axis=1) * width[sl] / (n_grid - 1)
        return out / (area * nu)

    s_w = spectral(w_arr)  # the absorptive part, pi S(omega)
    if part == 'real' and diagonal:
        return 2 * PI * PI * s_w.real  # e^2/hbar = 2 pi e^2/h
    # the reactive part, i P int S(nu) / (omega - nu), on a grid of
    # midpoints (never nu = 0) over the whole range of transition energies
    span = hi - lo
    n_half = max(512, 4 * n_moments)
    nu = -span + (np.arange(2 * n_half) + 0.5) * span / n_half
    s_nu = spectral(nu)
    reactive = np.zeros(len(w_arr), 'c16')
    for i, w in enumerate(w_arr):
        if w >= nu[-1]:  # no transition: S(omega) = 0, and no singularity
            reactive[i] = trapezoid(s_nu / (w - nu), nu)
            continue
        s0 = np.interp(w, nu, s_nu.real) + 1j * np.interp(w, nu, s_nu.imag)
        # subtract the pole: (S(nu) - S(omega)) / (omega - nu) is smooth (0 at
        # a grid point nu = omega), and P int dnu / (omega - nu) is a log
        d = w - nu
        integrand = (s_nu - s0) / np.where(d == 0, 1., d)
        reactive[i] = trapezoid(integrand, nu) + s0 * np.log((w + span) / (span - w))
    sigma = 2 * PI * (PI * s_w + 1j * reactive)
    if part == 'real':
        return sigma.real
    if part == 'imag':
        return sigma.imag
    return sigma
