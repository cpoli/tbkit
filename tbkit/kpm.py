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

import tbkit.error_handling as error_handling
import tbkit.occupation as occupation


PI = np.pi


def spectral_bounds(ham) -> tuple[float, float]:
    r'''
    Get bounds on the spectrum of a Hermitian matrix from Gershgorin's
    circle theorem: every eigenvalue lies within
    :math:`H_{ii} \pm \sum_{j\neq i}|H_{ij}|` for some :math:`i`.

    :param ham: Square sparse (or dense) Hermitian matrix.

    :returns:
        * **e_min**, **e_max** -- Real numbers.
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

    :param n_moments: Positive integer :math:`M`.

    :returns:
        * **g** -- Real ndarray, shape (n_moments,).
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

    :param n_moments: Positive integer :math:`M`.
    :param lam: Positive real. Default value 4.

    :returns:
        * **g** -- Real ndarray, shape (n_moments,).
    '''
    error_handling.positive_int(n_moments, 'n_moments')
    error_handling.positive_real(lam, 'lam')
    return np.sinh(lam * (1. - np.arange(n_moments) / n_moments)) / np.sinh(lam)


def _setup(ham, n_moments, kernel, bounds, e_grid, n_grid=1001):
    '''
    Private function. Validate, rescale the Hamiltonian, and build the
    coefficients c_m(E) (shape (n_moments, len(e_grid))).
    '''
    ham = sparse.csr_matrix(ham, dtype='c16')
    error_handling.square_matrix(ham, 'ham')
    error_handling.hermitian(ham)
    error_handling.positive_int(n_moments, 'n_moments')
    error_handling.kpm_kernel(kernel)
    if bounds is None:
        bounds = spectral_bounds(ham)
    error_handling.lims(bounds)
    e_min, e_max = bounds
    a = (e_max - e_min) / (2. - 0.02)  # keep the scaled spectrum inside (-0.99, 0.99)
    b = (e_max + e_min) / 2.
    if e_grid is None:
        e_grid = b + a * np.cos(PI * (np.arange(n_grid) + 0.5) / n_grid)[::-1]
    e_grid = np.atleast_1d(np.asarray(e_grid, dtype='f8'))
    error_handling.ndarray_empty(e_grid, 'e_grid')
    eps = (e_grid - b) / a
    error_handling.kpm_grid(eps)
    g = jackson_kernel(n_moments) if kernel == 'jackson' else lorentz_kernel(n_moments)
    tm = np.cos(np.arange(n_moments)[:, None] * np.arccos(eps)[None, :])
    weight = np.where(np.arange(n_moments) == 0, 1., 2.)
    coef = (weight * g)[:, None] * tm / (a * PI * np.sqrt(1. - eps ** 2))[None, :]
    ham_s = (ham - b * sparse.identity(ham.shape[0], format='csr')) / a
    return ham_s, e_grid, coef, a, b


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


def dos(
    ham, n_moments: int = 256, n_random: int | None = 10, e_grid: ArrayLike | None = None,
    kernel: str = 'jackson', seed=None, bounds: tuple[float, float] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    r'''
    Get the density of states :math:`\rho(E) = \mathrm{Tr}\,\delta(E - H)`
    (normalized, like *tbkit.dos.density_of_states*, to integrate to the
    number of states).

    :param ham: Square sparse Hermitian matrix (e.g. *System.ham*).
    :param n_moments: Positive integer. Default value 256. Number of
        Chebyshev moments (the energy resolution is about
        :math:`\pi a/M`, :math:`2a` the spectral width).
    :param n_random: Positive integer, or None. Default value 10. Number of
        random vectors for the trace; None for the exact trace (N vectors:
        small systems only).
    :param e_grid: Real array. Default value None: 1001 Chebyshev nodes over
        the spectrum. Energies, inside the spectral bounds.
    :param kernel: String. Default value 'jackson'. 'jackson' or 'lorentz'.
    :param seed: Default value None. Seed of the random vectors.
    :param bounds: Pair of reals. Default value None (Gershgorin bounds, see
        *spectral_bounds*). Bounds enclosing the whole spectrum.

    :returns:
        * **e_grid** -- Real ndarray.
        * **rho** -- Real ndarray, same shape.
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

    :param levels: Real array. Energy levels.
    :param n_moments: Positive integer. Number of Chebyshev moments.
    :param bounds: Pair of reals. Spectral bounds (those of *dos*, e.g.
        *spectral_bounds* of the Hamiltonian it is compared with).
    :param e_grid: Real array. Energies.
    :param kernel: String. Default value 'jackson'.

    :returns:
        * **rho** -- Real ndarray, same shape as *e_grid* (integrating to the number of levels).
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

    :param ham: Square sparse Hermitian matrix.
    :param site: Integer. Site (row) index.
    :param n_moments: Positive integer. Default value 256.
    :param e_grid: See *dos*.
    :param kernel: See *dos*.
    :param bounds: See *dos*.

    :returns:
        * **e_grid** -- Real ndarray.
        * **rho** -- Real ndarray, same shape.
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

    :param ham: Square sparse Hermitian matrix.
    :param x: Real array, one coordinate per site (e.g. *lat.coor['x']*).
    :param n_moments: Positive integer. Default value 128.
    :param n_random: Positive integer, or None. Default value 10. See *dos*.
    :param e_grid: See *dos*.
    :param kernel: See *dos*.
    :param seed: See *dos*.
    :param bounds: See *dos*.
    :param area: Positive real. Default value None (the number of sites):
        the area (volume) :math:`\Omega` of the sample.

    :returns:
        * **e_grid** -- Real ndarray.
        * **sigma** -- Real ndarray, same shape.
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
    vec = _vectors(n, n_random, seed)
    mu = np.zeros((n_moments, n_moments), 'c16')
    for r in vec.T:
        left = np.array(list(_chebyshev(ham_s, vel @ r, n_moments)))  # T_m v |r>
        right = np.array([vel @ t for t in _chebyshev(ham_s, r, n_moments)])  # v T_n |r>
        mu += left.conj() @ right.T
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

    :param ham: Square sparse Hermitian matrix.
    :param vx: Square sparse (or dense) Hermitian matrix, like *ham*. Velocity
        :math:`v_x = i[H, x]` (see *KSpace.finite_velocity*). For a spin
        Hall conductivity, pass the spin current :math:`\{s, v_x\}/2`
        instead (units :math:`e/2\pi`, see *KSpace.spin_hall_conductivity*).
    :param vy: Square sparse (or dense) Hermitian matrix. Velocity :math:`v_y`.
    :param n_moments: Positive integer. Default value 256. Number of
        Chebyshev moments :math:`M` (energy resolution about :math:`\pi a/M`).
    :param n_random: Positive integer, or None. Default value 10. See *dos*.
    :param e_grid: Real array. Default value None: 1001 Chebyshev nodes over
        the spectrum. Fermi energies :math:`\mu`.
    :param kernel: String. Default value 'jackson'. See *dos*.
    :param seed: See *dos*.
    :param bounds: See *dos*.
    :param area: Positive real. Default value None (the number of orbitals).
        Area :math:`\Omega` of the sample (e.g. number of cells times the
        cell area).
    :param temperature: Positive real or zero. Default value 0. In energy units.

    :returns:
        * **e_grid** -- Real ndarray.
        * **sigma** -- Real ndarray, same shape, in units of :math:`e^2/h`.
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
        sigma = prefactor * np.trapezoid(f * integrand[None, :], eps, axis=1)
    return e_grid, sigma

