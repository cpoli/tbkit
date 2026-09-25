r"""
Floquet theory of periodically driven Tight-Binding models,
:math:`H(t + T) = H(t)`, :math:`\omega = 2\pi/T` (:math:`\hbar = 1`).

The evolution over one period, :math:`U(T) = \mathcal{T}e^{-i\int_0^TH(t)dt}
= e^{-iH_FT}`, defines the Floquet Hamiltonian :math:`H_F`, whose
eigenvalues -- the quasienergies :math:`\epsilon`, defined modulo
:math:`\omega` -- replace the energies of a static model: the driven
system, observed once per period, behaves as if it were static with
Hamiltonian :math:`H_F`. Driving can therefore *engineer* band structures
and their topology: circularly polarized light turns graphene into a
Chern insulator (Oka and Aoki, Phys. Rev. B 79, 081406 (2009); Kitagawa,
Oka, Brataas, Fu and Demler, Phys. Rev. B 84, 235108 (2011)).

Two routes give the quasienergies, checked against each other in the tests:

* the time-ordered product of short-time propagators (*evolution_operator*,
  *quasienergies*, *effective_hamiltonian*);
* the Sambe (extended) space of the harmonics
  :math:`H(t) = \sum_m H_m e^{-im\omega t}`, a static eigenvalue problem
  (*harmonics*, *sambe_hamiltonian*).

:class:`FloquetKSpace` makes a driven **KSpace** model: its *get_ham* is
the Floquet Hamiltonian :math:`H_F(\mathbf{k})`, so its bands, Berry
curvature and Chern numbers are the driven ones.

Anomalous Floquet phases. The quasienergies are defined modulo
:math:`\omega`, so :math:`H_F` depends on where the branch cut of the
logarithm is put (the *epsilon* arguments). The Chern numbers of the
Floquet bands no longer count the edge states: every band can have
:math:`C = 0` while chiral edge states cross every gap (Kitagawa, Berg,
Rudner and Demler, Phys. Rev. B 82, 235114 (2010)). The winding number of
Rudner, Lindner, Berg and Levin (Phys. Rev. X 3, 031005 (2013)), which
uses the evolution at *all* times of the period, does
(*DrivenKSpace.winding_number*):

* :class:`DrivenKSpace` makes a driven Bloch model out of any
  time-dependent :math:`H(\mathbf{k}, t)`;
* *step_drive* builds a piecewise-constant drive from a list of models
  (**KSpace** models, ribbons made by *kspace.ribbon*, real-space
  **System** instances, or matrices), evolved exactly as a product of
  step exponentials (:class:`StepDrive` in real space);
* *DrivenKSpace.edge_state_count* counts the chiral edge states of a
  driven ribbon, independently of the bulk winding number.
"""
from __future__ import annotations

import copy
from typing import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray
import scipy.linalg as LA

import tbkit.error_handling as error_handling


def evolution_operator(ham_t: Callable, period: float, n_steps: int = 100) -> NDArray[np.complex128]:
    r'''
    Get the one-period evolution operator
    :math:`U(T) = \prod_m e^{-iH(t_m)\,\delta t}` (later times on the left,
    midpoints :math:`t_m = (m + 1/2)\delta t`, error :math:`O(\delta t^2)`).

    :param ham_t: Callable. ``ham_t(t)`` returns the Hamiltonian at time t (a square matrix).
    :param period: Positive real. :math:`T`.
    :param n_steps: Positive integer. Default value 100. Time steps per period.

    :returns:
        * **U** -- Complex ndarray, the shape of the Hamiltonian.
    '''
    error_handling.is_callable(ham_t, 'ham_t')
    error_handling.positive_real(period, 'period')
    error_handling.positive_int(n_steps, 'n_steps')
    dt = period / n_steps
    u = None
    for m in range(n_steps):
        h = np.asarray(ham_t((m + 0.5) * dt), dtype='c16')
        step = LA.expm(-1j * dt * h)
        u = step if u is None else step @ u
    return u


def _fold(eps: NDArray[np.float64], period: float, epsilon: float) -> NDArray[np.float64]:
    '''
    Private function. Quasienergies folded into [epsilon - 2 pi/T, epsilon).
    '''
    omega = 2 * np.pi / period
    rest = np.mod(eps - epsilon, omega)
    rest = np.where(rest >= omega, rest - omega, rest)  # np.mod may round up to omega
    return epsilon - omega + rest


def _floquet(
    u: NDArray[np.complex128], period: float, epsilon: float | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.complex128]]:
    '''
    Private function. Quasienergies (sorted, in [-pi/T, pi/T), or in
    [epsilon - 2 pi/T, epsilon)) and Floquet states of a unitary
    U = exp(-i H_F T), from its complex Schur form.
    '''
    tri, z = LA.schur(u, output='complex')
    eps = -np.angle(np.diag(tri)) / period
    if epsilon is not None:
        eps = _fold(eps, period, epsilon)
    order = np.argsort(eps, kind='stable')
    return eps[order], z[:, order]


def _heff(u: NDArray[np.complex128], period: float, epsilon: float | None = None) -> NDArray[np.complex128]:
    '''
    Private function. The Floquet Hamiltonian (i/T) ln U, branch cut at epsilon.
    '''
    eps, z = _floquet(u, period, epsilon)
    h_f = z @ np.diag(eps) @ z.conj().T
    return (h_f + h_f.conj().T) / 2


def quasienergies(
    ham_t: Callable, period: float, n_steps: int = 100, epsilon: float | None = None,
) -> NDArray[np.float64]:
    r'''
    Get the quasienergies :math:`\epsilon`, eigenvalues of the Floquet
    Hamiltonian, in the first Floquet zone :math:`[-\omega/2, \omega/2)`,
    or in :math:`[\epsilon_{cut} - \omega, \epsilon_{cut})`.

    :param ham_t: Callable. See *evolution_operator*.
    :param period: Positive real.
    :param n_steps: Positive integer. Default value 100.
    :param epsilon: Real number. Default value None (the zone
        :math:`[-\omega/2, \omega/2)`). Branch cut, see *effective_hamiltonian*.

    :returns:
        * **eps** -- Real ndarray, sorted.
    '''
    error_handling.is_callable(ham_t, 'ham_t')
    error_handling.positive_real(period, 'period')
    error_handling.positive_int(n_steps, 'n_steps')
    error_handling.branch_cut(epsilon)
    return _floquet(evolution_operator(ham_t, period, n_steps), period, epsilon)[0]


def effective_hamiltonian(
    ham_t: Callable, period: float, n_steps: int = 100, epsilon: float | None = None,
) -> NDArray[np.complex128]:
    r'''
    Get the Floquet Hamiltonian :math:`H_F = \frac{i}{T}\ln U(T)`, with its
    quasienergies in the first Floquet zone (Hermitian for a Hermitian drive).

    The logarithm is defined up to multiples of :math:`\omega` on each
    eigenvalue: *epsilon* puts its branch cut at the quasienergy
    :math:`\epsilon_{cut}` (the phase :math:`e^{-i\epsilon_{cut}T}` of
    :math:`U`), so that the quasienergies lie in
    :math:`[\epsilon_{cut} - \omega, \epsilon_{cut})`. The default, the zone
    :math:`[-\omega/2, \omega/2)`, is the cut :math:`\epsilon_{cut} = \omega/2`.
    Rudner et al. (2013) take :math:`(\epsilon_{cut}, \epsilon_{cut} + \omega]`
    instead: :math:`H_F` shifted by :math:`\omega` (the same projectors,
    so the same Chern numbers and winding numbers).

    :param ham_t: Callable. See *evolution_operator*.
    :param period: Positive real.
    :param n_steps: Positive integer. Default value 100.
    :param epsilon: Real number. Default value None (:math:`\epsilon_{cut} = \omega/2`,
        computed exactly as before this option existed). Branch cut,
        best placed in a quasienergy gap.

    :returns:
        * **H_F** -- Complex ndarray.
    '''
    error_handling.is_callable(ham_t, 'ham_t')
    error_handling.positive_real(period, 'period')
    error_handling.positive_int(n_steps, 'n_steps')
    error_handling.branch_cut(epsilon)
    return _heff(evolution_operator(ham_t, period, n_steps), period, epsilon)


def harmonics(ham_t: Callable, period: float, n_harmonics: int, n_samples: int | None = None) -> dict:
    r'''
    Get the Fourier components :math:`H_m = \frac1T\int_0^T H(t)\,e^{im\omega t}\,dt`
    of the drive, :math:`H(t) = \sum_m H_m e^{-im\omega t}`, for
    :math:`|m| \le` *n_harmonics* (exact for a drive with no higher harmonics).

    :param ham_t: Callable. See *evolution_operator*.
    :param period: Positive real.
    :param n_harmonics: Positive integer or zero.
    :param n_samples: Positive integer. Default value None (4 n_harmonics + 16).
        Time samples per period.

    :returns:
        * **H_m** -- Dictionary {m: complex ndarray}.
    '''
    error_handling.is_callable(ham_t, 'ham_t')
    error_handling.positive_real(period, 'period')
    error_handling.positive_int_zero(n_harmonics, 'n_harmonics')
    if n_samples is None:
        n_samples = 4 * n_harmonics + 16
    error_handling.positive_int(n_samples, 'n_samples')
    omega = 2 * np.pi / period
    ts = period * np.arange(n_samples) / n_samples
    hams = np.array([np.asarray(ham_t(t), dtype='c16') for t in ts])
    return {m: np.mean(hams * np.exp(1j * m * omega * ts)[:, None, None], axis=0)
               for m in range(-n_harmonics, n_harmonics + 1)}


def sambe_hamiltonian(harm: dict, omega: float, n_max: int) -> NDArray[np.complex128]:
    r'''
    Get the Sambe-space Floquet Hamiltonian, truncated to the photon
    sectors :math:`n = -n_{max}, \dots, n_{max}`:

    .. math::

        (\mathcal{H})_{nn'} = H_{n-n'} + n\omega\,\delta_{nn'}\, .

    Its eigenvalues are the quasienergies, repeated in every Floquet zone
    (shifted by multiples of :math:`\omega`); those of the central zone
    converge fastest as *n_max* grows.

    :param harm: Dictionary {m: square complex array}, see *harmonics*.
    :param omega: Positive real. Driving frequency.
    :param n_max: Positive integer or zero.

    :returns:
        * **H** -- Complex ndarray, shape ((2 n_max + 1) d, (2 n_max + 1) d).
    '''
    error_handling.harmonics(harm)
    error_handling.positive_real(omega, 'omega')
    error_handling.positive_int_zero(n_max, 'n_max')
    d = len(harm[0])
    size = 2 * n_max + 1
    out = np.zeros((size * d, size * d), 'c16')
    for a, n in enumerate(range(-n_max, n_max + 1)):
        for b, n2 in enumerate(range(-n_max, n_max + 1)):
            block = np.asarray(harm.get(n - n2, np.zeros((d, d))), dtype='c16')
            if n == n2:
                block = block + n * omega * np.eye(d)
            out[a*d:(a+1)*d, b*d:(b+1)*d] = block
    return out


# ----------------------------------------------------------------------
# Piecewise-constant (step) drives
# ----------------------------------------------------------------------


def _step_evolution(steps: list, t: float | None = None) -> NDArray[np.complex128]:
    r'''
    Private function. The exact evolution of a piecewise-constant drive,
    *steps* = [(H_1, dt_1), (H_2, dt_2), ...] in time order:
    U(t) = exp(-i H_n (t - t_{n-1})) ... exp(-i H_1 dt_1), U(T) if t is None.
    '''
    u = np.eye(len(steps[0][0]), dtype='c16')
    elapsed = 0.
    for h, dt in steps:
        tau = dt if t is None else min(dt, t - elapsed)
        if tau <= 0:
            break
        u = LA.expm(-1j * tau * h) @ u
        elapsed += dt
    return u


class StepDrive:
    r'''
    A finite (real-space) system driven by a piecewise-constant
    Hamiltonian: :math:`H(t) = H_n` for
    :math:`t_{n-1} \le t < t_n = \delta t_1 + \dots + \delta t_n`, repeated
    with period :math:`T = \sum_n \delta t_n`. The evolution is exact, a
    product of step exponentials,

    .. math::

        U(t) = e^{-iH_n(t - t_{n-1})}\,e^{-iH_{n-1}\delta t_{n-1}}\cdots e^{-iH_1\delta t_1}\, ,

    with no time grid. Built by *step_drive* from **System** instances or
    matrices (see *DrivenKSpace* for Bloch models and ribbons).

    :param hams: List of square complex ndarrays, one Hermitian matrix per step (all of one shape).
    :param durations: List of positive reals, the step durations :math:`\delta t_n`.
    :param epsilon: Real number. Default value None. Default branch cut of
        *quasienergies* and *effective_hamiltonian*.
    '''

    def __init__(self, hams: list, durations: ArrayLike, epsilon: float | None = None) -> None:
        error_handling.step_matrices(hams)
        error_handling.durations(durations, len(hams))
        error_handling.branch_cut(epsilon)
        self.hams = [np.asarray(h, dtype='c16') for h in hams]
        self.durations = np.array(durations, dtype='f8')
        self.period = float(self.durations.sum())
        self.epsilon = epsilon
        self.n_sites = len(self.hams[0])

    def get_ham(self, t: float) -> NDArray[np.complex128]:
        r'''
        Get the Hamiltonian :math:`H(t)` (periodic in :math:`T`).

        :param t: Real number. Time.

        :returns:
            * **H** -- Complex ndarray.
        '''
        error_handling.real_number(t, 't')
        n = np.searchsorted(np.cumsum(self.durations), t % self.period, side='right')
        return self.hams[min(n, len(self.hams) - 1)]

    def evolution_operator(self, t: float | None = None) -> NDArray[np.complex128]:
        r'''
        Get the exact evolution operator :math:`U(t)` from time 0.

        :param t: Real number in :math:`[0, T]`. Default value None (one period, :math:`U(T)`).

        :returns:
            * **U** -- Complex ndarray.
        '''
        error_handling.time_in_period(t, self.period)
        return _step_evolution(list(zip(self.hams, self.durations)), t)

    def quasienergies(self, epsilon: float | None = None) -> NDArray[np.float64]:
        r'''
        Get the quasienergies, sorted, in :math:`[-\omega/2, \omega/2)` or
        :math:`[\epsilon_{cut} - \omega, \epsilon_{cut})` (see *effective_hamiltonian*).

        :param epsilon: Real number. Default value None (the *epsilon* of the drive).

        :returns:
            * **eps** -- Real ndarray.
        '''
        error_handling.branch_cut(epsilon)
        epsilon = self.epsilon if epsilon is None else epsilon
        return _floquet(self.evolution_operator(), self.period, epsilon)[0]

    def effective_hamiltonian(self, epsilon: float | None = None) -> NDArray[np.complex128]:
        r'''
        Get the Floquet Hamiltonian :math:`H_F = \frac{i}{T}\ln U(T)`, branch
        cut at *epsilon* (see *tbkit.floquet.effective_hamiltonian*).

        :param epsilon: Real number. Default value None (the *epsilon* of the drive).

        :returns:
            * **H_F** -- Complex ndarray.
        '''
        error_handling.branch_cut(epsilon)
        epsilon = self.epsilon if epsilon is None else epsilon
        return _heff(self.evolution_operator(), self.period, epsilon)

    def evolve(self, psi: NDArray, n_periods: int) -> NDArray[np.complex128]:
        r'''
        Evolve a state stroboscopically, :math:`\psi(nT) = U(T)^n\psi(0)`.

        :param psi: Complex ndarray, the initial state (one amplitude per site).
        :param n_periods: Positive integer or zero.

        :returns:
            * **psi_n** -- Complex ndarray, shape (n_periods + 1, n_sites):
              the state after 0, 1, ..., n_periods periods.
        '''
        error_handling.ndarray(psi, 'psi', self.n_sites)
        error_handling.positive_int_zero(n_periods, 'n_periods')
        u = self.evolution_operator()
        out = np.zeros((n_periods + 1, self.n_sites), 'c16')
        out[0] = psi
        for n in range(n_periods):
            out[n + 1] = u @ out[n]
        return out


def step_drive(models: list, durations: ArrayLike, epsilon: float | None = None):
    r'''
    Build a piecewise-constant drive: the Hamiltonian of ``models[n]`` acts
    for ``durations[n]``, in order, and the cycle repeats with period
    :math:`T = \sum_n \delta t_n`. The evolution is exact (a product of step
    exponentials, no time grid, see *StepDrive*).

    :param models: List of models of one size, all of one kind:

        * **KSpace** instances (including ribbons made by *kspace.ribbon*)
          -- returns a *DrivenKSpace*, whose *get_ham* is the Floquet
          Hamiltonian :math:`H_F(\mathbf{k})`;
        * **System** instances (their *get_ham* is called) or square
          matrices (dense or scipy.sparse) -- returns a *StepDrive*.

        Each must be Hermitian; KSpace models need an orthogonal basis.
    :param durations: List of positive reals. :math:`\delta t_n`, one per model.
    :param epsilon: Real number. Default value None. Branch cut of the
        Floquet Hamiltonian (see *effective_hamiltonian*).

    :returns:
        * **drive** -- *DrivenKSpace* or *StepDrive*.

    Example usage::

        # Rudner et al. (2013): hopping along one bond direction per step
        drive = step_drive([ks1, ks2, ks3, ks4, ks5], [T/5] * 5)
        w = drive.winding_number(epsilon=np.pi / T)
    '''
    from tbkit.kspace import KSpace
    from tbkit.system import System
    kind = error_handling.step_models(models, KSpace, System)
    error_handling.durations(durations, len(models))
    error_handling.branch_cut(epsilon)
    if kind == 'kspace':
        return DrivenKSpace._from_steps(models, durations, epsilon)
    hams = []
    for m in models:
        if isinstance(m, System):
            m.get_ham()
            m = m.ham
        hams.append(m.toarray() if hasattr(m, 'toarray') else np.asarray(m))
    return StepDrive(hams, durations, epsilon)


# ----------------------------------------------------------------------
# Driven Bloch models
# ----------------------------------------------------------------------


def _bloch_grid(model, ks: NDArray[np.float64], a: ArrayLike | None = None) -> NDArray[np.complex128]:
    r'''
    Private function. The Bloch Hamiltonians of a static **KSpace** at many
    k-points at once, shape (nk, norb, norb): *get_ham* (or, with a vector
    potential *a*, *get_ham_peierls*) vectorized over k. A model whose class
    redefines these methods is evaluated point by point.
    '''
    from tbkit.kspace import KSpace
    if type(model).get_ham is not KSpace.get_ham or type(model).get_ham_peierls is not KSpace.get_ham_peierls:
        if a is None:
            return np.array([model.get_ham(k) for k in ks])
        return np.array([model.get_ham_peierls(k, a) for k in ks])
    return model._bloch_ham(ks @ model.k_basis.T, None if a is None else np.asarray(a, dtype='f8'))


def _spectral_derivative(u: NDArray[np.complex128], axis: int) -> NDArray[np.complex128]:
    r'''
    Private function. Derivative of a function sampled on a uniform grid
    of one period [0, 1) along *axis* (exact for a trigonometric
    polynomial, exponentially accurate for an analytic periodic function).
    '''
    n = u.shape[axis]
    m = np.fft.fftfreq(n, 1. / n)
    if n % 2 == 0:
        m[n // 2] = 0.  # the Nyquist mode has no odd derivative
    shape = [1] * u.ndim
    shape[axis] = n
    return np.fft.ifft(np.fft.fft(u, axis=axis) * (2j * np.pi * m).reshape(shape), axis=axis)


def _winding_density(u: NDArray[np.complex128], a_t: NDArray[np.complex128]) -> float:
    r'''
    Private function. Brillouin-zone average of
    Tr(A_t [A_1, A_2]), A_i = U^-1 d U / d f_i along the fractional
    coordinates f_1, f_2 (u has shape (n1, n2, N, N)).
    '''
    u_inv = np.conj(np.swapaxes(u, -1, -2))
    a1 = u_inv @ _spectral_derivative(u, 0)
    a2 = u_inv @ _spectral_derivative(u, 1)
    return float(np.mean(np.trace(a_t @ (a1 @ a2 - a2 @ a1), axis1=-2, axis2=-1)).real)


class DrivenKSpace:
    r'''
    A periodic model driven by any time-periodic Bloch Hamiltonian
    :math:`H(\mathbf{k}, t + T) = H(\mathbf{k}, t)`.

    It is a copy of the **KSpace** model whose *get_ham* returns the Floquet
    Hamiltonian :math:`H_F(\mathbf{k}) = \frac{i}{T}\ln U(\mathbf{k}, T)`
    (branch cut at *epsilon*, see *effective_hamiltonian*), so *get_bands*,
    *k_path*, *berry_curvature*, *chern_number*, ... describe the driven
    system, and *winding_number* gives its anomalous topology. The
    evolution :math:`U(\mathbf{k}, T)` is the product of *n_steps*
    midpoint propagators (see *evolution_operator*); drives built by
    *step_drive* are evolved exactly. Real-space methods that need static
    hoppings (*finite_ham*, *get_ham_beta*, *gbz*) are not available.

    :param ks: **KSpace** instance. The model driven: its lattice, orbitals
        and k-space conventions (periodic gauge) are those of the driven model.
    :param ham_kt: Callable. ``ham_kt(k, t)`` returns the Hermitian Bloch
        Hamiltonian :math:`H(\mathbf{k}, t)`, shape (norb, norb), in the
        periodic gauge of *ks* (:math:`H(\mathbf{k} + \mathbf{G}, t) = H(\mathbf{k}, t)`).
    :param period: Positive real. :math:`T`.
    :param n_steps: Positive integer. Default value 100. Time steps per period.
    :param epsilon: Real number. Default value None (quasienergies in
        :math:`[-\omega/2, \omega/2)`). Branch cut of *get_ham*.

    Example usage::

        # a sublattice mass switched on and off
        drive = DrivenKSpace(ks, lambda k, t: ks.get_ham(k) + m * np.cos(w * t) * sz, 2 * np.pi / w)
    '''

    def __new__(cls, ks, ham_kt: Callable, period: float, n_steps: int = 100, epsilon: float | None = None):
        from tbkit.kspace import KSpace
        error_handling.kspace(ks, KSpace)
        error_handling.is_callable(ham_kt, 'ham_kt')
        error_handling.positive_real(period, 'period')
        error_handling.positive_int(n_steps, 'n_steps')
        error_handling.branch_cut(epsilon)
        error_handling.drive_matrix(ham_kt(np.zeros(ks.dim), 0.), ks.norb)
        obj = cls._copy(ks, period, epsilon)
        obj.ham_kt = ham_kt
        obj.n_steps = n_steps
        return obj

    def __init__(self, ks, ham_kt, period, n_steps=100, epsilon=None) -> None:
        pass

    @classmethod
    def _copy(cls, ks, period: float, epsilon: float | None):
        '''
        Private method. A copy of *ks* whose class also derives from *cls*.
        '''
        driven_cls = type(cls.__name__, (cls, type(ks)), {})
        obj = object.__new__(driven_cls)
        obj.__dict__.update(copy.deepcopy(ks.__dict__))
        obj.static = ks
        obj.period = period
        obj.epsilon = epsilon
        obj.steps = None
        return obj

    @classmethod
    def _from_steps(cls, models: list, durations: ArrayLike, epsilon: float | None):
        '''
        Private method. The drive of *step_drive*: models[n] for durations[n].
        '''
        obj = cls._copy(models[0], float(np.sum(durations)), epsilon)
        obj.steps = list(models)
        obj.durations = np.array(durations, dtype='f8')
        obj.n_steps = len(models)
        obj.ham_kt = obj._step_ham
        return obj

    def _step_ham(self, k: ArrayLike, t: float) -> NDArray[np.complex128]:
        '''
        Private method. H(k, t) of a step drive.
        '''
        n = np.searchsorted(np.cumsum(self.durations), t % self.period, side='right')
        return self.steps[min(n, len(self.steps) - 1)].get_ham(k)

    def _step_list(self, k: NDArray[np.float64]) -> list:
        '''
        Private method. The drive at k as steps [(H_n(k), dt_n)]: exact for
        a step drive, the midpoint propagators otherwise.
        '''
        if self.steps is not None:
            return [(m.get_ham(k), d) for m, d in zip(self.steps, self.durations)]
        dt = self.period / self.n_steps
        return [(np.asarray(self.ham_kt(k, (m + 0.5) * dt), dtype='c16'), dt) for m in range(self.n_steps)]

    def _grid_steps(self, ks: NDArray[np.float64]) -> tuple[NDArray[np.complex128], list]:
        '''
        Private method. The steps of *_step_list* at many k-points at once:
        the Hamiltonians, shape (nk, n_steps, norb, norb), and the durations.
        '''
        if self.steps is not None:
            return np.stack([_bloch_grid(m, ks) for m in self.steps], axis=1), list(self.durations)
        dt = self.period / self.n_steps
        hams = np.array([[s[0] for s in self._step_list(k)] for k in ks])
        return hams, [dt] * self.n_steps

    def get_evolution(self, k: ArrayLike, t: float | None = None) -> NDArray[np.complex128]:
        r'''
        Get the evolution operator :math:`U(\mathbf{k}, t) = \mathcal{T}e^{-i\int_0^tH(\mathbf{k},t')dt'}`.

        :param k: k point (see *KSpace.get_ham*).
        :param t: Real number in :math:`[0, T]`. Default value None (one
            period: the Floquet operator :math:`U(\mathbf{k}, T)`).

        :returns:
            * **U** -- Complex ndarray, shape (norb, norb).
        '''
        error_handling.k_vector(k, 'k', self.dim)
        error_handling.time_in_period(t, self.period)
        k = np.asarray(k, dtype='f8')
        if t is None and self.steps is None:
            return evolution_operator(lambda tt: self.ham_kt(k, tt), self.period, self.n_steps)
        return _step_evolution(self._step_list(k), t)

    def get_ham(self, k: ArrayLike) -> NDArray[np.complex128]:
        r'''
        Get the Floquet Hamiltonian :math:`H_F(\mathbf{k})`, quasienergies in
        :math:`[\epsilon_{cut} - \omega, \epsilon_{cut})` (default
        :math:`[-\omega/2, \omega/2)`).

        :param k: k point (see *KSpace.get_ham*).

        :returns:
            * **H_F** -- Complex ndarray, shape (norb, norb).
        '''
        error_handling.k_vector(k, 'k', self.dim)
        return _heff(self.get_evolution(k), self.period, self.epsilon)

    def is_hermitian(self) -> bool:
        '''
        The Floquet Hamiltonian of a Hermitian drive is Hermitian.
        '''
        return True

    def _hop_cells(self):
        error_handling.not_floquet()

    def _check_static(self):
        error_handling.not_floquet()

    def winding_number(self, epsilon: float, nk: int | tuple[int, int] = 48, n_t: int = 40) -> float:
        r'''
        Get the winding number of Rudner, Lindner, Berg and Levin (Phys. Rev.
        X 3, 031005 (2013)), the number of chiral edge states in the
        quasienergy gap at *epsilon*:

        .. math::

            W[U_\epsilon] = \frac{1}{8\pi^2}\int_0^T dt\int_{BZ} dk_x dk_y\,
            \mathrm{Tr}\left(U_\epsilon^{-1}\partial_tU_\epsilon
            \left[U_\epsilon^{-1}\partial_{k_x}U_\epsilon, U_\epsilon^{-1}\partial_{k_y}U_\epsilon\right]\right),

        of the loop :math:`U_\epsilon(\mathbf{k}, t) = U(\mathbf{k}, 2t)` for
        :math:`t \le T/2`, then :math:`V_\epsilon(\mathbf{k}, 2T - 2t)`, with
        the return map :math:`V_\epsilon(\mathbf{k}, t) = e^{-iH_F^\epsilon(\mathbf{k})t}`
        and :math:`H_F^\epsilon` the Floquet Hamiltonian with its branch cut
        at *epsilon* (see *effective_hamiltonian*). As the integrand is a
        3-form, this is the winding of :math:`U` over :math:`[0, T]` minus
        that of :math:`V_\epsilon` over :math:`[0, T]`.

        Signs. :math:`W` counts the chiral edge states in the gap:
        :math:`W = n_{lower} = -n_{upper}` of *edge_state_count*, so
        :math:`W > 0` means states moving along :math:`+\mathbf{a}_1` on
        the edge of a ribbon on the side of :math:`-\hat{z}\times\mathbf{a}_1`
        (the bottom edge of a ribbon along x). Across a group of Floquet bands,
        :math:`W[U_{\epsilon'}] - W[U_\epsilon] = -\sum_n C_n` for
        :math:`\epsilon < \epsilon'`, the sum over the bands between the two
        gaps, with the Chern numbers :math:`C_n` of *chern_number* (the
        paper's :math:`\mathcal{C}_n = -\frac{1}{2\pi}\int\nabla\times\mathcal{A}_n`
        is minus ours, so there :math:`W' - W = +\sum\mathcal{C}`). A
        static model (whose bands fit in the zone below the cut) has
        :math:`W = 0` at the zone edge, and :math:`W = -\sum C` of the
        bands below any other gap. The anomalous phases are those with
        :math:`W \ne 0` in every gap while all the :math:`C_n = 0`.

        Discretization: :math:`U(\mathbf{k}, t)` on a uniform
        :math:`n_1\times n_2` Brillouin-zone grid, differentiated spectrally
        in :math:`\mathbf{k}` (exponentially accurate: both :math:`U` and,
        while the gap at *epsilon* is open, :math:`V_\epsilon` are analytic
        and periodic in :math:`\mathbf{k}`), and :math:`\partial_tU = -iHU`
        exactly, integrated by Gauss-Legendre quadrature on each time step
        (*n_t* nodes per period in total, at least 2 per step) and over
        :math:`[0, T]` for the return map. Step drives are exact; a
        continuous drive is replaced by its *n_steps* midpoint steps, whose
        loop has the same :math:`W` as long as the gap stays open.

        Tolerance. The result is returned raw. Its error falls off
        exponentially with *nk*, at a rate set by the k-width of the
        region where :math:`V_\epsilon` varies fast, gap / band velocity
        near the smallest gap at *epsilon* (the Berry curvature is
        concentrated there). For the five-step model
        (:math:`\delta_{AB} = 0.5\pi/T`) the error is below
        :math:`10^{-12}` at perfect transfer; with a gap of
        :math:`0.067\omega` it is :math:`2\times10^{-5}` at the default
        *nk* = 48; with a gap of :math:`0.021\omega` it is
        :math:`10^{-2}` at *nk* = 48 and :math:`4\times10^{-4}` at
        *nk* = 96. Circularly driven graphene (gap :math:`0.01\omega` at K)
        needs *nk* = 96 for :math:`3\times10^{-3}`. Rounding to the nearest
        integer is trustworthy when :math:`|W - \mathrm{round}(W)| < 0.01` and the
        value is unchanged within that tolerance when *nk* is doubled; a
        value far from an integer means the gap is too small for the grid
        (or closed).

        :param epsilon: Real number. Quasienergy of the gap (branch cut).
        :param nk: Positive integer (at least 4), or tuple of 2. Default
            value 48. k-points along each reciprocal vector.
        :param n_t: Positive integer. Default value 40. Gauss-Legendre
            nodes per period (at least 2 per step; 2 n_t, and at least 24,
            for the return map).

        :returns:
            * **W** -- Real number, close to an integer.
        '''
        error_handling.dim_exact(self.dim, 2)
        error_handling.real_number(epsilon, 'epsilon')
        error_handling.nk(nk, 2)
        if isinstance(nk, int):
            nk = (nk, nk)
        error_handling.nk_min(nk, 4)
        error_handling.positive_int(n_t, 'n_t')
        n1, n2 = nk
        fracs = np.stack(np.meshgrid(np.arange(n1) / n1, np.arange(n2) / n2, indexing='ij'), -1)
        ks = fracs @ self.rec_vec_k
        hams, durations = self._grid_steps(ks.reshape(-1, 2))
        hams = hams.reshape(n1, n2, len(durations), self.norb, self.norb)
        x, w = np.polynomial.legendre.leggauss(max(2, -(-n_t // len(durations))))
        # the driven evolution U(k, t), 0 <= t <= T
        u = np.broadcast_to(np.eye(self.norb, dtype='c16'), (n1, n2, self.norb, self.norb))
        integral = 0.
        for s, dt in enumerate(durations):
            h = hams[:, :, s]
            e, q = np.linalg.eigh(h)
            qh = np.conj(np.swapaxes(q, -1, -2))
            for xj, wj in zip(x, w):
                ut = (q * np.exp(-1j * e * dt * (1 + xj) / 2)[..., None, :]) @ qh @ u
                a_t = -1j * np.conj(np.swapaxes(ut, -1, -2)) @ h @ ut
                integral += wj * dt / 2 * _winding_density(ut, a_t)
            u = (q * np.exp(-1j * e * dt)[..., None, :]) @ qh @ u
        # the return map V(k, t) = exp(-i H_F t), 0 <= t <= T, run backwards
        eps = np.zeros((n1, n2, self.norb))
        z = np.zeros((n1, n2, self.norb, self.norb), 'c16')
        for i1 in range(n1):
            for i2 in range(n2):
                eps[i1, i2], z[i1, i2] = _floquet(u[i1, i2], self.period, epsilon)
        zh = np.conj(np.swapaxes(z, -1, -2))
        h_f = (z * eps[..., None, :]) @ zh
        x, w = np.polynomial.legendre.leggauss(max(24, 2 * n_t))
        for xj, wj in zip(x, w):
            tau = self.period * (1 + xj) / 2
            v = (z * np.exp(-1j * eps * tau)[..., None, :]) @ zh
            integral -= wj * self.period / 2 * _winding_density(v, -1j * h_f)
        return self._plane_orientation((0, 1)) * integral / (8 * np.pi**2)

    def edge_state_count(
        self, epsilon: float, nk: int = 200, window: float | None = None,
    ) -> tuple[float, float]:
        r'''
        Count the chiral edge states of a driven ribbon (a 1D model, e.g.
        made by *step_drive* from *kspace.ribbon* models) crossing the
        quasienergy gap at *epsilon*, on each of its two edges.

        Following the quasienergies :math:`\epsilon_n(k)` over the
        Brillouin zone (:math:`k` along :math:`\mathbf{a}_1`, *nk* points), a
        state crossing *epsilon* upwards (:math:`d\epsilon/dk > 0`) counts
        +1, downwards -1, weighted by its probability on each half of the
        ribbon (the halves below and above its middle line, along
        :math:`\hat{z}\times\mathbf{a}_1`). States are followed from one k to
        the next by their largest overlap. Only states within *window* of
        *epsilon* are followed; bulk states do not cross a gap, so only
        edge states contribute. The counts are integers, :math:`\pm1` per
        chiral mode, up to the tail of the edge states on the other half
        (exponentially small in the width) and the two edges sum to zero.
        With the conventions of *winding_number*,
        ``n_upper = -n_lower = W`` (checked in the tests and examples).

        :param epsilon: Real number. Quasienergy of the gap.
        :param nk: Positive integer (at least 4). Default value 200. k-points,
            fine enough that a state moves by less than *window* between two.
        :param window: Positive real. Default value None (:math:`\omega/8`).

        :returns:
            * **n_lower** -- Real number. Signed count on the lower edge.
            * **n_upper** -- Real number. Signed count on the upper edge.
        '''
        error_handling.dim_exact(self.dim, 1)
        error_handling.real_number(epsilon, 'epsilon')
        error_handling.positive_int(nk, 'nk')
        error_handling.nk_min((nk,), 4)
        omega = 2 * np.pi / self.period
        if window is None:
            window = omega / 8
        error_handling.positive_real(window, 'window')
        a1 = np.asarray(self.lat.prim_vec[0], dtype='f8')[:2]
        normal = np.array([-a1[1], a1[0]]) / np.linalg.norm(a1)
        across = np.array([dic['r0'][:2] for dic in self.lat.unit_cell], dtype='f8') @ normal
        if self.spin:
            across = np.repeat(across, 2)
        middle = (across.min() + across.max()) / 2
        lower = np.where(across < middle, 1., np.where(across > middle, 0., 0.5))
        ks = ((np.arange(nk) + 0.5) / nk)[:, None] * self.rec_vec_k[0][None, :]
        states = []
        for k in ks:
            eps, z = _floquet(self.get_evolution(k), self.period, epsilon + omega / 2)
            states.append((eps - epsilon, z))
        n_lower = n_upper = 0.
        for i in range(nk):
            x1, z1 = states[i]
            x2, z2 = states[(i + 1) % nk]
            overlap = np.abs(z1.conj().T @ z2)
            for a in np.where(np.abs(x1) < window)[0]:
                b = np.argmax(overlap[a])
                if abs(x2[b]) < window and np.sign(x1[a]) != np.sign(x2[b]):
                    direction = np.sign(x2[b] - x1[a])
                    w_low = (lower @ np.abs(z1[:, a])**2 + lower @ np.abs(z2[:, b])**2) / 2
                    n_lower += direction * w_low
                    n_upper += direction * (1 - w_low)
        return float(n_lower), float(n_upper)


class FloquetKSpace(DrivenKSpace):
    r'''
    A periodic model driven by a uniform, time-periodic vector potential
    :math:`\mathbf{A}(t)` (e.g. circularly polarized light,
    :math:`\mathbf{A}(t) = A_0(\cos\omega t, \sin\omega t)`), coupled through
    the Peierls phases of its bonds (see *KSpace.get_ham_peierls*).

    It is a copy of the **KSpace** model whose *get_ham* returns the Floquet
    Hamiltonian :math:`H_F(\mathbf{k})` (see *effective_hamiltonian*), so
    *get_bands*, *k_path*, *berry_curvature*, *chern_number*,
    *wannier_centers*, ... describe the driven system (quasienergies in the
    first Floquet zone: for Chern numbers, pick bands separated by gaps
    that do not wrap around the zone edge, or move the zone edge with
    *epsilon*). It is a *DrivenKSpace*, with
    :math:`H(\mathbf{k}, t)` = ``ks.get_ham_peierls(k, vector_potential(t))``,
    so *winding_number* applies. Real-space methods that need
    static hoppings (*finite_ham*, *get_ham_beta*, *gbz*) are not available.

    :param ks: **KSpace** instance (the static model).
    :param vector_potential: Callable. ``vector_potential(t)`` returns
        :math:`\mathbf{A}(t)` (space_dim components).
    :param period: Positive real. :math:`T`.
    :param n_steps: Positive integer. Default value 100. Time steps per period.
    :param epsilon: Real number. Default value None (quasienergies in
        :math:`[-\omega/2, \omega/2)`, as before this option existed).
        Branch cut of *get_ham*, see *effective_hamiltonian*.
    '''

    def __new__(cls, ks, vector_potential: Callable, period: float, n_steps: int = 100,
                     epsilon: float | None = None):
        from tbkit.kspace import KSpace
        error_handling.kspace(ks, KSpace)
        error_handling.is_callable(vector_potential, 'vector_potential')
        error_handling.positive_real(period, 'period')
        error_handling.positive_int(n_steps, 'n_steps')
        error_handling.branch_cut(epsilon)
        obj = cls._copy(ks, period, epsilon)
        obj.vector_potential = vector_potential
        obj.n_steps = n_steps
        obj.ham_kt = lambda k, t: obj.static.get_ham_peierls(k, obj.vector_potential(t))
        return obj

    def __init__(self, ks, vector_potential, period, n_steps=100, epsilon=None) -> None:
        pass

    def _grid_steps(self, ks: NDArray[np.float64]) -> tuple[NDArray[np.complex128], list]:
        '''
        Private method. The midpoint Hamiltonians at many k-points at once (see *DrivenKSpace*).
        '''
        dt = self.period / self.n_steps
        hams = [_bloch_grid(self.static, ks, np.asarray(self.vector_potential((m + 0.5) * dt), dtype='f8'))
                   for m in range(self.n_steps)]
        return np.stack(hams, axis=1), [dt] * self.n_steps

    def is_hermitian(self) -> bool:
        '''
        The Floquet Hamiltonian of a Hermitian drive is Hermitian.
        '''
        return self.static.is_hermitian()
