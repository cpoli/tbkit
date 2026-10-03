r"""
Linear and nonlinear optical response of 2D Bloch models (:class:`~tbkit.kspace.KSpace`)
over a uniform Brillouin-zone mesh.

* :func:`optical_conductivity` -- the Kubo-Greenwood conductivity
  :math:`\sigma_{ab}(\omega)`, interband and intraband (Drude), with a
  broadening :math:`\eta`.
* :func:`joint_dos` -- the joint density of states of vertical transitions.
* :func:`berry_curvature_dipole` -- the Berry curvature dipole of the
  nonlinear Hall effect (Sodemann and Fu 2015).
* :func:`shift_current` -- the shift-current conductivity
  :math:`\sigma^{abc}(0;\omega,-\omega)` (Sipe and Shkrebtii 2000).

Every function works in units :math:`\hbar = e = 1`, energies (and
:math:`\hbar\omega`) in the units of the hoppings, lengths in those of
*prim_vec*, and counts each orbital once: a spinless model gives the
response *per spin*.

The velocities are the exact :math:`\partial H/\partial k_a` of
*KSpace._bloch_derivatives* along the Cartesian axes :math:`x, y` (for a 2D
lattice embedded in 3D space, the axes of *KSpace.k_basis*). With
``positions=True`` (the default) the bond vectors include the orbital
positions, :math:`\mathbf{d} = \mathbf{R} + \boldsymbol\tau_j -
\boldsymbol\tau_i`, so that the velocity is the physical :math:`i[H,
\mathbf{r}]` (see *KSpace.hall_conductivity*).

Example usage::

    from tbkit.optics import optical_conductivity
    omega = np.linspace(0.05, 3., 60)
    sigma = optical_conductivity(graphene, omega, e_fermi=0., nk=300)
    # Re sigma ~ pi/4 e^2/h per spin for hbar omega << t
"""
from __future__ import annotations

from typing import Any, Iterator

import numpy as np
from numpy.typing import ArrayLike, NDArray

import tbkit.error_handling as error_handling
import tbkit.occupation as occupation
from tbkit.kspace import KSpace


_AXES = {'x': 0, 'y': 1}
_DEGENERATE = 1e-9


def _mesh(ks: KSpace, nk: int | tuple[int, int], positions: bool) -> NDArray[np.float64]:
    '''
    Private function. Validate the model (a static, Hermitian, 2D KSpace)
    and return the k-points of a uniform Brillouin-zone mesh.
    '''
    error_handling.kspace(ks, KSpace)
    ks._check_static()
    error_handling.dim_exact(ks.dim, 2)
    error_handling.hermitian_model(ks.is_hermitian())
    error_handling.boolean(positions, 'positions')
    return ks.mesh_grid(nk)[1]


def cell_area(ks: KSpace) -> float:
    r'''
    Get the area :math:`A_c = |\mathbf{a}_1\times\mathbf{a}_2|` of the unit
    cell of a 2D model (in the units of *prim_vec*), which turns a
    Brillouin-zone average into an integral:
    :math:`\int\frac{d^2k}{(2\pi)^2}g = \frac{1}{N_kA_c}\sum_{\mathbf{k}}g`.

    :param ks: **KSpace** instance, 2D.

    :returns:
        * **area** -- Positive real.
    '''
    error_handling.kspace(ks, KSpace)
    error_handling.dim_exact(ks.dim, 2)
    a1, a2 = (np.zeros(3) for _ in range(2))
    a1[:ks.space_dim] = ks.lat.prim_vec[0]
    a2[:ks.space_dim] = ks.lat.prim_vec[1]
    return float(np.linalg.norm(np.cross(a1, a2)))


def _bands(
    ks: KSpace, kpts: NDArray[np.float64], positions: bool, second: bool = False,
) -> Iterator[tuple]:
    r'''
    Private function. Over chunks of the k-points *kpts*: the band energies
    (nk, n), the velocity matrices :math:`v^a_{nm} = \langle n|\partial_aH|m\rangle`
    in the eigenbasis (2, nk, n, n) and, with *second*, the matrices
    :math:`w^{ab}_{nm} = \langle n|\partial_a\partial_bH|m\rangle` (2, 2, nk, n, n).
    '''
    directions = [ks.k_basis[:, 0], ks.k_basis[:, 1]]
    chunk = max(1, 200000 // ks.norb ** 2)
    if second:
        # d^2 H / dk_a dk_b is the first derivative of the Bloch sum of the
        # hoppings t * i(d.u_a): the same _bloch_sum, with rescaled amplitudes.
        hops = ks._hop
        d = np.array([h[2] for h in hops], dtype='f8').reshape(len(hops), ks.space_dim)
        if positions:
            tau = ks.orbital_positions()
            d = d + tau[[h[1] for h in hops]] - tau[[h[0] for h in hops]]
        scaled = [[(h[0], h[1], h[2], h[3] * 1j * (dd @ u)) for h, dd in zip(hops, d)]
                       for u in directions]
    for c0 in range(0, len(kpts), chunk):
        kc = kpts[c0:c0 + chunk]
        ham, dham = ks._bloch_derivatives(kc, directions, positions)
        en, vec = np.linalg.eigh(ham)
        vh = vec.conj().transpose(0, 2, 1)
        vel = np.array([vh @ dh @ vec for dh in dham])
        if not second:
            yield en, vel
            continue
        k_cart = kc @ ks.k_basis.T
        w = np.array([ks._bloch_sum(h, k_cart, directions, positions)[1] for h in scaled])
        yield en, vel, vh[None, None] @ w @ vec[None, None]


def _as_output(values: NDArray, like: ArrayLike) -> Any:
    '''
    Private function. A scalar for a scalar input, else shaped like it.
    '''
    if np.ndim(like) == 0:
        return values[0]
    return values.reshape(np.shape(like) + values.shape[1:])


def optical_conductivity(
    ks: KSpace, omega: float | ArrayLike, e_fermi: float = 0., temperature: float = 0.,
    eta: float = 0.05, nk: int | tuple[int, int] = 100, component: str = 'xx',
    positions: bool = True,
) -> complex | NDArray[np.complex128]:
    r'''
    Get the optical conductivity :math:`\sigma_{ab}(\omega)` of a 2D model
    from the Kubo-Greenwood formula over a uniform Brillouin-zone mesh,

    .. math::

        \sigma_{ab}(\omega) = \frac{i e^2}{\hbar}\int\frac{d^2k}{(2\pi)^2}
        \sum_{n,m}\frac{f_n - f_m}{E_m - E_n}\,
        \frac{v^a_{nm}\,v^b_{mn}}{\hbar\omega + E_n - E_m + i\eta}\, ,

    with :math:`v^a_{nm} = \langle n|\partial H/\partial k_a|m\rangle`,
    :math:`f_n` the Fermi-Dirac occupations at *e_fermi* and *temperature*,
    and :math:`\eta` a phenomenological broadening (the inverse lifetime).
    Pairs of degenerate states, :math:`n = m` included, enter through the
    limit :math:`(f_n-f_m)/(E_m-E_n)\to-\partial f/\partial E`: the
    intraband (Drude) term :math:`\frac{i}{\hbar\omega+i\eta}\sum_n
    (-f'_n)v^a_{nn}v^b_{nn}`. It samples the Fermi surface only through the
    thermal window, so it needs a *temperature* larger than the level
    spacing of the mesh; at ``temperature=0`` only the interband part is
    computed.

    It is the Ohm's-law tensor, :math:`j_a = \sigma_{ab}E_b`. In a gap its
    static Hall part is the TKNN conductance:
    :math:`\sigma_{yx}(0) = -\sigma_{xy}(0)` equals
    *KSpace.hall_conductivity* (which returns :math:`\sigma^{\mathrm{Ohm}}_{yx}`,
    see its docstring). The absorptive part obeys the f-sum rule
    :math:`\int_0^\infty\mathrm{Re}\,\sigma_{aa}\,d\omega =
    \frac{\pi e^2}{2\hbar^2}\int\frac{d^2k}{(2\pi)^2}\sum_n f_n\langle n|\partial_a^2H|n\rangle`.
    For graphene at neutrality, :math:`\mathrm{Re}\,\sigma_{xx} \to
    \frac{e^2}{16\hbar}` per spin and valley for :math:`\eta \ll \hbar\omega
    \ll t`: the universal :math:`\sigma_0 = e^2/4\hbar` once spin and
    valleys are counted, :math:`\pi/4` in units of :math:`e^2/h` for a
    spinless model.

    :param ks: **KSpace** instance, 2D, Hermitian.
    :param omega: Real number or array of real numbers. Photon energies
        :math:`\hbar\omega`.
    :param e_fermi: Real number. Default value 0. Fermi energy.
    :param temperature: Positive real or zero. Default value 0 (:math:`k_B = 1`).
    :param eta: Positive real. Default value 0.05. Broadening, larger than
        the spacing of the transition energies on the mesh.
    :param nk: Positive integer, or tuple of 2. Default value 100. k-mesh.
    :param component: 'xx', 'xy', 'yx' or 'yy'. Default value 'xx'.
    :param positions: Boolean. Default value True. Orbital positions in the
        bond vectors (see *KSpace.hall_conductivity*).

    :returns:
        * **sigma** -- Complex number (or ndarray shaped like *omega*), in
          units of :math:`e^2/h`.
    '''
    kpts = _mesh(ks, nk, positions)
    error_handling.frequencies(omega, 'omega')
    error_handling.real_number(e_fermi, 'e_fermi')
    error_handling.positive_real_zero(temperature, 'temperature')
    error_handling.positive_real(eta, 'eta')
    a, b = error_handling.tensor_component(component, 2)
    w_arr = np.atleast_1d(np.asarray(omega, dtype='f8')).ravel()
    total = np.zeros(len(w_arr), 'c16')
    for en, vel in _bands(ks, kpts, positions):
        f = occupation.fermi_dirac(en, e_fermi, temperature)
        minus_df = f * (1. - f) / temperature if temperature > 0 else np.zeros_like(f)
        de = en[:, None, :] - en[:, :, None]  # E_m - E_n
        degenerate = np.abs(de) < _DEGENERATE
        ratio = np.where(degenerate, minus_df[:, :, None],
                                (f[:, :, None] - f[:, None, :]) / np.where(degenerate, 1., de))
        weight = ratio * vel[a] * vel[b].transpose(0, 2, 1)
        keep = weight != 0
        wk, dk = weight[keep], np.where(degenerate, 0., de)[keep]
        step = max(1, 2000000 // len(w_arr))
        for c0 in range(0, len(wk), step):
            total += (wk[None, c0:c0 + step]
                          / (w_arr[:, None] - dk[None, c0:c0 + step] + 1j * eta)).sum(axis=1)
    sigma = 1j * 2 * np.pi * total / (len(kpts) * cell_area(ks))
    return _as_output(sigma, omega)


def joint_dos(
    ks: KSpace, omega: float | ArrayLike, e_fermi: float = 0., temperature: float = 0.,
    eta: float = 0.05, nk: int | tuple[int, int] = 100,
) -> float | NDArray[np.float64]:
    r'''
    Get the joint density of states of the vertical (interband) transitions,
    per unit cell,

    .. math::

        J(\omega) = \frac{1}{N_k}\sum_{\mathbf{k}}\sum_{n\neq m}
        f_n\,(1 - f_m)\,\delta(\hbar\omega - E_m + E_n)\, ,

    with the delta function a Gaussian of standard deviation :math:`\eta`.
    Its integral over :math:`\omega` is the number of (occupied, empty)
    pairs per k-point. For graphene at neutrality,
    :math:`J = A_c\,\hbar\omega/(4\pi\hbar^2 v_F^2)` near the Dirac points
    (both valleys, per spin).

    :param ks: **KSpace** instance, 2D, Hermitian.
    :param omega: Real number or array of real numbers. Photon energies.
    :param e_fermi: Real number. Default value 0.
    :param temperature: Positive real or zero. Default value 0.
    :param eta: Positive real. Default value 0.05. Gaussian width.
    :param nk: Positive integer, or tuple of 2. Default value 100.

    :returns:
        * **jdos** -- Real number (or ndarray shaped like *omega*), in states
          per unit cell and per unit energy.
    '''
    kpts = _mesh(ks, nk, True)
    error_handling.frequencies(omega, 'omega')
    error_handling.real_number(e_fermi, 'e_fermi')
    error_handling.positive_real_zero(temperature, 'temperature')
    error_handling.positive_real(eta, 'eta')
    w_arr = np.atleast_1d(np.asarray(omega, dtype='f8')).ravel()
    total = np.zeros(len(w_arr))
    for en, _ in _bands(ks, kpts, False):
        f = occupation.fermi_dirac(en, e_fermi, temperature)
        weight = f[:, :, None] * (1. - f[:, None, :])
        weight[:, np.arange(ks.norb), np.arange(ks.norb)] = 0.
        keep = weight > 0
        wk = weight[keep]
        dk = (en[:, None, :] - en[:, :, None])[keep]
        total += _gaussian_sum(w_arr, dk, wk, eta)
    return _as_output(total / len(kpts), omega)


def _gaussian_sum(
    x: NDArray[np.float64], centers: NDArray[np.float64], weights: NDArray[np.float64], eta: float,
) -> NDArray[np.float64]:
    r'''
    Private function. :math:`\sum_i w_i\,g(x - c_i)` with the normalized
    Gaussian :math:`g` of standard deviation *eta*, in chunks of the centers.
    '''
    out = np.zeros(len(x))
    step = max(1, 2000000 // len(x))
    for c0 in range(0, len(centers), step):
        u = (x[:, None] - centers[None, c0:c0 + step]) / eta
        out += (weights[None, c0:c0 + step] * np.exp(-0.5 * u ** 2)).sum(axis=1)
    return out / (np.sqrt(2 * np.pi) * eta)


def berry_curvature_dipole(
    ks: KSpace, e_fermi: float | ArrayLike, temperature: float,
    nk: int | tuple[int, int] = 100, positions: bool = True,
) -> NDArray[np.float64]:
    r'''
    Get the Berry curvature dipole of a 2D model (Sodemann and Fu, Phys.
    Rev. Lett. 115, 216806 (2015)),

    .. math::

        D_a = \int\frac{d^2k}{(2\pi)^2}\sum_n f_n\,\partial_a\Omega_n
            = \int\frac{d^2k}{(2\pi)^2}\sum_n
              \left(-\frac{\partial f}{\partial E}\right)_{E_n}
              v^a_{nn}\,\Omega_n\, ,

    the first moment of the Berry curvature :math:`\Omega_n` (the
    convention of *KSpace.berry_curvature* and *hall_conductivity*,
    :math:`\Omega_n = -2\,\mathrm{Im}\sum_{m\neq n}v^x_{nm}v^y_{mn}/(E_n-E_m)^2`)
    over the occupied states. It is a Fermi-surface property: the second
    form, used here, samples it through the thermal window
    :math:`-\partial f/\partial E`, so *temperature* must be positive and
    larger than the level spacing of the mesh near :math:`E_F`. A finite
    dipole needs broken inversion symmetry and a single mirror line at
    most. In a DC field it drives the nonlinear Hall current
    :math:`j_a = \chi_{abc}E_bE_c`,
    :math:`\chi_{abc} = -\varepsilon_{adc}\frac{e^3\tau}{2\hbar^2(1+i\omega\tau)}D_{bd}`,
    with :math:`D_{bd} = D_b` along :math:`d = z` in 2D.

    For the tilted massive Dirac cone
    :math:`H = t k_x + v(k_x\sigma_x + k_y\sigma_y) + m\sigma_z` doped into
    the conduction band (:math:`\mu > |m|`), the first order in the tilt is
    :math:`D_x = -\frac{3\,t\,m\,(\mu^2 - m^2)}{8\pi\mu^4}`,
    :math:`D_y = 0`, from :math:`\Omega_c = -mv^2/2\varepsilon^3`
    (Sodemann and Fu's model, in these conventions).

    :param ks: **KSpace** instance, 2D, Hermitian.
    :param e_fermi: Real number, or array of real numbers. Fermi energies.
    :param temperature: Positive real. Temperature (:math:`k_B = 1`).
    :param nk: Positive integer, or tuple of 2. Default value 100.
    :param positions: Boolean. Default value True. See *KSpace.hall_conductivity*.

    :returns:
        * **dipole** -- Real ndarray of shape (2,): :math:`(D_x, D_y)` (or
          ``e_fermi.shape + (2,)``), in units of length (of *prim_vec*).
    '''
    kpts = _mesh(ks, nk, positions)
    error_handling.fermi_energies(e_fermi)
    error_handling.positive_real(temperature, 'temperature')
    mus = np.atleast_1d(np.asarray(e_fermi, dtype='f8')).ravel()
    total = np.zeros((len(mus), 2))
    for en, vel in _bands(ks, kpts, positions):
        de = en[:, :, None] - en[:, None, :]
        degenerate = np.abs(de) < _DEGENERATE
        inv2 = np.where(degenerate, 0., 1. / np.where(degenerate, 1., de) ** 2)
        omega_n = -2 * np.sum(np.imag(vel[0] * vel[1].transpose(0, 2, 1)) * inv2, axis=2)
        vdiag = np.real(np.diagonal(vel, axis1=2, axis2=3))  # (2, nk, n)
        for e, mu in enumerate(mus):
            f = occupation.fermi_dirac(en, float(mu), temperature)
            window = f * (1. - f) / temperature
            total[e] += np.einsum('kn,akn->a', window * omega_n, vdiag)
    return _as_output(total / (len(kpts) * cell_area(ks)), e_fermi)


def generalized_derivative(
    en: NDArray[np.float64], vel: NDArray[np.complex128], w: NDArray[np.complex128],
) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    r'''
    Get the interband position matrix elements and their generalized
    derivatives from the band energies and the first and second derivatives
    of :math:`H` in the eigenbasis (a sum over states, Sipe and Shkrebtii
    2000; Cook et al. 2017; :math:`\hbar = 1`, :math:`\omega_{nm} = E_n - E_m`,
    :math:`\Delta^a_{nm} = v^a_{nn} - v^a_{mm}`):

    .. math::

        r^b_{nm} = \frac{v^b_{nm}}{i\omega_{nm}}\, ,\qquad
        r^b_{nm;a} = \frac{i}{\omega_{nm}}\left[
        \frac{v^b_{nm}\Delta^a_{nm} + v^a_{nm}\Delta^b_{nm}}{\omega_{nm}}
        - w^{ba}_{nm} + \sum_{p\neq n,m}\left(
        \frac{v^b_{np}v^a_{pm}}{\omega_{pm}} - \frac{v^a_{np}v^b_{pm}}{\omega_{np}}
        \right)\right]\, ,

    :math:`r^b_{nm;a} = \partial_ar^b_{nm} - i(A^a_{nn} - A^a_{mm})r^b_{nm}`,
    gauge covariant. Degenerate pairs (:math:`|\omega_{nm}| < 10^{-9}`) are
    set to zero, and dropped from the sum over :math:`p`.

    :param en: Real ndarray, shape (nk, n). Band energies.
    :param vel: Complex ndarray, shape (2, nk, n, n). :math:`v^a_{nm}`.
    :param w: Complex ndarray, shape (2, 2, nk, n, n). :math:`w^{ab}_{nm} =
        \langle n|\partial_a\partial_bH|m\rangle`.

    :returns:
        * **r** -- Complex ndarray, shape (2, nk, n, n): r[b] is :math:`r^b_{nm}`.
        * **r_der** -- Complex ndarray, shape (2, 2, nk, n, n): r_der[b, a]
          is :math:`r^b_{nm;a}`.
    '''
    en, vel, w = np.asarray(en), np.asarray(vel), np.asarray(w)
    error_handling.band_derivatives(en, vel, w)
    om = en[:, :, None] - en[:, None, :]
    degenerate = np.abs(om) < _DEGENERATE
    inv = np.where(degenerate, 0., 1. / np.where(degenerate, 1., om))
    diag = np.diagonal(vel, axis1=2, axis2=3)  # (2, nk, n)
    delta = diag[:, :, :, None] - diag[:, :, None, :]
    r = -1j * vel * inv[None]
    r_der = np.zeros((2, 2) + om.shape, 'c16')
    for b in range(2):
        for a in range(2):
            # the full sums over p include p = n, m: those terms add
            # Delta^b v^a / omega, removed again below
            full = vel[b] @ (vel[a] * inv) - (vel[a] * inv) @ vel[b]
            bracket = ((vel[b] * delta[a] + vel[a] * delta[b]) * inv - w[b, a] + full
                           - vel[a] * delta[b] * inv)
            r_der[b, a] = 1j * inv * bracket
    return r, r_der


def shift_current(
    ks: KSpace, omega: float | ArrayLike, e_fermi: float = 0., temperature: float = 0.,
    eta: float = 0.05, nk: int | tuple[int, int] = 100, component: str = 'xxx',
    positions: bool = True,
) -> float | NDArray[np.float64]:
    r'''
    Get the shift-current conductivity of a 2D model under linearly
    polarized light (Sipe and Shkrebtii, Phys. Rev. B 61, 5337 (2000)),
    :math:`j^a = 2\sigma^{abc}(0;\omega,-\omega)E^b(\omega)E^c(-\omega)`,

    .. math::

        \sigma^{abc}(\omega) = -\frac{\pi e^3}{2\hbar^2}\int\frac{d^2k}{(2\pi)^2}
        \sum_{n,m}f_{nm}\,\mathrm{Im}\left[r^b_{mn}r^c_{nm;a}
        + r^c_{mn}r^b_{nm;a}\right]\delta(\omega_{mn} - \omega)\, ,

    :math:`f_{nm} = f_n - f_m`, with the position matrix elements
    :math:`r` and their generalized derivatives from
    :func:`generalized_derivative` (a sum over states, with the exact second
    derivatives of :math:`H`). The delta function is a Gaussian of standard
    deviation :math:`\eta`. For :math:`b = c`,
    :math:`\mathrm{Im}[r^b_{mn}r^b_{nm;a}] = |r^b_{nm}|^2R^{a,b}_{nm}` with
    :math:`R^{a,b}_{nm}` the shift vector, the real-space displacement of
    the electron in the transition. It vanishes with inversion symmetry.
    Unlike the Hall conductance of full bands, it depends on where the
    orbitals sit in the cell (Ibanez-Azpiroz, Tsirkin and Souza, Phys. Rev.
    B 97, 245143 (2018)): ``positions=False``, which puts them all at the
    cell origin, describes a different crystal.

    :param ks: **KSpace** instance, 2D, Hermitian, without overlap.
    :param omega: Real number or array of real numbers. Photon energies :math:`\hbar\omega`.
    :param e_fermi: Real number. Default value 0.
    :param temperature: Positive real or zero. Default value 0.
    :param eta: Positive real. Default value 0.05. Gaussian width.
    :param nk: Positive integer, or tuple of 2. Default value 100.
    :param component: Three letters among 'x', 'y' (current direction, then
        the two field directions). Default value 'xxx'.
    :param positions: Boolean. Default value True. See *KSpace.hall_conductivity*.

    :returns:
        * **sigma** -- Real number (or ndarray shaped like *omega*), in units
          of :math:`e^3/\hbar\times` length / energy (:math:`\hbar = e = 1`),
          counting each orbital once.
    '''
    kpts = _mesh(ks, nk, positions)
    error_handling.no_overlap(bool(ks._overlap_hop))
    error_handling.frequencies(omega, 'omega')
    error_handling.real_number(e_fermi, 'e_fermi')
    error_handling.positive_real_zero(temperature, 'temperature')
    error_handling.positive_real(eta, 'eta')
    a, b, c = error_handling.tensor_component(component, 3)
    w_arr = np.atleast_1d(np.asarray(omega, dtype='f8')).ravel()
    total = np.zeros(len(w_arr))
    for en, vel, w in _bands(ks, kpts, positions, second=True):
        r, r_der = generalized_derivative(en, vel, w)
        f = occupation.fermi_dirac(en, e_fermi, temperature)
        fnm = f[:, :, None] - f[:, None, :]
        integrand = np.imag(r[b].transpose(0, 2, 1) * r_der[c, a]
                                  + r[c].transpose(0, 2, 1) * r_der[b, a]) * fnm
        keep = integrand != 0
        wmn = (en[:, None, :] - en[:, :, None])[keep]
        total += _gaussian_sum(w_arr, wmn, integrand[keep], eta)
    sigma = -np.pi / 2 * total / (len(kpts) * cell_area(ks))
    return _as_output(sigma, omega)
