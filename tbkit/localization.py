r'''
Localization lengths of disordered quasi-1D strips by the transfer matrix
(MacKinnon and Kramer 1981, 1983).

The amplitudes :math:`A_i` on the slices of a strip obey

.. math::

    \sum_{l=1}^{l_0}\left(V_l^\dagger A_{i-l} + V_l A_{i+l}\right) + h_i A_i = E\,A_i\, ,

with :math:`h_i = h_0 + \mathrm{diag}(\epsilon_i)`, so that
:math:`A_{i+l_0} = -V_{l_0}^{-1}[\dots]` is propagated slice by slice. The
Lyapunov exponents of the product of transfer matrices are the averaged
logarithms of the diagonal of R in a QR factorization repeated along the
strip; at a real energy they come in pairs :math:`\pm\gamma`, and the
smallest positive one sets the localization length
:math:`\lambda_M = 1/\gamma_{\min}` of a strip of cross-section *M*.
'''
from __future__ import annotations

from typing import NamedTuple

import numpy as np
import scipy.linalg as LA
from numpy.typing import ArrayLike, NDArray

import tbkit.error_handling as error_handling


class Lyapunov(NamedTuple):
    r'''
    Result of *lyapunov_exponents*, one row per energy.

    Attributes
    ----------
    exponents : Real ndarray, shape (n_e, l0*m)
        The positive Lyapunov exponents per slice, in decreasing order.
    error : Real ndarray, shape (n_e,)
        Standard error of the smallest one, from the scatter of its means
        over *n_chunks* consecutive pieces of the strip.
    localization_length : Real ndarray, shape (n_e,)
        :math:`\lambda = 1/\gamma_{\min}`, in slices.
    length_error : Real ndarray, shape (n_e,)
        Standard error of *localization_length*.
    '''
    exponents: NDArray[np.float64]
    error: NDArray[np.float64]
    localization_length: NDArray[np.float64]
    length_error: NDArray[np.float64]


def lyapunov_exponents(
    h0: ArrayLike, v, energies: ArrayLike, disorder: float = 0., n_slices: int = 10000,
    n_warmup: int = 200, n_chunks: int = 20, qr_every: int = 1, seed=None,
) -> Lyapunov:
    r'''
    Get the Lyapunov exponents and the localization length of a long strip
    with uniform onsite disorder (the transfer-matrix method of MacKinnon
    and Kramer), without building its Hamiltonian: every slice is drawn,
    used and discarded.

    Parameters
    ----------
    h0 : Hermitian matrix, shape (m, m)
        Hamiltonian of one slice of the clean strip (e.g. from
        *transport.lead_from_kspace*, or the ``ham`` of a *System* of the
        cross-section).
    v : Matrix of shape (m, m), or array of shape (l0, m, m)
        Coupling :math:`V_1 = H_{i,i+1}` to the next slice, or the couplings
        :math:`[V_1, \dots, V_{l_0}]` to the next :math:`l_0` slices. The
        last one must be invertible: group several cells into one slice if
        it is not (e.g. a zigzag ribbon).
    energies : Real number or array
        Several energies are propagated along the same strip (the same
        disorder), which correlates their errors.
    disorder : Positive real or zero, default 0
        Width *W*: onsite energies uniform in :math:`[-W/2, W/2]`,
        independent on every site of every slice.
    n_slices : Positive integer, default 10000
        Slices averaged over (rounded down to a multiple of
        ``n_chunks * qr_every``). The relative error of
        :math:`\gamma_{\min}` decreases as :math:`1/\sqrt{n\gamma_{\min}}`.
    n_warmup : Positive integer or zero, default 200
        Slices propagated first, to align the random initial basis.
    n_chunks : Integer, at least 2, default 20
        Pieces of the strip for the error estimate.
    qr_every : Positive integer, default 1
        Slices between two QR factorizations, which must keep
        :math:`e^{q(\gamma_1-\gamma_{\min})}` well below :math:`10^{16}`.
    seed : Integer or None, default None
        Seed of the disorder and of the initial basis.

    Returns
    -------
    result : :class:`Lyapunov`

    Examples
    --------
    Example usage::

        res = lyapunov_exponents([[0.]], [[1.]], [0.5, 1.], disorder=1., n_slices=10 ** 5)
        res.localization_length   # about 24 (4 - E^2) / W^2 sites (Thouless)
    '''
    h0 = np.atleast_2d(np.asarray(h0, dtype='c16'))
    error_handling.square_matrix(h0, 'h0')
    error_handling.hermitian_dense(h0)
    v = np.asarray(v, dtype='c16')
    vs = list(v) if v.ndim == 3 else [np.atleast_2d(v)]
    for c in vs:
        error_handling.lead(h0, c)
    error_handling.invertible_coupling(vs[-1])
    error_handling.frequencies(energies, 'energies')
    energies = np.atleast_1d(np.asarray(energies, dtype='f8'))
    error_handling.positive_real_zero(disorder, 'disorder')
    error_handling.positive_int(n_slices, 'n_slices')
    error_handling.positive_int_zero(n_warmup, 'n_warmup')
    error_handling.positive_int(qr_every, 'qr_every')
    error_handling.transfer_chunks(n_chunks, n_slices, qr_every)

    m, l0, n_e = len(h0), len(vs), len(energies)
    k = l0 * m
    rng = np.random.default_rng(seed)
    # the rows of the state hold A_{i+l0-1}, ..., A_{i-l0} (blocks of m), so
    # the equation at slice i is one product with [V_{l0-1} .. V_1 h0 V_1^+ .. V_l0^+]
    lu = LA.lu_factor(vs[-1])
    row = np.hstack(vs[-2::-1] + [h0] + [c.conj().T for c in vs])
    prop = LA.lu_solve(lu, row)
    inv = LA.lu_solve(lu, np.eye(m))
    q0, _ = LA.qr(rng.standard_normal((2 * k, k)) + 1j * rng.standard_normal((2 * k, k)), mode='economic')
    q = np.tile(q0, n_e)            # one basis per energy, side by side
    shift = np.repeat(energies, k)
    mid = slice((l0 - 1) * m, l0 * m)
    n_warmup = qr_every * -(-n_warmup // qr_every)
    chunk_len = n_slices // (n_chunks * qr_every) * qr_every
    logs = np.zeros((n_e, k))
    ends = []
    for i in range(n_warmup + chunk_len * n_chunks):
        eps = disorder * (rng.random(m) - 0.5)
        top = prop @ q + inv @ ((eps[:, None] - shift) * q[mid])
        q = np.vstack([-top, q[:-m]])
        if (i + 1) % qr_every:
            continue
        for e in range(n_e):
            q[:, e * k:(e + 1) * k], r = LA.qr(q[:, e * k:(e + 1) * k], mode='economic', check_finite=False)
            if i >= n_warmup:
                logs[e] += np.log(np.abs(np.diag(r)))
        if i >= n_warmup and (i + 1 - n_warmup) % chunk_len == 0:
            ends.append(logs[:, -1].copy())
    n = chunk_len * n_chunks
    means = np.diff(np.vstack([np.zeros(n_e)] + ends), axis=0) / chunk_len
    gammas = logs / n
    error = means.std(axis=0, ddof=1) / np.sqrt(n_chunks)
    length = 1 / gammas[:, -1]
    return Lyapunov(gammas, error, length, error * length ** 2)
