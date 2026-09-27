r"""
The Hubbard model in the (collinear, unrestricted Hartree-Fock) mean-field
approximation:

.. math::

    H = \sum_{ij\sigma} t_{ij}c^\dagger_{i\sigma}c_{j\sigma}
        + U\sum_i n_{i\uparrow}n_{i\downarrow}
    \;\to\;
    \sum_\sigma\Big[\sum_{ij} t_{ij}c^\dagger_{i\sigma}c_{j\sigma}
        + U\sum_i\langle n_{i\bar\sigma}\rangle n_{i\sigma}\Big]
        - U\sum_i\langle n_{i\uparrow}\rangle\langle n_{i\downarrow}\rangle\, .

Each spin moves in the density of the other; the densities are iterated
(with linear mixing) until self-consistent. With :math:`U > 0` on a
bipartite lattice at half filling it reproduces Lieb's theorem (the total
moment :math:`S = ||A| - |B||/2`) and the edge magnetism of zigzag graphene.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
import scipy.linalg as LA

import tbkit.error_handling as error_handling
import tbkit.occupation as occupation


class MeanFieldResult():
    '''
    Self-consistent solution of *hubbard_mean_field*.

    :ivar n_up: Real ndarray. Spin-up density on every site.
    :ivar n_dn: Real ndarray. Spin-down density on every site.
    :ivar magnetization: Real ndarray. :math:`m_i = (n_{i\\uparrow} - n_{i\\downarrow})/2`.
    :ivar en_up: Real ndarray. Spin-up mean-field energies.
    :ivar en_dn: Real ndarray. Spin-down mean-field energies.
    :ivar e_fermi: Real number. Fermi level.
    :ivar energy: Real number. Mean-field ground-state (free, at T > 0) energy
        :math:`\\sum f E - U\\sum_i n_{i\\uparrow}n_{i\\downarrow}`.
    :ivar iterations: Integer. Number of iterations.
    '''

    def __init__(self, n_up, n_dn, en_up, en_dn, e_fermi, energy, iterations) -> None:
        self.n_up, self.n_dn = n_up, n_dn
        self.magnetization = (n_up - n_dn) / 2
        self.en_up, self.en_dn = en_up, en_dn
        self.e_fermi = e_fermi
        self.energy = energy
        self.iterations = iterations

    @property
    def total_spin(self) -> float:
        r'''
        Total :math:`S_z = \sum_i m_i`.
        '''
        return float(np.sum(self.magnetization))


def _occupy(en_up, en_dn, n_electrons, temperature):
    '''
    Private function. Occupations of both spins sharing one Fermi level: at
    T = 0 the lowest levels, the electrons left for a degenerate level at
    the Fermi energy shared equally among its states (a fixed, arbitrary
    choice among them would make the iteration oscillate), else Fermi-Dirac.
    '''
    energies = np.concatenate([en_up, en_dn])
    if temperature == 0:
        n = int(round(n_electrons))
        order = np.sort(energies)
        f = np.zeros(len(energies))
        if n:
            e_f = order[n - 1]
            tol = 1e-8 * max(1., np.max(np.abs(energies)))
            below = energies < e_f - tol
            shell = np.abs(energies - e_f) <= tol
            f[below] = 1.
            f[shell] = (n - below.sum()) / shell.sum()
        mu = occupation.fermi_level(energies, n_electrons)
    else:
        mu = occupation.fermi_level(energies, n_electrons, temperature)
        f = occupation.fermi_dirac(energies, mu, temperature)
    return f[:len(en_up)], f[len(en_up):], mu


def hubbard_mean_field(
    ham, U: float | ArrayLike, n_electrons: float, temperature: float = 0.,
    n_up: ArrayLike | None = None, n_dn: ArrayLike | None = None, mixing: float = 0.5,
    tol: float = 1e-8, max_iter: int = 20000, seed=None,
) -> MeanFieldResult:
    r'''
    Solve the Hubbard model in mean field, self-consistently.

    The iteration converges to *a* self-consistent solution, not
    necessarily the one of lowest energy: compare the *energy* of the
    solutions reached from several starts (random seeds, or a staggered
    magnetization on the two sublattices of a bipartite lattice).

    :param ham: Square Hermitian matrix (sparse or dense), the spin-independent
        single-particle Hamiltonian (e.g. *System.ham*).
    :param U: Real number, or array of one per site. Onsite repulsion.
    :param n_electrons: Positive real, at most 2N. Number of electrons (both
        spins; N for half filling). An integer at T = 0.
    :param temperature: Positive real or zero. Default value 0.
    :param n_up: Real array. Default value None. Initial spin-up density.
    :param n_dn: Real array. Default value None. Initial spin-down density.
        By default, a small random magnetization on top of the uniform
        density (to break the spin symmetry).
    :param mixing: Real in (0, 1]. Default value 0.5. Fraction of the new
        density mixed in at each iteration. (Plain linear mixing converges
        only to stable solutions, but slowly near a magnetic transition:
        raise *max_iter* there.)
    :param tol: Positive real. Default value 1e-8. Convergence threshold on
        the largest density change.
    :param max_iter: Positive integer. Default value 20000.
    :param seed: Default value None. Seed of the random initial magnetization.

    :returns:
        * **result** -- :class:`MeanFieldResult`.
    '''
    ham = ham.toarray() if hasattr(ham, 'toarray') else np.asarray(ham)
    error_handling.square_matrix(ham, 'ham')
    n = len(ham)
    error_handling.hermitian_dense(ham)
    U = np.broadcast_to(np.asarray(U, dtype='f8'), (n,)).copy() if np.ndim(U) == 0 \
        else np.asarray(U, dtype='f8')
    error_handling.ndarray(U, 'U', n)
    error_handling.electrons(n_electrons, 2 * n)
    error_handling.positive_real_zero(temperature, 'temperature')
    if temperature == 0:
        error_handling.integer_electrons(n_electrons)
    error_handling.mixing(mixing)
    error_handling.positive_real(tol, 'tol')
    error_handling.positive_int(max_iter, 'max_iter')
    if n_up is None or n_dn is None:
        rng = np.random.default_rng(seed)
        kick = 0.1 * rng.uniform(-1., 1., n)
        n_up = np.full(n, n_electrons / (2 * n)) + kick
        n_dn = np.full(n, n_electrons / (2 * n)) - kick
    n_up, n_dn = np.asarray(n_up, dtype='f8').copy(), np.asarray(n_dn, dtype='f8').copy()
    error_handling.ndarray(n_up, 'n_up', n)
    error_handling.ndarray(n_dn, 'n_dn', n)
    for it in range(1, max_iter + 1):
        en_up, v_up = LA.eigh(ham + np.diag(U * n_dn))
        en_dn, v_dn = LA.eigh(ham + np.diag(U * n_up))
        f_up, f_dn, mu = _occupy(en_up, en_dn, n_electrons, temperature)
        new_up = (np.abs(v_up) ** 2) @ f_up
        new_dn = (np.abs(v_dn) ** 2) @ f_dn
        change = max(np.max(np.abs(new_up - n_up)), np.max(np.abs(new_dn - n_dn)))
        n_up = (1 - mixing) * n_up + mixing * new_up
        n_dn = (1 - mixing) * n_dn + mixing * new_dn
        if change < tol:
            break
    else:
        error_handling.converged(False, 'hubbard_mean_field')
    energy = float(np.sum(f_up * en_up) + np.sum(f_dn * en_dn) - np.sum(U * new_up * new_dn))
    return MeanFieldResult(new_up, new_dn, en_up, en_dn, mu, energy, it)


#: Pauli matrices sigma_x, sigma_y, sigma_z.
_SIGMA = np.array([[[0., 1.], [1., 0.]], [[0., -1j], [1j, 0.]], [[1., 0.], [0., -1.]]], dtype='c16')


class NonCollinearResult():
    r'''
    Self-consistent solution of *hubbard_mean_field_noncollinear*.

    :ivar rho: Complex ndarray, shape (N, 2, 2). Onsite spin density
        matrices :math:`\rho_{i,\alpha\beta} = \langle c^\dagger_{i\beta}c_{i\alpha}\rangle`.
    :ivar density: Real ndarray. :math:`n_i = \mathrm{Tr}\,\rho_i`.
    :ivar magnetization: Real ndarray, shape (N, 3).
        :math:`\mathbf{m}_i = \mathrm{Tr}(\rho_i\boldsymbol\sigma)/2`.
    :ivar energies: Real ndarray. Mean-field energies (2N).
    :ivar states: Complex ndarray, shape (2N, 2N). Eigenvectors (columns),
        rows site-major (spin up, down).
    :ivar e_fermi: Real number. Fermi level.
    :ivar energy: Real number. Mean-field ground-state energy
        :math:`\sum f E - U\sum_i(\rho_{i\uparrow\uparrow}\rho_{i\downarrow\downarrow}
        - |\rho_{i\uparrow\downarrow}|^2)`.
    :ivar iterations: Integer. Number of iterations.
    '''

    def __init__(self, rho, energies, states, e_fermi, energy, iterations) -> None:
        self.rho = rho
        self.density = np.real(np.trace(rho, axis1=1, axis2=2))
        self.magnetization = np.real(np.einsum('iab,sba->is', rho, _SIGMA)) / 2
        self.energies, self.states = energies, states
        self.e_fermi = e_fermi
        self.energy = energy
        self.iterations = iterations

    @property
    def total_spin(self) -> NDArray[np.float64]:
        r'''
        Total spin :math:`\sum_i\mathbf{m}_i`, shape (3,).
        '''
        return self.magnetization.sum(axis=0)


def hubbard_mean_field_noncollinear(
    ham, U: float | ArrayLike, n_electrons: float, temperature: float = 0.,
    magnetization: ArrayLike | None = None, spinful: bool = False, mixing: float = 0.5,
    tol: float = 1e-8, max_iter: int = 20000, seed=None,
) -> NonCollinearResult:
    r'''
    Solve the Hubbard model in the non-collinear (spin-rotation invariant)
    Hartree-Fock approximation: the onsite spin density matrix
    :math:`\rho_i` is kept whole, so the moments may point in any
    direction (spirals, the 120-degree order of frustrated lattices,
    moments canted by spin-orbit coupling). The Hartree and Fock terms of
    :math:`Un_{i\uparrow}n_{i\downarrow}` add, on every site,

    .. math::

        V_i = U\,(n_i\mathbb{1} - \rho_i) = U\begin{pmatrix}\rho_{\downarrow\downarrow} & -\rho_{\uparrow\downarrow}\\
        -\rho_{\downarrow\uparrow} & \rho_{\uparrow\uparrow}\end{pmatrix}
        = U\Big(\frac{n_i}{2}\mathbb{1} - \mathbf{m}_i\cdot\boldsymbol\sigma\Big)\, ,

    iterated with linear mixing until :math:`\rho` is self-consistent. With
    :math:`\rho_{\uparrow\downarrow} = 0` it is *hubbard_mean_field*
    (collinear moments along :math:`z` stay collinear, and give the same
    solution). As there, several starts should be compared by *energy*.

    :param ham: Square Hermitian matrix: the single-particle Hamiltonian,
        (N, N) spin independent, or, with ``spinful=True``, (2N, 2N) with
        rows site-major (spin up, down) -- e.g. with spin-orbit coupling.
    :param U: Real number, or array of one per site. Onsite repulsion.
    :param n_electrons: Positive real, at most 2N. An integer at T = 0.
    :param temperature: Positive real or zero. Default value 0.
    :param magnetization: Real array, shape (N, 3). Default value None
        (random directions of length 0.1). Initial moments, on top of a
        uniform density.
    :param spinful: Boolean. Default value False. See *ham*.
    :param mixing: Real in (0, 1]. Default value 0.5.
    :param tol: Positive real. Default value 1e-8. Convergence threshold on
        the largest change of :math:`\rho`.
    :param max_iter: Positive integer. Default value 20000.
    :param seed: Default value None. Seed of the random initial moments.

    :returns:
        * **result** -- :class:`NonCollinearResult`.
    '''
    ham = ham.toarray() if hasattr(ham, 'toarray') else np.asarray(ham)
    error_handling.square_matrix(ham, 'ham')
    error_handling.hermitian_dense(ham)
    error_handling.boolean(spinful, 'spinful')
    if spinful:
        error_handling.even_dimension(len(ham))
        h0 = ham.astype('c16')
    else:
        h0 = np.kron(ham, np.eye(2)).astype('c16')
    n = len(h0) // 2
    U = np.full(n, float(U)) if np.ndim(U) == 0 else np.asarray(U, dtype='f8')
    error_handling.ndarray(U, 'U', n)
    error_handling.electrons(n_electrons, 2 * n)
    error_handling.positive_real_zero(temperature, 'temperature')
    if temperature == 0:
        error_handling.integer_electrons(n_electrons)
    error_handling.mixing(mixing)
    error_handling.positive_real(tol, 'tol')
    error_handling.positive_int(max_iter, 'max_iter')
    if magnetization is None:
        vec = np.random.default_rng(seed).normal(size=(n, 3))
        magnetization = 0.1 * vec / np.linalg.norm(vec, axis=1)[:, None]
    magnetization = np.asarray(magnetization, dtype='f8')
    error_handling.magnetization(magnetization, n)
    # rho = n/2 + m.sigma (so that Tr(rho sigma)/2 = m)
    rho = (n_electrons / (2 * n) * np.eye(2)[None]
              + np.einsum('is,sab->iab', magnetization, _SIGMA)).astype('c16')
    blocks = np.arange(n)
    for it in range(1, max_iter + 1):
        pot = U[:, None, None] * (np.trace(rho, axis1=1, axis2=2)[:, None, None] * np.eye(2)[None] - rho)
        h = h0.copy()
        for a in range(2):
            for b in range(2):
                h[2 * blocks + a, 2 * blocks + b] += pot[:, a, b]
        en, vec = np.linalg.eigh(h)
        f, _, mu = _occupy(en, en[:0], n_electrons, temperature)
        full = (vec * f[None, :]) @ vec.conj().T
        new = full.reshape(n, 2, n, 2)[blocks, :, blocks, :]
        change = np.max(np.abs(new - rho))
        rho = (1 - mixing) * rho + mixing * new
        if change < tol:
            break
    else:
        error_handling.converged(False, 'hubbard_mean_field_noncollinear')
    double = U * np.real(new[:, 0, 0] * new[:, 1, 1] - np.abs(new[:, 0, 1]) ** 2)
    energy = float(np.sum(f * en) - np.sum(double))
    return NonCollinearResult(new, en, vec, mu, energy, it)
