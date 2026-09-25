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
