r"""
Quantum transport through a finite Tight-Binding device attached to
semi-infinite leads: the Landauer-Buttiker picture, in which the (linear,
zero-temperature) conductance between two leads is the transmission
probability at the Fermi energy,

.. math::

    G = \frac{e^2}{h}\,T(E_F)\, ,\qquad
    T(E) = \mathrm{Tr}\left[\Gamma_{out}\,G^r\,\Gamma_{in}\,G^a\right]\, ,

(the Caroli formula) with :math:`G^r = [E + i\eta - H - \sum_l\Sigma_l]^{-1}`
the device's retarded Green's function, :math:`\Sigma_l` the self-energy of
lead :math:`l` and :math:`\Gamma_l = i(\Sigma_l - \Sigma_l^\dagger)`.

A lead is a semi-infinite repetition of a unit cell with Hamiltonian
:math:`h_0`, each cell coupled to the next one (away from the device) by
:math:`v`; its surface cell is coupled to some device sites by
:math:`\tau`. Its surface Green's function comes from the decimation
algorithm of Lopez Sancho, Lopez Sancho and Rubio (J. Phys. F 15, 851
(1985)), and :math:`\Sigma = \tau\, g_s\,\tau^\dagger`.

Example usage::

    # a perfect chain of 10 sites between two chain leads: T = 1 in the band
    tr = Transport(sys.ham)
    h0, v = np.zeros((1, 1)), np.ones((1, 1))
    tr.add_lead(h0, v, coupling=np.ones((1, 1)), sites=[0])
    tr.add_lead(h0, v, coupling=np.ones((1, 1)), sites=[9])
    T = tr.transmission(np.linspace(-1.9, 1.9, 50))
"""
from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
import scipy.linalg as LA

import tbkit.error_handling as error_handling


def surface_green(
    h0: ArrayLike, v: ArrayLike, energy: float, eta: float = 1e-9, tol: float = 1e-12,
    max_iter: int = 10000,
) -> NDArray[np.complex128]:
    r'''
    Get the surface Green's function of a semi-infinite lead: cells
    :math:`0, 1, 2, \dots`, with :math:`H_{nn} = h_0` and
    :math:`H_{n,n+1} = v`, :math:`H_{n+1,n} = v^\dagger`, by the
    Lopez Sancho-Rubio decimation (each iteration doubles the number of
    cells accounted for, so it converges in a few tens of steps inside the
    bands, as :math:`\eta\to0^+`).

    :param h0: Square complex array. Cell Hamiltonian.
    :param v: Square complex array, same shape. Coupling of a cell to the next one.
    :param energy: Real number. Energy.
    :param eta: Positive real. Default value 1e-9. Broadening.
    :param tol: Positive real. Default value 1e-12. Convergence threshold on
        the norm of the renormalized couplings.
    :param max_iter: Positive integer. Default value 10000.

    :returns:
        * **gs** -- Complex ndarray, same shape as *h0*: :math:`g_s = [E+i\eta - h_0 - \Sigma_{bulk}]^{-1}`.
    '''
    h0 = np.atleast_2d(np.asarray(h0, dtype='c16'))
    v = np.atleast_2d(np.asarray(v, dtype='c16'))
    error_handling.lead(h0, v)
    error_handling.real_number(energy, 'energy')
    error_handling.positive_real(eta, 'eta')
    error_handling.positive_real(tol, 'tol')
    error_handling.positive_int(max_iter, 'max_iter')
    m = len(h0)
    # The first decimation step inverts E + i eta - h0: at an energy equal to
    # an eigenvalue of h0 (E = 0 for a plain chain) that is ~1/eta, and the
    # precision is lost. Regrouping n = 1, 2 or 3 cells into one (the same
    # lead) moves those energies: use the best-conditioned grouping.
    best = None
    for n in (1, 2, 3):
        hn = np.kron(np.eye(n), h0) + np.kron(np.eye(n, k=1), v) + np.kron(np.eye(n, k=-1), v.conj().T)
        vn = np.kron(np.eye(n, k=1 - n), v) if n > 1 else v.copy()
        smin = np.linalg.svd((energy + 1j * eta) * np.eye(n * m) - hn, compute_uv=False)[-1]
        if best is None or smin > best[0] * (1. + 1e-12):
            best = (smin, hn, vn)
    _, hn, vn = best
    z = (energy + 1j * eta) * np.eye(len(hn))
    eps_s, eps = hn.copy(), hn.copy()
    alpha, beta = vn.copy(), vn.conj().T.copy()
    for _ in range(max_iter):
        g = LA.inv(z - eps)
        ag, bg = alpha @ g, beta @ g
        eps_s = eps_s + ag @ beta
        eps = eps + ag @ beta + bg @ alpha
        alpha, beta = ag @ alpha, bg @ beta
        if np.linalg.norm(alpha) + np.linalg.norm(beta) < tol:
            break
    else:
        error_handling.converged(False, 'surface_green')
    return LA.inv(z - eps_s)[:m, :m]


def lead_from_kspace(ks, direction: int = 1) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    r'''
    Get the cell Hamiltonian :math:`h_0` and coupling :math:`v` of a lead
    from a 1D periodic model (e.g. a ribbon, see *kspace.ribbon*), whose
    hoppings reach at most the neighbouring cells.

    :param ks: **KSpace** instance, 1D.
    :param direction: +1 or -1. Default value 1. Direction, along the
        model's primitive vector, in which the lead extends away from the
        device: :math:`v` gathers the hoppings to the cell at
        :math:`\mathbf{R} = \pm\mathbf{a}_1`.

    :returns:
        * **h0** -- Complex ndarray, shape (norb, norb).
        * **v** -- Complex ndarray, shape (norb, norb).
    '''
    error_handling.dim_exact(ks.dim, 1)
    error_handling.lead_direction(direction)
    h0 = np.diag(ks.onsite).astype('c16') + ks._onsite_offdiag
    v = np.zeros((ks.norb, ks.norb), 'c16')
    for i, j, (n,), t in ks._hop_cells():
        error_handling.nearest_cells(n)
        if n == 0:
            h0[i, j] += t
        elif n == direction:
            v[i, j] += t
    return h0, v


class Transport():
    r'''
    A finite device, of Hamiltonian *ham* (e.g. *System.ham*), to which
    semi-infinite leads are attached with *add_lead*; *transmission* then
    gives the Landauer transmission between two of them.

    :param ham: Square matrix (sparse or dense). Device Hamiltonian.
    '''

    def __init__(self, ham) -> None:
        ham = ham.toarray() if hasattr(ham, 'toarray') else np.asarray(ham)
        error_handling.square_matrix(ham, 'ham')
        self.ham = ham.astype('c16')
        self.leads = []  # list of (h0, v, coupling, sites)

    def add_lead(self, h0: ArrayLike, v: ArrayLike, coupling: ArrayLike, sites: list[int]) -> None:
        r'''
        Attach a semi-infinite lead (see *surface_green*).

        :param h0: Square complex array, shape (m, m). Lead cell Hamiltonian.
        :param v: Square complex array, shape (m, m). Coupling of a lead cell
            to the next one, away from the device.
        :param coupling: Complex array, shape (len(sites), m). Hoppings
            :math:`\tau` between the device *sites* and the lead's surface cell.
        :param sites: List of device site (row) indices the lead touches.
        '''
        h0 = np.atleast_2d(np.asarray(h0, dtype='c16'))
        v = np.atleast_2d(np.asarray(v, dtype='c16'))
        error_handling.lead(h0, v)
        coupling = np.atleast_2d(np.asarray(coupling, dtype='c16'))
        error_handling.lead_coupling(coupling, sites, len(h0), len(self.ham))
        self.leads.append((h0, v, coupling, list(sites)))

    def self_energy(self, lead: int, energy: float, eta: float = 1e-9) -> NDArray[np.complex128]:
        r'''
        Get the self-energy :math:`\Sigma = \tau g_s \tau^\dagger` of a lead,
        embedded in the device space.

        :param lead: Integer. Lead index (in the order of *add_lead*).
        :param energy: Real number.
        :param eta: Positive real. Default value 1e-9.

        :returns:
            * **sigma** -- Complex ndarray, shape of *ham*.
        '''
        error_handling.lead_index(lead, len(self.leads))
        h0, v, tau, sites = self.leads[lead]
        sigma = np.zeros_like(self.ham)
        sigma[np.ix_(sites, sites)] = tau @ surface_green(h0, v, energy, eta) @ tau.conj().T
        return sigma

    def get_green(self, energy: float, eta: float = 1e-9) -> NDArray[np.complex128]:
        r'''
        Get the retarded Green's function of the device with all its leads,
        :math:`G^r = [E + i\eta - H - \sum_l \Sigma_l]^{-1}`.

        :param energy: Real number.
        :param eta: Positive real. Default value 1e-9.

        :returns:
            * **G** -- Complex ndarray, shape of *ham*.
        '''
        error_handling.real_number(energy, 'energy')
        error_handling.positive_real(eta, 'eta')
        sigma = sum((self.self_energy(l, energy, eta) for l in range(len(self.leads))),
                            np.zeros_like(self.ham))
        return LA.inv((energy + 1j * eta) * np.eye(len(self.ham)) - self.ham - sigma)

    def transmission(
        self, energies: ArrayLike, lead_in: int = 0, lead_out: int = 1, eta: float = 1e-9,
    ) -> NDArray[np.float64]:
        r'''
        Get the transmission :math:`T(E) = \mathrm{Tr}[\Gamma_{out} G^r
        \Gamma_{in} G^a]` from lead *lead_in* to lead *lead_out*: the
        two-terminal conductance in units of :math:`e^2/h` (per spin, for a
        spinless model).

        :param energies: Real array. Energies.
        :param lead_in: Integer. Default value 0.
        :param lead_out: Integer. Default value 1.
        :param eta: Positive real. Default value 1e-9.

        :returns:
            * **T** -- Real ndarray, same length as *energies*.
        '''
        error_handling.lead_index(lead_in, len(self.leads))
        error_handling.lead_index(lead_out, len(self.leads))
        energies = np.atleast_1d(np.asarray(energies, dtype='f8'))
        out = np.zeros(len(energies))
        for n, e in enumerate(energies):
            sig_in = self.self_energy(lead_in, e, eta)
            sig_out = self.self_energy(lead_out, e, eta)
            gam_in = 1j * (sig_in - sig_in.conj().T)
            gam_out = 1j * (sig_out - sig_out.conj().T)
            g = self.get_green(e, eta)
            out[n] = np.trace(gam_out @ g @ gam_in @ g.conj().T).real
        return out
