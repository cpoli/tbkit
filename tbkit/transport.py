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

*Transport.attach_lead* builds the coupling and finds the device sites
itself, from a lead given as a 1D *KSpace* model placed in the device's
coordinates (e.g. ``tr.attach_lead(sys, strip, 1)``).

*Transport.smatrix* gives the scattering matrix itself. It uses the exact
propagating and evanescent modes of the leads (*lead_modes*, with no
broadening) and one sparse factorization of the device per energy (the
formulation of Groth et al. 2014, as in Kwant). *Transport.wave_function*
and *Transport.ldos* give the scattering states and the local density of
states they carry. *Transport.transmission* uses the scattering matrix by
default.

With more leads, *Transport.conductance_matrix* and
*four_terminal_resistance* solve the Landauer-Buttiker equations (Hall
bars); *bond_currents* and *local_currents* map the current;
*transmission_eigenvalues*, *shot_noise* and *fano_factor* give the shot
noise. *RecursiveTransport* computes the two-terminal transmission of long
quasi-1D devices slice by slice (recursive Green's function), without
inverting the whole device.
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np
from numpy.typing import ArrayLike, NDArray
import scipy.linalg as LA
import scipy.sparse as sp
import scipy.sparse.linalg as spla

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
    gs, _ = _green_surface_bulk(np.array([energy + 1j * eta]), h0, v, v.conj().T, tol, max_iter,
                                             'surface_green')
    return gs[0]


def _green_surface_bulk(
    z: NDArray[np.complex128], h0: NDArray[np.complex128], alpha: NDArray[np.complex128],
    beta: NDArray[np.complex128], tol: float, max_iter: int, name: str,
) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    r'''
    Private function. The surface and bulk Green's functions of one cell
    of the semi-infinite chain of *_decimation* (*h0*, *alpha*, *beta* of
    shape (m, m)) at the complex energies *z*, shapes (nz, m, m).

    The first decimation step inverts :math:`z - h_0`: at an energy equal to
    an eigenvalue of :math:`h_0` (E = 0 for a plain chain) that is
    :math:`\sim1/\eta`, and the precision is lost. Regrouping n = 1, 2 or 3
    cells into one (the same chain) moves those energies: at each energy,
    the best-conditioned grouping is used.
    '''
    m = len(h0)
    # Scaling cell n by r**n is a similarity transform that leaves the
    # Green's function of every cell unchanged: balancing the couplings of a
    # non-reciprocal chain keeps the renormalized ones from overflowing
    # (r = 1 when beta = alpha^dagger).
    n_a, n_b = np.linalg.norm(alpha), np.linalg.norm(beta)
    if n_a > 0 and n_b > 0:
        r = np.sqrt(n_b / n_a)
        alpha, beta = alpha * r, beta / r
    groups, best, choice = [], None, np.zeros(len(z), int)
    for c, n in enumerate((1, 2, 3)):
        hn = (np.kron(np.eye(n), h0) + np.kron(np.eye(n, k=1), alpha)
               + np.kron(np.eye(n, k=-1), beta))
        an = np.kron(np.eye(n, k=1 - n), alpha)
        bn = np.kron(np.eye(n, k=n - 1), beta)
        groups.append((hn, an, bn))
        smin = np.linalg.svd(z[:, None, None] * np.eye(n * m)[None] - hn, compute_uv=False)[:, -1]
        if best is None:
            best = smin
        else:
            better = smin > best * (1. + 1e-12)
            choice[better], best = c, np.where(better, smin, best)
    gs, gb = np.empty((len(z), m, m), 'c16'), np.empty((len(z), m, m), 'c16')
    for c, (hn, an, bn) in enumerate(groups):
        sel = choice == c
        if np.any(sel):
            s_, b_ = _decimation(z[sel], hn, an, bn, tol, max_iter, name)
            gs[sel], gb[sel] = s_[:, :m, :m], b_[:, :m, :m]
    return gs, gb


def _decimation(
    z: NDArray[np.complex128], h0: NDArray[np.complex128], alpha: NDArray[np.complex128],
    beta: NDArray[np.complex128], tol: float, max_iter: int, name: str,
) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    r'''
    Private function. The Lopez Sancho-Rubio decimation of a semi-infinite
    chain of cells :math:`0, 1, 2, \dots` (:math:`H_{nn} = h_0`,
    :math:`H_{n,n+1} = \alpha`, :math:`H_{n+1,n} = \beta`, with
    :math:`\beta = \alpha^\dagger` unless the chain is non-reciprocal) at
    many complex energies *z* (shape (nz,)) at once; *h0*, *alpha* and *beta*
    have shape (m, m) or (nz, m, m). Used by *surface_green* and
    *KSpace.surface_spectral_function*, which validate the arguments. It
    stops when the renormalized couplings are below *tol* at every energy
    (a RuntimeError naming *name* after *max_iter* iterations).

    :returns:
        * **gs** -- Complex ndarray, shape (nz, m, m). Surface Green's function (cell 0).
        * **gb** -- Complex ndarray, shape (nz, m, m). Bulk Green's function
          (one cell of the infinite chain).
    '''
    m = h0.shape[-1]
    zi = np.asarray(z, dtype='c16')[:, None, None] * np.eye(m)[None]
    eps_s = np.broadcast_to(h0, zi.shape).astype('c16')
    eps = eps_s.copy()
    alpha = np.broadcast_to(alpha, zi.shape).astype('c16')
    beta = np.broadcast_to(beta, zi.shape).astype('c16')
    for _ in range(max_iter):
        g = np.linalg.inv(zi - eps)
        ag, bg = alpha @ g, beta @ g
        eps_s = eps_s + ag @ beta
        eps = eps + ag @ beta + bg @ alpha
        alpha, beta = ag @ alpha, bg @ beta
        if np.all(np.linalg.norm(alpha, axis=(1, 2)) + np.linalg.norm(beta, axis=(1, 2)) < tol):
            break
    else:
        error_handling.converged(False, name)
    return np.linalg.inv(zi - eps_s), np.linalg.inv(zi - eps)


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


class LeadModes(NamedTuple):
    r'''
    The propagating modes of a lead at one energy (see *lead_modes*):
    the incoming ones first (moving towards the device, velocity < 0), then
    the outgoing ones, each group by block of the conservation law, then in
    increasing momentum.

    * **wave_functions** -- Complex ndarray, shape (m, n_modes). Mode :math:`\phi`
      on the lead's surface cell: cell :math:`n` carries :math:`\lambda^n\phi`,
      with :math:`\lambda = e^{ik}`. Normalized to unit current.
    * **momenta** -- Real ndarray, shape (n_modes,). :math:`k` in :math:`(-\pi, \pi]`.
    * **velocities** -- Real ndarray, shape (n_modes,). :math:`dE/dk`,
      positive away from the device.
    * **blocks** -- Integer ndarray, shape (n_modes,). The block of the
      conservation law each mode belongs to: the index of its eigenvalue,
      in increasing order (all 0 without a conservation law).
    '''
    wave_functions: NDArray[np.complex128]
    momenta: NDArray[np.float64]
    velocities: NDArray[np.float64]
    blocks: NDArray[np.int64]


def lead_modes(
    h0: ArrayLike, v: ArrayLike, energy: float, conservation_law: ArrayLike | None = None,
) -> LeadModes:
    r'''
    Get the propagating modes of a semi-infinite lead (cells
    :math:`0, 1, 2, \dots` away from the device, :math:`H_{nn} = h_0`,
    :math:`H_{n,n+1} = v`) at one energy. A mode :math:`\psi_n = \lambda^n\phi`
    solves :math:`(E - h_0 - \lambda v - \lambda^{-1}v^\dagger)\phi = 0`, the
    generalized eigenproblem of the transfer matrix

    .. math::

        \begin{pmatrix} 0 & 1\\ -v^\dagger & E - h_0\end{pmatrix}
        \begin{pmatrix}\phi\\ \lambda\phi\end{pmatrix} = \lambda
        \begin{pmatrix} 1 & 0\\ 0 & v\end{pmatrix}
        \begin{pmatrix}\phi\\ \lambda\phi\end{pmatrix}\, ,

    solved by the QZ algorithm, so that a singular :math:`v` is allowed. The
    modes with :math:`|\lambda| = 1` propagate with the velocity
    :math:`dE/dk = -2\,\mathrm{Im}(\lambda\phi^\dagger v\phi)` (diagonalized
    within degenerate modes). There is no broadening :math:`\eta`, so the
    modes are exact up to the band edges.

    A *conservation_law* :math:`Q` (spin :math:`\sigma_z`, the electron-hole
    :math:`\tau_z` of a normal lead, a valley) commutes with :math:`h_0` and
    :math:`v`. The modes are then found in each eigenspace of :math:`Q`
    separately, so that every mode carries one of its eigenvalues, even
    when modes of different blocks are degenerate.

    :param h0: Square complex array, Hermitian. Cell Hamiltonian.
    :param v: Square complex array, same shape. Coupling of a cell to the
        next one, away from the device.
    :param energy: Real number. Energy, not at a band edge of the lead.
    :param conservation_law: Hermitian complex array, shape of *h0*, or None.
        Default value None.

    :returns:
        * **modes** -- **LeadModes**: the wave functions, momenta, velocities and blocks.
    '''
    h0 = np.atleast_2d(np.asarray(h0, dtype='c16'))
    v = np.atleast_2d(np.asarray(v, dtype='c16'))
    error_handling.lead(h0, v)
    error_handling.real_number(energy, 'energy')
    law = _law_blocks(h0, v, conservation_law)
    return _lead_modes(h0, v, float(energy), law)[0]


def _law_blocks(
    h0: NDArray[np.complex128], v: NDArray[np.complex128], law: ArrayLike | None,
) -> list[NDArray[np.complex128]] | None:
    r'''
    Private function. Validate a conservation law of a lead and return an
    orthonormal basis (shape (m, d)) of each of its eigenspaces, in
    increasing eigenvalue (None without a conservation law).
    '''
    if law is None:
        return None
    law = np.atleast_2d(np.asarray(law, dtype='c16'))
    error_handling.conservation_law(law, h0, v)
    w, u = np.linalg.eigh(law)
    breaks = np.flatnonzero(np.diff(w) > 1e-8 * max(1., np.max(np.abs(w)))) + 1
    return [u[:, g] for g in np.split(np.arange(len(w)), breaks)]


def _lead_modes(
    h0: NDArray[np.complex128], v: NDArray[np.complex128], energy: float,
    law: list[NDArray[np.complex128]] | None = None, tol: float = 1e-6,
) -> tuple[LeadModes, int, NDArray[np.complex128], NDArray[np.complex128]]:
    r'''
    Private function. The modes of a lead (see *lead_modes*), the number of
    incoming ones, and a basis of every solution that leaves the device or
    decays away from it: its values on cells 0 and 1, *u0* and *u1* of shape
    (m, m). Its first columns are the outgoing modes of *LeadModes*, the
    others span the evanescent modes (a Schur basis of the pencil, which
    stays well conditioned when :math:`v` is singular and many
    :math:`\lambda` vanish). Modes with :math:`||\lambda| - 1| \le` *tol*
    propagate. With the eigenspaces *law* of a conservation law
    (*_law_blocks*), each block :math:`P^\dagger h_0P`, :math:`P^\dagger vP`
    is solved on its own and embedded back.
    '''
    if law is not None:
        # the incoming modes of every block, then the outgoing ones, then
        # the evanescent ones: the order of the single-block case
        phi, k, vel, block, out0, out1, ev0, ev1 = ([] for _ in range(8))
        for b, p in enumerate(law):
            md, n, u0, u1 = _lead_modes(p.conj().T @ h0 @ p, p.conj().T @ v @ p, energy, None, tol)
            phi.append((p @ md.wave_functions[:, :n], p @ md.wave_functions[:, n:]))
            k.append((md.momenta[:n], md.momenta[n:]))
            vel.append((md.velocities[:n], md.velocities[n:]))
            block.append(np.full(n, b))
            out0.append(p @ u0[:, :n])
            out1.append(p @ u1[:, :n])
            ev0.append(p @ u0[:, n:])
            ev1.append(p @ u1[:, n:])
        modes = LeadModes(np.hstack([f[0] for f in phi] + [f[1] for f in phi]),
                          np.concatenate([f[0] for f in k] + [f[1] for f in k]),
                          np.concatenate([f[0] for f in vel] + [f[1] for f in vel]),
                          np.concatenate(block * 2))
        return modes, sum(len(b) for b in block), np.hstack(out0 + ev0), np.hstack(out1 + ev1)
    error_handling.hermitian_operator(h0)
    m = len(h0)
    eye, zero = np.eye(m), np.zeros((m, m))
    a = np.block([[zero, eye], [-v.conj().T, energy * eye - h0]])
    b = np.block([[eye, zero], [zero, v]])
    # decaying (|lambda| < 1, including lambda = 0) first; beta = 0 is lambda = infinity
    _, _, alpha, beta, _, z = LA.ordqz(a, b, sort=lambda al, be: np.abs(al) < (1 - tol) * np.abs(be),
                                       output='complex')
    decay = np.abs(alpha) < (1 - tol) * np.abs(beta)
    unit = ~decay & (np.abs(np.abs(alpha) - np.abs(beta)) <= tol * np.abs(beta)) & (beta != 0)
    lams = alpha[unit] / beta[unit]
    phis, ks, vels = [], [], []
    left = list(range(len(lams)))
    while left:
        # a group of degenerate lambdas: its modes span the kernel of the
        # Hermitian E - h0 - lam v - lam^* v^dagger; diagonalize the velocity there
        group = [i for i in left if abs(lams[i] - lams[left[0]]) < tol]
        left = [i for i in left if i not in group]
        lam = np.mean(lams[group])
        lam /= abs(lam)
        w, u = np.linalg.eigh(energy * eye - h0 - lam * v - np.conj(lam) * v.conj().T)
        phi = u[:, np.argsort(np.abs(w))[:len(group)]]
        vel, rot = np.linalg.eigh(1j * (lam * phi.conj().T @ v @ phi
                                        - np.conj(lam) * phi.conj().T @ v.conj().T @ phi))
        error_handling.lead_band_edge(np.min(np.abs(vel)) > tol, energy)
        phis.append(phi @ rot / np.sqrt(np.abs(vel)))
        ks += [np.angle(lam)] * len(group)
        vels.append(vel)
    phis = np.hstack(phis) if phis else np.zeros((m, 0), 'c16')
    ks, vels = np.array(ks), np.concatenate(vels) if vels else np.zeros(0)
    order = np.lexsort((ks, vels > 0))  # incoming (v < 0) first, then by momentum
    phis, ks, vels = phis[:, order], ks[order], vels[order]
    n_in, n_dec = int(np.sum(vels < 0)), int(np.sum(decay))
    error_handling.lead_band_edge(2 * n_in == len(vels) and n_in + n_dec == m, energy)
    out = phis[:, n_in:]
    u0 = np.hstack([out, z[:m, :n_dec]])
    u1 = np.hstack([out * np.exp(1j * ks[n_in:]), z[m:, :n_dec]])
    return LeadModes(phis, ks, vels, np.zeros(len(ks), int)), n_in, u0, u1


class SMatrix():
    r'''
    The scattering matrix of a device at one energy (see
    *Transport.smatrix*): ``data[a, b]`` is the amplitude of outgoing mode
    *a* for an incoming mode *b* of unit current. The rows are the outgoing
    modes of lead 0, then lead 1, ...; the columns the incoming ones, in the
    order of *lead_info*. It is unitary for a Hermitian device.

    A lead is an integer, or a tuple ``(lead, block)`` that keeps only the
    modes in one block of the lead's conservation law (*Transport.add_lead*),
    for example ``(0, 1)`` for the electrons and ``(0, 0)`` for the holes of
    a normal lead with :math:`Q = \tau_z`.

    :ivar data: Complex ndarray, shape (n_out, n_in).
    :ivar lead_info: List of **LeadModes**, one per lead.
    '''

    def __init__(
        self, data: NDArray[np.complex128], lead_info: list[LeadModes], n_blocks: list[int] | None = None,
    ) -> None:
        self.data = data
        self.lead_info = lead_info
        self._n_blocks = [1] * len(lead_info) if n_blocks is None else list(n_blocks)
        counts = [int(np.sum(m.velocities < 0)) for m in lead_info]
        self._bounds = np.concatenate([[0], np.cumsum(counts)]).astype(int)

    def _modes(self, lead: int | tuple[int, int], outgoing: bool) -> NDArray[np.int64]:
        r'''
        Private method. The rows (*outgoing*) or columns of the modes of a
        lead, or of one block of it.
        '''
        error_handling.lead_block(lead, self._n_blocks)
        l = lead[0] if isinstance(lead, tuple) else lead
        idx = np.arange(self._bounds[l], self._bounds[l + 1])
        if isinstance(lead, tuple):
            n = len(idx)
            blocks = self.lead_info[l].blocks
            idx = idx[(blocks[n:] if outgoing else blocks[:n]) == lead[1]]
        return idx

    def num_propagating(self, lead: int | tuple[int, int]) -> int:
        r'''
        Get the number of propagating modes (incoming, as many as outgoing)
        of a lead, or of one block of it.

        :param lead: Integer or tuple (lead, block).

        :returns:
            * **n** -- Integer.
        '''
        return len(self._modes(lead, False))

    def submatrix(
        self, lead_out: int | tuple[int, int], lead_in: int | tuple[int, int],
    ) -> NDArray[np.complex128]:
        r'''
        Get the block of the scattering matrix from lead *lead_in* to lead
        *lead_out*: the transmission amplitudes :math:`t`, or the reflection
        amplitudes :math:`r` if the two are equal.

        :param lead_out: Integer or tuple (lead, block).
        :param lead_in: Integer or tuple (lead, block).

        :returns:
            * **s** -- Complex ndarray, shape (num_propagating(lead_out), num_propagating(lead_in)).
        '''
        return self.data[np.ix_(self._modes(lead_out, True), self._modes(lead_in, False))]

    def transmission(self, lead_out: int | tuple[int, int], lead_in: int | tuple[int, int]) -> float:
        r'''
        Get the transmission :math:`\sum_{ab}|S_{ab}|^2` from lead *lead_in*
        to lead *lead_out* (the reflection if they are equal). With blocks,
        ``transmission((0, 0), (0, 1))`` is, for example, the Andreev
        reflection of electrons into holes.

        :param lead_out: Integer or tuple (lead, block).
        :param lead_in: Integer or tuple (lead, block).

        :returns:
            * **T** -- Real number.
        '''
        return float(np.sum(np.abs(self.submatrix(lead_out, lead_in)) ** 2))


class Transport():
    r'''
    A finite device, of Hamiltonian *ham* (e.g. *System.ham*), to which
    semi-infinite leads are attached with *add_lead*; *transmission* then
    gives the Landauer transmission between two of them.

    The device is stored sparse. *smatrix*, *wave_function*, *ldos* and
    *transmission* (by default) only factorize the sparse matrix. The
    methods with a broadening *eta* use the dense Green's function, and the
    dense matrix *ham* is built once, the first time it is needed.

    :param ham: Square matrix (sparse or dense). Device Hamiltonian.
    '''

    def __init__(self, ham) -> None:
        ham = sp.csr_matrix(ham, dtype='c16')
        error_handling.square_matrix(ham, 'ham')
        self._ham = ham
        self._dense = None
        self.leads = []  # list of (h0, v, coupling, sites)
        self._laws = []  # eigenspaces of each lead's conservation law, or None

    @property
    def ham(self) -> NDArray[np.complex128]:
        r'''
        The device Hamiltonian, as a dense complex ndarray.
        '''
        if self._dense is None:
            self._dense = self._ham.toarray()
        return self._dense

    def add_lead(
        self, h0: ArrayLike, v: ArrayLike, coupling: ArrayLike, sites: list[int],
        conservation_law: ArrayLike | None = None,
    ) -> None:
        r'''
        Attach a semi-infinite lead (see *surface_green*).

        A *conservation_law* :math:`Q`, Hermitian and commuting with
        :math:`h_0` and :math:`v`, sorts the lead's modes into the eigenspaces
        of :math:`Q` (*lead_modes*), and *smatrix* then resolves its blocks:
        :math:`\sigma_z` for spin-resolved transport, the electron-hole
        :math:`\tau_z` of a normal lead for Andreev reflection. The device
        itself need not conserve :math:`Q`.

        :param h0: Square complex array, shape (m, m). Lead cell Hamiltonian.
        :param v: Square complex array, shape (m, m). Coupling of a lead cell
            to the next one, away from the device.
        :param coupling: Complex array, shape (len(sites), m). Hoppings
            :math:`\tau` between the device *sites* and the lead's surface cell.
        :param sites: List of device site (row) indices the lead touches.
        :param conservation_law: Hermitian complex array, shape (m, m), or None.
            Default value None. Its blocks are numbered in increasing eigenvalue.
        '''
        h0 = np.atleast_2d(np.asarray(h0, dtype='c16'))
        v = np.atleast_2d(np.asarray(v, dtype='c16'))
        error_handling.lead(h0, v)
        coupling = np.atleast_2d(np.asarray(coupling, dtype='c16'))
        error_handling.lead_coupling(coupling, sites, len(h0), self._ham.shape[0])
        law = _law_blocks(h0, v, conservation_law)
        self.leads.append((h0, v, coupling, list(sites)))
        self._laws.append(law)

    def attach_lead(
        self, sys, ks, direction: int = 1, conservation_law: ArrayLike | None = None,
    ) -> None:
        r'''
        Attach a semi-infinite lead given as a 1D *KSpace* model (e.g. a
        ribbon, see *kspace.ribbon*) to the device built from *sys*, without
        writing the coupling by hand.

        The lead's sites sit at :math:`\boldsymbol\tau_o + n\mathbf{a}_1`
        (its *unit_cell* positions, in the coordinates of *sys*, and its
        primitive vector). The device sites found at these positions, with
        the same tag, are copies of lead orbitals; the outermost cell along
        *direction* that holds some of them is the interface. The lead
        starts at the next cell, and couples to the interface sites through
        its own hoppings: :math:`\tau = v` restricted to their orbitals (see
        *lead_from_kspace*, which also gives :math:`h_0` and :math:`v`).
        Every lead orbital with a hopping into the next cell must be present
        in the interface cell; the device is not extended.

        :param sys: **System** instance (one orbital per site) whose
            Hamiltonian, in the same site order, is the device of this
            *Transport*.
        :param ks: **KSpace** instance, 1D and spinless, with hoppings
            between neighbouring cells only.
        :param direction: +1 or -1. Default value 1. Direction, along the
            lead's primitive vector, in which the lead extends away from the device.
        :param conservation_law: Hermitian complex array, shape (norb, norb),
            or None. Default value None. See *add_lead*.

        Example usage::

            strip = ribbon(lattices.square(), hoppings, width=5, direction=1)
            tr = Transport(sys.ham)
            tr.attach_lead(sys, strip, -1)   # at the left end of the device
            tr.attach_lead(sys, strip, 1)    # at the right end
        '''
        from tbkit.orbital import OrbitalSystem
        error_handling.sys(sys)
        error_handling.not_orbital_system(sys, OrbitalSystem)
        error_handling.empty_coor(sys.lat.coor)
        error_handling.lead_device_size(sys.lat.sites, self._ham.shape[0])
        error_handling.spinless(ks.spin)
        h0, v = lead_from_kspace(ks, direction)
        error_handling.lead_space_dim(ks.space_dim, sys.lat.space_dim)
        pos = np.stack([sys.lat.coor[f] for f in ('x', 'y', 'z')[:sys.lat.space_dim]], axis=1)
        a = np.array(ks.lat.prim_vec[0], dtype='f8')
        tau = ks.orbital_positions()
        cell = np.zeros(len(pos), int)
        orb = np.full(len(pos), -1)
        for o in range(ks.norb):
            rel = pos - tau[o]
            n = np.rint(rel @ a / (a @ a)).astype(int)
            hit = (orb < 0) & (np.linalg.norm(rel - n[:, None] * a, axis=1) < error_handling.ATOL) \
                & (sys.lat.coor['tag'] == ks.tags[o])
            cell[hit], orb[hit] = n[hit], o
        on_lead = orb >= 0
        error_handling.lead_overlap(bool(on_lead.any()))
        edge = direction * np.max(direction * cell[on_lead])
        interface = np.flatnonzero(on_lead & (cell == edge))
        bonded = np.flatnonzero(np.any(v != 0, axis=1))  # orbitals coupled to the next cell
        error_handling.lead_interface(sorted(set(bonded) - set(orb[interface])))
        sites = interface[np.isin(orb[interface], bonded)]
        self.add_lead(h0, v, v[orb[sites]], [int(s) for s in sites], conservation_law)

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
        self, energies: ArrayLike, lead_in: int = 0, lead_out: int = 1, eta: float | None = None,
    ) -> NDArray[np.float64]:
        r'''
        Get the transmission from lead *lead_in* to lead *lead_out*: the
        two-terminal conductance in units of :math:`e^2/h` (per spin, for a
        spinless model).

        By default (*eta* None) it is :math:`\sum_{ab}|S_{ab}|^2` over the block of
        the scattering matrix (*smatrix*), exact up to the band edges of the
        leads. With a broadening *eta* (and, as a fallback, for non-Hermitian
        leads or *lead_in* = *lead_out*, with *eta* = 1e-9) it is the Caroli
        formula :math:`T(E) = \mathrm{Tr}[\Gamma_{out} G^r \Gamma_{in} G^a]`.
        The two agree to :math:`O(\eta)`.

        :param energies: Real array. Energies.
        :param lead_in: Integer. Default value 0.
        :param lead_out: Integer. Default value 1.
        :param eta: Positive real or None. Default value None.

        :returns:
            * **T** -- Real ndarray, same length as *energies*.
        '''
        error_handling.lead_index(lead_in, len(self.leads))
        error_handling.lead_index(lead_out, len(self.leads))
        energies = np.atleast_1d(np.asarray(energies, dtype='f8'))
        hermitian = all(np.allclose(h0, h0.conj().T) for h0, _, _, _ in self.leads)
        if eta is None and hermitian and lead_in != lead_out:
            out = np.zeros(len(energies))
            for n, e in enumerate(energies):
                modes, sol = self._scattering(float(e), [lead_in])
                a = self._ham.shape[0] + sum(len(h0) for h0, _, _, _ in self.leads[:lead_out])
                n_out = int(np.sum(modes[lead_out].velocities > 0))
                out[n] = np.sum(np.abs(sol[a:a + n_out]) ** 2)
            return out
        eta = 1e-9 if eta is None else eta
        out = np.zeros(len(energies))
        for n, e in enumerate(energies):
            sig_in = self.self_energy(lead_in, e, eta)
            sig_out = self.self_energy(lead_out, e, eta)
            gam_in = 1j * (sig_in - sig_in.conj().T)
            gam_out = 1j * (sig_out - sig_out.conj().T)
            g = self.get_green(e, eta)
            out[n] = np.trace(gam_out @ g @ gam_in @ g.conj().T).real
        return out

    # ------------------------------------------------------------------
    # Scattering matrix and scattering states
    # ------------------------------------------------------------------

    def _scattering(self, energy: float, inject: list[int]) -> tuple:
        r'''
        Private method. Solve the scattering problem for every incoming mode
        of the leads *inject* (Groth et al. 2014). In lead :math:`l`, the
        wave function on cells 0 and 1 is the incoming mode plus
        :math:`U_0 b_l`, :math:`U_1 b_l` (*_lead_modes*: outgoing and
        evanescent). The unknowns are the device wave function
        :math:`\psi` and the amplitudes :math:`b_l`, and the equations are
        the Schrodinger equation on the device and on the surface cell of
        every lead,

        .. math::

            (E - H)\psi - \sum_l\tau_l U_{0,l}b_l = \sum_l\tau_l\phi^{in}_l\, ,\qquad
            -\tau_l^\dagger\psi + [(E - h_0)U_0 - vU_1]b_l = -(E - h_0 - \lambda_{in}v)\phi^{in}_l\, ,

        one sparse system of size n_sites + sum_l m_l, factorized once
        (*scipy.sparse.linalg.splu*) for every right-hand side.

        :returns:
            * **modes** -- List of **LeadModes**, one per lead.
            * **sol** -- Complex ndarray, shape (n_sites + sum_l m_l, n_in): one
              column per incoming mode of the leads *inject*, in order.
        '''
        error_handling.real_number(energy, 'energy')
        n = self._ham.shape[0]
        found = [_lead_modes(h0, v, energy, law) for (h0, v, _, _), law in zip(self.leads, self._laws)]
        size = n + sum(len(h0) for h0, _, _, _ in self.leads)
        device = (energy * sp.eye(n, dtype='c16') - self._ham).tocoo()
        rows, cols, vals = [device.row], [device.col], [device.data]

        def block(r, c, values):
            rr, cc = np.meshgrid(r, c, indexing='ij')
            rows.append(rr.ravel())
            cols.append(cc.ravel())
            vals.append(values.ravel())

        rhs, a = [], n
        for l, ((h0, v, tau, sites), (modes, n_in, u0, u1)) in enumerate(zip(self.leads, found)):
            m = len(h0)
            lead = np.arange(a, a + m)
            e_h0 = energy * np.eye(m) - h0
            block(sites, lead, -tau @ u0)
            block(lead, sites, -tau.conj().T)
            block(lead, lead, e_h0 @ u0 - v @ u1)
            if l in inject:
                phi = modes.wave_functions[:, :n_in]
                col = np.zeros((size, n_in), 'c16')
                col[sites] = tau @ phi
                col[a:a + m] = -(e_h0 @ phi - v @ phi * np.exp(1j * modes.momenta[:n_in]))
                rhs.append(col)
            a += m
        rhs = np.hstack(rhs) if rhs else np.zeros((size, 0), 'c16')
        if rhs.shape[1] == 0:
            return [f[0] for f in found], rhs
        mat = sp.csc_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                            shape=(size, size))
        sol = spla.splu(mat).solve(rhs)
        return [f[0] for f in found], sol

    def smatrix(self, energy: float) -> SMatrix:
        r'''
        Get the scattering matrix at one energy: the amplitudes of the
        outgoing modes of every lead for each incoming mode of unit
        current, from the exact lead modes (*lead_modes*, no broadening) and
        one sparse factorization of the device. It is unitary for a
        Hermitian device. Every lead must be Hermitian. The blocks of the
        leads' conservation laws (*add_lead*) are addressed as
        ``(lead, block)`` in *SMatrix.submatrix* and *SMatrix.transmission*.

        :param energy: Real number. Energy, not at a band edge of a lead.

        :returns:
            * **S** -- **SMatrix**: *data*, *lead_info*, *submatrix*, *transmission*.

        Example usage::

            s = tr.smatrix(0.3)
            t = s.submatrix(1, 0)           # transmission amplitudes, lead 0 to lead 1
            T = s.transmission(1, 0)        # = tr.transmission(0.3, 0, 1)
        '''
        modes, sol = self._scattering(energy, list(range(len(self.leads))))
        rows, a = [], self._ham.shape[0]
        for (h0, _, _, _), mode in zip(self.leads, modes):
            rows.append(sol[a:a + int(np.sum(mode.velocities > 0))])
            a += len(h0)
        data = np.vstack(rows) if rows else np.zeros((0, 0), 'c16')
        return SMatrix(data, modes, [1 if law is None else len(law) for law in self._laws])

    def wave_function(self, energy: float, lead: int) -> NDArray[np.complex128]:
        r'''
        Get the scattering states injected from a lead: the wave function on
        the device of each incoming mode of unit current (*lead_modes*),
        with the scattered part leaving through every lead.

        :param energy: Real number. Energy, not at a band edge of a lead.
        :param lead: Integer. Injecting lead.

        :returns:
            * **psi** -- Complex ndarray, shape (n_modes, n_sites): one row
              per incoming mode of the lead.
        '''
        error_handling.lead_index(lead, len(self.leads))
        _, sol = self._scattering(energy, [lead])
        return sol[:self._ham.shape[0]].T

    def ldos(self, energy: float) -> NDArray[np.float64]:
        r'''
        Get the local density of states carried by the scattering states,
        :math:`\rho_i(E) = \frac{1}{2\pi}\sum_{l,a}|\psi_{la}(i)|^2` over every
        incoming mode :math:`a` of every lead :math:`l` (unit current
        normalization). The bound states of the device, which no lead
        feeds, are left out.

        :param energy: Real number. Energy, not at a band edge of a lead.

        :returns:
            * **rho** -- Real ndarray, shape (n_sites,).
        '''
        _, sol = self._scattering(energy, list(range(len(self.leads))))
        return np.sum(np.abs(sol[:self._ham.shape[0]]) ** 2, axis=1) / (2 * np.pi)

    # ------------------------------------------------------------------
    # Finite temperature: conductance and thermoelectric coefficients
    # ------------------------------------------------------------------

    def _onsager(
        self, mu: float, temperature: float, lead_in: int, lead_out: int, eta: float | None,
        n_points: int,
    ) -> tuple[float, float, float]:
        r'''
        Private method. The integrals :math:`L_n = \int dE\,(E - \mu)^n\,T(E)
        (-\partial f/\partial E)`, n = 0, 1, 2.
        '''
        energies, weights = _fermi_window(mu, temperature, n_points)
        t = self.transmission(energies, lead_in, lead_out, eta) * weights
        de = energies - mu
        return float(t.sum()), float(np.sum(t * de)), float(np.sum(t * de**2))

    def conductance(
        self, mu: float, temperature: float = 0., lead_in: int = 0, lead_out: int = 1,
        eta: float | None = None, n_points: int = 101,
    ) -> float:
        r'''
        Get the two-terminal linear conductance at a finite temperature, in
        units of :math:`e^2/h`,

        .. math::

            G = \int dE\,\left(-\frac{\partial f}{\partial E}\right)T(E)\, ,

        with the Fermi-Dirac :math:`f` at chemical potential :math:`\mu`. The
        integral runs over :math:`|E - \mu| \le 36k_BT` on *n_points* equally
        spaced energies (the trapezoidal rule, spectrally accurate for
        a smooth :math:`T(E)`). Resolving features of :math:`T(E)` narrower than
        :math:`k_BT` needs more points. At zero temperature it is :math:`T(\mu)`.

        :param mu: Real number. Chemical potential.
        :param temperature: Positive real or zero. Default value 0 (:math:`k_B = 1`).
        :param lead_in: Integer. Default value 0.
        :param lead_out: Integer. Default value 1.
        :param eta: Positive real or None. Default value None (see *transmission*).
        :param n_points: Positive integer. Default value 101.

        :returns:
            * **G** -- Real number.
        '''
        error_handling.real_number(mu, 'mu')
        error_handling.positive_real_zero(temperature, 'temperature')
        error_handling.positive_int(n_points, 'n_points')
        if temperature == 0:
            return float(self.transmission([mu], lead_in, lead_out, eta)[0])
        return self._onsager(mu, temperature, lead_in, lead_out, eta, n_points)[0]

    def thermoelectric(
        self, mu: float, temperature: float, lead_in: int = 0, lead_out: int = 1,
        eta: float | None = None, n_points: int = 101,
    ) -> tuple[float, float, float]:
        r'''
        Get the two-terminal thermoelectric coefficients in linear response
        (Sivan and Imry 1986), from :math:`L_n = \int dE\,(E - \mu)^n\,T(E)
        (-\partial f/\partial E)`:

        .. math::

            G = L_0\, ,\qquad S = -\frac{L_1}{T L_0}\, ,\qquad
            \kappa = \frac{1}{T}\left(L_2 - \frac{L_1^2}{L_0}\right)\, ,

        the conductance (:math:`e^2/h`), the thermopower (:math:`k_B/e`, for
        electrons of charge :math:`-e`) and the electronic thermal conductance
        at zero current (:math:`k_B/h` times the energy unit). At low
        temperature they obey the Mott formula
        :math:`S = -\frac{\pi^2}{3}T\,\frac{d\ln T(E)}{dE}` and the
        Wiedemann-Franz law :math:`\kappa = \frac{\pi^2}{3}TG`. The integrals
        are those of *conductance*.

        :param mu: Real number. Chemical potential.
        :param temperature: Positive real (:math:`k_B = 1`).
        :param lead_in: Integer. Default value 0.
        :param lead_out: Integer. Default value 1.
        :param eta: Positive real or None. Default value None (see *transmission*).
        :param n_points: Positive integer. Default value 101.

        :returns:
            * **G** -- Real number.
            * **S** -- Real number (NaN when nothing is transmitted).
            * **kappa** -- Real number.
        '''
        error_handling.real_number(mu, 'mu')
        error_handling.positive_real(temperature, 'temperature')
        error_handling.positive_int(n_points, 'n_points')
        l0, l1, l2 = self._onsager(mu, temperature, lead_in, lead_out, eta, n_points)
        if l0 < 1e-14:
            return l0, np.nan, l2 / temperature
        return l0, -l1 / (temperature * l0), (l2 - l1**2 / l0) / temperature

    # ------------------------------------------------------------------
    # Multi-terminal conductance, currents and noise
    # ------------------------------------------------------------------

    def _broadenings(self, energy: float, eta: float) -> list[NDArray[np.complex128]]:
        r'''
        Private method. The broadening :math:`\Gamma_l = i(\Sigma_l - \Sigma_l^\dagger)`
        of every lead, restricted to its sites (shape (len(sites), len(sites))).
        '''
        out = []
        for h0, v, tau, _ in self.leads:
            sig = tau @ surface_green(h0, v, energy, eta) @ tau.conj().T
            out.append(1j * (sig - sig.conj().T))
        return out

    def transmission_matrix(self, energy: float, eta: float = 1e-9) -> NDArray[np.float64]:
        r'''
        Get the transmissions between every pair of leads at one energy,
        :math:`T_{pq} = \mathrm{Tr}[\Gamma_pG^r\Gamma_qG^a]` from lead
        :math:`q` to lead :math:`p` (the diagonal, a reflection, is set to 0).

        :param energy: Real number.
        :param eta: Positive real. Default value 1e-9.

        :returns:
            * **T** -- Real ndarray, shape (n_leads, n_leads).
        '''
        g = self.get_green(energy, eta)
        gam = self._broadenings(energy, eta)
        n = len(self.leads)
        out = np.zeros((n, n))
        for p in range(n):
            for q in range(n):
                if p != q:
                    gpq = g[np.ix_(self.leads[p][3], self.leads[q][3])]
                    out[p, q] = np.trace(gam[p] @ gpq @ gam[q] @ gpq.conj().T).real
        return out

    def conductance_matrix(
        self, energy: float, eta: float = 1e-9, temperature: float = 0., n_points: int = 101,
    ) -> NDArray[np.float64]:
        r'''
        Get the Landauer-Buttiker conductance matrix (Buttiker 1986), in units
        of :math:`e^2/h`: the linear-response currents flowing *into* the
        device from each lead, :math:`I_p = \sum_q G_{pq}V_q`, with

        .. math::

            G_{pq} = -T_{pq}\ (p\neq q)\, ,\qquad G_{pp} = \sum_{q\neq p}T_{qp}\, .

        Its rows and columns add up to zero (current conservation, and no
        current at equal voltages). At a finite *temperature* it is averaged
        over the Fermi window, :math:`\int dE\,(-\partial f/\partial E)\,G(E)`
        (see *conductance*).

        :param energy: Real number. Fermi energy (chemical potential).
        :param eta: Positive real. Default value 1e-9.
        :param temperature: Positive real or zero. Default value 0 (:math:`k_B = 1`).
        :param n_points: Positive integer. Default value 101. Energies in the
            Fermi window, at a finite temperature.

        :returns:
            * **G** -- Real ndarray, shape (n_leads, n_leads).
        '''
        error_handling.positive_real_zero(temperature, 'temperature')
        error_handling.positive_int(n_points, 'n_points')
        if temperature > 0:
            energies, weights = _fermi_window(energy, temperature, n_points)
            return sum(w * self.conductance_matrix(float(e), eta) for e, w in zip(energies, weights))
        t = self.transmission_matrix(energy, eta)
        return np.diag(t.sum(axis=0)) - t

    def four_terminal_resistance(
        self, energy: float, current: tuple[int, int], voltage: tuple[int, int], eta: float = 1e-9,
        temperature: float = 0., n_points: int = 101,
    ) -> float:
        r'''
        Get a four-terminal (or two-terminal) resistance from the
        Landauer-Buttiker equations: a current :math:`I` enters through lead
        ``current[0]`` and leaves through ``current[1]``, every other lead is
        an ideal voltage probe (no net current), and the result is
        :math:`R = (V_a - V_b)/I` with ``voltage = (a, b)``, in units of
        :math:`h/e^2`. On a Hall bar, a longitudinal pair gives
        :math:`R_{xx}` and a transverse pair :math:`R_{xy}`; in the quantum
        Hall regime with :math:`\nu` edge channels, :math:`|R_{xy}| = 1/\nu`
        and :math:`R_{xx} = 0` (Buttiker 1988).

        :param energy: Real number. Fermi energy.
        :param current: Tuple of two distinct lead indices (source, drain).
        :param voltage: Tuple of two lead indices (a, b).
        :param eta: Positive real. Default value 1e-9.
        :param temperature: Positive real or zero. Default value 0 (see *conductance_matrix*).
        :param n_points: Positive integer. Default value 101.

        :returns:
            * **R** -- Real number.
        '''
        error_handling.lead_pair(current, len(self.leads), 'current', True)
        error_handling.lead_pair(voltage, len(self.leads), 'voltage', False)
        g = self.conductance_matrix(energy, eta, temperature, n_points)
        source, drain = current
        keep = [p for p in range(len(self.leads)) if p != drain]  # the drain is grounded
        currents = np.zeros(len(keep))
        currents[keep.index(source)] = 1.
        volts = np.zeros(len(self.leads))
        volts[keep] = LA.solve(g[np.ix_(keep, keep)], currents)
        return float(volts[voltage[0]] - volts[voltage[1]])

    def bond_currents(self, energy: float, lead: int, eta: float = 1e-9) -> NDArray[np.float64]:
        r'''
        Get the bond currents carried, per unit energy, by the electrons
        injected from one lead (the non-equilibrium current when that lead's
        electrochemical potential is raised),

        .. math::

            J_{i\to j} = -2\,\mathrm{Im}\left[H_{ij}\,A^{(q)}_{ji}\right]\, ,\qquad
            A^{(q)} = G^r\Gamma_qG^a\, ,

        in units of :math:`e/h` per unit energy: summed over a cross-section
        of a two-terminal device, it is the transmission :math:`T(E)`. The
        matrix is antisymmetric, and the currents are conserved at every
        site not touched by a lead.

        :param energy: Real number.
        :param lead: Integer. Injecting lead.
        :param eta: Positive real. Default value 1e-9.

        :returns:
            * **J** -- Real ndarray, shape of *ham*: J[i, j] is the current
              from site i to site j.
        '''
        error_handling.lead_index(lead, len(self.leads))
        g = self.get_green(energy, eta)
        sites = self.leads[lead][3]
        gam = self._broadenings(energy, eta)[lead]
        spectral = g[:, sites] @ gam @ g[:, sites].conj().T
        return -2 * np.imag(self.ham * spectral.T)

    def local_currents(
        self, energy: float, lead: int, positions: ArrayLike, eta: float = 1e-9,
    ) -> NDArray[np.float64]:
        r'''
        Get the current density vector on every site from the bond currents
        (*bond_currents*): half of the currents of the bonds leaving the
        site, along their bond vectors,
        :math:`\mathbf{j}_i = \frac12\sum_jJ_{i\to j}(\mathbf{r}_j - \mathbf{r}_i)`.

        :param energy: Real number.
        :param lead: Integer. Injecting lead.
        :param positions: Real array, shape (n_sites, d). Site positions
            (e.g. ``np.column_stack([lat.coor['x'], lat.coor['y']])``).
        :param eta: Positive real. Default value 1e-9.

        :returns:
            * **j** -- Real ndarray, shape (n_sites, d).
        '''
        positions = np.asarray(positions, dtype='f8')
        error_handling.site_positions(positions, len(self.ham))
        bonds = self.bond_currents(energy, lead, eta)
        return 0.5 * (bonds @ positions - bonds.sum(axis=1)[:, None] * positions)

    def transmission_eigenvalues(
        self, energy: float, lead_in: int = 0, lead_out: int = 1, eta: float = 1e-9,
    ) -> NDArray[np.float64]:
        r'''
        Get the transmission eigenvalues :math:`T_n`, eigenvalues of
        :math:`t^\dagger t` with :math:`t = \Gamma_{out}^{1/2}G^r\Gamma_{in}^{1/2}`:
        their sum is the transmission.

        :param energy: Real number.
        :param lead_in: Integer. Default value 0.
        :param lead_out: Integer. Default value 1.
        :param eta: Positive real. Default value 1e-9.

        :returns:
            * **T_n** -- Real ndarray, in decreasing order, of length
              min(len(sites_in), len(sites_out)).
        '''
        error_handling.lead_index(lead_in, len(self.leads))
        error_handling.lead_index(lead_out, len(self.leads))
        g = self.get_green(energy, eta)
        gam = self._broadenings(energy, eta)
        s_in, s_out = self.leads[lead_in][3], self.leads[lead_out][3]
        return _eigen_transmissions(gam[lead_out], g[np.ix_(s_out, s_in)], gam[lead_in])

    def shot_noise(
        self, energies: ArrayLike, lead_in: int = 0, lead_out: int = 1, eta: float = 1e-9,
    ) -> NDArray[np.float64]:
        r'''
        Get the zero-temperature shot-noise factor :math:`\sum_nT_n(1-T_n)`
        (Lesovik 1989; Buttiker 1990): the noise power at bias :math:`V` is
        :math:`S = 2\frac{e^3|V|}{h}\sum_nT_n(1-T_n)` (per spin).

        :param energies: Real array. Energies.
        :param lead_in: Integer. Default value 0.
        :param lead_out: Integer. Default value 1.
        :param eta: Positive real. Default value 1e-9.

        :returns:
            * **noise** -- Real ndarray, same length as *energies*.
        '''
        error_handling.frequencies(energies, 'energies')
        energies = np.atleast_1d(np.asarray(energies, dtype='f8'))
        return np.array([_noise(self.transmission_eigenvalues(float(e), lead_in, lead_out, eta))
                                for e in energies])

    def fano_factor(
        self, energies: ArrayLike, lead_in: int = 0, lead_out: int = 1, eta: float = 1e-9,
    ) -> NDArray[np.float64]:
        r'''
        Get the Fano factor of the zero-temperature shot noise,

        .. math::

            F = \frac{\sum_nT_n(1-T_n)}{\sum_nT_n}\, ,

        the noise in units of the Poisson value :math:`2e|I|`: 0 for a
        ballistic conductor (every :math:`T_n` is 0 or 1), close to 1 for a
        tunnel barrier, 1/3 for a diffusive wire (Beenakker and Buttiker 1992).

        :param energies: Real array. Energies.
        :param lead_in: Integer. Default value 0.
        :param lead_out: Integer. Default value 1.
        :param eta: Positive real. Default value 1e-9.

        :returns:
            * **F** -- Real ndarray, same length as *energies* (NaN where
              nothing is transmitted).
        '''
        error_handling.frequencies(energies, 'energies')
        energies = np.atleast_1d(np.asarray(energies, dtype='f8'))
        return np.array([_fano(self.transmission_eigenvalues(float(e), lead_in, lead_out, eta))
                                for e in energies])


def _fermi_window(
    mu: float, temperature: float, n_points: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    r'''
    Private function. Energies and weights of the trapezoidal rule for
    :math:`\int dE\,(-\partial f/\partial E)\,F(E)`, on *n_points* equally
    spaced energies with :math:`|E - \mu| \le 36T` (the weight is below
    1e-15 beyond). The weights add up to 1.
    '''
    x = np.linspace(-36., 36., n_points) if n_points > 1 else np.zeros(1)
    w = 1 / (4 * np.cosh(x / 2) ** 2)
    return mu + temperature * x, w / w.sum()


def _sqrt_psd(mat: NDArray[np.complex128]) -> NDArray[np.complex128]:
    '''
    Private function. Square root of a positive semi-definite Hermitian matrix.
    '''
    w, u = np.linalg.eigh(0.5 * (mat + mat.conj().T))
    return (u * np.sqrt(np.clip(w, 0., None))) @ u.conj().T


def _eigen_transmissions(
    gam_out: NDArray[np.complex128], g: NDArray[np.complex128], gam_in: NDArray[np.complex128],
) -> NDArray[np.float64]:
    r'''
    Private function. Transmission eigenvalues: the squared singular values
    of :math:`\Gamma_{out}^{1/2}G\,\Gamma_{in}^{1/2}` (*g* the block of the
    Green's function from the input to the output sites).
    '''
    t = _sqrt_psd(gam_out) @ g @ _sqrt_psd(gam_in)
    return np.linalg.svd(t, compute_uv=False) ** 2


def _noise(t: NDArray[np.float64]) -> float:
    '''
    Private function. Shot-noise factor sum T_n (1 - T_n).
    '''
    return float(np.sum(t * (1 - t)))


def _fano(t: NDArray[np.float64]) -> float:
    '''
    Private function. Fano factor from transmission eigenvalues (NaN if none transmit).
    '''
    total = t.sum()
    return _noise(t) / float(total) if total > 1e-12 else np.nan


def slices_from_positions(x: ArrayLike, tol: float = 1e-6) -> list[list[int]]:
    r'''
    Group the sites of a device into slices of equal coordinate *x* (e.g.
    the columns of a strip along :math:`x`), ordered by increasing *x*, for
    *RecursiveTransport*.

    :param x: Real array. One coordinate per site (e.g. ``lat.coor['x']``).
    :param tol: Positive real. Default value 1e-6. Coordinates closer than
        *tol* belong to the same slice.

    :returns:
        * **slices** -- List of lists of site indices.
    '''
    x = np.asarray(x, dtype='f8').ravel()
    error_handling.ndarray_empty(x, 'x')
    error_handling.positive_real(tol, 'tol')
    order = np.argsort(x, kind='stable')
    breaks = np.flatnonzero(np.diff(x[order]) > tol) + 1
    return [sorted(s.tolist()) for s in np.split(order, breaks)]


class RecursiveTransport():
    r'''
    Two-terminal transport through a long quasi-1D device by the recursive
    Green's function method (Thouless and Kirkpatrick 1981; Lee and Fisher
    1981; MacKinnon 1985). The device is cut into slices :math:`1\dots N`,
    each coupled only to its neighbours, with the left lead on slice 1 and
    the right lead on slice :math:`N`. The left-connected Green's functions

    .. math::

        g_1 = [z - H_{11} - \Sigma_L]^{-1}\, ,\qquad
        g_{i} = [z - H_{ii} - H_{i,i-1}\,g_{i-1}H_{i-1,i}]^{-1}\, ,

    (:math:`\Sigma_R` added in the last one, :math:`z = E + i\eta`) and the
    propagator :math:`G_{i1} = g_iH_{i,i-1}G_{i-1,1}` give
    :math:`T = \mathrm{Tr}[\Gamma_RG_{N1}\Gamma_LG_{N1}^\dagger]`. The cost
    grows linearly with the length, and only slice-sized matrices are
    inverted, never the whole device (compare *Transport*, which inverts
    the dense device matrix).

    :param ham: Square matrix, sparse (preferred for long devices) or dense.
        Device Hamiltonian.
    :param slices: List of lists of site indices: a partition of the device
        into slices, ordered from the left lead to the right one, with
        hoppings between neighbouring slices only (see *slices_from_positions*).
    :param left: Tuple (h0, v, coupling): the left lead (see *Transport.add_lead*),
        with a coupling of shape (len(slices[0]), len(h0)), its rows in the
        order of ``slices[0]``.
    :param right: Tuple (h0, v, coupling): the right lead, coupled to ``slices[-1]``.

    Example usage::

        slices = slices_from_positions(lat.coor['x'])
        rgf = RecursiveTransport(sys.ham, slices, (h0, v_left, tau), (h0, v_right, tau))
        T = rgf.transmission(energies)
    '''

    def __init__(self, ham, slices: list[list[int]], left: tuple, right: tuple) -> None:
        ham = sp.csr_matrix(ham, dtype='c16')
        error_handling.square_matrix(ham, 'ham')
        error_handling.slices(slices, ham.shape[0])
        label = np.empty(ham.shape[0], dtype=int)
        pos = np.empty(ham.shape[0], dtype=int)
        for n, s in enumerate(slices):
            label[s] = n
            pos[s] = np.arange(len(s))
        coo = ham.tocoo()
        coo.sum_duplicates()
        lr, lc = label[coo.row], label[coo.col]
        error_handling.block_tridiagonal(int(np.abs(lr - lc).max(initial=0)))
        self.slices = [list(s) for s in slices]
        sizes = [len(s) for s in self.slices]
        self.blocks = [np.zeros((m, m), 'c16') for m in sizes]
        self.couplings = [np.zeros((m, k), 'c16') for m, k in zip(sizes[:-1], sizes[1:])]
        # scatter every nonzero entry into its block (lc == lr) or coupling
        # (lc == lr + 1), grouped by slice; the lower couplings are their adjoints
        for target, keep in ((self.blocks, lr == lc), (self.couplings, lc == lr + 1)):
            order = np.argsort(lr[keep], kind='stable')
            rows, cols = pos[coo.row[keep]][order], pos[coo.col[keep]][order]
            vals, owner = coo.data[keep][order], lr[keep][order]
            bounds = np.searchsorted(owner, np.arange(len(target) + 1))
            for n, mat in enumerate(target):
                a, b = bounds[n], bounds[n + 1]
                mat[rows[a:b], cols[a:b]] = vals[a:b]
        self.leads = []
        for lead, s in ((left, self.slices[0]), (right, self.slices[-1])):
            error_handling.lead_tuple(lead)
            h0 = np.atleast_2d(np.asarray(lead[0], dtype='c16'))
            v = np.atleast_2d(np.asarray(lead[1], dtype='c16'))
            error_handling.lead(h0, v)
            tau = np.atleast_2d(np.asarray(lead[2], dtype='c16'))
            error_handling.lead_coupling(tau, list(range(len(s))), len(h0), len(s))
            self.leads.append((h0, v, tau))

    def _propagator(self, energy: float, eta: float) -> tuple:
        r'''
        Private method. :math:`G_{N1}` and the broadenings of the two leads.
        '''
        sig = [tau @ surface_green(h0, v, energy, eta) @ tau.conj().T for h0, v, tau in self.leads]
        gam = [1j * (s - s.conj().T) for s in sig]
        z = energy + 1j * eta
        last = len(self.blocks) - 1
        g = g_n1 = None
        for i, h in enumerate(self.blocks):
            a = z * np.eye(len(h)) - h
            if i == 0:
                a = a - sig[0]
            else:
                v = self.couplings[i - 1]
                a = a - v.conj().T @ g @ v
            if i == last:
                a = a - sig[1]
            g = LA.inv(a)
            g_n1 = g if i == 0 else g @ self.couplings[i - 1].conj().T @ g_n1
        return g_n1, gam

    def transmission(self, energies: ArrayLike, eta: float = 1e-9) -> NDArray[np.float64]:
        r'''
        Get the transmission from the left lead to the right one: the same
        as *Transport.transmission* for the same device and leads.

        :param energies: Real array. Energies.
        :param eta: Positive real. Default value 1e-9.

        :returns:
            * **T** -- Real ndarray, same length as *energies*.
        '''
        error_handling.frequencies(energies, 'energies')
        energies = np.atleast_1d(np.asarray(energies, dtype='f8'))
        error_handling.positive_real(eta, 'eta')
        out = np.zeros(len(energies))
        for n, e in enumerate(energies):
            g, (gam_l, gam_r) = self._propagator(float(e), eta)
            out[n] = np.trace(gam_r @ g @ gam_l @ g.conj().T).real
        return out

    def transmission_eigenvalues(self, energy: float, eta: float = 1e-9) -> NDArray[np.float64]:
        r'''
        Get the transmission eigenvalues (see *Transport.transmission_eigenvalues*).

        :param energy: Real number.
        :param eta: Positive real. Default value 1e-9.

        :returns:
            * **T_n** -- Real ndarray, in decreasing order.
        '''
        error_handling.real_number(energy, 'energy')
        error_handling.positive_real(eta, 'eta')
        g, (gam_l, gam_r) = self._propagator(energy, eta)
        return _eigen_transmissions(gam_r, g, gam_l)

    def fano_factor(self, energies: ArrayLike, eta: float = 1e-9) -> NDArray[np.float64]:
        r'''
        Get the Fano factor of the shot noise (see *Transport.fano_factor*).

        :param energies: Real array. Energies.
        :param eta: Positive real. Default value 1e-9.

        :returns:
            * **F** -- Real ndarray, same length as *energies*.
        '''
        error_handling.frequencies(energies, 'energies')
        energies = np.atleast_1d(np.asarray(energies, dtype='f8'))
        return np.array([_fano(self.transmission_eigenvalues(float(e), eta)) for e in energies])
