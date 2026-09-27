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

With more leads, *Transport.conductance_matrix* and
*four_terminal_resistance* solve the Landauer-Buttiker equations (Hall
bars); *bond_currents* and *local_currents* map the current;
*transmission_eigenvalues*, *shot_noise* and *fano_factor* give the shot
noise. *RecursiveTransport* computes the two-terminal transmission of long
quasi-1D devices slice by slice (recursive Green's function), without
inverting the whole device.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
import scipy.linalg as LA
import scipy.sparse as sp

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

    def conductance_matrix(self, energy: float, eta: float = 1e-9) -> NDArray[np.float64]:
        r'''
        Get the Landauer-Buttiker conductance matrix (Buttiker 1986), in units
        of :math:`e^2/h`: the linear-response currents flowing *into* the
        device from each lead, :math:`I_p = \sum_q G_{pq}V_q`, with

        .. math::

            G_{pq} = -T_{pq}\ (p\neq q)\, ,\qquad G_{pp} = \sum_{q\neq p}T_{qp}\, .

        Its rows and columns add up to zero (current conservation, and no
        current at equal voltages).

        :param energy: Real number. Fermi energy.
        :param eta: Positive real. Default value 1e-9.

        :returns:
            * **G** -- Real ndarray, shape (n_leads, n_leads).
        '''
        t = self.transmission_matrix(energy, eta)
        return np.diag(t.sum(axis=0)) - t

    def four_terminal_resistance(
        self, energy: float, current: tuple[int, int], voltage: tuple[int, int], eta: float = 1e-9,
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

        :returns:
            * **R** -- Real number.
        '''
        error_handling.lead_pair(current, len(self.leads), 'current', True)
        error_handling.lead_pair(voltage, len(self.leads), 'voltage', False)
        g = self.conductance_matrix(energy, eta)
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
