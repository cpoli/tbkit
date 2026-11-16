r'''
Exact diagonalization of small interacting clusters: the reference against
which the mean-field solvers of :mod:`tbkit.meanfield` are checked.

The Hamiltonian is the extended Hubbard model of any single-particle
matrix :math:`t_{ij}` (e.g. *System.ham*, or *KSpace.finite_ham* for a
torus),

.. math::

    H = \sum_{ij\sigma} t_{ij}c^\dagger_{i\sigma}c_{j\sigma}
        + \sum_i U_i\,n_{i\uparrow}n_{i\downarrow}
        + \sum_{i<j} V_{ij}\,n_i n_j\, ,\qquad n_i = n_{i\uparrow} + n_{i\downarrow}\, ,

or its spinless version (one species, no *U*). It conserves
:math:`N_\uparrow` and :math:`N_\downarrow`, so it is built in one sector
at a time (:class:`FockSector`). A basis state is a pair of bit patterns,
bit *i* the occupation of site *i*, in increasing order; the fermion
operators are ordered by site, all spins up before all spins down, so
:math:`c^\dagger_i c_j` carries the sign :math:`(-1)^m`, *m* the number of
occupied sites strictly between *i* and *j*. Every hopping is applied to
all basis states at once (one vectorized pass per bond), and the lowest
states come from ``scipy.sparse.linalg.eigsh``: about 12 sites at half
filling (853776 states) take seconds.
'''
from __future__ import annotations

import functools
import itertools
from math import comb

import numpy as np
import scipy.linalg as LA
import scipy.sparse as sp
import scipy.sparse.linalg as sla
from numpy.typing import ArrayLike, NDArray

import tbkit.error_handling as error_handling

#: Largest sector diagonalized densely (all eigenvalues), above it eigsh.
DENSE_MAX = 1500

_M1 = np.uint64(0x5555555555555555)
_M2 = np.uint64(0x3333333333333333)
_M4 = np.uint64(0x0F0F0F0F0F0F0F0F)
_H01 = np.uint64(0x0101010101010101)


def _popcount(x: NDArray) -> NDArray[np.int64]:
    '''
    Private function. Number of set bits of every element of an integer array.
    '''
    x = x.astype(np.uint64)
    x = x - ((x >> np.uint64(1)) & _M1)
    x = (x & _M2) + ((x >> np.uint64(2)) & _M2)
    x = (x + (x >> np.uint64(4))) & _M4
    return ((x * _H01) >> np.uint64(56)).astype(np.int64)


@functools.lru_cache(maxsize=8)
def _all_patterns(n_sites: int, n_particles: int) -> NDArray[np.int64]:
    '''
    Private function. Cached (read-only) result of *_patterns*, reused by
    the momentum sectors of one cluster.
    '''
    count = comb(n_sites, n_particles)
    if n_particles == 0:
        out = np.zeros(1, np.int64)
    else:
        sites = np.fromiter(itertools.chain.from_iterable(itertools.combinations(range(n_sites), n_particles)),
                            dtype=np.int64, count=count * n_particles).reshape(count, n_particles)
        out = np.sort(np.sum(np.left_shift(1, sites), axis=1))
    out.flags.writeable = False
    return out


def _patterns(n_sites: int, n_particles: int) -> NDArray[np.int64]:
    '''
    Private function. Every way to put *n_particles* fermions on *n_sites*
    sites, as bit patterns in increasing order.
    '''
    return _all_patterns(n_sites, n_particles).copy()


def _occupations(patterns: NDArray[np.int64], n_sites: int) -> NDArray[np.float64]:
    '''
    Private function. Occupation numbers, shape (len(patterns), n_sites).
    '''
    return ((patterns[:, None] >> np.arange(n_sites)[None, :]) & 1).astype('f8')


def _hop(patterns: NDArray[np.int64], i: int, j: int):
    r'''
    Private function. :math:`c^\dagger_i c_j` (i != j) on every pattern: the
    patterns it acts on (indices), their images (indices) and the signs.
    '''
    occ_j = (patterns >> j) & 1
    occ_i = (patterns >> i) & 1
    cols = np.flatnonzero(occ_j & (1 - occ_i))
    src = patterns[cols]
    lo, hi = min(i, j), max(i, j)
    between = ((1 << hi) - 1) ^ ((1 << (lo + 1)) - 1)
    signs = 1 - 2 * (_popcount(src & between) & 1)
    rows = np.searchsorted(patterns, src ^ ((1 << i) | (1 << j)))
    return rows, cols, signs


def _hopping_matrix(patterns: NDArray[np.int64], h: NDArray, occ: NDArray[np.float64]) -> sp.csr_matrix:
    '''
    Private function. One species' kinetic energy in the basis *patterns*.
    '''
    dim = len(patterns)
    rows, cols, vals = [np.arange(dim)], [np.arange(dim)], [occ @ np.diag(h)]
    for i, j in zip(*np.nonzero(h - np.diag(np.diag(h)))):
        r, c, s = _hop(patterns, int(i), int(j))
        rows.append(r)
        cols.append(c)
        vals.append(s * h[i, j])
    return sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                         shape=(dim, dim)).tocsr()


def _pair_coefficients(w: NDArray, tol: float = 1e-12):
    r'''
    Private function. The two-body operator
    :math:`\tfrac12\sum W_{abcd}a^\dagger_aa^\dagger_ba_da_c` as
    :math:`\sum_{a<b,\,c<d} C_{ab,cd}\,a^\dagger_aa^\dagger_ba_da_c`: the
    pairs (c, d) annihilated, and for each the pairs (a, b) created with
    their nonzero coefficients.
    '''
    c = (w - w.transpose(1, 0, 2, 3) - w.transpose(0, 1, 3, 2) + w.transpose(1, 0, 3, 2)) / 2
    n = len(w)
    lo, hi = np.triu_indices(n, 1)
    block = c[lo[:, None], hi[:, None], lo[None, :], hi[None, :]]  # (created pair, annihilated pair)
    scale = max(1., float(np.max(np.abs(block)))) if block.size else 1.
    out = []
    for q in range(len(lo)):
        keep = np.flatnonzero(np.abs(block[:, q]) > tol * scale)
        if len(keep):
            out.append((int(lo[q]), int(hi[q]), lo[keep], hi[keep], block[keep, q]))
    return out


def _two_body_matrix(patterns: NDArray[np.int64], w: NDArray) -> sp.csr_matrix:
    '''
    Private function. A two-body operator of spinless fermions in the basis
    *patterns* (each annihilated pair applied to every state at once).
    '''
    dim = len(patterns)
    rows, cols, vals = [np.zeros(0, int)], [np.zeros(0, int)], [np.zeros(0, w.dtype)]
    for c, d, created_a, created_b, coef in _pair_coefficients(w):
        cols0 = np.flatnonzero(((patterns >> c) & 1) & ((patterns >> d) & 1))
        if not len(cols0):
            continue
        s = patterns[cols0]
        # a_d a_c: (-1)^(occupied below c) (-1)^(occupied below d, after removing c)
        sign0 = _popcount(s & ((1 << c) - 1)) + _popcount(s & ((1 << d) - 1)) - 1
        s2 = s ^ ((1 << c) | (1 << d))
        for a, b, x in zip(created_a, created_b, coef):
            empty = np.flatnonzero((((s2 >> a) | (s2 >> b)) & 1) == 0)
            if not len(empty):
                continue
            t = s2[empty]
            # a+_a a+_b: (-1)^(occupied below b) (-1)^(occupied below a), a < b
            sign = sign0[empty] + _popcount(t & ((1 << int(b)) - 1)) + _popcount(t & ((1 << int(a)) - 1))
            rows.append(np.searchsorted(patterns, t | (1 << int(a)) | (1 << int(b))))
            cols.append(cols0[empty])
            vals.append((1 - 2 * (sign & 1)) * x)
    return sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                         shape=(dim, dim)).tocsr()


def project_interaction(states: ArrayLike, V) -> NDArray[np.complex128]:
    r'''
    Get a density-density interaction projected onto a set of
    single-particle states :math:`\phi_a` (e.g. the Bloch states of one
    band of a torus): with :math:`c_r \to \sum_a\phi_a(r)\,a_a`,

    .. math::

        \sum_{r<r'}V_{rr'}n_rn_{r'} \to \frac12\sum_{abcd}W_{abcd}\,
        a^\dagger_aa^\dagger_ba_da_c\, ,\qquad
        W_{abcd} = \sum_{rr'}V_{rr'}\,\phi^*_a(r)\phi^*_b(r')\phi_c(r)\phi_d(r')\, .

    Pass *W* to *exact_diagonalization*, with the projected one-body
    Hamiltonian (``diag`` of the band energies) as *ham*. This is how a
    partially filled flat band is studied on clusters too large for the
    full Fock space (e.g. a fractional Chern insulator).

    Parameters
    ----------
    states : Complex array, shape (N, n)
        Orthonormal single-particle states (columns) on the N sites.
    V : Real symmetric (N, N) matrix with zero diagonal
        Density-density interaction on the sites.

    Returns
    -------
    W : Complex ndarray, shape (n, n, n, n)
    '''
    phi = np.asarray(states, dtype='c16')
    error_handling.orthonormal_states(phi)
    v = V.toarray() if hasattr(V, 'toarray') else np.asarray(V)
    error_handling.density_interaction(v, len(phi))
    v = v.astype('f8')
    n = phi.shape[1]
    pair = (phi.conj()[:, :, None] * phi[:, None, :]).reshape(len(phi), n * n)  # (r, (a, c))
    w = (pair.T @ v @ pair).reshape(n, n, n, n)  # (a, c, b, d)
    return w.transpose(0, 2, 1, 3)


class FockSector():
    r'''
    Basis of the many-body states with :math:`N_\uparrow` spin-up and
    :math:`N_\downarrow` spin-down fermions on *n_sites* sites (or
    :math:`N_\uparrow` spinless ones, with ``n_dn=None``).

    The state of index ``a * len(dn) + b`` has spin-up pattern ``up[a]``
    and spin-down pattern ``dn[b]``: a wave function reshaped to
    ``(len(up), len(dn))`` has the spin-up configurations along its rows.

    Parameters
    ----------
    n_sites : Positive integer, at most 62
    n_up : Integer from 0 to *n_sites*
        Spin-up fermions (or spinless ones).
    n_dn : Integer from 0 to *n_sites*, or None, default None
        Spin-down fermions; None for spinless fermions.
    momentum : Tuple (labels, n_cells, total), or None, default None
        Spinless fermions only: keep the states of one total crystal
        momentum. *labels* (integers, shape (n_sites, d)) is the momentum
        :math:`(m_1, \dots)` of each orbital, in units of
        :math:`\mathbf{b}_\alpha/N_\alpha`, *n_cells* the
        :math:`(N_1, \dots)` of the torus and *total* the sector
        :math:`\sum m \bmod N`. The orbitals are then Bloch states (see
        *project_interaction*), and the Hamiltonian must conserve momentum.

    Attributes
    ----------
    up : Integer ndarray
        Spin-up (or spinless) occupation patterns, bit *i* for site *i*,
        in increasing order.
    dn : Integer ndarray
        Spin-down patterns (the single empty pattern ``[0]`` when spinless).
    dim : Integer
        Number of basis states.
    spinless : Boolean
    '''

    def __init__(self, n_sites: int, n_up: int, n_dn: int | None = None, momentum: tuple | None = None) -> None:
        if isinstance(n_up, np.integer):
            n_up = int(n_up)
        if isinstance(n_dn, np.integer):
            n_dn = int(n_dn)
        error_handling.fock_sector(n_sites, n_up, n_dn)
        self.n_sites, self.n_up = n_sites, n_up
        self.n_dn: int | None = n_dn
        self.spinless = n_dn is None
        self.up = _patterns(n_sites, n_up)
        self.dn = _patterns(n_sites, 0 if n_dn is None else n_dn)
        self.momentum = None
        if momentum is not None:
            error_handling.momentum_sector(momentum, n_sites, self.spinless)
            labels = np.asarray(momentum[0], dtype=np.int64).reshape(n_sites, -1)
            cells = np.broadcast_to(np.asarray(momentum[1], dtype=np.int64), labels.shape[1:])
            total = np.broadcast_to(np.asarray(momentum[2], dtype=np.int64), labels.shape[1:])
            self.momentum = (labels, cells, total % cells)
            k = (_occupations(self.up, n_sites) @ labels).astype(np.int64) % cells
            self.up = self.up[np.all(k == total % cells, axis=1)]
        self.dim = len(self.up) * len(self.dn)

    @property
    def sz(self) -> float:
        r'''
        :math:`S_z = (N_\uparrow - N_\downarrow)/2` (0 when spinless).
        '''
        return 0. if self.n_dn is None else (self.n_up - self.n_dn) / 2

    def occupations(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        '''
        Occupation numbers of the spin-up and spin-down patterns.

        Returns
        -------
        occ_up, occ_dn
            Real ndarrays, shapes (len(up), n_sites) and (len(dn), n_sites).
        '''
        return _occupations(self.up, self.n_sites), _occupations(self.dn, self.n_sites)

    def hamiltonian(self, ham, U: float | ArrayLike = 0., V=None, W: ArrayLike | None = None) -> sp.csr_matrix:
        r'''
        Get the sparse Hamiltonian of the sector.

        Parameters
        ----------
        ham
            Hermitian (N, N) matrix, sparse or dense: the hoppings
            :math:`t_{ij}` and onsite energies :math:`t_{ii}`, the same
            for both spins.
        U : Real number, or array of one per site, default 0
            Onsite repulsion :math:`U_i n_{i\uparrow}n_{i\downarrow}`
            (must be 0 for spinless fermions).
        V : Real symmetric (N, N) matrix with zero diagonal, or None, default None
            Density-density interaction :math:`\sum_{i<j}V_{ij}n_in_j`
            (e.g. a constant on the nearest-neighbour bonds).
        W : Complex array, shape (N, N, N, N), or None, default None
            Spinless fermions only: a general two-body interaction
            :math:`\tfrac12\sum_{abcd}W_{abcd}c^\dagger_ac^\dagger_bc_dc_c`,
            Hermitian (:math:`W_{abcd} = W^*_{cdab}`), e.g. from
            *project_interaction*.

        Returns
        -------
        ham : CSR matrix
            Of shape (dim, dim). Real when *ham* and *W* are real, else complex.
        '''
        h, u, v = self._interactions(ham, U, V)
        occ_up, occ_dn = self.occupations()
        h_up = _hopping_matrix(self.up, h, occ_up)
        diag_up = 0.5 * np.sum((occ_up @ v) * occ_up, axis=1)
        if self.spinless:
            mat = h_up + sp.diags(diag_up)
            if W is not None:
                w = np.asarray(W)
                error_handling.two_body(w, self.n_sites)
                if self.momentum is not None:
                    error_handling.conserves_momentum(h, w, *self.momentum)
                mat = mat + _two_body_matrix(self.up, w.real if np.allclose(w.imag, 0.) else w)
            return mat.tocsr()
        error_handling.spinless_two_body(W)
        h_dn = _hopping_matrix(self.dn, h, occ_dn)
        diag_dn = 0.5 * np.sum((occ_dn @ v) * occ_dn, axis=1)
        diag = (diag_up[:, None] + diag_dn[None, :] + (occ_up * u) @ occ_dn.T + occ_up @ v @ occ_dn.T)
        n_up, n_dn = len(self.up), len(self.dn)
        return (sp.kron(h_up, sp.identity(n_dn), format='csr') + sp.kron(sp.identity(n_up), h_dn, format='csr')
                + sp.diags(diag.ravel())).tocsr()

    def _interactions(self, ham, U, V):
        '''
        Private method. Check and densify the hoppings and interactions.
        '''
        h = ham.toarray() if hasattr(ham, 'toarray') else np.asarray(ham)
        error_handling.square_matrix(h, 'ham')
        error_handling.ndarray(h, 'ham', self.n_sites)
        error_handling.hermitian_dense(h)
        if np.allclose(h.imag, 0.):
            h = h.real.astype('f8')
        if self.momentum is not None:
            error_handling.conserves_momentum(h, None, *self.momentum)
        u = np.broadcast_to(np.asarray(U, dtype='f8'), (self.n_sites,)).copy() if np.ndim(U) == 0 \
            else np.asarray(U, dtype='f8')
        error_handling.ndarray(u, 'U', self.n_sites)
        error_handling.spinless_onsite(u, self.spinless)
        if V is None:
            v = np.zeros((self.n_sites, self.n_sites))
        else:
            v = V.toarray() if hasattr(V, 'toarray') else np.asarray(V)
        error_handling.density_interaction(v, self.n_sites)
        return h, u, v.astype('f8')


class ExactResult():
    r'''
    Lowest eigenstates of a sector, from *exact_diagonalization*.

    Attributes
    ----------
    energies : Real ndarray
        The lowest energies, in increasing order.
    states : ndarray
        The eigenvectors (columns, shape (dim, n_states)), in the basis of
        *sector*.
    sector : :class:`FockSector`
    '''

    def __init__(self, energies, states, sector) -> None:
        self.energies, self.states, self.sector = energies, states, sector
        self._occ = sector.occupations()

    def _psi(self, state):
        '''
        Private method. Wave function *state*, shape (len(up), len(dn)).
        '''
        error_handling.state_index(state, len(self.energies))
        return self.states[:, state].reshape(len(self.sector.up), len(self.sector.dn))

    def densities(self, state: int = 0):
        r'''
        Get the densities :math:`\langle n_{i\uparrow}\rangle` and
        :math:`\langle n_{i\downarrow}\rangle` of an eigenstate.

        Parameters
        ----------
        state : Integer, default 0
            Index of the eigenstate (0 the lowest).

        Returns
        -------
        n_up, n_dn : Real ndarrays
            The densities on every site (*n_dn* all zero when spinless).
        '''
        p = np.abs(self._psi(state)) ** 2
        return p.sum(axis=1) @ self._occ[0], p.sum(axis=0) @ self._occ[1]

    def density_correlation(self, state: int = 0) -> NDArray[np.float64]:
        r'''
        Get :math:`\langle n_in_j\rangle` (:math:`n_i` summed over spin).

        Parameters
        ----------
        state : Integer, default 0

        Returns
        -------
        corr : Real ndarray, shape (N, N)
        '''
        p = np.abs(self._psi(state)) ** 2
        occ_up, occ_dn = self._occ
        uu = occ_up.T @ (p.sum(axis=1)[:, None] * occ_up)
        dd = occ_dn.T @ (p.sum(axis=0)[:, None] * occ_dn)
        ud = occ_up.T @ p @ occ_dn
        return uu + dd + ud + ud.T

    def double_occupancy(self, state: int = 0) -> NDArray[np.float64]:
        r'''
        Get :math:`\langle n_{i\uparrow}n_{i\downarrow}\rangle` on every site.
        The local moment is
        :math:`\langle (n_{i\uparrow} - n_{i\downarrow})^2\rangle = \langle n_i\rangle - 2\langle n_{i\uparrow}n_{i\downarrow}\rangle`.

        Parameters
        ----------
        state : Integer, default 0

        Returns
        -------
        d : Real ndarray
        '''
        error_handling.spinful_sector(self.sector.spinless)
        p = np.abs(self._psi(state)) ** 2
        return np.sum((p @ self._occ[1]) * self._occ[0], axis=0)

    def spin_correlation(self, state: int = 0) -> NDArray[np.float64]:
        r'''
        Get :math:`\langle\mathbf{S}_i\cdot\mathbf{S}_j\rangle`. The sum of
        all its elements is :math:`S(S+1)`; its diagonal is
        :math:`\tfrac34\langle(n_{i\uparrow} - n_{i\downarrow})^2\rangle`.

        Parameters
        ----------
        state : Integer, default 0

        Returns
        -------
        corr : Real ndarray, shape (N, N)
        '''
        error_handling.spinful_sector(self.sector.spinless)
        psi = self._psi(state)
        p = np.abs(psi) ** 2
        occ_up, occ_dn = self._occ
        uu = occ_up.T @ (p.sum(axis=1)[:, None] * occ_up)
        dd = occ_dn.T @ (p.sum(axis=0)[:, None] * occ_dn)
        ud = occ_up.T @ p @ occ_dn
        corr = (uu + dd - ud - ud.T) / 4
        n = self.sector.n_sites
        for i in range(n):
            for j in range(i + 1, n):
                # S+_i S-_j = -(c+_{i up} c_{j up})(c+_{j dn} c_{i dn})
                ru, cu, su = _hop(self.sector.up, i, j)
                rd, cd, sd = _hop(self.sector.dn, j, i)
                if not (len(ru) and len(rd)):
                    continue
                moved = (su[:, None] * sd[None, :]) * psi[np.ix_(cu, cd)]
                x = -np.vdot(psi[np.ix_(ru, rd)], moved)
                # <S_i.S_j> gets (<S+_i S-_j> + <S-_i S+_j>)/2 = Re <S+_i S-_j>
                corr[i, j] += x.real
                corr[j, i] += x.real
        occ = self.densities(state)
        corr[np.diag_indices(n)] = 0.75 * (occ[0] + occ[1] - 2 * self.double_occupancy(state))
        return corr

    def total_spin(self, state: int = 0) -> float:
        r'''
        Get the total spin *S*, from
        :math:`S(S+1) = \|S^+\psi\|^2 + S_z(S_z+1)`. Within a degenerate
        multiplet the eigensolver may return a mixture of spins, whose *S*
        is then not a half-integer.

        Parameters
        ----------
        state : Integer, default 0

        Returns
        -------
        S : Real number
        '''
        error_handling.spinful_sector(self.sector.spinless)
        psi = self._psi(state)
        sec = self.sector
        sz = sec.sz
        norm = 0.
        if sec.n_dn and sec.n_up < sec.n_sites:
            up2, dn2 = _patterns(sec.n_sites, sec.n_up + 1), _patterns(sec.n_sites, sec.n_dn - 1)
            target = np.zeros((len(up2), len(dn2)), dtype=psi.dtype)
            for i in range(sec.n_sites):
                cu = np.flatnonzero(((sec.up >> i) & 1) == 0)
                cd = np.flatnonzero((sec.dn >> i) & 1)
                below = (1 << i) - 1
                su = 1 - 2 * (_popcount(sec.up[cu] & below) & 1)
                sd = 1 - 2 * (_popcount(sec.dn[cd] & below) & 1)
                ru = np.searchsorted(up2, sec.up[cu] | (1 << i))
                rd = np.searchsorted(dn2, sec.dn[cd] ^ (1 << i))
                target[np.ix_(ru, rd)] += (su[:, None] * sd[None, :]) * psi[np.ix_(cu, cd)]
            norm = float(np.sum(np.abs(target) ** 2))
        s_s1 = norm + sz * (sz + 1)
        return float((np.sqrt(1 + 4 * s_s1) - 1) / 2)


def exact_diagonalization(
    ham, n_up: int, n_dn: int | None = None, U: float | ArrayLike = 0., V=None, n_states: int = 1,
    W: ArrayLike | None = None, momentum: tuple | None = None,
) -> ExactResult:
    r'''
    Get the lowest eigenstates of the (extended) Hubbard model of a small
    cluster, exactly, in the sector of :math:`N_\uparrow` and
    :math:`N_\downarrow` electrons (see :mod:`tbkit.manybody`).

    The spin-up and spin-down hoppings are those of *ham*, so this is the
    exact counterpart of :func:`tbkit.meanfield.hubbard_mean_field` on the
    same matrix and *U*. Sectors of up to :data:`DENSE_MAX` states are
    diagonalized densely; larger ones by ``eigsh`` (Lanczos), from a fixed
    starting vector so that results are reproducible.

    Parameters
    ----------
    ham
        Hermitian (N, N) matrix, sparse or dense, e.g. *System.ham*.
    n_up : Integer from 0 to N
        Spin-up electrons (or spinless fermions, with ``n_dn=None``).
    n_dn : Integer from 0 to N, or None, default None
        Spin-down electrons; None for spinless fermions.
    U : Real number, or array of one per site, default 0
        Onsite repulsion (0 for spinless fermions).
    V : Real symmetric (N, N) matrix with zero diagonal, or None, default None
        Density-density interaction :math:`\sum_{i<j}V_{ij}n_in_j`.
    n_states : Positive integer, default 1
        Number of lowest states, at most the dimension of the sector.
    W : Complex array, shape (N, N, N, N), or None, default None
        Spinless fermions only: a general two-body interaction (see
        *FockSector.hamiltonian*), e.g. from *project_interaction*.
    momentum : Tuple (labels, n_cells, total), or None, default None
        Spinless fermions only: the sector of one total crystal momentum
        (see :class:`FockSector`).

    Returns
    -------
    result : :class:`ExactResult`

    Examples
    --------
    The Hubbard dimer at half filling::

        res = exact_diagonalization([[0., -1.], [-1., 0.]], 1, 1, U=4.)
        res.energies[0]             # (U - sqrt(U^2 + 16 t^2)) / 2 = -0.828
        res.total_spin()            # 0, a singlet
    '''
    h = ham.toarray() if hasattr(ham, 'toarray') else np.asarray(ham)
    error_handling.square_matrix(h, 'ham')
    sector = FockSector(len(h), n_up, n_dn, momentum)
    error_handling.positive_int(n_states, 'n_states')
    error_handling.n_states(n_states, sector.dim)
    mat = sector.hamiltonian(h, U, V, W)
    if sector.dim <= DENSE_MAX or n_states >= sector.dim - 1:
        energies, states = LA.eigh(mat.toarray(), subset_by_index=(0, n_states - 1))
    else:
        v0 = np.random.default_rng(0).standard_normal(sector.dim).astype(mat.dtype)
        energies, states = sla.eigsh(mat, k=n_states, which='SA', v0=v0, tol=1e-12)
        order = np.argsort(energies)
        energies, states = energies[order], states[:, order]
    return ExactResult(energies, states, sector)
