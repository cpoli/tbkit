r"""
Real-space Tight-Binding models with several orbitals per site and an
optional spin: :class:`OrbitalSystem` extends :class:`tbkit.system.System`
with Slater-Koster hoppings between s, p and d orbitals (see
:mod:`tbkit.slater_koster`), non-orthogonal bases (an overlap matrix),
and spin-dependent terms -- Kane-Mele intrinsic spin-orbit coupling,
Rashba coupling, Zeeman fields and atomic :math:`\lambda\mathbf{L}\cdot\mathbf{S}`.

Rows of the Hamiltonian are ordered site by site, then orbital by orbital
(in the order given for the site's tag), then spin (up, down): the row of
orbital *o* with spin *s* on site *i* is *sys.row(i, o, s)*.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.typing import NDArray
import scipy.linalg as LA
import scipy.sparse as sparse

import tbkit.error_handling as error_handling
from tbkit.lattice import Lattice
from tbkit.slater_koster import ORBITALS, sk_block, orbital_angular_momentum
from tbkit.system import System, PI

SIGMA = [np.array([[0., 1.], [1., 0.]], dtype='c16'),
              np.array([[0., -1j], [1j, 0.]], dtype='c16'),
              np.array([[1., 0.], [0., -1.]], dtype='c16')]


class OrbitalSystem(System):
    r'''
    Solve a real-space Tight-Binding model with several orbitals per site
    and, optionally, spin.

    :param lat: **Lattice** class instance.
    :param orbitals: Dictionary. Default value None (one 's' orbital per
        site). key: sublattice tag, val: list of orbital names (see
        :data:`tbkit.slater_koster.ORBITALS`), e.g. ``{'a': ['s', 'px', 'py', 'pz']}``.
    :param spin: Boolean. Default value False. Give every orbital a
        spin-1/2 degree of freedom.

    Everything of **System** still applies: *set_hopping* (and the disorder,
    defect and Peierls methods) sets hoppings :math:`t` that connect equal
    orbitals and equal spins of the two sites (:math:`t\,\mathbb{1}`), and
    *set_onsite* with plain numbers shifts every orbital of a site. Onsite
    energies can also be given orbital by orbital, and full hopping blocks
    come from *set_slater_koster*, *set_spin_orbit*, *set_rashba* or
    *set_hopping_block*. Call *set_peierls_phase* / *set_magnetic_field*
    last: they multiply every hopping (and block) set so far.

    Example usage::

        # graphene's sp3 bands: four orbitals per carbon
        sys = OrbitalSystem(lat, orbitals={'a': ['s', 'px', 'py', 'pz'],
                                                        'b': ['s', 'px', 'py', 'pz']})
        sys.set_onsite({'a': {'s': -8.87}, 'b': {'s': -8.87}})
        sys.set_slater_koster(1, {'ss_sigma': -6.77, 'sp_sigma': 5.58,
                                             'pp_sigma': 5.04, 'pp_pi': -3.03})
        sys.get_ham()
    '''

    def __init__(self, lat: Lattice, orbitals: dict[str, list[str]] | None = None,
                 spin: bool = False) -> None:
        System.__init__(self, lat)
        error_handling.boolean(spin, 'spin')
        if orbitals is None:
            orbitals = {tag: ['s'] for tag in lat.tags}
        error_handling.orbital_dict(orbitals, lat.tags)
        self.orbitals = {tag: list(orbs) for tag, orbs in orbitals.items()}
        self.spin = spin
        self.ns = 2 if spin else 1
        self.onsite_orb = {}  # tag -> {orbital: energy}
        self.blocks = {}  # name -> list of (i, j, matrix): bond blocks, upper part
        self.overlap_blocks = {}  # name -> list of (i, j, matrix)
        self.onsite_blocks = {}  # name -> {site: matrix}
        self.overlap = None
        self._index()

    # ------------------------------------------------------------------
    # indexing
    # ------------------------------------------------------------------

    def _index(self) -> None:
        '''
        Private method. Rows of every site, orbital and spin.
        '''
        tags = self.lat.coor['tag']
        self.norb_site = np.array([len(self.orbitals[t]) * self.ns for t in tags], dtype=int)
        self.offset = np.concatenate([[0], np.cumsum(self.norb_site)[:-1]]).astype(int)
        self.n_rows = int(self.norb_site.sum())
        self.orb_site = np.repeat(np.arange(self.lat.sites), self.norb_site)

    def row(self, site: int, orbital: str, spin: int = 0) -> int:
        '''
        Get the row of an orbital.

        :param site: Integer. Site index.
        :param orbital: String. Orbital name (one of the site's).
        :param spin: 0 (up) or 1 (down). Default value 0.

        :returns:
            * **row** -- Integer.
        '''
        self._index()
        error_handling.site_index(site, self.lat.sites)
        orbs = self.orbitals[self.lat.coor['tag'][site]]
        error_handling.orbital_of_site(orbital, orbs)
        error_handling.spin_index(spin, self.ns)
        return int(self.offset[site] + orbs.index(orbital) * self.ns + spin)

    def _site_orbs(self, site: int) -> list[str]:
        return self.orbitals[self.lat.coor['tag'][site]]

    def _kron_spin(self, mat: NDArray) -> NDArray:
        '''
        Private method. Orbital matrix times the spin identity.
        '''
        if not self.spin:
            return np.asarray(mat, dtype='c16')
        return np.kron(mat, np.eye(2))

    def _same_orbitals(self, i: int, j: int) -> NDArray:
        '''
        Private method. Matrix connecting equal orbitals of sites i and j.
        '''
        oi, oj = self._site_orbs(i), self._site_orbs(j)
        return np.array([[1. if a == b else 0. for b in oj] for a in oi])

    # ------------------------------------------------------------------
    # onsite terms
    # ------------------------------------------------------------------

    def set_onsite(self, dict_onsite: dict) -> None:
        '''
        Set onsite energies.

        :param dict_onsite: Dictionary. key: sublattice tag, val: onsite
            energy (a number, applied to every orbital of the sites), or a
            dictionary {orbital: energy} (orbitals not given get 0).

        Example usage::

            sys.set_onsite({'a': 0.5, 'b': {'s': -8.87, 'pz': 0.}})
        '''
        error_handling.sites(self.lat.sites)
        error_handling.set_onsite_orb(dict_onsite, self.lat.tags, self.orbitals)
        numbers = {tag: val for tag, val in dict_onsite.items() if not isinstance(val, dict)}
        System.set_onsite(self, numbers)
        self.onsite_orb = {tag: dict(val) for tag, val in dict_onsite.items() if isinstance(val, dict)}

    def set_zeeman(self, b: tuple[float, float, float]) -> None:
        r'''
        Set a Zeeman field :math:`\mathbf{b}\cdot\boldsymbol\sigma` on every
        orbital (``spin=True`` only).

        :param b: Tuple of three real numbers :math:`(b_x, b_y, b_z)`.
        '''
        error_handling.spinful(self.spin)
        error_handling.vector3(b, 'b')
        field = sum(c * s for c, s in zip(b, SIGMA))
        self.onsite_blocks['zeeman'] = {
            i: np.kron(np.eye(len(self._site_orbs(i))), field) for i in range(self.lat.sites)}

    def set_atomic_soc(self, lam: float, tags: list[str] | None = None) -> None:
        r'''
        Set the atomic spin-orbit coupling
        :math:`\lambda\,\mathbf{L}\cdot\mathbf{S} = \frac{\lambda}{2}\sum_k L_k\sigma_k`
        on the p orbitals (``spin=True`` only; the sites must carry all three
        of px, py, pz). It splits them into :math:`j = 3/2` (at
        :math:`+\lambda/2`) and :math:`j = 1/2` (at :math:`-\lambda`) levels.

        :param lam: Real number. :math:`\lambda`.
        :param tags: List of tags. Default value None (every tag with p orbitals).
        '''
        error_handling.spinful(self.spin)
        error_handling.real_number(lam, 'lam')
        if tags is None:
            tags = [t for t, orbs in self.orbitals.items() if 'px' in orbs]
        error_handling.p_shell(tags, self.orbitals)
        L = orbital_angular_momentum()
        blocks = {}
        for i in range(self.lat.sites):
            orbs = self._site_orbs(i)
            if self.lat.coor['tag'][i] not in tags:
                continue
            mat = np.zeros((2 * len(orbs), 2 * len(orbs)), 'c16')
            idx = [orbs.index(o) for o in ('px', 'py', 'pz')]
            for k in range(3):
                emb = np.zeros((len(orbs), len(orbs)), 'c16')
                emb[np.ix_(idx, idx)] = L[k]
                mat += 0.5 * lam * np.kron(emb, SIGMA[k])
            blocks[i] = mat
        self.onsite_blocks['atomic_soc'] = blocks

    # ------------------------------------------------------------------
    # bond blocks
    # ------------------------------------------------------------------

    def _bonds(self, n: int) -> tuple[NDArray, NDArray, NDArray]:
        '''
        Private method. Bonds of order n (oriented, see *get_bonds*) and their vectors.
        '''
        self.get_distances(n)
        self.nmax = len(self.dist_uni) - 1
        error_handling.positive_int_lim(n, 'n', self.nmax)
        i, j, _ = self.get_bonds(n)
        coords = self._coords()
        return i, j, coords[j] - coords[i]

    def set_slater_koster(self, n: int, params: dict, overlap: dict | None = None) -> None:
        r'''
        Set the hoppings between the orbitals of every pair of *n*-th
        neighbours from Slater-Koster bond integrals (see
        :func:`tbkit.slater_koster.sk_block`), replacing those set before for
        the same *n*. The blocks are spin independent.

        :param n: Positive integer. Neighbour order (1 for nearest neighbours).
        :param params: Dictionary of bond integrals (e.g. ``{'pp_sigma': 5.04,
            'pp_pi': -3.03}``), or dictionary {tag pair: bond integrals} (e.g.
            ``{'ab': {...}, 'ba': {...}}``, for bonds from an 'a' to a 'b' site).
        :param overlap: Dictionary. Default value None. Bond integrals of the
            overlap matrix :math:`S_{ij} = \langle i|j\rangle`, in the same
            format: the basis is then non-orthogonal, and *get_eig* solves
            :math:`H\psi = ES\psi`.
        '''
        error_handling.positive_int(n, 'n')
        i, j, d = self._bonds(n)
        tags = self.lat.coor['tag']
        for store, par in ((self.blocks, params), (self.overlap_blocks, overlap)):
            if par is None:
                store.pop(('sk', n), None)
                continue
            by_pair = error_handling.sk_pair_params(par, tags[i], tags[j])
            blocks = []
            for a, b, vec in zip(i, j, d):
                p = par[tags[a] + tags[b]] if by_pair else par
                blocks.append((a, b, self._kron_spin(sk_block(self._site_orbs(a), self._site_orbs(b), vec, p))))
            store[('sk', n)] = blocks

    def set_spin_orbit(self, lam: float, n: int = 2) -> None:
        r'''
        Set the Kane-Mele intrinsic spin-orbit coupling on the *n*-th
        neighbours (``spin=True`` only),

        .. math::

            H_{SO} = i\lambda\sum_{\langle\langle ij\rangle\rangle}\nu_{ij}\,
                     c_i^\dagger\sigma_z c_j\, ,\qquad
            \nu_{ij} = \sum_k \mathrm{sign}\left[(\mathbf{d}_{ik}\times\mathbf{d}_{kj})_z\right]\, ,

        the sum running over the nearest neighbours :math:`k` shared by
        :math:`i` and :math:`j` (one on the honeycomb lattice, where
        :math:`\nu_{ij} = \pm1` says whether the path turns left or right).
        It connects equal orbitals of the two sites.

        :param lam: Real number. :math:`\lambda`.
        :param n: Positive integer. Default value 2. Neighbour order.
        '''
        error_handling.spinful(self.spin)
        error_handling.real_number(lam, 'lam')
        error_handling.positive_int(n, 'n')
        ni, nj, _ = self._bonds(1)
        neighbours = [set() for _ in range(self.lat.sites)]
        for a, b in zip(ni, nj):
            neighbours[a].add(b)
            neighbours[b].add(a)
        i, j, _ = self._bonds(n)
        coords = self._coords()
        blocks = []
        for a, b in zip(i, j):
            nu = 0.
            for k in neighbours[a] & neighbours[b]:
                d1, d2 = coords[k] - coords[a], coords[b] - coords[k]
                nu += np.sign(d1[0] * d2[1] - d1[1] * d2[0])
            if nu:
                blocks.append((a, b, 1j * lam * nu * np.kron(self._same_orbitals(a, b), SIGMA[2])))
        self.blocks[('soc', n)] = blocks

    def set_rashba(self, lam: float, n: int = 1) -> None:
        r'''
        Set a Rashba spin-orbit coupling on the *n*-th neighbours
        (``spin=True`` only),
        :math:`H_R = i\lambda\sum_{\langle ij\rangle} c_i^\dagger
        (\boldsymbol\sigma\times\hat{\mathbf{d}}_{ij})_z\, c_j`,
        :math:`\hat{\mathbf{d}}_{ij}` the bond direction. It connects equal
        orbitals of the two sites.

        :param lam: Real number. :math:`\lambda`.
        :param n: Positive integer. Default value 1.
        '''
        error_handling.spinful(self.spin)
        error_handling.real_number(lam, 'lam')
        error_handling.positive_int(n, 'n')
        i, j, d = self._bonds(n)
        blocks = []
        for a, b, vec in zip(i, j, d):
            u = vec / np.linalg.norm(vec)
            spin = 1j * lam * (SIGMA[0] * u[1] - SIGMA[1] * u[0])
            blocks.append((a, b, np.kron(self._same_orbitals(a, b), spin)))
        self.blocks[('rashba', n)] = blocks

    def set_hopping_block(self, dict_block: dict[tuple[int, int], NDArray]) -> None:
        r'''
        Set hopping blocks by hand, added to the others: key: site pair
        :math:`(i, j)`, val: matrix of shape (rows of i, rows of j) --
        orbitals, then spin, as *row* orders them. Their Hermitian conjugates
        are added.

        :param dict_block: Dictionary.
        '''
        error_handling.set_hopping_def(None, {k: 0. for k in dict_block}, self.lat.sites, 'dict_block')
        self._index()
        blocks = self.blocks.setdefault(('manual', 0), [])
        for (a, b), mat in dict_block.items():
            mat = np.asarray(mat, dtype='c16')
            error_handling.block_shape(mat, (self.norb_site[a], self.norb_site[b]))
            blocks.append((a, b, mat))

    def clear_hopping(self) -> None:
        '''
        Clear the hoppings: *hop*, and every hopping block.
        '''
        System.clear_hopping(self)
        self.blocks = {}
        self.overlap_blocks = {}

    def set_peierls_phase(self, phase: Callable[..., NDArray]) -> None:
        r'''
        Apply the Peierls substitution (see *System.set_peierls_phase*) to
        the hoppings *hop* and to every hopping block set so far.

        :param phase: Callable ``phase(xi, yi, xj, yj)``.
        '''
        error_handling.is_callable(phase, 'phase')
        if self.hop.size:
            System.set_peierls_phase(self, phase)
        x, y = self.lat.coor['x'], self.lat.coor['y']
        for key, blocks in self.blocks.items():
            self.blocks[key] = [(a, b, mat * np.exp(1j * phase(x[a], y[a], x[b], y[b])))
                                          for a, b, mat in blocks]

    # ------------------------------------------------------------------
    # Hamiltonian and eigenstates
    # ------------------------------------------------------------------

    def _assemble(self, blocks: list, n: int) -> sparse.csr_matrix:
        '''
        Private method. Sparse matrix of bond blocks, upper part only.
        '''
        rows, cols, vals = [], [], []
        for a, b, mat in blocks:
            r = self.offset[a] + np.arange(mat.shape[0])
            c = self.offset[b] + np.arange(mat.shape[1])
            rr, cc = np.meshgrid(r, c, indexing='ij')
            rows.append(rr.ravel())
            cols.append(cc.ravel())
            vals.append(mat.ravel())
        if not rows:
            return sparse.csr_matrix((n, n), dtype='c16')
        return sparse.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                                          shape=(n, n))

    def get_ham(self) -> None:
        '''
        Get the Hamiltonian (and, for a non-orthogonal basis, the overlap
        matrix *overlap*).
        '''
        error_handling.orbital_terms(self.hop, self.blocks)
        self._index()
        n = self.n_rows
        ham = sparse.csr_matrix((n, n), dtype='c16')
        if self.hop.size:
            error_handling.hop_sites(self.hop, self.lat.sites)
            rows, cols, vals = [], [], []
            names = sorted({o for orbs in self.orbitals.values() for o in orbs})
            index = {o: np.full(self.lat.sites, -1) for o in names}
            for site in range(self.lat.sites):
                for k, o in enumerate(self._site_orbs(site)):
                    index[o][site] = self.offset[site] + k * self.ns
            hi, hj = self.hop['i'].astype(int), self.hop['j'].astype(int)
            for o in names:
                keep = (index[o][hi] >= 0) & (index[o][hj] >= 0)
                for s in range(self.ns):
                    rows.append(index[o][hi[keep]] + s)
                    cols.append(index[o][hj[keep]] + s)
                    vals.append(self.hop['t'][keep])
            scalar = sparse.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                                                shape=(n, n))
            if np.all(self.hop['ang'] >= 0) or np.all(self.hop['ang'] < 0):
                scalar = scalar + scalar.conj().T
            ham = ham + scalar
        bonds = self._assemble([blk for blocks in self.blocks.values() for blk in blocks], n)
        ham = ham + bonds + bonds.conj().T
        diag = np.zeros(n, 'c16')
        if self.onsite.size == self.lat.sites:
            diag += self.onsite[self.orb_site]
        for tag, energies in self.onsite_orb.items():
            for site in np.flatnonzero(self.lat.coor['tag'] == tag):
                for o, e in energies.items():
                    k = self._site_orbs(site).index(o)
                    diag[self.offset[site] + k * self.ns: self.offset[site] + (k + 1) * self.ns] += e
        ham = ham + sparse.diags(diag)
        for blocks in self.onsite_blocks.values():
            ham = ham + self._assemble([(i, i, mat) for i, mat in blocks.items()], n)
        self.ham = sparse.csr_matrix(ham)
        self.overlap = None
        overlap = [blk for blocks in self.overlap_blocks.values() for blk in blocks]
        if overlap:
            o = self._assemble(overlap, n)
            self.overlap = sparse.csr_matrix(sparse.identity(n, dtype='c16') + o + o.conj().T)

    def _set_states(self) -> None:
        '''
        Private method. Intensities summed over each site's orbitals and spin
        (Mulliken populations for a non-orthogonal basis), and sublattice
        polarizations.
        '''
        if self.overlap is None:
            weight = np.abs(self.rn) ** 2
        else:
            weight = np.real(self.rn.conj() * (self.overlap @ self.rn))
        self.intensity = np.zeros((self.lat.sites, self.rn.shape[1]))
        np.add.at(self.intensity, self.orb_site, weight)
        self.pola = np.zeros((self.rn.shape[1], len(self.lat.tags)))
        for i, tag in enumerate(self.lat.tags):
            self.pola[:, i] = np.sum(self.intensity[self.lat.coor['tag'] == tag, :], axis=0)

    def get_eig(self, eigenvec: bool = False, left: bool = False) -> None:
        r'''
        Get the eigenenergies (and eigenvectors), as *System.get_eig*. With a
        non-orthogonal basis (see *set_slater_koster*), solve the generalized
        problem :math:`H\psi = ES\psi` (eigenvectors normalized as
        :math:`\psi^\dagger S\psi = 1` for a Hermitian :math:`H`).

        :param eigenvec: Boolean. Default value False.
        :param left: Boolean. Default value False (orthogonal basis only).
        '''
        if self.overlap is None:
            return System.get_eig(self, eigenvec, left)
        error_handling.empty_ham(self.ham)
        error_handling.boolean(eigenvec, 'eigenvec')
        error_handling.no_left(left)
        self.ln = np.array([], 'c16')
        ham, s = self.ham.toarray(), self.overlap.toarray()
        if (self.ham.conj().T != self.ham).nnz:
            en, vec = LA.eig(ham, s)
            ind = np.argsort(en.real, kind='stable')
            self.en, self.rn = en[ind], vec[:, ind]
        else:
            self.en, self.rn = LA.eigh(ham, s)
        if eigenvec:
            self._set_states()

    def _row_sites(self) -> NDArray:
        '''
        Private method. Site of every row of the Hamiltonian.
        '''
        self._index()
        return self.orb_site

    def get_eig_sparse(self, n_eig: int = 10, sigma: complex = 0., eigenvec: bool = False) -> None:
        '''
        Get the eigenpairs closest to *sigma* (see *System.get_eig_sparse*;
        orthogonal basis only).
        '''
        error_handling.orthogonal(self.overlap)
        System.get_eig_sparse(self, n_eig, sigma, eigenvec)

    def get_local_chern_marker(self, e_fermi: float = 0., area: float | None = None) -> NDArray[np.float64]:
        '''
        Get the local Chern marker (see *System.get_local_chern_marker*),
        summed over the orbitals and spin of each site (orthogonal basis only).

        :param e_fermi: Real number. Default value 0.
        :param area: Positive real number. Default value None.

        :returns:
            * **marker** -- Real ndarray, shape (sites,).
        '''
        error_handling.orthogonal(self.overlap)
        return System.get_local_chern_marker(self, e_fermi, area)
