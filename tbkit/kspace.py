from __future__ import annotations

from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import scipy.linalg as LA
import scipy.sparse as sp
import tbkit.error_handling as error_handling
import tbkit.dos as dos
import tbkit.occupation as occupation
from tbkit.lattice import Lattice


PI = np.pi

#: Pauli matrices (plus the identity, key ``'0'``), for building spinful
#: hoppings/onsite terms (spin-orbit coupling, Zeeman splitting, ...) when
#: :class:`KSpace` is constructed with ``spin=True``.
PAULI = {
    '0': np.eye(2, dtype='c16'),
    'x': np.array([[0., 1.], [1., 0.]], dtype='c16'),
    'y': np.array([[0., -1j], [1j, 0.]], dtype='c16'),
    'z': np.array([[1., 0.], [0., -1.]], dtype='c16'),
}


#################################
# CLASS KSPACE
#################################


def reciprocal_vectors(prim_vec: list[tuple[float, ...]]) -> list[tuple[float, ...]]:
    r'''
    Get the reciprocal lattice vectors :math:`\mathbf{b}_i` such that
    :math:`\mathbf{a}_i\cdot\mathbf{b}_j = 2\pi\delta_{ij}`, lying in the
    span of the :math:`\mathbf{a}_i`.

    :param prim_vec: List of one/two/three tuples. Primitive vectors (see class **lattice**).

    :returns:
        * **rec_vec** -- List of one/two/three tuples. Reciprocal vectors, with
          as many components as the primitive vectors.
    '''
    if len(prim_vec[0]) == 3:
        # b = 2 pi (A^+)^T: the dual basis within the span of the a_i
        rec = 2 * PI * np.linalg.pinv(np.array(prim_vec, dtype='f8')).T
        return [tuple(float(c) for c in b) for b in rec]
    if len(prim_vec) == 1:
        ax, ay = prim_vec[0]
        norm2 = ax ** 2 + ay ** 2
        return [(2*PI*ax/norm2, 2*PI*ay/norm2)]
    (a1x, a1y), (a2x, a2y) = prim_vec
    area = a1x * a2y - a1y * a2x
    b1 = (2*PI*a2y/area, -2*PI*a2x/area)
    b2 = (-2*PI*a1y/area, 2*PI*a1x/area)
    return [b1, b2]


class KSpace():
    r'''
    Build and solve the Tight-Binding Bloch Hamiltonian :math:`H(\mathbf{k})`
    of a periodic lattice defined by the class **lattice**.

    Hoppings are defined between orbitals of the unit cell, separated by a
    lattice vector :math:`\mathbf{R} = n_1\mathbf{a}_1+n_2\mathbf{a}_2`:

    .. math::

        H_{ij}(\mathbf{k}) = \sum_{\mathbf{R}} t_{ij}(\mathbf{R})\,
        e^{i\mathbf{k}\cdot\mathbf{R}}

    :param lat: **lattice** class instance. Only *unit_cell* and *prim_vec*
        are used (the instance need not call *get_lattice*).
    :param spin: Boolean. Default value False. If True, every site of
        *unit_cell* carries a spin-1/2 degree of freedom (*norb* doubles to
        ``2*len(unit_cell)``, ordered site-major: orbitals ``2*i, 2*i+1``
        are the up/down components of site *i*). *set_onsite* and
        *set_hopping* then accept 2x2 (spin) matrices in addition to plain
        numbers, to build spin-orbit coupling or Zeeman terms -- see
        :data:`PAULI` for ready-made Pauli matrices.

    Example usage::

        # graphene, nearest-neighbor hopping t
        DX, DY = 0.5 * 3 ** 0.5, 0.5
        unit_cell = [{'tag': 'a', 'r0': (0., 0.)}, {'tag': 'b', 'r0': (DX, DY)}]
        prim_vec = [(2*DX, 0.), (DX, 1.5)]
        lat = Lattice(unit_cell=unit_cell, prim_vec=prim_vec)
        gra = KSpace(lat)
        gra.set_hopping([{'i': 0, 'j': 1, 'R': (0, 0), 't': 1.},
                                {'i': 0, 'j': 1, 'R': (-1, 0), 't': 1.},
                                {'i': 0, 'j': 1, 'R': (0, -1), 't': 1.}])
    '''

    def __init__(self, lat: Lattice, spin: bool = False) -> None:
        error_handling.lat(lat)
        error_handling.boolean(spin, 'spin')
        error_handling.independent(lat.prim_vec)
        self.lat = lat
        self.dim = len(lat.prim_vec)
        self.space_dim = len(lat.prim_vec[0])
        self.spin = spin
        self.n_sites = len(lat.unit_cell)
        self.norb = 2*self.n_sites if spin else self.n_sites
        self.tags = np.array([dic['tag'] for dic in lat.unit_cell])
        self.onsite = np.zeros(self.norb, 'c16')
        # spin-off-diagonal onsite terms (in-plane Zeeman, onsite Rashba):
        # they have no place on the diagonal `onsite` array.
        self._onsite_offdiag = np.zeros((self.norb, self.norb), 'c16')
        self._hop = []  # list of (i, j, R_cartesian (np.ndarray), t)
        self._nonreciprocal = False  # set_hopping(hermitian=False) was used
        self._overlap_hop = []  # overlaps, as _hop: list of (i, j, R_cartesian, s)
        self.rec_vec = reciprocal_vectors(lat.prim_vec)
        # Orthonormal basis (columns) of the span of prim_vec, in which k is
        # given: the identity when the lattice fills its space, a1/|a1| for a
        # chain, the Gram-Schmidt basis of (a1, a2) for a 2D lattice in 3D.
        self.k_basis = np.linalg.qr(np.array(lat.prim_vec, dtype='f8').T)[0]
        if self.dim == self.space_dim:
            self.k_basis = np.eye(self.space_dim)
        elif self.dim == 1:
            a1 = np.asarray(lat.prim_vec[0], dtype='f8')
            self.k_basis = (a1 / np.linalg.norm(a1))[:, None]
        else:
            # keep the first axis along a1 (QR may flip signs)
            signs = np.sign(np.diag(np.array(lat.prim_vec, dtype='f8') @ self.k_basis))
            self.k_basis = self.k_basis * np.where(signs == 0, 1., signs)[None, :]
        #: Reciprocal vectors in the k coordinates used by *get_ham* (rows).
        self.rec_vec_k = np.array(self.rec_vec, dtype='f8') @ self.k_basis
        self.ks = np.array([])  # k-points of the last band-structure calculation
        self.ks_dist = np.array([])  # cumulative distance along the k-path
        self.nodes = np.array([])  # positions, along ks_dist, of the k-path nodes
        self.en = np.array([])  # bands, shape (len(ks), norb)

    def set_onsite(self, dict_onsite: dict[str, complex | Sequence[complex]]) -> None:
        '''
        Set the onsite energies, by sublattice tag.

        :param dict_onsite: Dictionary. key: tag, val: onsite energy
            (a plain number), or, if ``spin=True``, a plain number (applied
            equally to both spins), a pair ``(E_up, E_down)`` of numbers (a
            spin splitting along z), or a 2x2 complex matrix (a general spin
            structure, e.g. an in-plane Zeeman field built from :data:`PAULI`).
            A complex onsite energy (gain/loss), or a non-Hermitian 2x2
            block, makes :math:`H(\\mathbf{k})` non-Hermitian: *get_bands*
            then returns complex energies, sorted by their real part.

        Example usage::

            kag.set_onsite({'a': 1., 'b': -1.})
            # spinful: same onsite energy for both spins on 'a', a Zeeman
            # splitting along z on 'b':
            kag_spin.set_onsite({'a': 1., 'b': (1., -1.)})
            # spinful: an in-plane Zeeman field on 'a':
            kag_spin.set_onsite({'a': Bx*PAULI['x']})
        '''
        error_handling.set_onsite_kspace(dict_onsite, self.lat.tags, self.spin)
        for tag, val in dict_onsite.items():
            sites = np.where(self.tags == tag)[0]
            if self.spin:
                if np.ndim(val) == 0:
                    block = val * PAULI['0']
                elif np.ndim(val) == 2:
                    block = np.asarray(val, 'c16')
                else:
                    block = np.diag(np.asarray(val, 'c16'))
                self.onsite[2*sites] = block[0, 0]
                self.onsite[2*sites + 1] = block[1, 1]
                for site in sites:
                    self._onsite_offdiag[2*site, 2*site + 1] = block[0, 1]
                    self._onsite_offdiag[2*site + 1, 2*site] = block[1, 0]
            else:
                self.onsite[sites] = val

    def set_hopping(self, list_hop: list[dict], hermitian: bool = True) -> None:
        r'''
        Set the hoppings between orbitals of the unit cell.

        Only one representative of each hopping needs to be given: its
        Hermitian conjugate (:math:`j\to i`, :math:`\mathbf{R}\to-\mathbf{R}`)
        is added automatically (unless ``hermitian=False``).

        :param list_hop: List of dictionaries with keys ('i', 'j', 'R', 't'):

            * 'i', 'j': Positive integers. Site indices within the unit cell
              (following the order of *unit_cell*).
            * 'R': Tuple of one/two integers :math:`(n_1, n_2)`. Lattice vector
              :math:`\mathbf{R}=n_1\mathbf{a}_1+n_2\mathbf{a}_2` separating the
              two sites.
            * 't': Complex number, or, if ``spin=True``, either a complex
              number (spin-independent hopping) or a 2x2 complex matrix (a
              general, possibly spin-mixing, hopping -- e.g. built from
              :data:`PAULI` for Rashba or intrinsic spin-orbit coupling).
        :param hermitian: Boolean. Default value True. If False, the Hermitian
            conjugates are *not* added: each dictionary is one matrix element
            :math:`H_{ij}(\mathbf{R})` only, so that non-reciprocal
            (non-Hermitian) hoppings can be built -- give the reverse hopping
            explicitly, with its own amplitude.

        Example usage::

            # 1D chain, nearest-neighbor hopping t between the only orbital
            # and its right neighbor:
            chain.set_hopping([{'i': 0, 'j': 0, 'R': (1,), 't': 1.}])
            # spinful: spin-independent hopping t, plus a Rashba-like
            # spin-flip term of strength alpha:
            chain_spin.set_hopping([{'i': 0, 'j': 0, 'R': (1,),
                                                    't': t*PAULI['0'] + 1j*alpha*PAULI['y']}])
        '''
        error_handling.set_hopping_kspace(list_hop, self.n_sites, self.dim, self.spin)
        error_handling.boolean(hermitian, 'hermitian')
        if not hermitian:
            self._nonreciprocal = True
        for dic in list_hop:
            R_cart = np.zeros(self.space_dim)
            for n, a in zip(dic['R'], self.lat.prim_vec):
                R_cart += n * np.array(a)
            i, j, t = dic['i'], dic['j'], dic['t']
            if self.spin:
                block = t*PAULI['0'] if np.ndim(t) == 0 else np.asarray(t, 'c16')
                for a in range(2):
                    for b in range(2):
                        self._hop.append((2*i+a, 2*j+b, R_cart, block[a, b]))
                        if hermitian:
                            self._hop.append((2*j+b, 2*i+a, -R_cart, np.conj(block[a, b])))
            else:
                self._hop.append((i, j, R_cart, t))
                if hermitian:
                    self._hop.append((j, i, -R_cart, np.conj(t)))

    def set_overlap(self, list_hop: list[dict]) -> None:
        r'''
        Set the overlaps :math:`s_{ij}(\mathbf{R}) = \langle i,\mathbf{0}|j,\mathbf{R}\rangle`
        of a non-orthogonal basis, in the format of *set_hopping* (key 't'
        for the overlap; the Hermitian conjugates are added):

        .. math::

            S(\mathbf{k}) = \mathbb{1} + \sum_{\mathbf{R}} s(\mathbf{R})\,e^{i\mathbf{k}\cdot\mathbf{R}}\, ,

        after which the bands solve :math:`H(\mathbf{k})v = E\,S(\mathbf{k})v`.
        (With graphene's nearest-neighbour overlap :math:`s`, the
        :math:`\pi` bands become :math:`E = (\epsilon \pm t|f|)/(1 \pm s|f|)`:
        no longer symmetric about :math:`\epsilon`.) The topological tools use
        the Lowdin-orthonormalized states :math:`S^{1/2}v`.

        :param list_hop: List of dictionaries ('i', 'j', 'R', 't'), see *set_hopping*.
        '''
        error_handling.set_hopping_kspace(list_hop, self.n_sites, self.dim, self.spin)
        saved, self._hop = self._hop, []
        nonrec = self._nonreciprocal
        self.set_hopping(list_hop)
        self._overlap_hop += self._hop
        self._hop, self._nonreciprocal = saved, nonrec

    def get_overlap(self, k: ArrayLike) -> NDArray[np.complex128]:
        r'''
        Get the overlap matrix :math:`S(\mathbf{k})` (the identity for an
        orthogonal basis), see *set_overlap*.

        :param k: k point (see *get_ham*).

        :returns:
            * **S** -- Complex ndarray, shape (norb, norb).
        '''
        error_handling.k_vector(k, 'k', self.dim)
        k_cart = self.k_basis @ np.asarray(k, dtype='f8')
        return np.eye(self.norb) + self._bloch_sum(self._overlap_hop, k_cart[None])[0][0]

    def clear_hopping(self) -> None:
        '''
        Clear the hoppings set by *set_hopping* (and the overlaps of *set_overlap*).
        '''
        self._hop = []
        self._nonreciprocal = False
        self._overlap_hop = []

    def get_ham(self, k: ArrayLike) -> NDArray[np.complex128]:
        r'''
        Get the dense Bloch Hamiltonian :math:`H(\mathbf{k})`.

        :param k: Tuple/list/ndarray of *dim* real numbers. When the lattice
            fills its space (2D in the plane, 3D in space), the
            :math:`\mathbf{k}` point in the same Cartesian frame as *prim_vec*.
            In 1D, the crystal momentum *along* the primitive vector (so the
            Brillouin zone spans :math:`2\pi/|\mathbf{a}_1|`), which
            coincides with :math:`k_x` for a chain aligned with :math:`x`.
            For a 2D lattice in 3D space (a slab), the components along the
            orthonormal basis *k_basis* of its plane (the first along
            :math:`\mathbf{a}_1`). *rec_vec_k* holds the reciprocal vectors in
            these coordinates.

        :returns:
            * **ham** -- Complex ndarray, shape (norb, norb).
        '''
        error_handling.k_vector(k, 'k', self.dim)
        return self._bloch_ham((self.k_basis @ np.asarray(k, dtype='f8'))[None])[0]

    def get_ham_peierls(self, k: ArrayLike, A: ArrayLike) -> NDArray[np.complex128]:
        r'''
        Get the Bloch Hamiltonian in a uniform vector potential
        :math:`\mathbf{A}` (:math:`\hbar = e = 1`): every hopping picks up the
        Peierls phase :math:`e^{i\mathbf{A}\cdot\mathbf{d}}` of its bond
        vector :math:`\mathbf{d} = \mathbf{R} + \boldsymbol\tau_j - \boldsymbol\tau_i`
        (see *System.set_peierls_phase*). A time-dependent :math:`\mathbf{A}(t)`
        is a uniform electric field :math:`-\partial_t\mathbf{A}`, e.g. light
        (see *tbkit.floquet*).

        :param k: k point (see *get_ham*).
        :param A: Tuple/ndarray of space_dim real numbers. Vector potential.

        :returns:
            * **ham** -- Complex ndarray, shape (norb, norb).
        '''
        error_handling.k_vector(k, 'k', self.dim)
        A = np.asarray(A, dtype='f8')
        error_handling.ndarray(A, 'A', self.space_dim)
        return self._bloch_ham((self.k_basis @ np.asarray(k, dtype='f8'))[None], A)[0]

    def _bloch_ham(self, k_cart: NDArray[np.float64], A: NDArray[np.float64] | None = None) -> NDArray[np.complex128]:
        '''
        Private method. *get_ham* (or, with a vector potential *A*,
        *get_ham_peierls*) at many k-points at once: *k_cart* has shape
        (nk, space_dim) and the result (nk, norb, norb).
        '''
        return self._bloch_sum(self._hop, k_cart, A=A)[0] + (np.diag(self.onsite) + self._onsite_offdiag)[None]

    def get_bands(
        self, ks: ArrayLike, eigenvec: bool = False,
    ) -> NDArray[np.float64] | tuple[NDArray[np.float64], NDArray[np.complex128]]:
        r'''
        Diagonalize :math:`H(\mathbf{k})` over a set of k-points, and keep
        the result for *plot_bands* (plotted against the cumulative distance
        between consecutive k-points).

        :param ks: ndarray, shape (nk, dim). k-points.
        :param eigenvec: Boolean. Default value False. If True, also return
            the eigenvectors.

        :returns:
            * **en** -- Real ndarray, shape (nk, norb). Band energies, sorted ascending.
              Complex, and sorted by real part, if :math:`H(\mathbf{k})` is
              non-Hermitian (complex onsite energies, see *set_onsite*).
            * **vn** -- Complex ndarray, shape (nk, norb, norb), only if *eigenvec* is True.
              vn[k, :, n] is the nth (right) eigenvector at ks[k].
        '''
        ks = np.atleast_2d(np.asarray(ks, dtype='f8'))
        error_handling.ks(ks, self.dim)
        out = self._diagonalize(ks, eigenvec)
        self.ks = ks
        self.en = out[0] if eigenvec else out
        steps = np.linalg.norm(np.diff(ks, axis=0), axis=1)
        self.ks_dist = np.concatenate([[0.], np.cumsum(steps)])[:len(ks)]
        self.nodes = []
        return out

    def _diagonalize(
        self, ks: NDArray[np.float64], eigenvec: bool = False,
    ) -> NDArray[np.float64] | tuple[NDArray[np.float64], NDArray[np.complex128]]:
        r'''
        Private method. Diagonalize :math:`H(\mathbf{k})` over the k-points
        *ks* (shape (nk, dim)), without touching the stored band structure.
        '''
        en = np.zeros((len(ks), self.norb), 'f8' if self.is_hermitian() else 'c16')
        if eigenvec:
            vn = np.zeros((len(ks), self.norb, self.norb), 'c16')
        for i, k in enumerate(ks):
            if eigenvec:
                en[i], vn[i] = self._eig(self.get_ham(k), eigenvec=True, k=k)
            else:
                en[i] = self._eig(self.get_ham(k), k=k)
        if eigenvec:
            return en, vn
        return en

    def is_hermitian(self) -> bool:
        '''
        Check whether :math:`H(\\mathbf{k})` is Hermitian: it is, unless an
        onsite term is not, or *set_hopping* was called with
        ``hermitian=False`` (such a model is always treated as
        non-Hermitian, and diagonalized with the general solver).

        :returns:
            * **hermitian** -- Boolean.
        '''
        return bool(not self._nonreciprocal and np.all(self.onsite.imag == 0.)
                        and np.array_equal(self._onsite_offdiag, self._onsite_offdiag.conj().T))

    def _eig(
        self, ham: NDArray[np.complex128], eigenvec: bool = False, k: NDArray | None = None,
    ) -> NDArray | tuple[NDArray, NDArray[np.complex128]]:
        '''
        Private method. Eigenvalues (and right eigenvectors) of one Bloch
        Hamiltonian: the Hermitian solver when *is_hermitian*, otherwise the
        general one, with the eigenvalues sorted by real part. With an
        overlap (see *set_overlap*), the generalized problem H v = E S(k) v.
        '''
        s = self.get_overlap(k) if (self._overlap_hop and k is not None) else None
        if self.is_hermitian():
            return LA.eigh(ham, s) if eigenvec else LA.eigvalsh(ham, s)
        if not eigenvec:
            en = LA.eigvals(ham, s)
            return en[np.argsort(en.real, kind='stable')]
        en, vn = LA.eig(ham, s)
        ind = np.argsort(en.real, kind='stable')
        return en[ind], vn[:, ind]

    def k_path(
        self, points: list[ArrayLike], nk: int,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        r'''
        Build a k-path through a list of high-symmetry points, and get the
        associated bands.

        :param points: List of at least two k-points (each a tuple/list of
            one/two real numbers).
        :param nk: Positive integer. Number of k-points per path segment.

        :returns:
            * **ks_dist** -- Real ndarray. Cumulative distance along the path,
              to be used as the x-axis of a band-structure plot.
            * **en** -- Real ndarray, shape (len(ks_dist), norb). Band energies.
        '''
        error_handling.k_path_points(points, self.dim)
        error_handling.positive_int(nk, 'nk')
        points = np.atleast_2d(np.asarray(points, dtype='f8'))
        segments = [np.linspace(points[i], points[i+1], nk, endpoint=False)
                          for i in range(len(points) - 1)]
        ks = np.concatenate(segments + [points[-1:]])
        en = self.get_bands(ks)
        self.nodes = self.ks_dist[::nk][:len(points)-1].tolist() + [self.ks_dist[-1]]
        return self.ks_dist, en

    def mesh_grid(
        self, nk: int | tuple[int, int],
    ) -> tuple[list[NDArray[np.float64]], NDArray[np.float64]]:
        '''
        Private method. Build a uniform grid of fractional coordinates
        spanning the Brillouin zone (each in [0, 1)), and the corresponding
        Cartesian k-points.

        :param nk: Positive integer, or tuple of *dim* positive integers.
            Number of k-points along each reciprocal lattice vector.

        :returns:
            * **fracs** -- List of *dim* real ndarrays, shape (nk1, nk2) each
              (or (nk1,) in 1D): fractional coordinates of the grid.
            * **ks** -- Real ndarray, shape (nk1*nk2, dim) (or (nk1, dim) in 1D).
        '''
        error_handling.nk(nk, self.dim)
        if isinstance(nk, int):
            nk = (nk,) * self.dim
        fracs = np.meshgrid(*[np.arange(n)/n for n in nk], indexing='ij')
        ks = sum(f.ravel()[:, None] * b[None, :] for f, b in zip(fracs, self.rec_vec_k))
        if self.dim == 1:
            return [fracs[0]], ks
        return list(fracs), ks

    def mesh_bands(self, nk: int | tuple[int, int]) -> NDArray[np.float64]:
        '''
        Diagonalize :math:`H(\\mathbf{k})` over a uniform mesh spanning the
        Brillouin zone. Unlike *get_bands*, it leaves the band structure kept
        for *plot_bands* untouched.

        :param nk: Positive integer, or tuple of *dim* positive integers.
            Number of k-points along each reciprocal lattice vector.

        :returns:
            * **en** -- Real ndarray, shape (nk1*nk2, norb). Band energies
              over the mesh (flattened).
        '''
        _, ks = self.mesh_grid(nk)
        return self._diagonalize(ks)

    def get_fermi_level(
        self, n_electrons: float, nk: int | tuple[int, ...] = 30, temperature: float = 0.,
    ) -> float:
        '''
        Get the Fermi level holding *n_electrons* electrons per unit cell
        (one electron per band and k-point), from the bands over a uniform
        Brillouin-zone mesh (see *tbkit.occupation.fermi_level*).

        :param n_electrons: Positive real or zero, at most norb. Electrons per unit cell.
        :param nk: Positive integer, or tuple. Default value 30. k-mesh.
        :param temperature: Positive real or zero. Default value 0.

        :returns:
            * **e_fermi** -- Real number.
        '''
        en = self.mesh_bands(nk)
        return occupation.fermi_level(en, n_electrons, temperature,
                                                   weights=np.full(en.shape, 1. / len(en)))

    def berry_curvature(
        self, bands: int | list[int], nk: int | tuple[int, int] = 30,
        plane: tuple[int, int] = (0, 1), k_fixed: float = 0.,
    ) -> NDArray[np.float64]:
        r'''
        Get the Berry curvature of a group of bands over a uniform
        Brillouin-zone mesh, using the gauge-invariant lattice method of
        Fukui, Hatsugai and Suzuki (J. Phys. Soc. Jpn. 74, 1674 (2005)):
        the flux through each mesh plaquette is minus the phase of the
        product of the (Slater-determinant) overlaps between the occupied
        subspaces at its four corners,

        .. math::

            \Omega_{\square} = -\arg\prod_{\langle\mathbf{k}\mathbf{k}'\rangle\in\partial\square}
            \det\langle u(\mathbf{k})|u(\mathbf{k}')\rangle\, ,

        the corners being visited counterclockwise. This is the discrete
        version of :math:`\Omega = \partial_{k_x}A_y - \partial_{k_y}A_x`,
        :math:`\mathbf{A} = i\langle u|\nabla_{\mathbf{k}}u\rangle`, and does
        not depend on the orientation (handedness) of *prim_vec*. The local
        curvature is that of the periodic gauge of *get_ham* (orbitals on the
        cell origin), which differs point by point from the one with the
        orbital positions included (see *quantum_geometric_tensor*); their
        integral, the Chern number, is the same. For a
        non-Hermitian :math:`H(\mathbf{k})` (see *set_onsite*), the bands
        are ordered by the real part of their energy and the overlaps are
        taken between right eigenvectors (the "RR" curvature of
        *biorthogonal_berry_curvature*). That ordering follows the bands
        continuously only if a real line gap separates *bands* from the
        others over the whole zone: without one (an imaginary line gap,
        or exceptional points), a ValueError is raised instead of a
        meaningless result -- see *biorthogonal_berry_curvature*.

        :param bands: Positive integer, or list of positive integers. Band
            index, or indices of a group of bands (e.g. all occupied bands
            below a gap).
        :param nk: Positive integer, or tuple of 2 positive integers.
            Default value 30. Number of k-points along each of the two
            reciprocal lattice vectors spanning the plane.
        :param plane: Tuple of two distinct integers. Default value (0, 1).
            3D lattices only: the reciprocal vectors
            :math:`\mathbf{b}_i, \mathbf{b}_j` spanning the plane of the
            Brillouin zone. The flux is oriented along the remaining
            primitive vector :math:`\mathbf{a}_l` (along +z in 2D).
        :param k_fixed: Real number. Default value 0. 3D lattices only: the
            fractional coordinate of the plane along the remaining reciprocal
            vector :math:`\mathbf{b}_l` (e.g. :math:`k_z` in units of
            :math:`|\mathbf{b}_3|` for ``plane=(0, 1)``).

        :returns:
            * **curv** -- Real ndarray, shape (nk1, nk2). Berry curvature
              (flux through each plaquette, in radians). Summing *curv* and
              dividing by :math:`2\pi` gives the Chern number, see
              *chern_number*.
        '''
        error_handling.dim_min(self.dim, 2)
        if isinstance(bands, int):
            bands = [bands]
        error_handling.band_indices(bands, self.norb)
        if isinstance(nk, int):
            nk = (nk, nk)
        error_handling.nk(nk, 2)
        error_handling.plane(plane, self.dim)
        error_handling.real_number(k_fixed, 'k_fixed')
        n1, n2 = nk
        ks = self._plane_mesh(nk, plane, k_fixed)
        if not self.is_hermitian():
            # bands labelled by Re E are continuous only across a real line gap
            en = np.array([self._eig(self.get_ham(k), k=k) for k in ks.reshape(-1, self.dim)])
            error_handling.line_gap(en.real, bands, 'real')
            error_handling.band_continuity(en.reshape(n1, n2, self.norb), bands)
        v = np.zeros((n1, n2, self.norb, len(bands)), 'c16')
        for i1 in range(n1):
            for i2 in range(n2):
                v[i1, i2] = self._subspace(ks[i1, i2], bands)
        curv = np.zeros((n1, n2))
        for i1 in range(n1):
            for i2 in range(n2):
                v1 = v[i1, i2]
                v2 = v[(i1+1) % n1, i2]
                v3 = v[(i1+1) % n1, (i2+1) % n2]
                v4 = v[i1, (i2+1) % n2]
                link = (np.linalg.det(v1.conj().T @ v2)
                             * np.linalg.det(v2.conj().T @ v3)
                             * np.linalg.det(v3.conj().T @ v4)
                             * np.linalg.det(v4.conj().T @ v1))
                curv[i1, i2] = -np.angle(link)
        # The corners k, k+b_i/n1, k+b_i/n1+b_j/n2, k+b_j/n2 run
        # counterclockwise about the normal only if (b_i x b_j).normal > 0;
        # otherwise every flux comes out with the opposite sign.
        return self._plane_orientation(plane) * curv

    def _plane_mesh(
        self, nk: tuple[int, int], plane: tuple[int, int], k_fixed: float,
    ) -> NDArray[np.float64]:
        '''
        Private method. k-points, shape (n1, n2, dim), of the uniform mesh of
        the Brillouin-zone plane spanned by b_i, b_j (plane = (i, j)), at
        fractional coordinate k_fixed along the remaining reciprocal vector.
        '''
        n1, n2 = nk
        i, j = plane
        rest = [l for l in range(self.dim) if l not in plane]
        fracs = np.zeros((n1, n2, self.dim))
        fracs[:, :, i] = (np.arange(n1) / n1)[:, None]
        fracs[:, :, j] = (np.arange(n2) / n2)[None, :]
        for l in rest:
            fracs[:, :, l] = k_fixed
        return fracs @ self.rec_vec_k

    def _plane_orientation(self, plane: tuple[int, int]) -> float:
        '''
        Private method. +1 if (b_i, b_j) is right-handed about the plane's
        normal (+z in 2D; the remaining primitive vector in 3D), else -1.
        '''
        b = [np.zeros(3) for _ in range(2)]
        for bb, idx in zip(b, plane):
            bb[:self.space_dim] = self.rec_vec[idx]
        if self.dim == 3:
            normal = np.array(self.lat.prim_vec[3 - sum(plane)], dtype='f8')
        else:
            a = [np.zeros(3), np.zeros(3)]
            for aa, vec in zip(a, self.lat.prim_vec):
                aa[:self.space_dim] = vec
            normal = np.cross(a[0], a[1])
            if abs(normal[2]) > 1e-12:
                normal = np.array([0., 0., 1.])
        return float(np.sign(np.dot(np.cross(b[0], b[1]), normal)))

    def _subspace(self, k: NDArray[np.float64], bands: list[int]) -> NDArray[np.complex128]:
        '''
        Private method. Orthonormal basis (columns) of the eigenvectors of
        *bands* at *k* (in the periodic gauge, orbital positions ignored).
        With an overlap S(k), the Lowdin-orthonormalized S^(1/2) v.
        '''
        _, vn = self._eig(self.get_ham(k), eigenvec=True, k=k)
        if self._overlap_hop:
            vn = LA.sqrtm(self.get_overlap(k)) @ vn
        return vn[:, bands]

    def biorthogonal_berry_curvature(
        self, bands: int | list[int], nk: int | tuple[int, int] = 30, kind: str = 'LR',
        gap: str = 'real', plane: tuple[int, int] = (0, 1), k_fixed: float = 0.,
    ) -> NDArray[np.float64]:
        r'''
        Get the Berry curvature of a group of bands of a (non-Hermitian)
        model from its right and left eigenvectors, :math:`H|R_n\rangle =
        E_n|R_n\rangle` and :math:`H^\dagger|L_n\rangle = E_n^*|L_n\rangle`,
        normalized by :math:`\langle L_m|R_n\rangle = \delta_{mn}` (Shen, Zhen
        and Fu, Phys. Rev. Lett. 120, 146402 (2018)). The four connections

        .. math::

            \mathbf{A}^{\alpha\beta} = i\langle u^\alpha|\nabla_{\mathbf{k}}u^\beta\rangle\, ,
            \qquad \alpha, \beta \in \{L, R\}\, ,

        have different curvatures point by point, but their integrals, the
        Chern numbers :math:`C^{LR} = C^{RL} = C^{RR} = C^{LL}`, are the same
        integer. The flux through each plaquette is computed as in
        *berry_curvature* (Fukui-Hatsugai-Suzuki), with the links
        :math:`\det\langle u^\alpha(\mathbf{k})|u^\beta(\mathbf{k}')\rangle` (for
        RR and LL, the phase does not depend on the normalization). For a
        Hermitian model all four are the curvature of *berry_curvature*.

        The bands are ordered at every k-point by the real part of their
        energy (*gap* = 'real') or by its imaginary part ('imaginary'), and
        *bands* must be separated from the others by a line gap of that kind
        over the whole zone -- a line :math:`\mathrm{Re}\,E = E_0` (or
        :math:`\mathrm{Im}\,E = E_0`) that no band crosses. Otherwise a
        ValueError is raised: the gap closes at exceptional points (see
        *tbkit.exceptional.find_exceptional_points*), where the Chern number
        can change.

        :param bands: Positive integer, or list of positive integers. Band
            indices in the order set by *gap*.
        :param nk: Positive integer, or tuple of 2 positive integers.
            Default value 30. See *berry_curvature*.
        :param kind: 'LR', 'RL', 'RR' or 'LL'. Default value 'LR'.
            :math:`\alpha\beta` above.
        :param gap: 'real' or 'imaginary'. Default value 'real'. Kind of line
            gap, and the order of the bands.
        :param plane: Tuple of two integers. Default value (0, 1). 3D only, see
            *berry_curvature*.
        :param k_fixed: Real number. Default value 0. 3D only, see *berry_curvature*.

        :returns:
            * **curv** -- Real ndarray, shape (nk1, nk2). Flux through each
              plaquette, in radians (sign convention of *berry_curvature*).
        '''
        error_handling.dim_min(self.dim, 2)
        if isinstance(bands, int):
            bands = [bands]
        error_handling.band_indices(bands, self.norb)
        if isinstance(nk, int):
            nk = (nk, nk)
        error_handling.nk(nk, 2)
        error_handling.biorthogonal_kind(kind)
        error_handling.gap_kind(gap)
        error_handling.plane(plane, self.dim)
        error_handling.real_number(k_fixed, 'k_fixed')
        error_handling.no_overlap(self._overlap_hop)
        n1, n2 = nk
        ks = self._plane_mesh(nk, plane, k_fixed)
        right = np.zeros((n1, n2, self.norb, len(bands)), 'c16')
        left = np.zeros_like(right)
        en = np.zeros((n1, n2, self.norb), 'c16')
        for i1 in range(n1):
            for i2 in range(n2):
                w, vl, vr = LA.eig(self.get_ham(ks[i1, i2]), left=True, right=True)
                order = np.argsort(w.real if gap == 'real' else w.imag, kind='stable')
                en[i1, i2] = w[order]
                r, l = vr[:, order][:, bands], vl[:, order][:, bands]
                # biorthonormal: <L_m|R_n> = delta_mn
                right[i1, i2] = r
                left[i1, i2] = l @ np.linalg.inv(l.conj().T @ r).conj().T
        error_handling.line_gap(en.reshape(-1, self.norb).real if gap == 'real'
                                           else en.reshape(-1, self.norb).imag, bands, gap)
        error_handling.band_continuity(en, bands)
        bra, ket = {'LR': (left, right), 'RL': (right, left),
                         'RR': (right, right), 'LL': (left, left)}[kind]
        # one link per oriented edge, U = det<u^alpha(k)|u^beta(k + dk)>, and
        # its inverse when the edge is run backwards: <L_a|R_b> and <L_b|R_a>
        # are not conjugate, so using both would spoil the exact
        # cancellation of the shared edges (for RR, 1/U has the phase of U*).
        link = np.zeros((2, n1, n2), 'c16')
        for i1 in range(n1):
            for i2 in range(n2):
                link[0, i1, i2] = np.linalg.det(bra[i1, i2].conj().T @ ket[(i1 + 1) % n1, i2])
                link[1, i1, i2] = np.linalg.det(bra[i1, i2].conj().T @ ket[i1, (i2 + 1) % n2])
        flux = (link[0] * np.roll(link[1], -1, axis=0)
                   / (np.roll(link[0], -1, axis=1) * link[1]))
        return self._plane_orientation(plane) * -np.angle(flux)

    def biorthogonal_chern_number(
        self, bands: int | list[int], nk: int | tuple[int, int] = 30, kind: str = 'LR',
        gap: str = 'real', plane: tuple[int, int] = (0, 1), k_fixed: float = 0.,
    ) -> float:
        r'''
        Get the Chern number of a line-gapped group of bands of a
        (non-Hermitian) model, :math:`C^{\alpha\beta} = \frac{1}{2\pi}\int_{BZ}
        \Omega^{\alpha\beta}\,d^2k`, the same integer for the four kinds
        LR, RL, RR, LL (see *biorthogonal_berry_curvature*). It equals the
        Hermitian Chern number of *chern_number* as long as the line gap stays
        open.

        :param bands: See *biorthogonal_berry_curvature*.
        :param nk: See *biorthogonal_berry_curvature*.
        :param kind: 'LR', 'RL', 'RR' or 'LL'. Default value 'LR'.
        :param gap: 'real' or 'imaginary'. Default value 'real'.
        :param plane: See *berry_curvature*.
        :param k_fixed: See *berry_curvature*.

        :returns:
            * **chern** -- Real number, close to an integer.
        '''
        return self.biorthogonal_berry_curvature(bands, nk, kind, gap, plane, k_fixed).sum() / (2*PI)

    def chern_number(
        self, bands: int | list[int], nk: int | tuple[int, int] = 30,
        plane: tuple[int, int] = (0, 1), k_fixed: float = 0.,
    ) -> float:
        r'''
        Get the Chern number of a group of bands:

        .. math::

            C = \frac{1}{2\pi}\int_{BZ} \Omega(\mathbf{k})\, d^2k

        an integer (up to the numerical precision set by *nk*) for a group
        of bands that is isolated from the rest of the spectrum by a gap
        everywhere in the Brillouin zone. See *berry_curvature*. For a
        non-Hermitian model, *bands* (ordered by Re E) must be isolated by a
        real line gap, or a ValueError is raised; *biorthogonal_chern_number*
        also handles imaginary line gaps and the left eigenvectors.

        :param bands: Positive integer, or list of positive integers. Band
            index, or indices of a group of bands (e.g. all occupied bands
            below a gap).
        :param nk: Positive integer, or tuple of 2 positive integers.
            Default value 30. Number of k-points along each reciprocal
            lattice vector.
        :param plane: Tuple of two integers. Default value (0, 1). 3D lattices
            only, see *berry_curvature*.
        :param k_fixed: Real number. Default value 0. 3D lattices only, see
            *berry_curvature*: the Chern number of the plane at this
            fractional coordinate (it jumps across a Weyl point).

        :returns:
            * **chern** -- Real number, close to an integer.
        '''
        return self.berry_curvature(bands, nk, plane, k_fixed).sum() / (2*PI)

    # ------------------------------------------------------------------
    # Anomalous and spin Hall conductivities (Kubo formula)
    # ------------------------------------------------------------------

    def _check_static(self) -> None:
        '''
        Private method. Methods that differentiate the stored hoppings call
        it first (a Floquet model overrides it to refuse).
        '''

    def _bloch_sum(
        self, hops: list, k_cart: NDArray[np.float64], directions: Sequence[NDArray[np.float64]] = (),
        positions: bool = False, A: NDArray[np.float64] | None = None,
    ) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
        r'''
        Private method. The Bloch sum :math:`\sum t\,e^{i\mathbf{k}\cdot\mathbf{d}}`
        of a hopping list over many k-points at once (*k_cart*, shape
        (nk, space_dim)), and its derivatives
        :math:`\sum i(\mathbf{u}\cdot\mathbf{d})\,t\,e^{i\mathbf{k}\cdot\mathbf{d}}`
        along the Cartesian unit vectors *directions*, with
        :math:`\mathbf{d} = \mathbf{R}` (plus :math:`\boldsymbol\tau_j-\boldsymbol\tau_i`
        with *positions*). A vector potential *A* multiplies each term by the
        Peierls phase :math:`e^{i\mathbf{A}\cdot(\mathbf{R}+\boldsymbol\tau_j-\boldsymbol\tau_i)}`
        (see *get_ham_peierls*). Shapes (nk, norb, norb) and
        (len(directions), nk, norb, norb). Every Bloch matrix of the model
        (*get_ham*, *get_overlap*, *get_ham_peierls*, *_bloch_derivatives*)
        is built here.
        '''
        n, nk = self.norb, len(k_cart)
        i = np.array([h[0] for h in hops], dtype=int)
        j = np.array([h[1] for h in hops], dtype=int)
        d = np.array([h[2] for h in hops], dtype='f8').reshape(len(hops), self.space_dim)
        t = np.array([h[3] for h in hops], dtype='c16')
        if positions or A is not None:
            tau = self.orbital_positions()
            bond = d + tau[j] - tau[i]
            if positions:
                d = bond
        phase = k_cart @ d.T
        if A is not None:
            phase = phase + bond @ A
        w = np.exp(1j * phase) * t[None, :]
        # accumulate term m of k-point p in entry (p, i_m, j_m): repeated
        # (i, j) pairs add up, as they must
        index = (np.arange(nk)[:, None] * n * n + i * n + j).ravel()

        def scatter(v):
            size = nk * n * n
            return (np.bincount(index, v.real.ravel(), size)
                        + 1j * np.bincount(index, v.imag.ravel(), size)).reshape(nk, n, n)
        mat = scatter(w)
        dmat = np.array([scatter(w * (1j * (d @ u))[None, :]) for u in directions],
                                  dtype='c16').reshape(len(directions), nk, n, n)
        return mat, dmat

    def _bloch_derivatives(
        self, ks: NDArray[np.float64], directions: list[NDArray[np.float64]], positions: bool,
    ) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
        r'''
        Private method. :math:`H(\mathbf{k})` and its exact derivatives
        :math:`\partial H/\partial k_u` along the Cartesian unit vectors
        *directions*, at the k-points *ks* (shape (nk, dim), coordinates of
        *get_ham*). With an overlap, the Lowdin-orthogonalized
        :math:`H' = S^{-1/2}HS^{-1/2}` (the eigenvectors :math:`S^{1/2}v` of
        the topology tools) and its derivative, with
        :math:`\partial S^{-1/2}` from the Daleckii-Krein formula in the
        eigenbasis :math:`S = \sum_a s_a|a\rangle\langle a|`:
        :math:`\langle a|\partial S^{-1/2}|b\rangle = \langle a|\partial S|b\rangle
        (s_a^{-1/2}-s_b^{-1/2})/(s_a-s_b)`, :math:`-s_a^{-3/2}/2` when :math:`s_a = s_b`.
        '''
        k_cart = ks @ self.k_basis.T
        ham, dham = self._bloch_sum(self._hop, k_cart, directions, positions)
        ham = ham + (np.diag(self.onsite) + self._onsite_offdiag)[None]
        if not self._overlap_hop:
            return ham, dham
        s, ds = self._bloch_sum(self._overlap_hop, k_cart, directions, positions)
        s = s + np.eye(self.norb)[None]
        sv, vs = np.linalg.eigh(s)
        error_handling.positive_overlap(sv.min())
        r = sv ** -0.5
        diff = sv[:, :, None] - sv[:, None, :]
        close = np.abs(diff) < 1e-10
        lk = np.where(close, -0.5 * sv[:, :, None] ** -1.5,
                           (r[:, :, None] - r[:, None, :]) / np.where(close, 1., diff))
        vh = vs.conj().transpose(0, 2, 1)
        s_inv = (vs * r[:, None, :]) @ vh
        ds_inv = vs @ ((vh @ ds @ vs) * lk) @ vh
        dham = ds_inv @ ham @ s_inv + s_inv @ dham @ s_inv + s_inv @ ham @ ds_inv
        return s_inv @ ham @ s_inv, dham

    def _kubo(
        self, ks: NDArray[np.float64], e_fermi: NDArray[np.float64], temperature: float,
        directions: list[NDArray[np.float64]], pairs: list[tuple[int, int]], positions: bool,
        spin_op: NDArray[np.complex128] | None,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        r'''
        Private method. The Kubo curvature
        :math:`F(\mathbf{k}) = -\sum_{n\neq m}(f_n-f_m)\,\mathrm{Im}[X_{nm}Y_{mn}]/(E_n-E_m)^2`
        (:math:`=\sum_n f_n\Omega_n`) at each k-point, Fermi energy and pair
        (X, Y) of velocities (X the spin current :math:`\{s, v\}/2` if
        *spin_op*). Shape (nk, len(e_fermi), len(pairs)). Pairs of
        degenerate states (:math:`f_n = f_m`) are skipped, so :math:`F` stays
        finite at band crossings below or above the Fermi level. Also
        returns, per k-point, the occupation-independent size of the
        curvature, :math:`\sum_{n\neq m}|\mathrm{Im}[X_{nm}Y_{mn}]|/(E_n-E_m)^2`,
        which locates the hot spots of the adaptive refinement.
        '''
        out = np.zeros((len(ks), len(e_fermi), len(pairs)))
        hot = np.zeros(len(ks))
        chunk = max(1, 400000 // self.norb ** 2)
        for c0 in range(0, len(ks), chunk):
            ham, dham = self._bloch_derivatives(ks[c0:c0 + chunk], directions, positions)
            en, vec = np.linalg.eigh(ham)
            vh = vec.conj().transpose(0, 2, 1)
            vel = [vh @ dh @ vec for dh in dham]
            de = en[:, :, None] - en[:, None, :]
            degenerate = np.abs(de) < 1e-9
            inv2 = np.where(degenerate, 0., 1. / np.where(degenerate, 1., de) ** 2)
            if spin_op is not None:
                s_e = vh @ spin_op @ vec
            for p, (a, b) in enumerate(pairs):
                x = vel[a] if spin_op is None else 0.5 * (s_e @ vel[a] + vel[a] @ s_e)
                w = np.imag(x * vel[b].transpose(0, 2, 1)) * inv2
                hot[c0:c0 + chunk] += np.abs(w).sum(axis=(1, 2))
                for e, mu in enumerate(e_fermi):
                    f = occupation.fermi_dirac(en, float(mu), temperature)
                    out[c0:c0 + chunk, e, p] = -np.einsum('knm,knm->k', f[:, :, None] - f[:, None, :], w)
        return out, hot

    def _hall(
        self, e_fermi, temperature, nk, plane, k_fixed, positions, refine, refine_fraction,
        spin_axis,
    ):
        '''
        Private method. Validate, set the geometry and mesh, integrate the
        Kubo curvature (with the adaptive refinement), for
        *hall_conductivity* (spin_axis None) and *spin_hall_conductivity*.
        '''
        self._check_static()
        error_handling.dim_min(self.dim, 2)
        error_handling.hermitian_model(self.is_hermitian())
        error_handling.fermi_energies(e_fermi)
        error_handling.positive_real_zero(temperature, 'temperature')
        error_handling.plane(plane, self.dim)
        error_handling.k_fixed_hall(k_fixed, self.dim, spin_axis is None)
        error_handling.boolean(positions, 'positions')
        error_handling.positive_int(refine, 'refine')
        error_handling.refine_fraction(refine_fraction)
        full = self.dim == 3 and k_fixed is None
        axes = [0, 1, 2] if full else list(plane)
        if isinstance(nk, int):
            nk = (nk,) * len(axes)
        error_handling.nk(nk, len(axes))
        spin_op = None
        if spin_axis is not None:
            error_handling.spinful(self.spin)
            error_handling.spin_axis(spin_axis)
            spin_op = np.kron(np.eye(self.n_sites), PAULI[spin_axis] / 2.)
        # Directions (Cartesian unit vectors) of the velocities, the pairs
        # (u, w) whose curvature Omega_uw = Omega.(u x w) is integrated, and
        # the prefactor turning the mesh average into the conductivity.
        if full:
            directions = list(np.eye(3))
            pairs = [(1, 2), (2, 0), (0, 1)]
            volume = abs(np.linalg.det(np.array(self.lat.prim_vec, dtype='f8')))
            prefactor = 2 * PI / volume
        elif self.dim == 3:
            i, j = plane
            a_l = np.array(self.lat.prim_vec[3 - i - j], dtype='f8')
            b_i, b_j = (np.array(self.rec_vec[p], dtype='f8') for p in plane)
            u = b_i / np.linalg.norm(b_i)
            directions = [u, np.cross(a_l / np.linalg.norm(a_l), u)]
            pairs = [(0, 1)]
            prefactor = np.linalg.norm(np.cross(b_i, b_j)) / (2 * PI)
        else:
            directions = [self.k_basis[:, 0], self.k_basis[:, 1]]
            pairs = [(0, 1)]
            det = np.linalg.det(self.rec_vec_k)
            prefactor = self._plane_orientation((0, 1)) * np.sign(det) * abs(det) / (2 * PI)
        e_arr = np.atleast_1d(np.asarray(e_fermi, dtype='f8')).ravel()
        grids = np.meshgrid(*[np.arange(n) / n for n in nk], indexing='ij')
        fracs = np.zeros((grids[0].size, self.dim))
        for ax, g in zip(axes, grids):
            fracs[:, ax] = g.ravel()
        for ax in range(self.dim):
            if ax not in axes:
                fracs[:, ax] = k_fixed
        args = (e_arr, temperature, directions, pairs, positions, spin_op)
        coarse, size = self._kubo(fracs @ self.rec_vec_k, *args)
        total = coarse.sum(axis=0)
        if refine > 1:
            # Wang, Yates, Souza and Vanderbilt (2006): resample the cells
            # where the curvature peaks on a refine^d submesh (midpoints of
            # the sub-cells), and replace their coarse value by the average.
            # The cells are ranked by the size of the curvature whatever the
            # occupations, so that cells straddling the Fermi surface are
            # refined together with their neighbours.
            n_hot = int(np.ceil(refine_fraction * len(fracs)))
            hot = np.argsort(size)[::-1][:n_hot]
            sub = np.array(np.meshgrid(*[(np.arange(refine) + 0.5) / refine - 0.5] * len(axes),
                                                    indexing='ij')).reshape(len(axes), -1).T
            offsets = np.zeros((len(sub), self.dim))
            for c, (ax, n) in enumerate(zip(axes, nk)):
                offsets[:, ax] = sub[:, c] / n
            fine = (fracs[hot][:, None, :] + offsets[None, :, :]).reshape(-1, self.dim)
            fine_val = self._kubo(fine @ self.rec_vec_k, *args)[0].reshape(n_hot, len(sub), len(e_arr), len(pairs))
            total = total + (fine_val.mean(axis=1) - coarse[hot]).sum(axis=0)
        sigma = prefactor * total / len(fracs)
        if not full:
            sigma = sigma[:, 0]
        if np.ndim(e_fermi) == 0:
            return sigma[0] if full else float(sigma[0])
        return sigma.reshape(np.shape(e_fermi) + sigma.shape[1:])

    def hall_conductivity(
        self, e_fermi: float | ArrayLike, temperature: float = 0., nk: int | tuple[int, ...] = 60,
        plane: tuple[int, int] = (0, 1), k_fixed: float | None = None, positions: bool = True,
        refine: int = 1, refine_fraction: float = 0.05,
    ) -> float | NDArray[np.float64]:
        r'''
        Get the intrinsic anomalous Hall conductivity at any Fermi level --
        in a gap, in a band, or at a band crossing -- from the Kubo formula
        over a uniform Brillouin-zone mesh,

        .. math::

            \sigma_{xy} = \frac{e^2}{\hbar}\int_{BZ}\frac{d^dk}{(2\pi)^d}\,
            F(\mathbf{k})\, ,\qquad
            F = -\sum_{n\neq m}(f_n - f_m)\,
            \frac{\mathrm{Im}\left[\langle n|\partial_x H|m\rangle\langle m|\partial_y H|n\rangle\right]}
            {(E_n - E_m)^2} = \sum_n f_n\,\Omega_n\, ,

        with :math:`f_n = f(E_n)` the Fermi-Dirac occupations and

        .. math::

            \Omega_n = -2\,\mathrm{Im}\sum_{m\neq n}
            \frac{\langle n|\partial_x H|m\rangle\langle m|\partial_y H|n\rangle}{(E_n-E_m)^2}
            = -2\,\mathrm{Im}\langle\partial_x u_n|\partial_y u_n\rangle

        the Berry curvature of band *n* (:math:`\partial_\mu H = \partial H/\partial k_\mu`,
        the convention :math:`\Omega = \partial_xA_y - \partial_yA_x`,
        :math:`\mathbf{A} = i\langle u|\nabla_{\mathbf{k}}u\rangle` of
        *berry_curvature*). The form with :math:`f_n - f_m` stays finite
        where bands touch: degenerate pairs have equal occupations and
        drop out, instead of cancelling between two diverging
        :math:`\Omega_n`. :math:`\partial H/\partial\mathbf{k}` is built
        analytically from the stored hoppings, :math:`\sum_{\mathbf{R}}
        i\mathbf{d}\,t\,e^{i\mathbf{k}\cdot\mathbf{d}}` with the Cartesian bond
        vector :math:`\mathbf{d}`.

        **Sign.** In a gap, :math:`\sigma_{xy} = C\,e^2/h` with :math:`C`
        the value *chern_number* returns for the occupied bands: this is the
        sign of TKNN (Phys. Rev. Lett. 49, 405 (1982)), whose
        :math:`\sigma_H = (e^2/h)\,\frac{i}{2\pi}\int d^2k\,(\langle\partial_1u|\partial_2u\rangle
        - \langle\partial_2u|\partial_1u\rangle)` has the same integrand,
        :math:`\Omega/2\pi`. The adiabatic (semiclassical) response of a
        charge :math:`q` of either sign, :math:`\dot{\mathbf{r}} = \partial_{\mathbf{k}}E
        - \dot{\mathbf{k}}\times\boldsymbol\Omega` with
        :math:`\hbar\dot{\mathbf{k}} = q\mathbf{E}`, gives the opposite sign
        for the tensor of Ohm's law :math:`j_x = \sigma^{\mathrm{Ohm}}_{xy}E_y`:
        :math:`\sigma^{\mathrm{Ohm}}_{xy} = -(q^2/\hbar)\int F\,d^dk/(2\pi)^d`,
        the convention of Wang, Yates, Souza and Vanderbilt (Phys. Rev. B
        74, 195118 (2006)) and of the Kubo-Bastin formula (*tbkit.kpm.hall_conductivity*
        returns, likewise, minus it). So the value returned is
        :math:`\sigma^{\mathrm{Ohm}}_{yx} = -\sigma^{\mathrm{Ohm}}_{xy}`; e.g. a
        massive Dirac cone :math:`v(k_x\sigma_x + k_y\sigma_y) + m\sigma_z`
        gives :math:`+\frac{e^2}{2h}\frac{m}{|E_F|}` outside its gap (the
        literature's :math:`\sigma_{xy} = -\frac{e^2}{2h}\frac{m}{|E_F|}`), and
        :math:`\frac{e^2}{2h}\mathrm{sgn}(m)` inside.

        **Orbital positions.** The velocity is :math:`\partial H/\partial\mathbf{k}`
        in the gauge of *positions*: with ``positions=True`` (the default)
        the bond vector is :math:`\mathbf{d} = \mathbf{R} + \boldsymbol\tau_j
        - \boldsymbol\tau_i`, the matrix of the physical velocity
        :math:`i[H, \mathbf{r}]` of a tight-binding model whose position
        operator is diagonal on its orbitals (the one *kpm.hall_conductivity*
        uses on a torus). With ``positions=False``, every orbital sits on
        its cell origin (the periodic gauge of *get_ham*). The two differ by
        :math:`i(E_n-E_m)\langle n|\boldsymbol\tau|m\rangle` in the
        velocity's matrix elements, so the band curvatures differ by
        :math:`\nabla\times\langle n|\boldsymbol\tau|n\rangle`, a total
        derivative: the conductivities agree whenever the bands below
        :math:`E_F` are all full (in a gap, at any temperature well below
        the gap) and differ in general at partial filling, by the circulation
        :math:`\frac{1}{2\pi}\oint_{FS}\langle n|\boldsymbol\tau|n\rangle\cdot d\mathbf{k}`
        around the Fermi surface. The default is the physical one.

        **Units and dimensions.** In 2D, :math:`\sigma` is in units of
        :math:`e^2/h` (:math:`\frac{1}{2\pi}\int d^2k\,F`). In 3D, with
        *k_fixed* given, the value of the plane (the conductance of that
        slice of the Brillouin zone, in :math:`e^2/h`, oriented like
        *chern_number* along :math:`\mathbf{a}_l`); with ``k_fixed=None``,
        the full 3D conductivity as the Hall (pseudo)vector
        :math:`(\sigma_{yz}, \sigma_{zx}, \sigma_{xy}) = \frac{1}{(2\pi)^2}\int d^3k\,\mathbf{F}`
        in units of :math:`e^2/(h\cdot\mathrm{length})` (length the unit of
        *prim_vec*): :math:`\frac{1}{2\pi}\sum_l\bar C_l\,\mathbf{b}_l`, e.g.
        :math:`C/c` along :math:`z` for a stack of Chern layers :math:`c` apart.

        **Convergence.** The integrand peaks where bands nearly touch close
        to :math:`E_F` (hot spots), and jumps across the Fermi surface. A
        *temperature* smooths the latter. *refine* resamples the fraction
        *refine_fraction* of mesh cells with the largest :math:`|F|` on a
        finer submesh (*refine* points per direction), in the spirit of the
        adaptive mesh refinement of Wang, Yates, Souza and Vanderbilt
        (2006); the cells are ranked by the size of the curvature whatever
        the occupations. It is meant for metals: in a gap the plain uniform
        mesh already converges exponentially, and refining part of it
        costs a little accuracy (about :math:`4\cdot10^{-4}` for the
        Haldane model on a 60 x 60 mesh). With an overlap (*set_overlap*), :math:`H` is first
        Lowdin-orthogonalized, :math:`S^{-1/2}HS^{-1/2}`, as by the topology
        tools (exact derivatives of :math:`S^{-1/2}` included).

        :param e_fermi: Real number, or array of real numbers. Fermi energies
            (all of them are computed from one diagonalization of the mesh).
        :param temperature: Positive real or zero. Default value 0. In
            energy units (:math:`k_B = 1`).
        :param nk: Positive integer, or tuple (two integers for a plane,
            three for the full 3D conductivity). Default value 60. k-mesh.
        :param plane: Tuple of two integers. Default value (0, 1). 3D only:
            the reciprocal vectors spanning the plane (see *berry_curvature*).
        :param k_fixed: Real number or None. Default value None. 3D only:
            the fractional coordinate of the plane along the remaining
            reciprocal vector; None for the full 3D conductivity. Ignored in 2D.
        :param positions: Boolean. Default value True. Include the orbital
            positions in the bond vectors (see above).
        :param refine: Positive integer. Default value 1 (no refinement).
            Submesh points per direction in the refined cells.
        :param refine_fraction: Real in (0, 1]. Default value 0.05. Fraction
            of the mesh cells that are refined.

        :returns:
            * **sigma** -- Real number (or ndarray shaped like *e_fermi*); in
              3D with ``k_fixed=None``, ndarray of shape (3,) (or
              ``e_fermi.shape + (3,)``).

        Example usage::

            e_f = np.linspace(-3., 3., 301)
            sigma = hal.hall_conductivity(e_f, nk=120)  # plateau at C in the gap
        '''
        return self._hall(e_fermi, temperature, nk, plane, k_fixed, positions, refine,
                                refine_fraction, None)

    def spin_hall_conductivity(
        self, e_fermi: float | ArrayLike, temperature: float = 0., nk: int | tuple[int, int] = 60,
        plane: tuple[int, int] = (0, 1), k_fixed: float | None = None, positions: bool = True,
        spin_axis: str = 'z', refine: int = 1, refine_fraction: float = 0.05,
    ) -> float | NDArray[np.float64]:
        r'''
        Get the intrinsic spin Hall conductivity of a spinful model
        (``spin=True``): the Kubo formula of *hall_conductivity* with the
        charge current along :math:`x` replaced by the spin current
        (Sinova et al., Phys. Rev. Lett. 92, 126603 (2004))

        .. math::

            j^{s}_x = \tfrac12\{s, v_x\}\, ,\qquad s = \tfrac{\hbar}{2}\sigma_{a}\, ,

        i.e. :math:`\sigma^{s}_{xy} = \frac{e}{2\pi}\cdot\frac{1}{2\pi}\int d^2k\,F^s`,
        :math:`F^s = -\sum_{n\neq m}(f_n-f_m)\,\mathrm{Im}[\langle n|j^s_x|m\rangle\langle m|\partial_yH|n\rangle]/(E_n-E_m)^2`
        (:math:`j^s_x` with :math:`s = \sigma_a/2`, in units of :math:`\hbar`).
        The unit is :math:`e/2\pi`: one :math:`e/\hbar` from the electric
        field, one :math:`\hbar/2` from the spin, and :math:`1/2\pi` from the
        Brillouin-zone integral, as for :math:`e^2/h = (e^2/\hbar)/2\pi`.
        When :math:`s_z` is conserved, the two spins decouple and
        :math:`j^s_x = \pm\frac{\hbar}{2}v_x` in each, so in a gap
        :math:`\sigma^s_{xy} = \frac{e}{2\pi}\cdot\frac{C_\uparrow - C_\downarrow}{2}`,
        :math:`C_\sigma` the Chern number of each spin sector (the sign of
        *hall_conductivity* and *chern_number*): :math:`\pm e/2\pi` for the
        Kane-Mele quantum spin Hall insulator (Kane and Mele 2005). Rashba
        coupling breaks the conservation of :math:`s_z`, and the value is
        then no longer quantized (the :math:`\mathbb{Z}_2` invariant, see
        *z2_invariant*, still is).

        :param e_fermi: See *hall_conductivity*.
        :param temperature: See *hall_conductivity*.
        :param nk: Positive integer, or tuple of 2 integers. Default value 60.
        :param plane: See *hall_conductivity*.
        :param k_fixed: Real number or None. Default value None. 3D only: the
            plane (required in 3D, where only planes are computed).
        :param positions: Boolean. Default value True. See *hall_conductivity*.
        :param spin_axis: 'x', 'y' or 'z'. Default value 'z'. Spin component
            :math:`a` carried by the current.
        :param refine: See *hall_conductivity*.
        :param refine_fraction: See *hall_conductivity*.

        :returns:
            * **sigma_s** -- Real number (or ndarray shaped like *e_fermi*), in
              units of :math:`e/2\pi`.
        '''
        return self._hall(e_fermi, temperature, nk, plane, k_fixed, positions, refine,
                                refine_fraction, spin_axis)

    # ------------------------------------------------------------------
    # Finite samples and non-Hermitian band theory
    # ------------------------------------------------------------------

    def _hop_cells(self) -> list[tuple[int, int, tuple[int, ...], complex]]:
        r'''
        Private method. The stored hoppings with their lattice vector as
        integers :math:`(n_1, ...)` (from :math:`\mathbf{R}\cdot\mathbf{b}_i/2\pi`).
        '''
        rec = np.array(self.rec_vec, dtype='f8')
        return [(i, j, tuple(int(n) for n in np.rint(rec @ R / (2 * PI))), t)
                   for i, j, R, t in self._hop]

    def _finite_entries(
        self, n_cells: int | tuple[int, ...], periodic: bool,
    ) -> tuple[tuple[int, ...], int, list[tuple[NDArray, NDArray, int, int, tuple, NDArray, complex]]]:
        '''
        Private method. For each stored hopping, the row and column indices
        it fills in a finite sample of *n_cells* cells (see *finite_ham*),
        with its orbitals, lattice vector (integers and Cartesian) and amplitude.
        '''
        if isinstance(n_cells, int):
            n_cells = (n_cells,) * self.dim
        n_tot = int(np.prod(n_cells))
        cells = np.array(np.meshgrid(*[np.arange(n) for n in n_cells], indexing='ij'))
        cells = cells.reshape(self.dim, -1).T  # (n_tot, dim)
        strides = np.cumprod((1,) + tuple(n_cells[:-1]))
        index = cells @ strides
        order = np.argsort(index)
        cells = cells[order]
        dims = np.array(n_cells)
        entries = []
        for (i, j, R, t), (_, _, R_cart, _) in zip(self._hop_cells(), self._hop):
            target = cells + np.array(R)[None, :]
            if periodic:
                target = target % dims
                keep = np.ones(n_tot, bool)
            else:
                keep = np.all((target >= 0) & (target < dims), axis=1)
            rows = np.arange(n_tot)[keep] * self.norb + i
            cols = (target[keep] @ strides) * self.norb + j
            entries.append((rows, cols, i, j, R, R_cart, t))
        return n_cells, n_tot, entries

    def finite_ham(
        self, n_cells: int | tuple[int, ...], periodic: bool = False, sparse: bool = False,
    ) -> NDArray[np.complex128] | sp.csr_matrix:
        r'''
        Get the real-space Hamiltonian of a finite sample of
        :math:`N_1\times N_2\times\dots` unit cells of the model -- open
        boundaries (a flake, a finite chain), or periodic ones (a torus).
        Orbital *o* of cell :math:`(n_1, n_2, n_3)` is row
        :math:`((n_3 N_2 + n_2) N_1 + n_1)\,\mathrm{norb} + o`.

        :param n_cells: Positive integer, or tuple of *dim* positive integers.
        :param periodic: Boolean. Default value False. Periodic boundary
            conditions (the spectrum is then that of *get_bands* on the
            :math:`N_1\times N_2\times\dots` k-mesh), else open ones.
        :param sparse: Boolean. Default value False. Return a sparse CSR
            matrix (for large samples, e.g. with *tbkit.kpm*) instead of a
            dense array.

        :returns:
            * **ham** -- Complex ndarray (or CSR matrix), shape (N*norb, N*norb), N the number of cells.
        '''
        error_handling.nk(n_cells, self.dim)
        error_handling.boolean(periodic, 'periodic')
        error_handling.boolean(sparse, 'sparse')
        _, n_tot, entries = self._finite_entries(n_cells, periodic)
        onsite = np.diag(self.onsite) + self._onsite_offdiag
        if sparse:
            onsite_rows, onsite_cols = np.nonzero(onsite)
            rows = [c * self.norb + onsite_rows for c in range(n_tot)]
            cols = [c * self.norb + onsite_cols for c in range(n_tot)]
            vals = [onsite[onsite_rows, onsite_cols]] * n_tot
            for r, c, _, _, _, _, t in entries:
                rows.append(r)
                cols.append(c)
                vals.append(np.full(len(r), t, dtype='c16'))
            n = n_tot * self.norb
            return sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                                           shape=(n, n)).tocsr()
        ham = np.zeros((n_tot * self.norb, n_tot * self.norb), 'c16')
        for c in range(n_tot):
            ham[c*self.norb:(c+1)*self.norb, c*self.norb:(c+1)*self.norb] += onsite
        for rows, cols, _, _, _, _, t in entries:
            np.add.at(ham, (rows, cols), t)
        return ham

    def finite_velocity(
        self, n_cells: int | tuple[int, ...], periodic: bool = False, sparse: bool = False,
    ) -> list[NDArray[np.complex128] | sp.csr_matrix]:
        r'''
        Get the velocity operators :math:`\mathbf{v} = i[H, \mathbf{r}]`
        (:math:`\hbar = 1`) of the finite sample of *finite_ham*, from the
        Cartesian bond vectors of the hoppings rather than from the site
        coordinates:

        .. math::

            (v_\alpha)_{ab} = i\,H_{ab}\,d_\alpha\, ,\qquad
            \mathbf{d} = \mathbf{R} + \boldsymbol\tau_j - \boldsymbol\tau_i\, ,

        so that they are well defined on a torus (``periodic=True``), where
        the position operator is not: a bond that wraps around the sample
        keeps its short displacement. They are the real-space counterpart of
        :math:`\partial H/\partial\mathbf{k}` with ``positions=True``
        (*hall_conductivity*), and the input of *tbkit.kpm.hall_conductivity*.

        :param n_cells: See *finite_ham*.
        :param periodic: Boolean. Default value False. See *finite_ham*.
        :param sparse: Boolean. Default value False. See *finite_ham*.

        :returns:
            * **vel** -- List of *space_dim* complex ndarrays (or CSR
              matrices), shape (N*norb, N*norb): :math:`v_x, v_y, (v_z)`.
        '''
        error_handling.nk(n_cells, self.dim)
        error_handling.boolean(periodic, 'periodic')
        error_handling.boolean(sparse, 'sparse')
        _, n_tot, entries = self._finite_entries(n_cells, periodic)
        tau = self.orbital_positions()
        n = n_tot * self.norb
        vel = []
        for alpha in range(self.space_dim):
            rows, cols, vals = [np.zeros(0, int)], [np.zeros(0, int)], [np.zeros(0, 'c16')]
            for r, c, i, j, _, R_cart, t in entries:
                d = R_cart[alpha] + tau[j, alpha] - tau[i, alpha]
                rows.append(r)
                cols.append(c)
                vals.append(np.full(len(r), 1j * t * d, dtype='c16'))
            mat = sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                                          shape=(n, n)).tocsr()
            vel.append(mat if sparse else mat.toarray())
        return vel

    def get_ham_beta(self, beta: complex) -> NDArray[np.complex128]:
        r'''
        Get the non-Bloch Hamiltonian of a 1D model,
        :math:`H(\beta) = \sum_{n} t(n)\, \beta^{n}`: the Bloch Hamiltonian
        continued off the unit circle, :math:`\beta = e^{ik}` with complex
        :math:`k`. On the generalized Brillouin zone (see *gbz*), its
        spectrum is the open-chain spectrum of a non-Hermitian chain.

        :param beta: Complex number, nonzero.

        :returns:
            * **ham** -- Complex ndarray, shape (norb, norb).
        '''
        error_handling.dim_exact(self.dim, 1)
        error_handling.nonzero_number(beta, 'beta')
        ham = np.diag(self.onsite).astype('c16') + self._onsite_offdiag
        for i, j, (n,), t in self._hop_cells():
            ham[i, j] += t * complex(beta) ** n
        return ham

    def spectral_winding(self, e_ref: complex = 0., nk: int = 400) -> float:
        r'''
        Get the spectral winding number of a 1D (non-Hermitian) model about
        the reference energy :math:`E_r`,

        .. math::

            W(E_r) = \frac{1}{2\pi i}\oint_0^{2\pi} dk\,
                     \partial_k \ln\det\left[H(k) - E_r\right]\, ,

        the number of times the periodic-chain spectrum winds around
        :math:`E_r` in the complex plane. :math:`W \neq 0` for some
        :math:`E_r` is the criterion for the non-Hermitian skin effect: the
        open-chain eigenstates then pile up at one end (Okuma, Kawabata,
        Shiozaki and Sato, Phys. Rev. Lett. 124, 086801 (2020)).

        :param e_ref: Complex number. Default value 0. Reference energy.
        :param nk: Positive integer. Default value 400. k-points.

        :returns:
            * **W** -- Real number, close to an integer (it is ill defined if
              :math:`E_r` lies on the periodic spectrum).
        '''
        error_handling.dim_exact(self.dim, 1)
        error_handling.number(e_ref, 'e_ref')
        error_handling.positive_int(nk, 'nk')
        ks = np.arange(nk + 1) / nk * self.rec_vec_k[0, 0]
        dets = np.array([np.linalg.det(self.get_ham([k]) - e_ref * np.eye(self.norb)) for k in ks])
        steps = np.angle(dets[1:] / dets[:-1])
        return float(np.sum(steps) / (2 * PI))

    def gbz(self, energies: ArrayLike) -> NDArray[np.complex128]:
        r'''
        Get the generalized Brillouin zone of a 1D non-Hermitian model
        (Yao and Wang, Phys. Rev. Lett. 121, 086803 (2018); Yokomizo and
        Murakami, Phys. Rev. Lett. 123, 066404 (2019)): for each energy
        :math:`E`, the roots of :math:`\det[H(\beta)-E] = 0` sorted by modulus,
        :math:`|\beta_1|\le\dots\le|\beta_{p+s}|` (:math:`p` the order of its
        pole at :math:`\beta = 0`), and the middle pair
        :math:`\beta_p, \beta_{p+1}`. The open-chain spectrum, in the limit of
        a long chain, is where :math:`|\beta_p| = |\beta_{p+1}|`; those
        :math:`\beta` trace the generalized Brillouin zone, on which the
        non-Bloch bands :math:`H(\beta)` (see *get_ham_beta*) replace the
        Bloch ones. For a Hermitian model it is the unit circle.

        :param energies: Complex array of energies (e.g. open-chain eigenvalues
            from *finite_ham*).

        :returns:
            * **beta** -- Complex ndarray, shape (len(energies), 2): :math:`\beta_p, \beta_{p+1}`.
        '''
        error_handling.dim_exact(self.dim, 1)
        energies = np.atleast_1d(np.asarray(energies, dtype='c16'))
        error_handling.ndarray_empty(energies, 'energies')
        powers = [n for _, _, (n,), _ in self._hop_cells()] + [0]
        low, high = self.norb * min(powers), self.norb * max(powers)
        n_samples = 2 * (high - low) + 8
        theta = 2 * PI * np.arange(n_samples) / n_samples
        hams = np.array([self.get_ham_beta(np.exp(1j * th)) for th in theta])
        out = np.zeros((len(energies), 2), 'c16')
        exps = np.arange(low, high + 1)
        for n, e in enumerate(energies):
            dets = np.linalg.det(hams - e * np.eye(self.norb)[None])
            # Laurent coefficients c_m of det[H(beta) - E] = sum_m c_m beta^m
            coef = np.array([np.mean(dets * np.exp(-1j * m * theta)) for m in exps])
            nonzero = np.abs(coef) > 1e-12 * np.max(np.abs(coef))
            first, last = np.argmax(nonzero), len(coef) - 1 - np.argmax(nonzero[::-1])
            p = -exps[first]
            roots = np.roots(coef[first:last + 1][::-1])
            roots = roots[np.argsort(np.abs(roots))]
            error_handling.gbz_roots(len(roots), p)
            out[n] = roots[p - 1], roots[p]
        return out

    # ------------------------------------------------------------------
    # Berry phases, Wannier centres, Z2, symmetries, quantum geometry
    # ------------------------------------------------------------------

    def orbital_positions(self) -> NDArray[np.float64]:
        r'''
        Get the positions :math:`\boldsymbol\tau` of the orbitals within the
        unit cell (each site twice, for its two spin components, if
        ``spin=True``).

        :returns:
            * **tau** -- Real ndarray, shape (norb, space_dim).
        '''
        tau = np.array([dic['r0'] for dic in self.lat.unit_cell], dtype='f8')
        return np.repeat(tau, 2, axis=0) if self.spin else tau

    def _loop(
        self, bands: list[int], nk: int, direction: int, fracs_perp: dict[int, float],
        positions: bool,
    ) -> NDArray[np.complex128]:
        '''
        Private method. Wilson loop matrix of *bands* along the closed path
        k0 -> k0 + b_direction (nk steps), at fixed fractional coordinates
        *fracs_perp* along the other reciprocal vectors.
        '''
        frac = np.zeros(self.dim)
        for d, f in fracs_perp.items():
            frac[d] = f
        b = self.rec_vec_k[direction]
        k0 = frac @ self.rec_vec_k
        # each link carries exp(-i dk.tau): the Bloch sums then include the
        # orbital positions, so that the phases locate the Wannier centres
        # in space rather than on the cell origin.
        dk_space = self.k_basis @ (b / nk)
        phase = np.exp(-1j * self.orbital_positions() @ dk_space) if positions \
            else np.ones(self.norb)
        vs = [self._subspace(k0 + m * b / nk, bands) for m in range(nk)]
        wilson = np.eye(len(bands), dtype='c16')
        for m in range(nk):
            link = vs[m].conj().T @ (phase[:, None] * vs[(m + 1) % nk])
            # keep only the unitary part of each link (its singular values
            # tend to 1 as nk grows): the product then stays unitary
            u, _, vh = np.linalg.svd(link)
            wilson = wilson @ (u @ vh)
        return wilson

    def _perp(self, direction: int, k_perp: float | tuple | None) -> dict[int, float]:
        '''
        Private method. Fractional coordinates of a loop along the other
        reciprocal vectors, from *k_perp* (a number in 2D, a pair in 3D).
        '''
        others = [d for d in range(self.dim) if d != direction]
        if k_perp is None:
            k_perp = (0.,) * len(others)
        if not isinstance(k_perp, tuple):
            k_perp = (k_perp,)
        error_handling.k_perp(k_perp, len(others))
        return dict(zip(others, k_perp))

    def berry_phase(
        self, bands: int | list[int], nk: int = 100, direction: int = 0,
        k_perp: float | tuple[float, float] | None = None, positions: bool = True,
    ) -> float:
        r'''
        Get the Berry phase of a group of bands along a closed loop across the
        Brillouin zone, :math:`\mathbf{k}\to\mathbf{k}+\mathbf{b}_{direction}`,

        .. math::

            \gamma = \oint \mathbf{A}\cdot d\mathbf{k} = -\arg\det W\, ,\qquad
            W = \prod_{m} \langle u(\mathbf{k}_m)|u(\mathbf{k}_{m+1})\rangle\, ,

        with :math:`\mathbf{A} = i\langle u|\nabla_{\mathbf{k}}u\rangle` (the
        convention of *berry_curvature*). In 1D it is the Zak phase: its
        value :math:`\gamma/2\pi`, modulo 1, is the centre of the band's
        Wannier functions in units of :math:`\mathbf{a}_{direction}` (the
        polarization).

        :param bands: Band index, or list of band indices.
        :param nk: Positive integer. Default value 100. Number of k-points on the loop.
        :param direction: Integer. Default value 0. Reciprocal vector the loop follows.
        :param k_perp: Real number (2D) or pair of real numbers (3D). Default
            value None (zero). Fractional coordinates of the loop along the
            other reciprocal vectors.
        :param positions: Boolean. Default value True. Include the orbital
            positions in the Bloch functions (the physical choice: the phase
            then locates the Wannier centres in the crystal). If False, all
            orbitals are placed on the cell origin (the periodic gauge of
            *get_ham*); the two differ by :math:`\sum_n\mathbf{b}\cdot\boldsymbol\tau`
            weighted by the orbital content, and agree on quantized
            differences between phases.

        :returns:
            * **gamma** -- Real number in :math:`(-\pi, \pi]`.
        '''
        bands = self._check_loop(bands, nk, direction, positions)
        wilson = self._loop(bands, nk, direction, self._perp(direction, k_perp), positions)
        return float(-np.angle(np.linalg.det(wilson)))

    def wannier_centers(
        self, bands: int | list[int], nk: int = 100, direction: int = 0,
        k_perp: float | tuple[float, float] | None = None, positions: bool = True,
    ) -> NDArray[np.float64]:
        r'''
        Get the (hybrid) Wannier centres of a group of bands: minus the
        phases of the eigenvalues of the Wilson loop (see *berry_phase*),
        divided by :math:`2\pi`. They are the positions, in units of
        :math:`\mathbf{a}_{direction}` and modulo 1, of the maximally
        localized Wannier functions along that direction (hybrid: still
        Bloch waves along the others, at *k_perp*). Their sum is
        :math:`\gamma/2\pi`, modulo 1.

        :param bands: Band index, or list of band indices.
        :param nk: Positive integer. Default value 100. Number of k-points on the loop.
        :param direction: Integer. Default value 0.
        :param k_perp: See *berry_phase*.
        :param positions: Boolean. Default value True. See *berry_phase*.

        :returns:
            * **centers** -- Real ndarray, sorted, in :math:`[0, 1)`.
        '''
        bands = self._check_loop(bands, nk, direction, positions)
        wilson = self._loop(bands, nk, direction, self._perp(direction, k_perp), positions)
        return np.sort((-np.angle(np.linalg.eigvals(wilson)) / (2*PI)) % 1.)

    def wannier_flow(
        self, bands: int | list[int], nk: int = 100, nk_perp: int = 51, direction: int = 0,
        flow: int | None = None, k_fixed: float = 0., positions: bool = True,
        k_range: tuple[float, float] = (0., 1.),
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        r'''
        Get the flow of the hybrid Wannier centres (see *wannier_centers*)
        along *direction*, as the loop is moved across the Brillouin zone
        along the reciprocal vector *flow*. In a Chern insulator the centres
        wind by :math:`C` cells over a full period; in a
        :math:`\mathbb{Z}_2` topological insulator, the Kramers pairs swap
        partners over half a period (see *z2_invariant*).

        :param bands: Band index, or list of band indices.
        :param nk: Positive integer. Default value 100. k-points on each loop.
        :param nk_perp: Positive integer. Default value 51. Number of loops.
        :param direction: Integer. Default value 0. Direction of the loops.
        :param flow: Integer. Default value None (the first other direction).
            Reciprocal vector along which the loops are moved.
        :param k_fixed: Real number. Default value 0. 3D only: fractional
            coordinate along the remaining reciprocal vector.
        :param positions: Boolean. Default value True. See *berry_phase*.
        :param k_range: Pair of real numbers. Default value (0, 1). Range of
            the fractional coordinate along *flow* (both ends included).

        :returns:
            * **fracs** -- Real ndarray, shape (nk_perp,). Fractional coordinates along *flow*.
            * **centers** -- Real ndarray, shape (nk_perp, len(bands)). Sorted centres, in [0, 1).
        '''
        bands = self._check_loop(bands, nk, direction, positions)
        error_handling.dim_min(self.dim, 2)
        error_handling.positive_int(nk_perp, 'nk_perp')
        others = [d for d in range(self.dim) if d != direction]
        if flow is None:
            flow = others[0]
        error_handling.flow(flow, direction, self.dim)
        error_handling.real_number(k_fixed, 'k_fixed')
        error_handling.lims(k_range)
        fracs = np.linspace(k_range[0], k_range[1], nk_perp)
        centers = np.zeros((nk_perp, len(bands)))
        for n, f in enumerate(fracs):
            perp = {d: (f if d == flow else k_fixed) for d in others}
            wilson = self._loop(bands, nk, direction, perp, positions)
            centers[n] = np.sort((-np.angle(np.linalg.eigvals(wilson)) / (2*PI)) % 1.)
        return fracs, centers

    def z2_invariant(
        self, bands: list[int], nk: int = 100, nk_perp: int = 51, direction: int = 0,
        flow: int | None = None, k_fixed: float = 0.,
    ) -> int:
        r'''
        Get the :math:`\mathbb{Z}_2` invariant of a time-reversal-symmetric
        group of bands (an even number, e.g. all occupied bands of the
        Kane-Mele model) from the flow of its hybrid Wannier centres over
        half the Brillouin zone (Soluyanov and Vanderbilt, Phys. Rev. B 83,
        235401 (2011); Yu, Qi, Bernevig, Fang and Dai, Phys. Rev. B 84,
        075119 (2011)): follow the midpoint of the largest gap between the
        centres, and count, modulo 2, how many centres it jumps over. It is
        odd when the Kramers partners exchange along the flow -- a
        topological insulator -- and even otherwise.

        Unlike the parity criterion (*parity_z2*), it needs no inversion
        symmetry, so it also works with Rashba coupling.

        :param bands: List of band indices (an even number of bands).
        :param nk: Positive integer. Default value 100. k-points on each loop.
        :param nk_perp: Positive integer. Default value 51. Number of loops
            over the half zone (the flow must be resolved finely enough that
            no two centres cross between neighbouring loops unnoticed).
        :param direction: Integer. Default value 0. See *wannier_flow*.
        :param flow: Integer. Default value None. See *wannier_flow*.
        :param k_fixed: Real number. Default value 0. 3D only: a time-reversal
            invariant plane has *k_fixed* 0 or 0.5 (the weak and strong
            indices follow from the six such planes).

        :returns:
            * **nu** -- 0 (trivial) or 1 (topological).
        '''
        error_handling.even_bands(bands)
        _, centers = self.wannier_flow(bands, nk, nk_perp, direction, flow, k_fixed,
                                                    positions=False, k_range=(0., 0.5))
        gap = [self._largest_gap(c) for c in centers]
        jumps = sum(self._between(gap[n], gap[n + 1], centers[n + 1])
                           for n in range(len(gap) - 1))
        return int(jumps % 2)

    @staticmethod
    def _largest_gap(centers: NDArray[np.float64]) -> float:
        '''
        Private method. Midpoint of the largest gap between sorted centres on the circle [0, 1).
        '''
        ext = np.append(centers, centers[0] + 1.)
        n = np.argmax(np.diff(ext))
        return float((ext[n] + ext[n + 1]) / 2 % 1.)

    @staticmethod
    def _between(a: float, b: float, centers: NDArray[np.float64]) -> int:
        '''
        Private method. Number of centres on the shorter arc between a and b.
        '''
        d = (b - a) % 1.
        start, length = (a, d) if d <= 0.5 else (b, 1. - d)
        return int(np.sum((centers - start) % 1. < length))

    def _check_loop(self, bands, nk, direction, positions) -> list[int]:
        '''
        Private method. Validate the arguments shared by the Wilson-loop methods.
        '''
        if isinstance(bands, int):
            bands = [bands]
        error_handling.band_indices(bands, self.norb)
        error_handling.positive_int(nk, 'nk')
        error_handling.direction(direction, self.dim)
        error_handling.boolean(positions, 'positions')
        return bands

    def _trims(self) -> list[NDArray[np.float64]]:
        '''
        Private method. Time-reversal-invariant momenta: fractional
        coordinates in {0, 1/2}^dim, as k points.
        '''
        grid = np.array(np.meshgrid(*[[0., 0.5]] * self.dim, indexing='ij')).reshape(self.dim, -1).T
        return [f @ self.rec_vec_k for f in grid]

    def _operator(self, op, k: NDArray[np.float64]) -> NDArray[np.complex128]:
        '''
        Private method. Matrix of a symmetry operator at k (op is a matrix, or a callable of k).
        '''
        mat = np.asarray(op(k) if callable(op) else op, dtype='c16')
        error_handling.operator(mat, self.norb)
        return mat

    def parity_z2(self, inversion, bands: list[int]) -> int:
        r'''
        Get the :math:`\mathbb{Z}_2` invariant of an inversion- and
        time-reversal-symmetric model from the parity eigenvalues of its
        occupied bands at the time-reversal-invariant momenta
        :math:`\Gamma_i` (Fu and Kane, Phys. Rev. B 76, 045302 (2007)):

        .. math::

            (-1)^\nu = \prod_i \delta_i\, ,\qquad
            \delta_i = \prod_{m=1}^{N} \xi_{2m}(\Gamma_i)\, ,

        one parity :math:`\xi = \pm1` per Kramers pair. In 3D, :math:`\nu`
        is the strong index.

        :param inversion: Complex ndarray, shape (norb, norb), or callable of
            k returning one: the inversion operator :math:`P`, with
            :math:`P H(\mathbf{k}) P^{-1} = H(-\mathbf{k})` (e.g. swapping the
            two sublattices of graphene, and the identity on spin).
        :param bands: List of band indices (the occupied Kramers pairs).

        :returns:
            * **nu** -- 0 (trivial) or 1 (topological).
        '''
        error_handling.even_bands(bands)
        error_handling.band_indices(bands, self.norb)
        product = 1
        for k in self._trims():
            v = self._subspace(k, bands)
            xi = np.linalg.eigvals(v.conj().T @ self._operator(inversion, k) @ v)
            error_handling.parities(xi)
            n_minus = int(np.sum(xi.real < 0))
            product *= (-1) ** (n_minus // 2)
        return 0 if product == 1 else 1

    def symmetry_error(
        self, op, k_map: str = 'minus', antiunitary: bool = False, anti: bool = False,
        nk: int = 5, seed: int = 0,
    ) -> float:
        r'''
        Check a symmetry of the Bloch Hamiltonian,

        .. math::

            H(g\mathbf{k}) = \pm\, U\, \mathcal{K}H(\mathbf{k})\mathcal{K}^{-1} U^\dagger\, ,

        with :math:`g\mathbf{k} = -\mathbf{k}` or :math:`\mathbf{k}`,
        :math:`\mathcal{K}` complex conjugation if *antiunitary*, and the
        minus sign if *anti*: time reversal is antiunitary with
        :math:`\mathbf{k}\to-\mathbf{k}` and the plus sign, particle-hole
        antiunitary with :math:`\mathbf{k}\to-\mathbf{k}` and the minus sign,
        chiral unitary with :math:`\mathbf{k}\to\mathbf{k}` and the minus sign,
        inversion unitary with :math:`\mathbf{k}\to-\mathbf{k}` and the plus
        sign.

        :param op: Complex ndarray, shape (norb, norb), or callable of k: :math:`U`.
        :param k_map: 'minus' or 'identity'. Default value 'minus'.
        :param antiunitary: Boolean. Default value False.
        :param anti: Boolean. Default value False.
        :param nk: Positive integer. Default value 5. Points per direction of
            the (randomly shifted) test mesh.
        :param seed: Integer. Default value 0. Seed of the random shift.

        :returns:
            * **error** -- Real number. Largest deviation over the mesh (0 for a symmetry).
        '''
        error_handling.k_map(k_map)
        error_handling.boolean(antiunitary, 'antiunitary')
        error_handling.boolean(anti, 'anti')
        error_handling.positive_int(nk, 'nk')
        rng = np.random.default_rng(seed)
        fracs = np.array(np.meshgrid(*[np.arange(nk) / nk] * self.dim, indexing='ij')).reshape(self.dim, -1).T
        fracs = fracs + rng.uniform(0., 1. / nk, self.dim)
        sign = -1. if anti else 1.
        err = 0.
        for f in fracs:
            k = f @ self.rec_vec_k
            gk = -k if k_map == 'minus' else k
            ham = self.get_ham(k)
            u = self._operator(op, k)
            rhs = u @ (ham.conj() if antiunitary else ham) @ u.conj().T
            err = max(err, float(np.max(np.abs(self.get_ham(gk) - sign * rhs))))
        return err

    def tenfold_class(
        self, time_reversal=None, particle_hole=None, chiral=None, nk: int = 5, tol: float = 1e-8,
    ) -> str:
        r'''
        Get the Altland-Zirnbauer symmetry class (the "tenfold way") of the
        model from its time-reversal :math:`T = U_T\mathcal{K}`, particle-hole
        :math:`C = U_C\mathcal{K}` and chiral :math:`S = U_S` symmetries. Each
        given operator is first checked (see *symmetry_error*) to be a
        symmetry; the class then follows from :math:`T^2 = U_TU_T^* = \pm1`,
        :math:`C^2 = \pm1`, and the presence of :math:`S`. Any two of the
        three imply the third (:math:`S = TC`), which need not be given.

        :param time_reversal: Matrix :math:`U_T`, or None (absent).
        :param particle_hole: Matrix :math:`U_C`, or None (absent).
        :param chiral: Matrix :math:`U_S`, or None (absent).
        :param nk: Positive integer. Default value 5. See *symmetry_error*.
        :param tol: Positive real. Default value 1e-8. Tolerance of the checks.

        :returns:
            * **name** -- String: 'A', 'AIII', 'AI', 'BDI', 'D', 'DIII', 'AII', 'CII', 'C' or 'CI'.
        '''
        error_handling.positive_real(tol, 'tol')
        k0 = np.zeros(self.dim)
        for op, k_map, antiu, anti in ((time_reversal, 'minus', True, False),
                                                      (particle_hole, 'minus', True, True),
                                                      (chiral, 'identity', False, True)):
            if op is not None:
                error_handling.is_symmetry(self.symmetry_error(op, k_map, antiu, anti, nk), tol)
        u_t = None if time_reversal is None else self._operator(time_reversal, k0)
        u_c = None if particle_hole is None else self._operator(particle_hole, k0)
        # two of the three imply the third: C = S T, T = S C
        if chiral is not None and (u_t is None) != (u_c is None):
            u_s = self._operator(chiral, k0)
            if u_c is None:
                u_c = u_s @ u_t
            else:
                u_t = u_s @ u_c
        squares = []
        for u in (u_t, u_c):
            if u is None:
                squares.append(0)
                continue
            square = u @ u.conj()
            error_handling.square(square, tol)
            squares.append(int(np.sign(square[0, 0].real)))
        has_s = chiral is not None or (u_t is not None and u_c is not None)
        table = {(0, 0, False): 'A', (0, 0, True): 'AIII', (1, 0, False): 'AI',
                     (1, 1, True): 'BDI', (0, 1, False): 'D', (-1, 1, True): 'DIII',
                     (-1, 0, False): 'AII', (-1, -1, True): 'CII', (0, -1, False): 'C',
                     (1, -1, True): 'CI'}
        return table[(squares[0], squares[1], has_s)]

    def quantum_geometric_tensor(
        self, bands: int | list[int], k: ArrayLike, dk: float = 1e-4, positions: bool = True,
    ) -> NDArray[np.complex128]:
        r'''
        Get the quantum geometric tensor of a group of bands at :math:`\mathbf{k}`,

        .. math::

            Q_{\mu\nu} = \mathrm{Tr}\left[P\,\partial_\mu P\,\partial_\nu P\right]
                       = \sum_{n\in\mathrm{bands}}\langle\partial_\mu u_n|(1-P)|\partial_\nu u_n\rangle\, ,

        with :math:`P(\mathbf{k})` the projector on the bands (gauge
        invariant, so degeneracies within the group are harmless). Its real
        part is the quantum metric :math:`g_{\mu\nu}` (Provost and Vallee,
        1980), minus twice its imaginary part the Berry curvature
        :math:`\Omega_{\mu\nu} = -2\,\mathrm{Im}\,Q_{\mu\nu}` (the convention of
        *berry_curvature*, per unit k-area here). They obey
        :math:`\sqrt{\det g} \ge |\Omega_{xy}|/2` in 2D.

        :param bands: Band index, or list of band indices.
        :param k: k point (see *get_ham*).
        :param dk: Positive real. Default value 1e-4. Finite-difference step.
        :param positions: Boolean. Default value True. Include the orbital
            positions in the Bloch functions. The local metric and curvature
            depend on this choice (the Berry curvature of *berry_curvature* is
            that of ``positions=False``); integrals such as the Chern number
            do not.

        :returns:
            * **Q** -- Complex ndarray, shape (dim, dim), in the k coordinates of *get_ham*.
        '''
        if isinstance(bands, int):
            bands = [bands]
        error_handling.band_indices(bands, self.norb)
        error_handling.k_vector(k, 'k', self.dim)
        error_handling.positive_real(dk, 'dk')
        error_handling.boolean(positions, 'positions')
        k = np.asarray(k, dtype='f8')
        tau = self.orbital_positions()

        def proj(kk):
            v = self._subspace(kk, bands)
            if positions:
                v = np.exp(-1j * tau @ (self.k_basis @ kk))[:, None] * v
            return v @ v.conj().T

        p0 = proj(k)
        dp = []
        for mu in range(self.dim):
            e = np.zeros(self.dim)
            e[mu] = dk
            dp.append((proj(k + e) - proj(k - e)) / (2 * dk))
        return np.array([[np.trace(p0 @ dp[mu] @ dp[nu]) for nu in range(self.dim)]
                               for mu in range(self.dim)])

    def plot_dos(
        self,
        nk: int | tuple[int, int] = 30,
        broadening: float = 0.05,
        kernel: str = 'gaussian',
        e_grid: ArrayLike | None = None,
        fs: float = 20,
        lw: float = 2.,
        figsize: tuple[float, float] | None = None,
    ) -> Figure:
        '''
        Plot the (broadened) density of states, obtained by diagonalizing
        :math:`H(\\mathbf{k})` over a uniform Brillouin-zone mesh -- see
        *tbkit.dos.density_of_states*.

        :param nk: Positive integer, or tuple of *dim* positive integers.
            Default value 30. Number of k-points along each reciprocal
            lattice vector.
        :param broadening: Positive real number. Default value 0.05. Kernel width.
        :param kernel: String. Default value 'gaussian'. 'gaussian' or 'lorentzian'.
        :param e_grid: Real ndarray. Default value None. Energies at which to
            evaluate the density of states.
        :param fs: Positive number. Default value 20. Fontsize.
        :param lw: Positive number. Default value 2. Linewidth.
        :param figsize: Tuple. Default value None. Figure size.

        :returns:
            * **fig** -- Figure.
        '''
        error_handling.positive_real(fs, 'fs')
        error_handling.positive_real(lw, 'lw')
        error_handling.tuple_2elem(figsize, 'figsize')
        en = self.mesh_bands(nk)
        return dos._plot_density_of_states(en, e_grid, broadening, kernel, fs, lw, figsize)

    def plot_bands(
        self,
        node_labels: list[str] | None = None,
        fs: float = 20,
        lw: float = 2.,
        ms: float = 0.,
        c: str = 'b',
        lims: tuple[float, float] | None = None,
        figsize: tuple[float, float] | None = None,
    ) -> Figure:
        '''
        Plot the band structure computed by *k_path* or *get_bands*, against
        the cumulative distance along the k-points (vertical lines mark the
        nodes of a *k_path*). For a non-Hermitian model, the real part of
        the energies is plotted.

        :param node_labels: List of strings. Default value None. Labels of the
            high-symmetry points passed to *k_path*.
        :param fs: Positive number. Default value 20. Fontsize.
        :param lw: Positive number. Default value 2. Linewidth.
        :param ms: Positive number. Default value 0. Marker size.
        :param c: Default value 'b'. Line color.
        :param lims: List. Default value None. Energy plot limits.
        :param figsize: Tuple. Default value None. Figure size.

        :returns:
            * **fig** -- Figure.
        '''
        error_handling.empty_ndarray(self.en, 'get_bands or k_path')
        error_handling.positive_real(fs, 'fs')
        error_handling.positive_real(lw, 'lw')
        error_handling.lims(lims)
        error_handling.tuple_2elem(figsize, 'figsize')
        fig, ax = plt.subplots(figsize=figsize)
        for n in range(self.norb):
            ax.plot(self.ks_dist, self.en[:, n].real, c=c, lw=lw, marker='o', ms=ms)
        for node in self.nodes:
            ax.axvline(node, color='k', lw=0.5)
        ax.set_xlim([self.ks_dist[0], self.ks_dist[-1]])
        if lims is not None:
            ax.set_ylim(lims)
        if node_labels is not None:
            error_handling.ndarray(np.array(node_labels), 'node_labels', len(self.nodes))
            ax.set_xticks(self.nodes)
            ax.set_xticklabels(node_labels, fontsize=fs)
        ax.set_ylabel('$E$', fontsize=fs)
        for label in ax.yaxis.get_majorticklabels():
            label.set_fontsize(fs)
        fig.set_layout_engine('tight')
        plt.draw()
        return fig

    def show(self) -> None:
        '''
        Emulate Matplotlib method plt.show().
        '''
        plt.show()


def ribbon(
    lat: Lattice,
    list_hop: list[dict],
    width: int,
    direction: int = 1,
    onsite: dict | None = None,
    spin: bool = False,
) -> KSpace:
    r'''
    Cut a ribbon out of a 2D periodic model: periodic along one primitive
    vector, finite (open boundary, *width* unit cells) along the other.
    This is the standard way to see edge states in a band structure (e.g.
    the zero-energy edge band of a zigzag graphene ribbon, or the helical
    edge states of a Kane-Mele ribbon). Cut out of a 3D model, it gives a
    slab: periodic along the two other primitive vectors, the way to see
    surface states (e.g. the Fermi arcs of a Weyl semimetal).

    :param lat: **Lattice** class instance (2D or 3D, i.e. two or three primitive
        vectors). Only *unit_cell* and *prim_vec* are used.
    :param list_hop: List of dictionaries, in the same format passed to
        *KSpace.set_hopping* -- the hoppings of the periodic (2D) model
        that the ribbon is cut from.
    :param width: Positive integer. Number of unit cells across the ribbon.
    :param direction: 0 or 1 (or 2, in 3D). Default value 1. Which primitive
        vector (``lat.prim_vec[direction]``) becomes finite; the others stay
        periodic.
    :param onsite: Dictionary. Default value None. Onsite energies, in the
        same format passed to *KSpace.set_onsite* -- applied identically
        on every row of the ribbon.
    :param spin: Boolean. Default value False. See *KSpace*.

    :returns:
        * **rib** -- **KSpace** instance, 1D-periodic (2D-periodic for a slab), with
          ``width * len(lat.unit_cell)`` sites (each site of *lat*,
          repeated once per row across the ribbon; row *w*'s copy of site
          *i* is orbital ``w*len(lat.unit_cell) + i``).

    Example usage::

        # zigzag graphene ribbon, 20 unit cells wide
        list_hop = [{'i': 0, 'j': 1, 'R': (0, 0), 't': 1.},
                          {'i': 0, 'j': 1, 'R': (-1, 0), 't': 1.},
                          {'i': 0, 'j': 1, 'R': (0, -1), 't': 1.}]
        rib = ribbon(lat, list_hop, width=20)
    '''
    error_handling.lat(lat)
    error_handling.dim_min(len(lat.prim_vec), 2)
    error_handling.positive_int(width, 'width')
    error_handling.direction(direction, len(lat.prim_vec))
    periodic = [d for d in range(len(lat.prim_vec)) if d != direction]
    n_sites = len(lat.unit_cell)
    a_dir = np.array(lat.prim_vec[direction])
    new_unit_cell = []
    for w in range(width):
        for dic in lat.unit_cell:
            r0 = np.array(dic['r0']) + w*a_dir
            new_unit_cell.append({'tag': dic['tag'], 'r0': tuple(float(c) for c in r0)})
    new_lat = Lattice(unit_cell=new_unit_cell, prim_vec=[lat.prim_vec[d] for d in periodic])
    rib = KSpace(new_lat, spin=spin)
    new_list_hop = []
    for dic in list_hop:
        w2_shift = dic['R'][direction]
        n_periodic = tuple(dic['R'][d] for d in periodic)
        for w in range(width):
            w2 = w + w2_shift
            if 0 <= w2 < width:
                new_list_hop.append({'i': w*n_sites + dic['i'],
                                                    'j': w2*n_sites + dic['j'],
                                                    'R': n_periodic,
                                                    't': dic['t']})
    rib.set_hopping(new_list_hop)
    if onsite is not None:
        rib.set_onsite(onsite)
    return rib


def magnetic_supercell(
    lat: Lattice,
    list_hop: list[dict],
    p: int,
    q: int,
    onsite: dict | None = None,
    spin: bool = False,
) -> KSpace:
    r'''
    Thread a uniform magnetic field through a 2D periodic model, with a
    rational flux :math:`\alpha = p/q` flux quanta per unit cell, and return
    the Bloch Hamiltonian of its magnetic unit cell: *q* cells along
    :math:`\mathbf{a}_1`. Each hopping gets its Peierls phase (see
    *System.set_peierls_phase*)

    .. math::

        \phi_{ij} = \frac{2\pi}{\Phi_0}\int_{\mathbf{r}_i}^{\mathbf{r}_j}\mathbf{A}\cdot d\mathbf{l}
                  = 2\pi\alpha\,\frac{s_{1,i}+s_{1,j}}{2}\,(s_{2,j}-s_{2,i})

    in the gauge :math:`\mathbf{A} = \alpha\Phi_0\, s_1\nabla s_2`, with
    :math:`(s_1, s_2)` the fractional coordinates of a point along
    :math:`(\mathbf{a}_1, \mathbf{a}_2)`, periodic along :math:`\mathbf{a}_2`.
    A translation by :math:`q\mathbf{a}_1` is a symmetry only up to the
    gauge transformation :math:`e^{-2\pi i p s_2}`, which the hoppings
    crossing the magnetic cell boundary carry. The bands of the result are
    the Hofstadter subbands, and their *chern_number* the TKNN integers.

    :param lat: **Lattice** class instance, 2D (two primitive vectors in the plane).
    :param list_hop: List of dictionaries, the hoppings of the field-free
        model, in the format of *KSpace.set_hopping*.
    :param p: Integer. Numerator of the flux per unit cell.
    :param q: Positive integer. Denominator of the flux per unit cell: the
        magnetic cell has *q* unit cells.
    :param onsite: Dictionary. Default value None. Onsite energies, in the
        format of *KSpace.set_onsite*, applied to every copy.
    :param spin: Boolean. Default value False. See *KSpace*.

    :returns:
        * **mag** -- **KSpace** instance, with ``q * len(lat.unit_cell)``
          sites: copy *m* of site *i* is site ``m*len(lat.unit_cell) + i``,
          displaced by :math:`m\mathbf{a}_1`.

    Example usage::

        # square lattice, flux 1/3 per plaquette: three Hofstadter bands
        mag = magnetic_supercell(lattices.square(), list_hop, p=1, q=3)
        chern = [mag.chern_number(bands=[n]) for n in range(3)]
    '''
    error_handling.lat(lat)
    error_handling.planar_cell(lat.prim_vec)
    error_handling.set_hopping_kspace(list_hop, len(lat.unit_cell), 2, spin)
    error_handling.integer(p, 'p')
    error_handling.positive_int(q, 'q')
    alpha = p / q
    n_sites = len(lat.unit_cell)
    a = np.array(lat.prim_vec, dtype='f8')
    frac = np.linalg.solve(a.T, np.array([dic['r0'] for dic in lat.unit_cell], dtype='f8').T).T
    new_unit_cell = []
    for m in range(q):
        for dic in lat.unit_cell:
            r0 = np.array(dic['r0'], dtype='f8') + m * a[0]
            new_unit_cell.append({'tag': dic['tag'], 'r0': (float(r0[0]), float(r0[1]))})
    new_lat = Lattice(unit_cell=new_unit_cell, prim_vec=[tuple(q * a[0]), lat.prim_vec[1]])
    mag = KSpace(new_lat, spin=spin)
    new_list_hop = []
    for dic in list_hop:
        n1, n2 = dic['R']
        for m in range(q):
            s1_i, s2_i = m + frac[dic['i'], 0], frac[dic['i'], 1]
            s1_j, s2_j = m + n1 + frac[dic['j'], 0], n2 + frac[dic['j'], 1]
            big, m_j = divmod(m + n1, q)
            phase = 2 * PI * alpha * 0.5 * (s1_i + s1_j) * (s2_j - s2_i) - 2 * PI * p * big * s2_j
            new_list_hop.append({'i': m * n_sites + dic['i'], 'j': m_j * n_sites + dic['j'],
                                                'R': (big, n2), 't': dic['t'] * np.exp(1j * phase)})
    mag.set_hopping(new_list_hop)
    if onsite is not None:
        mag.set_onsite(onsite)
    return mag
