from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from itertools import product
from typing import Literal, Sequence, overload

import numpy as np
from numpy.typing import ArrayLike, NDArray
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.colors import LogNorm, Normalize
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import scipy.linalg as LA
import scipy.sparse as sp
from scipy.integrate import cumulative_trapezoid
from scipy.special import expit, spence
import tbkit.error_handling as error_handling
import tbkit.dos as dos
import tbkit.occupation as occupation
import tbkit.neighbours as neighbours
import tbkit.values as values
from tbkit.lattice import Lattice
from tbkit.transport import _green_surface_bulk


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


def reciprocal_vectors(prim_vec: Sequence[tuple[float, ...]]) -> list[tuple[float, ...]]:
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

    workers = 1  # threads diagonalizing over k, see set_workers

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
        self._hop_const: list = []  # list of (i, j, R_cartesian (np.ndarray), t)
        # value functions (see tbkit.values): (i, j, R_cartesian, function,
        # hermitian), with arrays of sites i, j and bond vectors R
        self._hop_values: list = []
        #: Parameters of the value functions (see *set_params*).
        self.params: dict = {}
        self._nonreciprocal = False  # set_hopping(hermitian=False) was used
        self._overlap_hop: list = []  # overlaps, as _hop: list of (i, j, R_cartesian, s)
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

    @property
    def _hop(self) -> list:
        '''
        Private. The hoppings, as a list of (i, j, R_cartesian, t): those
        given as numbers, then the value functions evaluated with *params*.
        Assigning it replaces both.
        '''
        return self._hops(self.params)

    @_hop.setter
    def _hop(self, hops: list) -> None:
        self._hop_const = list(hops)
        self._hop_values = []

    def _hops(self, params: dict) -> list:
        '''
        Private method. *_hop*, with the value functions evaluated with *params*.
        '''
        hops = list(self._hop_const)
        if not self._hop_values:
            return hops
        tau = np.array([dic['r0'] for dic in self.lat.unit_cell], dtype='f8').reshape(self.n_sites, -1)
        for i, j, R, func, hermitian in self._hop_values:
            site_i = values.sites(tau[i], self.tags[i], i)
            site_j = values.sites(tau[j] + R, self.tags[j], j)
            t = values.evaluate(func, (site_i, site_j), params, len(i), self.spin)
            if self.spin:
                pairs = [(2*ii + a, 2*jj + b, RR, tt[a, b], np.conj(tt[a, b]))
                            for ii, jj, RR, tt in zip(i, j, R, t) for a in range(2) for b in range(2)]
            else:
                pairs = [(ii, jj, RR, tt, np.conj(tt)) for ii, jj, RR, tt in zip(i, j, R, t)]
            for ii, jj, RR, tt, tc in pairs:
                hops.append((int(ii), int(jj), RR, tt))
                if hermitian:
                    hops.append((int(jj), int(ii), -RR, tc))
        return hops

    def set_params(self, **params) -> None:
        '''
        Set the parameters of the value functions of *set_hopping*: every
        method (*get_bands*, *chern_number*, *finite_ham*, ...) then uses
        these values. *get_ham* also takes them per call.

        :param params: Parameter values, added to (or replacing those of) *params*.

        Example usage::

            ks.set_params(m=0.5)
            ks.chern_number()
        '''
        self.params.update(params)

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
              Or a value function ``t(site_i, site_j, **params)``,
              evaluated by ``get_ham(k, **params)`` (and by every other
              method, with the parameters of *set_params*): *site_i* and
              *site_j* are structured arrays of the bonds' end points,
              *site_i* in the home cell and *site_j* in cell
              :math:`\mathbf{R}` (fields 'x', 'y', ('z'), 'tag', 'index' --
              the site index within *unit_cell*). It is called once with
              every bond of the dictionaries that share it, and returns one
              value, or one per bond (or, if ``spin=True``, a 2x2 matrix,
              or one per bond).

            *list_hop* may instead use the neighbour-order form of
            *System.set_hopping*: dictionaries with keys ('n', 't') and
            optionally 'ang' and/or 'tag' -- 'n' the neighbour order (1st,
            2nd, ... shortest distance between orbitals of the infinite
            lattice), 'ang' a bond angle in degrees, 'tag' a sublattice pair
            such as ``'ab'``. The bonds are found from *unit_cell* and
            *prim_vec* (in 1D, 2D and 3D) and turned into the explicit form
            by *tbkit.neighbours.neighbour_hoppings* (see there for the bond
            orientations; a negative angle addresses the reversed bonds).
            A list uses one form or the other, not both.
        :param hermitian: Boolean. Default value True. If False, the Hermitian
            conjugates are *not* added: each dictionary is one matrix element
            :math:`H_{ij}(\mathbf{R})` only, so that non-reciprocal
            (non-Hermitian) hoppings can be built -- give the reverse hopping
            explicitly, with its own amplitude.

        Example usage::

            # graphene, by neighbour order: nearest (t) and next-nearest
            # (t2) neighbours, as in System.set_hopping
            gra.set_hopping([{'n': 1, 't': t}, {'n': 2, 't': t2}])
            # square lattice, anisotropic nearest neighbours
            sq.set_hopping([{'n': 1, 'ang': 0., 't': tx}, {'n': 1, 'ang': 90., 't': ty}])

            # 1D chain, nearest-neighbor hopping t between the only orbital
            # and its right neighbor:
            chain.set_hopping([{'i': 0, 'j': 0, 'R': (1,), 't': 1.}])
            # spinful: spin-independent hopping t, plus a Rashba-like
            # spin-flip term of strength alpha:
            chain_spin.set_hopping([{'i': 0, 'j': 0, 'R': (1,),
                                                    't': t*PAULI['0'] + 1j*alpha*PAULI['y']}])
            # value function: a staggered hopping, swept without rebuilding
            chain.set_hopping([{'n': 1, 't': lambda si, sj, t, d: t + d * (-1) ** si['index']}])
            ham = chain.get_ham((0.,), t=1., d=0.2)
        '''
        error_handling.boolean(hermitian, 'hermitian')
        if error_handling.hopping_form(list_hop):
            list_hop = neighbours.neighbour_hoppings(self.lat, list_hop, self.spin, hermitian)
        error_handling.set_hopping_kspace(list_hop, self.n_sites, self.dim, self.spin, values=True)
        if not hermitian:
            self._nonreciprocal = True
        funcs: dict = {}  # the bonds of each value function: id -> (function, [(i, j, R_cart)])
        for dic in list_hop:
            R_cart = np.zeros(self.space_dim)
            for n, vec in zip(dic['R'], self.lat.prim_vec):
                R_cart += n * np.array(vec)
            i, j, t = dic['i'], dic['j'], dic['t']
            if callable(t):
                funcs.setdefault(id(t), (t, []))[1].append((i, j, R_cart))
            elif self.spin:
                block = t*PAULI['0'] if np.ndim(t) == 0 else np.asarray(t, 'c16')
                for a in range(2):
                    for b in range(2):
                        self._hop_const.append((2*i+a, 2*j+b, R_cart, block[a, b]))
                        if hermitian:
                            self._hop_const.append((2*j+b, 2*i+a, -R_cart, np.conj(block[a, b])))
            else:
                self._hop_const.append((i, j, R_cart, t))
                if hermitian:
                    self._hop_const.append((j, i, -R_cart, np.conj(t)))
        for func, bonds in funcs.values():
            i, j, R = zip(*bonds)
            self._hop_values.append((np.array(i), np.array(j), np.array(R), func, hermitian))

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

        :param list_hop: List of dictionaries ('i', 'j', 'R', 't'), or by
            neighbour order ('n', 't', optionally 'ang', 'tag'), see
            *set_hopping* (numbers only, no value functions).
        '''
        if error_handling.hopping_form(list_hop):
            list_hop = neighbours.neighbour_hoppings(self.lat, list_hop, self.spin)
        error_handling.set_hopping_kspace(list_hop, self.n_sites, self.dim, self.spin)
        saved = self._hop_const, self._hop_values, self._nonreciprocal
        self._hop = []
        self.set_hopping(list_hop)
        self._overlap_hop += self._hop_const
        self._hop_const, self._hop_values, self._nonreciprocal = saved

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

    def get_ham(self, k: ArrayLike, **params) -> NDArray[np.complex128]:
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
        :param params: Values of the parameters of the value functions of
            *set_hopping*, for this call only: they override those of
            *set_params*. Each function gets the ones its signature names
            (all of them with ``**kwargs``); the others are ignored.

        :returns:
            * **ham** -- Complex ndarray, shape (norb, norb).

        Example usage::

            hams = [ks.get_ham((0., 0.), m=m) for m in np.linspace(-1., 1., 21)]
        '''
        error_handling.k_vector(k, 'k', self.dim)
        return self._bloch_ham((self.k_basis @ np.asarray(k, dtype='f8'))[None],
                                          params={**self.params, **params})[0]

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

    def _bloch_ham(
        self, k_cart: NDArray[np.float64], A: NDArray[np.float64] | None = None, params: dict | None = None,
    ) -> NDArray[np.complex128]:
        '''
        Private method. *get_ham* (or, with a vector potential *A*,
        *get_ham_peierls*) at many k-points at once: *k_cart* has shape
        (nk, space_dim) and the result (nk, norb, norb). The value functions
        take *params* (default: *self.params*).
        '''
        hops = self._hops(self.params if params is None else params)
        return self._bloch_sum(hops, k_cart, A=A)[0] + (np.diag(self.onsite) + self._onsite_offdiag)[None]

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
        self.en = out[0] if isinstance(out, tuple) else out
        steps = np.linalg.norm(np.diff(ks, axis=0), axis=1)
        self.ks_dist = np.concatenate([[0.], np.cumsum(steps)])[:len(ks)]
        self.nodes = np.array([])
        return out

    @overload
    def _diagonalize(
        self, ks: NDArray[np.float64], eigenvec: Literal[False] = ...,
    ) -> NDArray[np.float64]: ...
    @overload
    def _diagonalize(
        self, ks: NDArray[np.float64], eigenvec: Literal[True],
    ) -> tuple[NDArray[np.float64], NDArray[np.complex128]]: ...
    @overload
    def _diagonalize(
        self, ks: NDArray[np.float64], eigenvec: bool,
    ) -> NDArray[np.float64] | tuple[NDArray[np.float64], NDArray[np.complex128]]: ...

    def _diagonalize(
        self, ks: NDArray[np.float64], eigenvec: bool = False,
    ) -> NDArray[np.float64] | tuple[NDArray[np.float64], NDArray[np.complex128]]:
        r'''
        Private method. Diagonalize :math:`H(\mathbf{k})` over the k-points
        *ks* (shape (nk, dim)), without touching the stored band structure.
        '''
        return self._eigs(ks, eigenvec)

    def set_workers(self, workers: int) -> None:
        r'''
        Set the number of threads that diagonalize :math:`H(\mathbf{k})`
        over many k-points (*get_bands*, *mesh_bands*, *berry_curvature*,
        the Wilson loops, ...). The k-points are split into chunks, each
        diagonalized by one stacked LAPACK call, which releases the GIL, so
        the chunks run in parallel. The results depend on *workers* only up
        to rounding (and the phases of the eigenvectors).

        :param workers: Positive integer. Number of threads (default 1, no threads).

        Example usage::

            ks.set_workers(4)
            curv = ks.berry_curvature(0, nk=400)
        '''
        error_handling.positive_int(workers, 'workers')
        self.workers = workers

    def _hams(self, ks: NDArray[np.float64]) -> NDArray[np.complex128]:
        '''
        Private method. *get_ham* at the k-points *ks* (shape (nk, dim)),
        shape (nk, norb, norb): one Bloch sum, unless a subclass overrides
        *get_ham* (a Floquet model), which is then called per k-point.
        '''
        if type(self).get_ham is KSpace.get_ham:
            return self._bloch_ham(ks @ self.k_basis.T)
        return np.array([self.get_ham(k) for k in ks], dtype='c16')

    def _overlaps(self, ks: NDArray[np.float64]) -> NDArray[np.complex128]:
        '''
        Private method. *get_overlap* at the k-points *ks*, shape (nk, norb, norb).
        '''
        return np.eye(self.norb) + self._bloch_sum(self._overlap_hop, ks @ self.k_basis.T)[0]

    @overload
    def _eigs(self, ks: NDArray[np.float64], eigenvec: Literal[False] = ...) -> NDArray: ...
    @overload
    def _eigs(
        self, ks: NDArray[np.float64], eigenvec: Literal[True],
    ) -> tuple[NDArray, NDArray[np.complex128]]: ...
    @overload
    def _eigs(
        self, ks: NDArray[np.float64], eigenvec: bool,
    ) -> NDArray | tuple[NDArray, NDArray[np.complex128]]: ...

    def _eigs(
        self, ks: NDArray[np.float64], eigenvec: bool = False,
    ) -> NDArray | tuple[NDArray, NDArray[np.complex128]]:
        '''
        Private method. *_eig* at the k-points *ks* (shape (nk, dim)): the
        eigenvalues, shape (nk, norb), and the eigenvectors, shape
        (nk, norb, norb). The k-points go in chunks of bounded memory,
        spread over *workers* threads (see *set_workers*).
        '''
        chunk = max(1, min(400000 // self.norb ** 2, -(-len(ks) // self.workers)))
        parts = [ks[c0:c0 + chunk] for c0 in range(0, len(ks), chunk)] or [ks]
        if self.workers > 1 and len(parts) > 1:
            with ThreadPoolExecutor(self.workers) as pool:
                out = list(pool.map(lambda part: self._eig_chunk(part, eigenvec), parts))
        else:
            out = [self._eig_chunk(part, eigenvec) for part in parts]
        if eigenvec:
            return np.concatenate([o[0] for o in out]), np.concatenate([o[1] for o in out])
        return np.concatenate(out)

    def _eig_chunk(
        self, ks: NDArray[np.float64], eigenvec: bool,
    ) -> NDArray | tuple[NDArray, NDArray[np.complex128]]:
        r'''
        Private method. One chunk of *_eigs*: the Hermitian solver when
        *is_hermitian*, otherwise the general one, with the eigenvalues
        sorted by real part, both stacked over the k-points. With an overlap
        (see *set_overlap*), the generalized problem :math:`Hv = ES(\mathbf{k})v`:
        the Hermitian one is reduced by the Cholesky factor
        :math:`S = LL^\dagger` to :math:`L^{-1}HL^{-\dagger}` (eigenvectors
        :math:`L^{-\dagger}v`, normalized to :math:`v^\dagger Sv = 1` as by
        *scipy.linalg.eigh*); the non-Hermitian one has no stacked solver,
        and is solved one k-point at a time.
        '''
        ham = self._hams(ks)
        hermitian = self.is_hermitian()
        en: NDArray
        vn: NDArray
        if self._overlap_hop and hermitian:
            l_inv = np.linalg.inv(np.linalg.cholesky(self._overlaps(ks)))
            l_inv_h = l_inv.conj().transpose(0, 2, 1)
            ham = l_inv @ ham @ l_inv_h
        if hermitian:
            if not eigenvec:
                return np.linalg.eigvalsh(ham)
            en, vn = np.linalg.eigh(ham)
            return en, (l_inv_h @ vn if self._overlap_hop else vn)
        if self._overlap_hop:
            solved = [LA.eig(h, s) for h, s in zip(ham, self._overlaps(ks))]
            en = np.array([w for w, _ in solved], dtype='c16').reshape(len(ks), self.norb)
            vn = np.array([v for _, v in solved], dtype='c16').reshape(ham.shape)
        elif eigenvec:
            en, vn = np.linalg.eig(ham)
        else:
            en = np.linalg.eigvals(ham)
        ind = np.argsort(en.real, axis=1, kind='stable')
        if not eigenvec:
            return np.take_along_axis(en, ind, axis=1)
        return np.take_along_axis(en, ind, axis=1), np.take_along_axis(vn, ind[:, None, :], axis=2)

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

    def k_path(
        self, points: list[ArrayLike], nk: int,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        r'''
        Build a k-path through a list of high-symmetry points, and get the
        associated bands.

        :param points: List of at least two k-points (each a tuple/list of
            *dim* real numbers), e.g. from *high_symmetry_path*.
        :param nk: Positive integer. Number of k-points per path segment.

        :returns:
            * **ks_dist** -- Real ndarray. Cumulative distance along the path,
              to be used as the x-axis of a band-structure plot.
            * **en** -- Real ndarray, shape (len(ks_dist), norb). Band energies.
        '''
        error_handling.k_path_points(points, self.dim)
        error_handling.positive_int(nk, 'nk')
        ks, _, nodes = _path(points, nk)
        self.get_bands(ks)
        self.nodes = np.array(nodes)
        return self.ks_dist, self.en

    def mesh_grid(
        self, nk: int | tuple[int, ...],
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

    def mesh_bands(self, nk: int | tuple[int, ...]) -> NDArray[np.float64]:
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
            en = self._eigs(ks.reshape(-1, self.dim))
            error_handling.line_gap(en.real, bands, 'real')
            error_handling.band_continuity(en.reshape(n1, n2, self.norb), bands)
        v1 = self._subspaces(ks.reshape(-1, self.dim), bands).reshape(n1, n2, self.norb, len(bands))
        return self._flux(v1, plane)

    def _flux(self, v1: NDArray[np.complex128], plane: tuple[int, int]) -> NDArray[np.float64]:
        '''
        Private method. Fukui-Hatsugai-Suzuki flux through each plaquette of
        the mesh of *_plane_mesh*, from orthonormal bases *v1* of a subspace
        at its nodes (shape (n1, n2, norb, n)), oriented as in *berry_curvature*.
        '''
        # the other corners of each plaquette: k + b_i/n1, k + b_i/n1 + b_j/n2, k + b_j/n2
        v2 = np.roll(v1, -1, axis=0)
        v3 = np.roll(v2, -1, axis=1)
        v4 = np.roll(v1, -1, axis=1)

        def overlap(a, b):
            return np.linalg.det(a.conj().swapaxes(-1, -2) @ b)
        link = overlap(v1, v2) * overlap(v2, v3) * overlap(v3, v4) * overlap(v4, v1)
        curv = -np.angle(link)
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
        return self._subspaces(np.asarray(k, dtype='f8')[None], bands)[0]

    def _subspaces(self, ks: NDArray[np.float64], bands: list[int]) -> NDArray[np.complex128]:
        '''
        Private method. *_subspace* at the k-points *ks* (shape (nk, dim)),
        shape (nk, norb, len(bands)). S^(1/2) comes from the eigenvalues of
        the (Hermitian, positive) overlap.
        '''
        _, vn = self._eigs(ks, eigenvec=True)
        if self._overlap_hop:
            sv, us = np.linalg.eigh(self._overlaps(ks))
            vn = (us * np.sqrt(sv)[:, None, :]) @ (us.conj().transpose(0, 2, 1) @ vn)
        return vn[:, :, bands]

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

    def sector_chern_numbers(
        self, op, bands: int | list[int], nk: int | tuple[int, int] = 30,
        plane: tuple[int, int] = (0, 1), k_fixed: float = 0.,
    ) -> tuple[float, float]:
        r'''
        Get the Chern numbers of the two sectors into which a Hermitian
        operator :math:`O` splits a group of bands: at every k-point, the
        projected operator :math:`P(\mathbf{k})\,O\,P(\mathbf{k})`
        (:math:`P` the projector on *bands*) is diagonalized within the
        bands, and its eigenvectors with positive and negative eigenvalues
        span the two sectors, whose Chern numbers :math:`C_\pm` follow as in
        *berry_curvature*. They are well defined as long as the spectrum of
        :math:`POP` keeps a gap around zero over the whole plane, even if
        :math:`O` does not commute with :math:`H` (Prodan, Phys. Rev. B 80,
        125327 (2009)); otherwise a ValueError is raised. See
        *spin_chern_number* and *mirror_chern_number*.

        :param op: Complex ndarray, shape (norb, norb), or callable of k
            returning one: the Hermitian operator :math:`O`.
        :param bands: Band index, or list of band indices.
        :param nk: Positive integer, or tuple of 2 positive integers. Default value 30.
        :param plane: Tuple of two integers. Default value (0, 1). 3D only, see *berry_curvature*.
        :param k_fixed: Real number. Default value 0. 3D only, see *berry_curvature*.

        :returns:
            * **c_plus** -- Real number, close to an integer: Chern number
              of the sector with :math:`POP > 0`.
            * **c_minus** -- Real number: that of the sector with :math:`POP < 0`.
              :math:`C_+ + C_-` is the Chern number of *bands*.
        '''
        bands, nk, ks = self._check_sectors(bands, nk, plane, k_fixed)
        if callable(op):
            ops = np.array([self._operator(op, k) for k in ks])
        else:
            ops = self._operator(op, ks[0])[None]
        error_handling.hermitian_operator(ops)
        return self._sector_cherns(ops, bands, nk, ks, plane)

    def _check_sectors(
        self, bands, nk, plane, k_fixed,
    ) -> tuple[list[int], tuple[int, int], NDArray[np.float64]]:
        '''
        Private method. Validate the arguments of the sector Chern numbers;
        return them with the k-points of the plane mesh, shape (n1 n2, dim).
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
        error_handling.hermitian_kspace(self.is_hermitian(), self._overlap_hop)
        return bands, nk, self._plane_mesh(nk, plane, k_fixed).reshape(-1, self.dim)

    def _sector_cherns(
        self, ops: NDArray[np.complex128], bands: list[int], nk: tuple[int, int],
        ks: NDArray[np.float64], plane: tuple[int, int],
    ) -> tuple[float, float]:
        '''
        Private method. *sector_chern_numbers* with the Hermitian operator
        evaluated at the k-points *ks* (shape (n1 n2, norb, norb), or
        (1, norb, norb) if constant).
        '''
        n1, n2 = nk
        v = self._subspaces(ks, bands)
        w, u = np.linalg.eigh(v.conj().transpose(0, 2, 1) @ ops @ v)
        n_minus = error_handling.projected_gap(w)
        sectors = v @ u
        c_minus = self._flux(sectors[:, :, :n_minus].reshape(n1, n2, self.norb, -1), plane)
        c_plus = self._flux(sectors[:, :, n_minus:].reshape(n1, n2, self.norb, -1), plane)
        return float(c_plus.sum() / (2*PI)), float(c_minus.sum() / (2*PI))

    def spin_chern_number(
        self, bands: list[int], nk: int | tuple[int, int] = 30, s_z=None,
        plane: tuple[int, int] = (0, 1), k_fixed: float = 0.,
    ) -> float:
        r'''
        Get the spin Chern number of a group of bands (Prodan, Phys. Rev. B
        80, 125327 (2009); Sheng, Weng, Sheng and Haldane, Phys. Rev. Lett.
        97, 036808 (2006)),

        .. math::

            C_s = \frac{C_+ - C_-}{2}\, ,

        with :math:`C_\pm` the Chern numbers of the two sectors of the
        projected spin operator :math:`Ps_zP` (see *sector_chern_numbers*).
        When :math:`s_z` is conserved they are the Chern numbers of the two
        spins; Rashba coupling mixes the spins, but :math:`C_s` stays
        quantized, and unchanged, as long as the spectrum of :math:`Ps_zP`
        keeps a gap around zero. With time reversal, :math:`C_s` modulo 2 is
        the :math:`\mathbb{Z}_2` invariant of *z2_invariant*.

        :param bands: List of band indices (e.g. the occupied bands).
        :param nk: Positive integer, or tuple of 2 positive integers. Default value 30.
        :param s_z: Complex ndarray, shape (norb, norb), or callable of k.
            Default value None: :math:`\sigma_z` on every site of a model
            built with ``spin=True``. The spin operator.
        :param plane: Tuple of two integers. Default value (0, 1). 3D only, see *berry_curvature*.
        :param k_fixed: Real number. Default value 0. 3D only, see *berry_curvature*.

        :returns:
            * **c_s** -- Real number, close to an integer.
        '''
        if s_z is None:
            error_handling.spinful(self.spin)
            s_z = np.kron(np.eye(self.norb // 2), PAULI['z'])
        c_plus, c_minus = self.sector_chern_numbers(s_z, bands, nk, plane, k_fixed)
        return (c_plus - c_minus) / 2

    def mirror_chern_number(
        self, mirror, bands: list[int], nk: int | tuple[int, int] = 30,
        plane: tuple[int, int] = (0, 1), k_fixed: float = 0., tol: float = 1e-8,
    ) -> float:
        r'''
        Get the mirror Chern number of a group of bands on a mirror-invariant
        plane of the Brillouin zone (Teo, Fu and Kane, Phys. Rev. B 78,
        045426 (2008); Hsieh et al., Nat. Commun. 3, 982 (2012)),

        .. math::

            n_M = \frac{C_{+} - C_{-}}{2}\, ,

        with :math:`C_\pm` the Chern numbers of the bands of mirror
        eigenvalue :math:`+i` and :math:`-i` (:math:`M^2 = -1`, spinful
        electrons) or :math:`+1` and :math:`-1` (:math:`M^2 = +1`). The
        mirror must commute with :math:`H(\mathbf{k})` at every point of the
        plane: in 3D, the plane fixed by the reflection (e.g.
        :math:`k_z = 0` or :math:`\pi` for :math:`z\to-z`); in 2D, the whole
        zone, for a reflection of the third axis. A topological crystalline
        insulator such as SnTe has :math:`n_M = 2` and every
        :math:`\mathbb{Z}_2` index zero (:math:`\nu = n_M` modulo 2 on the plane).

        :param mirror: Complex ndarray, shape (norb, norb), or callable of k:
            the unitary mirror operator :math:`M`, with :math:`M^2 = \pm1`.
        :param bands: List of band indices (e.g. the occupied bands).
        :param nk: Positive integer, or tuple of 2 positive integers. Default value 30.
        :param plane: Tuple of two integers. Default value (0, 1). 3D only, see *berry_curvature*.
        :param k_fixed: Real number. Default value 0. 3D only: the
            mirror-invariant plane (0 or 0.5 for a reflection of the remaining axis).
        :param tol: Positive real. Default value 1e-8. Tolerance of the
            checks :math:`M^2 = \pm1` and :math:`[M, H(\mathbf{k})] = 0`.

        :returns:
            * **n_m** -- Real number, close to an integer.
        '''
        bands, nk, ks = self._check_sectors(bands, nk, plane, k_fixed)
        error_handling.positive_real(tol, 'tol')
        if callable(mirror):
            mats = np.array([self._operator(mirror, k) for k in ks])
        else:
            mats = self._operator(mirror, ks[0])[None]
        sign = error_handling.mirror_square(mats @ mats, tol)
        hams = self._hams(ks)
        error_handling.commutes(float(np.max(np.abs(mats @ hams - hams @ mats))), tol)
        # Hermitian, with eigenvalue +1 on the sector M = +i (M = +1 if M^2 = 1)
        herm = mats if sign > 0 else -1j * mats
        c_plus, c_minus = self._sector_cherns(herm, bands, nk, ks, plane)
        return (c_plus - c_minus) / 2

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
        params: dict | None = None,
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
        The value functions take *params* (default: *self.params*).
        '''
        k_cart = ks @ self.k_basis.T
        hops = self._hop if params is None else self._hops(params)
        ham, dham = self._bloch_sum(hops, k_cart, directions, positions)
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
        spin_op: NDArray[np.complex128] | None, response: str = 'hall',
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        r'''
        Private method. The Kubo curvature
        :math:`F(\mathbf{k}) = \sum_{n\neq m}P_{nm}\,\mathrm{Im}[X_{nm}Y_{mn}]/(E_n-E_m)^2`
        at each k-point, Fermi energy and pair (X, Y) of velocities (X the
        spin current :math:`\{s, v\}/2` if *spin_op*), with the weights
        :math:`P_{nm}` of *response* (see *_response_weights*), e.g.
        :math:`P_{nm} = -(f_n-f_m)` for the Hall conductivity
        (:math:`F = \sum_n f_n\Omega_n`). Shape (nk, len(e_fermi), len(pairs)).
        Pairs of degenerate states (equal weights) are skipped, so :math:`F`
        stays finite at band crossings below or above the Fermi level. Also
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
                    weights = _response_weights(en, float(mu), temperature, response)
                    out[c0:c0 + chunk, e, p] = np.einsum('knm,knm->k', weights, w)
        return out, hot

    def _hall(
        self, e_fermi, temperature, nk, plane, k_fixed, positions, refine, refine_fraction,
        spin_axis, response='hall',
    ):
        '''
        Private method. Validate, set the geometry and mesh, integrate the
        Kubo curvature (with the adaptive refinement), for
        *hall_conductivity* (spin_axis None) and *spin_hall_conductivity*,
        and, with the weights of *response*, for *orbital_magnetization*,
        *anomalous_nernst_conductivity* and *thermal_hall_conductivity*.
        '''
        self._check_static()
        error_handling.dim_min(self.dim, 2)
        error_handling.hermitian_model(self.is_hermitian())
        error_handling.fermi_energies(e_fermi)
        if response in ('nernst', 'thermal'):
            error_handling.positive_real(temperature, 'temperature')
        else:
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
        args = (e_arr, temperature, directions, pairs, positions, spin_op, response)
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
    # Berry-phase response beyond the Hall conductivity
    # ------------------------------------------------------------------

    def orbital_magnetization(
        self, e_fermi: float | ArrayLike, temperature: float = 0., nk: int | tuple[int, ...] = 60,
        plane: tuple[int, int] = (0, 1), k_fixed: float | None = None, positions: bool = True,
        refine: int = 1, refine_fraction: float = 0.05,
    ) -> float | NDArray[np.float64]:
        r'''
        Get the orbital magnetization in the modern theory (Thonhauser,
        Ceresoli, Vanderbilt and Resta, Phys. Rev. Lett. 95, 137205 (2005);
        Xiao, Shi and Niu, Phys. Rev. Lett. 95, 137204 (2005); Ceresoli et
        al., Phys. Rev. B 74, 024408 (2006)), at any Fermi level and
        temperature:

        .. math::

            M = \frac{e}{h}\,\frac{1}{2\pi}\int_{BZ} d^2k\sum_n\left[f_n\,m_n
            + g_n\,\Omega_n\right],\qquad
            m_n = \mathrm{Im}\langle\partial_xu_n|(H-E_n)|\partial_yu_n\rangle\, ,

        with :math:`g = k_BT\ln(1+e^{-(E-\mu)/k_BT})` (:math:`\mu - E` below
        :math:`\mu` and 0 above, at :math:`T = 0`). The first term is the
        self-rotation of the wavepackets, the second the circulation of
        the Berry-curvature currents at the edges. At :math:`T = 0` both
        combine into :math:`\sum_nf_n\,\mathrm{Im}\langle\partial_xu_n|(H+E_n-2\mu)|\partial_yu_n\rangle`,
        the formula of Thonhauser et al. It is evaluated, like
        *hall_conductivity*, from :math:`\langle n|\partial H|m\rangle`
        (no derivatives of the eigenstates), so it is finite where bands
        touch.

        **Sign and units.** The sign is that of *hall_conductivity*, so
        that the Streda formula reads :math:`\partial M/\partial\mu = \sigma`:
        in a gap (:math:`T = 0`), :math:`M` is linear in :math:`\mu` with
        slope the Chern number :math:`C`. In 2D, :math:`M` (a magnetic
        moment per area, i.e. a current) is in units of :math:`e/h` times
        the energy unit; with ``k_fixed=None`` in 3D, the vector
        :math:`(M_x, M_y, M_z)` per length. *positions* matters as for
        *hall_conductivity*, and the physical choice is the default.

        :param e_fermi: See *hall_conductivity*.
        :param temperature: See *hall_conductivity*.
        :param nk: See *hall_conductivity*.
        :param plane: See *hall_conductivity*.
        :param k_fixed: See *hall_conductivity*.
        :param positions: See *hall_conductivity*.
        :param refine: See *hall_conductivity*.
        :param refine_fraction: See *hall_conductivity*.

        :returns:
            * **M** -- Real number (or ndarray shaped like *e_fermi*); in 3D
              with ``k_fixed=None``, ndarray of shape (3,) (or ``e_fermi.shape + (3,)``).

        Example usage::

            mu = np.linspace(-0.5, 0.5, 11)  # in the gap of a Chern insulator
            M = hal.orbital_magnetization(mu)  # slope C = hall_conductivity(mu)
        '''
        return self._hall(e_fermi, temperature, nk, plane, k_fixed, positions, refine,
                                refine_fraction, None, 'magnetization')

    def anomalous_nernst_conductivity(
        self, e_fermi: float | ArrayLike, temperature: float, nk: int | tuple[int, ...] = 60,
        plane: tuple[int, int] = (0, 1), k_fixed: float | None = None, positions: bool = True,
        refine: int = 1, refine_fraction: float = 0.05,
    ) -> float | NDArray[np.float64]:
        r'''
        Get the intrinsic anomalous Nernst (transverse thermoelectric)
        conductivity :math:`\alpha_{xy}`, :math:`j_x = \alpha_{xy}(-\partial_yT)`,
        from the Berry curvature weighted by the entropy of each state
        (Xiao, Yao, Fang and Niu, Phys. Rev. Lett. 97, 026603 (2006)):

        .. math::

            \alpha_{xy} = \frac{ek_B}{h}\,\frac{1}{2\pi}\int_{BZ}d^2k\sum_n s_n\,\Omega_n\, ,
            \qquad s = -f\ln f - (1-f)\ln(1-f)\, .

        Only the states within a few :math:`k_BT` of :math:`\mu` carry
        entropy, so it vanishes in a gap at low temperature. It obeys the
        Mott relation exactly,
        :math:`\alpha_{xy}(\mu, T) = \frac{1}{eT}\int dE\,(E-\mu)\left(-\frac{\partial f}{\partial E}\right)\sigma_{xy}(E)`,
        with :math:`\sigma_{xy}(E)` the :math:`T = 0` *hall_conductivity*
        at Fermi energy :math:`E`, which fixes the sign. The pair form of
        *hall_conductivity* keeps it finite where bands touch.

        **Units.** :math:`ek_B/h` in 2D (:math:`k_B = 1`, temperatures in
        energy units), per length for the 3D vector (``k_fixed=None``).

        :param e_fermi: See *hall_conductivity*.
        :param temperature: Positive real number. In energy units (:math:`k_B = 1`).
        :param nk: See *hall_conductivity*.
        :param plane: See *hall_conductivity*.
        :param k_fixed: See *hall_conductivity*.
        :param positions: See *hall_conductivity*.
        :param refine: See *hall_conductivity*.
        :param refine_fraction: See *hall_conductivity*.

        :returns:
            * **alpha** -- Real number (or ndarray, as *hall_conductivity*).

        Example usage::

            alpha = hal.anomalous_nernst_conductivity(np.linspace(-3, 3, 301), 0.05)
        '''
        return self._hall(e_fermi, temperature, nk, plane, k_fixed, positions, refine,
                                refine_fraction, None, 'nernst')

    def thermal_hall_conductivity(
        self, e_fermi: float | ArrayLike, temperature: float, nk: int | tuple[int, ...] = 60,
        plane: tuple[int, int] = (0, 1), k_fixed: float | None = None, positions: bool = True,
        refine: int = 1, refine_fraction: float = 0.05,
    ) -> float | NDArray[np.float64]:
        r'''
        Get the intrinsic thermal Hall conductivity of the electrons
        :math:`\kappa_{xy}`, :math:`j^Q_x = \kappa_{xy}(-\partial_yT)`
        (Qin, Niu and Shi, Phys. Rev. Lett. 107, 236601 (2011)):

        .. math::

            \kappa_{xy} = \frac{k_B^2T}{h}\,\frac{1}{2\pi}\int_{BZ}d^2k
            \sum_n c_2(x_n)\,\Omega_n\, ,\qquad
            c_2(x) = \int_x^\infty y^2\left(-\frac{\partial f}{\partial y}\right)dy\, ,

        :math:`x_n = (E_n-\mu)/k_BT` (:math:`c_2 = \pi^2/3` deep below
        :math:`\mu`, 0 far above). Equivalently, exactly,
        :math:`\kappa_{xy}(\mu, T) = \frac{1}{e^2T}\int dE\,(E-\mu)^2\left(-\frac{\partial f}{\partial E}\right)\sigma_{xy}(E)`
        with the :math:`T = 0` *hall_conductivity*. In a gap at low
        temperature it is the Wiedemann-Franz value
        :math:`\kappa_{xy} = \frac{\pi^2k_B^2}{3e^2}T\sigma_{xy}`: for a Chern
        insulator the quantized :math:`C\,\pi^2k_B^2T/3h`.

        **Units.** :math:`k_B^2/h` times the temperature (in energy units,
        :math:`k_B = 1`) in 2D, so that :math:`\kappa/T = \pi^2C/3` on a
        Chern plateau; per length for the 3D vector (``k_fixed=None``).
        The sign is that of *hall_conductivity*.

        :param e_fermi: See *hall_conductivity*.
        :param temperature: Positive real number. In energy units (:math:`k_B = 1`).
        :param nk: See *hall_conductivity*.
        :param plane: See *hall_conductivity*.
        :param k_fixed: See *hall_conductivity*.
        :param positions: See *hall_conductivity*.
        :param refine: See *hall_conductivity*.
        :param refine_fraction: See *hall_conductivity*.

        :returns:
            * **kappa** -- Real number (or ndarray, as *hall_conductivity*).

        Example usage::

            kappa = hal.thermal_hall_conductivity(0., 0.01)  # = pi^2/3 C T in the gap
        '''
        return self._hall(e_fermi, temperature, nk, plane, k_fixed, positions, refine,
                                refine_fraction, None, 'thermal')

    def _axion_rate(
        self, ks: NDArray[np.float64], bands: list[int], params: dict, param: str,
        positions: bool,
    ) -> float:
        r'''
        Private method. The mesh average, over *ks*, of
        :math:`\mathrm{tr}[F_{\lambda x}F_{yz} + F_{\lambda y}F_{zx} + F_{\lambda z}F_{xy}]`
        for the bands *bands* at the parameters *params*, the derivative
        along *param* by central differences. The non-Abelian curvature is
        :math:`F_{\mu\nu} = i(X_\mu^\dagger X_\nu - X_\nu^\dagger X_\mu)`,
        :math:`(X_\mu)_{lm} = \langle l|\partial_\mu H|m\rangle/(E_m-E_l)`
        (:math:`m` in *bands*, :math:`l` not), i.e.
        :math:`Q|\partial_\mu u_m\rangle` in the eigenbasis: gauge covariant.
        '''
        value = params[param]
        step = 1e-5 * max(1., abs(value))
        others = [b for b in range(self.norb) if b not in bands]
        total, gap = 0., np.inf
        chunk = max(1, 400000 // self.norb ** 2)
        for c0 in range(0, len(ks), chunk):
            k = ks[c0:c0 + chunk]
            ham, dham = self._bloch_derivatives(k, list(np.eye(3)), positions, params)
            plus = self._bloch_derivatives(k, [], positions, dict(params, **{param: value + step}))[0]
            minus = self._bloch_derivatives(k, [], positions, dict(params, **{param: value - step}))[0]
            en, vec = np.linalg.eigh(ham)
            occ, emp = vec[:, :, bands], vec[:, :, others]
            de = en[:, others][:, :, None] - en[:, bands][:, None, :]
            gap = min(gap, np.abs(de).min())
            emp_h = emp.conj().transpose(0, 2, 1)
            x = [-(emp_h @ d @ occ) / de for d in [(plus - minus) / (2 * step), *dham]]
            f = {}
            for a, b in ((0, 1), (0, 2), (0, 3), (1, 2), (2, 3), (3, 1)):
                xa_h = x[a].conj().transpose(0, 2, 1)
                xb_h = x[b].conj().transpose(0, 2, 1)
                f[a, b] = 1j * (xa_h @ x[b] - xb_h @ x[a])
            trace = (np.einsum('kab,kba->k', f[0, 1], f[2, 3]) + np.einsum('kab,kba->k', f[0, 2], f[3, 1])
                       + np.einsum('kab,kba->k', f[0, 3], f[1, 2]))
            total += trace.real.sum()
        error_handling.path_gap(gap, param, value)
        return total / len(ks)

    def axion_angle(
        self, bands: list[int], param: str, values: ArrayLike, nk: int | tuple[int, int, int] = 16,
        theta0: float = 0., positions: bool = True,
    ) -> NDArray[np.float64]:
        r'''
        Get the axion angle :math:`\theta` of a 3D insulator along a path of
        a parameter of its value functions (*set_hopping*), from the second
        Chern form (Qi, Hughes and Zhang, Phys. Rev. B 78, 195424 (2008);
        Essin, Moore and Vanderbilt, Phys. Rev. Lett. 102, 146805 (2009)):

        .. math::

            \frac{d\theta}{d\lambda} = \frac{1}{2\pi}\int_{BZ}d^3k\;
            \mathrm{tr}\left[F_{\lambda x}F_{yz} + F_{\lambda y}F_{zx} + F_{\lambda z}F_{xy}\right],

        :math:`F_{\mu\nu}` the non-Abelian Berry curvature of the bands
        *bands* in :math:`(\lambda, k_x, k_y, k_z)` space, so that
        :math:`\theta(\lambda) = \theta_0 + \int_{\lambda_0}^{\lambda}\frac{d\theta}{d\lambda'}d\lambda'`.
        This is the variation of the Chern-Simons axion coupling, computed
        from gauge-invariant quantities only (no smooth gauge is needed),
        and :math:`\theta_0` is its value at ``values[0]``: e.g. 0
        for a path starting from an atomic insulator. A closed path (a
        pumping cycle) changes :math:`\theta` by :math:`2\pi C_2`, with
        :math:`C_2` the second Chern number of the 4D family. At a point
        with time-reversal or inversion symmetry, :math:`\theta` is 0 or
        :math:`\pi` (mod :math:`2\pi`): :math:`\pi` for a strong topological
        insulator, cf. *z2_indices_3d*. The sign of :math:`\theta` follows
        the orientation :math:`(\lambda, k_x, k_y, k_z)` above.

        The path must stay gapped: the bands *bands* must be separated
        from the others at every k-point and every value (a ValueError
        names the value where they touch). The integrand is smooth, so the
        uniform k-mesh converges exponentially; the :math:`\lambda`
        integral is a trapezoid over *values* (error :math:`O(\Delta\lambda^2)`).
        :math:`\partial H/\partial\lambda` is a central difference, and
        :math:`\partial H/\partial\mathbf{k}` is exact (see
        *hall_conductivity*, whose *positions* convention applies).

        :param bands: List of integers. The occupied bands (e.g. the lower
            half of the spectrum).
        :param param: String. Name of the parameter of the value functions
            that the path varies (see *set_params*; the other parameters keep
            their values in *params*).
        :param values: 1D array of at least two real numbers, increasing
            or decreasing. The parameter values along the path.
        :param nk: Positive integer, or tuple of 3 positive integers.
            Default value 16. k-mesh.
        :param theta0: Real number. Default value 0. :math:`\theta` at ``values[0]``.
        :param positions: Boolean. Default value True. See *hall_conductivity*.

        :returns:
            * **theta** -- ndarray, shape ``(len(values),)``: :math:`\theta`
              at each value, not reduced modulo :math:`2\pi`.

        Example usage::

            phi = np.linspace(0., 2 * np.pi, 41)
            theta = ks.axion_angle([0, 1], 'phi', phi)  # theta[-1] = 2 pi C2
        '''
        self._check_static()
        error_handling.dim_exact(self.dim, 3)
        error_handling.hermitian_model(self.is_hermitian())
        error_handling.band_indices(bands, self.norb)
        error_handling.path_parameter(param)
        error_handling.path_values(values)
        if isinstance(nk, int):
            nk = (nk,) * 3
        error_handling.nk(nk, 3)
        error_handling.real_number(theta0, 'theta0')
        error_handling.boolean(positions, 'positions')
        values = np.asarray(values, dtype='f8')
        fracs = np.array(np.meshgrid(*[np.arange(n) / n for n in nk], indexing='ij')).reshape(3, -1).T
        ks = fracs @ self.rec_vec_k
        volume = abs(np.linalg.det(self.rec_vec_k))
        rates = [self._axion_rate(ks, bands, dict(self.params, **{param: float(v)}), param, positions)
                     for v in values]
        return theta0 + cumulative_trapezoid(np.array(rates) * volume / (2 * PI), values, initial=0.)

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
        for cell in range(n_tot):
            ham[cell*self.norb:(cell+1)*self.norb, cell*self.norb:(cell+1)*self.norb] += onsite
        for r, c, _, _, _, _, t in entries:
            np.add.at(ham, (r, c), t)
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
        vs = self._subspaces(k0 + np.arange(nk)[:, None] * b[None] / nk, bands)
        links = vs.conj().transpose(0, 2, 1) @ (phase[:, None] * np.roll(vs, -1, axis=0))
        # keep only the unitary part of each link (its singular values
        # tend to 1 as nk grows): the product then stays unitary
        u, _, vh = np.linalg.svd(links)
        wilson = np.eye(len(bands), dtype='c16')
        for unitary in u @ vh:
            wilson = wilson @ unitary
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

    def z2_indices_3d(
        self, bands: list[int], nk: int = 100, nk_perp: int = 51,
    ) -> tuple[int, int, int, int]:
        r'''
        Get the four :math:`\mathbb{Z}_2` indices
        :math:`(\nu_0;\nu_1\nu_2\nu_3)` of a 3D time-reversal-invariant
        insulator (Fu, Kane and Mele, Phys. Rev. Lett. 98, 106803 (2007);
        Moore and Balents, Phys. Rev. B 75, 121306(R) (2007)) from the
        :math:`\mathbb{Z}_2` invariants of its six time-reversal-invariant
        planes :math:`k_i = 0` and :math:`k_i = 1/2` (fractional coordinates
        along :math:`\mathbf{b}_i`), each computed by *z2_invariant*:

        .. math::

            \nu_0 = \nu(k_i = 0) + \nu(k_i = 1/2) \bmod 2\, ,\qquad
            \nu_i = \nu(k_i = 1/2)\, .

        The strong index :math:`\nu_0` comes out the same for the three
        :math:`i`, which is checked (a ValueError asks for a finer flow
        otherwise). :math:`\nu_0 = 1` is a strong topological insulator; a
        weak one has :math:`\nu_0 = 0` and some :math:`\nu_i = 1`, i.e.
        stacked quantum spin Hall layers normal to
        :math:`\nu_1\mathbf{b}_1 + \nu_2\mathbf{b}_2 + \nu_3\mathbf{b}_3`.
        No inversion symmetry is needed (compare *parity_z2*).

        :param bands: List of band indices (an even number of bands).
        :param nk: Positive integer. Default value 100. See *z2_invariant*.
        :param nk_perp: Positive integer. Default value 51. See *z2_invariant*.

        :returns:
            * **indices** -- Tuple of four integers, 0 or 1: :math:`(\nu_0, \nu_1, \nu_2, \nu_3)`.
        '''
        error_handling.dim_exact(self.dim, 3)
        nu = np.zeros((3, 2), int)
        for i in range(3):
            direction, flow = [d for d in range(3) if d != i]
            for n, k_fixed in enumerate((0., 0.5)):
                nu[i, n] = self.z2_invariant(bands, nk, nk_perp, direction, flow, k_fixed)
        strong = nu.sum(axis=1) % 2
        error_handling.strong_index(strong)
        return int(strong[0]), int(nu[0, 1]), int(nu[1, 1]), int(nu[2, 1])

    def entanglement_spectrum(
        self, ks: ArrayLike, region: list[int], bands: int | list[int],
    ) -> NDArray[np.float64]:
        r'''
        Get the k-resolved entanglement spectrum of a group of bands
        (Peschel, J. Phys. A 36, L205 (2003); Li and Haldane, Phys. Rev. Lett.
        101, 010504 (2008); Fidkowski, Phys. Rev. Lett. 104, 130502 (2010)):
        the eigenvalues :math:`\xi_n(\mathbf{k})` of the correlation matrix
        :math:`P(\mathbf{k})` of the filled *bands*, restricted to the
        orbitals of *region*,

        .. math::

            C_A(\mathbf{k}) = \left[P(\mathbf{k})\right]_{A}\, ,\qquad
            P(\mathbf{k}) = \sum_{n\in\mathrm{bands}}|u_n(\mathbf{k})\rangle\langle u_n(\mathbf{k})|\, ,

        between 0 and 1. For a ribbon or a supercell
        (*tbkit.moire.supercell*) cut in two halves across its width, the
        states of the whole are either inside the region
        (:math:`\xi \approx 1`) or outside (:math:`\xi \approx 0`), except
        for those localized at the cuts: a topological phase leaves
        entanglement modes inside :math:`(0, 1)` -- a mode pinned at 1/2 in
        a chiral-symmetric chain, a branch that flows from 0 to 1 across
        the zone in a Chern insulator. The single-particle entanglement
        energies are :math:`\epsilon = \ln[(1-\xi)/\xi]`.

        :param ks: ndarray, shape (nk, dim). k-points (see *get_bands*).
        :param region: List of orbital indices (the region :math:`A`).
        :param bands: Band index, or list of band indices (the filled bands).

        :returns:
            * **xi** -- Real ndarray, shape (nk, len(region)), sorted ascending at every k.
        '''
        ks = np.atleast_2d(np.asarray(ks, dtype='f8'))
        error_handling.ks(ks, self.dim)
        error_handling.region(region, self.norb)
        if isinstance(bands, int):
            bands = [bands]
        error_handling.band_indices(bands, self.norb)
        error_handling.hermitian_kspace(self.is_hermitian(), self._overlap_hop)
        v = self._subspaces(ks, bands)[:, region, :]
        return np.linalg.eigvalsh(v @ v.conj().transpose(0, 2, 1))

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
            if u_t is not None:
                u_c = u_s @ u_t
            elif u_c is not None:
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

    def surface_spectral_function(
        self, ks: ArrayLike, energies: ArrayLike, direction: int, side: int = 1,
        eta: float = 1e-2, bulk: bool = False, max_iter: int = 10000,
    ) -> NDArray[np.float64]:
        r'''
        Get the spectral function of the surface of a semi-infinite crystal,

        .. math::

            A_s(\mathbf{k}_\parallel, E) = -\frac{1}{\pi}\,
            \mathrm{Im}\,\mathrm{Tr}\, G_{00}(\mathbf{k}_\parallel, E + i\eta)\, ,

        with :math:`G_{00}` the Green's function of the outermost unit cell:
        the crystal fills the cells :math:`n\,\mathbf{a}_d`,
        :math:`n = 0, 1, 2, \dots` along the primitive vector
        :math:`\mathbf{a}_d` (``lat.prim_vec[direction]``), times *side*, and
        is periodic along the others. It comes from the Lopez Sancho-Rubio
        decimation (see *transport.surface_green*) of the chain of principal
        layers (as many cells as the longest hopping along
        :math:`\mathbf{a}_d` spans), so the crystal is truly semi-infinite: no
        finite slab, and no states of the opposite surface. It is the
        tight-binding picture of an ARPES map -- surface states (edge states,
        in 2D; the end state of a 1D chain) show up as sharp lines inside the
        bulk gaps. With *bulk*, the same quantity for a unit cell deep inside
        the crystal: the bulk bands projected on the surface Brillouin zone.

        :param ks: Real array, shape (nk, dim). k-points, in the coordinates
            of *get_ham* (e.g. from a *k_path* of the surface Brillouin zone).
            Only :math:`\mathbf{k}_\parallel` matters: the result does not
            depend on the component along the reciprocal vector
            :math:`\mathbf{b}_d`.
        :param energies: Real number or real array. Energies :math:`E`.
        :param direction: Integer, between 0 and dim-1. The primitive vector
            normal to the surface (the one along which the crystal is
            semi-infinite).
        :param side: +1 or -1. Default value 1. The crystal extends along
            :math:`+\mathbf{a}_d` (the surface faces :math:`-\mathbf{a}_d`) or
            along :math:`-\mathbf{a}_d` (the opposite surface).
        :param eta: Positive real. Default value 1e-2. Broadening: the
            width of the lines, and the decay length (about :math:`v/\eta`
            cells) over which the bulk is resolved.
        :param bulk: Boolean. Default value False. If True, the spectral
            function of a bulk unit cell instead of the surface one.
        :param max_iter: Positive integer. Default value 10000. Maximum
            number of decimation steps.

        :returns:
            * **A** -- Real ndarray, shape (nk, len(energies)).

        Example usage::

            # the (001) surface of a 3D model, along a path of the surface zone
            A = ks.surface_spectral_function(k_path, np.linspace(-1, 1, 201), direction=2)
        '''
        error_handling.no_overlap(self._overlap_hop)
        ks = np.atleast_2d(np.asarray(ks, dtype='f8'))
        error_handling.ks(ks, self.dim)
        error_handling.frequencies(energies, 'energies')
        energies = np.atleast_1d(np.asarray(energies, dtype='f8'))
        error_handling.direction(direction, self.dim)
        error_handling.surface_side(side)
        error_handling.positive_real(eta, 'eta')
        error_handling.boolean(bulk, 'bulk')
        error_handling.positive_int(max_iter, 'max_iter')
        # H_m(k): the hoppings from a cell to the cell m a_d away, with the
        # phase of their in-plane part only (independent of k along b_d)
        a_dir = np.array(self.lat.prim_vec[direction], dtype='f8')
        layers: dict[int, list] = {}
        for (i, j, cell, t), (_, _, R, _) in zip(self._hop_cells(), self._hop):
            layers.setdefault(cell[direction], []).append((i, j, R - cell[direction] * a_dir, t))
        k_cart = ks @ self.k_basis.T
        blocks: dict[int, NDArray] = {m: self._bloch_sum(hops, k_cart)[0] for m, hops in layers.items()}
        blocks[0] = blocks.get(0, 0.) + (np.diag(self.onsite) + self._onsite_offdiag)[None]
        width = max([abs(m) for m in layers] + [1])
        norb, nk = self.norb, len(ks)

        def layer(shift):
            # block (p, q) couples cell p of a principal layer to cell q of
            # the layer *shift* cells further into the crystal
            out = np.zeros((nk, width, norb, width, norb), 'c16')
            for p in range(width):
                for q in range(width):
                    m = side * (q - p + shift)
                    if m in blocks:
                        out[:, p, :, q, :] = blocks[m]
            return out.reshape(nk, width * norb, width * norb)
        h0, alpha, beta = layer(0), layer(width), layer(-width)
        spec = np.empty((nk, len(energies)))
        for n in range(nk):
            gs, gb = _green_surface_bulk(energies + 1j * eta, h0[n], alpha[n], beta[n], 1e-12,
                                                       max_iter, 'surface_spectral_function')
            g = gb if bulk else gs
            spec[n] = -np.trace(g[:, :norb, :norb], axis1=1, axis2=2).imag / PI
        return spec

    # ------------------------------------------------------------------
    # Projections, constant-energy contours and plot helpers
    # ------------------------------------------------------------------

    def spin_operator(self, axis: str) -> NDArray[np.complex128]:
        r'''
        Get the Pauli matrix :math:`\sigma_{axis}` acting on the spin of
        every site, :math:`\mathbb{1}_{sites}\otimes\sigma_{axis}`, for a model
        built with ``spin=True`` (for *band_weights* and *spin_texture*).

        :param axis: 'x', 'y' or 'z'.

        :returns:
            * **op** -- Complex ndarray, shape (norb, norb).
        '''
        error_handling.spinful(self.spin)
        error_handling.spin_axis(axis)
        return np.kron(np.eye(self.n_sites), PAULI[axis])

    def _projector(self, projector) -> NDArray[np.complex128]:
        '''
        Private method. The (norb, norb) operator of *band_weights* from an
        orbital index, a list of them, a sublattice tag or a matrix.
        '''
        error_handling.projector(projector, self.norb, self.tags)
        if isinstance(projector, np.ndarray):
            return projector.astype('c16')
        if isinstance(projector, str):
            sites = np.nonzero(self.tags == projector)[0]
            projector = np.concatenate([2 * sites, 2 * sites + 1]) if self.spin else sites
        diag = np.zeros(self.norb)
        diag[projector] = 1.
        return np.diag(diag).astype('c16')

    def _expectations(self, ks: NDArray[np.float64], ops: list) -> NDArray[np.float64]:
        r'''
        Private method. :math:`\mathrm{Re}\langle u_n(\mathbf{k})|O|u_n(\mathbf{k})\rangle
        /\langle u_n|u_n\rangle` for each operator of *ops*, band and k-point:
        shape (len(ops), nk, norb), from one diagonalization. Within a group
        of degenerate bands (energies within 1e-8 of each other, relative to
        the largest one), each band gets the average over the group: the
        trace of O over the degenerate subspace divided by its dimension,
        which does not depend on the eigenvectors the solver returns.
        '''
        en, vn = self._eigs(ks, eigenvec=True)
        norm = np.sum(np.abs(vn) ** 2, axis=1)
        values = np.array([np.einsum('kin,ij,kjn->kn', vn.conj(), op, vn).real / norm for op in ops])
        # label the degenerate groups (consecutive bands, sorted by energy)
        tol = 1e-8 * max(1., np.abs(en).max())
        new_group = np.concatenate([np.ones((len(en), 1), bool),
                                              np.abs(np.diff(en, axis=1)) > tol], axis=1)
        groups = np.cumsum(new_group.ravel()) - 1
        counts = np.bincount(groups)
        return np.array([(np.bincount(groups, v.ravel()) / counts)[groups].reshape(v.shape)
                              for v in values])

    def band_weights(self, projector, ks: ArrayLike | None = None) -> NDArray[np.float64]:
        r'''
        Get the weight of each band on a set of orbitals, or the expectation
        value of an operator, for "fat band" plots (see *plot_bands*):

        .. math::

            w_n(\mathbf{k}) = \langle u_n(\mathbf{k})|P|u_n(\mathbf{k})\rangle\, ,

        with :math:`P` the projector on the chosen orbitals (a weight
        between 0 and 1; the weights on all the sublattices add up to 1), or
        any Hermitian operator, e.g. ``ks.spin_operator('z')`` for the spin
        polarization (between -1 and 1). Within a group of degenerate bands
        (e.g. Kramers pairs with inversion and time reversal), the weight of
        a single eigenvector depends on the choice of eigenvectors; each band
        of the group gets instead the average over the group, which does not
        (a Kramers pair has zero spin polarization).
        For a non-Hermitian model, the right eigenvectors are used; with an
        overlap (*set_overlap*), :math:`v^\dagger Pv/v^\dagger v`.

        :param projector: Orbital index, list of orbital indices, sublattice
            tag (all the orbitals, both spins, of the sites with that tag),
            or a (norb, norb) Hermitian matrix.
        :param ks: Real array, shape (nk, dim). Default value None: the
            k-points of the last *get_bands* or *k_path*.

        :returns:
            * **weights** -- Real ndarray, shape (nk, norb), ordered as the bands.

        Example usage::

            ks.k_path(points, nk=100)
            ks.plot_bands(weights=ks.band_weights('a'))
        '''
        op = self._projector(projector)
        if ks is None:
            error_handling.empty_ndarray(self.en, 'get_bands or k_path')
            ks = self.ks
        ks = np.atleast_2d(np.asarray(ks, dtype='f8'))
        error_handling.ks(ks, self.dim)
        return self._expectations(ks, [op])[0]

    def spin_texture(self, ks: ArrayLike, band: int) -> NDArray[np.float64]:
        r'''
        Get the spin expectation values
        :math:`\langle\boldsymbol\sigma\rangle_n(\mathbf{k}) =
        \langle u_n|\boldsymbol\sigma|u_n\rangle` of a band, for a model built
        with ``spin=True``. Their length is 1 if the band is fully polarized,
        less if spin-orbit coupling entangles the spin with the orbitals.
        Within a group of degenerate bands, each band gets the average spin
        of the group (see *band_weights*): zero for a Kramers pair, with both
        inversion and time reversal.

        :param ks: Real array, shape (nk, dim). k-points.
        :param band: Integer. Band index.

        :returns:
            * **spin** -- Real ndarray, shape (nk, 3):
              :math:`(\langle\sigma_x\rangle, \langle\sigma_y\rangle, \langle\sigma_z\rangle)`.
        '''
        error_handling.spinful(self.spin)
        error_handling.band_index(band, self.norb)
        ks = np.atleast_2d(np.asarray(ks, dtype='f8'))
        error_handling.ks(ks, self.dim)
        ops = [self.spin_operator(a) for a in 'xyz']
        return self._expectations(ks, ops)[:, :, band].T

    def _fold(self, elements: NDArray[np.float64]) -> NDArray[np.float64]:
        '''
        Private method. Bring segments or triangles (fractional coordinates,
        shape (n, nv, dim)) into the first Brillouin zone (the Wigner-Seitz
        cell of the reciprocal lattice, by their centres), in the k
        coordinates of *get_ham*.
        '''
        elements = elements - np.floor(elements.mean(axis=1) + 0.5)[:, None, :]
        cart = elements @ self.rec_vec_k
        shifts = np.array(list(product((-1, 0, 1), repeat=self.dim))) @ self.rec_vec_k
        dist = np.linalg.norm(cart.mean(axis=1)[:, None, :] - shifts[None], axis=2)
        return cart - shifts[np.argmin(dist, axis=1)][:, None, :]

    def fermi_surface(
        self, energy: float = 0., nk: int | tuple[int, ...] | None = None,
        bands: int | list[int] | None = None,
    ) -> list[NDArray[np.float64]]:
        r'''
        Get the constant-energy surfaces :math:`E_n(\mathbf{k}) = E` (the
        Fermi surface at :math:`E = E_F`): lines in a 2D Brillouin zone,
        surfaces in 3D. The bands are interpolated linearly in the triangles
        (tetrahedra in 3D) of a uniform mesh, the same decomposition as
        *dos.tetrahedron_dos*, and the level set of each simplex is a segment
        (a triangle or a quadrilateral, split in two, in 3D): marching
        triangles and tetrahedra, exact for the interpolated bands. Each
        piece is brought into the first Brillouin zone, the Wigner-Seitz
        cell around :math:`\Gamma`. For a non-Hermitian model, the real part
        of the energies is used.

        :param energy: Real number. Default value 0. The energy :math:`E`.
        :param nk: Positive integer, or tuple of *dim* positive integers.
            Default value None: 100 in 2D, 40 in 3D. Number of k-points along
            each reciprocal lattice vector.
        :param bands: Band index, or list of band indices. Default value
            None: all bands.

        :returns:
            * **surfaces** -- List, one entry per band of *bands*: a real
              ndarray of shape (n, 2, 2) in 2D (n segments, each two k-points)
              or (n, 3, 3) in 3D (n triangles, each three k-points), in the
              k coordinates of *get_ham*. Empty (n = 0) if the band does not
              cross *energy*.

        Example usage::

            # the Fermi surface of the half-filled square lattice
            segments = sq.fermi_surface(0.)[0]
        '''
        error_handling.dim_min(self.dim, 2)
        error_handling.real_number(energy, 'energy')
        if nk is None:
            nk = 100 if self.dim == 2 else 40
        error_handling.nk(nk, self.dim)
        if isinstance(nk, int):
            nk = (nk,) * self.dim
        if bands is None:
            bands = list(range(self.norb))
        elif isinstance(bands, int):
            bands = [bands]
        error_handling.band_indices(bands, self.norb)
        _, ks = self.mesh_grid(nk)
        en = self._eigs(ks).real.reshape(*nk, self.norb)
        return [self._fold(_level_set(en[..., n], energy)) for n in bands]

    def _k_labels(self) -> list[str]:
        '''
        Private method. Axis labels of the k coordinates of *get_ham*:
        '$k_x$' for a coordinate along the Cartesian x axis (the slab of a
        3D model along x and z gives '$k_x$', '$k_z$'), '$k_1$', ... otherwise.
        '''
        labels = []
        for n, col in enumerate(self.k_basis.T):
            axis = np.nonzero(np.isclose(np.abs(col), 1.))[0]
            labels.append('$k_{}$'.format('xyz'[axis[0]] if len(axis) else n + 1))
        return labels

    def _zone_axes(self, ax, fs: float) -> None:
        '''
        Private method. Draw the first Brillouin zone of a 2D model on *ax*,
        and label and scale its axes.
        '''
        zone = _zone_polygon(self.rec_vec_k)
        ax.plot(*np.vstack([zone, zone[:1]]).T, 'k', lw=1)
        ax.set_aspect('equal')
        pad = 0.05 * np.ptp(zone, axis=0).max()
        ax.set_xlim(zone[:, 0].min() - pad, zone[:, 0].max() + pad)
        ax.set_ylim(zone[:, 1].min() - pad, zone[:, 1].max() + pad)
        ax.set_xlabel(self._k_labels()[0], fontsize=fs)
        ax.set_ylabel(self._k_labels()[1], fontsize=fs)

    def plot_fermi_surface(
        self, energy: float = 0., nk: int | tuple[int, ...] | None = None,
        bands: int | list[int] | None = None, lw: float = 2., alpha: float = 0.6,
        fs: float = 20, figsize: tuple[float, float] | None = None,
    ) -> Figure:
        '''
        Plot the constant-energy contours of *fermi_surface* (one color per
        band, with a legend if several bands cross the energy): lines inside
        the first Brillouin zone (drawn in black) in 2D, surfaces in 3D.

        :param energy: Real number. Default value 0. The energy.
        :param nk: See *fermi_surface*.
        :param bands: See *fermi_surface*.
        :param lw: Positive number. Default value 2. Linewidth (2D).
        :param alpha: Real number in (0, 1]. Default value 0.6. Opacity of
            the surfaces (3D).
        :param fs: Positive number. Default value 20. Fontsize.
        :param figsize: Tuple. Default value None. Figure size.

        :returns:
            * **fig** -- Figure.
        '''
        error_handling.positive_real(lw, 'lw')
        error_handling.positive_real(alpha, 'alpha')
        error_handling.positive_real(fs, 'fs')
        error_handling.tuple_2elem(figsize, 'figsize')
        surfaces = self.fermi_surface(energy, nk, bands)
        bands = list(range(self.norb)) if bands is None else np.atleast_1d(bands).tolist()
        fig = plt.figure(figsize=figsize)
        if self.dim == 2:
            ax = fig.add_subplot()
            for n, (band, segs) in enumerate(zip(bands, surfaces)):
                if len(segs):
                    ax.add_collection(LineCollection(list(segs), colors='C{}'.format(n % 10), lw=lw,
                                                                   label='band {}'.format(band)))
            self._zone_axes(ax, fs)
        else:
            ax = fig.add_subplot(projection='3d')
            for n, (band, tris) in enumerate(zip(bands, surfaces)):
                if len(tris):
                    ax.add_collection3d(Poly3DCollection(tris, facecolor='C{}'.format(n % 10),
                                                                         alpha=min(alpha, 1.), edgecolor='none',
                                                                         label='band {}'.format(band)))
            extent = np.abs(np.concatenate([t.reshape(-1, 3) for t in surfaces] +
                                                         [self.rec_vec_k / 2])).max()
            ax.set(xlim=(-extent, extent), ylim=(-extent, extent), zlim=(-extent, extent))
            ax.set_box_aspect((1, 1, 1))
            ax.set_xlabel(self._k_labels()[0], fontsize=fs)
            ax.set_ylabel(self._k_labels()[1], fontsize=fs)
            ax.set_zlabel(self._k_labels()[2], fontsize=fs)
        if sum(len(s) > 0 for s in surfaces) > 1:
            ax.legend(fontsize=0.6 * fs)
        ax.set_title('$E = {:g}$'.format(energy), fontsize=fs)
        plt.draw()
        return fig

    def plot_spin_texture(
        self, band: int, energy: float = 0., nk: int = 100, n_arrows: int = 40,
        cmap: str = 'RdBu_r', lw: float = 1., fs: float = 20,
        figsize: tuple[float, float] | None = None,
    ) -> Figure:
        r'''
        Plot the spin texture of a band of a 2D spinful model on its
        constant-energy contour (see *fermi_surface*), zoomed on the
        contour: arrows of
        :math:`(\langle\sigma_x\rangle, \langle\sigma_y\rangle)` along the
        contour, colored by :math:`\langle\sigma_z\rangle` (see *spin_texture*).
        Rashba coupling, for instance, locks the spin perpendicular to
        :math:`\mathbf{k}`, winding once around the contour.

        :param band: Integer. Band index.
        :param energy: Real number. Default value 0. Energy of the contour.
        :param nk: Positive integer. Default value 100. Mesh of the contour.
        :param n_arrows: Positive integer. Default value 40. Approximate
            number of arrows, evenly spaced along the contour. A fully
            polarized in-plane spin is drawn 80% as long as their spacing.
        :param cmap: Default value 'RdBu_r'. Colormap of :math:`\langle\sigma_z\rangle`.
        :param lw: Positive number. Default value 1. Linewidth of the contour.
        :param fs: Positive number. Default value 20. Fontsize.
        :param figsize: Tuple. Default value None. Figure size.

        :returns:
            * **fig** -- Figure.
        '''
        error_handling.dim_2(self.dim)
        error_handling.spinful(self.spin)
        error_handling.band_index(band, self.norb)
        error_handling.positive_int(nk, 'nk')
        error_handling.positive_int(n_arrows, 'n_arrows')
        error_handling.positive_real(lw, 'lw')
        error_handling.positive_real(fs, 'fs')
        error_handling.tuple_2elem(figsize, 'figsize')
        segs = self.fermi_surface(energy, nk, [band])[0]
        error_handling.contour_found(len(segs), energy)
        # arrows at segment midpoints, at least one contour length / n_arrows apart
        mids = segs.mean(axis=1)
        spacing = np.linalg.norm(segs[:, 1] - segs[:, 0], axis=1).sum() / n_arrows
        keep = [0]
        for i in range(1, len(mids)):
            if np.min(np.linalg.norm(mids[keep] - mids[i], axis=1)) >= spacing:
                keep.append(i)
        spin = self.spin_texture(mids[keep], band)
        fig, ax = plt.subplots(figsize=figsize)
        ax.add_collection(LineCollection(list(segs), colors='0.6', lw=lw))
        arrows = ax.quiver(mids[keep, 0], mids[keep, 1], spin[:, 0], spin[:, 1], spin[:, 2],
                                  cmap=cmap, norm=Normalize(-1., 1.), pivot='mid', edgecolor='k',
                                  linewidth=0.5, angles='xy', scale_units='xy', scale=1.25 / spacing)
        fig.colorbar(arrows, ax=ax).set_label(r'$\langle\sigma_z\rangle$', fontsize=fs)
        self._zone_axes(ax, fs)
        # zoom on the contour (within the zone)
        low, high = segs.reshape(-1, 2).min(axis=0), segs.reshape(-1, 2).max(axis=0)
        pad = 0.1 * (high - low).max() + spacing
        ax.set_xlim(max(ax.get_xlim()[0], low[0] - pad), min(ax.get_xlim()[1], high[0] + pad))
        ax.set_ylim(max(ax.get_ylim()[0], low[1] - pad), min(ax.get_ylim()[1], high[1] + pad))
        ax.set_title('Spin texture of band {}, $E = {:g}$'.format(band, energy), fontsize=fs)
        plt.draw()
        return fig

    def plot_berry_curvature(
        self, bands: int | list[int], nk: int | tuple[int, int] = 60,
        plane: tuple[int, int] = (0, 1), k_fixed: float = 0., cmap: str = 'RdBu_r',
        fs: float = 20, figsize: tuple[float, float] | None = None,
    ) -> Figure:
        r'''
        Plot the Berry curvature of *berry_curvature* as a density,
        :math:`\Omega(\mathbf{k})` = flux of each plaquette / its area. In
        2D, the plaquettes are brought into the first Brillouin zone (drawn
        in black), in the k coordinates of *get_ham*. In 3D, they fill the
        parallelogram spanned by the two reciprocal vectors of *plane*, in
        orthonormal coordinates of the plane (:math:`k_1` along
        :math:`\mathbf{b}_i`). The title gives the Chern number, the total
        flux over :math:`2\pi`.

        :param bands: See *berry_curvature*.
        :param nk: Default value 60. See *berry_curvature*.
        :param plane: See *berry_curvature*.
        :param k_fixed: See *berry_curvature*.
        :param cmap: Default value 'RdBu_r'. Colormap, centred on zero.
        :param fs: Positive number. Default value 20. Fontsize.
        :param figsize: Tuple. Default value None. Figure size.

        :returns:
            * **fig** -- Figure.
        '''
        error_handling.positive_real(fs, 'fs')
        error_handling.tuple_2elem(figsize, 'figsize')
        curv = self.berry_curvature(bands, nk, plane, k_fixed)
        n1, n2 = curv.shape
        # the corners of each plaquette, counterclockwise, in fractional coordinates
        f1, f2 = np.meshgrid(np.arange(n1) / n1, np.arange(n2) / n2, indexing='ij')
        corner = np.stack([f1.ravel(), f2.ravel()], axis=-1)[:, None, :]
        quads = corner + np.array([[0., 0.], [1. / n1, 0.], [1. / n1, 1. / n2], [0., 1. / n2]])
        b_i, b_j = self.rec_vec_k[plane[0]], self.rec_vec_k[plane[1]]
        if self.dim == 2:
            quads = self._fold(quads)
        else:
            # orthonormal coordinates of the plane, the first along b_i
            u = b_i / np.linalg.norm(b_i)
            v = b_j - (b_j @ u) * u
            b_i, b_j = np.array([b_i @ u, 0.]), np.array([b_j @ u, b_j @ v / np.linalg.norm(v)])
            quads = quads @ np.array([b_i, b_j])
        area = abs(b_i[0] * b_j[1] - b_i[1] * b_j[0]) / (n1 * n2)
        omega = curv.ravel() / area
        vmax = np.abs(omega).max() or 1.
        fig, ax = plt.subplots(figsize=figsize)
        patches = PolyCollection(quads, array=omega, cmap=cmap, norm=Normalize(-vmax, vmax),
                                          edgecolors='face', linewidths=0.2)
        ax.add_collection(patches)
        fig.colorbar(patches, ax=ax).set_label(r'$\Omega(\mathbf{k})$', fontsize=fs)
        if self.dim == 2:
            self._zone_axes(ax, fs)
        else:
            ax.autoscale_view()
            ax.set_aspect('equal')
            ax.set_xlabel('$k_1$', fontsize=fs)
            ax.set_ylabel('$k_2$', fontsize=fs)
        ax.set_title('Berry curvature, $C = {:.3f}$'.format(curv.sum() / (2 * PI)), fontsize=fs)
        fig.set_layout_engine('tight')
        plt.draw()
        return fig

    def plot_wannier_flow(
        self, bands: int | list[int], nk: int = 100, nk_perp: int = 51, direction: int = 0,
        flow: int | None = None, k_fixed: float = 0., positions: bool = True,
        k_range: tuple[float, float] = (0., 1.), ms: float = 4., c: str = 'b', fs: float = 20,
        figsize: tuple[float, float] | None = None,
    ) -> Figure:
        r'''
        Plot the flow of the hybrid Wannier centres (the Wilson-loop
        spectrum) of *wannier_flow*: the centres along *direction*, in
        units of :math:`\mathbf{a}_{direction}`, as the loop moves along
        the reciprocal vector *flow*. They wind :math:`C` times in a Chern
        insulator, and the Kramers pairs swap partners in a
        :math:`\mathbb{Z}_2` topological insulator.

        :param bands: See *wannier_flow*.
        :param nk: See *wannier_flow*.
        :param nk_perp: See *wannier_flow*.
        :param direction: See *wannier_flow*.
        :param flow: See *wannier_flow*.
        :param k_fixed: See *wannier_flow*.
        :param positions: See *wannier_flow*.
        :param k_range: See *wannier_flow*.
        :param ms: Positive number. Default value 4. Marker size.
        :param c: Default value 'b'. Marker color.
        :param fs: Positive number. Default value 20. Fontsize.
        :param figsize: Tuple. Default value None. Figure size.

        :returns:
            * **fig** -- Figure.
        '''
        error_handling.positive_real(ms, 'ms')
        error_handling.positive_real(fs, 'fs')
        error_handling.tuple_2elem(figsize, 'figsize')
        fracs, centers = self.wannier_flow(bands, nk, nk_perp, direction, flow, k_fixed,
                                                         positions, k_range)
        if flow is None:
            flow = [d for d in range(self.dim) if d != direction][0]
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(fracs, centers, 'o', c=c, ms=ms)
        ax.set_xlim(fracs[0], fracs[-1])
        ax.set_ylim(0., 1.)
        ax.set_xlabel(r'$k_{0}/|\mathbf{{b}}_{0}|$'.format(flow + 1), fontsize=fs)
        ax.set_ylabel(r'Wannier centres ($\mathbf{{a}}_{}$)'.format(direction + 1), fontsize=fs)
        fig.set_layout_engine('tight')
        plt.draw()
        return fig

    def plot_surface_spectral_function(
        self, points: list[ArrayLike], energies: ArrayLike, direction: int, nk: int = 60,
        side: int = 1, eta: float = 1e-2, bulk: bool = False,
        node_labels: list[str] | None = None, log: bool = True, cmap: str = 'magma',
        fs: float = 20, figsize: tuple[float, float] | None = None,
    ) -> Figure:
        r'''
        Plot the surface spectral function of *surface_spectral_function*
        along a path through the surface Brillouin zone (built as in
        *k_path*), as an ARPES-like map of :math:`E` against
        :math:`\mathbf{k}_\parallel`. Surface states are the sharp lines in
        the bulk gaps.

        :param points: List of at least two k-points, the nodes of the path
            (see *k_path*).
        :param energies: Real array. Energies :math:`E`.
        :param direction: See *surface_spectral_function*.
        :param nk: Positive integer. Default value 60. Number of k-points per path segment.
        :param side: See *surface_spectral_function*.
        :param eta: See *surface_spectral_function*.
        :param bulk: See *surface_spectral_function*.
        :param node_labels: List of strings. Default value None. Labels of *points*.
        :param log: Boolean. Default value True. Logarithmic color scale, over
            four decades below the maximum (the surface states are much
            sharper than the bulk continuum).
        :param cmap: Default value 'magma'. Colormap.
        :param fs: Positive number. Default value 20. Fontsize.
        :param figsize: Tuple. Default value None. Figure size.

        :returns:
            * **fig** -- Figure.
        '''
        error_handling.k_path_points(points, self.dim)
        error_handling.positive_int(nk, 'nk')
        error_handling.boolean(log, 'log')
        error_handling.positive_real(fs, 'fs')
        error_handling.tuple_2elem(figsize, 'figsize')
        if node_labels is not None:
            error_handling.ndarray(np.array(node_labels), 'node_labels', len(points))
        ks, dist, nodes = _path(points, nk)
        spec = self.surface_spectral_function(ks, energies, direction, side, eta, bulk)
        energies = np.atleast_1d(np.asarray(energies, dtype='f8'))
        vmax = spec.max() if spec.max() > 0 else 1.
        norm = LogNorm(vmax * 1e-4, vmax) if log else Normalize(0., vmax)
        fig, ax = plt.subplots(figsize=figsize)
        mesh = ax.pcolormesh(dist, energies, np.maximum(spec, vmax * 1e-4).T, cmap=cmap,
                                       norm=norm, shading='nearest')
        fig.colorbar(mesh, ax=ax).set_label(r'$A(\mathbf{k}_\parallel, E)$', fontsize=fs)
        for node in nodes:
            ax.axvline(node, color='w', lw=0.5)
        if node_labels is not None:
            ax.set_xticks(nodes)
            ax.set_xticklabels(node_labels, fontsize=fs)
        ax.set_ylabel('$E$', fontsize=fs)
        ax.set_title('Bulk' if bulk else 'Surface', fontsize=fs)
        fig.set_layout_engine('tight')
        plt.draw()
        return fig

    def plot_dos(
        self,
        nk: int | tuple[int, ...] = 30,
        broadening: float = 0.05,
        kernel: str = 'gaussian',
        e_grid: ArrayLike | None = None,
        fs: float = 20,
        lw: float = 2.,
        figsize: tuple[float, float] | None = None,
    ) -> Figure:
        '''
        Plot the density of states, obtained by diagonalizing
        :math:`H(\\mathbf{k})` over a uniform Brillouin-zone mesh: broadened
        (see *tbkit.dos.density_of_states*), or by the linear tetrahedron
        method (``kernel='tetrahedron'``, see *tbkit.dos.tetrahedron_dos*),
        which needs no broadening and keeps band edges and van Hove
        singularities sharp.

        :param nk: Positive integer, or tuple of *dim* positive integers.
            Default value 30. Number of k-points along each reciprocal
            lattice vector.
        :param broadening: Positive real number. Default value 0.05. Kernel
            width (unused by the tetrahedron method).
        :param kernel: String. Default value 'gaussian'. 'gaussian',
            'lorentzian' or 'tetrahedron'.
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
        error_handling.kspace_dos_kernel(kernel)
        en = self.mesh_bands(nk)
        if kernel != 'tetrahedron':
            return dos._plot_density_of_states(en, e_grid, broadening, kernel, fs, lw, figsize)
        nk = (nk,) * self.dim if isinstance(nk, int) else nk
        e_grid, rho = dos.tetrahedron_dos(en.reshape(*nk, self.norb), e_grid)
        return dos._plot_dos_curve(e_grid, rho, fs, lw, figsize)

    def plot_bands(
        self,
        node_labels: list[str] | None = None,
        fs: float = 20,
        lw: float = 2.,
        ms: float = 0.,
        c: str = 'b',
        lims: tuple[float, float] | None = None,
        figsize: tuple[float, float] | None = None,
        weights: ArrayLike | None = None,
        style: str = 'color',
        cmap: str | None = None,
    ) -> Figure:
        '''
        Plot the band structure computed by *k_path* or *get_bands*, against
        the cumulative distance along the k-points (vertical lines mark the
        nodes of a *k_path*). For a non-Hermitian model, the real part of
        the energies is plotted. With *weights* (e.g. from *band_weights*:
        an orbital, sublattice or spin projection), a "fat band" plot: the
        bands are colored by the weights (with a colorbar), or drawn with
        dots sized by their absolute value.

        :param node_labels: List of strings. Default value None. Labels of the
            high-symmetry points passed to *k_path*.
        :param fs: Positive number. Default value 20. Fontsize.
        :param lw: Positive number. Default value 2. Linewidth.
        :param ms: Positive number. Default value 0. Marker size.
        :param c: Default value 'b'. Line color.
        :param lims: List. Default value None. Energy plot limits.
        :param figsize: Tuple. Default value None. Figure size.
        :param weights: Real array, shape (nk, norb). Default value None.
            One weight per k-point and band, e.g. from *band_weights*.
        :param style: String. Default value 'color'. 'color': the bands
            colored by *weights*; 'size': dots of size proportional to
            :math:`|w|` (*ms*, or 8 if *ms* is 0, for the largest) on thin
            lines of color *c*.
        :param cmap: Default value None ('viridis', or 'RdBu_r', centred on
            zero, if some weights are negative). Colormap of *style* 'color'.

        :returns:
            * **fig** -- Figure.
        '''
        error_handling.empty_ndarray(self.en, 'get_bands or k_path')
        error_handling.positive_real(fs, 'fs')
        error_handling.positive_real(lw, 'lw')
        error_handling.lims(lims)
        error_handling.tuple_2elem(figsize, 'figsize')
        fig, ax = plt.subplots(figsize=figsize)
        if weights is None:
            for n in range(self.norb):
                ax.plot(self.ks_dist, self.en[:, n].real, c=c, lw=lw, marker='o', ms=ms)
        else:
            self._plot_fat_bands(fig, ax, weights, style, cmap, c, lw, ms)
        for node in self.nodes:
            ax.axvline(node, color='k', lw=0.5)
        ax.set_xlim(self.ks_dist[0], self.ks_dist[-1])
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

    def _plot_fat_bands(self, fig, ax, weights, style, cmap, c, lw, ms) -> None:
        '''
        Private method. The bands of *plot_bands* colored (or with dots
        sized) by *weights*.
        '''
        error_handling.band_weights(weights, self.en.shape)
        error_handling.weight_style(style)
        weights = np.asarray(weights, dtype='f8')
        signed = weights.min() < 0
        vmax = np.abs(weights).max() or 1.
        if style == 'size':
            size = (ms or 8.) ** 2
            for n in range(self.norb):
                ax.plot(self.ks_dist, self.en[:, n].real, c=c, lw=lw / 4)
                ax.scatter(self.ks_dist, self.en[:, n].real, s=size * np.abs(weights[:, n]) / vmax,
                                c=c, lw=0)
            return
        norm = Normalize(-vmax if signed else 0., vmax)
        cmap = cmap or ('RdBu_r' if signed else 'viridis')
        for n in range(self.norb):
            pts = np.stack([self.ks_dist, self.en[:, n].real], axis=-1)
            lines = LineCollection(list(np.stack([pts[:-1], pts[1:]], axis=1)), cmap=cmap, norm=norm, lw=lw)
            lines.set_array((weights[:-1, n] + weights[1:, n]) / 2)
            ax.add_collection(lines)
        ax.autoscale_view()
        fig.colorbar(lines, ax=ax)

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


#################################
# K-PATHS AND CONSTANT-ENERGY CONTOURS
#################################


def _response_weights(
    en: NDArray[np.float64], mu: float, temperature: float, response: str,
) -> NDArray[np.float64]:
    r'''
    Private. The weights :math:`P_{nm}` of the Kubo pair sum of
    *KSpace._kubo*, shape (nk, n, n), from the bands *en* (nk, n). With
    :math:`x_n = (E_n-\mu)/T`, the conductivities are
    :math:`\sum_n w_n\Omega_n`, i.e. :math:`P_{nm} = -(w_n - w_m)`, with
    :math:`w = f` ('hall'), the entropy per state
    :math:`s = -f\ln f-(1-f)\ln(1-f)` ('nernst') and
    :math:`c_2/T = T\int_x^\infty y^2(-f'(y))\,dy` ('thermal'). The orbital
    magnetization ('magnetization'),
    :math:`\sum_n[f_n\,\mathrm{Im}\langle\partial_xu_n|(H-E_n)|\partial_yu_n\rangle + g_n\Omega_n]`
    with :math:`g = T\ln(1+e^{-x})` (:math:`\max(\mu-E, 0)` at :math:`T = 0`),
    has :math:`P_{nm} = \frac12(f_n+f_m)(E_m-E_n) - (g_n-g_m)`.
    '''
    if response == 'magnetization':
        f = occupation.fermi_dirac(en, mu, temperature)
        if temperature == 0:
            g = np.maximum(mu - en, 0.)
        else:
            g = temperature * np.logaddexp(0., -(en - mu) / temperature)
        return 0.5 * (f[:, :, None] + f[:, None, :]) * (en[:, None, :] - en[:, :, None]) \
            - (g[:, :, None] - g[:, None, :])
    if response == 'hall':
        w = occupation.fermi_dirac(en, mu, temperature)
    else:
        a = np.abs(en - mu) / temperature
        if response == 'nernst':
            w = np.log1p(np.exp(-a)) + a * expit(-a)
        else:
            # int_a^inf y^2 (-f') dy = a^2 f(a) + 2 [a ln(1 + e^-a) - Li2(-e^-a)]
            # for a >= 0, and pi^2/3 minus it below the Fermi level
            # (Li2(z) = spence(1 - z) in scipy's convention)
            c2 = a ** 2 * expit(-a) + 2 * (a * np.log1p(np.exp(-a)) - spence(1 + np.exp(-a)))
            w = temperature * np.where(en >= mu, c2, PI ** 2 / 3 - c2)
    return -(w[:, :, None] - w[:, None, :])


def _path(
    points: list[ArrayLike], nk: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64], list[float]]:
    '''
    Private function. The k-points of a path through *points* (*nk* per
    segment, the last point included), their cumulative distance, and the
    positions of the nodes along it.
    '''
    pts = np.atleast_2d(np.asarray(points, dtype='f8'))
    ks = np.concatenate([np.linspace(pts[i], pts[i+1], nk, endpoint=False)
                                  for i in range(len(pts) - 1)] + [pts[-1:]])
    dist = np.concatenate([[0.], np.cumsum(np.linalg.norm(np.diff(ks, axis=0), axis=1))])
    return ks, dist, dist[::nk][:len(pts)-1].tolist() + [dist[-1]]


def _marching_table(dim: int) -> dict[int, list[list[tuple[int, int]]]]:
    '''
    Private function. For each pattern of the vertices of a triangle
    (*dim* = 2) or tetrahedron (*dim* = 3) above a level (bit m set if
    vertex m is above), the pieces of the level set, as lists of the edges
    (a, b) that they cross: one segment in 2D; one triangle, or two
    triangles forming a quadrilateral when two vertices are above, in 3D.
    '''
    table = {}
    for pattern in range(1, 2 ** (dim + 1) - 1):
        up = [m for m in range(dim + 1) if pattern >> m & 1]
        down = [m for m in range(dim + 1) if not pattern >> m & 1]
        if dim == 3 and len(up) == 2:
            (a, b), (c, d) = up, down
            table[pattern] = [[(a, c), (a, d), (b, d)], [(a, c), (b, d), (b, c)]]
        else:
            table[pattern] = [[(a, b) for a in up for b in down]]
    return table


def _level_set(values: NDArray[np.float64], level: float) -> NDArray[np.float64]:
    '''
    Private function. The level set ``values == level`` of a function on a
    periodic 2D or 3D mesh (*values* of shape (n1, n2) or (n1, n2, n3)),
    linear in each simplex of *dos._mesh_simplices*: segments (shape
    (n, 2, 2)) or triangles (shape (n, 3, 3)), in fractional coordinates
    (not folded: between 0 and 1 + 1/n_d).
    '''
    shape, dim = values.shape, values.ndim
    offsets = dos._mesh_simplices(dim)
    vals = dos._simplex_values(values - level, offsets).reshape(dim + 1, -1)
    pattern = sum((vals[m] > 0).astype(int) << m for m in range(dim + 1))
    pieces = [np.empty((0, dim, dim))]
    for pat, polygons in _marching_table(dim).items():
        sel = np.nonzero(pattern == pat)[0]
        if not len(sel):
            continue
        simplex, cell = np.divmod(sel, int(np.prod(shape)))
        corner = np.stack(np.unravel_index(cell, shape), axis=-1)
        pos = corner[None] + offsets[simplex].transpose(1, 0, 2)
        v = vals[:, sel]
        for polygon in polygons:
            # the crossing on edge (a, b): v[a] > 0 >= v[b] or the reverse
            pieces.append(np.stack([pos[a] + (v[a] / (v[a] - v[b]))[:, None] * (pos[b] - pos[a])
                                              for a, b in polygon], axis=1))
    return np.concatenate(pieces) / np.array(shape)


def _zone_polygon(rec: NDArray[np.float64]) -> NDArray[np.float64]:
    '''
    Private function. Vertices, counterclockwise, of the first Brillouin
    zone (Wigner-Seitz cell) of the 2D reciprocal lattice with basis *rec*
    (rows): the points at least as close to the origin as to any other
    reciprocal lattice vector, where two of the bisectors meet.
    '''
    gs = np.array([n @ rec for n in product(range(-2, 3), repeat=2) if any(n)])
    half = np.sum(gs ** 2, axis=1) / 2
    tol = 1e-9 * half.max()
    found = []
    for i in range(len(gs)):
        for j in range(i + 1, len(gs)):
            mat = gs[[i, j]]
            if abs(np.linalg.det(mat)) > tol:
                k = np.linalg.solve(mat, half[[i, j]])
                if np.all(gs @ k <= half + tol):
                    found.append(k)
    verts = np.unique(np.round(np.array(found), 9), axis=0)
    return verts[np.argsort(np.arctan2(verts[:, 1], verts[:, 0]))]


def _gauss_reduce(u: NDArray[np.float64], v: NDArray[np.float64]):
    '''
    Private function. Gauss-reduced basis of a 2D lattice: the two
    shortest independent vectors, |u| <= |v|, at an obtuse (or right)
    angle, so that the edges of the Wigner-Seitz cell are normal to
    u, v and u + v.
    '''
    while True:
        if u @ u > v @ v:
            u, v = v, u
        m = np.round(u @ v / (u @ u))
        if m == 0:
            break
        v = v - m * u
    return (u, v - u) if u @ v > 0 else (u, v)


def _successive_minima(vecs: NDArray[np.float64]):
    '''
    Private function. The shortest lattice vectors of a 3D lattice with
    basis *vecs* (rows): the three successive minima, the shells of
    lattice vectors sorted by length (list of arrays), and their lengths.
    '''
    pts = np.array([n @ vecs for n in product(range(-3, 4), repeat=3) if any(n)])
    lengths = np.linalg.norm(pts, axis=1)
    # by length, then (among equal lengths) the most positive first: x, then y, then z
    order = np.lexsort((-pts[:, 2], -pts[:, 1], -pts[:, 0], np.round(lengths / lengths.min(), 6)))
    pts, lengths = pts[order], lengths[order]
    minima: list = []
    for p in pts:
        if np.linalg.matrix_rank(np.array(minima + [p]), tol=1e-8 * lengths[0]) == len(minima) + 1:
            minima.append(p)
        if len(minima) == 3:
            break
    bounds = np.nonzero(np.diff(lengths) > 1e-6 * lengths[0])[0] + 1
    return np.array(minima), np.split(pts, bounds), [s[0] for s in np.split(lengths, bounds)]


def _orthogonal_axes(shell: NDArray[np.float64]) -> NDArray[np.float64]:
    '''
    Private function. Three mutually orthogonal unit vectors among the
    vectors of *shell* (the conventional cubic axes).
    '''
    x = shell[0] / np.linalg.norm(shell[0])
    y = next(s for s in shell if abs(s @ x) < 1e-6 * np.linalg.norm(s))
    y = y / np.linalg.norm(y)
    return np.array([x, y, np.cross(x, y)])


def _bravais_3d(vecs: NDArray[np.float64]):
    '''
    Private function. The points and labels of the high-symmetry path of a
    3D lattice (basis *vecs*, rows), or (None, None) if it is not one of
    the simple cubic, fcc, bcc, simple tetragonal or hexagonal lattices.
    '''
    minima, shells, radii = _successive_minima(vecs)
    lengths = np.linalg.norm(minima, axis=1)
    pairs = [(0, 1), (0, 2), (1, 2)]
    cosine = {p: abs(minima[p[0]] @ minima[p[1]]) / (lengths[p[0]] * lengths[p[1]]) for p in pairs}
    equal = {p: abs(lengths[p[0]] - lengths[p[1]]) < 1e-6 * lengths[2] for p in pairs}
    right = {p: cosine[p] < 1e-6 for p in pairs}
    G = r'$\Gamma$'

    def cubic_path(pts, names, axes, a):
        return [2 * PI / a * np.array(pts[n]) @ axes for n in names], names
    if all(right.values()):
        if all(equal.values()):  # simple cubic
            return cubic_path({G: (0, 0, 0), 'X': (0, .5, 0), 'M': (.5, .5, 0), 'R': (.5, .5, .5)},
                                   [G, 'X', 'M', G, 'R', 'X'], minima / lengths[:, None], lengths[0])
        if sum(equal.values()) != 1:  # orthorhombic
            return None, None
        # simple tetragonal: a, a, c
        (i, j), = [p for p in pairs if equal[p]]
        k = 3 - i - j
        a, c = lengths[i], lengths[k]
        axes = np.array([minima[i] / a, minima[j] / a, minima[k] / c])
        pts = {G: (0, 0, 0), 'X': (0, .5, 0), 'M': (.5, .5, 0), 'Z': (0, 0, .5),
                 'R': (0, .5, .5), 'A': (.5, .5, .5)}
        names = [G, 'X', 'M', G, 'Z', 'R', 'A', 'Z']
        scale = 2 * PI * np.array([1 / a, 1 / a, 1 / c])
        return [(scale * pts[n]) @ axes for n in names], names
    for (i, j) in pairs:
        k = 3 - i - j
        if equal[(i, j)] and abs(cosine[(i, j)] - .5) < 1e-6 and \
                all(right[p] for p in pairs if p != (i, j)):
            # hexagonal: reciprocal vectors of (a1, a2, c), in-plane ones at 120 degrees
            rec = 2 * PI * np.linalg.inv(minima[[i, j, k]]).T
            b1, b2 = _gauss_reduce(rec[0], rec[1])
            M, K, A = b1 / 2, (2 * b1 + b2) / 3, rec[2] / 2
            hex_pts = {G: np.zeros(3), 'M': M, 'K': K, 'A': A, 'L': M + A, 'H': K + A}
            names = [G, 'M', 'K', G, 'A', 'L', 'H', 'A']
            return [hex_pts[n] for n in names], names
    if not all(equal.values()):
        return None, None
    if len(shells[0]) == 12:  # fcc: 12 nearest neighbours at a/sqrt(2), then 6 at a
        return cubic_path({G: (0, 0, 0), 'X': (0, 1, 0), 'W': (.5, 1, 0), 'K': (.75, .75, 0),
                                  'L': (.5, .5, .5), 'U': (.25, 1, .25)},
                                 [G, 'X', 'W', 'K', G, 'L', 'U', 'W', 'L', 'K'],
                                 _orthogonal_axes(shells[1]), radii[1])
    if len(shells[0]) == 8 and all(abs(c - 1 / 3) < 1e-6 for c in cosine.values()):
        # bcc: 8 nearest neighbours at a sqrt(3)/2, then 6 at a
        return cubic_path({G: (0, 0, 0), 'H': (0, 0, 1), 'N': (.5, 0, .5), 'P': (.5, .5, .5)},
                                 [G, 'H', 'N', G, 'P', 'H'], _orthogonal_axes(shells[1]), radii[1])
    return None, None


def high_symmetry_path(lat: Lattice) -> tuple[list[NDArray[np.float64]], list[str]]:
    r'''
    Get a standard path through the high-symmetry points of the Brillouin
    zone of a lattice, for *KSpace.k_path* and the labels of
    *KSpace.plot_bands*. The Bravais lattice is recognized from
    *lat.prim_vec*, whatever the choice of primitive vectors and the
    orientation of the lattice:

    * 1D: :math:`-X, \Gamma, X` (the whole zone, :math:`X = \mathbf{b}/2`).
    * 2D (the five Bravais lattices): square :math:`\Gamma X M \Gamma`;
      rectangular :math:`\Gamma X S Y \Gamma`; hexagonal
      :math:`\Gamma M K \Gamma`; centred rectangular and oblique
      :math:`\Gamma X H C H_1 Y \Gamma`, along the edges of the
      zone: :math:`X, Y, C` are the midpoints of the edges normal to the
      reciprocal vectors :math:`\mathbf{b}_1, \mathbf{b}_2,
      \mathbf{b}_1+\mathbf{b}_2` of the reduced basis (with
      :math:`|\mathbf{b}_1| \le |\mathbf{b}_2|`), :math:`H` and
      :math:`H_1` the corners between them.
    * 3D: simple cubic :math:`\Gamma X M \Gamma R X`; fcc
      :math:`\Gamma X W K \Gamma L U W L K`; bcc :math:`\Gamma H N \Gamma P H`;
      simple tetragonal :math:`\Gamma X M \Gamma Z R A Z`; hexagonal
      :math:`\Gamma M K \Gamma A L H A`. These are the paths of Setyawan
      and Curtarolo (Comput. Mater. Sci. 49, 299 (2010)), up to their
      first discontinuity (a path here is one continuous line).

    The other 3D lattices (orthorhombic, monoclinic, body-centred
    tetragonal, rhombohedral, ...) raise a ValueError: give their points
    to *k_path* directly.

    :param lat: **lattice** class instance (only *prim_vec* is used).

    :returns:
        * **points** -- List of real ndarrays, the k-points, in the
          coordinates of *KSpace.get_ham*.
        * **labels** -- List of strings (LaTeX), for *node_labels*.

    Example usage::

        points, labels = high_symmetry_path(lat)
        ks.k_path(points, nk=60)
        ks.plot_bands(node_labels=labels)
    '''
    ks = KSpace(lat)
    vecs = np.array(lat.prim_vec, dtype='f8') @ ks.k_basis  # in the k coordinates
    G = r'$\Gamma$'
    if ks.dim == 1:
        x = PI / vecs[0]
        return [-x, np.zeros(1), x], ['$-X$', G, 'X']
    if ks.dim == 3:
        points, labels = _bravais_3d(vecs)
        error_handling.bravais_lattice(labels, 3)
        return points, labels
    rec = 2 * PI * np.linalg.inv(vecs).T
    b1, b2 = _gauss_reduce(rec[0], rec[1])
    l1, l2, l3 = (np.linalg.norm(b) for b in (b1, b2, b1 + b2))
    close = lambda a, b: abs(a - b) < 1e-6 * max(abs(a), abs(b))
    cosine = b1 @ b2 / (l1 * l2)
    if abs(cosine) < 1e-6:
        if close(l1, l2):
            return [np.zeros(2), b1 / 2, (b1 + b2) / 2, np.zeros(2)], [G, 'X', 'M', G]
        return [np.zeros(2), b1 / 2, (b1 + b2) / 2, b2 / 2, np.zeros(2)], [G, 'X', 'S', 'Y', G]
    if close(l1, l2) and close(cosine, -.5):
        return [np.zeros(2), b1 / 2, (2 * b1 + b2) / 3, np.zeros(2)], [G, 'M', 'K', G]
    # centred rectangular and oblique: around half of the hexagonal zone
    b3 = b1 + b2
    corner = lambda u, v: np.linalg.solve(np.array([u, v]), np.array([u @ u, v @ v]) / 2)
    return ([np.zeros(2), b1 / 2, corner(b1, b3), b3 / 2, corner(b2, b3), b2 / 2, np.zeros(2)],
               [G, 'X', 'H', 'C', '$H_1$', 'Y', G])
