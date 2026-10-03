r"""
Supercells, band unfolding and moiré bilayers.

* :func:`supercell` repeats a periodic model (:class:`tbkit.kspace.KSpace`)
  in a larger cell spanned by integer combinations of its primitive vectors,
  :math:`\mathbf{A}_i = \sum_j M_{ij}\mathbf{a}_j`: the natural home of
  disorder, defects and superstructures that the primitive cell cannot hold.
* :func:`unfold` and :func:`spectral_function` project the supercell
  eigenstates back onto the primitive Brillouin zone (Ku, Berlijn and Lee,
  Phys. Rev. Lett. 104, 216401 (2010); Popescu and Zunger, Phys. Rev. B
  85, 085201 (2012)): the weight of state :math:`|\mathbf{K}J\rangle` at the
  primitive :math:`\mathbf{k}` (folding onto :math:`\mathbf{K}`) is

  .. math::

      W_J(\mathbf{k}) = \frac{1}{N}\sum_o\Big|\sum_{m=1}^{N}
      e^{-i\mathbf{k}\cdot\mathbf{r}_m}\,\psi_J(m, o)\Big|^2\, ,

  :math:`\mathbf{r}_m` the origins of the :math:`N` primitive cells in the
  supercell and :math:`o` the primitive orbitals, so that
  :math:`A(\mathbf{k},\omega) = \sum_J W_J(\mathbf{k})\,\delta(\omega - E_J)`.
  A pristine supercell unfolds exactly onto the primitive bands
  (:math:`W = 1` on them, 0 elsewhere); disorder smears them.
* :func:`twisted_bilayer` builds the commensurate supercells of two
  honeycomb (twisted bilayer graphene) or square layers twisted by
  :math:`\theta`, with every hopping a function of the distance
  (:func:`pz_hopping`, the Slater-Koster :math:`p_z` form of Moon and
  Koshino, Phys. Rev. B 85, 195458 (2012), by default).
  :func:`magic_angle_parameter` gives the Bistritzer-MacDonald
  :math:`\alpha = w/\hbar v k_\theta` and Fermi velocity
  :math:`v^*/v = (1-3\alpha^2)/(1+6\alpha^2)` of the same model.

Every Bloch matrix comes from :meth:`KSpace._bloch_ham <tbkit.kspace.KSpace>`
(the supercells are ordinary **KSpace** models: *k_path*, *mesh_bands*,
*finite_ham*, the topology tools all apply).
"""
from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.integrate import quad
from scipy.spatial import cKDTree
from scipy.special import j0

import tbkit.error_handling as error_handling
from tbkit.kspace import KSpace
from tbkit.lattice import Lattice


PI = np.pi


class SupercellKSpace(KSpace):
    r'''
    A **KSpace** made of :math:`N` copies of a primitive model (see
    *supercell*). Orbital *o* of primitive cell *m* is orbital
    ``m*prim.norb + o``.

    :ivar prim: **KSpace** instance. The primitive model.
    :ivar matrix: Integer ndarray, shape (dim, dim). Supercell matrix *M*.
    :ivar cells: Real ndarray, shape (N, space_dim). Origins
        :math:`\mathbf{r}_m` of the primitive cells in the supercell.
    :ivar n_cells: Integer. :math:`N = |\det M|`.
    '''

    def __init__(self, lat: Lattice, spin: bool, prim: KSpace, matrix: NDArray, cells: NDArray) -> None:
        super().__init__(lat, spin)
        self.prim = prim
        self.matrix = matrix
        self.cells = cells
        self.n_cells = len(cells)


class MoireKSpace(KSpace):
    r'''
    A **KSpace** of a twisted bilayer (see *twisted_bilayer*): the orbitals
    of layer 0 (tags 'a', 'b', rotated by :math:`-\theta/2`) then those of
    layer 1 (tags 'c', 'd', rotated by :math:`+\theta/2`, a height *d*
    above).

    :ivar theta: Real number. Twist angle, in radians.
    :ivar layer: Integer ndarray. Layer (0 or 1) of every orbital.
    :ivar z: Real ndarray. Height of every orbital.
    :ivar n_layer_cells: Integer. Primitive cells per layer in the supercell.
    '''

    def __init__(self, lat: Lattice, theta: float, layer: NDArray, z: NDArray, n_layer_cells: int) -> None:
        super().__init__(lat)
        self.theta = theta
        self.layer = layer
        self.z = z
        self.n_layer_cells = n_layer_cells


def _inner_cells(matrix: NDArray) -> NDArray:
    '''
    Private function. Integer lattice points n (rows) inside the supercell
    spanned by the rows of *matrix*: n M^-1 in [0, 1)^dim, sorted.
    '''
    dim = len(matrix)
    corners = np.array(np.meshgrid(*[[0, 1]] * dim, indexing='ij')).reshape(dim, -1).T @ matrix
    ranges = [np.arange(corners[:, d].min(), corners[:, d].max() + 1) for d in range(dim)]
    pts = np.array(np.meshgrid(*ranges, indexing='ij')).reshape(dim, -1).T
    frac = pts @ np.linalg.inv(matrix)
    eps = 1e-9
    keep = np.all((frac > -eps) & (frac < 1 - eps), axis=1)
    return pts[keep]


def supercell(ks: KSpace, matrix: ArrayLike) -> SupercellKSpace:
    r'''
    Get the supercell of a periodic model spanned by
    :math:`\mathbf{A}_i = \sum_j M_{ij}\mathbf{a}_j`, a **KSpace** whose unit
    cell holds the :math:`N = |\det M|` primitive cells inside it. The bands
    of the supercell are those of the primitive model folded into the
    smaller Brillouin zone; *unfold* undoes the folding. Break the
    translation symmetry afterwards by changing the supercell's
    ``onsite`` array (one entry per orbital), e.g. with random energies.

    :param ks: **KSpace** instance (any dimension; spinful or not).
    :param matrix: Integer array, shape (dim, dim), nonzero determinant
        (an integer *n* in 1D is also accepted).

    :returns:
        * **sc** -- :class:`SupercellKSpace` instance, ``N*ks.norb`` orbitals:
          orbital *o* of cell *m* (origin ``sc.cells[m]``) is ``m*ks.norb + o``.

    Example usage::

        # a 3 x 3 supercell of graphene, 18 orbitals
        sc = supercell(gra, [[3, 0], [0, 3]])
    '''
    error_handling.kspace(ks, KSpace)
    if isinstance(matrix, int):
        matrix = [[matrix]]
    matrix = np.asarray(matrix)
    error_handling.supercell_matrix(matrix, ks.dim)
    matrix = matrix.astype(int)
    a = np.array(ks.lat.prim_vec, dtype='f8')
    cells = _inner_cells(matrix)
    origins = cells @ a
    cell = []
    for r in origins:
        for dic in ks.lat.unit_cell:
            cell.append({'tag': dic['tag'], 'r0': tuple(float(c) for c in np.array(dic['r0']) + r)})
    big = matrix @ a
    lat = Lattice(unit_cell=cell, prim_vec=[tuple(float(c) for c in v) for v in big])
    sc = SupercellKSpace(lat, ks.spin, ks, matrix, origins)
    n_cells = len(cells)
    lookup = {tuple(c): m for m, c in enumerate(cells)}
    inv = np.linalg.inv(matrix)
    rec = np.array(ks.rec_vec, dtype='f8')

    def fold(hops):
        out = []
        for i, j, R_cart, t in hops:
            R = np.rint(rec @ R_cart / (2 * PI)).astype(int)
            target = cells + R[None, :]
            shift = np.floor(target @ inv + 1e-9).astype(int)
            inner = target - shift @ matrix
            for m in range(n_cells):
                out.append((m * ks.norb + i, lookup[tuple(inner[m])] * ks.norb + j,
                                 shift[m] @ big, t))
        return out

    sc._hop = fold(ks._hop)
    sc._overlap_hop = fold(ks._overlap_hop)
    sc._nonreciprocal = ks._nonreciprocal
    sc.onsite = np.tile(ks.onsite, n_cells)
    sc._onsite_offdiag = np.kron(np.eye(n_cells), ks._onsite_offdiag)
    return sc


def unfold(sc: SupercellKSpace, ks: ArrayLike) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    r'''
    Unfold the bands of a supercell onto k-points of the primitive
    Brillouin zone (see the module docstring): diagonalize the supercell
    Hamiltonian at each primitive :math:`\mathbf{k}` (which folds onto
    itself modulo the supercell reciprocal lattice), and weigh each
    eigenstate by its overlap with the primitive Bloch states at
    :math:`\mathbf{k}`. The weights at one :math:`\mathbf{k}` add up to the
    number of primitive orbitals; for a pristine supercell they are 1 on
    the primitive bands :math:`E_n(\mathbf{k})` and 0 on the others.

    :param sc: :class:`SupercellKSpace` instance (Hermitian, without overlap).
    :param ks: Real array, shape (nk, dim). Primitive k-points, in the
        coordinates of ``sc.prim.get_ham``.

    :returns:
        * **en** -- Real ndarray, shape (nk, sc.norb). Supercell energies at each k.
        * **weights** -- Real ndarray, shape (nk, sc.norb). Spectral weights :math:`W_J(\mathbf{k})`.
    '''
    error_handling.supercell_model(sc, SupercellKSpace)
    error_handling.hermitian_kspace(sc.is_hermitian(), sc._overlap_hop)
    ks = np.atleast_2d(np.asarray(ks, dtype='f8'))
    error_handling.ks(ks, sc.dim)
    k_cart = ks @ sc.prim.k_basis.T
    en, vec = np.linalg.eigh(sc._bloch_ham(k_cart))
    norb = sc.prim.norb
    psi = vec.reshape(len(ks), sc.n_cells, norb, sc.norb)
    phase = np.exp(-1j * k_cart @ sc.cells.T)  # (nk, N)
    amp = np.einsum('km,kmos->kos', phase, psi)
    weights = (np.abs(amp) ** 2).sum(axis=1) / sc.n_cells
    return en, weights


def spectral_function(
    sc: SupercellKSpace, ks: ArrayLike, energies: ArrayLike, broadening: float = 0.05,
    kernel: str = 'lorentzian',
) -> NDArray[np.float64]:
    r'''
    Get the unfolded spectral function of a supercell,

    .. math::

        A(\mathbf{k}, \omega) = \sum_J W_J(\mathbf{k})\, g(\omega - E_J)\, ,

    with the weights of *unfold* and a normalized kernel *g* (Lorentzian or
    Gaussian of width *broadening*), so that :math:`\int A\,d\omega` is
    the number of primitive orbitals. For a disordered supercell, average
    it over disorder configurations.

    :param sc: :class:`SupercellKSpace` instance.
    :param ks: Real array, shape (nk, dim). Primitive k-points.
    :param energies: Real array. Energies :math:`\omega`.
    :param broadening: Positive real. Default value 0.05. Kernel width.
    :param kernel: 'lorentzian' or 'gaussian'. Default value 'lorentzian'.

    :returns:
        * **A** -- Real ndarray, shape (nk, len(energies)).
    '''
    energies = np.atleast_1d(np.asarray(energies, dtype='f8'))
    error_handling.ndarray_empty(energies, 'energies')
    error_handling.positive_real(broadening, 'broadening')
    error_handling.dos_kernel(kernel)
    en, weights = unfold(sc, ks)
    diff = energies[None, :, None] - en[:, None, :]
    if kernel == 'gaussian':
        g = np.exp(-diff ** 2 / (2 * broadening ** 2)) / (broadening * np.sqrt(2 * PI))
    else:
        g = (broadening / PI) / (diff ** 2 + broadening ** 2)
    return np.einsum('kej,kj->ke', g, weights)


def pz_hopping(
    d: ArrayLike, t_pi: float = -2.7, t_sigma: float = 0.48, a0: float = 0.142,
    d0: float = 0.335, delta: float = 0.0453,
) -> NDArray[np.float64]:
    r'''
    Get the hopping between two :math:`p_z` orbitals a vector
    :math:`\mathbf{d}` apart, in the Slater-Koster form of Moon and Koshino
    (Phys. Rev. B 85, 195458 (2012)), used for twisted bilayer graphene:

    .. math::

        t(\mathbf{d}) = V_{pp\pi}(d)\Big[1-\Big(\frac{d_z}{d}\Big)^2\Big]
        + V_{pp\sigma}(d)\Big(\frac{d_z}{d}\Big)^2\, ,\quad
        V_{pp\pi} = t_\pi e^{-(d-a_0)/\delta}\, ,\quad
        V_{pp\sigma} = t_\sigma e^{-(d-d_0)/\delta}\, .

    In-plane it is :math:`t_\pi` between nearest neighbours and decays with
    distance; between the layers of AA-stacked graphene it is
    :math:`t_\sigma`. Defaults: graphene, in eV and nm (:math:`\delta =
    0.184\,a`, :math:`a = 0.246` nm).

    :param d: Real array, shape (n, 3). Bond vectors.
    :param t_pi: Real number. Default value -2.7.
    :param t_sigma: Real number. Default value 0.48.
    :param a0: Positive real. Default value 0.142. In-plane nearest-neighbour distance.
    :param d0: Positive real. Default value 0.335. Interlayer distance.
    :param delta: Positive real. Default value 0.0453. Decay length.

    :returns:
        * **t** -- Real ndarray, shape (n,).
    '''
    d = np.atleast_2d(np.asarray(d, dtype='f8'))
    error_handling.bond_vectors(d)
    error_handling.real_number(t_pi, 't_pi')
    error_handling.real_number(t_sigma, 't_sigma')
    for val, name in ((a0, 'a0'), (d0, 'd0'), (delta, 'delta')):
        error_handling.positive_real(val, name)
    r = np.linalg.norm(d, axis=1)
    c2 = (d[:, 2] / r) ** 2
    return t_pi * np.exp(-(r - a0) / delta) * (1 - c2) + t_sigma * np.exp(-(r - d0) / delta) * c2


def _twist_matrices(m: int, lattice: str) -> tuple[NDArray, NDArray]:
    '''
    Private function. Supercell matrices of the two layers: the rows are the
    moire vectors in units of each layer's primitive vectors.
    '''
    if lattice == 'honeycomb':  # R60 a1 = a2, R60 a2 = a2 - a1
        return (np.array([[m, m + 1], [-(m + 1), 2 * m + 1]]),
                  np.array([[m + 1, m], [-m, 2 * m + 1]]))
    # square: R90 a1 = a2, R90 a2 = -a1
    return (np.array([[m, m + 1], [-(m + 1), m]]), np.array([[m + 1, m], [-m, m + 1]]))


def _layer_lattice(lattice: str, a: float) -> tuple[NDArray, NDArray, list[str]]:
    '''
    Private function. Primitive vectors, orbital positions and tags of one layer.
    '''
    if lattice == 'honeycomb':
        dx = 0.5 * np.sqrt(3) * a
        return np.array([[2 * dx, 0.], [dx, 1.5 * a]]), np.array([[0., 0.], [dx, 0.5 * a]]), ['a', 'b']
    return np.array([[a, 0.], [0., a]]), np.array([[0., 0.]]), ['a']


def commensurate_angle(m: int, lattice: str = 'honeycomb') -> float:
    r'''
    Get the commensurate twist angle :math:`\theta_m` of two layers: the
    rotation that maps the lattice vector :math:`(m+1)\mathbf{a}_1 + m\mathbf{a}_2`
    onto :math:`m\mathbf{a}_1 + (m+1)\mathbf{a}_2`,

    .. math::

        \cos\theta_m = \frac{3m^2+3m+1/2}{3m^2+3m+1}\ \text{(honeycomb)},\qquad
        \cos\theta_m = \frac{2m(m+1)}{2m^2+2m+1}\ \text{(square)}

    (Lopes dos Santos, Peres and Castro Neto, Phys. Rev. B 86, 155449
    (2012)). The moiré cell holds :math:`3m^2+3m+1` (honeycomb) or
    :math:`2m^2+2m+1` (square) primitive cells per layer; :math:`m = 31`
    gives graphene's first magic angle, :math:`\theta \approx 1.05°`.

    :param m: Positive integer.
    :param lattice: 'honeycomb' or 'square'. Default value 'honeycomb'.

    :returns:
        * **theta** -- Real number, in radians.
    '''
    error_handling.positive_int(m, 'm')
    error_handling.twist_lattice(lattice)
    a1, _ = _twist_matrices(m, lattice)
    prim, _, _ = _layer_lattice(lattice, 1.)
    v, w = a1[0] @ prim, a1[0][::-1] @ prim
    return float(abs(np.arctan2(v[0] * w[1] - v[1] * w[0], v @ w)))


def twisted_bilayer(
    m: int, lattice: str = 'honeycomb', a: float = 0.142, d: float = 0.335,
    hopping: Callable[[NDArray], NDArray] | None = None, cutoff: float | None = None,
    interlayer_scale: float = 1.,
) -> MoireKSpace:
    r'''
    Get the commensurate twisted bilayer of two honeycomb or square layers,
    twisted by :math:`\theta_m` (see *commensurate_angle*) about a common
    site (AA stacking at the origin): a **KSpace** with the moiré cell as
    unit cell, and every hopping, within a layer and between the layers,
    given by the function *hopping* of the bond vector
    :math:`\mathbf{d} = (d_x, d_y, d_z)`, for all bonds shorter than
    *cutoff* (periodic images included).

    Near the first magic angle (:math:`\theta \approx 1.05°` for graphene,
    :math:`m = 31`, 11908 orbitals) the two bands at charge neutrality
    become almost flat (Bistritzer and MacDonald, PNAS 108, 12233 (2011);
    Cao et al., Nature 556, 43 (2018)). The physics depends on the ratio
    :math:`\alpha = w/\hbar v k_\theta` only (see *magic_angle_parameter*),
    so *interlayer_scale* (which multiplies every interlayer hopping)
    moves the magic angle to larger angles and smaller cells.

    :param m: Positive integer. See *commensurate_angle*.
    :param lattice: 'honeycomb' or 'square'. Default value 'honeycomb'.
    :param a: Positive real. Default value 0.142 (graphene, nm). Nearest-neighbour distance.
    :param d: Positive real. Default value 0.335 (nm). Interlayer distance.
    :param hopping: Callable. Default value None (*pz_hopping*). Maps bond
        vectors, shape (n, 3), to hoppings, shape (n,).
    :param cutoff: Positive real. Default value None (3.8 *a*, between the
        honeycomb neighbour shells at :math:`\sqrt{13}a` and :math:`4a`, so
        that no shell is cut by rounding). Longest bond.
    :param interlayer_scale: Positive real. Default value 1. Factor on the
        interlayer hoppings.

    :returns:
        * **tbl** -- :class:`MoireKSpace` instance.
    '''
    error_handling.positive_int(m, 'm')
    error_handling.twist_lattice(lattice)
    error_handling.positive_real(a, 'a')
    error_handling.positive_real(d, 'd')
    if hopping is None:
        hopping = pz_hopping
    error_handling.is_callable(hopping, 'hopping')
    if cutoff is None:
        cutoff = 3.8 * a
    error_handling.positive_real(cutoff, 'cutoff')
    error_handling.positive_real(interlayer_scale, 'interlayer_scale')
    theta = commensurate_angle(m, lattice)
    prim, tau, tags = _layer_lattice(lattice, a)
    mats = _twist_matrices(m, lattice)
    pos_parts: list = []
    z_parts: list = []
    layer_parts: list = []
    cell_tags: list[str] = []
    for l, (mat, sign) in enumerate(zip(mats, (-1., 1.))):
        c, s = np.cos(sign * theta / 2), np.sin(sign * theta / 2)
        rot = np.array([[c, -s], [s, c]])
        origins = _inner_cells(mat) @ prim
        p = (origins[:, None, :] + tau[None]).reshape(-1, 2) @ rot.T
        pos_parts.append(p)
        z_parts.append(np.full(len(p), l * d))
        layer_parts.append(np.full(len(p), l))
        cell_tags += [chr(ord(t) + 2 * l) for t in tags] * len(origins)
        big = mat @ prim @ rot.T
        if l == 0:
            moire = big
    pos, z, layer = np.concatenate(pos_parts), np.concatenate(z_parts), np.concatenate(layer_parts)
    error_handling.commensurate(moire, big)
    cell = [{'tag': t, 'r0': (float(x), float(y))} for t, (x, y) in zip(cell_tags, pos)]
    tbl = MoireKSpace(Lattice(unit_cell=cell, prim_vec=[tuple(float(c) for c in v) for v in moire]),
                                 theta, layer, z, len(pos) // (2 * len(tau)))
    # bonds to the periodic images within the cutoff
    area = abs(np.linalg.det(moire))
    n_img = [int(np.ceil(cutoff / (area / np.linalg.norm(moire[1 - i])))) + 1 for i in range(2)]
    tree = cKDTree(pos)
    hops = []
    for l1 in range(-n_img[0], n_img[0] + 1):
        for l2 in range(-n_img[1], n_img[1] + 1):
            if (l1, l2) < (0, 0):
                continue  # the conjugates of these are added below
            R = l1 * moire[0] + l2 * moire[1]
            pairs = tree.sparse_distance_matrix(cKDTree(pos + R), cutoff, output_type='ndarray')
            i, j = pairs['i'], pairs['j']
            keep = i < j if (l1, l2) == (0, 0) else np.ones(len(i), bool)
            i, j = i[keep], j[keep]
            vec = np.column_stack([pos[j] + R - pos[i], z[j] - z[i]])
            short = np.linalg.norm(vec, axis=1) <= cutoff
            i, j, vec = i[short], j[short], vec[short]
            if len(vec) == 0:
                continue
            t = np.asarray(hopping(vec))
            error_handling.hopping_values(t, len(vec))
            t = np.where(layer[i] != layer[j], interlayer_scale * t, t)
            for ii, jj, tt in zip(i, j, t):
                hops.append((int(ii), int(jj), R, tt))
                hops.append((int(jj), int(ii), -R, np.conj(tt)))
    tbl._hop = hops
    return tbl


def magic_angle_parameter(
    theta: float, a: float = 0.142, d: float = 0.335, t: float = -2.7,
    interlayer: Callable[[NDArray], NDArray] | None = None, interlayer_scale: float = 1.,
) -> tuple[float, float, float]:
    r'''
    Get the Bistritzer-MacDonald parameters of twisted bilayer graphene
    for a tight-binding model with nearest-neighbour hopping *t* and the
    interlayer hopping *interlayer* (a function of the bond vector): the
    interlayer tunnelling :math:`w = \frac{1}{\Omega}\int d^2r\,
    t_\perp(\mathbf{r}, d)\,e^{-i\mathbf{K}\cdot\mathbf{r}}`
    (:math:`\Omega = \sqrt3a_{\mathrm{lat}}^2/2` the cell area,
    :math:`|\mathbf{K}| = 4\pi/3a_{\mathrm{lat}}`, :math:`a_{\mathrm{lat}} = \sqrt3 a`),
    the Dirac velocity :math:`\hbar v = \frac32|t|a`, the ratio
    :math:`\alpha = w/\hbar vk_\theta` (:math:`k_\theta = 2|\mathbf{K}|\sin\frac\theta2`),
    and the renormalized velocity at the Dirac points of the moiré bands
    (Bistritzer and MacDonald, PNAS 108, 12233 (2011)),

    .. math::

        \frac{v^*}{v} = \frac{1-3\alpha^2}{1+6\alpha^2}\, ,

    which vanishes at the first magic angle, :math:`\alpha = 1/\sqrt3`.

    :param theta: Positive real. Twist angle, in radians.
    :param a: Positive real. Default value 0.142. Nearest-neighbour distance.
    :param d: Positive real. Default value 0.335. Interlayer distance.
    :param t: Real number. Default value -2.7. Nearest-neighbour hopping.
    :param interlayer: Callable. Default value None (*pz_hopping*). Interlayer hopping.
    :param interlayer_scale: Positive real. Default value 1. See *twisted_bilayer*.

    :returns:
        * **w** -- Real number. Interlayer tunnelling.
        * **alpha** -- Real number.
        * **v_ratio** -- Real number. :math:`v^*/v`.
    '''
    error_handling.positive_real(theta, 'theta')
    error_handling.positive_real(a, 'a')
    error_handling.positive_real(d, 'd')
    error_handling.real_number(t, 't')
    error_handling.positive_real(interlayer_scale, 'interlayer_scale')
    if interlayer is None:
        interlayer = pz_hopping
    error_handling.is_callable(interlayer, 'interlayer')
    a_lat = np.sqrt(3) * a
    k_d = 4 * PI / (3 * a_lat)
    omega = np.sqrt(3) / 2 * a_lat ** 2

    def integrand(r):
        return r * float(np.real(interlayer(np.array([[r, 0., d]]))[0])) * j0(k_d * r)
    w = interlayer_scale * 2 * PI * quad(integrand, 0., 20 * d, limit=200)[0] / omega
    k_theta = 2 * k_d * np.sin(theta / 2)
    alpha = abs(w) / (1.5 * abs(t) * a * k_theta)
    return float(abs(w)), float(alpha), float((1 - 3 * alpha ** 2) / (1 + 6 * alpha ** 2))
