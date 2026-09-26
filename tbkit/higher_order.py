r"""
Higher-order topology: nested Wilson loops, Wannier-sector polarizations,
the quadrupole moment and corner charges (Benalcazar, Bernevig and Hughes,
Science 357, 61 (2017); Phys. Rev. B 96, 245115 (2017)).

A two-dimensional insulator can have no dipole moment (no Chern number, no
bulk polarization) and still be topological: its hybrid Wannier functions
split into two gapped *Wannier sectors*, each carrying a quantized
polarization of its own, and the bulk quadrupole moment

.. math::

    q_{xy} = 2\,p_y^{\nu_x^-}\,p_x^{\nu_y^-} \pmod 1

(1/2 or 0 with reflection symmetries) shows up as a fractional charge
:math:`\pm e/2` bound to each corner of a finite flake, with zero-energy
corner states at the corners in the chiral-symmetric limit.

The recipe of the nested Wilson loop, for occupied bands :math:`|u^n_{\mathbf{k}}\rangle`:

1. The Wilson loop along :math:`x` from every base point :math:`\mathbf{k}`,
   :math:`W_{x,\mathbf{k}}|\nu^j_{x,\mathbf{k}}\rangle = e^{-2\pi i\nu_x^j(k_y)}|\nu^j_{x,\mathbf{k}}\rangle`;
   its phases :math:`\nu_x^j(k_y)` are the Wannier bands (see
   :meth:`tbkit.kspace.KSpace.wannier_flow`).
2. The Wannier-sector states :math:`|w^j_{x,\mathbf{k}}\rangle = \sum_n|u^n_{\mathbf{k}}\rangle[\nu^j_{x,\mathbf{k}}]^n`
   of the sector :math:`\nu_x^-` (the Wannier bands below the Wannier gap).
3. The Wilson loop of those states along :math:`y` (the *nested* Wilson
   loop), whose phase, averaged over :math:`k_x`, is the Wannier-sector
   polarization :math:`p_y^{\nu_x^-}`.

The Bloch matrices of the whole mesh come at once from
:meth:`KSpace._bloch_ham <tbkit.kspace.KSpace>`; the links carry the orbital
positions (``positions=True``), as in *KSpace.berry_phase*.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
import scipy.linalg as LA

import tbkit.error_handling as error_handling
from tbkit.kspace import KSpace
from tbkit.meanfield import _occupy
from tbkit.lattice import Lattice


PI = np.pi


def bbh_model(gamma: float = 0.5, lam: float = 1., delta: float = 0.) -> KSpace:
    r'''
    Get the Benalcazar-Bernevig-Hughes (BBH) quadrupole insulator: four
    orbitals per square cell, intra-cell hoppings :math:`\gamma`, inter-cell
    hoppings :math:`\lambda`, and a :math:`\pi` flux through every plaquette,

    .. math::

        H(\mathbf{k}) = [\gamma+\lambda\cos k_x]\Gamma_4 + \lambda\sin k_x\,\Gamma_3
        + [\gamma+\lambda\cos k_y]\Gamma_2 + \lambda\sin k_y\,\Gamma_1 + \delta\,\Gamma_0\, ,

    with :math:`\Gamma_0 = \tau_3\sigma_0`, :math:`\Gamma_k = -\tau_2\sigma_k`
    (:math:`k = 1, 2, 3`), :math:`\Gamma_4 = \tau_1\sigma_0` (orbital
    :math:`2\tau + \sigma`). The hoppings are read off :math:`H(\mathbf{k})`,
    and the four orbitals sit at the corners of a square of side 1/2 centred
    on the cell origin, each next to the orbitals it is bonded to by
    :math:`\gamma`. For :math:`|\gamma| < |\lambda|` (and :math:`\delta = 0`) the model is a
    quadrupole insulator, :math:`q_{xy} = 1/2`; for :math:`|\gamma| > |\lambda|` it is trivial.
    :math:`\delta` breaks the chiral symmetry and splits the corner states
    to :math:`\pm\delta`.

    :param gamma: Real number. Default value 0.5. Intra-cell hopping.
    :param lam: Real number. Default value 1. Inter-cell hopping.
    :param delta: Real number. Default value 0. Onsite :math:`\pm\delta`.

    :returns:
        * **bbh** -- **KSpace** instance, 4 orbitals, lattice constant 1.
    '''
    error_handling.real_number(gamma, 'gamma')
    error_handling.real_number(lam, 'lam')
    error_handling.real_number(delta, 'delta')
    s0, sx = np.eye(2), np.array([[0., 1.], [1., 0.]])
    sy, sz = np.array([[0., -1j], [1j, 0.]]), np.diag([1., -1.])
    g1, g2, g3 = (-np.kron(sy, s) for s in (sx, sy, sz))
    g4 = np.kron(sx, s0)
    onsite = gamma * (g4 + g2)
    bx = lam / 2 * (g4 - 1j * g3)  # lam cos(kx) G4 + lam sin(kx) G3
    by = lam / 2 * (g2 - 1j * g1)
    r = 0.25
    # orbital 0 top right, 1 bottom left, 2 top left, 3 bottom right
    cell = [{'tag': t, 'r0': p} for t, p in zip('abcd', [(r, r), (-r, -r), (-r, r), (r, -r)])]
    bbh = KSpace(Lattice(unit_cell=cell, prim_vec=[(1., 0.), (0., 1.)]))
    hops = []
    for mat, R in ((onsite, (0, 0)), (bx, (1, 0)), (by, (0, 1))):
        for i in range(4):
            for j in range(4):
                if abs(mat[i, j]) > 1e-14 and (R != (0, 0) or i < j):
                    t = complex(mat[i, j])
                    hops.append({'i': i, 'j': j, 'R': R, 't': t.real if t.imag == 0 else t})
    bbh.set_hopping(hops)
    bbh.set_onsite({'a': float(delta), 'b': float(delta), 'c': -float(delta), 'd': -float(delta)})
    return bbh


def _check_model(ks: KSpace, bands) -> list[int]:
    '''
    Private function. Validate a 2D Hermitian KSpace and its band list.
    '''
    error_handling.kspace(ks, KSpace)
    error_handling.dim_exact(ks.dim, 2)
    error_handling.hermitian_kspace(ks.is_hermitian(), ks._overlap_hop)
    if isinstance(bands, int):
        bands = [bands]
    error_handling.band_indices(bands, ks.norb)
    return bands


def _occupied_mesh(ks: KSpace, bands: list[int], nk: tuple[int, int]) -> NDArray[np.complex128]:
    '''
    Private function. Eigenvectors of *bands* on the uniform mesh, shape
    (n1, n2, norb, len(bands)), from one vectorized Bloch sum.
    '''
    n1, n2 = nk
    fracs = np.stack(np.meshgrid(np.arange(n1) / n1, np.arange(n2) / n2, indexing='ij'), axis=-1)
    k_cart = (fracs.reshape(-1, 2) @ ks.rec_vec_k) @ ks.k_basis.T
    _, vec = np.linalg.eigh(ks._bloch_ham(k_cart))
    return vec[:, :, bands].reshape(n1, n2, ks.norb, len(bands))


def _link_phase(ks: KSpace, direction: int, n: int, positions: bool) -> NDArray[np.complex128]:
    '''
    Private function. The phases exp(-i dk.tau) carried by each link of a
    loop of n steps along b_direction (ones without positions).
    '''
    if not positions:
        return np.ones(ks.norb, 'c16')
    dk = ks.k_basis @ (ks.rec_vec_k[direction] / n)
    return np.exp(-1j * ks.orbital_positions() @ dk)


def _unitary(mat: NDArray[np.complex128]) -> NDArray[np.complex128]:
    '''
    Private function. Unitary part u vh of each matrix (last two axes).
    '''
    u, _, vh = np.linalg.svd(mat)
    return u @ vh


def _wilson_all(vec: NDArray[np.complex128], phase: NDArray[np.complex128]) -> NDArray[np.complex128]:
    r'''
    Private function. Wilson loops along the first axis of *vec* (shape
    (n, m, norb, nb)) from every base point: W[i, j] = F_i F_{i+1} ... F_{i-1},
    F_i = <u_i|e^{-i dk tau}|u_{i+1}> (unitarized). Shape (n, m, nb, nb).
    '''
    n = vec.shape[0]
    nxt = np.roll(vec, -1, axis=0)
    links = _unitary(vec.conj().transpose(0, 1, 3, 2) @ (phase[:, None] * nxt))
    out = np.zeros(links.shape, 'c16')
    for i in range(n):
        w = np.broadcast_to(np.eye(links.shape[-1], dtype='c16'), links.shape[1:]).copy()
        for m in range(n):
            w = w @ links[(i + m) % n]
        out[i] = w
    return out


def wannier_bands(
    ks: KSpace, bands: int | list[int], nk: int | tuple[int, int] = 40, direction: int = 0,
    positions: bool = True,
) -> tuple[NDArray[np.float64], NDArray[np.complex128]]:
    r'''
    Get the Wannier bands :math:`\nu_{d}^j(\mathbf{k})` of a group of bands
    of a 2D model, and the Wannier-sector states, from the Wilson loop
    :math:`W_{d,\mathbf{k}}` along :math:`\mathbf{b}_d` starting from every
    point :math:`\mathbf{k}` of the mesh. The Wannier bands do not depend on
    the base point along :math:`d` (only the eigenvectors do); they are the
    hybrid Wannier centres of *KSpace.wannier_centers*, in units of
    :math:`\mathbf{a}_d`, here in :math:`(-1/2, 1/2]` and sorted.

    :param ks: **KSpace** instance, 2D, Hermitian, without overlap.
    :param bands: Band index or list of band indices (e.g. the occupied bands).
    :param nk: Positive integer, or tuple of two. Default value 40. Mesh
        along :math:`\mathbf{b}_1, \mathbf{b}_2`.
    :param direction: 0 or 1. Default value 0. Direction :math:`d` of the Wilson loop.
    :param positions: Boolean. Default value True. Include the orbital
        positions in the links (see *KSpace.berry_phase*).

    :returns:
        * **nu** -- Real ndarray, shape (n1, n2, len(bands)). Wannier bands at
          every base point, sorted, in :math:`(-1/2, 1/2]`.
        * **w** -- Complex ndarray, shape (n1, n2, norb, len(bands)). The
          Wannier-sector states :math:`|w^j_{d,\mathbf{k}}\rangle`, column *j*
          for ``nu[..., j]``.
    '''
    bands = _check_model(ks, bands)
    error_handling.nk(nk, 2)
    error_handling.direction(direction, 2)
    error_handling.boolean(positions, 'positions')
    if isinstance(nk, int):
        nk = (nk, nk)
    vec = _occupied_mesh(ks, bands, nk)
    phase = _link_phase(ks, direction, nk[direction], positions)
    if direction == 1:
        vec = vec.transpose(1, 0, 2, 3)
    wil = _wilson_all(vec, phase)
    val, eig = np.linalg.eig(wil)
    nu = -np.angle(val) / (2 * PI)
    nu = np.where(nu <= -0.5, nu + 1., nu)
    order = np.argsort(nu, axis=-1, kind='stable')
    nu = np.take_along_axis(nu, order, axis=-1)
    eig = np.take_along_axis(eig, order[..., None, :], axis=-1)
    # W is unitary: orthonormalize the eigenvectors (degenerate ones included)
    eig = np.linalg.qr(eig)[0]
    w = vec @ eig
    if direction == 1:
        nu, w = nu.transpose(1, 0, 2), w.transpose(1, 0, 2, 3)
    return nu, w


def wannier_sector_polarization(
    ks: KSpace, bands: int | list[int], sector: int | list[int] | None = None,
    nk: int | tuple[int, int] = 40, direction: int = 0, positions: bool = True,
) -> float:
    r'''
    Get the polarization of a Wannier sector, :math:`p_{d'}^{\nu_d}`, by the
    nested Wilson loop of Benalcazar, Bernevig and Hughes (see the module
    docstring): the Wannier-sector states of *sector* (from *wannier_bands*
    along *direction* :math:`d`) are carried around the Brillouin zone along
    the other direction :math:`d'`,

    .. math::

        p_{d'}^{\nu_d} = -\frac{1}{N_d}\sum_{k_d}\frac{1}{2\pi}\arg\det
        \prod_{k_{d'}}\langle w_{d,\mathbf{k}}|e^{-i\delta\mathbf{k}\cdot\boldsymbol\tau}|w_{d,\mathbf{k}+\delta\mathbf{k}_{d'}}\rangle
        \pmod 1\, ,

    in units of :math:`\mathbf{a}_{d'}`. The sector must be separated from
    the other Wannier bands by a Wannier gap at every :math:`\mathbf{k}`
    (a ValueError is raised otherwise). For the BBH model it is 1/2 in the
    quadrupole phase and 0 in the trivial one.

    :param ks: **KSpace** instance, 2D, Hermitian, without overlap.
    :param bands: Band index or list of band indices (the occupied bands).
    :param sector: Integer or list of integers. Default value None (the
        lower half, :math:`\nu_d^-`). Indices of the Wannier bands (sorted
        in :math:`(-1/2, 1/2]`, see *wannier_bands*) forming the sector.
    :param nk: Positive integer, or tuple of two. Default value 40. Mesh.
    :param direction: 0 or 1. Default value 0. Direction :math:`d` of the
        first Wilson loop; the polarization is along the other one.
    :param positions: Boolean. Default value True. See *wannier_bands*.

    :returns:
        * **p** -- Real number in :math:`[0, 1)`.
    '''
    bands = _check_model(ks, bands)
    if sector is None:
        sector = list(range(max(1, len(bands) // 2)))
    if isinstance(sector, int):
        sector = [sector]
    error_handling.band_indices(sector, len(bands))
    nu, w = wannier_bands(ks, bands, nk, direction, positions)
    error_handling.wannier_gap(nu, sector)
    other = 1 - direction
    w = w[..., sector]
    n_other = w.shape[other]
    phase = _link_phase(ks, other, n_other, positions)
    if other == 0:
        w = w.transpose(1, 0, 2, 3)
    # w: (n_d, n_other, norb, ns); links along the second axis
    nxt = np.roll(w, -1, axis=1)
    links = _unitary(w.conj().transpose(0, 1, 3, 2) @ (phase[:, None] * nxt))
    loop = np.broadcast_to(np.eye(len(sector), dtype='c16'), (w.shape[0], len(sector), len(sector))).copy()
    for m in range(n_other):
        loop = loop @ links[:, m]
    # the phases are continuous in k_d: unwrap them before averaging
    p = -np.unwrap(np.angle(np.linalg.det(loop))) / (2 * PI)
    return _mod1(float(np.mean(p)))


def _mod1(x: float) -> float:
    '''
    Private function. x modulo 1, in [0, 1), with 1 - 1e-10 < x < 1 taken as 0.
    '''
    x = x % 1.
    return 0. if x > 1. - 1e-10 else x


def quadrupole_moment(
    ks: KSpace, bands: int | list[int], nk: int | tuple[int, int] = 40, positions: bool = True,
) -> float:
    r'''
    Get the bulk quadrupole moment of a 2D insulator with reflection
    symmetries, from its Wannier-sector polarizations (Benalcazar, Bernevig
    and Hughes 2017):

    .. math::

        q_{xy} = 2\,p_y^{\nu_x^-}\,p_x^{\nu_y^-} \pmod 1\, ,

    in units of :math:`e` per unit cell area. With the reflections
    :math:`M_x, M_y` (and :math:`C_4`, or the chiral symmetry, of the BBH
    model) both polarizations are quantized to 0 or 1/2, and so is
    :math:`q_{xy}`. The occupied bands must have a Wannier gap in both
    directions (a ValueError is raised otherwise, see
    *wannier_sector_polarization*).

    :param ks: **KSpace** instance, 2D, Hermitian, without overlap.
    :param bands: List of band indices (the occupied bands, at least two).
    :param nk: Positive integer, or tuple of two. Default value 40. Mesh.
    :param positions: Boolean. Default value True. See *wannier_bands*.

    :returns:
        * **q_xy** -- Real number in :math:`[0, 1)`.
    '''
    p_y = wannier_sector_polarization(ks, bands, None, nk, 0, positions)
    p_x = wannier_sector_polarization(ks, bands, None, nk, 1, positions)
    # each p is taken in (-1/2, 1/2] first, so that 1 - eps counts as 0
    p_y, p_x = (p - 1. if p > 0.5 + 1e-9 else p for p in (p_y, p_x))
    return _mod1(2 * p_y * p_x)


def flake_positions(ks: KSpace, n_cells: int | tuple[int, ...]) -> NDArray[np.float64]:
    r'''
    Get the positions of the orbitals of the finite sample of
    *KSpace.finite_ham*, in its order: orbital *o* of cell
    :math:`(n_1, n_2, \dots)` at :math:`\sum_i n_i\mathbf{a}_i + \boldsymbol\tau_o`.

    :param ks: **KSpace** instance.
    :param n_cells: Positive integer, or tuple of *dim* positive integers.

    :returns:
        * **positions** -- Real ndarray, shape (N*norb, space_dim).
    '''
    error_handling.kspace(ks, KSpace)
    error_handling.nk(n_cells, ks.dim)
    if isinstance(n_cells, int):
        n_cells = (n_cells,) * ks.dim
    # cell (n1, n2, n3) is row ((n3 N2 + n2) N1 + n1): n1 runs fastest
    grids = np.meshgrid(*[np.arange(n) for n in n_cells[::-1]], indexing='ij')
    cells = np.stack([g.ravel() for g in grids[::-1]], axis=1)
    origin = cells @ np.array(ks.lat.prim_vec, dtype='f8')
    return (origin[:, None, :] + ks.orbital_positions()[None]).reshape(-1, ks.space_dim)


class CornerCharges():
    r'''
    Charges of the four corners of a finite 2D flake, see *corner_charges*.

    :ivar energies: Real ndarray. Spectrum of the flake.
    :ivar density: Real ndarray. Electron density :math:`\rho_i` on every orbital.
    :ivar positions: Real ndarray, shape (N*norb, 2). Orbital positions.
    :ivar background: Real number. Uniform background :math:`\bar\rho`
        (electrons per orbital) that neutralizes the flake.
    :ivar e_fermi: Real number. Fermi level.
    :ivar charges: Real ndarray, shape (2, 2). Excess electron number
        :math:`\sum_{i\in Q}(\rho_i - \bar\rho)` in each quadrant *Q*:
        ``charges[a, b]`` for the half :math:`a` (0: low, 1: high) along
        :math:`\mathbf{a}_1` and :math:`b` along :math:`\mathbf{a}_2`.
    '''

    def __init__(self, energies, density, positions, background, e_fermi, charges) -> None:
        self.energies = energies
        self.density = density
        self.positions = positions
        self.background = background
        self.e_fermi = e_fermi
        self.charges = charges


def corner_charges(
    ks: KSpace, n_cells: int | tuple[int, int], n_electrons: float | None = None,
    temperature: float = 0.,
) -> CornerCharges:
    r'''
    Get the corner charges of a finite flake of :math:`N_1\times N_2` cells
    of a 2D model (*KSpace.finite_ham*, open boundaries): the electron
    density :math:`\rho_i = \sum_n f_n|\psi_n(i)|^2` of the lowest states,
    minus a uniform neutralizing background :math:`\bar\rho = N_e/(N\,\mathrm{norb})`,
    summed over each quadrant of the flake (the halves being split at the
    middle of the sample along :math:`\mathbf{a}_1` and
    :math:`\mathbf{a}_2`). A quadrant charge is the corner charge of
    Benalcazar, Bernevig and Hughes,
    :math:`Q^{\mathrm{corner}} = p^{\mathrm{edge}}_x + p^{\mathrm{edge}}_y - q_{xy}`:
    :math:`\pm 1/2` for the BBH quadrupole insulator (with
    :math:`\delta \ne 0` to fill two of its four corner states), 0 for a
    trivial insulator. The sign is that of the *electron* number (charge
    in units of :math:`-e`).

    At :math:`T = 0`, the electrons left for a degenerate level at the
    Fermi energy are shared equally among its states (e.g. the four exact
    zero modes of the BBH model at :math:`\delta = 0`, which then gives
    zero corner charges): split the degeneracy, or use a temperature.

    :param ks: **KSpace** instance, 2D, Hermitian, without overlap.
    :param n_cells: Positive integer, or tuple of two. Cells along
        :math:`\mathbf{a}_1, \mathbf{a}_2`.
    :param n_electrons: Positive real. Default value None (half filling).
        Number of electrons in the flake. An integer at T = 0.
    :param temperature: Positive real or zero. Default value 0.

    :returns:
        * **result** -- :class:`CornerCharges`.
    '''
    _check_model(ks, 0)
    error_handling.nk(n_cells, 2)
    if isinstance(n_cells, int):
        n_cells = (n_cells, n_cells)
    n_states = int(np.prod(n_cells)) * ks.norb
    if n_electrons is None:
        n_electrons = n_states // 2
    error_handling.electrons(n_electrons, n_states)
    error_handling.positive_real_zero(temperature, 'temperature')
    if temperature == 0:
        error_handling.integer_electrons(n_electrons)
    en, vec = LA.eigh(ks.finite_ham(n_cells))
    f, _, mu = _occupy(en, en[:0], n_electrons, temperature)
    density = (np.abs(vec) ** 2) @ f
    background = n_electrons / n_states
    positions = flake_positions(ks, n_cells)
    # fractional coordinates along a1, a2, split at the middle of the sample
    a = np.array(ks.lat.prim_vec, dtype='f8')
    frac = np.linalg.solve(a.T, positions.T).T
    mid = (np.array(n_cells) - 1) / 2 + np.mean(np.linalg.solve(a.T, ks.orbital_positions().T).T, axis=0)
    high = frac > mid[None, :]
    excess = density - background
    charges = np.array([[excess[(high[:, 0] == bool(s1)) & (high[:, 1] == bool(s2))].sum()
                                   for s2 in (0, 1)] for s1 in (0, 1)])
    return CornerCharges(en, density, positions, background, float(mu), charges)

