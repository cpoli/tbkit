r'''
A library of named Tight-Binding models, each in one call.

Every function takes the parameters of the paper that introduced the
model as keyword arguments and returns a :class:`tbkit.kspace.KSpace`, so
every band, topology and response tool applies to it. For a finite
sample in real space, cut it with :mod:`tbkit.bridges`:
*finite_system* gives a ready-to-use :class:`tbkit.system.System` for
the spinless models, *finite_model* the sparse Hamiltonian (with the
site positions) for every model, spinful ones included.

==============================  ===========  ======================================
Function                        Dimension    Invariant
==============================  ===========  ======================================
:func:`ssh`                     1D           Zak phase (*berry_phase*)
:func:`rice_mele`               1D           (polarization, not quantized)
:func:`rice_mele_pump`          1D + cycle   *chern_number* of the pump
:func:`kitaev_chain`            1D, BdG      *bdg.majorana_number*
:func:`haldane`                 2D           *chern_number*
:func:`qi_wu_zhang`             2D           *chern_number*
:func:`kane_mele`               2D, spin     *z2_invariant*
:func:`bhz`                     2D, spin     *z2_invariant*
:func:`tmd_three_band`          2D           (trivial; valley Berry curvature)
:func:`topological_insulator_3d` 3D, spin    *z2_indices_3d*
:func:`weyl_semimetal`          3D           *topology.find_weyl_points*
==============================  ===========  ======================================

Examples
--------
Example usage::

    from tbkit import models
    from tbkit.bridges import finite_system

    hal = models.haldane(t2=0.1, phi=np.pi / 2, M=0.2)
    hal.chern_number(bands=[0])            # 1.0
    flake = finite_system(hal, (20, 20))   # the same model, in real space
'''
from __future__ import annotations

from math import cos, sqrt
from typing import Sequence

import numpy as np

import tbkit.error_handling as error_handling
from tbkit.bdg import bdg_kspace
from tbkit.kspace import KSpace, PAULI
from tbkit.lattice import Lattice
import tbkit.lattices as lattices

# Liu, Shan, Yao, Yao and Xiao, Phys. Rev. B 88, 085433 (2013): GGA
# nearest-neighbour parameters (Table II; a in angstrom, energies in eV)
# and spin-orbit couplings lam (Table IV).
TMD_PARAMETERS: dict[str, dict[str, float]] = {
    'MoS2': dict(a=3.190, eps1=1.046, eps2=2.104, t0=-0.184, t1=0.401, t2=0.507,
                 t11=0.218, t12=0.338, t22=0.057, lam=0.073),
    'WS2': dict(a=3.191, eps1=1.130, eps2=2.275, t0=-0.206, t1=0.567, t2=0.536,
                t11=0.286, t12=0.384, t22=-0.061, lam=0.211),
    'MoSe2': dict(a=3.326, eps1=0.919, eps2=2.065, t0=-0.188, t1=0.317, t2=0.456,
                  t11=0.211, t12=0.290, t22=0.130, lam=0.091),
    'WSe2': dict(a=3.325, eps1=0.943, eps2=2.179, t0=-0.207, t1=0.457, t2=0.486,
                 t11=0.263, t12=0.329, t22=0.034, lam=0.228),
    'MoTe2': dict(a=3.557, eps1=0.605, eps2=1.972, t0=-0.169, t1=0.228, t2=0.390,
                  t11=0.207, t12=0.239, t22=0.252, lam=0.107),
    'WTe2': dict(a=3.560, eps1=0.606, eps2=2.102, t0=-0.175, t1=0.342, t2=0.410,
                 t11=0.233, t12=0.270, t22=0.190, lam=0.237),
}


def _two_orbitals(tags: tuple[str, str], prim_vec: Sequence[tuple[float, ...]]) -> Lattice:
    '''
    Private function. Two orbitals on one site at the origin.
    '''
    r0 = (0.,) * len(prim_vec[0])
    return Lattice(unit_cell=[{'tag': tags[0], 'r0': r0}, {'tag': tags[1], 'r0': r0}],
                   prim_vec=prim_vec)


_SQUARE = [(1., 0.), (0., 1.)]
_CUBIC = [(1., 0., 0.), (0., 1., 0.), (0., 0., 1.)]


def _chain_lattice() -> Lattice:
    '''
    Private function. Two sites per cell, half a lattice constant apart.
    '''
    return Lattice(unit_cell=[{'tag': 'a', 'r0': (0., 0.)}, {'tag': 'b', 'r0': (0.5, 0.)}],
                   prim_vec=[(1., 0.)])


def ssh(v: float = 0.5, w: float = 1.) -> KSpace:
    r'''
    Su-Schrieffer-Heeger chain (Phys. Rev. Lett. 42, 1698 (1979)): two
    sites 'a', 'b' per cell, intracell hopping *v* and intercell hopping
    *w*,

    .. math::

        H(k) = \begin{pmatrix} 0 & v + w e^{-ik} \\ v + w e^{ik} & 0\end{pmatrix}\, .

    The gap :math:`2|v - w|` closes at :math:`|v| = |w|`. For :math:`|w| > |v|`
    the lower band's Zak phase (*berry_phase* with ``positions=False``) is
    :math:`\pi` and an open chain has a zero mode at each end; for
    :math:`|w| < |v|` it is 0.

    Parameters
    ----------
    v : Real number, default 0.5
        Intracell hopping.
    w : Real number, default 1
        Intercell hopping.

    Returns
    -------
    ks
        **KSpace** instance (1D, spinless).
    '''
    error_handling.real_numbers({'v': v, 'w': w})
    return rice_mele(t=0.5 * (v + w), delta=0.5 * (v - w))


def rice_mele(t: float = 1., delta: float = -0.25, Delta: float = 0.) -> KSpace:
    r'''
    Rice-Mele chain (Phys. Rev. Lett. 49, 1455 (1982)): the SSH chain
    with alternating hoppings :math:`t\pm\delta` and a staggered onsite
    energy :math:`\pm\Delta`,

    .. math::

        H(k) = \begin{pmatrix} \Delta & v + w e^{-ik} \\ v + w e^{ik} & -\Delta\end{pmatrix}\, ,
        \qquad v = t + \delta,\ w = t - \delta\, .

    Its gap, at :math:`k = \pi`, is :math:`2\sqrt{\Delta^2 + 4\delta^2}`.
    With :math:`\Delta = 0` it is :func:`ssh`.

    Parameters
    ----------
    t : Real number, default 1
        Mean hopping.
    delta : Real number, default -0.25
        Dimerization: intracell hopping :math:`t+\delta`, intercell :math:`t-\delta`.
    Delta : Real number, default 0
        Staggered onsite energy, :math:`+\Delta` on 'a' and :math:`-\Delta` on 'b'.

    Returns
    -------
    ks
        **KSpace** instance (1D, spinless).
    '''
    error_handling.real_numbers({'t': t, 'delta': delta, 'Delta': Delta})
    ks = KSpace(_chain_lattice())
    ks.set_hopping([{'i': 0, 'j': 1, 'R': (0,), 't': t + delta},
                    {'i': 0, 'j': 1, 'R': (-1,), 't': t - delta}])
    ks.set_onsite({'a': Delta, 'b': -Delta})
    return ks


def rice_mele_pump(t: float = 1., delta: float = 0.5, Delta: float = 0.6,
                   offset: float = 0.) -> KSpace:
    r'''
    Thouless pump of the Rice-Mele chain (D. J. Thouless, Phys. Rev. B 27,
    6083 (1983)): the chain of :func:`rice_mele` with dimerization
    :math:`\delta\cos\phi` and staggering :math:`\Delta\sin\phi + \Delta_0`,
    cycled over :math:`\phi\in[0, 2\pi)`. The cycle parameter is the second
    crystal momentum of a 2D Bloch Hamiltonian :math:`H(k, \phi)`
    (primitive vectors (1, 0) and (0, 1)), whose lower band's
    *chern_number* is the charge pumped per cycle: :math:`\pm 1` when the
    loop encircles the gap closing :math:`\delta\cos\phi = \Delta\sin\phi + \Delta_0 = 0`
    (:math:`|\Delta_0| < |\Delta|` and :math:`\delta \neq 0`), 0 otherwise.

    Parameters
    ----------
    t : Real number, default 1
        Mean hopping.
    delta : Real number, default 0.5
        Amplitude of the dimerization.
    Delta : Real number, default 0.6
        Amplitude of the staggered onsite energy.
    offset : Real number, default 0
        Constant staggered onsite energy :math:`\Delta_0`.

    Returns
    -------
    ks
        **KSpace** instance (2D: :math:`k` and :math:`\phi`, spinless).
    '''
    error_handling.real_numbers({'t': t, 'delta': delta, 'Delta': Delta, 'offset': offset})
    lat = Lattice(unit_cell=[{'tag': 'a', 'r0': (0., 0.)}, {'tag': 'b', 'r0': (0.5, 0.)}],
                  prim_vec=_SQUARE)
    ks = KSpace(lat)
    ks.set_hopping([{'i': 0, 'j': 1, 'R': (0, 0), 't': t},
                    {'i': 0, 'j': 1, 'R': (-1, 0), 't': t},
                    {'i': 0, 'j': 1, 'R': (0, 1), 't': delta / 2},
                    {'i': 0, 'j': 1, 'R': (0, -1), 't': delta / 2},
                    {'i': 0, 'j': 1, 'R': (-1, 1), 't': -delta / 2},
                    {'i': 0, 'j': 1, 'R': (-1, -1), 't': -delta / 2},
                    {'i': 0, 'j': 0, 'R': (0, 1), 't': Delta / 2j},
                    {'i': 1, 'j': 1, 'R': (0, 1), 't': -Delta / 2j}])
    ks.set_onsite({'a': offset, 'b': -offset})
    return ks


def kitaev_chain(t: float = 1., delta: float = 1., mu: float = 0.) -> KSpace:
    r'''
    Kitaev chain (A. Yu. Kitaev, Phys.-Usp. 44, 131 (2001)): spinless
    fermions with nearest-neighbour hopping *t* and p-wave pairing
    *delta*, in the Bogoliubov-de Gennes form of :func:`tbkit.bdg.bdg_kspace`
    (particle then hole),

    .. math::

        E(k) = \pm\sqrt{(2t\cos k - \mu)^2 + 4\Delta^2\sin^2 k}\, .

    For :math:`|\mu| < 2|t|` (and :math:`\Delta\neq 0`) it is topological:
    *bdg.majorana_number* is :math:`-1`, and an open chain has a Majorana
    zero mode at each end; for :math:`|\mu| > 2|t|` it is :math:`+1`.

    Parameters
    ----------
    t : Real number, default 1
        Hopping.
    delta : Real number, default 1
        p-wave pairing.
    mu : Real number, default 0
        Chemical potential.

    Returns
    -------
    ks
        **KSpace** instance (1D, two orbitals: particle and hole).
    '''
    error_handling.real_numbers({'t': t, 'delta': delta, 'mu': mu})
    chain = KSpace(lattices.chain())
    chain.set_hopping([{'i': 0, 'j': 0, 'R': (1,), 't': t}])
    return bdg_kspace(chain, [{'i': 0, 'j': 0, 'R': (1,), 'delta': delta}], mu=mu)


def _honeycomb_nnn(ks: KSpace, t_a: complex | np.ndarray, t_b: complex | np.ndarray) -> None:
    '''
    Private function. Second-neighbour hoppings of the honeycomb lattice
    along a2, -a1 and a1 - a2 (120 degrees apart): *t_a* within sublattice
    'a', *t_b* within 'b'; the reverse bonds carry the conjugates.
    '''
    for R in [(0, 1), (-1, 0), (1, -1)]:
        ks.set_hopping([{'i': 0, 'j': 0, 'R': R, 't': t_a}, {'i': 1, 'j': 1, 'R': R, 't': t_b}])


_HONEYCOMB_NN = [(0, 0), (-1, 0), (0, -1)]


def haldane(t1: float = 1., t2: float = 0.1, phi: float = np.pi / 2, M: float = 0.) -> KSpace:
    r'''
    Haldane model (F. D. M. Haldane, Phys. Rev. Lett. 61, 2015 (1988)),
    the first Chern insulator: the honeycomb lattice
    (:func:`tbkit.lattices.honeycomb`, nearest neighbours a distance 1
    apart) with nearest-neighbour hopping :math:`t_1`, second-neighbour
    hopping :math:`t_2 e^{\pm i\phi}` of opposite chirality on the two
    sublattices (a staggered flux, no net field) and a staggered onsite
    energy :math:`\pm M`.

    The gaps at K and K' are :math:`2|M \pm 3\sqrt3\,t_2\sin\phi|`: the
    lower band's *chern_number* is :math:`\mathrm{sign}(t_2\sin\phi)` for
    :math:`|M| < 3\sqrt3\,|t_2\sin\phi|` (with :math:`t_1>0`), 0 above.

    Parameters
    ----------
    t1 : Real number, default 1
        Nearest-neighbour hopping.
    t2 : Real number, default 0.1
        Second-neighbour hopping amplitude.
    phi : Real number, default pi/2
        Phase of the second-neighbour hopping.
    M : Real number, default 0
        Staggered onsite energy, :math:`+M` on 'a' and :math:`-M` on 'b'.

    Returns
    -------
    ks
        **KSpace** instance (2D, spinless).
    '''
    error_handling.real_numbers({'t1': t1, 't2': t2, 'phi': phi, 'M': M})
    ks = KSpace(lattices.honeycomb())
    ks.set_hopping([{'i': 0, 'j': 1, 'R': R, 't': t1} for R in _HONEYCOMB_NN])
    _honeycomb_nnn(ks, t2 * np.exp(1j * phi), t2 * np.exp(-1j * phi))
    ks.set_onsite({'a': M, 'b': -M})
    return ks


def qi_wu_zhang(m: float = -1.) -> KSpace:
    r'''
    Qi-Wu-Zhang model (X.-L. Qi, Y.-S. Wu and S.-C. Zhang, Phys. Rev. B
    74, 085308 (2006)), the simplest Chern insulator: two orbitals 'a',
    'b' on one site of the square lattice,

    .. math::

        H(\mathbf{k}) = \sin k_x\,\sigma_x + \sin k_y\,\sigma_y
        + (m + \cos k_x + \cos k_y)\,\sigma_z\, .

    The gap closes at :math:`m = 0, \pm2`: the lower band's
    *chern_number* is :math:`+1` for :math:`-2 < m < 0`, :math:`-1` for
    :math:`0 < m < 2`, 0 for :math:`|m| > 2`.

    Parameters
    ----------
    m : Real number, default -1
        Mass.

    Returns
    -------
    ks
        **KSpace** instance (2D, spinless, two orbitals per site).
    '''
    error_handling.real_number(m, 'm')
    ks = KSpace(_two_orbitals(('a', 'b'), _SQUARE))
    hops = []
    for R, s in (((1, 0), PAULI['x']), ((0, 1), PAULI['y'])):
        t = s / 2j + PAULI['z'] / 2
        hops += [{'i': i, 'j': j, 'R': R, 't': t[i, j]} for i in range(2) for j in range(2) if t[i, j] != 0]
    ks.set_hopping(hops)
    ks.set_onsite({'a': m, 'b': -m})
    return ks


def kane_mele(t: float = 1., lambda_so: float = 0.06, lambda_r: float = 0.,
              lambda_v: float = 0.) -> KSpace:
    r'''
    Kane-Mele model (C. L. Kane and E. J. Mele, Phys. Rev. Lett. 95,
    226801 (2005)), the quantum spin Hall insulator of graphene: the
    honeycomb lattice with spin, nearest-neighbour hopping *t*, intrinsic
    spin-orbit coupling :math:`i\lambda_{SO}\nu_{ij}s_z` between second
    neighbours, Rashba coupling :math:`i\lambda_R(\mathbf{s}\times\hat{\mathbf{d}}_{ij})_z`
    between first neighbours and a staggered onsite energy
    :math:`\pm\lambda_v`,

    .. math::

        H = t\sum_{\langle ij\rangle} c_i^\dagger c_j
        + i\lambda_{SO}\sum_{\langle\langle ij\rangle\rangle}\nu_{ij}c_i^\dagger s_z c_j
        + i\lambda_R\sum_{\langle ij\rangle}c_i^\dagger(\mathbf{s}\times\hat{\mathbf{d}}_{ij})_z c_j
        + \lambda_v\sum_i\xi_i c_i^\dagger c_i\, .

    The :math:`\mathbb{Z}_2` invariant of the two lower bands
    (*z2_invariant*) is 1 for :math:`|\lambda_v| < 3\sqrt3\,|\lambda_{SO}|`
    at :math:`\lambda_R = 0` (the boundary moves with :math:`\lambda_R`),
    0 above.

    Parameters
    ----------
    t : Real number, default 1
        Nearest-neighbour hopping.
    lambda_so : Real number, default 0.06
        Intrinsic spin-orbit coupling.
    lambda_r : Real number, default 0
        Rashba coupling.
    lambda_v : Real number, default 0
        Staggered onsite energy, :math:`+\lambda_v` on 'a' and :math:`-\lambda_v` on 'b'.

    Returns
    -------
    ks
        **KSpace** instance (2D, ``spin=True``).
    '''
    error_handling.real_numbers({'t': t, 'lambda_so': lambda_so, 'lambda_r': lambda_r,
                                 'lambda_v': lambda_v})
    lat = lattices.honeycomb()
    ks = KSpace(lat, spin=True)
    ks.set_hopping([{'i': 0, 'j': 1, 'R': R, 't': t} for R in _HONEYCOMB_NN])
    _honeycomb_nnn(ks, 1j * lambda_so * PAULI['z'], -1j * lambda_so * PAULI['z'])
    if lambda_r:
        tau = np.array([d['r0'] for d in lat.unit_cell])
        a = np.array(lat.prim_vec)
        for R in _HONEYCOMB_NN:
            d = tau[1] + R[0] * a[0] + R[1] * a[1] - tau[0]
            d = d / np.linalg.norm(d)
            ks.set_hopping([{'i': 0, 'j': 1, 'R': R,
                             't': 1j * lambda_r * (PAULI['x'] * d[1] - PAULI['y'] * d[0])}])
    ks.set_onsite({'a': lambda_v, 'b': -lambda_v})
    return ks


def bhz(A: float = 1., B: float = 1., C: float = 0., D: float = 0., M: float = 1.) -> KSpace:
    r'''
    Bernevig-Hughes-Zhang model of HgTe quantum wells (B. A. Bernevig,
    T. L. Hughes and S.-C. Zhang, Science 314, 1757 (2006)), in its
    square-lattice form (lattice constant 1): an s-like orbital 's'
    (:math:`E_1`) and a p-like orbital 'p' (:math:`H_1`) with spin, on one
    site,

    .. math::

        H(\mathbf{k}) = \begin{pmatrix} h(\mathbf{k}) & 0 \\ 0 & h^*(-\mathbf{k})\end{pmatrix},
        \quad h = \epsilon(\mathbf{k}) + A\sin k_x\,\tau_x + A\sin k_y\,\tau_y + d_z(\mathbf{k})\,\tau_z\, ,

    :math:`d_z = M - 2B(2 - \cos k_x - \cos k_y)` and
    :math:`\epsilon = C - 2D(2 - \cos k_x - \cos k_y)` (:math:`\tau`:
    orbital). The basis is that of *KSpace* with ``spin=True``
    (:math:`E_1\uparrow, E_1\downarrow, H_1\uparrow, H_1\downarrow`), so
    the two blocks give :math:`A\sin k_x\,\tau_x s_z + A\sin k_y\,\tau_y`.
    Inversion is ``diag(1, -1)`` on the orbitals. The :math:`\mathbb{Z}_2`
    invariant of the two lower bands (*z2_invariant*, *parity_z2*; for
    :math:`|D| < |B|`) is 1 for :math:`0 < M/B < 4` and :math:`4 < M/B < 8`,
    0 otherwise.

    Parameters
    ----------
    A : Real number, default 1
        Velocity (orbital mixing).
    B : Real number, default 1
        Band curvature of the mass.
    C : Real number, default 0
        Energy offset.
    D : Real number, default 0
        Band curvature common to both orbitals.
    M : Real number, default 1
        Mass: the bands are inverted at :math:`\Gamma` for :math:`M/B > 0`.

    Returns
    -------
    ks
        **KSpace** instance (2D, ``spin=True``, two orbitals per site).
    '''
    error_handling.real_numbers({'A': A, 'B': B, 'C': C, 'D': D, 'M': M})
    ks = KSpace(_two_orbitals(('s', 'p'), _SQUARE), spin=True)
    ks.set_onsite({'s': C - 4 * D + M - 4 * B, 'p': C - 4 * D - M + 4 * B})
    for R in [(1, 0), (0, 1)]:
        ks.set_hopping([{'i': 0, 'j': 0, 'R': R, 't': (D + B) * PAULI['0']},
                        {'i': 1, 'j': 1, 'R': R, 't': (D - B) * PAULI['0']}])
    ks.set_hopping([{'i': 0, 'j': 1, 'R': (1, 0), 't': -0.5j * A * PAULI['z']},
                    {'i': 0, 'j': 1, 'R': (-1, 0), 't': 0.5j * A * PAULI['z']},
                    {'i': 0, 'j': 1, 'R': (0, 1), 't': -0.5 * A * PAULI['0']},
                    {'i': 0, 'j': 1, 'R': (0, -1), 't': 0.5 * A * PAULI['0']}])
    return ks


def tmd_three_band(
    a: float = 3.190, eps1: float = 1.046, eps2: float = 2.104, t0: float = -0.184,
    t1: float = 0.401, t2: float = 0.507, t11: float = 0.218, t12: float = 0.338,
    t22: float = 0.057, lam: float = 0.073, spin: bool = False,
) -> KSpace:
    r'''
    Three-band model of the monolayer group-VIB transition-metal
    dichalcogenides MX\ :sub:`2` (G.-B. Liu, W.-Y. Shan, Y. Yao, W. Yao and
    D. Xiao, Phys. Rev. B 88, 085433 (2013)): the :math:`d_{z^2}`,
    :math:`d_{xy}`, :math:`d_{x^2-y^2}` orbitals of the metal (tags 'z2',
    'xy', 'x2y2') on a triangular lattice of constant *a*, with
    nearest-neighbour hoppings. The hopping matrix to the neighbour at
    :math:`\mathbf{R}_1 = (a, 0)` is

    .. math::

        E(\mathbf{R}_1) = \begin{pmatrix} t_0 & t_1 & t_2 \\ -t_1 & t_{11} & t_{12}
        \\ t_2 & -t_{12} & t_{22}\end{pmatrix}\, ,

    and those to the other neighbours follow by the :math:`C_3` rotations
    (which turn :math:`(d_{xy}, d_{x^2-y^2})` by twice the angle) and
    Hermiticity, so that :math:`H(\mathbf{k})` is Eq. (4)-(10) of the
    paper. The defaults are the GGA parameters of MoS\ :sub:`2`;
    ``tmd_three_band(**TMD_PARAMETERS['WSe2'])`` gives another material
    (Mo/W, S/Se/Te). The smallest direct gap is at K and K' (1.663 eV for
    MoS\ :sub:`2`, the conduction band there being pure :math:`d_{z^2}` at
    :math:`\epsilon_1 - 3t_0`); the model is trivial (*chern_number* 0),
    with opposite Berry curvatures at K and K' (the valley Hall effect).

    With ``spin=True`` the spin-orbit coupling :math:`\lambda L_z s_z/2`
    of the paper's Eq. (25)-(26) is added: it splits the valence band at
    K by :math:`2\lambda`, the upper state having spin down at K and spin
    up at K' (spin-valley locking).

    Parameters
    ----------
    a : Positive real number, default 3.190
        Lattice constant, in angstrom.
    eps1, eps2 : Real numbers, default 1.046, 2.104
        Onsite energies of :math:`d_{z^2}` and of :math:`d_{xy}, d_{x^2-y^2}` (eV).
    t0, t1, t2, t11, t12, t22 : Real numbers
        Nearest-neighbour hoppings (eV), defaults
        -0.184, 0.401, 0.507, 0.218, 0.338, 0.057.
    lam : Real number, default 0.073
        Spin-orbit coupling (eV), used with ``spin=True``.
    spin : Boolean, default False
        Spin-1/2 orbitals, with spin-orbit coupling.

    Returns
    -------
    ks
        **KSpace** instance (2D, three orbitals per site).
    '''
    error_handling.positive_real(a, 'a')
    error_handling.real_numbers({'eps1': eps1, 'eps2': eps2, 't0': t0, 't1': t1, 't2': t2,
                                 't11': t11, 't12': t12, 't22': t22, 'lam': lam})
    error_handling.boolean(spin, 'spin')
    lat = Lattice(unit_cell=[{'tag': tag, 'r0': (0., 0.)} for tag in ('z2', 'xy', 'x2y2')],
                  prim_vec=[(a, 0.), (0.5 * a, 0.5 * sqrt(3) * a)])
    ks = KSpace(lat, spin=spin)
    e_r1 = np.array([[t0, t1, t2], [-t1, t11, t12], [t2, -t12, t22]])
    hops = []
    # R1 = a1, then R1 turned by 120 and 240 degrees; the opposite
    # neighbours are the Hermitian conjugates
    for n, R in enumerate([(1, 0), (-1, 1), (0, -1)]):
        c, s = np.cos(4 * np.pi * n / 3), np.sin(4 * np.pi * n / 3)
        rot = np.array([[1., 0., 0.], [0., c, s], [0., -s, c]])
        e_r = rot @ e_r1 @ rot.T
        hops += [{'i': i, 'j': j, 'R': R, 't': float(e_r[i, j])}
                 for i in range(3) for j in range(3) if abs(e_r[i, j]) > 1e-14]
    if spin:
        hops.append({'i': 1, 'j': 2, 'R': (0, 0), 't': 1j * lam * PAULI['z']})
    ks.set_hopping(hops)
    ks.set_onsite({'z2': eps1, 'xy': eps2, 'x2y2': eps2})
    return ks


def topological_insulator_3d(M: float = 2., A: float = 1., B: float = 1.) -> KSpace:
    r'''
    Lattice model of a 3D topological insulator (the cubic-lattice
    regularization of the Bi\ :sub:`2`\ Se\ :sub:`3` model of H. Zhang et
    al., Nature Phys. 5, 438 (2009)): orbitals 's' and 'p' of opposite
    parity, with spin, on one site of the cubic lattice,

    .. math::

        H(\mathbf{k}) = m(\mathbf{k})\,\tau_z + A\sum_{i=x,y,z}\sin k_i\,\tau_x\sigma_i\, ,
        \qquad m(\mathbf{k}) = M - 2B\sum_i(1 - \cos k_i)\, .

    Inversion is ``diag(1, -1)`` on the orbitals. The four
    :math:`\mathbb{Z}_2` indices of the two lower bands (*z2_indices_3d*),
    for :math:`B > 0` and :math:`A \neq 0`, are :math:`(0;000)` for
    :math:`M < 0` or :math:`M > 12B`, :math:`(1;000)` for
    :math:`0 < M < 4B` (a strong topological insulator), :math:`(0;111)`
    for :math:`4B < M < 8B` (weak), :math:`(1;111)` for :math:`8B < M < 12B`.

    Parameters
    ----------
    M : Real number, default 2
        Mass at :math:`\Gamma`.
    A : Real number, default 1
        Velocity.
    B : Real number, default 1
        Band curvature of the mass.

    Returns
    -------
    ks
        **KSpace** instance (3D, ``spin=True``, two orbitals per site).
    '''
    error_handling.real_numbers({'M': M, 'A': A, 'B': B})
    ks = KSpace(_two_orbitals(('s', 'p'), _CUBIC), spin=True)
    sigma = [PAULI['x'], PAULI['y'], PAULI['z']]
    hops = []
    for d in range(3):
        R = tuple(int(x) for x in np.eye(3, dtype=int)[d])
        hops += [{'i': 0, 'j': 0, 'R': R, 't': B * PAULI['0']},
                 {'i': 1, 'j': 1, 'R': R, 't': -B * PAULI['0']},
                 {'i': 0, 'j': 1, 'R': R, 't': -0.5j * A * sigma[d]},
                 {'i': 1, 'j': 0, 'R': R, 't': -0.5j * A * sigma[d]}]
    ks.set_hopping(hops)
    ks.set_onsite({'s': M - 6 * B, 'p': -(M - 6 * B)})
    return ks


def weyl_semimetal(k0: float = np.pi / 2) -> KSpace:
    r'''
    Minimal Weyl semimetal (two Weyl points, broken time reversal) on the
    cubic lattice: a two-component orbital ('a', ``spin=True``) with

    .. math::

        H(\mathbf{k}) = \sin k_x\,\sigma_x + \sin k_y\,\sigma_y
        + (2 + \cos k_0 - \cos k_x - \cos k_y - \cos k_z)\,\sigma_z\, ,

    whose bands touch at :math:`(0, 0, \pm k_0)` (*topology.find_weyl_points*),
    with chiralities :math:`\pm 1`: the planes :math:`|k_z| < k_0` have
    *chern_number* :math:`-1`, the others 0 (for :math:`0 < k_0 < \pi`).

    Parameters
    ----------
    k0 : Real number, default pi/2
        Position of the Weyl points along :math:`k_z`.

    Returns
    -------
    ks
        **KSpace** instance (3D, ``spin=True``, one site).
    '''
    error_handling.real_number(k0, 'k0')
    ks = KSpace(Lattice(unit_cell=[{'tag': 'a', 'r0': (0., 0., 0.)}], prim_vec=_CUBIC), spin=True)
    ks.set_hopping([{'i': 0, 'j': 0, 'R': (1, 0, 0), 't': -0.5j * PAULI['x'] - 0.5 * PAULI['z']},
                    {'i': 0, 'j': 0, 'R': (0, 1, 0), 't': -0.5j * PAULI['y'] - 0.5 * PAULI['z']},
                    {'i': 0, 'j': 0, 'R': (0, 0, 1), 't': -0.5 * PAULI['z']}])
    ks.set_onsite({'a': (2. + cos(k0), -2. - cos(k0))})  # (2 + cos k0) sigma_z
    return ks
