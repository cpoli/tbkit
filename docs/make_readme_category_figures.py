"""Regenerate the README's per-category example figures.

    python docs/make_readme_category_figures.py                  # all categories
    python docs/make_readme_category_figures.py topology driven  # just these

Writes docs/source/_static/images/readme_<category>.png, three panels each,
at the same size as readme_hero.png (see make_readme_figure.py). README.md
embeds them by their raw.githubusercontent.com URLs so they also render on
PyPI. Each panel is a condensed version of one gallery example in examples/.
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = Path(__file__).parent / 'source' / '_static' / 'images'

GRAPHENE_HOP = [{'i': 0, 'j': 1, 'R': R, 't': 1.} for R in [(0, 0), (-1, 0), (0, -1)]]


def haldane(M=0., t2=0.2):
    import tbkit.lattices as lattices
    from tbkit.kspace import KSpace

    hal = KSpace(lattices.honeycomb())
    hal.set_hopping(GRAPHENE_HOP)
    for R in [(0, 1), (-1, 0), (1, -1)]:
        hal.set_hopping([{'i': 0, 'j': 0, 'R': R, 't': 1j*t2}, {'i': 1, 'j': 1, 'R': R, 't': -1j*t2}])
    hal.set_onsite({'a': M, 'b': -M})
    return hal


def models(axes):
    '''Building & solving models: an impurity, Slater-Koster orbitals, the KPM.'''
    import tbkit.kpm as kpm
    import tbkit.lattices as lattices
    from tbkit.kspace import reciprocal_vectors
    from tbkit.slater_koster import sk_kspace
    from tbkit.system import System

    ax1, ax2, ax3 = axes
    # examples/tight_binding/plot_defects_and_impurities.py
    lat = lattices.honeycomb()
    lat.get_lattice(n1=10, n2=10)
    x, y = lat.coor['x'], lat.coor['y']
    imp = int(np.argmin(np.hypot(x - x.mean(), y - y.mean())))
    flake = System(lat)
    flake.set_hopping([{'n': 1, 't': -1.}])
    flake.set_onsite({'a': 0., 'b': 0.})
    flake.set_onsite_def({imp: -8.})
    flake.get_ham()
    en, vec = np.linalg.eigh(flake.ham.toarray())
    ax1.scatter(x, y, c=np.abs(vec[:, 0]) ** 2, s=22, cmap='viridis', vmax=0.05)
    ax1.set_aspect('equal')
    ax1.axis('off')
    ax1.set_title('An impurity binds a state below the band, $E = {:.2f}$'.format(en[0]))

    # examples/orbitals/plot_slater_koster.py
    orbitals = {'a': ['s', 'px', 'py', 'pz'], 'b': ['s', 'px', 'py', 'pz']}
    onsite = {'a': {'s': -8.87}, 'b': {'s': -8.87}}
    hopping = {'ss_sigma': -6.77, 'sp_sigma': 5.58, 'pp_sigma': 5.04, 'pp_pi': -3.03}
    overlap = {'ss_sigma': 0.212, 'sp_sigma': -0.102, 'pp_sigma': -0.146, 'pp_pi': 0.129}
    hc = lattices.honeycomb()
    sk = sk_kspace(hc, orbitals, {1: hopping}, onsite=onsite, overlap={1: overlap})
    b1, b2 = (np.array(v) for v in reciprocal_vectors(hc.prim_vec))
    dist, en = sk.k_path([np.zeros(2), (b1 - b2) / 3, b1 / 2, np.zeros(2)], nk=80)
    ax2.plot(dist, en, color='tab:blue', lw=1.2)
    ax2.plot([], [], color='tab:blue', label=r'$\sigma$ and $\pi$ bands')
    for x0 in sk.nodes:
        ax2.axvline(x0, color='gray', lw=0.6)
    ax2.set_xticks(sk.nodes, [r'$\Gamma$', 'K', 'M', r'$\Gamma$'])
    ax2.set_xlim(dist[0], dist[-1])
    ax2.set_ylim(-25, 25)
    ax2.set_ylabel('$E$ (eV)')
    ax2.set_title(r'Slater-Koster $sp^3$ graphene, with overlaps')

    # examples/large_scale/plot_kernel_polynomial_method.py
    e_grid = np.linspace(-3., 3., 241)
    for vacancies, color, label in ((0., 'b', 'clean'), (0.02, 'r', '2% vacancies')):
        big = lattices.honeycomb()
        big.get_lattice(150, 150)
        if vacancies:
            rng = np.random.default_rng(0)
            big.remove_sites(sorted(rng.choice(big.sites, int(vacancies * big.sites), replace=False).tolist()))
        sheet = System(big)
        sheet.set_hopping([{'n': 1, 't': 1.}])
        sheet.get_ham()
        _, rho = kpm.dos(sheet.ham, n_moments=256, n_random=24, e_grid=e_grid, seed=1)
        ax3.plot(e_grid, rho / big.sites, color, label=label)
    ax3.set_ylim(0, None)
    ax3.set_xlabel('$E/t$')
    ax3.set_ylabel(r'$\rho(E)$ per site')
    ax3.set_title('Kernel polynomial method, 45,000 sites')
    ax3.legend(fontsize=9)


def fields(axes):
    '''Fields, strain & disorder: Landau levels, pseudo-fields, localization.'''
    from tbkit.graphene import GrapheneLattice, GrapheneSystem
    from tbkit.lattice import Lattice
    from tbkit.system import System

    ax1, ax2, ax3 = axes
    # examples/magnetic_field/plot_landau_levels.py
    alpha, v_F = 0.02, 1.5
    glat = GrapheneLattice()
    glat.triangle_zigzag(n=44)
    gsys = System(glat)
    gsys.set_hopping([{'n': 1, 't': 1.}])
    gsys.set_magnetic_field(alpha=alpha)
    gsys.get_ham()
    gsys.get_eig()
    en = np.sort(gsys.en.real)
    window = en[np.abs(en) < 1.2]
    ax1.plot(window, 'o', ms=1.5, color='tab:red')
    for n in range(-3, 4):
        ax1.axhline(np.sign(n) * v_F * np.sqrt(4 * np.pi * alpha * abs(n)), color='k', ls='--', lw=0.6)
    ax1.set_xlabel('state index')
    ax1.set_ylabel('$E/t$')
    ax1.set_title(r'Graphene in a field: the $\sqrt{n}$ Landau ladder')

    # examples/strain/plot_pseudo_magnetic_field.py
    lat = GrapheneLattice()
    lat.circle(n=30)
    lat.coor['x'] -= lat.coor['x'].mean()
    lat.coor['y'] -= lat.coor['y'].mean()
    strained = GrapheneSystem(lat)
    strained.set_hop_linear_strain(t=1., beta=-0.05)
    x, y = lat.coor['x'], lat.coor['y']
    amp = strained.hop['t'].real
    norm = plt.Normalize(amp.min(), amp.max())
    cmap = plt.get_cmap('coolwarm')
    for i, j, a in zip(strained.hop['i'], strained.hop['j'], amp):
        ax2.plot([x[i], x[j]], [y[i], y[j]], color=cmap(norm(a)), lw=1.4)
    ax2.set_aspect('equal')
    ax2.axis('off')
    ax2.set_title('Triaxial strain: a pseudo-magnetic field')

    # examples/disorder/plot_anderson_localization.py
    n = 200
    for w, seed, color, label in ((0., 0, 'b', 'clean: extended'), (3., 2, 'r', 'disorder $W=3$: localized')):
        chain = Lattice(unit_cell=[{'tag': 'a', 'r0': (0., 0.)}], prim_vec=[(1., 0.)])
        chain.get_lattice(n1=n)
        sys = System(chain)
        sys.set_hopping([{'n': 1, 't': 1.}])
        sys.set_onsite({'a': 0.})
        if w:
            state = np.random.get_state()
            np.random.seed(seed)
            sys.set_onsite_dis(alpha=w)
            np.random.set_state(state)
        sys.get_ham()
        sys.get_eig(eigenvec=True)
        ax3.plot(np.abs(sys.rn[:, n // 2]) ** 2, color, lw=1, label=label)
    ax3.set_xlabel('site')
    ax3.set_ylabel(r'$|\psi_i|^2$')
    ax3.set_title('Anderson localization')
    ax3.legend(fontsize=9)


def topology(axes):
    '''Band topology: Berry curvature, the Z2 Wannier flow, corner states.'''
    import scipy.linalg as LA

    import tbkit.lattices as lattices
    from tbkit.higher_order import bbh_model, flake_positions
    from tbkit.kspace import KSpace, PAULI

    ax1, ax2, ax3 = axes
    # examples/topology/plot_haldane_topology.py
    curv = haldane(M=0.).berry_curvature(bands=[0], nk=60)
    ax1.imshow(curv.T, origin='lower', extent=[0, 1, 0, 1], cmap='RdBu')
    ax1.set_xlabel('$k_1$ (fractional)')
    ax1.set_ylabel('$k_2$ (fractional)')
    ax1.set_title('Haldane model: Berry curvature, $C = 1$')

    # examples/topology/plot_kane_mele_z2.py
    lam = 0.06
    km = KSpace(lattices.honeycomb(), spin=True)
    km.set_hopping(GRAPHENE_HOP)
    for R in [(0, 1), (-1, 0), (1, -1)]:
        km.set_hopping([{'i': 0, 'j': 0, 'R': R, 't': 1j*lam*PAULI['z']},
                        {'i': 1, 'j': 1, 'R': R, 't': -1j*lam*PAULI['z']}])
    km.set_onsite({'a': 0.1, 'b': -0.1})
    fracs, centres = km.wannier_flow([0, 1], nk=60, nk_perp=41, k_range=(0., 0.5), positions=False)
    ax2.plot(fracs, centres, 'o', color='tab:blue', ms=3)
    ax2.set_xlabel(r'$k_2/|\mathbf{b}_2|$')
    ax2.set_ylabel('hybrid Wannier centres')
    ax2.set_title(r'Kane-Mele $\mathbb{Z}_2$: Kramers pairs switch partners')

    # examples/higher_order/plot_quadrupole_insulator.py
    n = 16
    bbh = bbh_model(0.5, 1.)
    en, vec = LA.eigh(bbh.finite_ham(n))
    zero = np.abs(en) < 1e-4
    pos = flake_positions(bbh, n)
    weight = (np.abs(vec[:, zero]) ** 2).sum(axis=1)
    ax3.scatter(pos[:, 0], pos[:, 1], c=weight, s=4 + 400 * weight, cmap='viridis')
    ax3.set_aspect('equal')
    ax3.axis('off')
    ax3.set_title('Quadrupole insulator: four corner states')


def response(axes):
    '''Response & transport: Hall conductivity, a point contact, optics.'''
    import tbkit.lattices as lattices
    from tbkit.kspace import KSpace, ribbon
    from tbkit.optics import optical_conductivity
    from tbkit.system import System
    from tbkit.transport import Transport, lead_from_kspace

    ax1, ax2, ax3 = axes
    # examples/hall_effects/plot_anomalous_hall_effect.py
    e_f = np.linspace(-3.6, 3.6, 721)
    for M, color, label in ((0.3, 'C0', 'topological, $M = 0.3$'), (1.5, 'C3', 'trivial, $M = 1.5$')):
        ax1.plot(e_f, haldane(M).hall_conductivity(e_f, nk=150), color=color, label=label)
    ax1.axhline(1., color='k', lw=0.5, ls=':')
    ax1.axhline(0., color='k', lw=0.5)
    ax1.set_xlabel('$E_F$')
    ax1.set_ylabel(r'$\sigma_{xy}$ ($e^2/h$)')
    ax1.set_title('Anomalous Hall effect of the Haldane model')
    ax1.legend(fontsize=9)

    # examples/transport/plot_landauer_conductance.py
    t, width, length = -1., 24, 40
    square = [{'i': 0, 'j': 0, 'R': (1, 0), 't': t}, {'i': 0, 'j': 0, 'R': (0, 1), 't': t}]
    lat = lattices.square()
    lat.get_lattice(length, width)
    x = lat.coor['x'] - (length - 1) / 2
    y = lat.coor['y'] - (width - 1) / 2
    left = list(np.flatnonzero(np.isclose(lat.coor['x'], 0.)))
    right = list(np.flatnonzero(np.isclose(lat.coor['x'], length - 1.)))
    strip = ribbon(lattices.square(), square, width=width, direction=1)
    h_left, v_left = lead_from_kspace(strip, -1)
    h_right, v_right = lead_from_kspace(strip, 1)
    envelope = np.cos(np.pi * x / (length - 1)) ** 2
    gates = np.linspace(1.0, -0.4, 57)
    conductance = []
    for vg in gates:
        sys = System(lat)
        sys.set_hopping([{'n': 1, 't': t}])
        sys.set_onsite({'a': 0.})
        sys.onsite[:] = (vg + 0.02 * y**2) * envelope
        sys.get_ham()
        tr = Transport(sys.ham)
        tr.add_lead(h_left, v_left, t * np.eye(width), left)
        tr.add_lead(h_right, v_right, t * np.eye(width), right)
        conductance.append(tr.transmission([-3.])[0])
    ax2.plot(gates, conductance, 'o-b', ms=3)
    ax2.set_xlabel('gate potential $V_g$')
    ax2.set_ylabel('$G$ ($e^2/h$)')
    ax2.set_yticks(range(0, 7))
    ax2.grid(axis='y', alpha=0.4)
    ax2.invert_xaxis()
    ax2.set_title('Quantum point contact: conductance steps')

    # examples/optics/plot_graphene_universal_absorption.py
    gra = KSpace(lattices.honeycomb())
    gra.set_hopping([{'i': 0, 'j': 1, 'R': R, 't': -1.} for R in [(0, 0), (-1, 0), (0, -1)]])
    sigma_0 = 2 * np.pi / 4
    w = np.linspace(0.05, 3., 120)
    for e_fermi, style, label in ((0., '-b', '$E_F = 0$'), (0.4, '--r', '$E_F = 0.4|t|$')):
        sigma = 2 * optical_conductivity(gra, w, e_fermi=e_fermi, eta=0.03, nk=300).real / sigma_0
        ax3.plot(w, sigma, style, label=label)
    ax3.axhline(1., color='k', lw=0.8)
    ax3.set_xlabel(r'$\hbar\omega / |t|$')
    ax3.set_ylabel(r'Re $\sigma_{xx} / \sigma_0$')
    ax3.set_title(r'Graphene: universal absorption $\pi\alpha$')
    ax3.legend(fontsize=9)


def interactions(axes):
    '''Flat bands & interactions: twisted bilayer, Hubbard edges, Majoranas.'''
    import tbkit.lattices as lattices
    from tbkit.bdg import bdg_ham, pairing_bonds
    from tbkit.graphene import GrapheneLattice
    from tbkit.meanfield import hubbard_mean_field
    from tbkit.moire import magic_angle_parameter, twisted_bilayer
    from tbkit.system import System

    ax1, ax2, ax3 = axes
    # examples/moire/plot_magic_angle_twisted_bilayer.py
    tbl = twisted_bilayer(5, interlayer_scale=4.6)
    b1, b2 = tbl.rec_vec_k
    c = b2 if b1 @ b2 > 0 else b1 + b2
    k = np.linalg.solve(2 * np.array([b1, c]), np.array([b1 @ b1, c @ c]))
    nodes = [k, 0 * k, b1 / 2, k]
    ks = np.concatenate([np.linspace(nodes[i], nodes[i + 1], 15, endpoint=False) for i in range(3)] + [nodes[-1:]])
    en = np.linalg.eigvalsh(tbl._bloch_ham(ks @ tbl.k_basis.T))
    n = tbl.norb // 2
    mid = en[:, n - 2:n + 2]
    e0 = mid.mean()
    x = np.arange(len(ks))
    ax1.plot(x, 1e3 * (en[:, n - 8:n + 8] - e0), c='0.6', lw=1)
    ax1.plot(x, 1e3 * (mid - e0), c='C3', lw=2)
    _, alpha, _ = magic_angle_parameter(tbl.theta, interlayer_scale=4.6)
    ax1.set_xticks([0, 15, 30, 45], [r'$K$', r'$\Gamma$', r'$M$', r'$K$'])
    ax1.set_ylim(-400, 400)
    ax1.set_ylabel('$E - E_0$ (meV)')
    ax1.set_title(r'Twisted bilayer at magic $\alpha = {:.2f}$: flat bands'.format(alpha))

    # examples/correlations/plot_hubbard_edge_magnetism.py
    lat = GrapheneLattice()
    lat.hexagon_zigzag(n=6)
    flake = System(lat)
    flake.set_hopping([{'n': 1, 't': -1.}])
    flake.get_ham()
    up = np.where(lat.coor['tag'] == 'a', 0.55, 0.45)
    m = hubbard_mean_field(flake.ham, 2., lat.sites, n_up=up, n_dn=1 - up).magnetization
    ax2.scatter(lat.coor['x'], lat.coor['y'], c=m, cmap='RdBu', vmin=-0.2, vmax=0.2, s=30)
    ax2.set_aspect('equal')
    ax2.axis('off')
    ax2.set_title('Hubbard mean field: magnetic zigzag edges')

    # examples/superconductivity/plot_kitaev_chain.py
    n = 40
    chain = lattices.chain()
    chain.get_lattice(n)
    wire = System(chain)
    wire.set_hopping([{'n': 1, 't': 1.}])
    wire.get_ham()
    en, vec = np.linalg.eigh(bdg_ham(wire.ham, pairing_bonds(wire, 0.6), mu=0.5).toarray())
    zero = np.argsort(np.abs(en))[:2]
    weight = np.abs(vec[:n, zero]) ** 2 + np.abs(vec[n:, zero]) ** 2
    ax3.plot(np.arange(n), weight.sum(axis=1), 'o-r', ms=4)
    ax3.set_xlabel('site')
    ax3.set_ylabel('zero-mode weight')
    ax3.set_title('Kitaev chain: Majorana end modes')


def driven(axes):
    '''Driven & open systems: Bloch oscillations, Floquet bands, skin effect.'''
    import tbkit.lattices as lattices
    from tbkit.floquet import FloquetKSpace
    from tbkit.kspace import KSpace, reciprocal_vectors
    from tbkit.lattice import Lattice
    from tbkit.propagation import Propagation
    from tbkit.system import System

    ax1, ax2, ax3 = axes
    # examples/dynamics/plot_bloch_oscillations.py
    n, F, dz = 161, 0.3, 0.02
    lat = Lattice(unit_cell=[{'tag': 'a', 'r0': (0., 0.)}], prim_vec=[(1., 0.)])
    lat.get_lattice(n1=n)
    chain = System(lat)
    chain.set_hopping([{'n': 1, 't': 1.}])
    chain.set_onsite({'a': 0.})
    chain.set_onsite_def({i: F*i for i in range(n)})
    chain.get_ham()
    T_B = 2 * np.pi / F
    steps = int(2.2 * T_B / dz)
    x = np.arange(n)
    psi0 = np.exp(-(x - n // 2)**2 / (4 * 8.**2)).astype('c16')
    psi0 /= np.linalg.norm(psi0)
    prop = Propagation(lat)
    prop.get_propagation(chain.ham, psi0, steps=steps, dz=dz, norm=False)
    ax1.imshow(np.abs(prop.prop) ** 2, aspect='auto', origin='lower', cmap='magma',
               extent=[0, steps * dz / T_B, 0, n])
    ax1.set_ylim(n // 2 - 30, n // 2 + 30)
    ax1.set_xlabel(r'$t / T_B$')
    ax1.set_ylabel('site')
    ax1.set_title('Bloch oscillations in a tilted chain')

    # examples/floquet/plot_floquet_chern_insulator.py
    graphene = KSpace(lattices.honeycomb())
    graphene.set_hopping(GRAPHENE_HOP)
    omega, a0 = 12., 0.6
    floquet = FloquetKSpace(graphene, lambda t: (a0 * np.cos(omega * t), a0 * np.sin(omega * t)),
                            2 * np.pi / omega, n_steps=60)
    b1, b2 = (np.array(v) for v in reciprocal_vectors(graphene.lat.prim_vec))
    dist, en = floquet.k_path([np.zeros(2), (b1 - b2) / 3, b1 / 2, np.zeros(2)], nk=60)
    ax2.plot(dist, en, color='tab:blue', lw=1.8)
    for x0 in floquet.nodes:
        ax2.axvline(x0, color='gray', lw=0.6)
    ax2.set_xticks(floquet.nodes, [r'$\Gamma$', 'K', 'M', r'$\Gamma$'])
    ax2.set_xlim(dist[0], dist[-1])
    ax2.set_ylabel('quasienergy')
    ax2.set_title('Circularly polarized light gaps graphene')

    # examples/non_hermitian/plot_skin_effect.py
    hn = KSpace(lattices.chain())
    hn.set_hopping([{'i': 0, 'j': 0, 'R': (1,), 't': 1.}, {'i': 0, 'j': 0, 'R': (-1,), 't': 0.5}], hermitian=False)
    ring = np.linalg.eigvals(hn.finite_ham(60, periodic=True))
    open_chain = np.linalg.eigvals(hn.finite_ham(60))
    ax3.plot(ring.real, ring.imag, 'ob', ms=3, label='ring: a loop')
    ax3.plot(open_chain.real, open_chain.imag, 'or', ms=3, label='open chain: a line')
    ax3.set_ylim(-0.6, 0.9)
    ax3.set_xlabel('Re $E$')
    ax3.set_ylabel('Im $E$')
    ax3.set_title('Non-Hermitian skin effect: the spectrum collapses')
    ax3.legend(fontsize=9, loc='upper center', ncol=2)


FIGURES = {f.__name__: f for f in (models, fields, topology, response, interactions, driven)}


def main(names):
    for name in names or FIGURES:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)
        FIGURES[name](axes)
        out = OUT_DIR / 'readme_{}.png'.format(name)
        fig.savefig(out, dpi=110)
        plt.close(fig)
        print('wrote {}'.format(out))


if __name__ == '__main__':
    main(sys.argv[1:])
