"""Regenerate the README hero figure: python docs/make_readme_figure.py

Writes docs/source/_static/images/readme_hero.png, which README.md embeds by
its raw.githubusercontent.com URL (so it also renders on PyPI) and the docs
homepage (docs/source/index.rst) includes directly.
"""
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from tbkit.kspace import KSpace, reciprocal_vectors, ribbon, PAULI
from tbkit.lattice import Lattice
from tbkit.system import System

OUT = Path(__file__).parent / 'source' / '_static' / 'images' / 'readme_hero.png'

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)

# Graphene: two bands touching linearly at the Dirac point K.
DX, DY = 0.5 * 3 ** 0.5, 0.5
unit_cell = [{'tag': 'a', 'r0': (0., 0.)}, {'tag': 'b', 'r0': (DX, DY)}]
prim_vec = [(2 * DX, 0.), (DX, 1.5)]
lat = Lattice(unit_cell=unit_cell, prim_vec=prim_vec)
graphene_hop = [{'i': 0, 'j': 1, 'R': (0, 0), 't': 1.},
                {'i': 0, 'j': 1, 'R': (-1, 0), 't': 1.},
                {'i': 0, 'j': 1, 'R': (0, -1), 't': 1.}]
graphene = KSpace(lat)
graphene.set_hopping(graphene_hop)
b1, b2 = (np.array(v) for v in reciprocal_vectors(prim_vec))
ks_dist, en = graphene.k_path([np.zeros(2), (b1 - b2) / 3, b1 / 2, np.zeros(2)], nk=120)
ax1.plot(ks_dist, en, color='tab:blue', lw=1.8)
for x in graphene.nodes:
    ax1.axvline(x, color='gray', lw=0.6)
ax1.set_xticks(graphene.nodes, [r'$\Gamma$', 'K', 'M', r'$\Gamma$'])
ax1.set_xlim(ks_dist[0], ks_dist[-1])
ax1.set_ylabel('$E / t$')
ax1.set_title('Graphene bands: the Dirac cone at K')

# Hofstadter's butterfly: a square-lattice flake threaded by a sweeping flux.
alphas = np.linspace(0., 1., 241)
for alpha in alphas:
    sq = Lattice(unit_cell=[{'tag': 'a', 'r0': (0., 0.)}], prim_vec=[(1., 0.), (0., 1.)])
    sq.get_lattice(n1=16, n2=16)
    flake = System(sq)
    flake.set_hopping([{'n': 1, 't': 1.}])
    if alpha:
        flake.set_magnetic_field(alpha=alpha)
    flake.get_ham()
    flake.get_eig()
    ax2.plot(np.full(flake.en.size, alpha), flake.en.real, ',', color='k', alpha=0.6)
ax2.set_xlabel(r'flux per plaquette $\Phi / \Phi_0$')
ax2.set_ylabel('$E / t$')
ax2.set_title("Hofstadter's butterfly")

# Kane-Mele ribbon: helical edge states crossing the spin-orbit bulk gap.
km_hop = list(graphene_hop)
for R in [(0, 1), (-1, 0), (1, -1)]:
    km_hop.append({'i': 0, 'j': 0, 'R': R, 't': 1j * 0.1 * PAULI['z']})
    km_hop.append({'i': 1, 'j': 1, 'R': R, 't': -1j * 0.1 * PAULI['z']})
rib = ribbon(lat, km_hop, width=30, direction=1, spin=True)
ks = np.linspace(-np.pi, np.pi, 500)
en_rib = rib.get_bands(ks[:, None])
ax3.plot(ks, en_rib, color='tab:blue', lw=0.7)
edge = np.abs(en_rib) < 0.45
ax3.plot(np.broadcast_to(ks[:, None], en_rib.shape)[edge], en_rib[edge], '.', ms=1.5, color='tab:red')
ax3.set_xlim(ks[0], ks[-1])
ax3.set_ylim(-1.2, 1.2)
ax3.set_xlabel('$k$')
ax3.set_ylabel('$E / t$')
ax3.set_title('Kane-Mele ribbon: helical edge states')

OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=110)
print(f'wrote {OUT}')
