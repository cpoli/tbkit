r"""
Irreducible Representations of the Bands: Why Graphene Has Dirac Points
=========================================================================

L. P. Bouckaert, R. Smoluchowski and E. Wigner (1936) read a band
structure through its symmetry. At a k-point :math:`\mathbf{k}`, the
operations :math:`g` of the crystal with :math:`g\mathbf{k} = \mathbf{k}`
(modulo a reciprocal lattice vector) form the *little group* of
:math:`\mathbf{k}`; the Bloch states of one energy carry a representation
of it, and that representation is irreducible unless the degeneracy is
accidental. A two-dimensional irrep is a degeneracy that no perturbation
respecting the symmetry can lift; one-dimensional irreps of different
labels may cross along a line, equal ones must repel.

Here the operations of graphene (:func:`~tbkit.symmetry.find_symmetries`:
the twelve of C6v, about the centre of a hexagon) act on its two
:math:`p_z` orbitals, and :func:`~tbkit.symmetry.irreps` decomposes the
characters :math:`\chi(g) = \mathrm{Tr}\,V^\dagger U_g(\mathbf{k})V` of
each level :math:`V` on the character tables. The Dirac point is the
two-dimensional irrep E of the group C3v of K. A sublattice mass removes
the mirrors of K and leaves C3: E splits into the two one-dimensional
irreps ¹E and ²E, and a gap opens.
"""
import numpy as np
import matplotlib.pyplot as plt

from tbkit import lattices
from tbkit.kspace import KSpace, high_symmetry_path
from tbkit.symmetry import find_symmetries, irreps


def graphene(mass=0.):
    gra = KSpace(lattices.honeycomb())
    gra.set_hopping([{'n': 1, 't': -1.}])
    gra.set_onsite({'a': mass, 'b': -mass})
    return gra


# %%
# The symmetry group
# ------------------
# Twelve unitary operations, each also times time reversal; every one is a
# rotation or reflection about the centre of a hexagon.

gra = graphene()
ops = find_symmetries(gra)
unitary = [op for op in ops if not op.antiunitary]
print(len(unitary), 'unitary:', [op.name for op in unitary])
assert len(unitary) == 12 and len(ops) == 24
massive = [op.name for op in find_symmetries(graphene(0.3)) if not op.antiunitary]
print('with a mass:', massive)
assert len(massive) == 6 and 'C6' not in massive

# %%
# Irreps at the high-symmetry points
# ----------------------------------
# The group of Gamma is C6v, that of M is C2v and that of K is C3v.

points, labels = high_symmetry_path(gra.lat)
for k, name in zip(points[:3], ('Gamma', 'M', 'K')):
    res = irreps(gra, k, unitary)
    print('{:6s} {:4s} {}  at E = {}'.format(name, res.group, res.labels, np.round(res.energies, 6)))
dirac = irreps(gra, points[2], unitary)
assert (dirac.group, dirac.labels, dirac.bands) == ('C3v', ['E'], [[0, 1]])
assert abs(dirac.energies[0]) < 1e-12

# %%
# The mass splits the Dirac point
# -------------------------------

gapped = irreps(graphene(0.3), points[2])
print('with a mass, at K:', gapped.group, gapped.labels, np.round(gapped.energies, 6))
assert gapped.group == 'C3' and set(gapped.labels) == {'¹E', '²E'}
np.testing.assert_allclose(gapped.energies, [-0.3, 0.3], atol=1e-12)

# %%
# Compatibility along the lines
# -----------------------------
# Between two points the little group shrinks to the operations they share
# with the line; each level at the end decomposes into the levels on the
# line, with the same characters on those operations (compatibility
# relations).

for p, q, name in zip(points[:-1], points[1:], ('Gamma-M', 'M-K', 'K-Gamma')):
    line = irreps(gra, 0.5 * (p + q), unitary)
    for end in (p, q):
        res = irreps(gra, end, unitary)
        idx = [[o.name for o in res.operations].index(o.name) for o in line.operations]
        for lv, bands in enumerate(res.bands):
            sub = sum(line.characters[i] for i, b in enumerate(line.bands) if set(b) <= set(bands))
            np.testing.assert_allclose(res.characters[lv, idx], sub, atol=1e-10)
    print(name, line.group, line.labels)

# %%
# Labelled band structures
# ------------------------
# *plot_bands* writes the irrep of every level at the nodes of the path.

fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharey=True)
for ax, mass, title in zip(axes, (0., 0.3), ('graphene: C6v', 'with a sublattice mass: C3v')):
    model = graphene(mass)
    model.k_path(points, 60)
    for n in range(2):
        ax.plot(model.ks_dist, model.en[:, n], 'b', lw=2)
    group = [op for op in find_symmetries(model) if not op.antiunitary]
    for node, k in zip(model.nodes, points):
        res = irreps(model, k, group)
        for e, label in zip(res.energies, res.labels):
            ax.annotate(label, (node, e), xytext=(4, 3), textcoords='offset points', fontsize=12)
    for node in model.nodes:
        ax.axvline(node, color='k', lw=0.5)
    ax.set_xticks(model.nodes)
    ax.set_xticklabels(labels)
    ax.set_xlim(model.ks_dist[0], model.ks_dist[-1])
    ax.set_title(title)
axes[0].set_ylabel('$E$')
fig.set_layout_engine('tight')

# %%
# The same labels from *plot_bands*, in one call.

gra.k_path(points, 60)
fig2 = gra.plot_bands(node_labels=labels, irreps=unitary, fs=14)
assert 'E' in [t.get_text() for t in fig2.axes[0].texts]
