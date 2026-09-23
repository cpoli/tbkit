"""
Figure helpers used by the narrative docs (``tutorial.rst``, ``history.rst``)
through matplotlib's ``.. plot::`` directive.

They exist so every page can show *what lattice it is talking about* --
the unit cell, the primitive vectors, and a patch of the lattice they
generate -- without each page repeating twenty lines of matplotlib.

These are documentation helpers, not part of the public **tbkit** API.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from tbkit.lattice import Lattice

#: One colour per sublattice tag, in the order the tags sort.
COLORS = ['#3b76af', '#ef8636', '#519e3e', '#c53a32', '#8d69b8', '#84584e']


def _patch(lat, n1, n2):
    """A fresh finite patch of *lat*, leaving the argument untouched."""
    out = Lattice(unit_cell=lat.unit_cell, prim_vec=lat.prim_vec)
    out.get_lattice(n1=n1, n2=n2)
    return out


def _bonds(coor, tol=1e-3):
    """Index pairs at the shortest inter-site distance."""
    dx = coor['x'][:, None] - coor['x'][None, :]
    dy = coor['y'][:, None] - coor['y'][None, :]
    dist = np.hypot(dx, dy)
    off = dist[dist > tol]
    if not off.size:
        return [], 0.
    d_min = off.min()
    i, j = np.where(np.isclose(dist, d_min, atol=tol) & (np.arange(len(coor))[:, None]
                                                                       < np.arange(len(coor))[None, :]))
    return list(zip(i, j)), d_min


def plot_lattice(
    lat,
    n1=4,
    n2=3,
    title=None,
    ax=None,
    figsize=(5.4, 3.8),
    show_vectors=True,
    show_cell=True,
    label_cell=True,
):
    r"""
    Draw a patch of *lat* with its unit cell and primitive vectors marked.

    The sites of the home unit cell (the one at :math:`\mathbf{R}=0`) are
    drawn filled and labelled by tag; every other site is drawn faded, so
    the repeating motif is obvious at a glance. The primitive vectors are
    drawn as arrows from the origin and the unit cell they span is shaded.

    :param lat: **Lattice** instance. Only *unit_cell* and *prim_vec* are
        used, and the instance is not modified.
    :param n1: Positive integer. Unit cells drawn along :math:`\mathbf{a}_1`.
    :param n2: Positive integer. Unit cells drawn along :math:`\mathbf{a}_2`
        (ignored for a 1D lattice).
    :param title: String. Default value None. Axis title.
    :param ax: Axes. Default value None. Axis to draw on; one is created
        (with its own figure) if omitted.
    :param figsize: Tuple. Default value (5.4, 3.8). Figure size, when *ax*
        is not given.
    :param show_vectors: Boolean. Draw the primitive vectors as arrows.
    :param show_cell: Boolean. Shade the unit cell.
    :param label_cell: Boolean. Label the home unit cell's sites by tag.

    :returns:
        * **ax** -- Axes.
    """
    dim = len(lat.prim_vec)
    if dim == 1:
        n2 = 1
    patch = _patch(lat, n1, n2)
    coor = patch.coor
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    pairs, d_min = _bonds(coor)
    for i, j in pairs:
        ax.plot([coor['x'][i], coor['x'][j]], [coor['y'][i], coor['y'][j]],
                    '-', color='0.75', lw=1.1, zorder=1)

    # which sites belong to the home unit cell
    home = np.zeros(len(coor), bool)
    for dic in lat.unit_cell:
        hit = (np.isclose(coor['x'], dic['r0'][0], atol=1e-6)
                  & np.isclose(coor['y'], dic['r0'][1], atol=1e-6))
        home |= hit

    tags = sorted({dic['tag'] for dic in lat.unit_cell})
    for k, tag in enumerate(tags):
        color = COLORS[k % len(COLORS)]
        sel = coor['tag'] == tag
        ax.plot(coor['x'][sel & ~home], coor['y'][sel & ~home], 'o',
                    color=color, ms=8, alpha=0.3, markeredgecolor='none', zorder=2)
        ax.plot(coor['x'][sel & home], coor['y'][sel & home], 'o',
                    color=color, ms=11, markeredgecolor='k', markeredgewidth=1.2,
                    zorder=4)

    a1 = np.array(lat.prim_vec[0], dtype='f8')
    a2 = np.array(lat.prim_vec[1], dtype='f8') if dim == 2 else None

    if show_cell:
        if dim == 2:
            cell = np.array([[0, 0], a1, a1 + a2, a2])
        else:
            # 1D: the cell is one a1 long, and as wide as the motif itself --
            # thin for a chain, the full width of the column for a ribbon.
            perp = np.array([-a1[1], a1[0]])
            perp = perp / np.linalg.norm(perp)
            r0 = np.array([dic['r0'] for dic in lat.unit_cell], dtype='f8')
            proj = r0 @ perp
            pad = 0.3*(d_min or np.linalg.norm(a1))
            # slide the parallelogram along the motif's own extent, so a
            # slanted column (a ribbon) is enclosed as tightly as a chain
            p_lo = r0[np.argmin(proj)] - pad*perp
            p_hi = r0[np.argmax(proj)] + pad*perp
            cell = np.array([p_lo, p_lo + a1, p_hi + a1, p_hi])
        ax.add_patch(plt.Polygon(cell, closed=True, facecolor='0.85', edgecolor='0.55',
                                              lw=1.0, alpha=0.55, zorder=0))

    if show_vectors:
        if dim == 1:
            motif = np.array([dic['r0'] for dic in lat.unit_cell], dtype='f8').mean(axis=0)
            away = motif if np.linalg.norm(motif) > 1e-9 else None
        else:
            away = a2
        for vec, name, other in ((a1, r'$\mathbf{a}_1$', away),
                                            (a2, r'$\mathbf{a}_2$', a1)):
            if vec is None:
                continue
            ax.annotate('', xy=vec, xytext=(0, 0), zorder=5,
                            arrowprops=dict(arrowstyle='-|>', lw=1.8, color='k',
                                                    shrinkA=0, shrinkB=0))
            perp = np.array([-vec[1], vec[0]])
            norm = np.linalg.norm(perp)
            perp = perp / norm if norm else perp
            # keep the label outside the shaded cell, where the unit-cell
            # sites and their tags are
            if other is not None and np.dot(perp, other) > 0:
                perp = -perp
            ax.annotate(name, xy=0.5*vec, xytext=18*perp, textcoords='offset points',
                            fontsize=13, zorder=5, ha='center', va='center')

    if label_cell:
        for dic in lat.unit_cell:
            at_origin = np.allclose(dic['r0'], 0., atol=1e-9)
            offset = (-16, -16) if at_origin else (10, 9)
            ax.annotate(dic['tag'], xy=dic['r0'], xytext=offset,
                            textcoords='offset points', fontsize=12, zorder=5,
                            ha='center', va='center')

    ax.set_aspect('equal')
    ax.axis('off')
    if title:
        ax.set_title(title, fontsize=13)
    ax.margins(0.12)
    ax.figure.set_layout_engine('tight')
    return ax


def plot_flake(coor, title=None, ax=None, figsize=(4.6, 4.6), ms=5, c='#3b76af'):
    """
    Draw a finite flake from a structured *coor* array (x, y, tag), bonds
    included -- for the real-space geometries (rings, flakes, ribbons)
    that are cut rather than tiled.

    :param coor: Structured array with keys {'x', 'y', 'tag'}.
    :param title: String. Default value None. Axis title.
    :param ax: Axes. Default value None.
    :param figsize: Tuple. Default value (4.6, 4.6). Figure size.
    :param ms: Positive number. Default value 5. Marker size.
    :param c: Default value '#3b76af'. Marker colour, or a list of colours,
        one per sorted tag.

    :returns:
        * **ax** -- Axes.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    pairs, _ = _bonds(coor)
    for i, j in pairs:
        ax.plot([coor['x'][i], coor['x'][j]], [coor['y'][i], coor['y'][j]],
                    '-', color='0.75', lw=0.9, zorder=1)
    tags = sorted(set(coor['tag']))
    colors = [c] * len(tags) if isinstance(c, str) else c
    for k, tag in enumerate(tags):
        sel = coor['tag'] == tag
        ax.plot(coor['x'][sel], coor['y'][sel], 'o', color=colors[k % len(colors)],
                    ms=ms, markeredgecolor='none', zorder=2,
                    label="'{}'".format(tag) if len(tags) > 1 else None)
    ax.set_aspect('equal')
    ax.axis('off')
    if title:
        ax.set_title(title, fontsize=13)
    if len(tags) > 1:
        ax.legend(loc='upper right', frameon=False, fontsize=11)
    ax.figure.set_layout_engine('tight')
    return ax


def plot_ssh_chain(
    n_cells=5,
    v=0.45,
    w=1.0,
    gain_loss=False,
    title=None,
    ax=None,
    figsize=(7.0, 2.6),
):
    r"""
    Draw the dimerized (SSH) chain, drawing the intracell bond *v* and the
    intercell bond *w* with thicknesses proportional to their amplitude, so
    the dimerization pattern -- and therefore which end the chain is cut on
    -- is visible rather than merely asserted.

    :param n_cells: Positive integer. Number of unit cells drawn.
    :param v: Positive real number. Intracell hopping.
    :param w: Positive real number. Intercell hopping.
    :param gain_loss: Boolean. Default value False. Mark sublattice 'a' with
        ``+i`` gain and 'b' with ``-i`` loss.
    :param title: String. Default value None. Axis title.
    :param ax: Axes. Default value None.
    :param figsize: Tuple. Default value (7.0, 2.6). Figure size.

    :returns:
        * **ax** -- Axes.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    scale = 7.5 / max(v, w)
    xs_a = np.arange(n_cells, dtype='f8')
    xs_b = xs_a + 0.5
    for n in range(n_cells):
        ax.plot([xs_a[n], xs_b[n]], [0, 0], '-', color='0.35', lw=v*scale, zorder=1)
        if n < n_cells - 1:
            ax.plot([xs_b[n], xs_a[n+1]], [0, 0], '-', color='0.35', lw=w*scale, zorder=1)
    ax.plot(xs_a, np.zeros(n_cells), 'o', color=COLORS[0], ms=15,
                markeredgecolor='k', markeredgewidth=1.2, zorder=3)
    ax.plot(xs_b, np.zeros(n_cells), 'o', color=COLORS[1], ms=15,
                markeredgecolor='k', markeredgewidth=1.2, zorder=3)
    ax.annotate('a', xy=(xs_a[1], 0), xytext=(0, -21), textcoords='offset points',
                    ha='center', va='center', fontsize=12)
    ax.annotate('b', xy=(xs_b[1], 0), xytext=(0, -21), textcoords='offset points',
                    ha='center', va='center', fontsize=12)
    ax.annotate('$v$', xy=(0.5*(xs_a[1] + xs_b[1]), 0), xytext=(0, 20),
                    textcoords='offset points', ha='center', va='center', fontsize=13)
    ax.annotate('$w$', xy=(0.5*(xs_b[1] + xs_a[2]), 0), xytext=(0, 20),
                    textcoords='offset points', ha='center', va='center', fontsize=13)
    if gain_loss:
        # one label per sublattice: the colours already say which is which
        ax.annotate(r'every $a$: $+i\gamma$ (gain)', xy=(xs_a[0], 0), xytext=(-4, 30),
                        textcoords='offset points', ha='left', va='center',
                        fontsize=11, color=COLORS[0])
        ax.annotate(r'every $b$: $-i\gamma$ (loss)', xy=(xs_b[0], 0), xytext=(-4, -34),
                        textcoords='offset points', ha='left', va='center',
                        fontsize=11, color=COLORS[1])
    ax.set_aspect('equal')
    ax.axis('off')
    ax.margins(0.08, 1.0 if gain_loss else 0.7)
    if title:
        ax.set_title(title, fontsize=13, pad=30 if gain_loss else 18)
    ax.figure.set_layout_engine('tight')
    return ax


def strip_unit_cell(lat):
    r"""
    Fold a ribbon's unit cell back into a level strip.

    :func:`tbkit.kspace.ribbon` stacks its rows along the *other* primitive
    vector, which for a honeycomb lattice has a component along the periodic
    direction too. Tiling that cell therefore draws the ribbon as a slanted
    parallelogram. Sliding each site back by whole multiples of
    :math:`\mathbf{a}_1` changes nothing physically -- it is the same site of
    the same cell, relabelled by a different :math:`\mathbf{R}` -- but draws
    the ribbon the way one pictures it: a level strip with two open edges.

    :param lat: **Lattice** instance with a single primitive vector (a
        ribbon's ``rib.lat``). It is not modified.

    :returns:
        * **lat** -- A new **Lattice** with the folded unit cell.
    """
    a1 = np.array(lat.prim_vec[0], dtype='f8')
    a1_sq = a1 @ a1
    folded = []
    for dic in lat.unit_cell:
        r0 = np.array(dic['r0'], dtype='f8')
        r0 = r0 - np.round((r0 @ a1) / a1_sq) * a1
        folded.append({'tag': dic['tag'], 'r0': (float(r0[0]), float(r0[1]))})
    return Lattice(unit_cell=folded, prim_vec=lat.prim_vec)
