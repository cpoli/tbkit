r"""
Building Finite Lattices: Unit Cells, Cuts, and Lattice Arithmetic
======================================================================

Every real-space model starts from a finite set of sites. A
:class:`~tbkit.lattice.Lattice` repeats a unit cell along its primitive
vectors (*get_lattice*), and the result can be cut and combined:

* *boundary_line* keeps the sites on one side of a line, *ellipse_in* and
  *ellipse_out* inside or outside an ellipse;
* *remove_dangling* strips the sites with a single nearest neighbour,
  *remove_sites* removes sites by index;
* ``lat1 - lat2`` removes the sites of ``lat2`` from ``lat1``, ``lat1 += lat2``
  merges them, and *clean_coor* drops duplicated positions;
* *shift_x*, *shift_y*, *center* and *rotation* move the sites.

(This script replaces the pre-0.2 notebook ``examples_lattice.ipynb``.)
Every site count below is checked against a direct count.
"""
import numpy as np
import matplotlib.pyplot as plt

import tbkit.lattices as lattices
from tbkit.lattice import Lattice

fig, axes = plt.subplots(2, 3, figsize=(12, 8))
axes = axes.ravel()


def show(ax, lat, title):
    for tag in np.unique(lat.coor['tag']):
        sel = lat.coor['tag'] == tag
        ax.plot(lat.coor['x'][sel], lat.coor['y'][sel], 'o', ms=5, label=tag)
    ax.set_aspect('equal')
    ax.set_title('{}: {} sites'.format(title, lat.sites))
    ax.axis('off')


# %%
# A Lieb lattice without dangling sites
# ---------------------------------------
# An :math:`n\times n` Lieb flake has :math:`3n^2` sites; the edge sites of
# the last row and column have a single neighbour and are removed:
# :math:`3n^2 - 2n` remain.

n = 6
lieb = lattices.lieb()
lieb.get_lattice(n1=n, n2=n)
assert lieb.sites == 3 * n * n
lieb.remove_dangling()
assert lieb.sites == 3 * n * n - 2 * n
show(axes[0], lieb, 'Lieb, dangling removed')

# %%
# A face-centred square cut by two lines
# ------------------------------------------

fcs = Lattice(unit_cell=[{'tag': 'a', 'r0': (0., 0.)}, {'tag': 'b', 'r0': (1., 1.)}],
                     prim_vec=[(2., 0.), (0., 2.)])
fcs.get_lattice(n1=8, n2=8)
x, y = fcs.coor['x'].copy(), fcs.coor['y'].copy()
fcs.boundary_line(cx=0., cy=-1., co=-12.5)  # y < 12.5
fcs.boundary_line(cx=-1., cy=0., co=-10.5)  # x < 10.5
assert fcs.sites == np.sum((y < 12.5) & (x < 10.5))
show(axes[1], fcs, 'face-centred square, cut')

# %%
# A graphene ring: a disk minus a smaller disk
# ------------------------------------------------

disk = lattices.honeycomb()
disk.get_lattice(n1=20, n2=20)
disk.center()
hole = lattices.honeycomb()
hole.get_lattice(n1=20, n2=20)
hole.center()
r = np.hypot(disk.coor['x'], disk.coor['y'])
disk.ellipse_in(rx=7., ry=7., x0=0., y0=0.)
hole.ellipse_in(rx=3.5, ry=3.5, x0=0., y0=0.)
ring = disk - hole
assert ring.sites == np.sum((r < 7.) & (r >= 3.5))
ring.remove_dangling()
show(axes[2], ring, 'graphene ring')

# %%
# A dumbbell: two squares and a bridge
# ---------------------------------------

left = lattices.square()
left.get_lattice(n1=7, n2=7)
bridge = lattices.square()
bridge.get_lattice(n1=6, n2=3)
bridge.shift_x(shift=7.)
bridge.shift_y(shift=2.)
right = lattices.square()
right.get_lattice(n1=7, n2=7)
right.shift_x(shift=13.)
left += bridge
left += right
assert left.sites == 49 + 18 + 49
show(axes[3], left, 'dumbbell')

# %%
# Overlapping copies and clean_coor
# ------------------------------------
# Two Lieb flakes shifted by (4, 4) share a 3 x 3-cell block of 27 sites.

a = lattices.lieb()
a.get_lattice(n1=5, n2=5)
b = lattices.lieb()
b.get_lattice(n1=5, n2=5)
b.shift_x(shift=4.)
b.shift_y(shift=4.)
both = a + b
assert both.sites == 150
both.clean_coor()
assert both.sites == 150 - 27
show(axes[4], both, 'overlap cleaned')

# %%
# A rotated ribbon
# -------------------
# A rotation by 45 degrees keeps every distance.

ladder = Lattice(unit_cell=[{'tag': 'a', 'r0': (0., 0.)}, {'tag': 'b', 'r0': (0., 1.)}],
                         prim_vec=[(1., 0.)])
ladder.get_lattice(n1=8)
before = np.hypot(np.subtract.outer(ladder.coor['x'], ladder.coor['x']),
                           np.subtract.outer(ladder.coor['y'], ladder.coor['y']))
ladder.rotation(theta=45.)
after = np.hypot(np.subtract.outer(ladder.coor['x'], ladder.coor['x']),
                         np.subtract.outer(ladder.coor['y'], ladder.coor['y']))
assert np.allclose(before, after)
show(axes[5], ladder, 'ladder rotated by 45 deg')
fig.tight_layout()
