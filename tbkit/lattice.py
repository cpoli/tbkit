from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from scipy.spatial import cKDTree
import tbkit.error_handling as error_handling


PI = np.pi
COOR_DTYPE = [('x', 'f8'), ('y', 'f8'), ('tag', 'U1')]
#: Site dtype of a lattice in 3D space (*unit_cell* and *prim_vec* given as 3-tuples).
COOR_DTYPE_3D = [('x', 'f8'), ('y', 'f8'), ('z', 'f8'), ('tag', 'U1')]


#################################
# CLASS LATTICE
#################################


class Lattice():
    r'''
    Build up 1D, 2D or 3D lattice.
    Lattice is defined by the discrete operation:

    .. math::

        \mathbf{R} = n_1\mathbf{a}_1 + n_2\mathbf{a}_2 + n_3\mathbf{a}_3

    where :math:`\mathbf{a}_1`, :math:`\mathbf{a}_2`, :math:`\mathbf{a}_3` are the
    primitive vectors and :math:`n_1`, :math:`n_2`, :math:`n_3` are the number of
    unit cells along them.

    :param unit_cell: List of dictionaries.
     One dictionary per site within the unit cell. Each dictionary has two keys:

        * 'tag', one-character string. Label of the associated sublattice.
        * 'r0', Tuple. Position: :math:`(x, y)`, or :math:`(x, y, z)` for a
          lattice in 3D space.
    :param prim_vec: List of tuples.
     Define the primitive vectors. List of one/two/three tuples for 1D/2D/3D
     lattices, each the cartesian coordinates of a primitive vector: 2-tuples
     in the plane, or 3-tuples in 3D space (then *r0* must be 3-tuples too,
     and *coor* gains a 'z' field, see :data:`COOR_DTYPE_3D`).

    In 3D space, the geometry methods acting in the plane (*rotation*,
    *ellipse_in*, *boundary_line*, ...) act on :math:`(x, y)` only (rotation
    about the z axis, cylinders, vertical planes), and *plot* draws the
    projection on the :math:`(x, y)` plane.

    Example usage::

        # Line-Centered Square lattice
        unit_cell = [{'tag': 'a', 'r0': (0., 0.)}, {'tag': 'a', 'r0': (0., 1.)}]
        prim_vec = [(0, 2), (2, 0)]
        lat = lattice(unit_cell=unit_cell, prim_vec=prim_vec)
    '''

    def __init__(self, unit_cell: list[dict], prim_vec: list[tuple[float, float]]) -> None:
        error_handling.unit_cell(unit_cell)
        error_handling.prim_vec(prim_vec)
        error_handling.space_dim(unit_cell, prim_vec)
        self.unit_cell = unit_cell
        self.prim_vec = prim_vec
        self.tags = np.unique(np.array([dic['tag'] for dic in self.unit_cell]))
        self.space_dim = len(prim_vec[0])  # 2, or 3 for a lattice in 3D space
        self.dtype = COOR_DTYPE if self.space_dim == 2 else COOR_DTYPE_3D
        self.n1, self.n2, self.n3 = 0, 0, 0
        self.coor = np.array([], dtype=self.dtype)
        self.sites = 0

    def get_lattice(self, n1: int, n2: int = 1, n3: int = 1) -> None:
        r'''
        Get the lattice positions, sorted by (y, x) -- by (z, y, x) in 3D space.

        :param n1: Positive Integer.
            Number of unit cells along :math:`\mathbf{a}_1`.
        :param n2: Positive Integer. Default value 1.
            Number of unit cells along :math:`\mathbf{a}_2`.
        :param n3: Positive Integer. Default value 1.
            Number of unit cells along :math:`\mathbf{a}_3` (3D lattices only).

        Example usage::

            # Line-Centered Square lattice
            unit_cell = [{'tag': 'a', 'r0': (0., 0.)}, {'tag': 'a', 'r0': (0., 1.)}]
            prim_vec = [(0, 2), (2, 0)]
            lat = lattice(unit_cell=unit_cell, prim_vec=prim_vec)
            lat.get_lattice(n1=4, n2=5)
        '''
        error_handling.get_lattice(self.prim_vec, n1, n2, n3)
        sites_uc = len(self.unit_cell)
        sites_tag = n1*n2*n3
        self.sites = sites_uc * sites_tag
        self.coor = np.empty(self.sites, dtype=self.dtype)
        self.n1, self.n2, self.n3 = n1, n2, n3
        # translations R = i1 a1 + i2 a2 (+ i3 a3), i1 running fastest
        axes = [f for f in ('x', 'y', 'z')[:self.space_dim]]
        trans = np.zeros((sites_tag, self.space_dim))
        counts = (n1, n2, n3)
        index = np.indices(counts[:max(len(self.prim_vec), 1)][::-1]).reshape(
                                  max(len(self.prim_vec), 1), -1)[::-1]
        for n, a in zip(index, self.prim_vec):
            trans += n[:, None] * np.array(a, dtype='f8')[None, :]
        for i, dic in enumerate(self.unit_cell):
            for c, f in enumerate(axes):
                self.coor[f][i*sites_tag: (i+1)*sites_tag] = trans[:, c] + dic['r0'][c]
            self.coor['tag'][i*sites_tag: (i+1)*sites_tag] = dic['tag']
        self.coor = np.sort(self.coor, order=self.sort_order())

    def sort_order(self) -> tuple[str, ...]:
        '''
        Private method. Fields the sites are sorted by: ('y', 'x'), or
        ('z', 'y', 'x') in 3D space.
        '''
        return ('y', 'x') if self.space_dim == 2 else ('z', 'y', 'x')

    def distances(self) -> tuple[NDArray, ...]:
        '''
        Private method. Pairwise coordinate differences, *d[i, j]* from site
        *i* to site *j*: (dx, dy), or (dx, dy, dz) in 3D space.
        '''
        return tuple(self.coor[f] - self.coor[f].reshape(self.sites, 1)
                          for f in ('x', 'y', 'z')[:self.space_dim])

    def add_sites(self, coor: NDArray) -> None:
        '''
        Add sites.

        :param coor: Structured array with keys: {'x', 'y', 'tag'}
            ({'x', 'y', 'z', 'tag'} for a lattice in 3D space).

        Example usage::

            # Square lattice
            unit_cell = [{'tag': 'a', 'r0': (0., 0.)}]
            prim_vec = [(0, 1), (1, 0)]
            lat = lattice(unit_cell=unit_cell, prim_vec=prim_vec)
            lat.get_lattice(n1=2, n2=2)
            coor = np.array([(-1., -1, 'b'), (-2., -2, 'c')],
                                      dtype=[('x', 'f8'), ('y', 'f8'), ('tag', 'U1')])
            lat.add_sites(coor)
        '''
        error_handling.coor(coor, self.dtype)
        self.coor = np.concatenate([self.coor, coor])
        self.sites += len(coor)
        self.tags = np.unique(np.concatenate([self.tags, coor['tag']]))
        self.coor = np.sort(self.coor, order=self.sort_order())

    def remove_sites(self, index: list[int]) -> None:
        '''
        Remove sites defined by their indices
        (use method lattice.plot(plt_index=True)
        to get access to the site indices).

        :param index: List. Site indices to be removed.

        Example usage::

            # Square lattice
            unit_cell = [{'tag': 'a', 'r0': (0., 0.)}]
            prim_vec = [(0, 1), (1, 0)]
            lat = lattice(unit_cell=unit_cell, prim_vec=prim_vec)
            lat.get_lattice(n1=2, n2=2)
            lat.remove_sites([0, 2])
        '''
        error_handling.empty_coor(self.coor)
        error_handling.remove_sites(index, self.sites)
        mask = np.ones(self.sites, bool)
        mask[index] = False
        self.coor = self.coor[mask]
        self.sites = self.coor.size

    def remove_dangling(self) -> None:
        '''
        Remove dangling sites
        (sites connected with just another site).
        '''
        error_handling.empty_coor(self.coor)
        while True:
            coords = np.stack([self.coor[f] for f in ('x', 'y', 'z')[:self.space_dim]], axis=1)
            tree = cKDTree(coords)
            near = tree.query(coords, k=min(self.sites, 8))[0][:, 1:]
            len_hop = np.min(near[near > 0])  # the shortest distance between sites
            pairs = tree.query_pairs(len_hop * (1. + 1e-5) + 1e-8, output_type='ndarray')
            dis = np.linalg.norm(coords[pairs[:, 1]] - coords[pairs[:, 0]], axis=1)
            pairs = pairs[np.isclose(dis, len_hop)]
            degree = np.bincount(pairs.ravel(), minlength=self.sites)
            dang = list(np.flatnonzero(degree == 1))
            self.coor = np.delete(self.coor, dang, axis=0)
            self.sites -= len(dang)
            if dang == []:
                break

    def shift_x(self, shift: float) -> None:
        '''
        Shift the x coordinates.

        :param shift: Real number. Shift value.
        '''
        error_handling.empty_coor(self.coor)
        error_handling.real_number(shift, 'shift')
        self.coor['x'] += shift

    def shift_y(self, shift: float) -> None:
        '''
        Shift the y coordinates.

        :param shift: Real number. Shift value.
        '''
        error_handling.empty_coor(self.coor)
        error_handling.real_number(shift, 'shift')
        self.coor['y'] += shift

    def change_sign_x(self) -> None:
        '''
        Change x coordinates sign.
        '''
        error_handling.empty_coor(self.coor)
        self.coor['x'] *= -1

    def change_sign_y(self) -> None:
        '''
        Change y coordinates sign.
        '''
        error_handling.empty_coor(self.coor)
        self.coor['y'] *= -1

    def shift_z(self, shift: float) -> None:
        '''
        Shift the z coordinates (3D lattices only).

        :param shift: Real number. Shift value.
        '''
        error_handling.empty_coor(self.coor)
        error_handling.space_3d(self.space_dim)
        error_handling.real_number(shift, 'shift')
        self.coor['z'] += shift

    def change_sign_z(self) -> None:
        '''
        Change z coordinates sign (3D lattices only).
        '''
        error_handling.empty_coor(self.coor)
        error_handling.space_3d(self.space_dim)
        self.coor['z'] *= -1

    def slab(self, z_min: float, z_max: float) -> None:
        r'''
        Keep only the sites with :math:`z_{min} < z < z_{max}` (3D lattices only).

        :param z_min: Real number.
        :param z_max: Real number, larger than *z_min*.
        '''
        error_handling.empty_coor(self.coor)
        error_handling.space_3d(self.space_dim)
        error_handling.real_number(z_min, 'z_min')
        error_handling.real_number(z_max, 'z_max')
        error_handling.smaller(z_min, 'z_min', z_max, 'z_max')
        self.coor = self.coor[(self.coor['z'] > z_min) & (self.coor['z'] < z_max)]
        self.sites = len(self.coor)

    def boundary_line(self, cx: float, cy: float, co: float) -> None:
        r'''
        Keep only the sites with :math:`c_yy+c_xx > c_0`.

        :param cx: Real number. :math:`c_x` value.
        :param cy: Real number. :math:`c_y` value.
        :param co: Real number. :math:`c_0` value.
        '''
        error_handling.empty_coor(self.coor)
        error_handling.real_number(cx, 'cx')
        error_handling.real_number(cy, 'cy')
        error_handling.real_number(co, 'co')
        self.coor = self.coor[cy * self.coor['y'] + cx * self.coor['x'] > co]
        self.sites = len(self.coor)

    def ellipse_in(self, rx: float, ry: float, x0: float, y0: float) -> None:
        r'''
        Select sites according to

        .. math::

            (x-x_0)^2/r_x^2+(y-y_0)^2/r_y^2 < 1\,  .

        :param rx: Positive Real number. Radius along :math:`x`.
        :param ry: Positive Real number. Radius along :math:`y`.
        :param x0: Real number. :math:`x` center.
        :param y0: Real number. :math:`y` center.
        '''
        error_handling.empty_coor(self.coor)
        error_handling.positive_real(rx, 'rx')
        error_handling.positive_real(ry, 'ry')
        error_handling.real_number(x0, 'x0')
        error_handling.real_number(y0, 'y0')
        self.coor = self.coor[(self.coor['x'] -x0) ** 2 / rx ** 2 +  \
                                        (self.coor['y'] -y0) ** 2 / ry ** 2 < 1.]
        self.sites = len(self.coor)

    def ellipse_out(self, rx: float, ry: float, x0: float, y0: float) -> None:
        r'''
        Select sites according to

        .. math::

            (x-x_0)^2/r_x^2+(y-y_0)^2/r_y^2 > 1\,  .


        :param rx: Positive Real number. Radius along :math:`x`.
        :param ry: Positive Real number. Radius along :math:`y`.
        :param x0: Real number. :math:`x` center.
        :param y0: Real number. :math:`y` center.
        '''
        error_handling.empty_coor(self.coor)
        error_handling.positive_real(rx, 'rx')
        error_handling.positive_real(ry, 'ry')
        error_handling.real_number(x0, 'x0')
        error_handling.real_number(y0, 'y0')
        self.coor = self.coor[(self.coor['x'] -x0) ** 2 / rx ** 2 +  \
                                        (self.coor['y'] -y0) ** 2 / ry ** 2 > 1.]
        self.sites = len(self.coor)

    def center(self) -> None:
        '''
        Fix the center of mass of the lattice at (0, 0) -- (0, 0, 0) in 3D space.
        '''
        error_handling.empty_coor(self.coor)
        for f in ('x', 'y', 'z')[:self.space_dim]:
            self.coor[f] -= np.mean(self.coor[f])

    def rotation(self, theta: float) -> None:
        r'''
        Rotate the lattice structure about the origin (about the z axis, in
        3D space) by the angle :math:`\theta`.

        :param theta: Rotation angle in degrees.
        '''
        error_handling.empty_coor(self.coor)
        error_handling.real_number(theta, 'theta')
        theta *= PI / 180
        x = self.coor['x'].copy()
        y = self.coor['y'].copy()
        self.coor['x'] = x * np.cos(theta) - y * np.sin(theta)
        self.coor['y'] = y * np.cos(theta) + x * np.sin(theta)

    def clean_coor(self) -> None:
        '''
        Keep only the sites with different coordinates.
        '''
        error_handling.empty_coor(self.coor)
        fields = list(('x', 'y', 'z')[:self.space_dim])
        coor = self.coor[fields].copy()
        for f in fields:
            coor[f] = self.coor[f].round(4)
        _, idx = np.unique(coor, return_index=True)
        self.coor = self.coor[idx]
        self.sites = len(self.coor)

    def __add__(self, other: 'Lattice') -> 'Lattice':
        '''
        Overloading operator +.
        '''
        error_handling.lat(other)
        error_handling.empty_coor(self.coor)
        error_handling.empty_coor(other.coor)
        coor = np.concatenate([self.coor, other.coor])
        tags = np.concatenate([self.tags, other.tags])
        lat = lattice(unit_cell=self.unit_cell, prim_vec=self.prim_vec)
        lat.add_sites(coor)
        lat.sites = self.sites + other.sites
        lat.tags = np.unique(tags)
        return lat

    def __iadd__(self, other: 'Lattice') -> 'Lattice':
        '''
        Overloading operator +=.
        '''
        error_handling.lat(other)
        error_handling.empty_coor(self.coor)
        error_handling.empty_coor(other.coor)
        self.coor = np.concatenate([self.coor, other.coor])
        self.sites += other.sites
        self.tags = np.unique(np.concatenate([self.tags, other.tags]))
        return self

    def __sub__(self, other: 'Lattice') -> 'Lattice':
        '''
        Overloading operator -.

        .. note::

            The tags are not considered in the lattice subtraction.
        '''
        error_handling.lat(other)
        error_handling.empty_coor(self.coor)
        error_handling.empty_coor(other.coor)
        boo = self.overlaps(other)
        coor = self.coor[np.logical_not(boo)]
        lat = lattice(unit_cell=self.unit_cell, prim_vec=self.prim_vec)
        lat.add_sites(coor)
        lat.sites = len(lat.coor)
        lat.tags = self.tags
        return lat

    def __isub__(self, other: 'Lattice') -> 'Lattice':
        '''
        Overloading operator -=.

        .. note::

            The tags are not considered in the lattice subtraction.
        '''
        error_handling.lat(other)
        error_handling.empty_coor(self.coor)
        error_handling.empty_coor(other.coor)
        boo = self.overlaps(other)
        self.coor = self.coor[np.logical_not(boo)]
        self.sites = int(np.sum(np.logical_not(boo)))
        return self

    def overlaps(self, other: 'Lattice') -> NDArray[np.bool_]:
        '''
        Private method. Mask of the sites of *self* that coincide with a site
        of *other*.
        '''
        fields = [f for f in ('x', 'y', 'z') if f in self.coor.dtype.names]
        boo = np.zeros(self.sites, bool)
        for c in other.coor:
            match = np.ones(self.sites, bool)
            for f in fields:
                match &= np.isclose(c[f] if f in other.coor.dtype.names else 0., self.coor[f])
            boo += match
        return boo

    def plot(
        self,
        ms: float = 20,
        fs: float = 20,
        plt_index: bool = False,
        axis: bool = False,
        figsize: tuple[float, float] | None = None,
    ) -> Figure:
        '''
        Plot the lattice in real space (its projection on the (x, y) plane,
        in 3D space).

        :param ms: Positive number. Default value 20. Markersize.
        :param fs: Positive number. Default value 20. Fontsize.
        :param plt_index: Boolean. Default value False. Plot site labels.
        :param axis: Boolean. Default value False. Plot axis.
        :param figsize: Tuple. Default value None. Figsize.

        :returns:
            * **fig** -- Figure.
        '''
        error_handling.empty_coor(self.coor)
        error_handling.positive_real(ms, 'ms')
        error_handling.positive_real(fs, 'fs')
        error_handling.boolean(plt_index, 'plt_index')
        error_handling.boolean(axis, 'axis')
        if figsize is None:
            figsize = (5, 5)
        error_handling.list_tuple_2elem(figsize, 'figsize')
        error_handling.positive_real(figsize[0], 'figsize[0]')
        error_handling.positive_real(figsize[1], 'figsize[1]')
        fig, ax = plt.subplots(figsize=figsize)
        # plot sites
        colors = ['b', 'r', 'g', 'y', 'm', 'k']
        for color, tag in zip(colors, self.tags):
            plt.plot(self.coor['x'][self.coor['tag'] == tag],
                        self.coor['y'][self.coor['tag'] == tag],
                       'o', color=color, ms=ms, markeredgecolor='none')
        ax.set_aspect('equal')
        ax.set_xlim([np.min(self.coor['x'])-1., np.max(self.coor['x'])+1.])
        ax.set_ylim([np.min(self.coor['y'])-1., np.max(self.coor['y'])+1.])
        if not axis:
            ax.axis('off')
        # plot indices
        if plt_index:
            indices = ['{}'.format(i) for i in range(self.sites)]
            for l, x, y in zip(indices, self.coor['x'], self.coor['y']):
                plt.annotate(l, xy=(x, y), xytext=(0, 0),
                                    textcoords='offset points',
                                    ha='right', va='bottom', size=fs)
        plt.draw()
        return fig

    def show(self) -> None:
        """
        Emulate Matplotlib method plt.show().
        """
        plt.show()


# Backward-compatible lowercase alias (pre-0.2 API).
lattice = Lattice
