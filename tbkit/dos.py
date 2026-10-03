"""
Density of states from a set of eigenenergies, real-space
(:class:`tbkit.system.System`) or reciprocal-space
(:class:`tbkit.kspace.KSpace`, sampled over a k-mesh).
"""
from __future__ import annotations

from itertools import permutations

import numpy as np
from numpy.typing import ArrayLike, NDArray
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

import tbkit.error_handling as error_handling


def density_of_states(
    energies: ArrayLike,
    e_grid: ArrayLike | None = None,
    broadening: float = 0.05,
    kernel: str = 'gaussian',
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    r'''
    Get the density of states, broadened by a Gaussian or Lorentzian
    kernel of width *broadening*:

    .. math::

        \rho(E) = \sum_n g(E-E_n)\, ,\quad
        g(x) = \frac{1}{\sqrt{2\pi}\sigma}e^{-x^2/2\sigma^2}\ \text{(gaussian)}
        \ \text{or}\
        g(x) = \frac{1}{\pi}\frac{\sigma}{x^2+\sigma^2}\ \text{(lorentzian)}

    Each level contributes a kernel of unit area, so
    :math:`\int\rho(E)dE` equals the number of levels in *energies*,
    for an *e_grid* wide enough to contain the tails.

    :param energies: Array of (real) eigenenergies. Any shape (e.g. the
        *en* attribute of **System**, or of **KSpace** after *get_bands*
        over a k-mesh -- flattened automatically).
    :param e_grid: Real ndarray. Default value None. Energies at which to
        evaluate the density of states. If None, a grid of 401 points
        spanning ``[min(energies)-3*broadening, max(energies)+3*broadening]``
        is used.
    :param broadening: Positive real number. Default value 0.05. Kernel width
        :math:`\sigma`.
    :param kernel: String. Default value 'gaussian'. 'gaussian' or 'lorentzian'.

    :returns:
        * **e_grid** -- Real ndarray. The energy grid used.
        * **dos** -- Real ndarray, same shape as *e_grid*. Density of states.
    '''
    error_handling.ndarray_empty(np.asarray(energies), 'energies')
    error_handling.positive_real(broadening, 'broadening')
    error_handling.dos_kernel(kernel)
    energies = np.asarray(energies).real.astype('f8').ravel()
    if e_grid is None:
        pad = 3 * broadening
        e_grid = np.linspace(energies.min() - pad, energies.max() + pad, 401)
    else:
        error_handling.ndarray_empty(np.asarray(e_grid), 'e_grid')
        e_grid = np.asarray(e_grid, dtype='f8')
    diff = e_grid[:, None] - energies[None, :]
    if kernel == 'gaussian':
        weight = np.exp(-diff**2 / (2*broadening**2)) / (broadening*np.sqrt(2*np.pi))
    else:
        weight = (broadening/np.pi) / (diff**2 + broadening**2)
    return e_grid, weight.sum(axis=1)


def _mesh_simplices(dim: int) -> NDArray[np.int64]:
    '''
    Private function. Split each cell of a periodic *dim*-dimensional mesh
    (*dim* = 1, 2, 3) into *dim*! simplices of equal volume (segments,
    triangles, tetrahedra), all sharing the cell's main diagonal (the
    decomposition of Blochl et al.). The vertices are given as integer
    offsets from the cell's lowest corner: shape (dim!, dim+1, dim), the
    vertex m of simplex p being the sum of the first m unit vectors of
    the permutation p of the axes.
    '''
    eye = np.eye(dim, dtype=int)
    return np.array([np.cumsum([np.zeros(dim, int)] + [eye[a] for a in perm], axis=0)
                            for perm in permutations(range(dim))])


def _simplex_values(values: NDArray, offsets: NDArray[np.int64]) -> NDArray:
    '''
    Private function. Values of *values* (shape (n1, ..., n_dim, ...),
    periodic over its first dim axes) at the vertices of every simplex of
    *_mesh_simplices*: shape (dim+1, dim!, n1, ..., n_dim, ...).
    '''
    dim = offsets.shape[-1]
    axes = tuple(range(dim))
    return np.stack([np.stack([np.roll(values, tuple(-o), axis=axes) for o in simplex])
                           for simplex in offsets], axis=1)


def _simplex_fraction(v: NDArray[np.float64], e: float) -> NDArray[np.float64]:
    r'''
    Private function. Fraction of the volume of each simplex where the
    linear interpolation of its vertex values is below *e*, for simplices
    that *e* cuts (``v[0] <= e < v[-1]``): *v* has shape (dim+1, n), sorted
    along the first axis. In 3D, the integrated density of states of the
    linear tetrahedron method (Blochl et al. 1994, appendix).
    '''
    dim = len(v) - 1
    if dim == 1:
        return (e - v[0]) / (v[1] - v[0])
    out = np.empty(v.shape[1])
    if dim == 2:
        low = e < v[1]
        a, b, c = v[:, low]
        out[low] = (e - a) ** 2 / ((b - a) * (c - a))
        a, b, c = v[:, ~low]
        out[~low] = 1. - (c - e) ** 2 / ((c - a) * (c - b))
        return out
    low, high = e < v[1], e >= v[2]
    mid = ~low & ~high
    a, b, c, d = v[:, low]
    out[low] = (e - a) ** 3 / ((b - a) * (c - a) * (d - a))
    a, b, c, d = v[:, high]
    out[high] = 1. - (d - e) ** 3 / ((d - a) * (d - b) * (d - c))
    a, b, c, d = v[:, mid]
    x = e - b
    out[mid] = ((b - a) ** 2 + 3 * (b - a) * x + 3 * x ** 2
                    - (c - a + d - b) / ((c - b) * (d - b)) * x ** 3) / ((c - a) * (d - a))
    return out


def tetrahedron_dos(
    energies: ArrayLike,
    e_grid: ArrayLike | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    r'''
    Get the density of states by the linear tetrahedron method (Jepsen and
    Andersen 1971, Lehmann and Taut 1972, Blochl et al. 1994): the bands
    are interpolated linearly inside the simplices (segments in 1D,
    triangles in 2D, tetrahedra in 3D) of a uniform mesh of the Brillouin
    zone, and the density of states of the interpolated bands is
    integrated exactly. There is no broadening: band edges and van Hove
    singularities (the logarithmic peak of the 2D square lattice, the
    :math:`\sqrt{E}` edges of 3D bands) come out sharp, and converge with
    the mesh instead of with a kernel width.

    The value at ``e_grid[i]`` is the exact average of that density of
    states over the bin between the midpoints to the neighbouring grid
    points, so a flat band (a delta function) still carries its full
    weight, in one or two bins. The normalization is that of
    *density_of_states* on the same energies: :math:`\int\rho(E)dE` equals
    the number of levels, ``energies.size``, for an *e_grid* that spans
    them all. This is the linear method, without Blochl's curvature
    correction. Complex energies (non-Hermitian bands) are replaced by
    their real part.

    :param energies: Real array, shape (n1, n2, ..., nbands): the bands on
        a periodic mesh of 1, 2 or 3 dimensions, ``energies[i1, i2, ..., n]``
        at the k-point :math:`\sum_d (i_d/n_d)\mathbf{b}_d` (as given by
        ``KSpace.mesh_bands(nk).reshape(*nk, norb)``).
    :param e_grid: Real ndarray, increasing. Default value None. Energies at
        which to evaluate the density of states. If None, 401 points
        spanning the bands, with a 2% margin on each side.

    :returns:
        * **e_grid** -- Real ndarray. The energy grid used.
        * **dos** -- Real ndarray, same shape as *e_grid*. Density of states.

    Example usage::

        en = ks.mesh_bands((40, 40)).reshape(40, 40, ks.norb)
        e_grid, rho = dos.tetrahedron_dos(en)
    '''
    error_handling.mesh_energies(energies)
    energies = np.asarray(energies).real.astype('f8')
    dim = energies.ndim - 1
    if e_grid is None:
        span = energies.max() - energies.min()
        pad = 0.02 * (span if span > 0 else 1.)
        e_grid = np.linspace(energies.min() - pad, energies.max() + pad, 401)
    else:
        error_handling.increasing_grid(e_grid, 'e_grid')
        e_grid = np.asarray(e_grid, dtype='f8')
    offsets = _mesh_simplices(dim)
    v = np.sort(_simplex_values(energies, offsets).reshape(dim + 1, -1), axis=0)
    weight = energies.shape[-1] * energies[..., 0].size / v.shape[1]  # levels per simplex
    # integrated density of states at the bin edges, halfway between grid points
    mid = (e_grid[1:] + e_grid[:-1]) / 2
    edges = np.concatenate([[2 * e_grid[0] - mid[0]], mid, [2 * e_grid[-1] - mid[-1]]])
    tops = np.sort(v[-1])
    count = np.empty(len(edges))
    for n, e in enumerate(edges):
        cut = (v[0] <= e) & (e < v[-1])
        count[n] = np.searchsorted(tops, e, side='right') + _simplex_fraction(v[:, cut], e).sum()
    return e_grid, weight * np.diff(count) / np.diff(edges)


def _plot_density_of_states(
    energies: ArrayLike,
    e_grid: ArrayLike | None,
    broadening: float,
    kernel: str,
    fs: float,
    lw: float,
    figsize: tuple[float, float] | None,
) -> Figure:
    '''
    Private function. Plot *density_of_states* (shared by *Plot.dos* and
    *KSpace.plot_dos*, which validate the plotting parameters).
    '''
    e_grid, rho = density_of_states(energies, e_grid=e_grid, broadening=broadening, kernel=kernel)
    return _plot_dos_curve(e_grid, rho, fs, lw, figsize)


def _plot_dos_curve(
    e_grid: NDArray[np.float64],
    rho: NDArray[np.float64],
    fs: float,
    lw: float,
    figsize: tuple[float, float] | None,
) -> Figure:
    '''
    Private function. Plot a density of states *rho* on *e_grid*.
    '''
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(e_grid, rho, 'b', lw=lw)
    ax.fill_between(e_grid, rho, color='b', alpha=0.2)
    ax.set_xlim([e_grid[0], e_grid[-1]])
    ax.set_ylim([0., None])
    ax.set_title('Density of states', fontsize=fs)
    ax.set_xlabel('$E$', fontsize=fs)
    ax.set_ylabel(r'$\rho(E)$', fontsize=fs)
    for label in ax.xaxis.get_majorticklabels():
        label.set_fontsize(fs)
    for label in ax.yaxis.get_majorticklabels():
        label.set_fontsize(fs)
    fig.set_layout_engine('tight')
    plt.draw()
    return fig
