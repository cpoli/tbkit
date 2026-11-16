from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
import scipy.sparse as sparse
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.animation import FuncAnimation
import os
import tbkit.error_handling as error_handling
import tbkit.kpm as kpm
from tbkit.lattice import Lattice



def _crank_nicolson(ham, dz: float):
    r'''
    Private function. One Crank-Nicolson step,
    :math:`\psi \mapsto (i - \tfrac{dz}{2}H)^{-1}(i + \tfrac{dz}{2}H)\psi`,
    with :math:`i - \tfrac{dz}{2}H` factorized once (sparse LU).
    '''
    ham = sparse.csc_matrix(ham, dtype='c16')
    eye = sparse.identity(ham.shape[0], dtype='c16', format='csc')
    lu = spla.splu((1j * eye - 0.5 * dz * ham).tocsc())
    rhs = (1j * eye + 0.5 * dz * ham).tocsr()
    return lambda psi: lu.solve(rhs @ psi)


#################################
# CLASS PROPAGATION
#################################


class Propagation():
    r'''
    Get lattice time evolution. The time-dependent Schrodinger equation
    :math:`i\,\partial_z\psi = H\psi` is solved by the Crank-Nicolson method,

    .. math::

        \psi(z+dz) = \left(1 + \tfrac{i}{2}H\,dz\right)^{-1}
                     \left(1 - \tfrac{i}{2}H\,dz\right)\psi(z)\, ,

    unitary for a Hermitian :math:`H`. The matrix
    :math:`1 + \tfrac{i}{2}H\,dz` is factorized once per Hamiltonian
    (sparse LU, *scipy.sparse.linalg.splu*) and never inverted, so each
    step costs one sparse product and two sparse triangular solves.
    *get_propagation* also takes ``method='chebyshev'``: the exact evolution
    :math:`e^{-iH\,dz}` from its Chebyshev expansion
    (*tbkit.kpm.time_evolution*), for Hermitian Hamiltonians, with no
    factorization at all and no time-step error.

    Parameters
    ----------
    lat : **lattice** class instance
    '''

    def __init__(self, lat: Lattice) -> None:
        error_handling.lat(lat)
        self.lat = lat
        self.prop = np.array([], 'c16')

    def get_propagation(
        self, ham: sparse.spmatrix, psi_init: NDArray, steps: int, dz: float, norm: bool = False,
        method: str = 'crank-nicolson',
    ) -> None:
        r'''
        Get the time evolution.

        Parameters
        ----------
        ham : sparse.csr_matrix
            Tight-Binding Hamiltonian.
        psi_init : np.ndarray
            Initial state.
        steps : Positive Integer
            Number of steps.
        dz : Positive number
            Step.
        norm : Boolean, default False
            Renormalize the state to
            :math:`\sum_i|\psi_i|^2 = 1` at each step. The Crank-Nicolson
            step is unitary, so this only matters for a non-Hermitian
            Hamiltonian (gain/loss), where the norm is not conserved.
        method : String, default 'crank-nicolson'
            'crank-nicolson' (one
            sparse LU factorization; any Hamiltonian, Hermitian or not) or
            'chebyshev' (the exact :math:`e^{-iH\,dz}` of
            *tbkit.kpm.time_evolution*; Hermitian Hamiltonians only).
        '''
        error_handling.empty_ham(ham)
        error_handling.ndarray(psi_init, 'psi_init', self.lat.sites)
        error_handling.positive_int(steps, 'steps')
        error_handling.positive_real(dz, 'dz')
        error_handling.boolean(norm, 'norm')
        error_handling.prop_method(method)
        self.steps = steps
        self.dz = dz
        if method == 'chebyshev':
            self.prop = kpm.time_evolution(ham, psi_init, self.dz * np.arange(self.steps))
            if norm:
                self.prop /= np.linalg.norm(self.prop, axis=0)
            return
        self.prop = np.empty((self.lat.sites, self.steps), 'c16')
        self.prop[:, 0] = psi_init
        step = _crank_nicolson(ham, self.dz)
        for i in range(1, self.steps):
            self.prop[:, i] = step(self.prop[:, i-1])
            if norm:
                self.prop[:, i] /= np.linalg.norm(self.prop[:, i])

    def get_pumping(
        self, hams: list[sparse.spmatrix], psi_init: NDArray, steps: int, dz: float, norm: bool = True,
    ) -> None:
        r'''
        Get the time evolution with adiabatic pumpings.

        Parameters
        ----------
        hams : List of sparse.csr_matrices
            Tight-Binding Hamiltonians.
        psi_init : np.ndarray
            Initial state.
        steps : Positive integer
            Number of steps.
        dz : Positive number
            Step.
        norm : Boolean, default True
            Renormalize the state to
            :math:`\sum_i|\psi_i|^2 = 1` at each step.

        Notes
        -----
        The *steps* are split into ``len(hams) + 1`` equal stages: evolution
        under ``hams[0]``, then ``len(hams) - 1`` stages each interpolating
        linearly from ``hams[j]`` to ``hams[j+1]``, then evolution under
        ``hams[-1]`` for the remaining steps. With fewer than
        ``len(hams) + 1`` steps the stages are empty: the state evolves under
        ``hams[-1]`` from the first step on.
        '''
        error_handling.get_pump(hams)
        error_handling.ndarray(psi_init, 'psi_init', self.lat.sites)
        error_handling.positive_int(steps, 'steps')
        error_handling.positive_real(dz, 'dz')
        error_handling.boolean(norm, 'norm')
        self.steps = steps
        self.dz = dz
        no = len(hams)
        self.prop = np.empty((self.lat.sites, self.steps), 'c16')
        self.prop[:, 0] = psi_init

        def advance(i, step):
            self.prop[:, i] = step(self.prop[:, i-1])
            if norm:
                self.prop[:, i] /= np.linalg.norm(self.prop[:, i])

        delta = self.steps // (1 + no)
        # before pumping
        step = _crank_nicolson(hams[0], self.dz)
        for i in range(1, delta):
            advance(i, step)
        # pumping: one factorization per interpolated Hamiltonian
        c = np.linspace(0, 1, delta)
        for j in range(0, no-1):
            for i in range(0, delta):
                advance((j+1)*delta+i, _crank_nicolson((1-c[i])*hams[j]+c[i]*hams[j+1], self.dz))
        # after pumping, under hams[-1] (what the last pumping step ends on).
        # Start at 1 at least: with delta == 0 there were no stages, and
        # index no*delta - 1 = -1 would overwrite the initial state.
        step = _crank_nicolson(hams[-1], self.dz)
        for i in range(max(no*delta, 1), self.steps):
            advance(i, step)

    def plt_propagation_1d(
        self, prop_type: str = 'real', fs: float = 20, figsize: tuple[float, float] | None = None,
    ) -> Figure:
        '''
        Plot time evolution for 1D systems.

        Parameters
        ----------
        prop_type : String, default 'real'
            Plot the real part
            ('real'), the imaginary part ('imag'), or the intensity ('norm').
        fs : Positive number, default 20
            Fontsize.
        figsize : Tuple, default None
            Figure size.

        Returns
        -------
        fig : Figure
        '''
        error_handling.empty_ndarray(self.prop, 'get_propagation or get_pumping')
        error_handling.positive_real(fs, 'fs')
        error_handling.prop_type(prop_type)
        error_handling.tuple_2elem(figsize, 'figsize')
        fig, ax = plt.subplots(figsize=figsize)
        plt.ylabel('n', fontsize=fs)
        plt.xlabel('z', fontsize=fs)
        if prop_type == 'real':
            color = self.prop_smooth_1d(self.prop.real)
            max_val = max(np.max(color), -np.min(color))
            ticks = [-max_val, max_val]
            cmap = 'seismic'
        elif prop_type == 'imag':
            color = self.prop_smooth_1d(self.prop.imag)
            max_val = max(np.max(color), -np.min(color))
            ticks = [-max_val, max_val]
            cmap = 'seismic'
        else:
            color = self.prop_smooth_1d(np.abs(self.prop) ** 2)
            ticks = [0., np.max(color[:, -1])]
            cmap = 'hot'
        extent = (-0, self.steps*self.dz, self.lat.sites-.5, -.5)
        interpolation = 'nearest'
        im = plt.imshow(color, cmap=cmap, aspect='auto',
                                  interpolation=interpolation, extent=extent,
                                  vmin=ticks[0], vmax=ticks[-1])
        for label in ax.xaxis.get_majorticklabels():
            label.set_fontsize(fs)
        for label in ax.yaxis.get_majorticklabels():
            label.set_fontsize(fs)
        ax.get_yaxis().set_major_locator(plt.MaxNLocator(integer=True))
        if prop_type == 'norm':
            cbar = fig.colorbar(im, ticks=ticks)
            cbar.ax.set_yticklabels(['0','max'])
        else:
            cbar = fig.colorbar(im, ticks=[ticks[0], 0, ticks[1]])
            cbar.ax.set_yticklabels(['min', '0','max'])
        cbar.ax.tick_params(labelsize=fs)
        return fig

    def prop_smooth_1d(self, prop: NDArray, a: float = 10, no: int = 40) -> NDArray:
        r'''
        Private function. Used in *plt_propagation_1d*.
        Smooth propagation for 1D systems.
        Perform Gaussian interpolation :math:`e^{-a(x-x_i)^2}`,

        Parameters
        ----------
        prop
            Propagation.
        a
            Default value 10. Gaussian Parameter.
        no
            Default value 40. Number of points of each Gaussian.

        Returns
        -------
        smooth
            Smoothed propagation.
        '''
        func = np.exp(- a * np.linspace(-0.5, 0.5, no) ** 2) 
        smooth = np.empty((self.lat.sites * no, self.steps))
        for iz in range(0, self.steps):
            for i in range(self.lat.sites):
                smooth[i*no: (i+1)*no, iz] = prop[i, iz] * func
        return smooth

    def get_animation(
        self, s: float = 300., fs: float = 20., prop_type: str = 'real',
        figsize: tuple[float, float] | None = None,
    ) -> FuncAnimation:
        '''
        Get time evolution animation.

        Parameters
        ----------
        s
            Default value 300. Circle size.
        fs
            Default value 20. Fontsize.
        prop_type : String, default 'real'
            Plot the real part
            ('real'), the imaginary part ('imag'), or the intensity ('norm').
        figsize : Tuple, default None
            Figsize.

        Returns
        -------
        ani
            Animation.
        '''
        error_handling.empty_ndarray(self.prop, 'get_propagation or get_pumping')
        error_handling.positive_real(s, 's')
        error_handling.positive_real(fs, 'fs')
        error_handling.prop_type(prop_type)
        error_handling.tuple_2elem(figsize, 'figsize')
        if os.name == 'posix':
            blit = False
        else:
            blit = True
        if prop_type == 'real':
            color = self.prop.real
            max_val = max(np.max(color), -np.min(color))
            ticks = [-max_val, max_val]
            cmap = 'seismic'
        elif prop_type == 'imag':
            color = self.prop.imag
            max_val = max(np.max(color), -np.min(color))
            ticks = [-max_val, max_val]
            cmap = 'seismic'
        else:
            color = np.abs(self.prop) ** 2
            ticks = [0., np.max(color)]
            cmap = 'Reds'
        fig, ax = plt.subplots(figsize=figsize)
        plt.xlim(np.min(self.lat.coor['x'])-1., np.max(self.lat.coor['x'])+1.)
        plt.ylim(np.min(self.lat.coor['y'])-1., np.max(self.lat.coor['y'])+1.)
        scat = plt.scatter(self.lat.coor['x'], self.lat.coor['y'], c=color[:, 0],
                                   s=s, vmin=ticks[0], vmax=ticks[1],
                                   cmap=plt.get_cmap(cmap))
        frame = plt.gca()
        frame.get_xaxis().set_ticks([])
        frame.get_yaxis().set_ticks([])
        ax.set_aspect('equal')
        if prop_type == 'norm':
            cbar = fig.colorbar(scat, ticks=ticks)
            cbar.ax.set_yticklabels(['0','max'])
        else:
            cbar = fig.colorbar(scat, ticks=[ticks[0], 0, ticks[1]])
            cbar.ax.set_yticklabels(['min', '0','max'])

        def update(i, color, scat):
            scat.set_array(color[:, i])
            return scat,

        ani = FuncAnimation(fig, update, frames=self.steps,
                                                  fargs=(color, scat), blit=blit, repeat=False)
        return ani

    def get_animation_nb(
        self, s: float = 300., fs: float = 20., prop_type: str = 'real',
        figsize: tuple[float, float] | None = None,
    ) -> FuncAnimation:
        '''
        Get time evolution animation for iPython notebooks.

        Parameters
        ----------
        s
            Default value 300. Circle size.
        fs
            Default value 20. Fontsize.
        prop_type : String, default 'real'
            Plot the real part
            ('real'), the imaginary part ('imag'), or the intensity ('norm').
        figsize : Tuple, default None
            Figsize.

        Returns
        -------
        ani
            Animation.
        '''
        error_handling.empty_ndarray(self.prop, 'get_propagation or get_pumping')
        error_handling.positive_real(s, 's')
        error_handling.positive_real(fs, 'fs')
        error_handling.prop_type(prop_type)
        error_handling.tuple_2elem(figsize, 'figsize')
        if prop_type == 'real' or prop_type == 'imag':
            color = self.prop.real if prop_type == 'real' else self.prop.imag
            max_val = max(np.max(color), -np.min(color))
            ticks = [-max_val, max_val]
            cmap = 'seismic'
        else:
            color = np.abs(self.prop) ** 2
            ticks = [0., np.max(color)]
            cmap = 'Reds'
        fig = plt.figure(figsize=figsize)
        ax = plt.axes(xlim=(np.min(self.lat.coor['x']-.5), np.max(self.lat.coor['x']+.5)), 
                             ylim=(np.min(self.lat.coor['y']-.5), np.max(self.lat.coor['y']+.5)))
        ax.set_aspect('equal')
        frame = plt.gca()
        frame.get_xaxis().set_ticks([])
        frame.get_yaxis().set_ticks([])
        scat = plt.scatter(self.lat.coor['x'], self.lat.coor['y'], c=color[:, 0],
                                    s=s, vmin=ticks[0], vmax=ticks[1],
                                    cmap=cmap)
        if prop_type == 'real' or prop_type == 'imag':
            cbar = fig.colorbar(scat, ticks=[ticks[0], 0, ticks[1]])
            cbar.ax.set_yticklabels(['min', '0','max'])
        else:
            cbar = fig.colorbar(scat, ticks=[0, ticks[1]])
            cbar.ax.set_yticklabels(['0','max'])

        def init():
            scat.set_array(color[:, 0])
            return scat,

        def animate(i):
            scat.set_array(color[:, i])    
            return scat,

        return FuncAnimation(fig, animate, init_func=init,
                                   frames=self.steps, interval=120, blit=True)

    def plt_prop_dimer(self, lw: float = 5, fs: float = 20) -> Figure:
        '''
        Plot time evolution for dimers.

        Parameters
        ----------
        lw : Positive number, default 5
            Linewidth.
        fs : Positive number, default 20
            Fontsize.

        Returns
        -------
        fig : Figure
        '''
        error_handling.empty_ndarray(self.prop, 'get_propagation or get_pumping')
        error_handling.positive_real(lw, 'lw')
        error_handling.positive_real(fs, 'fs')
        color = ['b', 'r']
        fig, ax = plt.subplots()
        z = self.dz * np.arange(self.steps)
        for i, c in zip([0, 1], color):
            plt.plot(z, np.abs(self.prop[i, :])**2, c, lw=lw)
        plt.title('Intensity', fontsize=fs)
        plt.xlabel('$z$', fontsize=fs)
        plt.ylabel(r'$|\psi_j|^2$', fontsize=fs)
        plt.xlim(0, z[-1])
        return fig


# Backward-compatible lowercase alias (pre-0.2 API).
propagation = Propagation
