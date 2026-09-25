"""
Occupation of single-particle states: the Fermi-Dirac distribution and the
Fermi level that holds a given number of electrons, at zero or finite
temperature. Used by :class:`tbkit.system.System` (a finite spectrum) and
:class:`tbkit.kspace.KSpace` (a Brillouin-zone mesh, with weights).

Energies and temperatures are in the same units (:math:`k_B = 1`). Each
state holds one electron: for spin-degenerate levels of a spinless model,
count two electrons per level.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import brentq
from scipy.special import expit

import tbkit.error_handling as error_handling


def fermi_dirac(energies: ArrayLike, mu: float, temperature: float = 0.) -> NDArray[np.float64]:
    r'''
    Get the Fermi-Dirac occupations

    .. math::

        f(E) = \frac{1}{e^{(E-\mu)/T} + 1}\, ,

    the step :math:`\Theta(\mu - E)` at :math:`T = 0` (with :math:`f = 1/2`
    exactly at :math:`E = \mu`).

    :param energies: Array of real energies (for complex ones, the real part is used).
    :param mu: Real number. Chemical potential.
    :param temperature: Positive real number or zero. Default value 0.

    :returns:
        * **f** -- Real ndarray, same shape as *energies*.
    '''
    error_handling.real_number(mu, 'mu')
    error_handling.positive_real_zero(temperature, 'temperature')
    energies = np.asarray(energies).real
    if temperature == 0:
        return np.heaviside(mu - energies, 0.5)
    return expit(-(energies - mu) / temperature)


def fermi_level(
    energies: ArrayLike, n_electrons: float, temperature: float = 0.,
    weights: ArrayLike | None = None,
) -> float:
    r'''
    Get the Fermi level :math:`\mu` at which :math:`\sum_n w_n f(E_n) = N_e`.

    At :math:`T = 0` with a gap above the last occupied level, :math:`\mu` is
    placed midway in that gap (one unit above the top level if every level
    is filled); if the last electrons only partly fill a (degenerate)
    level, :math:`\mu` is that level's energy, where *fermi_dirac* gives
    each state :math:`f = 1/2` -- the electron count is then exact only for
    a half-filled shell. For partly filled degenerate shells, use a small
    temperature.

    :param energies: Array of real energies, any shape (e.g. *System.en*, or
        the bands of *KSpace.mesh_bands*).
    :param n_electrons: Positive real number or zero. Number of electrons
        :math:`N_e` (per unit cell, with the weights of a k-mesh).
    :param temperature: Positive real number or zero. Default value 0.
    :param weights: Array of positive reals, same shape as *energies*.
        Default value None (all 1). Weight of each level (e.g. 1/nk on a k-mesh).

    :returns:
        * **mu** -- Real number.
    '''
    energies = np.asarray(energies).real.ravel()
    error_handling.ndarray_empty(energies, 'energies')
    weights = np.ones_like(energies) if weights is None else np.asarray(weights, dtype='f8').ravel()
    error_handling.weights(weights, len(energies))
    error_handling.positive_real_zero(temperature, 'temperature')
    error_handling.electrons(n_electrons, weights.sum())
    order = np.argsort(energies, kind='stable')
    en, w = energies[order], weights[order]
    if temperature == 0:
        cum = np.cumsum(w)
        tol = 1e-9 * max(1., cum[-1])
        if n_electrons <= tol:
            return float(en[0] - 1.)
        m = int(np.searchsorted(cum, n_electrons - tol))
        if abs(cum[m] - n_electrons) > tol or m == len(en) - 1:
            # a partly filled level, or every level filled
            return float(en[m] if abs(cum[m] - n_electrons) > tol else en[m] + 1.)
        above = en[m + 1:][en[m + 1:] > en[m] + 1e-12]
        if len(above) == 0 or np.any(np.isclose(en[m + 1:], en[m], atol=1e-12)):
            # the next level is degenerate with the last occupied one
            return float(en[m])
        return float((en[m] + above[0]) / 2)
    pad = 50. * temperature
    return float(brentq(lambda mu: np.sum(w * fermi_dirac(en, mu, temperature)) - n_electrons,
                                  en[0] - pad, en[-1] + pad, xtol=1e-14))
