'''
Value functions: hoppings and onsite energies given as callables of the
sites and of named parameters, ``t(site_i, site_j, **params)`` and
``onsite(site, **params)`` (see *System.set_hopping*, *System.set_onsite*
and *KSpace.set_hopping*). The model stores which bonds or sites each one
applies to; ``get_ham(**params)`` calls it once per Hamiltonian, with
arrays of sites (one entry per bond or site).
'''
from __future__ import annotations

import inspect
from typing import Callable

import numpy as np
from numpy.typing import NDArray
import tbkit.error_handling as error_handling


def sites(pos: NDArray[np.float64], tags: NDArray, index: NDArray) -> NDArray:
    '''
    Private function. The sites passed to a value function: a structured
    array with fields 'x', 'y' (and 'z' in 3D space), 'tag' and 'index'.

    :param pos: Real ndarray, shape (n, space_dim). Positions.
    :param tags: String ndarray, shape (n,). Sublattice tags.
    :param index: Integer ndarray, shape (n,). Site indices.
    '''
    names = ('x', 'y', 'z')[:pos.shape[1]]
    rec = np.zeros(len(pos), dtype=[(name, 'f8') for name in names] + [('tag', 'U1'), ('index', 'i8')])
    for k, name in enumerate(names):
        rec[name] = pos[:, k]
    rec['tag'] = tags
    rec['index'] = index
    return rec


def _kwargs(func: Callable, n_args: int, params: dict) -> dict:
    '''
    Private function. The entries of *params* that *func* takes (all of them
    if it has ``**kwargs``), past its first *n_args* positional arguments.
    '''
    args = list(inspect.signature(func).parameters.values())
    if any(p.kind is p.VAR_KEYWORD for p in args):
        return dict(params)
    named = [p for p in args[n_args:] if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)]
    error_handling.value_params([p.name for p in named if p.default is p.empty], params, func)
    return {p.name: params[p.name] for p in named if p.name in params}


def evaluate(func: Callable, args: tuple, params: dict, size: int, spin: bool = False) -> NDArray[np.complex128]:
    '''
    Private function. Call the value function *func* on the site arrays
    *args*, with the entries of *params* it takes.

    :returns:
        * **values** -- Complex ndarray, shape (size,), or (size, 2, 2) if
          *spin* (a number per bond then multiplies the identity).
    '''
    val = np.asarray(func(*args, **_kwargs(func, len(args), params)), dtype='c16')
    error_handling.value_shape(val, size, spin, func)
    if not spin:
        return np.broadcast_to(val, (size,))
    if val.ndim <= 1:
        return np.broadcast_to(val, (size,))[:, None, None] * np.eye(2)
    return np.broadcast_to(val, (size, 2, 2))


def reversed_conj(func: Callable) -> Callable:
    r'''
    Private function. The value function of the reversed bonds: given
    :math:`t(i, j) = H_{ij}`, the function :math:`(i, j) \mapsto H_{ji}^\dagger
    = t(j, i)^\dagger` (conjugate transpose for 2x2 spin blocks).
    '''
    def value(site_i, site_j, **params):
        val = np.asarray(func(site_j, site_i, **_kwargs(func, 2, params)), dtype='c16')
        return val.conj() if val.ndim < 2 else np.swapaxes(val.conj(), -1, -2)
    return value
