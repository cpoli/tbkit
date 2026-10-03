r'''
Continuum discretization: turn a k·p (continuum) Hamiltonian, a polynomial
in the momenta :math:`k_x, k_y, k_z` with constant (possibly symbolic)
matrix coefficients, into a Tight-Binding model on a square or cubic grid
of spacing *a*.

Each momentum is the derivative :math:`k_x = -i\partial_x`, and a power
:math:`k_x^n` is replaced by the half-step finite difference
:math:`D_hf(x) = [f(x+h/2) - f(x-h/2)]/h` applied *n* times, with
:math:`h = a` for an even *n* and :math:`h = 2a` for an odd one, so that
every shift is a whole number of sites (the scheme of Kwant's
``kwant.continuum``). In k-space this replaces

.. math::

    k^n \to \Big(\tfrac{2}{a}\sin\tfrac{ka}{2}\Big)^n \;(n\text{ even}),\qquad
    k^n \to \Big(\tfrac{1}{a}\sin ka\Big)^n \;(n\text{ odd}),

which agrees with :math:`k^n` up to :math:`O(a^2)` and keeps a Hermitian
Hamiltonian Hermitian: :math:`k_x\to-i(\psi_{n+1}-\psi_{n-1})/2a`,
:math:`k_x^2\to-(\psi_{n+1}-2\psi_n+\psi_{n-1})/a^2`. Products of momenta
along different axes multiply their stencils. The coefficients do not
depend on the position, so the order of the factors does not matter.

* *discretize_symbolic* returns the hopping matrices :math:`T(\mathbf{R})`
  as sympy matrices in the grid spacing *a*, to inspect.
* *discretize* builds the **KSpace** model with
  :math:`H(\mathbf{k}) = \sum_\mathbf{R} T(\mathbf{R})e^{i\mathbf{k}\cdot\mathbf{R}}`.
  Its free symbols (*M*, *B*, ...) become parameters of value functions,
  set by ``get_ham(k, M=...)`` or *KSpace.set_params*.
  *bridges.finite_system* then gives the matching real-space **System**.

sympy is an optional dependency: ``pip install tbkit[continuum]``.
'''
from __future__ import annotations

import inspect
import itertools
import string
from math import comb

import numpy as np

import tbkit.error_handling as error_handling
from tbkit.kspace import KSpace
from tbkit.lattice import Lattice

#: Names of the momenta, along x, y and z.
MOMENTA = ('k_x', 'k_y', 'k_z')


def _sympy():
    '''
    Private function. The sympy module, imported on first use.
    '''
    try:
        import sympy
    except ImportError:
        sympy = None
    error_handling.sympy_module(sympy)
    return sympy


def _namespace(sympy) -> dict:
    '''
    Private function. Names available in a string Hamiltonian: the Pauli
    matrices *sigma_0*, *sigma_x*, *sigma_y*, *sigma_z*, *kron* (Kronecker
    product), and plain symbols for the names sympy would otherwise read
    as its own objects (E, N, O, Q, S, beta, gamma, zeta).
    '''
    space = {name: sympy.Symbol(name) for name in ('E', 'N', 'O', 'Q', 'S', 'beta', 'gamma', 'zeta')}
    space.update({'sigma_0': sympy.eye(2),
                       'sigma_x': sympy.Matrix([[0, 1], [1, 0]]),
                       'sigma_y': sympy.Matrix([[0, -sympy.I], [sympy.I, 0]]),
                       'sigma_z': sympy.Matrix([[1, 0], [0, -1]]),
                       'kron': lambda *ms: sympy.kronecker_product(*ms)})
    return space


def _matrix(sympy, hamiltonian):
    '''
    Private function. The Hamiltonian as a square sympy Matrix whose free
    symbols are all real, and the map back to the original symbols.
    '''
    error_handling.continuum_type(hamiltonian, (str, sympy.Basic, sympy.MatrixBase))
    ham = hamiltonian
    if isinstance(hamiltonian, str):
        from sympy.parsing.sympy_parser import parse_expr
        error = None
        try:
            ham = parse_expr(hamiltonian, local_dict=_namespace(sympy))
        except Exception as err:  # parse_expr raises many types
            error = err
        error_handling.continuum_parsed(error, hamiltonian)
        error_handling.continuum_type(ham, (sympy.Basic, sympy.MatrixBase))
    ham = sympy.Matrix(ham) if isinstance(ham, sympy.MatrixBase) else sympy.Matrix([[ham]])
    error_handling.continuum_square(ham.shape)
    real = {s: sympy.Symbol(s.name, real=True) for s in ham.free_symbols}
    return ham.xreplace(real), {r: s for s, r in real.items()}


def _stencil(sympy, n: int) -> dict[int, object]:
    r'''
    Private function. The finite difference of :math:`k^n`, times
    :math:`a^n`: {shift in sites: coefficient}.
    '''
    h = 1 if n % 2 == 0 else 2
    return {(n - 2*m) * h // 2: (-sympy.I / h)**n * comb(n, m) * (-1)**m for m in range(n + 1)}


def _discretize(sympy, hamiltonian, dim: int | None):
    '''
    Private function. The hoppings of *discretize_symbolic*, the dimension,
    the number of orbitals, and whether the Hamiltonian is Hermitian (for
    real parameters).
    '''
    ham, originals = _matrix(sympy, hamiltonian)
    names = {s.name for s in ham.free_symbols}
    error_handling.continuum_symbols(names)
    used = [d for d, k in enumerate(MOMENTA) if k in names]
    if dim is None:
        dim = max(used, default=0) + 1
    error_handling.continuum_dim(dim, used)
    momenta = [sympy.Symbol(k, real=True) for k in MOMENTA[:dim]]
    a = sympy.Symbol('a')
    norb = ham.shape[0]
    hops: dict = {}
    for i, j in itertools.product(range(norb), repeat=2):
        entry = sympy.expand(ham[i, j])
        error_handling.continuum_polynomial((i, j), entry.is_polynomial(*momenta))
        for powers, coeff in sympy.Poly(entry, *momenta).terms():
            stencils = [_stencil(sympy, n).items() for n in powers]
            for steps in itertools.product(*stencils):
                R = tuple(shift for shift, _ in steps)
                c = coeff * sympy.Mul(*(c for _, c in steps)) / a**sum(powers)
                hops.setdefault(R, sympy.zeros(norb))[i, j] += c
    hops = {R: m.applyfunc(sympy.expand).xreplace(originals) for R, m in sorted(hops.items())}
    hops = {R: m for R, m in hops.items() if not m.is_zero_matrix}
    hermitian = (ham - ham.H).applyfunc(sympy.simplify).is_zero_matrix
    return hops, dim, norb, bool(hermitian)


def discretize_symbolic(hamiltonian, dim: int | None = None) -> dict:
    r'''
    Discretize a continuum Hamiltonian on a square or cubic grid, keeping
    everything symbolic: the hopping matrices
    :math:`T(\mathbf{R}) = \langle\mathbf{0}|H|\mathbf{R}\rangle` between
    a grid site and the one :math:`\mathbf{R}` sites further, in terms of
    the grid spacing *a* and the parameters of the Hamiltonian (see the
    module docstring for the scheme).

    :param hamiltonian: sympy expression or Matrix, or a string, in the
        momenta ``k_x``, ``k_y``, ``k_z``: a polynomial in them with
        constant coefficients. Any other symbol is a parameter, taken real
        (the symbols ``x``, ``y``, ``z`` and ``a`` are not allowed). A
        string may use the Pauli matrices ``sigma_0``, ``sigma_x``,
        ``sigma_y``, ``sigma_z`` and ``kron`` (Kronecker product); ``I``
        is the imaginary unit.
    :param dim: 1, 2 or 3. Default value None: the last axis whose momentum
        appears (1 if none).

    :returns:
        * **hoppings** -- Dictionary. key: tuple of *dim* integers
          :math:`\mathbf{R}`, val: sympy Matrix :math:`T(\mathbf{R})`, in
          the symbol ``a`` and those of *hamiltonian*.

    Example usage::

        discretize_symbolic('k_x**2')
        # {(-1,): Matrix([[-1/a**2]]), (0,): Matrix([[2/a**2]]), (1,): Matrix([[-1/a**2]])}
    '''
    return _discretize(_sympy(), hamiltonian, dim)[0]


def _value(sympy, expr):
    '''
    Private function. A value function returning the matrix element *expr*,
    whose free symbols are its keyword parameters.
    '''
    symbols = sorted(expr.free_symbols, key=lambda s: s.name)
    names = [s.name for s in symbols]
    func = sympy.lambdify(symbols, expr, modules='numpy')

    def value(site_i, site_j, **params):
        return complex(func(*(params[n] for n in names)))
    kind = inspect.Parameter
    value.__signature__ = inspect.Signature(
        [kind('site_i', kind.POSITIONAL_OR_KEYWORD), kind('site_j', kind.POSITIONAL_OR_KEYWORD)]
        + [kind(n, kind.KEYWORD_ONLY) for n in names])
    value.__name__ = str(expr)
    return value


def discretize(hamiltonian, a: float = 1., dim: int | None = None,
                      tags: str | list[str] | None = None) -> KSpace:
    r'''
    Discretize a continuum (k·p) Hamiltonian into a Tight-Binding **KSpace**
    model on a square (2D) or cubic (3D) grid, or a chain (1D), of spacing
    *a*. The grid has one site per unit cell carrying the *norb* orbitals of
    the Hamiltonian's matrix, all at the origin of the cell, so that
    :math:`H(\mathbf{k}) = \sum_\mathbf{R} T(\mathbf{R})e^{i\mathbf{k}\cdot\mathbf{R}}`
    with the hoppings of *discretize_symbolic*. It agrees with the
    continuum Hamiltonian up to :math:`O(a^2)` near
    :math:`\mathbf{k}=\mathbf{0}`.

    A matrix element with no free symbol becomes a number. One with
    parameters becomes a value function (see *KSpace.set_hopping*): pass
    the parameters to ``get_ham(k, M=...)``, or to *KSpace.set_params* for
    the other methods. A Hamiltonian that is Hermitian for real parameters
    is stored with ``hermitian=True`` (one representative per bond);
    otherwise every matrix element is stored, with ``hermitian=False``.
    Constant onsite terms are in *ks.onsite*.

    :param hamiltonian: sympy expression or Matrix, or a string (see
        *discretize_symbolic*).
    :param a: Real number, at least :math:`\sqrt{0.1}\approx 0.316` (the
        shortest primitive vector of a **Lattice**). Default value 1. Grid
        spacing: to refine the grid, measure lengths in a smaller unit.
    :param dim: 1, 2 or 3. Default value None: the last axis whose momentum
        appears (1 if none).
    :param tags: String or list of one-character strings. Default value
        None: 'a', 'b', 'c', ... Tag of each orbital.

    :returns:
        * **ks** -- **KSpace** instance (spinless, *norb* orbitals). Its
          lattice has *prim_vec* :math:`a\hat{\mathbf{x}}, a\hat{\mathbf{y}},
          (a\hat{\mathbf{z}})`.

    Example usage::

        # the BHZ model of HgTe quantum wells, one spin block
        bhz = discretize('A*(k_x*sigma_x - k_y*sigma_y)'
                                '+ (M - B*(k_x**2 + k_y**2))*sigma_z', a=0.5)
        bhz.set_params(A=1., B=1., M=1.)
        bhz.chern_number([0])
        sys = finite_system(bhz, (40, 40))   # a real-space flake
    '''
    error_handling.positive_real(a, 'a')
    error_handling.continuum_spacing(a)
    sympy = _sympy()
    hops, dim, norb, hermitian = _discretize(sympy, hamiltonian, dim)
    error_handling.continuum_tags(tags, norb)
    tags = list(tags) if tags is not None else list(string.ascii_letters[:norb])
    space_dim = max(dim, 2)
    origin = (0.,) * space_dim
    prim_vec = [tuple(float(a) * (c == d) for c in range(space_dim)) for d in range(dim)]
    ks = KSpace(Lattice(unit_cell=[{'tag': t, 'r0': origin} for t in tags], prim_vec=prim_vec))
    a_sym = sympy.Symbol('a')
    list_hop = []
    for R, mat in hops.items():
        mat = mat.subs(a_sym, a)
        for i, j in itertools.product(range(norb), repeat=2):
            expr = mat[i, j]
            if expr == 0:
                continue
            nonzero = [n for n in R if n != 0]
            onsite = not nonzero and i == j
            if hermitian and not onsite and not ((nonzero and nonzero[0] > 0) or (not nonzero and i < j)):
                continue  # the conjugate of a kept representative
            t = _value(sympy, expr) if expr.free_symbols else complex(expr)
            if not onsite:
                list_hop.append({'i': i, 'j': j, 'R': R, 't': t})
            elif callable(t):
                # a parametrized onsite energy: a value function on the
                # bond i -> i, R = 0, which set_hopping does not take
                ks._hop_values.append((np.array([i]), np.array([i]), np.zeros((1, ks.space_dim)), t, False))
            else:
                ks.onsite[i] = t
    if list_hop:
        ks.set_hopping(list_hop, hermitian=hermitian)
    if not hermitian:
        ks._nonreciprocal = True
    return ks
