r"""
Löwdin Orthogonalization and Non-Orthogonal Tight Binding
================================================================

Atomic orbitals on neighbouring atoms overlap:
:math:`s = \langle\phi_i|\phi_j\rangle \neq 0`. In such a basis the
Schrödinger equation is the generalized eigenproblem
:math:`H\mathbf{c} = ES\mathbf{c}`, with :math:`S` the overlap matrix.
P.-O. Löwdin (1950) removed :math:`S` with the symmetric
orthogonalization :math:`|\psi\rangle = S^{-1/2}|\phi\rangle`. Among all
orthonormal sets, it is the one closest to the original orbitals, and it
treats every orbital alike, so the new basis keeps the symmetry of the
crystal. In that basis the problem is an ordinary eigenproblem for
:math:`S^{-1/2}HS^{-1/2}`, with the same bands. The cost is range: the
orthogonalized orbitals have tails on the neighbours, so the hoppings
between them reach further than those of :math:`H`. An orthogonal
tight-binding model with a few hoppings is a truncation of this exact
one.

:meth:`tbkit.kspace.KSpace.set_overlap` keeps the overlap and solves the
generalized problem. This example uses a chain with nearest-neighbour
hopping :math:`t` and overlap :math:`s`, whose band is

.. math::

   E(k) = \frac{\epsilon + 2t\cos ka}{1 + 2s\cos ka}\, ,

and checks the generalized solution against it, the Löwdin form, the
exponential decay of the orthogonalized hoppings and the closest-orbitals
property on a ring.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import sqrtm

import tbkit.lattices as lattices
from tbkit.kspace import KSpace


eps, t, s = 0., -1., 0.2      # onsite energy, hopping, overlap (|s| < 1/2)
k = np.linspace(-np.pi, np.pi, 401)
exact = (eps + 2 * t * np.cos(k)) / (1 + 2 * s * np.cos(k))

# %%
# The generalized eigenproblem
# --------------------------------
# ``set_overlap`` takes the overlaps in the format of ``set_hopping``.
# The band is no longer symmetric: the overlap pushes the antibonding
# states up more than it pulls the bonding ones down. The orthogonal model
# with the same :math:`t` (overlap ignored) misses it by up to
# :math:`1.3|t|` at the top of the band.

chain = KSpace(lattices.chain())
chain.set_hopping([{'i': 0, 'j': 0, 'R': (1,), 't': t}])
chain.set_overlap([{'i': 0, 'j': 0, 'R': (1,), 't': s}])
bands = chain.get_bands(k[:, None])[:, 0]
print('Generalized eigenproblem vs closed form: max error {:.1e}'.format(np.abs(bands - exact).max()))
assert np.allclose(bands, exact, atol=1e-12)
naive = eps + 2 * t * np.cos(k)
print('Overlap ignored: max error {:.3f}'.format(np.abs(naive - exact).max()))
assert np.abs(naive - exact).max() > 1.3

# %%
# The Löwdin form :math:`S^{-1/2}HS^{-1/2}` has the same spectrum at every
# k (here a two-orbital chain, so that the matrices are not numbers).

two = KSpace(lattices.honeycomb())
two.set_hopping([{'n': 1, 't': t}])
two.set_overlap([{'n': 1, 't': 0.12}])
rng = np.random.default_rng(1)
for kpt in rng.uniform(-np.pi, np.pi, size=(20, 2)):
    h, sk = two.get_ham(kpt), two.get_overlap(kpt)
    s_inv_half = np.linalg.inv(sqrtm(sk))
    lowdin = s_inv_half @ h @ s_inv_half
    assert np.allclose(lowdin, lowdin.conj().T)
    assert np.allclose(np.linalg.eigvalsh(lowdin), two.get_bands([kpt])[0])
print('S^-1/2 H S^-1/2 has the generalized eigenvalues at 20 random k-points of overlapping graphene.')

# %%
# The hoppings between Löwdin orbitals
# ----------------------------------------
# In the orthogonalized basis the band :math:`E(k)` is an ordinary
# Fourier series, :math:`E(k) = \sum_R t_L(R)e^{ikR}`. Writing
# :math:`E = t/s + (\epsilon - t/s)/(1 + 2s\cos k)` and expanding the
# second term gives
#
# .. math::
#
#    t_L(R) \propto (-x)^{|R|}\quad (R \neq 0)\, ,\qquad
#    x = \frac{1 - \sqrt{1 - 4s^2}}{2s}\, :
#
# every further neighbour is coupled, with a hopping smaller by a factor
# :math:`x` (0.209 at :math:`s = 0.2`). Even the nearest-neighbour hopping
# and the onsite energy change.

n_fft = 256
k_fft = 2 * np.pi * np.arange(n_fft) / n_fft
t_lowdin = np.fft.fft((eps + 2 * t * np.cos(k_fft)) / (1 + 2 * s * np.cos(k_fft))).real / n_fft
x = (1 - np.sqrt(1 - 4 * s ** 2)) / (2 * s)
print('Lowdin onsite {:+.4f}, hoppings t_L(1..5) = {}'.format(
    t_lowdin[0], np.array2string(t_lowdin[1:6], precision=4)))
print('ratios t_L(R+1)/t_L(R) = {}, -x = {:.4f}'.format(
    np.array2string(t_lowdin[2:7] / t_lowdin[1:6], precision=4), -x))
assert np.allclose(t_lowdin[2:10] / t_lowdin[1:9], -x)

# %%
# An orthogonal model that keeps the hoppings :math:`t_L(R)` up to range
# :math:`R_{max}` is a truncation of this series. Its error falls by
# :math:`x` with each neighbour added.

ranges = np.arange(1, 8)
errors = []
for r_max in ranges:
    model = KSpace(lattices.chain())
    model.set_onsite({'a': t_lowdin[0]})
    model.set_hopping([{'i': 0, 'j': 0, 'R': (r,), 't': t_lowdin[r]} for r in range(1, r_max + 1)])
    errors.append(np.abs(model.get_bands(k[:, None])[:, 0] - exact).max())
errors = np.array(errors)
print('max band error vs range:', np.array2string(errors, precision=2))
assert np.allclose(errors[1:] / errors[:-1], x, rtol=1e-3)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
ax1.plot(k, exact, lw=2.5, label=r'$H c = E S c$ (overlap $s=0.2$)')
ax1.plot(k, naive, '--', lw=2, label='overlap ignored')
for r_max in (1, 2):
    model = KSpace(lattices.chain())
    model.set_onsite({'a': t_lowdin[0]})
    model.set_hopping([{'i': 0, 'j': 0, 'R': (r,), 't': t_lowdin[r]} for r in range(1, r_max + 1)])
    ax1.plot(k, model.get_bands(k[:, None])[:, 0], ':', lw=2,
             label=r'Löwdin, $R_{{max}} = {}$'.format(r_max))
ax1.set_xlabel('$ka$')
ax1.set_ylabel('$E/|t|$')
ax1.set_xlim(-np.pi, np.pi)
ax1.legend(fontsize=9)
ax1.set_title('Non-orthogonal chain')
ax2.semilogy(np.arange(1, 9), np.abs(t_lowdin[1:9]), 'o-', label=r'$|t_L(R)|$')
ax2.semilogy(ranges, errors, 's-', label=r'band error at range $R_{max}$')
ax2.semilogy(np.arange(1, 9), np.abs(t_lowdin[1]) * x ** np.arange(8), 'k--', lw=1,
             label=r'$\propto x^{R}$')
ax2.set_xlabel('$R$ (neighbours)')
ax2.legend()
ax2.set_title('Löwdin orbitals: hoppings of every range')
fig.tight_layout()

# %%
# The closest orthonormal orbitals
# ------------------------------------
# On a ring of 12 sites, an orthonormal set
# :math:`|\psi_i\rangle = \sum_j|\phi_j\rangle C_{ji}` satisfies
# :math:`C^\dagger SC = 1`, and its distance to the atomic orbitals is
# :math:`\sum_i\||\psi_i\rangle - |\phi_i\rangle\|^2 =
# \mathrm{Tr}[(C-1)^\dagger S(C-1)]`. Löwdin's :math:`C = S^{-1/2}` moves
# every orbital by the same amount, the least possible. Gram-Schmidt
# (:math:`C = L^{-\dagger}`, with the Cholesky factor :math:`S = LL^\dagger`)
# leaves the first orbital alone and deforms the last one most: it breaks
# the translation symmetry of the ring and moves the orbitals twice as
# far in total.

n_ring = 12
shift = np.roll(np.eye(n_ring), 1, axis=1)
s_ring = np.eye(n_ring) + s * (shift + shift.T)
c_lowdin = np.real(np.linalg.inv(sqrtm(s_ring)))
c_gram = np.linalg.inv(np.linalg.cholesky(s_ring).T)
moves = {}
for name, c in (('Lowdin', c_lowdin), ('Gram-Schmidt', c_gram)):
    assert np.allclose(c.T @ s_ring @ c, np.eye(n_ring))
    moves[name] = np.diag((c - np.eye(n_ring)).T @ s_ring @ (c - np.eye(n_ring)))
    print('{:<13} total {:.4f}, per orbital {}'.format(
        name, moves[name].sum(), np.array2string(moves[name][[0, 1, 6, -1]], precision=4)))
assert np.allclose(moves['Lowdin'], moves['Lowdin'][0])          # all orbitals alike
assert moves['Gram-Schmidt'].sum() > 1.9 * moves['Lowdin'].sum()
assert np.isclose(moves['Gram-Schmidt'][0], 0.)                   # the first is untouched

fig_ring, ax = plt.subplots(figsize=(6.5, 4))
sites = np.arange(n_ring)
ax.bar(sites - 0.2, moves['Lowdin'], 0.4, label='Löwdin $S^{-1/2}$')
ax.bar(sites + 0.2, moves['Gram-Schmidt'], 0.4, label='Gram-Schmidt')
ax.set_xlabel('orbital $i$ on the ring')
ax.set_ylabel(r'$\||\psi_i\rangle - |\phi_i\rangle\|^2$')
ax.set_title('How far orthogonalization moves each orbital')
ax.legend()
fig_ring.tight_layout()
