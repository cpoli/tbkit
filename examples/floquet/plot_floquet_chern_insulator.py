r"""
Floquet Engineering: A Chern Insulator From Circularly Polarized Light
===========================================================================

T. Oka and H. Aoki (2009) predicted that graphene illuminated by
circularly polarized light becomes a Hall insulator; Kitagawa, Oka,
Brataas, Fu and Demler (2011) showed it is a Floquet realization of
Haldane's model. A time-periodic drive :math:`H(t+T) = H(t)` acts,
stroboscopically, as the static Floquet Hamiltonian :math:`H_F`,
:math:`e^{-iH_FT} = \mathcal{T}e^{-i\int_0^TH(t)dt}`. Light enters the hoppings
through the Peierls phase of its vector potential,
:math:`t_{ij} \to t_{ij}e^{i\mathbf{A}(t)\cdot\mathbf{d}_{ij}}` with
:math:`\mathbf{A}(t) = A_0(\cos\omega t, \sin\omega t)`. At high frequency,

.. math::

    H_F \approx H_0 + \frac{1}{\omega}\sum_{m\ge1}\frac{[H_{-m}, H_m]}{m}\, ,

and the commutator is a complex second-neighbour hopping -- Haldane's --
which opens gaps of the same sign at both Dirac points: a Chern
insulator made by light, observed in graphene by McIver et al. (2020).

:class:`~tbkit.floquet.FloquetKSpace` makes the driven Bloch model;
its bands and Chern numbers are those of :math:`H_F`.
"""
import numpy as np
import matplotlib.pyplot as plt

import tbkit.lattices as lattices
from tbkit.floquet import FloquetKSpace, harmonics, quasienergies, sambe_hamiltonian
from tbkit.kspace import KSpace, reciprocal_vectors

graphene = KSpace(lattices.honeycomb())
graphene.set_hopping([{'i': 0, 'j': 1, 'R': R, 't': 1.} for R in [(0, 0), (-1, 0), (0, -1)]])
omega, a0 = 12., 0.6
period = 2 * np.pi / omega


def light(helicity):
    return lambda t: (a0 * np.cos(omega * t), helicity * a0 * np.sin(omega * t))


driven = FloquetKSpace(graphene, light(+1), period, n_steps=60)
b1, b2 = (np.array(v) for v in reciprocal_vectors(graphene.lat.prim_vec))
K = (b1 - b2) / 3

# %%
# Two routes to the quasienergies
# -----------------------------------
# The time-ordered evolution over a period, and the static Sambe-space
# problem of the drive's harmonics, agree.

ham_t = lambda t: graphene.get_ham_peierls(K, light(+1)(t))
eps = quasienergies(ham_t, period, 400)
sambe = np.linalg.eigvalsh(sambe_hamiltonian(harmonics(ham_t, period, 4), omega, 8))
assert np.allclose(eps, np.sort(sambe[np.abs(sambe) < omega / 2]), atol=1e-5)

# %%
# Light opens the Dirac points: a Chern insulator
# ------------------------------------------------------

gap_k = np.diff(np.linalg.eigvalsh(driven.get_ham(K)))[0]
gap_kp = np.diff(np.linalg.eigvalsh(driven.get_ham(-K)))[0]
chern = driven.chern_number(0, nk=16)
chern_reversed = FloquetKSpace(graphene, light(-1), period, n_steps=60).chern_number(0, nk=16)
print("Gaps at K and K': {:.4f}, {:.4f}; Chern number {:.3f}, {:.3f} for the opposite helicity."
          .format(gap_k, gap_kp, chern, chern_reversed))
assert gap_k > 0.05 and np.isclose(gap_k, gap_kp)
assert np.isclose(abs(chern), 1., atol=1e-6) and np.isclose(chern_reversed, -chern, atol=1e-6)
# linear light keeps time reversal: no Chern number, Dirac points only shifted
linear = FloquetKSpace(graphene, lambda t: (a0 * np.cos(omega * t), 0.), period, n_steps=60)
assert np.min(np.diff(linear.mesh_bands(30), axis=1)) < 0.2

# %%
# The gap grows as the intensity, and falls as 1/omega
# ----------------------------------------------------------

amps = np.array([0.2, 0.4, 0.6])
gaps = [np.diff(np.linalg.eigvalsh(FloquetKSpace(graphene, lambda t, a=a: (a*np.cos(omega*t), a*np.sin(omega*t)),
                                                                                  period, 60).get_ham(K)))[0] for a in amps]
ratios = np.array(gaps) / amps**2
print('gap / A0^2:', np.round(ratios, 4))
assert np.allclose(ratios, ratios[0], rtol=0.1)


def gap_at(w, a=0.4):
    drive = FloquetKSpace(graphene, lambda t: (a*np.cos(w*t), a*np.sin(w*t)), 2*np.pi/w, 60)
    return np.diff(np.linalg.eigvalsh(drive.get_ham(K)))[0]


ratio = gap_at(12.) / gap_at(24.)
print('gap(omega = 12) / gap(omega = 24) = {:.3f}'.format(ratio))
assert abs(ratio - 2.) < 0.2

driven.k_path([np.zeros(2), K, b1 / 2, np.zeros(2)], nk=60)
fig = driven.plot_bands(node_labels=[r'$\Gamma$', 'K', 'M', r'$\Gamma$'])
fig.axes[0].set_title('Graphene in circularly polarized light: Floquet bands')
