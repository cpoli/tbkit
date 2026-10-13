r"""
Discrete Optics in Waveguide Arrays
=========================================

Light guided by parallel waveguides leaks from one to the next through
their evanescent tails. In coupled-mode theory the field amplitudes obey
a Tight-Binding Schrödinger equation in which the propagation distance
:math:`z` plays the role of time,

.. math::

    i\frac{d\psi_n}{dz} = c\,(\psi_{n+1} + \psi_{n-1})\, ,

with the coupling constant :math:`c` as the hopping. Two waveguides form
a *directional coupler* (Jones, 1965; Somekh et al., 1973): the power
swaps between them as :math:`\cos^2(cz)`. Light launched into one guide
of an array undergoes *discrete diffraction* (Jones, 1965), with the
intensity :math:`|J_n(2cz)|^2` concentrated in two outer lobes instead
of the single Gaussian spot of a continuous medium. Its rate depends on
the phase difference :math:`k` between neighbouring guides,
:math:`\epsilon''(k) = -2c\cos k`, and vanishes at :math:`k = \pi/2`
(*diffraction management*, Eisenberg et al., 2000). Waveguide arrays
turned into a laboratory for lattice physics with discrete solitons
(Christodoulides and Joseph, 1988; Eisenberg et al., 1998) and optical
Bloch oscillations (Pertsch et al.; Morandotti et al., 1999), reviewed by
Christodoulides, Lederer and Silberberg (2003).

:class:`~tbkit.propagation.Propagation` integrates this equation along
:math:`z` (*get_propagation*), and *plt_prop_dimer* draws the coupler.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jv

from tbkit.lattice import Lattice
from tbkit.system import System
from tbkit.propagation import Propagation

c = 1.
dz = 0.01


def array(n):
    '''n identical waveguides, coupling c between neighbours.'''
    lat = Lattice(unit_cell=[{'tag': 'a', 'r0': (0., 0.)}], prim_vec=[(1., 0.)])
    lat.get_lattice(n1=n)
    sys = System(lat)
    sys.set_hopping([{'n': 1, 't': c}])
    sys.set_onsite({'a': 0.})
    sys.get_ham()
    return sys


# %%
# The directional coupler
# ---------------------------
# All the light returns to the first guide after the beat length
# :math:`\pi/c`, having crossed to the second at :math:`\pi/2c`.

coupler = array(2)
prop_coupler = Propagation(coupler.lat)
prop_coupler.get_propagation(coupler.ham, np.array([1., 0.], dtype='c16'), steps=700, dz=dz)
z = dz * np.arange(700)
assert np.allclose(np.abs(prop_coupler.prop[0]) ** 2, np.cos(c * z) ** 2, atol=1e-4)
print('Coupler: |psi_1|^2 = cos^2(cz) to 1e-4; full transfer at z = pi/2c = {:.4f}.'.format(np.pi / (2 * c)))
fig_coupler = prop_coupler.plt_prop_dimer(lw=3, fs=14)
fig_coupler.axes[0].set_title(r'directional coupler: $|\psi_1|^2 = \cos^2(cz)$', fontsize=14)

# %%
# Discrete diffraction from a single guide
# ----------------------------------------------

n = 161
centre = n // 2
sites = np.arange(n) - centre
guides = array(n)
psi0 = np.zeros(n, dtype='c16')
psi0[centre] = 1.
steps = 1501
prop = Propagation(guides.lat)
prop.get_propagation(guides.ham, psi0, steps=steps, dz=dz)
z = dz * np.arange(steps)
intensity = np.abs(prop.prop) ** 2
bessel = jv(sites[:, None], 2 * c * z[None]) ** 2
print('Largest deviation from |J_n(2cz)|^2: {:.1e}'.format(np.max(np.abs(intensity - bessel))))
assert np.allclose(intensity, bessel, atol=1e-4)
# the width grows linearly (ballistically): <n^2> = 2 c^2 z^2
second_moment = np.sum(sites[:, None] ** 2 * intensity, axis=0)
assert np.allclose(second_moment, 2 * c ** 2 * z ** 2, rtol=1e-3, atol=1e-6)

# %%
# Diffraction management
# --------------------------
# A broad beam tilted by a phase :math:`k` per guide spreads at a rate set
# by the band curvature: fastest at :math:`k = 0`, not at all, to second
# order, at :math:`k = \pi/2`, where it drifts sideways at the largest
# group velocity, :math:`2c`, without broadening.


def beam_width(k, z_end=20.):
    psi = (np.exp(-sites ** 2 / (4 * 2.5 ** 2)) * np.exp(1j * k * sites)).astype('c16')
    psi /= np.linalg.norm(psi)
    beam = Propagation(guides.lat)
    beam.get_propagation(guides.ham, psi, steps=int(z_end / dz) + 1, dz=dz)
    weight = np.abs(beam.prop) ** 2
    mean = sites @ weight
    return np.sqrt(np.sum(sites[:, None] ** 2 * weight, axis=0) - mean ** 2), beam


width_0, _ = beam_width(0.)
width_half, beam_half = beam_width(np.pi / 2)
print('Beam width after z = 20: {:.2f} at k = 0, {:.2f} at k = pi/2 (initially {:.2f}).'
          .format(width_0[-1], width_half[-1], width_0[0]))
assert width_0[-1] > 3 * width_0[0] and width_half[-1] < 1.2 * width_half[0]

fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
window = slice(centre - 30, centre + 31)
axes[0].imshow(intensity[window], aspect='auto', origin='lower', cmap='magma',
                   extent=[0, z[-1], sites[window][0], sites[window][-1]])
axes[0].set_xlabel('$z$')
axes[0].set_ylabel('waveguide $n$')
axes[0].set_title('discrete diffraction from one guide')
axes[1].plot(sites, intensity[:, -1], 'o', ms=4, label='tbkit')
axes[1].plot(sites, bessel[:, -1], 'k', lw=1, label=r'$|J_n(2cz)|^2$')
axes[1].set_xlim(-30, 30)
axes[1].set_xlabel('waveguide $n$')
axes[1].set_title(r'intensity at $z = {:.0f}$'.format(z[-1]))
axes[1].legend(fontsize=8)
axes[2].plot(dz * np.arange(len(width_0)), width_0, label='$k = 0$')
axes[2].plot(dz * np.arange(len(width_half)), width_half, label=r'$k = \pi/2$')
axes[2].set_xlabel('$z$')
axes[2].set_ylabel('beam width (guides)')
axes[2].set_title('diffraction management')
axes[2].legend(fontsize=8)
fig.set_layout_engine('tight')
