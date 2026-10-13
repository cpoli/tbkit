r"""
Landau-Zener Tunnelling and Zener Breakdown
=================================================

Two levels swept through an avoided crossing,

.. math::

    H(t) = \frac{\alpha t}{2}\sigma_z + \Delta\sigma_x\, ,

end up in the *other* adiabatic level with the probability found in 1932
by Landau, Zener, Stückelberg and Majorana,

.. math::

    P = e^{-2\pi\Delta^2/\alpha}\qquad(\hbar = 1)\, :

a slow sweep follows the adiabatic level, a fast one jumps the gap
:math:`2\Delta`. Zener (1934) applied it to a crystal in a uniform field,
where the Bloch momentum sweeps through the Brillouin zone,
:math:`\dot k = F`. At the zone edge two bands nearly cross, and a
Bloch-oscillating electron (see the Bloch oscillations example) tunnels into
the next band each time it reaches the gap: Zener breakdown. In a binary
chain (hopping :math:`t`, onsite energies :math:`\pm\delta`), the folded
bands :math:`\pm2t\cos k` cross at :math:`k = \pi/2` with relative slope
:math:`\alpha = 4tF` and are coupled by :math:`\delta`, so

.. math::

    P_Z = e^{-\pi\delta^2/(2tF)}\, .

:func:`~tbkit.floquet.evolution_operator` gives the time-ordered evolution
of the two-level sweep, and :class:`~tbkit.propagation.Propagation`
evolves a wavepacket in the tilted binary chain.
"""
import numpy as np
import matplotlib.pyplot as plt

from tbkit.floquet import evolution_operator
from tbkit.lattice import Lattice
from tbkit.system import System
from tbkit.propagation import Propagation

# %%
# Two levels swept through an avoided crossing
# --------------------------------------------------
# The sweep runs from :math:`t = -T/2` to :math:`T/2`, long enough for the
# levels to end far apart. The evolution over the sweep is the one-period
# evolution operator of the drive, with "period" the sweep time.

sz = np.diag([1., -1.])
sx = np.array([[0., 1.], [1., 0.]])
delta = 0.5
sweep = 400.

rates = np.array([1., 2., 4., 8.])
p_lz = []
for alpha in rates:
    u = evolution_operator(lambda t, a=alpha: a * (t - sweep / 2) / 2 * sz + delta * sx, sweep, 40000)
    # starting in the diabatic level |0> (the lower adiabatic level at t = -T/2),
    # staying in it means jumping to the upper adiabatic level
    p_lz.append(abs(u[0, 0]) ** 2)
p_lz = np.array(p_lz)
p_formula = np.exp(-2 * np.pi * delta ** 2 / rates)
print('Landau-Zener probability:', np.round(p_lz, 4), ' formula:', np.round(p_formula, 4))
assert np.allclose(p_lz, p_formula, atol=5e-3)

# %%
# Zener breakdown between two bands
# -------------------------------------
# A broad wavepacket starts at the bottom of the lower band
# (:math:`k = \pi`). After one Bloch period, :math:`T_B = \pi/F` for the
# two-site unit cell, it has crossed the gap at the zone edge once; the
# weight it has left in the upper band is the Zener probability. The
# tilt is centred on the packet, so that the Crank-Nicolson step sees only
# small onsite energies.

t, d, n = 1., 0.3, 600
x = np.arange(n)
x0 = n // 2
lat = Lattice(unit_cell=[{'tag': 'a', 'r0': (0., 0.)}], prim_vec=[(1., 0.)])
lat.get_lattice(n1=n)


def binary_chain(force):
    '''Hopping t, onsite energies +-d, and a uniform force centred on x0.'''
    sys = System(lat)
    sys.set_hopping([{'n': 1, 't': t}])
    sys.set_onsite({'a': 0.})
    sys.set_onsite_def({i: d * (-1) ** i + force * (i - x0) for i in range(n)})
    sys.get_ham()
    return sys


bands = binary_chain(0.)
bands.get_eig(eigenvec=True)
lower, upper = bands.rn[:, bands.en.real < 0], bands.rn[:, bands.en.real > 0]
psi0 = np.exp(-(x - x0) ** 2 / (4 * 20. ** 2)) * np.exp(1j * np.pi * x)
psi0 = lower @ (lower.conj().T @ psi0)
psi0 /= np.linalg.norm(psi0)

dz = 0.02
forces = np.array([0.05, 0.1, 0.2, 0.4, 0.8])
p_zener = []
for force in forces:
    prop = Propagation(lat)
    prop.get_propagation(binary_chain(force).ham, psi0, steps=int(np.pi / force / dz) + 1, dz=dz)
    p_zener.append(np.sum(abs(upper.conj().T @ prop.prop[:, -1]) ** 2))
p_zener = np.array(p_zener)
p_zener_formula = np.exp(-np.pi * d ** 2 / (2 * t * forces))
print('Weight tunnelled into the upper band:', np.round(p_zener, 4),
          ' formula:', np.round(p_zener_formula, 4))
assert np.allclose(p_zener, p_zener_formula, atol=0.01)

# %%
# Each passage through the gap, at half-integer multiples of :math:`T_B`,
# moves weight between the bands. From the second passage on, the two
# paths interfere (Landau-Zener-Stückelberg interference), so a jump can
# go either way.

force = 0.2
prop = Propagation(lat)
prop.get_propagation(binary_chain(force).ham, psi0, steps=int(3 * np.pi / force / dz), dz=dz)
times = dz * np.arange(prop.steps)
upper_weight = np.sum(abs(upper.conj().T @ prop.prop) ** 2, axis=0)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
# the two problems on one axis: P depends only on Delta^2 / alpha
ratio = np.linspace(0.005, 0.5, 100)
axes[0].plot(ratio, np.exp(-2 * np.pi * ratio), 'k', lw=1, label=r'$e^{-2\pi\Delta^2/\alpha}$')
axes[0].plot(delta ** 2 / rates, p_lz, 'ob', label='two levels')
axes[0].plot(d ** 2 / (4 * t * forces), p_zener, 'sr', label=r'binary chain ($\alpha = 4tF$, $\Delta = \delta$)')
axes[0].set_xlabel(r'$\Delta^2/\alpha$')
axes[0].set_ylabel('tunnelling probability')
axes[0].legend()
axes[0].set_title('Landau-Zener formula')
axes[1].plot(times * force / np.pi, upper_weight, 'r')
for passage in (0.5, 1.5, 2.5):
    axes[1].axvline(passage, color='k', lw=0.8, ls=':')
axes[1].set_xlabel(r'$t / T_B$')
axes[1].set_ylabel('weight in the upper band')
axes[1].set_title(r'Zener breakdown, $F = 0.2$: a jump at each passage')
fig.set_layout_engine('tight')
