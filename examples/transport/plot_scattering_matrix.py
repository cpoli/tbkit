r"""
Mode Matching: the Scattering Matrix, Lead Modes and Scattering States
=========================================================================

The Landauer conductance is a sum of transmission probabilities, but the
object behind it is the scattering matrix :math:`S`: the amplitude
:math:`S_{ab}` of every outgoing mode :math:`a` for every incoming mode
:math:`b` of the leads. Fisher and Lee (1981) related it to the Green's
function, and Ando (1991) showed how to compute it on a lattice by *mode
matching*. The modes of a lead, :math:`\psi_n = \lambda^n\phi` in its cell
:math:`n`, solve the generalized eigenproblem of the transfer matrix

.. math::

    (E - h_0 - \lambda v - \lambda^{-1}v^\dagger)\,\phi = 0\, ,

whose solutions with :math:`|\lambda| = 1` propagate and the others decay.
In the device, the wave function is the incoming mode plus a
combination of outgoing and decaying ones, and the Schrodinger equation
fixes its coefficients. Groth et al. (2014) wrote this as one sparse
linear system, the method of Kwant.

:meth:`~tbkit.transport.Transport.smatrix` does the same with exact lead
modes (:func:`~tbkit.transport.lead_modes`, with no broadening
:math:`\eta`) and one sparse LU factorization per energy.
:meth:`~tbkit.transport.Transport.wave_function` returns the scattering
states and :meth:`~tbkit.transport.Transport.ldos` the density of states
they carry. Here they are applied to a strip with an antidot.
"""
import numpy as np
import matplotlib.pyplot as plt

import tbkit.lattices as lattices
from tbkit.kspace import ribbon
from tbkit.system import System
from tbkit.transport import Transport, lead_from_kspace, lead_modes


t = -1.
width, length = 16, 30
e_fermi = -2.6
square = [{'i': 0, 'j': 0, 'R': (1, 0), 't': t}, {'i': 0, 'j': 0, 'R': (0, 1), 't': t}]

lat = lattices.square()
lat.get_lattice(length, width)
x, y = lat.coor['x'], lat.coor['y']
left = list(np.flatnonzero(np.isclose(x, 0.)))
right = list(np.flatnonzero(np.isclose(x, length - 1.)))

strip = ribbon(lattices.square(), square, width=width, direction=1)
h_left, v_left = lead_from_kspace(strip, -1)
h_right, v_right = lead_from_kspace(strip, 1)

sys = System(lat)
sys.set_hopping([{'n': 1, 't': t}])
sys.set_onsite({'a': 0.})
# a smooth antidot off the axis of the strip
sys.onsite[:] = 3. * np.exp(-((x - length / 2) ** 2 + (y - 0.6 * width) ** 2) / 8.)
sys.get_ham()
tr = Transport(sys.ham)
tr.add_lead(h_left, v_left, t * np.eye(width), left)
tr.add_lead(h_right, v_right, t * np.eye(width), right)

# %%
# The modes of a lead
# ---------------------
# Subband :math:`n` of the strip has the band :math:`E = 2t\cos k +
# 2t\cos(n\pi/(W+1))`. Each open subband gives one incoming mode
# (velocity towards the device) and one outgoing mode, each normalized to
# carry unit current.

modes = lead_modes(h_right, v_right, e_fermi)
n_open = sum(abs(e_fermi - 2 * t * np.cos(n * np.pi / (width + 1))) < 2 * abs(t)
             for n in range(1, width + 1))
assert len(modes.momenta) == 2 * n_open
assert np.sum(modes.velocities < 0) == n_open
subband = e_fermi - 2 * t * np.cos(modes.momenta)
allowed = 2 * t * np.cos(np.arange(1, width + 1) * np.pi / (width + 1))
assert np.all(np.min(np.abs(subband[:, None] - allowed[None]), axis=1) < 1e-9)
print('{} open subbands at E = {}: {} incoming and {} outgoing modes.'.format(
    n_open, e_fermi, n_open, n_open))

# %%
# The scattering matrix
# -----------------------
# The S-matrix is unitary, so every incoming electron is either
# transmitted or reflected. The sum of its transmission block is the
# Caroli transmission, which the broadening :math:`\eta` of the Green's
# function approximates.

s = tr.smatrix(e_fermi)
assert np.allclose(s.data.conj().T @ s.data, np.eye(2 * n_open), atol=1e-12)
t_modes = np.abs(s.submatrix(1, 0)) ** 2
transmission = t_modes.sum()
assert abs(transmission + s.transmission(0, 0) - n_open) < 1e-12
assert abs(transmission - tr.transmission([e_fermi], eta=1e-9)[0]) < 1e-6
print('T = {:.6f} of {} open modes; S unitary to 1e-12.'.format(transmission, n_open))

# %%
# A scattering state
# --------------------
# The scattering state of each incoming mode carries its own current,
# the same through every cross-section of the strip: the transmission
# :math:`\sum_a|t_{ab}|^2` of that mode.

psi = tr.wave_function(e_fermi, 0)
ham = sys.ham.tocoo()
for b in range(n_open):
    bond = -2 * np.imag(np.conj(psi[b, ham.row]) * ham.data * psi[b, ham.col])
    for cut in (3, length // 2, length - 4):
        crossing = np.isclose(x[ham.row], cut) & np.isclose(x[ham.col], cut + 1)
        assert abs(bond[crossing].sum() - t_modes[:, b].sum()) < 1e-10
print('Each scattering state carries its transmission through every cross-section.')

# Without bound states in the device, the scattering states hold all of
# the local density of states, -Im G / pi.
rho = tr.ldos(e_fermi)
assert np.allclose(rho, -np.imag(np.diag(tr.get_green(e_fermi))) / np.pi, atol=1e-6)

# %%
# No broadening: exact up to the band edges
# -------------------------------------------
# A single impurity :math:`\varepsilon` in a chain transmits
# :math:`T = v^2/(v^2 + \varepsilon^2)`, with :math:`v = 2\sin k` the
# velocity. As :math:`E` approaches the band edge, :math:`v \to 0`, and the
# Caroli formula with :math:`\eta = 10^{-9}` drifts away from that value,
# while the lead modes stay exact.

chain = 0.8 * np.diag(np.eye(12)[5]) + np.eye(12, k=1) + np.eye(12, k=-1)
wire = Transport(chain)
wire.add_lead([[0.]], [[1.]], [[1.]], [0])
wire.add_lead([[0.]], [[1.]], [[1.]], [11])
gaps = np.logspace(-9, -1, 25)
energies = 2. - gaps
v = 2 * np.sin(np.arccos(energies / 2))
exact = v**2 / (v**2 + 0.64)
err_s = np.abs(wire.transmission(energies) / exact - 1)
err_g = np.abs(wire.transmission(energies, eta=1e-9) / exact - 1)
assert np.all(err_s < 1e-6) and np.all(err_s[gaps >= 1e-6] < 1e-9)
assert err_g[0] > 0.1 and np.all(err_g > err_s)
print('At 1e-9 from the band edge: relative error {:.1e} (S-matrix), {:.1e} (eta = 1e-9).'.format(
    err_s[0], err_g[0]))

fig, axes = plt.subplots(2, 2, figsize=(10, 7.5))
ax = axes[0, 0]
ks = np.linspace(-np.pi, np.pi, 301)
bands = np.array([np.linalg.eigvalsh(h_right + v_right * np.exp(1j * k) + v_right.conj().T * np.exp(-1j * k))
                  for k in ks])
ax.plot(ks, bands, 'k', lw=0.6)
ax.axhline(e_fermi, color='gray', ls='--', lw=0.8)
inc = modes.velocities < 0
ax.plot(modes.momenta[inc], np.full(inc.sum(), e_fermi), 'o', mfc='white', mec='r', label='incoming')
ax.plot(modes.momenta[~inc], np.full((~inc).sum(), e_fermi), 'o', color='b', label='outgoing')
ax.set_ylim(-4.2, -1.)
ax.set_xlabel('$k$')
ax.set_ylabel('$E$')
ax.set_title('Lead modes at $E_F$')
ax.legend(loc='upper center', fontsize=8)

ax = axes[0, 1]
im = ax.imshow(t_modes, cmap='viridis', vmin=0, vmax=1)
fig.colorbar(im, ax=ax, shrink=0.8)
ax.set_xlabel('incoming mode $b$ (left lead)')
ax.set_ylabel('outgoing mode $a$ (right lead)')
ax.set_title(r'$|t_{ab}|^2$, $T = %.3f$' % transmission)

ax = axes[1, 0]
sc = ax.scatter(x, y, c=rho, s=18, marker='s', cmap='magma')
ax.add_patch(plt.Circle((length / 2, 0.6 * width), np.sqrt(8 * np.log(3.)), fill=False,
                        color='c', lw=0.8))  # the antidot, where V = 1
fig.colorbar(sc, ax=ax, shrink=0.8)
ax.set_aspect('equal')
ax.set_title('LDOS of the scattering states')
ax.set_xticks([])
ax.set_yticks([])

ax = axes[1, 1]
ax.loglog(gaps, err_g, 'o-', ms=3, label=r'Caroli, $\eta = 10^{-9}$')
ax.loglog(gaps, np.maximum(err_s, 1e-17), 's-', ms=3, label='S-matrix (lead modes)')
ax.set_xlabel('distance to the band edge $2 - E$')
ax.set_ylabel('relative error of $T$')
ax.set_title('Impurity in a chain')
ax.legend(fontsize=8)
fig.set_layout_engine('tight')
