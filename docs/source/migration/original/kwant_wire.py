"""Kwant's half of the worked model of "Coming from Kwant" (not run in CI).

A quantum wire (Kwant tutorial, "Transport through a quantum wire"), with a
magnetic field through a window of the scattering region. Run it with
Kwant 1.5 installed; it prints the conductances that the tbkit version
(``docs/source/migration/kwant_wire.py``) asserts.
"""
import numpy as np
import kwant

t, W, L = 1., 10, 30
x0, x1 = 10, 20                      # the field threads 10 <= x <= 20


def window(x):
    return np.where((x >= x0) & (x <= x1), 1., 0.)


def hopping(site_i, site_j, alpha):  # H[i, j], Landau gauge A = (-B y g(x), 0)
    (xi, yi), (xj, yj) = site_i.pos, site_j.pos
    return -t * np.exp(-2j * np.pi * alpha * yi * window((xi + xj) / 2) * (xj - xi))


lat = kwant.lattice.square(a=1, norbs=1)
syst = kwant.Builder()
syst[(lat(x, y) for x in range(L) for y in range(W))] = 4 * t
syst[lat.neighbors()] = hopping

lead = kwant.Builder(kwant.TranslationalSymmetry((-1, 0)))
lead[(lat(0, y) for y in range(W))] = 4 * t
lead[lat.neighbors()] = -t
syst.attach_lead(lead)
syst.attach_lead(lead.reversed())
fsyst = syst.finalized()

energies = [0.25, 0.5, 1.0, 1.5]
G0 = [kwant.smatrix(fsyst, e, params={'alpha': 0.}).transmission(1, 0) for e in energies]
GB = [kwant.smatrix(fsyst, e, params={'alpha': 0.05}).transmission(1, 0) for e in energies]
print('G(B = 0) =', np.round(G0, 13))
print('G(B)     =', np.round(GB, 13).tolist())
