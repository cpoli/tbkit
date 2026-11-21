"""The worked model of "Coming from Kwant", ported line by line to tbkit.

A quantum wire (Kwant tutorial, "Transport through a quantum wire"), with a
magnetic field through a window of the scattering region. The expected
conductances are those Kwant 1.5 prints for the same model
(``original/kwant_wire.py``).
"""
import numpy as np

import tbkit.lattices as lattices
from tbkit.kspace import ribbon
from tbkit.system import System
from tbkit.transport import Transport

t, W, L = 1., 10, 30
x0, x1 = 10, 20                      # the field threads 10 <= x <= 20


def window(x):
    return np.where((x >= x0) & (x <= x1), 1., 0.)


def peierls(alpha):                  # phase of H[i, j], gauge A = (-B y g(x), 0)
    return lambda xi, yi, xj, yj: -2 * np.pi * alpha * yi * window((xi + xj) / 2) * (xj - xi)


def wire(alpha):
    lat = lattices.square()          # a = 1, one site 'a' per cell
    lat.get_lattice(n1=L, n2=W)      # sites at x = 0..L-1, y = 0..W-1
    syst = System(lat)
    syst.set_onsite({'a': 4 * t})
    syst.set_hopping([{'n': 1, 't': -t}])
    syst.set_peierls_phase(peierls(alpha))
    syst.get_ham()

    lead = ribbon(lattices.square(), [{'i': 0, 'j': 0, 'R': (1, 0), 't': -t},
                                      {'i': 0, 'j': 0, 'R': (0, 1), 't': -t}],
                  width=W, direction=1, onsite={'a': 4 * t})
    tr = Transport(syst.ham)
    tr.attach_lead(syst, lead, direction=-1)   # lead 0, towards -x
    tr.attach_lead(syst, lead, direction=1)    # lead 1, towards +x
    return tr


energies = [0.25, 0.5, 1.0, 1.5]
G0 = wire(0.).transmission(energies, 0, 1)
GB = wire(0.05).transmission(energies, 0, 1)

# Kwant 1.5: kwant.smatrix(fsyst, e, params=...).transmission(1, 0)
assert np.allclose(G0, [1., 2., 3., 4.], atol=1e-12)
assert np.allclose(GB, [9.4824712e-06, 0.9939437048524, 1.8907644932474, 2.7304048090486],
                   atol=1e-12)
print('G(B = 0) =', np.round(G0, 13))
print('G(B)     =', np.round(GB, 13).tolist())
