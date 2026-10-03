"""
Cross-check of tbkit's scattering matrix against Kwant.

The same devices are built independently in both packages, on the same
site positions, and compared through gauge-invariant quantities: the lead
momenta and velocities, the transmission, the transmission eigenvalues
(singular values of t), and the spin-resolved blocks of S.

Devices:
1. A square-lattice strip with a smooth antidot (energies across the band).
2. A disordered graphene device with zigzag-ribbon leads.
3. A square strip with Rashba spin-orbit coupling in the device only and
   spin-conserving leads (conservation law sigma_z), compared block by block.

Each comparison is asserted; ``RESULTS`` holds the largest deviations for
the paper's validation table.
"""
import numpy as np
import kwant

import tbkit.lattices as lattices
from tbkit.kspace import ribbon
from tbkit.orbital import OrbitalSystem
from tbkit.system import System
from tbkit.transport import Transport, lead_from_kspace, lead_modes

TOL = 1e-8
RESULTS = {}
SZ = np.diag([1., -1.])
I2 = np.eye(2)


def record(name, dev):
    RESULTS[name] = float(dev)
    assert dev < TOL, (name, dev)
    print(f'{name:55s} max deviation {dev:.1e}')


def svals(t):
    return np.sort(np.linalg.svd(t, compute_uv=False) ** 2) if t.size else np.zeros(0)


# 1. Square strip with an antidot ------------------------------------------

t, width, length = -1., 12, 24
antidot = lambda x, y: 3. * np.exp(-((x - length / 2) ** 2 + (y - 0.6 * width) ** 2) / 8.)

lat = lattices.square()
lat.get_lattice(length, width)
x, y = lat.coor['x'], lat.coor['y']
sys = System(lat)
sys.set_hopping([{'n': 1, 't': t}])
sys.set_onsite({'a': 0.})
sys.onsite[:] = antidot(x, y)
sys.get_ham()
strip = ribbon(lattices.square(), [{'i': 0, 'j': 0, 'R': (1, 0), 't': t},
                                   {'i': 0, 'j': 0, 'R': (0, 1), 't': t}], width=width, direction=1)
tr = Transport(sys.ham)
tr.attach_lead(sys, strip, -1)
tr.attach_lead(sys, strip, 1)

klat = kwant.lattice.square(norbs=1)
kb = kwant.Builder()
kb[(klat(i, j) for i in range(length) for j in range(width))] = lambda s: antidot(*s.pos)
kb[klat.neighbors()] = t
klead = kwant.Builder(kwant.TranslationalSymmetry((-1, 0)))
klead[(klat(0, j) for j in range(width))] = 0.
klead[klat.neighbors()] = t
kb.attach_lead(klead)
kb.attach_lead(klead.reversed())
fsq = kb.finalized()

energies = np.linspace(-3.9, 3.9, 53)  # away from E = 0 and the subband edges
dev_t, dev_eig, dev_modes = 0., 0., 0.
for e in energies:
    s_tb, s_kw = tr.smatrix(e), kwant.smatrix(fsq, e)
    dev_t = max(dev_t, abs(s_tb.transmission(1, 0) - s_kw.transmission(1, 0)))
    dev_eig = max(dev_eig, np.max(np.abs(svals(s_tb.submatrix(1, 0)) - svals(s_kw.submatrix(1, 0))),
                                  initial=0.))
    # lead modes of the right lead: kwant orders incoming, then outgoing
    modes_tb = s_tb.lead_info[1]
    modes_kw = s_kw.lead_info[1]
    assert len(modes_tb.momenta) == len(modes_kw.momenta)
    dev_modes = max(dev_modes,
                    np.max(np.abs(np.sort(modes_tb.momenta) - np.sort(modes_kw.momenta)), initial=0.),
                    np.max(np.abs(np.sort(modes_tb.velocities) - np.sort(modes_kw.velocities)), initial=0.))
record('square strip + antidot: transmission', dev_t)
record('square strip + antidot: transmission eigenvalues', dev_eig)
record('square strip + antidot: lead momenta and velocities', dev_modes)

# 2. Disordered graphene with zigzag leads ---------------------------------

t, n1, n2, w_dis = 1., 20, 8, 1.5
rng = np.random.default_rng(7)
glat = lattices.honeycomb()
glat.get_lattice(n1, n2)
gsys = System(glat)
gsys.set_hopping([{'n': 1, 't': t}])
gsys.set_onsite({'a': 0., 'b': 0.})
gsys.onsite[:] = rng.uniform(-w_dis / 2, w_dis / 2, glat.sites)
gsys.get_ham()
key = lambda p: (round(p[0], 6), round(p[1], 6))
disorder = {key(p): v for p, v in zip(zip(glat.coor['x'], glat.coor['y']), gsys.onsite)}
hops = [{'i': 0, 'j': 1, 'R': (0, 0), 't': t}, {'i': 0, 'j': 1, 'R': (-1, 0), 't': t},
        {'i': 0, 'j': 1, 'R': (0, -1), 't': t}]
zz = ribbon(lattices.honeycomb(), hops, width=n2, direction=1)
gtr = Transport(gsys.ham)
gtr.attach_lead(gsys, zz, -1)
gtr.attach_lead(gsys, zz, 1)

a1, a2 = np.array(lattices.honeycomb().prim_vec)
khon = kwant.lattice.general([a1, a2], [(0., 0.), (np.sqrt(3) / 2, 0.5)], norbs=1)
ka, kb_ = khon.sublattices
gb = kwant.Builder()
for i in range(n1):
    for j in range(n2):
        for sub in (ka, kb_):
            gb[sub(i, j)] = lambda s: disorder[key(s.pos)]
gb[khon.neighbors()] = t
glead = kwant.Builder(kwant.TranslationalSymmetry(-a1))
for j in range(n2):
    glead[ka(0, j)] = glead[kb_(0, j)] = 0.
glead[khon.neighbors()] = t
gb.attach_lead(glead)
gb.attach_lead(glead.reversed())
fgr = gb.finalized()
assert len(fgr.sites) == glat.sites

dev_t, dev_eig = 0., 0.
for e in np.linspace(-2.7, 2.7, 37) + 0.013:
    s_tb, s_kw = gtr.smatrix(e), kwant.smatrix(fgr, e)
    dev_t = max(dev_t, abs(s_tb.transmission(1, 0) - s_kw.transmission(1, 0)))
    dev_eig = max(dev_eig, np.max(np.abs(svals(s_tb.submatrix(1, 0)) - svals(s_kw.submatrix(1, 0))),
                                  initial=0.))
record('disordered graphene, zigzag leads: transmission', dev_t)
record('disordered graphene, zigzag leads: transmission eigenvalues', dev_eig)

# 3. Rashba strip, spin-resolved through a conservation law ----------------

t, lam, width, length = -1., 0.3, 6, 16
lat = lattices.square()
lat.get_lattice(length, width)
x = lat.coor['x']
osys = OrbitalSystem(lat, spin=True)
osys.set_hopping([{'n': 1, 't': t}])
osys.set_rashba(lam)
osys.get_ham()
h0, v = lead_from_kspace(ribbon(lattices.square(), [{'i': 0, 'j': 0, 'R': (1, 0), 't': t},
                                                    {'i': 0, 'j': 0, 'R': (0, 1), 't': t}],
                                width=width, direction=1), 1)
h0s, vs = np.kron(h0, I2), np.kron(v, I2)
law = np.kron(np.eye(width), SZ)
left = np.flatnonzero(np.isclose(x, 0.))
right = np.flatnonzero(np.isclose(x, length - 1.))
orbs = lambda sites: [int(2 * s + k) for s in sites for k in (0, 1)]
rtr = Transport(osys.ham)
rtr.add_lead(h0s, vs.conj().T, np.kron(t * np.eye(width), I2), orbs(left), law)
rtr.add_lead(h0s, vs, np.kron(t * np.eye(width), I2), orbs(right), law)

sx = np.array([[0, 1], [1, 0]], complex)
sy = np.array([[0, -1j], [1j, 0]])


def k_hop(s1, s2):
    d = s2.pos - s1.pos  # H[s1, s2], as tbkit's block for the bond from s1 to s2
    return t * I2 + 1j * lam * (sx * d[1] - sy * d[0])


rb = kwant.Builder()
klat2 = kwant.lattice.square(norbs=2)
rb[(klat2(i, j) for i in range(length) for j in range(width))] = 0 * I2
rb[klat2.neighbors()] = k_hop
rlead = kwant.Builder(kwant.TranslationalSymmetry((-1, 0)), conservation_law=SZ)
rlead[(klat2(0, j) for j in range(width))] = 0 * I2
rlead[klat2.neighbors()] = t * I2
rb.attach_lead(rlead)
rb.attach_lead(rlead.reversed())
frb = rb.finalized()

dev_blocks, flips = 0., 0.
for e in np.linspace(-3.5, 3.5, 29) + 0.021:
    s_tb, s_kw = rtr.smatrix(e), kwant.smatrix(frb, e)
    for a in (0, 1):
        for b in (0, 1):
            dev_blocks = max(dev_blocks, abs(s_tb.transmission((1, a), (0, b))
                                             - s_kw.transmission((1, a), (0, b))))
    flips = max(flips, s_tb.transmission((1, 0), (0, 1)))
assert flips > 1e-2  # the device does flip spins
record('Rashba strip: spin-resolved transmission blocks', dev_blocks)
