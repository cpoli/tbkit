"""
Cross-check of tbkit's band topology against PythTB 2.0.

The Haldane and Kane-Mele models are built in PythTB (its own
``pythtb.models``) and carried over to a tbkit ``KSpace`` by
``pythtb_bridge.to_kspace``. Each package then computes, on its own:

- the band energies at random k-points,
- the Chern number (Haldane, topological and trivial),
- the Berry phase of a band along k1, for several k2 (modulo 2 pi),
- the quantum metric and Berry curvature at random k-points,
- the hybrid Wannier centres of the occupied pair of the Kane-Mele model
  along k1, for several k2 (as sets, modulo one lattice vector),

and tbkit's Z2 invariant is checked against the phase read off PythTB's
Wannier-centre flow.

Each comparison is asserted; ``RESULTS`` holds the largest deviations.
"""
import warnings

import numpy as np
from pythtb import Mesh, WFArray
from pythtb.models import haldane, kane_mele

from pythtb_bridge import reciprocal, to_kspace

warnings.filterwarnings('ignore', category=DeprecationWarning)
RESULTS = {}
rng = np.random.default_rng(1)


def record(name, dev, tol):
    RESULTS[name] = float(dev)
    assert dev < tol, (name, dev)
    print(f'{name:55s} max deviation {dev:.1e}')


def circle(x):
    """Coefficients of prod (z - exp(2 pi i x)): a set of centres modulo 1."""
    return np.poly(np.exp(2j * np.pi * np.asarray(x)))


def wilson_mesh(model, nk1, nk2):
    mesh = Mesh(['k', 'k'])
    mesh.build_grid((nk1, nk2))
    wfa = WFArray(model.lattice, mesh, spinful=model.spinful)
    wfa.solve_model(model)
    k2 = mesh.get_k_points()[0, :, 1]
    return wfa, k2


# Haldane model ---------------------------------------------------------------

dev_e = dev_c = dev_bp = dev_g = dev_o = 0.
for delta, expected in ((0.2, 1), (1.2, 0)):  # topological, trivial (3 sqrt3 t2 = 0.78)
    m = haldane(delta, -1., 0.15)
    ks, B = to_kspace(m), reciprocal(m)

    kr = rng.random((200, 2))
    e_tb = np.array([np.linalg.eigvalsh(ks.get_ham(k)) for k in kr @ B])
    dev_e = max(dev_e, np.abs(np.sort(m.solve_ham(kr), axis=1) - e_tb).max())

    c_py = m.chern_number((0, 1), (60, 60), occ_idxs=[0])
    c_tb = ks.chern_number(0, nk=60)
    assert round(c_tb) == round(c_py) and abs(round(c_tb)) == expected
    dev_c = max(dev_c, abs(c_tb - c_py))

    nk1 = 81
    wfa, k2 = wilson_mesh(m, nk1, 9)
    bp_py = wfa.berry_phase(0, state_idx=[0])
    bp_tb = np.array([ks.berry_phase(0, nk=nk1, direction=0, k_perp=float(q)) for q in k2])
    dev_bp = max(dev_bp, np.abs(np.angle(np.exp(1j * (bp_py - bp_tb)))).max())

    kr = rng.random((20, 2))
    g_py = m.quantum_metric(kr, occ_idxs=[0], cartesian=True)
    o_py = m.berry_curvature(kr, occ_idxs=[0], cartesian=True)
    for n, k in enumerate(kr @ B):
        q = ks.quantum_geometric_tensor(0, k, dk=1e-5, positions=True)
        dev_g = max(dev_g, np.abs(q.real - g_py[:, :, n]).max())
        dev_o = max(dev_o, np.abs(-2 * q.imag - o_py[:, :, n]).max())

record('Haldane: band energies (400 k-points)', dev_e, 1e-12)
record('Haldane: Chern numbers (raw values, 60x60 mesh)', dev_c, 1e-6)
record('Haldane: Berry phases along k1 (mod 2 pi)', dev_bp, 1e-10)
record('Haldane: quantum metric (finite differences)', dev_g, 1e-6)
record('Haldane: Berry curvature (finite differences)', dev_o, 1e-6)

# Kane-Mele model -------------------------------------------------------------

dev_e = dev_w = 0.
for delta, z2 in ((0.7, 1), (2.5, 0)):  # (delta, t, soc, rashba): QSH and trivial
    m = kane_mele(delta, 1., 0.24, 0.05)
    ks, B = to_kspace(m), reciprocal(m)

    kr = rng.random((200, 2))
    e_py = np.sort(m.solve_ham(kr), axis=1)
    e_tb = np.array([np.linalg.eigvalsh(ks.get_ham(k)) for k in kr @ B])
    dev_e = max(dev_e, np.abs(e_py - e_tb).max())

    nk1 = 81
    wfa, k2 = wilson_mesh(m, nk1, 11)
    phases = wfa.berry_phase(0, state_idx=[0, 1], berry_evals=True, contin=False)
    for n, q in enumerate(k2):
        c_py = phases[n] / (2 * np.pi)  # PythTB's Wilson-loop phases are 2 pi x centre
        c_tb = ks.wannier_centers([0, 1], nk=nk1, direction=0, k_perp=float(q))
        dev_w = max(dev_w, np.abs(circle(c_tb) - circle(c_py)).max())
    assert ks.z2_invariant([0, 1], nk=nk1, nk_perp=51) == z2

record('Kane-Mele: band energies (400 k-points)', dev_e, 1e-12)
record('Kane-Mele: hybrid Wannier centres (mod 1)', dev_w, 1e-10)
print('Kane-Mele: tbkit Z2 = 1 (QSH) and 0 (trivial), as the phase diagram requires.')
