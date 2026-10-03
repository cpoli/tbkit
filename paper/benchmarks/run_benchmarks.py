"""
Wall-clock time and peak memory of tbkit, PythTB and Kwant on shared tasks.

Each case runs in a fresh process with one BLAS thread (and
``KSpace.set_workers(1)``), so the numbers compare algorithms, not thread
counts. Peak memory is the growth of the resident set during the timed
call. Results go to ``paper/results/benchmarks.json``.

Tasks:
- k-space: band energies on an N x N mesh, the Chern number of the lowest
  band, and the Wilson-loop (Berry phase) flow, for the random multi-orbital
  model of ``tests/benchmark_kspace.py``, against PythTB 2.0.
- transport: the scattering matrix of a square W x W device at one energy,
  against Kwant (MUMPS).

Usage (from the repository root)::

    python paper/benchmarks/run_benchmarks.py [--quick]
"""
import json
import multiprocessing as mp
import os
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[var] = '1'


def kspace_case(package, task, norb, nk):
    import warnings

    import numpy as np
    sys.path.insert(0, str(ROOT / 'tests'))
    from benchmark_kspace import model, timed

    warnings.filterwarnings('ignore', category=DeprecationWarning)
    ks = model(norb)
    ks.set_workers(1)
    if package == 'tbkit':
        _, mesh = ks.mesh_grid(nk)
        run = {'bands': lambda: ks.get_bands(mesh),
               'chern': lambda: ks.chern_number(0, nk=nk),
               'wilson': lambda: ks.wannier_flow(0, nk=nk, nk_perp=nk)}[task]
    else:
        from pythtb import Lattice, Mesh, TBModel, WFArray
        lat = Lattice([[1., 0.], [0., 1.]], [[0., 0.]] * norb, periodic_dirs=[0, 1])
        m = TBModel(lattice=lat)
        m.set_onsite([float(n) for n in range(norb)])
        # the input hoppings of benchmark_kspace.model, one per conjugate pair
        rng = np.random.default_rng(1)
        for R in [(1, 0), (0, 1)]:
            for i in range(norb):
                for j in range(norb):
                    m.set_hop(0.1 * complex(*rng.normal(size=2)), i, j, list(R), mode='add')
        probe = np.array([[0.13, 0.71], [0.42, 0.05]])
        assert np.allclose(m.solve_ham(probe), ks.get_bands(probe * 2 * np.pi), atol=1e-12)
        kred = np.stack(np.meshgrid(np.arange(nk) / nk, np.arange(nk) / nk, indexing='ij'), -1).reshape(-1, 2)

        def wilson():
            mesh = Mesh(['k', 'k'])
            mesh.build_grid((nk, nk))
            wfa = WFArray(m.lattice, mesh)
            wfa.solve_model(m)
            return wfa.berry_phase(0, state_idx=[0], berry_evals=True)

        run = {'bands': lambda: m.solve_ham(kred),
               'chern': lambda: m.chern_number((0, 1), (nk, nk), occ_idxs=[0]),
               'wilson': wilson}[task]
    return measure(run, timed)


def transport_case(package, width):
    import numpy as np
    sys.path.insert(0, str(ROOT / 'tests'))
    from benchmark_kspace import timed

    energy, t = -2.6, -1.
    rng = np.random.default_rng(3)
    disorder = rng.uniform(-0.5, 0.5, (width, width))
    if package == 'tbkit':
        import tbkit.lattices as lattices
        from tbkit.kspace import ribbon
        from tbkit.system import System
        from tbkit.transport import Transport
        lat = lattices.square()
        lat.get_lattice(width, width)
        sys_ = System(lat)
        sys_.set_hopping([{'n': 1, 't': t}])
        sys_.set_onsite({'a': 0.})
        sys_.onsite[:] = disorder[lat.coor['x'].astype(int), lat.coor['y'].astype(int)]
        sys_.get_ham()
        strip = ribbon(lattices.square(), [{'i': 0, 'j': 0, 'R': (1, 0), 't': t},
                                           {'i': 0, 'j': 0, 'R': (0, 1), 't': t}], width=width, direction=1)
        tr = Transport(sys_.ham)
        tr.attach_lead(sys_, strip, -1)
        tr.attach_lead(sys_, strip, 1)
        run = lambda: tr.smatrix(energy).transmission(1, 0)
    else:
        import kwant
        lat = kwant.lattice.square(norbs=1)
        b = kwant.Builder()
        b[(lat(i, j) for i in range(width) for j in range(width))] = lambda s: disorder[s.tag[0], s.tag[1]]
        b[lat.neighbors()] = t
        lead = kwant.Builder(kwant.TranslationalSymmetry((-1, 0)))
        lead[(lat(0, j) for j in range(width))] = 0.
        lead[lat.neighbors()] = t
        b.attach_lead(lead)
        b.attach_lead(lead.reversed())
        fsys = b.finalized()
        run = lambda: kwant.smatrix(fsys, energy).transmission(1, 0)
    return measure(run, timed)


def measure(run, timed):
    import resource
    scale = 1 if sys.platform == 'darwin' else 1024  # ru_maxrss: bytes on macOS, KiB on Linux
    before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * scale
    out = {}
    seconds = timed(lambda: out.setdefault('value', run()))
    after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * scale
    value = out['value']
    try:
        value = float(value)
    except TypeError:
        value = None
    return {'seconds': seconds, 'peak_mb': (after - before) / 2**20, 'value': value}


def isolated(func, *args):
    with mp.get_context('spawn').Pool(1) as pool:
        return pool.apply(func, args)


def main(quick=False):
    import importlib.metadata as metadata
    results = {'versions': {p: metadata.version(p) for p in ('tbkit', 'kwant', 'pythtb', 'numpy', 'scipy')},
               'platform': platform.platform(), 'processor': platform.processor(),
               'python': platform.python_version(), 'threads': 1, 'kspace': [], 'transport': []}
    norbs = (2, 8) if quick else (2, 8, 32)
    nks = (20, 40) if quick else (25, 50, 100, 200)
    for task in ('bands', 'chern', 'wilson'):
        for norb in norbs:
            for nk in nks:
                for package in ('tbkit', 'pythtb'):
                    r = isolated(kspace_case, package, task, norb, nk)
                    results['kspace'].append({'task': task, 'norb': norb, 'nk': nk, 'package': package, **r})
                    print(f"{task:7s} norb={norb:3d} nk={nk:4d} {package:7s} {r['seconds']:9.3f} s "
                          f"{r['peak_mb']:8.1f} MB", flush=True)
    widths = (20, 40, 80) if quick else (20, 40, 80, 160, 320, 640)
    for width in widths:
        for package in ('tbkit', 'kwant'):
            r = isolated(transport_case, package, width)
            results['transport'].append({'width': width, 'sites': width**2, 'package': package, **r})
            print(f"smatrix W={width:4d} {package:7s} {r['seconds']:9.3f} s {r['peak_mb']:8.1f} MB "
                  f"T={r['value']:.6f}", flush=True)
    for width in widths:  # the two packages solve the same device
        a, b = (next(r for r in results['transport'] if r['width'] == width and r['package'] == p)
                for p in ('tbkit', 'kwant'))
        assert abs(a['value'] - b['value']) < 1e-8, (width, a['value'], b['value'])
    name = 'benchmarks_quick.json' if quick else 'benchmarks.json'
    out = ROOT / 'paper' / 'results' / name
    out.write_text(json.dumps(results, indent=2) + '\n')
    print(f'Wrote {out.relative_to(ROOT)}')


if __name__ == '__main__':
    main(quick='--quick' in sys.argv)
