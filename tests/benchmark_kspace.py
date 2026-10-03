'''
Benchmark of the k-space diagonalizations (not collected by pytest).

Times the band and topology tools of a multi-orbital model on fine meshes,
with one and several threads (*KSpace.set_workers*), against the reference
loop of one scipy.linalg.eigh call per k-point. Run it before and after a
change to the k-space solvers, to see regressions:

    python tests/benchmark_kspace.py [norb] [workers]
'''
import os
import sys
import time

import numpy as np
import scipy.linalg as LA

from tbkit import KSpace, Lattice


def model(norb: int) -> KSpace:
    '''
    A square lattice with *norb* orbitals per cell, coupled at random to
    their nearest neighbours (fixed seed), with a gap above the lowest band.
    '''
    rng = np.random.default_rng(1)
    unit_cell = [{'tag': chr(97 + n), 'r0': (0., 0.)} for n in range(norb)]
    lat = Lattice(unit_cell=unit_cell, prim_vec=[(1., 0.), (0., 1.)])
    ks = KSpace(lat)
    ks.set_onsite({chr(97 + n): float(n) for n in range(norb)})
    hops = [{'i': i, 'j': j, 'R': R, 't': 0.1 * complex(*rng.normal(size=2))}
            for R in [(1, 0), (0, 1)] for i in range(norb) for j in range(norb)]
    ks.set_hopping(hops)
    return ks


def timed(func) -> float:
    t0 = time.perf_counter()
    func()
    return time.perf_counter() - t0


def main(norb: int = 16, workers: int = os.cpu_count() or 1) -> None:
    ks = model(norb)
    _, mesh = ks.mesh_grid(200)
    cases = {
        'mesh_bands(200)': lambda: ks.mesh_bands(200),
        'get_bands(200^2, eigenvec)': lambda: ks.get_bands(mesh, eigenvec=True),
        'berry_curvature(0, nk=100)': lambda: ks.berry_curvature(0, nk=100),
        'wannier_flow(0, 200, 101)': lambda: ks.wannier_flow(0, nk=200, nk_perp=101),
    }
    reference = timed(lambda: [LA.eigvalsh(ks.get_ham(k)) for k in mesh])
    print(f'norb = {norb}, {len(mesh)} k-points, workers = 1 and {workers}')
    print(f'{"per-k scipy loop (eigvalsh)":32s} {reference:8.3f} s')
    for name, func in cases.items():
        ks.set_workers(1)
        serial = timed(func)
        ks.set_workers(workers)
        parallel = timed(func)
        print(f'{name:32s} {serial:8.3f} s {parallel:8.3f} s')


if __name__ == '__main__':
    main(*map(int, sys.argv[1:]))
