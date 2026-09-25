r"""
Superconductivity in the Bogoliubov-de Gennes (BdG) formalism. With a
pairing term :math:`\sum_{ij}\Delta_{ij}c^\dagger_ic^\dagger_j + h.c.`,
the Hamiltonian is :math:`\frac12\Psi^\dagger H_{BdG}\Psi` in the Nambu
basis :math:`\Psi = (c_1, \dots, c_N, c_1^\dagger, \dots, c_N^\dagger)^T`,

.. math::

    H_{BdG} = \begin{pmatrix} h - \mu & \Delta \\ \Delta^\dagger & -(h-\mu)^T \end{pmatrix}\, ,

with :math:`\Delta^T = -\Delta` (Fermi statistics). Its spectrum is
symmetric, :math:`\pm E_n`, by the particle-hole symmetry
:math:`\tau_x H_{BdG}^*\tau_x = -H_{BdG}`; the quasiparticle energies are
the :math:`E_n \ge 0`, and a zero mode is a Majorana mode. Real-space
models come from *bdg_ham*, periodic ones from *bdg_kspace* (a
:class:`tbkit.kspace.KSpace` with the holes as extra orbitals, so its bands,
Berry phases, *finite_ham*, ... all apply).
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
import scipy.sparse as sparse

import tbkit.error_handling as error_handling
from tbkit.lattice import Lattice


def bdg_ham(ham, pairing, mu: float = 0.) -> sparse.csr_matrix:
    r'''
    Get the real-space BdG Hamiltonian (see the module docstring).

    :param ham: Square Hermitian matrix, shape (N, N) (e.g. *System.ham*).
    :param pairing: Square antisymmetric matrix, shape (N, N): :math:`\Delta`
        (see *pairing_bonds*, *pairing_s_wave*).
    :param mu: Real number. Default value 0. Chemical potential.

    :returns:
        * **ham_bdg** -- Sparse CSR matrix, shape (2N, 2N).
    '''
    h = sparse.csr_matrix(ham, dtype='c16')
    delta = sparse.csr_matrix(pairing, dtype='c16')
    error_handling.square_matrix(h, 'ham')
    error_handling.hermitian(h)
    error_handling.pairing(delta, h.shape[0])
    error_handling.real_number(mu, 'mu')
    h = h - mu * sparse.identity(h.shape[0], dtype='c16', format='csr')
    return sparse.csr_matrix(sparse.bmat([[h, delta], [delta.conj().T, -h.T]]))


def pairing_bonds(sys, delta: complex) -> sparse.csr_matrix:
    r'''
    Get a p-wave (odd) pairing on the bonds of *sys.hop*:
    :math:`\Delta_{ij} = \Delta`, :math:`\Delta_{ji} = -\Delta` for every
    stored hopping :math:`i\to j` (oriented as *System.set_hopping* stores
    them: angle in :math:`[0, 180)`). On a chain it is Kitaev's
    :math:`\Delta\,c^\dagger_jc^\dagger_{j+1}`.

    :param sys: **System** instance, with hoppings.
    :param delta: Complex number. :math:`\Delta`.

    :returns:
        * **pairing** -- Sparse CSR matrix, shape (sites, sites).
    '''
    error_handling.empty_hop(sys.hop)
    error_handling.number(delta, 'delta')
    i, j = sys.hop['i'].astype(int), sys.hop['j'].astype(int)
    n = sys.lat.sites
    vals = np.full(len(i), delta, dtype='c16')
    return sparse.csr_matrix((np.concatenate([vals, -vals]), (np.concatenate([i, j]), np.concatenate([j, i]))),
                                      shape=(n, n))


def pairing_s_wave(n_sites: int, delta: complex) -> sparse.csr_matrix:
    r'''
    Get an onsite singlet (s-wave) pairing
    :math:`\Delta\sum_i c^\dagger_{i\uparrow}c^\dagger_{i\downarrow} + h.c.`
    for a spinful model with rows ordered site by site, spin up then down
    (as *KSpace* with ``spin=True`` and *OrbitalSystem* with one orbital).

    :param n_sites: Positive integer. Number of sites.
    :param delta: Complex number. :math:`\Delta`.

    :returns:
        * **pairing** -- Sparse CSR matrix, shape (2 n_sites, 2 n_sites).
    '''
    error_handling.positive_int(n_sites, 'n_sites')
    error_handling.number(delta, 'delta')
    block = np.array([[0., delta], [-delta, 0.]], dtype='c16')
    return sparse.csr_matrix(sparse.kron(sparse.identity(n_sites), block))


def bdg_kspace(ks, pairing: list[dict], mu: float = 0.):
    r'''
    Get the BdG Bloch Hamiltonian of a periodic model,

    .. math::

        H_{BdG}(\mathbf{k}) = \begin{pmatrix} h(\mathbf{k})-\mu & \Delta(\mathbf{k}) \\
        \Delta(\mathbf{k})^\dagger & -[h(-\mathbf{k})-\mu]^T\end{pmatrix}\, ,\qquad
        \Delta(\mathbf{k}) = \sum_{\mathbf{R}}\Delta_{ij}(\mathbf{R})\,e^{i\mathbf{k}\cdot\mathbf{R}}\, ,

    as a new **KSpace** whose orbitals are those of *ks* (particles,
    ``0 ... norb-1``) followed by their holes (``norb ... 2 norb-1``, at the
    same positions). Each pairing :math:`\Delta\,c^\dagger_{i,\mathbf{0}}c^\dagger_{j,\mathbf{R}} + h.c.`
    enters antisymmetrized, :math:`\Delta_{ij}(\mathbf{R}) = -\Delta_{ji}(-\mathbf{R})`.

    :param ks: **KSpace** instance (Hermitian, orthogonal basis).
    :param pairing: List of dictionaries with keys ('i', 'j', 'R', 'delta'):
        orbital indices of *ks* (``0 ... norb-1``; with ``spin=True``,
        ``2*site + spin``), lattice vector, amplitude.
    :param mu: Real number. Default value 0. Chemical potential.

    :returns:
        * **ks_bdg** -- **KSpace** instance, 2 norb orbitals.

    Example usage::

        # Kitaev chain: E(k) = +-sqrt((2t cos k - mu)^2 + 4 Delta^2 sin^2 k)
        chain = KSpace(lattices.chain())
        chain.set_hopping([{'i': 0, 'j': 0, 'R': (1,), 't': t}])
        kit = bdg_kspace(chain, [{'i': 0, 'j': 0, 'R': (1,), 'delta': delta}], mu=mu)
    '''
    from tbkit.kspace import KSpace
    error_handling.hermitian_kspace(ks.is_hermitian(), ks._overlap_hop)
    error_handling.pairing_kspace(pairing, ks.norb, ks.dim)
    error_handling.real_number(mu, 'mu')
    norb = ks.norb
    tau = ks.orbital_positions()
    tags = np.repeat(ks.tags, 2) if ks.spin else ks.tags
    cell = [{'tag': str(t), 'r0': tuple(float(c) for c in r)} for t, r in zip(tags, tau)]
    new = KSpace(Lattice(unit_cell=cell + [dict(c) for c in cell], prim_vec=ks.lat.prim_vec))
    h0 = np.diag(ks.onsite) + ks._onsite_offdiag - mu * np.eye(norb)
    new.onsite = np.concatenate([np.diag(h0), -np.diag(h0)]).astype('c16')
    off = h0 - np.diag(np.diag(h0))
    new._onsite_offdiag = np.block([[off, np.zeros((norb, norb))], [np.zeros((norb, norb)), -off.T]]).astype('c16')
    hops = [(i, j, R, t) for i, j, R, t in ks._hop]
    hops += [(j + norb, i + norb, -R, -t) for i, j, R, t in ks._hop]
    for dic in pairing:
        R = np.zeros(ks.space_dim)
        for n, a in zip(dic['R'], ks.lat.prim_vec):
            R += n * np.array(a)
        i, j, d = dic['i'], dic['j'], complex(dic['delta'])
        hops += [(i, j + norb, R, d), (j, i + norb, -R, -d),
                     (j + norb, i, -R, np.conj(d)), (i + norb, j, R, -np.conj(d))]
    new._hop = hops
    return new


def particle_hole(n: int) -> NDArray[np.float64]:
    r'''
    Get the unitary part :math:`\tau_x` of the BdG particle-hole symmetry
    :math:`C = \tau_x\mathcal{K}` (:math:`C^2 = +1`), for *n* particle
    orbitals: pass it to *KSpace.symmetry_error* (``antiunitary=True``,
    ``anti=True``) or *KSpace.tenfold_class* (``particle_hole=``).

    :param n: Positive integer. Number of particle orbitals.

    :returns:
        * **tau_x** -- Real ndarray, shape (2n, 2n).
    '''
    error_handling.positive_int(n, 'n')
    return np.kron(np.array([[0., 1.], [1., 0.]]), np.eye(n))
