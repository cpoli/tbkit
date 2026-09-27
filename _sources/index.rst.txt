.. tbkit documentation master file, created by
   sphinx-quickstart on Sat Feb 13 09:54:24 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

tbkit
=====

.. toctree::
    :hidden:

    tutorial
    history
    api/gallery/index
    tbkit

**tbkit** is a Python package to build and solve **Tight-Binding** models,
written in fully vectorized **NumPy**. It's built for physics students
working through a textbook problem, curious learners exploring a topic
on their own, and educators building a demonstration.

Start with the :doc:`tutorial`, browse the :doc:`example gallery
</api/gallery/index>`, or jump straight to the :doc:`tbkit` API reference.

For computational physics beyond tight-binding -- quantum mechanics,
classical mechanics, statistical physics, general relativity, and more
-- see `physicskit <https://cpoli.github.io/physicskit/>`_, whose
``physicskit.condensed`` subpackage covers tight-binding band theory and
topology alongside superconductivity.

**tbkit** is composed of the following classes and modules:

    * Lattice
    * System
    * KSpace
    * Plot
    * Propagation
    * Save
    * OrbitalSystem (orbital)
    * Transport (transport)
    * lattices
    * dos
    * occupation
    * kpm
    * slater_koster
    * meanfield
    * bdg
    * floquet


**tbkit** main features:

    * Complex lattice structures, in 1D, 2D and 3D.
    * Complex-valued onsite energies and hoppings.
    * Hermitian and non-Hermitian Tight-Binding Hamiltonians.
    * Sublattices.
    * Hoppings defined by their type, tags, and angles.
    * Any type of hoppings:

        * Neighbors hoppings,
        * Next-neighbors hoppings,
        * Next-next-neighbors hoppings,
        * etc..

    * Implementation of onsite energies and hopping patterns:

       * Dimerization defects.
       * Magnetic field (Peierls substitution).
       * Strain.
       * Hopping disorder.
       * Onsite disorder.

    * Reciprocal-space Bloch Hamiltonians and band structures.
    * Berry curvature and Chern numbers.
    * An optional spin-1/2 degree of freedom, for spin-orbit coupling and
      Zeeman terms.
    * Ribbons (edge states) cut from any periodic model.
    * Zak/Berry phases, Wannier centres and their flow, the Z2 invariant
      (Wilson loops and Fu-Kane parities), the quantum metric, symmetry
      checks and the tenfold way, and the local Chern marker.
    * Magnetic supercells (Hofstadter bands, TKNN Chern numbers).
    * Anomalous and spin Hall conductivities at any Fermi level (Kubo
      formula over the Brillouin zone), and the real-space Kubo-Bastin Hall
      conductivity of large disordered samples (kernel polynomial method).
    * Non-reciprocal models: spectral winding, the skin effect, and the
      generalized Brillouin zone.
    * Exceptional points of 2D non-Hermitian bands: eigenvalue vorticity,
      discriminant winding, an EP finder (charges, orders, exceptional
      rings, bulk Fermi arcs), encircling EPs and diabolical points, and
      biorthogonal Chern numbers of line-gapped bands.
    * Several orbitals per site: Slater-Koster integrals, non-orthogonal
      bases, and spinful multi-orbital real-space models.
    * Sparse eigensolvers, sparse neighbour search, and the kernel
      polynomial method for very large lattices.
    * Green's functions, Fermi levels and occupations, and Landauer
      transport through a device between semi-infinite leads.
    * Hubbard mean field, and Bogoliubov-de Gennes superconductivity.
    * Floquet theory of periodically driven lattices, and the winding
      number of anomalous Floquet phases.
    * Broadened density of states.
    * A small library of ready-made lattices.
    * Time propagation.

**tbkit** is available at https://github.com/cpoli/tbkit and on PyPI at
https://pypi.org/project/tbkit/


To use **tbkit**:

  * Install Python 3.10 or later and three additional packages:

      * numpy
      * scipy
      * matplotlib

  * ``pip install -e .`` from a clone of the repository.

Examples are available at https://github.com/cpoli/tbkit/tree/master/examples,
and rendered with their output in the :doc:`example gallery
</api/gallery/index>`.