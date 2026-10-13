History
=======

.. epigraph::

   "It was found that the wave functions [...] in a periodic potential
   have exactly the form of a modulated plane wave." -- F. Bloch, 1928

A chronology of the breakthroughs behind Tight-Binding theory, from
Bloch's 1928 theorem to the Weyl semimetals, Floquet and non-Hermitian
lattices of the 2010s, in one chapter per theme. Every stop has a
pointer to the corresponding **tbkit** functionality and a short,
runnable, numerically-verified example in ``examples/`` -- nothing here
is asserted without also being checked in code.

- :doc:`models_breakthroughs` -- Bloch's theorem, Slater-Koster and k·p
  models, Wannier functions, and solvers for very large lattices.
- :doc:`fields_breakthroughs` -- magnetic fields (Peierls substitution,
  Landau levels, Hofstadter's butterfly), strain, and disorder.
- :doc:`topology_breakthroughs` -- Chern, Z2, mirror and spin Chern
  numbers, Berry phases, the Bott index, Weyl points, and higher-order
  phases.
- :doc:`response_breakthroughs` -- Hall-type responses, Landauer
  transport between leads, and linear and nonlinear optics.
- :doc:`interactions_breakthroughs` -- flat bands from geometry and from
  a moiré twist, the Hubbard model, and superconductivity.
- :doc:`driven_breakthroughs` -- wavepacket dynamics, Floquet lattices,
  and non-Hermitian bands with exceptional points.

.. toctree::
   :maxdepth: 1
   :hidden:

   models_breakthroughs
   fields_breakthroughs
   topology_breakthroughs
   response_breakthroughs
   interactions_breakthroughs
   driven_breakthroughs
