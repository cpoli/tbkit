Examples
========

Runnable scripts demonstrating the conceptual breakthroughs behind
**tbkit**, from Bloch's band theory to Floquet topological insulators --
see :doc:`/history` for the full chronology each script illustrates.

See also the narrative walkthrough: :doc:`/tutorial`.

Each script in this gallery is self-contained and can be run directly with
``python examples/<section>/<script>.py``. Every script also carries an
RST module docstring as its title/description and uses ``# %%`` markers
to split narrative text from code, which is exactly what Sphinx-Gallery
renders into the pages below -- the script *is* the source of truth for
what you see, not a copy of it. Every numeric claim a script's narrative
makes is checked with an ``assert`` right there in the code -- nothing is
asserted in the docs that isn't also verified in code.

Sections
--------

- **tight_binding** -- the generic real-space/reciprocal-space
  machinery: a graphene flake and its Bloch band structure, including
  Wallace's 1947 linear (Dirac) dispersion near the K point, and the
  physics behind the 2004 isolation of graphene.
- **orbitals** -- several orbitals per site: Slater-Koster bond
  integrals between s, p and d orbitals, with non-orthogonal bases.
- **three_dimensions** -- 3D lattices: the Anderson metal-insulator
  transition and Weyl semimetals with their Fermi arcs.
- **magnetic_field** -- the Peierls substitution: gauge-invariant
  plaquette fluxes, an Aharonov-Bohm ring's flux-periodic spectrum,
  Landau levels, and the fractal Hofstadter butterfly.
- **strain** -- strain-induced pseudo-magnetic fields in graphene.
- **disorder** -- Anderson localization: the Inverse Participation Ratio
  vs. disorder strength, and an extended state next to a localized one.
- **topology** -- topological band theory: the SSH model and its Zak
  phase, the Haldane model's Chern number, the TKNN Hall conductance,
  the Kane-Mele Z2 invariant (Wannier-centre flow and Fu-Kane parities),
  the Thouless pump, the quantum metric, the local Chern marker, the
  tenfold way, and diabolical points.
- **hall_effects** -- Hall conductivities at any Fermi level: the
  anomalous Hall effect of the Haldane model (in k-space, and in real
  space with disorder by the kernel polynomial method) and the intrinsic
  spin Hall effect of the Kane-Mele model.
- **flat_bands** -- the kagome and Lieb lattices' exactly flat bands, the
  natural home for strong-correlation physics via Lieb's theorem.
- **correlations** -- the Hubbard model in mean field: zigzag-edge
  magnetism in graphene and Lieb's theorem.
- **superconductivity** -- the Bogoliubov-de Gennes formalism and the
  Majorana end modes of the Kitaev chain.
- **transport** -- Landauer conductance through a device between
  semi-infinite leads, and the quantized steps of a point contact.
- **large_scale** -- sparse solvers and the kernel polynomial method for
  lattices of hundreds of thousands of sites.
- **dynamics** -- real-time wavepacket propagation: Bloch oscillations and
  the Wannier-Stark ladder under a uniform tilt.
- **non_hermitian** -- PT symmetry, exceptional points, the
  non-Hermitian skin effect with its generalized Brillouin zone, and
  exceptional points in momentum space: vorticity, exceptional rings,
  bulk Fermi arcs, encircling, and biorthogonal Chern numbers.
- **floquet** -- periodically driven lattices: a Floquet Chern
  insulator made with circularly polarized light, and anomalous Floquet
  phases with edge states despite zero Chern numbers.
