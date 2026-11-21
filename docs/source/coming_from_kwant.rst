Coming from Kwant
=================

If you already write `Kwant <https://kwant-project.org>`_ scripts, this page
shows how the same tasks are written in **tbkit**, where the two codes use
different conventions, and one model ported line by line. A model you already
have converts in one line, either way, with :mod:`tbkit.interop`::

    from tbkit.interop import from_kwant, to_kwant

    ham, positions = from_kwant(fsyst, params={'alpha': 0.05})  # finalized Kwant system -> matrix
    builder = to_kwant(tr)    # tbkit System or Transport (with its leads) -> kwant.Builder

:func:`~tbkit.interop.from_kwant` does not import the leads: attach tbkit leads
to the matrix with :class:`~tbkit.transport.Transport`.

The same tasks side by side
---------------------------

.. list-table::
   :header-rows: 1
   :widths: 18 41 41

   * - Task
     - Kwant 1.5
     - tbkit
   * - A lattice
     - ``lat = kwant.lattice.square(a=1, norbs=1)``
     - ``lat = tbkit.lattices.square()``; any other with
       :class:`~tbkit.lattice.Lattice` ``(unit_cell, prim_vec)``
   * - The sites of a sample
     - ``syst[(lat(x, y) for x in range(L) for y in range(W))] = 4 * t``,
       or a shape function
     - ``lat.get_lattice(n1=L, n2=W)``, then ``syst = System(lat)`` and
       ``syst.set_onsite({'a': 4 * t})``; shapes with ``lat.ellipse_in``,
       ``remove_sites``, ``boundary_line``, ``lat1 + lat2``
   * - Hoppings
     - ``syst[lat.neighbors()] = -t``; ``lat.neighbors(2)`` for second
       neighbours
     - ``syst.set_hopping([{'n': 1, 't': -t}])``; ``{'n': 2, ...}`` for
       second neighbours, or select by ``'ang'`` and ``'tag'``
   * - Building the matrix
     - ``fsyst = syst.finalized()``
     - ``syst.get_ham()`` (sparse CSR in ``syst.ham``)
   * - Parameters
     - value functions ``onsite(site, V)``, ``hopping(site_i, site_j, phi)``,
       passed ``params={...}``
     - value functions ``onsite(site, **params)``,
       ``t(site_i, site_j, **params)``, passed to ``get_ham(**params)``
       (see ``tbkit.values``)
   * - A magnetic field
     - a Peierls phase written into a hopping value function
     - ``syst.set_magnetic_field(alpha)`` (uniform, symmetric gauge), or any
       gauge with ``syst.set_peierls_phase(phase)``
   * - A band structure
     - ``kwant.plotter.bands(flead)``, ``kwant.physics.Bands(flead)``
     - :func:`tbkit.kspace.ribbon` gives the lead (or any periodic model) as a
       :class:`~tbkit.kspace.KSpace`: ``ks.k_path(points, nk)``,
       ``ks.get_bands(ks_list)``, ``ks.plot_bands()``
   * - Leads
     - ``lead = kwant.Builder(kwant.TranslationalSymmetry((-1, 0)))``, then
       ``syst.attach_lead(lead)`` and ``syst.attach_lead(lead.reversed())``
     - ``lead = ribbon(...)``, ``tr = Transport(syst.ham)``, then
       ``tr.attach_lead(syst, lead, direction=-1)`` and
       ``tr.attach_lead(syst, lead, direction=1)``; ``extend=True`` completes
       a ragged interface, as Kwant does
   * - A conductance
     - ``kwant.smatrix(fsyst, E).transmission(1, 0)``
     - ``tr.transmission(energies, 0, 1)`` (many energies), or
       ``tr.smatrix(E).transmission(1, 0)``
   * - Conservation laws
     - ``kwant.Builder(sym, conservation_law=...)``
     - ``tr.attach_lead(..., conservation_law=...)``; blocks addressed as
       ``(lead, block)``, as in Kwant
   * - Scattering states, LDOS, currents
     - ``kwant.wave_function``, ``kwant.ldos``, ``kwant.operator.Current``
     - ``tr.wave_function(E, lead)``, ``tr.ldos(E)``, ``tr.bond_currents``,
       ``tr.local_currents``
   * - Kernel polynomial method
     - ``kwant.kpm``
     - :mod:`tbkit.kpm`

Differences of convention
-------------------------

- **Hopping values.** Both codes store the value you give as the matrix
  element :math:`H_{ij}`, with no sign added: Kwant's tutorial writes
  :math:`4t` on site and :math:`-t` for the hopping, and so does the port
  below. A Kwant hopping value function ``f(site_i, site_j)`` gives
  :math:`H_{ij}`; a tbkit Peierls phase ``phase(xi, yi, xj, yj)`` gives the
  phase of the same :math:`H_{ij}`, so a phase written for Kwant carries over
  unchanged. tbkit adds the conjugate of every bond, as Kwant does.
- **Value functions are vectorized.** Kwant calls a value function once per
  site or bond, with a ``Site`` whose position is ``site.pos``. tbkit calls
  it once per Hamiltonian with arrays: ``site`` is a structured array with
  fields ``'x'``, ``'y'`` (and ``'z'``), ``'tag'`` and ``'index'``, so write
  ``np.where`` rather than ``if``.
- **Lead order.** ``tr.transmission(energies, lead_in, lead_out)`` takes the
  incoming lead first, while ``SMatrix.transmission(lead_out, lead_in)``
  follows Kwant's order. Leads are numbered in the order they are attached in
  both codes.
- **Units.** Neither code has units: energies are in the units of the
  hoppings, lengths in those of the lattice vectors. ``set_magnetic_field``
  takes the flux density :math:`B/\Phi_0`, in flux quanta per unit area.
- **No finalization.** A tbkit model stays editable after ``get_ham``;
  call ``get_ham`` again after a change.

A quantum wire, ported line by line
-----------------------------------

The quantum wire of Kwant's tutorial (width 10, length 30, :math:`t = 1`),
with a magnetic field of 0.05 flux quanta per plaquette through the window
:math:`10 \le x \le 20`. The gauge :math:`\mathbf{A} = (-By\,g(x), 0)`,
with :math:`g` the window, vanishes in the leads. Without the field, the
conductance counts the open channels, 1, 2, 3, 4; with it, Kwant and tbkit
agree to :math:`10^{-12}` on values that are not quantized.

Kwant (shown, not run when the docs are built):

.. literalinclude:: migration/original/kwant_wire.py
   :language: python
   :start-at: import numpy as np

tbkit (run, with its assertions, when the docs are built):

.. literalinclude:: migration/kwant_wire.py
   :language: python
   :start-at: import numpy as np

What tbkit adds, and what it does not do
----------------------------------------

The same lattice also gives band topology (Chern and :math:`\mathbb{Z}_2`
invariants, Wilson loops), Hall, optical and thermoelectric responses,
non-Hermitian and Floquet models, crystal symmetry, mean field and exact
diagonalization, all on the same :class:`~tbkit.system.System` and
:class:`~tbkit.kspace.KSpace` objects (see the :doc:`tutorial` and the
:doc:`example gallery </api/gallery/index>`).

tbkit has no ``Builder``: a model is a lattice and its hoppings, not a graph
of sites to edit one by one, and :func:`~tbkit.interop.from_kwant` imports the
finalized matrix, not the ``Builder`` with its value functions. Kwant's
time-dependent extension, tkwant, has no tbkit counterpart yet.
