Coming from PythTB
==================

If you already write `PythTB <https://www.physics.rutgers.edu/pythtb/>`_
scripts, this page shows how the same tasks are written in **tbkit**, where
the two codes use different conventions, and one model ported line by line.
A model you already have converts in one line, either way, with
:mod:`tbkit.interop` (PythTB 2.0 ``TBModel`` or 1.x ``tb_model``)::

    from tbkit.interop import from_pythtb, to_pythtb

    ks = from_pythtb(model)   # PythTB model -> tbkit KSpace
    model = to_pythtb(ks)     # tbkit KSpace -> PythTB model

The models of PythTB's ``pythtb.models`` are also in :mod:`tbkit.models`
(``haldane``, ``kane_mele``, ``ssh``, ...), with tbkit's own parameter names.

The same tasks side by side
---------------------------

.. list-table::
   :header-rows: 1
   :widths: 18 41 41

   * - Task
     - PythTB 2.0
     - tbkit
   * - A lattice
     - ``Lattice(lat_vecs, orb_vecs, periodic_dirs=[0, 1])``, orbitals in
       reduced coordinates
     - :class:`~tbkit.lattice.Lattice` ``(unit_cell, prim_vec)``, a tag and a
       Cartesian position ``'r0'`` per orbital
   * - The model
     - ``model = TBModel(lattice=lat, spinful=False)``
     - ``model = KSpace(lat)`` (``spin=True`` for spinful)
   * - On-site energies
     - ``model.set_onsite([-delta, delta])``
     - ``model.set_onsite({'a': -delta, 'b': delta})``, by tag
   * - Hoppings
     - ``model.set_hop(t1, 0, 1, [0, 0])``, one call per hopping
     - ``model.set_hopping([{'i': 0, 'j': 1, 'R': (0, 0), 't': t1}, ...])``,
       one list
   * - Spin
     - 2x2 amplitudes, or ``[s0, sx, sy, sz]``
     - 2x2 amplitudes, with ``tbkit.kspace.PAULI``
   * - Parameters
     - string or callable amplitudes, ``set_parameters``
     - value functions, ``model.set_params(**params)``
   * - A band structure
     - ``k_vec, k_dist, k_node = model.k_path(nodes, nk)``, then
       ``model.solve_ham(k_vec)``
     - ``dist, bands = model.k_path(points, nk)``, or
       ``model.get_bands(k_points)``; ``high_symmetry_path(lat)`` gives
       standard points, and ``model.plot_bands()`` the figure
   * - A Berry phase
     - ``wfa = WFArray(model.lattice, mesh)``, ``wfa.solve_model(model)``,
       ``wfa.berry_phase(0, state_idx=[0])``
     - ``model.berry_phase(0, nk=81, direction=0, k_perp=k2)``, one loop,
       no wave-function array
   * - A Chern number
     - ``model.chern_number((0, 1), (60, 60), occ_idxs=[0])``
     - ``model.chern_number(0, nk=60)``
   * - A Wilson loop
     - ``wfa.berry_phase(..., berry_evals=True)``, ``wfa.wilson_loop``
     - ``model.wannier_centers(bands, ...)`` (hybrid Wannier centres),
       ``model.wannier_flow(bands, ...)`` (their flow),
       ``model.z2_invariant(bands)``
   * - Berry curvature, quantum geometry
     - ``model.berry_curvature``, ``model.quantum_metric``,
       ``model.quantum_geometric_tensor``
     - ``model.berry_curvature``, ``model.quantum_geometric_tensor``
   * - A ribbon, a finite piece
     - ``model.cut_piece(n, dir)``, ``model.make_finite``
     - :func:`tbkit.kspace.ribbon` (a 1D :class:`~tbkit.kspace.KSpace`),
       ``model.finite_ham(n_cells)``, or a :class:`~tbkit.system.System`
       from the same lattice
   * - Local Chern marker
     - ``model.local_chern_marker``
     - ``System.get_local_chern_marker``
   * - Wannier90 files
     - ``TBModel.from_w90``
     - :func:`tbkit.io.read_wannier90`

Differences of convention
-------------------------

- **Hoppings.** Both codes store the amplitude as the matrix element
  :math:`\langle i,\mathbf{0}|H|j,\mathbf{R}\rangle` and add its conjugate,
  so give each pair once and the hopping lists translate one to one, with
  the same signs.
- **k-points.** PythTB takes k in units of the reciprocal vectors (reduced
  coordinates), tbkit's ``get_ham`` and ``get_bands`` in Cartesian
  coordinates: multiply by the matrix whose rows are the reciprocal vectors,
  ``k_red @ np.array(reciprocal_vectors(lat.prim_vec))``. tbkit's loop
  positions (``k_perp``, ``k_fixed``) are fractions of a reciprocal vector,
  as PythTB's mesh points are.
- **Bloch phases.** PythTB's :math:`H(\mathbf{k})` includes the orbital
  positions, :math:`e^{i\mathbf{k}\cdot(\mathbf{R}+\boldsymbol\tau_j-\boldsymbol\tau_i)}`;
  tbkit's ``get_ham`` uses the periodic gauge :math:`e^{i\mathbf{k}\cdot\mathbf{R}}`.
  The two matrices differ by a diagonal unitary: the bands, Chern numbers and
  :math:`\mathbb{Z}_2` invariants are the same. Berry phases and Wannier
  centres depend on the positions, and tbkit's include them by default
  (``positions=True``), so they equal PythTB's.
- **Berry phases are wrapped.** ``wfa.berry_phase`` along a line of loops
  makes the phase continuous; ``model.berry_phase`` returns one loop in
  :math:`(-\pi, \pi]`. Use ``np.unwrap`` on a list of them.
- **Units.** Neither code has units: energies are in the units of the
  hoppings, lengths in those of the lattice vectors.

The Haldane model, ported line by line
--------------------------------------

The Haldane model as ``pythtb.models.haldane`` builds it
(:math:`\Delta = 0.2`, :math:`t_1 = -1`, :math:`t_2 = 0.15`,
:math:`\phi = \pi/2`): the bands at :math:`\Gamma`, K and M, the Chern number
of the lower band, :math:`C = -1`, and the Berry phase of the lower band
along :math:`\mathbf{b}_1` for five values of :math:`k_2`. PythTB and tbkit
agree to :math:`10^{-12}`.

PythTB (shown, not run when the docs are built):

.. literalinclude:: migration/original/pythtb_haldane.py
   :language: python
   :start-at: import numpy as np

tbkit (run, with its assertions, when the docs are built):

.. literalinclude:: migration/pythtb_haldane.py
   :language: python
   :start-at: import numpy as np

What tbkit adds, and what it does not do
----------------------------------------

The same lattice also gives real-space models (disorder, magnetic fields,
defects), quantum transport, Hall, optical and thermoelectric responses,
non-Hermitian and Floquet models, crystal symmetry and symmetry indicators,
mean field and exact diagonalization (see the :doc:`tutorial` and the
:doc:`example gallery </api/gallery/index>`).

tbkit has no ``WFArray``: Berry phases, Wilson loops and Chern numbers take
their k-mesh as an argument and diagonalize on it. It has no counterpart of
PythTB's parameter-space meshes (a Berry phase over a model parameter,
``Mesh(['k', 'l'])``); vary the parameter with ``set_params`` and compute the
phase at each value instead.
