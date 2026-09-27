# Copyright 2014 Charles Poli.
#
# This file is part of TBKIT.  It is subject to the license terms in the
# LICENSE file found in the top-level directory of this distribution and at
# https://github.com/cpoli/tbkit.

"""tbkit: build and solve Tight-Binding models."""

__version__ = "0.4.1"

__all__ = [
    "Lattice", "System", "Plot", "Propagation", "Save", "KSpace",
    "reciprocal_vectors", "OrbitalSystem", "Transport", "FloquetKSpace",
    "DrivenKSpace", "StepDrive", "step_drive",
    "MeanFieldResult", "hubbard_mean_field", "ExceptionalPoints", "Encircling",
    "find_exceptional_points", "error_handling",
]

# NOTE: these are explicit imports, not `from tbkit.<module> import *`.
# A wildcard import here would rebind the `tbkit.<module>` submodule
# attributes to the classes they define (since e.g. tbkit/lattice.py both
# *is* the submodule `tbkit.lattice` and defines a `lattice` alias of the
# same name), breaking `import tbkit.lattice as lattice`-style imports.
from tbkit.lattice import Lattice
from tbkit.system import System
from tbkit.plot import Plot
from tbkit.propagation import Propagation
from tbkit.save import Save
from tbkit.kspace import KSpace, reciprocal_vectors
from tbkit.orbital import OrbitalSystem
from tbkit.transport import Transport
from tbkit.floquet import FloquetKSpace, DrivenKSpace, StepDrive, step_drive
from tbkit.meanfield import MeanFieldResult, hubbard_mean_field
from tbkit.exceptional import ExceptionalPoints, Encircling, find_exceptional_points
import tbkit.error_handling
# Model I/O (tbkit.io): appended here to keep the list above untouched.
from tbkit.io import save_model, load_model, read_wannier90
__all__ += ["save_model", "load_model", "read_wannier90"]

# Higher-order topology, moire supercells, self-consistent interactions
from tbkit.higher_order import CornerCharges, quadrupole_moment, corner_charges
__all__ += ["CornerCharges", "quadrupole_moment", "corner_charges"]
from tbkit.moire import SupercellKSpace, MoireKSpace, supercell, twisted_bilayer
__all__ += ["SupercellKSpace", "MoireKSpace", "supercell", "twisted_bilayer"]
from tbkit.meanfield import NonCollinearResult, hubbard_mean_field_noncollinear
from tbkit.bdg import GapResult, s_wave_gap
__all__ += ["NonCollinearResult", "hubbard_mean_field_noncollinear", "GapResult", "s_wave_gap"]
from tbkit.transport import RecursiveTransport
__all__ += ["RecursiveTransport"]
