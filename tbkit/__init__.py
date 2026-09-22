# Copyright 2014 Charles Poli.
#
# This file is part of TBKIT.  It is subject to the license terms in the
# LICENSE file found in the top-level directory of this distribution and at
# https://github.com/cpoli/tbkit.

"""tbkit: build and solve Tight-Binding models."""

__version__ = "0.2.0"

__all__ = [
    "Lattice", "System", "Plot", "Propagation", "Save", "KSpace",
    "reciprocal_vectors", "error_handling",
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
import tbkit.error_handling
