"""Volume-excluded reaction-drift-diffusion master equation (RDDME) on a lattice.

A teaching and prototyping implementation of Wang–Peskin–Elston lattice transport
in a static potential, White-Bear/BMCSL hard-sphere exclusion, and Fröhner–Noé
reaction acceptance. Two dependencies: numpy, and matplotlib for the plotting
helpers only.

Not a production solver, and no numerical agreement with the Lattice Microbes 2.6
CUDA drift-RDME solver is claimed or tested. See README.md.

``vex_rddme.viz`` is not imported here: importing it pulls in matplotlib, and the
solver is meant to work in a headless or minimal environment. Import it explicitly
when you want to plot.
"""

from .hop import Hop, bernoulli
from .lattice import Lattice
from .observe import Series, project, report_comparison
from .react import Reaction, ReactionSet
from .sim import Simulation
from .state import Species, State
from .vex import ExclusionModel, bfex, mu_ex_carnahan_starling

__all__ = [
    "Lattice",
    "Species",
    "State",
    "ExclusionModel",
    "Hop",
    "Reaction",
    "ReactionSet",
    "Simulation",
    "Series",
    "project",
    "report_comparison",
    "bernoulli",
    "bfex",
    "mu_ex_carnahan_starling",
]

__version__ = "0.1.0"
