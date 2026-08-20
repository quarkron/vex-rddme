"""Assembly: one object that wires the layers and runs the construction guards.

``Simulation`` is where the guards that need more than one layer's information get
called — in particular the sigma/voxel consistency check, which needs the voxel edge
from the lattice and the diameters from the species.

Operator splitting order is transport then reactions, matching the production
solver's ordering within a timestep. That ordering is an O(tau) choice, not a
correctness one; it is fixed here so runs are comparable.
"""

from __future__ import annotations

import numpy as np

from .guards import GuardReport, check_sigma_voxel_consistency
from .hop import Hop
from .lattice import Lattice
from .react import ReactionSet
from .state import Species, State
from .vex import DEFAULT_MAX_TABLE_ENTRIES, ExclusionModel


class Simulation:
    """A volume-excluded drift RDDME on a 2D or 3D lattice.

    Parameters
    ----------
    shape : tuple of int
        Lattice voxel counts; length selects dim (2 or 3).
    voxel_nm : float
    species : sequence of Species
    occupancy_cap : int
    psi : array, optional
        Static basis fields. Required if any species declares a basis coupling.
    D_um2_s : float or sequence
    tau_s : float
    exclusion : bool
        Set False to disable volume exclusion even for species with a diameter.
    seed : int
    """

    def __init__(
        self,
        shape,
        voxel_nm,
        species,
        occupancy_cap,
        psi=None,
        D_um2_s=1.0,
        tau_s=1e-5,
        exclusion=True,
        max_table_entries=DEFAULT_MAX_TABLE_ENTRIES,
        seed=0,
        attach_log_handler=True,
    ):
        self.report = GuardReport(attach_handler=attach_log_handler)
        self.rng = np.random.default_rng(seed)
        self.lattice = Lattice(shape=shape, voxel_nm=voxel_nm)
        self.species = list(species)

        any_hard_core = any(s.is_hard_core for s in self.species)

        # Needs the lattice (voxel edge) and the species (diameters) together, so it
        # cannot live in either layer alone. Skipped when exclusion is off: the
        # diameters are then unused, so a cap that would over-pack them says nothing
        # about the run. This is what lets a transport-only study use a large cap
        # with species that happen to carry a diameter.
        if exclusion and any_hard_core:
            check_sigma_voxel_consistency(
                self.species, self.lattice.voxel_nm, occupancy_cap, self.report
            )

        self.state = State(self.lattice, self.species, occupancy_cap=occupancy_cap)
        self.exclusion = (
            ExclusionModel(
                self.species,
                self.lattice.voxel_volume_nm3,
                occupancy_cap,
                max_table_entries=max_table_entries,
                report=self.report,
            )
            if (exclusion and any_hard_core)
            else None
        )
        if not exclusion and any_hard_core:
            self.report.info(
                "exclusion",
                "volume exclusion explicitly disabled although species have "
                "non-zero diameters; transport and reactions will be ideal",
            )

        self.hop = Hop(
            self.lattice, self.state, self.exclusion, psi,
            D_um2_s=D_um2_s, tau_s=tau_s, report=self.report,
        )
        self.reactions = ReactionSet(
            self.state, self.exclusion, self.hop, tau_s=tau_s, report=self.report
        )
        self.tau_s = float(tau_s)
        self.step_index = 0

    # ------------------------------------------------------------------ setup

    def seed_uniform(self, name, total):
        self.state.seed_uniform(name, total, self.rng)
        return self

    def set_counts(self, name, values):
        self.state.set_counts(name, values)
        return self

    def add_reaction(self, name, reactants, products, k_forward, k_reverse=0.0,
                     typical_reactant_product=1.0):
        return self.reactions.add(
            name, reactants, products, k_forward, k_reverse,
            typical_reactant_product=typical_reactant_product,
        )

    def record_initial(self):
        self.state.record_initial()
        return self

    # ----------------------------------------------------------- observables

    @property
    def inert_indices(self):
        """Indices of species declared as inert crowders."""
        return [i for i, s in enumerate(self.species) if s.inert]

    def packing_fraction(self, species=None):
        """Mean packing fraction over the lattice, optionally from a subset.

        ``species`` accepts names or indices. Returns 0 when exclusion is off, since
        no species then occupies volume as far as the dynamics is concerned.
        """
        if self.exclusion is None:
            return 0.0
        idx = None if species is None else [self.state.index(s) for s in species]
        return float(self.exclusion.xi3_of(self.state.counts, idx).mean())

    def crowder_packing_fraction(self):
        """Mean packing fraction contributed by the inert crowders alone.

        This is the sweep variable for the crowding-shifted equilibrium measurement:
        it must be separable from the reacting species so that crowding can be varied
        independently of them.
        """
        idx = self.inert_indices
        if not idx:
            raise RuntimeError(
                "no species is declared inert; declare a crowder with "
                "Species(..., inert=True) to use it as the crowding variable"
            )
        if self.exclusion is None:
            return 0.0
        return float(self.exclusion.xi3_of(self.state.counts, idx).mean())

    # ---------------------------------------------------------------- running

    def step(self):
        """One timestep: transport, then reactions."""
        self.hop.step(self.rng, step=self.step_index)
        if self.reactions.reactions:
            self.reactions.step(self.rng, step=self.step_index)
        self.step_index += 1
        return self

    def run(self, n_steps, burn_in=0, sample_every=1, check_mass_every=0):
        """Run and return time-averaged per-species counts over the sampled window.

        Returns ``(mean_counts, n_samples)`` where ``mean_counts`` has shape
        ``(n_species, n_voxels)``.
        """
        acc = np.zeros((self.state.n_species, self.lattice.n_voxels), dtype=np.float64)
        n = 0
        for i in range(int(n_steps)):
            self.step()
            if check_mass_every and (i + 1) % check_mass_every == 0:
                self.state.check_mass()
            if i >= burn_in and (i - burn_in) % sample_every == 0:
                acc += self.state.counts
                n += 1
        if n == 0:
            raise RuntimeError(
                f"no samples collected: n_steps={n_steps}, burn_in={burn_in}. "
                "Increase n_steps or reduce burn_in."
            )
        return acc / n, n

    def __repr__(self) -> str:
        return (
            f"Simulation(shape={self.lattice.shape}, dim={self.lattice.dim}, "
            f"species={[s.name for s in self.species]}, "
            f"cap={self.state.occupancy_cap}, tau_s={self.tau_s:g}, "
            f"exclusion={'on' if self.exclusion else 'off'}, "
            f"reactions={len(self.reactions.reactions)})"
        )
