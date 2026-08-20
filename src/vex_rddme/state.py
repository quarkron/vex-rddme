"""Lattice state: per-species integer occupancy counts.

``n[species, voxel]`` is the whole state. This is the reaction-diffusion master
equation's own variable. The RDME is an equation over occupancy numbers, not over
particle trajectories, and nothing in this package needs particle identity.

The occupancy cap is a *total* across species, not per species: it is a statement
about how many hard spheres fit in a voxel, and spheres of different species
compete for the same room.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Species:
    """One chemical species.

    Parameters
    ----------
    name : str
        Identifier used in messages and observables.
    sigma_nm : float
        Hard-sphere diameter in nanometres. Zero means the species is ideal: it
        occupies no volume, contributes nothing to the weighted densities, and is
        excluded from the free-energy table index.
    gamma : array of float
        Coupling to each static basis field. Length must match the number of
        basis fields; all zeros means the species feels no drift.
    inert : bool
        A crowder: participates in volume exclusion but is required to have zero
        coupling to every basis field and to appear in no reaction. Enforced at
        construction time by :class:`~vex_rddme.react.ReactionSet` and here.
    """

    name: str
    sigma_nm: float = 0.0
    gamma: np.ndarray = field(default_factory=lambda: np.zeros(0))
    inert: bool = False

    def __post_init__(self):
        self.name = str(self.name)
        self.sigma_nm = float(self.sigma_nm)
        if not np.isfinite(self.sigma_nm) or self.sigma_nm < 0:
            raise ValueError(
                f"species {self.name!r}: sigma_nm must be finite and non-negative; "
                f"got {self.sigma_nm}"
            )
        self.gamma = np.atleast_1d(np.asarray(self.gamma, dtype=np.float64))
        if self.inert and np.any(self.gamma != 0.0):
            raise ValueError(
                f"species {self.name!r} is declared inert but has non-zero basis "
                f"coupling gamma={self.gamma}. An inert crowder must not drift, so "
                "that crowding can be varied independently of the reacting species."
            )

    @property
    def is_hard_core(self) -> bool:
        return self.sigma_nm > 0.0


class State:
    """Per-species occupancy counts on a lattice, with mass bookkeeping.

    Mass conservation is checked against the *recorded* initial totals adjusted by
    net reaction firings, so transport is required to conserve exactly while
    reactions are allowed to change totals only by their stoichiometry.
    """

    def __init__(self, lattice, species, occupancy_cap):
        self.lattice = lattice
        self.species = list(species)
        if not self.species:
            raise ValueError("at least one species is required")

        names = [s.name for s in self.species]
        if len(set(names)) != len(names):
            dupes = sorted({n for n in names if names.count(n) > 1})
            raise ValueError(f"duplicate species names: {dupes}")

        n_bases = {len(s.gamma) for s in self.species}
        if len(n_bases) != 1:
            raise ValueError(
                "every species must declare a coupling for the same number of basis "
                f"fields; got lengths {sorted(n_bases)} for {names}"
            )
        self.n_bases = n_bases.pop()

        occupancy_cap = int(occupancy_cap)
        if occupancy_cap < 1:
            raise ValueError(f"occupancy_cap must be at least 1; got {occupancy_cap}")
        self.occupancy_cap = occupancy_cap

        self.counts = np.zeros((self.n_species, lattice.n_voxels), dtype=np.int64)
        self._initial_totals = np.zeros(self.n_species, dtype=np.int64)
        self._reaction_delta = np.zeros(self.n_species, dtype=np.int64)
        self._initial_recorded = False

    # ------------------------------------------------------------- properties

    @property
    def n_species(self) -> int:
        return len(self.species)

    def index(self, name) -> int:
        """Species index from a name, or pass through an integer index."""
        if isinstance(name, (int, np.integer)):
            idx = int(name)
            if not 0 <= idx < self.n_species:
                raise KeyError(f"species index {idx} out of range [0, {self.n_species})")
            return idx
        for i, s in enumerate(self.species):
            if s.name == name:
                return i
        raise KeyError(
            f"unknown species {name!r}; known species are "
            f"{[s.name for s in self.species]}"
        )

    @property
    def hard_core_indices(self):
        """Indices of species with non-zero diameter, in declaration order."""
        return [i for i, s in enumerate(self.species) if s.is_hard_core]

    def totals(self) -> np.ndarray:
        return self.counts.sum(axis=1)

    def occupancy(self) -> np.ndarray:
        """Total occupancy per voxel, summed across species."""
        return self.counts.sum(axis=0)

    def lattice_view(self, s) -> np.ndarray:
        """Counts for one species reshaped to the lattice shape."""
        return self.counts[self.index(s)].reshape(self.lattice.shape)

    def headroom(self) -> np.ndarray:
        """Remaining capacity per voxel before the occupancy cap is reached."""
        return self.occupancy_cap - self.occupancy()

    # ---------------------------------------------------------------- seeding

    def set_counts(self, s, values) -> None:
        """Set one species' counts from a lattice-shaped or flat array."""
        idx = self.index(s)
        arr = np.asarray(values)
        if arr.shape == self.lattice.shape:
            arr = arr.reshape(-1)
        elif arr.shape != (self.lattice.n_voxels,):
            raise ValueError(
                f"counts for {self.species[idx].name!r} must have shape "
                f"{self.lattice.shape} or ({self.lattice.n_voxels},); got {arr.shape}"
            )
        if np.any(arr < 0):
            raise ValueError(f"counts for {self.species[idx].name!r} must be non-negative")

        trial = self.counts.copy()
        trial[idx] = arr.astype(np.int64)
        over = trial.sum(axis=0) > self.occupancy_cap
        if np.any(over):
            worst = int(trial.sum(axis=0).max())
            raise ValueError(
                f"setting counts for {self.species[idx].name!r} would put "
                f"{int(over.sum())} voxel(s) above the occupancy cap "
                f"{self.occupancy_cap} (worst voxel would hold {worst}). "
                "Raise occupancy_cap or reduce the counts."
            )
        self.counts = trial

    def seed_uniform(self, s, total, rng) -> None:
        """Scatter ``total`` particles of one species uniformly, respecting the cap.

        Placement is by draw-and-retry against the remaining capacity, so the
        requested total is placed exactly or an error is raised. The count is never
        silently truncated.
        """
        idx = self.index(s)
        total = int(total)
        if total < 0:
            raise ValueError(f"total must be non-negative; got {total}")
        if total == 0:
            return

        capacity = int(self.headroom().sum())
        if total > capacity:
            raise ValueError(
                f"cannot place {total} particles of {self.species[idx].name!r}: only "
                f"{capacity} slots remain below the occupancy cap "
                f"{self.occupancy_cap}. Raise the cap, enlarge the lattice, or "
                "reduce the count."
            )

        placed = np.zeros(self.lattice.n_voxels, dtype=np.int64)
        remaining = total
        for _ in range(1000):
            if remaining == 0:
                break
            draw = rng.integers(0, self.lattice.n_voxels, size=remaining)
            add = np.bincount(draw, minlength=self.lattice.n_voxels).astype(np.int64)
            room = self.headroom() - placed
            accepted = np.minimum(add, room)
            placed += accepted
            remaining = total - int(placed.sum())
        if remaining != 0:
            raise RuntimeError(
                f"uniform seeding failed to place {remaining} of {total} particles of "
                f"{self.species[idx].name!r} after 1000 attempts; the lattice is too "
                "close to the occupancy cap for rejection seeding. Use set_counts "
                "with an explicit arrangement."
            )
        self.counts[idx] += placed

    # ------------------------------------------------------- mass bookkeeping

    def record_initial(self) -> None:
        """Freeze the current totals as the reference for mass conservation."""
        self._initial_totals = self.totals().copy()
        self._reaction_delta[:] = 0
        self._initial_recorded = True

    def note_reaction(self, delta) -> None:
        """Record a net stoichiometric change applied by the reaction layer."""
        self._reaction_delta += np.asarray(delta, dtype=np.int64)

    def expected_totals(self) -> np.ndarray:
        if not self._initial_recorded:
            raise RuntimeError(
                "record_initial() has not been called; there is no reference to "
                "check mass against"
            )
        return self._initial_totals + self._reaction_delta

    def mass_residual(self) -> np.ndarray:
        """Per-species (actual - expected) totals. All zeros when conserved."""
        return self.totals() - self.expected_totals()

    def check_mass(self) -> None:
        """Raise if any species' total departs from its expected value.

        Exact: counts are integers and every channel is integer-valued, so there is
        no tolerance to set. A non-zero residual is a bug, not drift.
        """
        residual = self.mass_residual()
        if np.any(residual != 0):
            lines = [
                f"  {self.species[i].name}: actual {int(self.totals()[i])}, "
                f"expected {int(self.expected_totals()[i])} "
                f"(initial {int(self._initial_totals[i])} "
                f"{int(self._reaction_delta[i]):+d} from reactions), "
                f"residual {int(residual[i]):+d}"
                for i in range(self.n_species)
                if residual[i] != 0
            ]
            raise AssertionError("mass not conserved:\n" + "\n".join(lines))

    def check_occupancy_cap(self) -> None:
        """Raise if any voxel exceeds the occupancy cap."""
        occ = self.occupancy()
        if np.any(occ > self.occupancy_cap):
            worst = int(occ.max())
            where = int(np.argmax(occ))
            raise AssertionError(
                f"occupancy cap {self.occupancy_cap} exceeded: voxel {where} "
                f"{self.lattice.coord(where)} holds {worst} particles"
            )

    def __repr__(self) -> str:
        return (
            f"State(species={[s.name for s in self.species]}, "
            f"totals={self.totals().tolist()}, cap={self.occupancy_cap})"
        )
