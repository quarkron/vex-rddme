"""Volume exclusion: weighted densities, the BMCSL free energy, and the integer table.

Three layers, each exact:

1. **Weighted densities.** Fundamental-measure theory reduces a hard-sphere mixture
   to four scalar fields. On a lattice they are linear in the integer counts::

       dxi[s][n] = (pi/6) * sigma_s**n / V_voxel      (n = 0..3)
       xi[n](v)  = sum_s n_s(v) * dxi[s][n]

   ``xi[3]`` is the local packing fraction. The n=0..2 moments carry number,
   radius, and surface. That is why a reaction that merges two spheres into one
   of equal *volume* still changes the free energy.

2. **White-Bear / BMCSL excess free energy.** The functional of those four fields.

3. **The integer table.** Because step 1 is linear in *integers*, the free energy is
   a function of the integer count vector, and can be tabulated exactly. Adding or
   removing one particle of species ``s`` is an index shift of ``+-stride_s``; a
   whole reaction stoichiometry is a single shift ``sum_s dnu_s * stride_s``. So
   every kinetic channel, transport and reaction alike, costs one gather.

   Strides are **C-order** (``radix**(S-1-s)``) to match a table built by
   ``np.indices``. The transposed assignment produces a table that is silently
   wrong but still runs, so ``tests/test_vex.py`` pins it.

The table is exact only for point species. Real-valued contributions to the
weighted densities (a smeared multi-voxel body, for instance) break the integer
quantisation; those configurations must use the elementwise path, and selecting it
is always announced.
"""

from __future__ import annotations

import numpy as np

from .guards import (
    GuardReport,
    check_cap_binds_after_exclusion,
    check_table_radix,
    report_occupancy_at_half_packing,
)

# xi[3] at or above 1 means the voxel is full: the log in the functional diverges.
# The production CUDA kernel clamps here silently, which switches exclusion off in
# exactly the crowded regime the method exists for. We refuse instead.
XI3_SATURATION = 1.0 - 1e-9

# Default ceiling on table entries. 8 M entries is 64 MB in float64.
DEFAULT_MAX_TABLE_ENTRIES = 8_388_608


def dxi_increments(species, voxel_volume_nm3):
    """Per-species weighted-density increments, shape ``(n_species, 4)``.

    Row ``s`` is ``(pi/6) * sigma_s**n / V`` for ``n = 0..3``. Ideal species
    (``sigma == 0``) get an all-zero row: they occupy no volume and contribute
    nothing to any weighted density.
    """
    V = float(voxel_volume_nm3)
    if not np.isfinite(V) or V <= 0:
        raise ValueError(f"voxel volume must be positive and finite; got {V}")
    out = np.zeros((len(species), 4), dtype=np.float64)
    pi6 = np.pi / 6.0
    for i, sp in enumerate(species):
        if sp.sigma_nm > 0.0:
            for n in range(4):
                out[i, n] = pi6 * sp.sigma_nm ** n / V
    return out


def bfex(xi, voxel_volume_nm3):
    """White-Bear / BMCSL excess free energy per voxel, in units of kT.

    ``xi`` is shape ``(4, ...)``. Returns an array shaped like ``xi[0]``.

    The ``log(1 - xi3)`` inside the third term is what makes this BMCSL rather
    than the Rosenfeld-1989 SPT form. Evaluated in float64 throughout: the log and
    the divisions by ``(1 - xi3)`` are the sensitive spots as the voxel fills.

    ``voxel_volume_nm3`` is **required**, not defaulted. The bracketed expression is
    a free-energy *density* (units 1/nm^3, since ``xi0`` carries them); multiplying
    by the voxel volume turns it into a per-voxel energy in kT. Omitting the factor
    leaves every channel work smaller by exactly ``1/V``. With a 10 nm voxel that
    is 1000x too weak, which switches exclusion off while every profile still looks
    plausible. A pure scale error of that kind is invisible to a correlation
    coefficient, so the argument is mandatory and
    ``test_vex.py::test_insertion_work_matches_carnahan_starling`` pins the scale.

    Empty voxels return exactly zero. Voxels at or above saturation are *not*
    silently clamped. The caller is expected to have run the saturation guard,
    but the arithmetic is kept finite here so a guard can report a value rather
    than crash on a NaN.
    """
    xi = np.asarray(xi, dtype=np.float64)
    if xi.shape[0] != 4:
        raise ValueError(f"xi must have 4 leading components; got shape {xi.shape}")
    V = float(voxel_volume_nm3)
    if not np.isfinite(V) or V <= 0:
        raise ValueError(f"voxel_volume_nm3 must be positive and finite; got {V}")
    x0, x1, x2, x3 = xi[0], xi[1], xi[2], xi[3]

    occupied = x3 > 0.0
    x3c = np.minimum(x3, XI3_SATURATION)
    omx = 1.0 - x3c
    ln_omx = np.log(omx)
    inv_pi = 1.0 / np.pi

    t1 = -(6.0 * inv_pi) * x0 * ln_omx
    t2 = (18.0 * inv_pi) * (x1 * x2) / omx
    # x2**3 / x3**2 is 0/0 at an empty voxel; the `occupied` mask discards it.
    denom = np.where(occupied, x3c * x3c, 1.0)
    t3 = (6.0 * inv_pi) * (x2 ** 3) / denom * (x3c / (omx * omx) + ln_omx)

    return V * np.where(occupied, t1 + t2 + t3, 0.0)


def mu_ex_carnahan_starling(eta):
    """Single-species excess chemical potential, in kT, from Carnahan-Starling.

    ``mu_ex = (8*eta - 9*eta**2 + 3*eta**3) / (1 - eta)**3``

    The analytic reference for demonstration 1, part 1. BMCSL reduces to this for one
    species, so
    this function is a check on :func:`bfex`, not a component of the solver.
    """
    eta = np.asarray(eta, dtype=np.float64)
    omx = 1.0 - eta
    return (8.0 * eta - 9.0 * eta ** 2 + 3.0 * eta ** 3) / omx ** 3


class ExclusionModel:
    """Volume exclusion for one species set, with the exact integer table.

    Parameters
    ----------
    species : sequence of Species
    voxel_volume_nm3 : float
    occupancy_cap : int
        Total particles a voxel may hold. Fixes the table radix.
    max_table_entries : int
        Ceiling on ``radix ** n_hard_core``. Above it the elementwise path is used
        and the reason is reported.
    report : GuardReport, optional
        Destination for guard messages. One is created if not supplied.
    """

    def __init__(
        self,
        species,
        voxel_volume_nm3,
        occupancy_cap,
        max_table_entries=DEFAULT_MAX_TABLE_ENTRIES,
        report=None,
    ):
        self.species = list(species)
        self.n_species = len(self.species)
        self.voxel_volume_nm3 = float(voxel_volume_nm3)
        self.occupancy_cap = int(occupancy_cap)
        self.report = report if report is not None else GuardReport()

        self.dxi = dxi_increments(self.species, self.voxel_volume_nm3)
        self.hard_core = [i for i, s in enumerate(self.species) if s.is_hard_core]
        self.n_hard_core = len(self.hard_core)

        # Insertion lookups need one slot of headroom above the cap, removal needs
        # one below zero never to be reached; radix = cap + 2 gives both.
        self.radix = self.occupancy_cap + 2
        check_table_radix(self.radix, self.occupancy_cap, self.report)

        self._build_table(max_table_entries)
        report_occupancy_at_half_packing(self, self.report)
        check_cap_binds_after_exclusion(
            self.max_attainable_xi3(), self.occupancy_cap, self.report
        )

    # ------------------------------------------------------------------ table

    @property
    def uses_table(self) -> bool:
        return self._table is not None

    @property
    def table_entries(self) -> int:
        return 0 if self._table is None else int(self._table.size)

    def _build_table(self, max_table_entries):
        self._table = None
        self._strides = None

        if self.n_hard_core == 0:
            self.report.info(
                "exclusion",
                "no species has a non-zero diameter, so volume exclusion is inactive "
                "and all channel works are zero",
            )
            return

        requested = self.radix ** self.n_hard_core
        if requested > max_table_entries:
            self.report.warn(
                "exclusion-fallback",
                f"The free-energy table would need radix**S = {self.radix}**"
                f"{self.n_hard_core} = {requested:,} entries "
                f"({requested * 8 / 1e6:.1f} MB).\n"
                f"The ceiling is {max_table_entries:,} entries.\n"
                f"This model now evaluates the BMCSL functional elementwise.\n"
                f"This result is correct, but each step is 20-30x slower.\n"
                f"Do one of these steps to use the table:\n"
                f"  - Reduce the occupancy cap.\n"
                f"  - Reduce the number of hard-core species.\n"
                f"  - Raise max_table_entries.",
            )
            return

        # C-order strides, matching np.indices: the FIRST hard-core species varies
        # slowest. Assigning radix**s instead would build a consistent-looking but
        # wrong index; test_vex.py checks that the transposed order fails.
        self._strides = (
            self.radix ** np.arange(self.n_hard_core, dtype=np.int64)[::-1]
        ).astype(np.int64)

        grid = np.indices((self.radix,) * self.n_hard_core).reshape(
            self.n_hard_core, -1
        )
        xi = self._xi_from_hard_core_counts(grid)
        self._table = bfex(xi, self.voxel_volume_nm3)

        self.report.info(
            "exclusion",
            f"free-energy table built: {self.n_hard_core} hard-core species, "
            f"radix {self.radix}, {requested:,} entries "
            f"({self._table.nbytes / 1e6:.2f} MB)",
        )

    def _xi_from_hard_core_counts(self, hc_counts):
        """Weighted densities from counts of the hard-core species only."""
        dxi_hc = self.dxi[self.hard_core]          # (n_hard_core, 4)
        return (np.asarray(hc_counts).T @ dxi_hc).T  # (4, ...)

    # ------------------------------------------------------- public evaluation

    def xi(self, counts):
        """Weighted densities from full per-species counts ``(n_species, V)``."""
        counts = np.asarray(counts)
        if counts.shape[0] != self.n_species:
            raise ValueError(
                f"counts must have {self.n_species} species rows; got {counts.shape}"
            )
        return (counts.T @ self.dxi).T

    def xi3_of(self, counts, species=None):
        """Packing fraction per voxel from a subset of species.

        ``species`` is a sequence of indices; ``None`` means all of them. Restricting
        it gives the contribution of one group. The crowder's packing fraction is the
        independent variable of the crowding sweep, and it has to be separable from
        the reacting species' own contribution.
        """
        counts = np.asarray(counts)
        if species is None:
            return (counts.T @ self.dxi[:, 3])
        idx = list(species)
        return counts[idx].T @ self.dxi[idx, 3]

    def max_attainable_xi3(self):
        """Largest ``xi3`` reachable if the fullest-packing species fills a voxel.

        The cap is on total occupancy, so the worst case is ``cap`` particles of
        whichever species has the largest volume increment.
        """
        if self.n_hard_core == 0:
            return 0.0
        return float(self.occupancy_cap * self.dxi[:, 3].max())

    def occupancy_at_half_packing(self):
        """Per-species occupancy at which that species alone reaches ``xi3 = 0.5``."""
        out = {}
        for i, sp in enumerate(self.species):
            d3 = self.dxi[i, 3]
            out[sp.name] = float("inf") if d3 == 0.0 else 0.5 / d3
        return out

    def index(self, counts):
        """Table index per voxel from full per-species counts. Table path only."""
        if not self.uses_table:
            raise RuntimeError(
                "index() requires the table path; this model is using the "
                "elementwise fallback"
            )
        hc = np.asarray(counts)[self.hard_core]
        return (hc * self._strides[:, None]).sum(axis=0)

    def stride(self, species_index):
        """Table index shift for inserting one particle of a species.

        Zero for ideal species: they do not appear in the index, so inserting one
        changes no weighted density and shifts no index.
        """
        if not self.uses_table:
            raise RuntimeError("stride() requires the table path")
        if species_index in self.hard_core:
            return int(self._strides[self.hard_core.index(species_index)])
        return 0

    def stoichiometry_offset(self, dnu):
        """Single table shift for a stoichiometric change ``dnu`` over all species.

        ``sum_s dnu_s * stride_s``. So a whole reaction's exclusion work is one
        pair of gathers regardless of how many species it involves.
        """
        return int(sum(int(dnu[s]) * self.stride(s) for s in range(self.n_species)))

    @staticmethod
    def _is_integral(counts):
        """Whether counts can index the table.

        The table is defined on integer count vectors. A *fractional* composition,
        a time-averaged mean density for instance, is still a meaningful argument
        to the free energy, since it is a smooth function of the weighted densities;
        it simply cannot be looked up. Such input takes the elementwise path, which
        is not a degradation: the two paths agree to roundoff wherever both apply,
        and elementwise is the only correct option here.
        """
        return np.issubdtype(np.asarray(counts).dtype, np.integer)

    def free_energy(self, counts):
        """Excess free energy per voxel, table path or elementwise.

        Fractional counts (e.g. a mean composition) use the elementwise path; see
        :meth:`_is_integral`.
        """
        if self.uses_table and self._is_integral(counts):
            return self._table[self.index(counts)]
        return bfex(self.xi(counts), self.voxel_volume_nm3)

    def shifted_free_energy(self, counts, offset, dnu=None):
        """Excess free energy after applying a channel change.

        ``offset`` is the table shift; ``dnu`` is the equivalent per-species
        stoichiometry, needed only by the elementwise path. Supplying both keeps
        the two paths interchangeable at the call site.
        """
        if self.uses_table and self._is_integral(counts):
            return self._table[self.index(counts) + offset]
        if dnu is None:
            raise ValueError(
                "the elementwise path needs dnu as well as offset; pass both so "
                "call sites work on either path"
            )
        delta = np.asarray(dnu, dtype=np.float64) @ self.dxi   # (4,)
        return bfex(self.xi(counts) + delta[:, None], self.voxel_volume_nm3)

    def channel_work(self, counts, dnu):
        """Work of applying ``dnu`` at every voxel, in kT.

        ``F(n + dnu) - F(n)``. Used directly for reactions; the hop composes two
        of these (an insertion at the destination and a removal at the source) so
        that the hopping particle does not feel its own exclusion.
        """
        offset = self.stoichiometry_offset(dnu) if self.uses_table else 0
        return self.shifted_free_energy(counts, offset, dnu) - self.free_energy(counts)

    def __repr__(self) -> str:
        mode = f"table({self.table_entries:,} entries)" if self.uses_table else "elementwise"
        return (
            f"ExclusionModel(n_species={self.n_species}, "
            f"n_hard_core={self.n_hard_core}, radix={self.radix}, mode={mode})"
        )
