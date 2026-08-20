"""Transport: the Scharfetter-Gummel hop with volume exclusion.

One step, per species:

    u_d(v) = [phi_s(v + e_d) - phi_s(v)]           static field difference
           + [F(n_dst + dxi_s) - F(n_dst)]         insertion at the destination
           - [F(n_src) - F(n_src - dxi_s)]         removal at the source
                                     ^^^^^^^^^^
                          the self-exclusion: the hopping particle must not feel
                          its own volume at the voxel it is leaving

    p_d(v) = q * B(u_d(v)),      B(u) = u / (exp(u) - 1)

then the ``n_s(v)`` particles at each voxel are partitioned across the ``2*dim``
directions and "stay" by a sequence of conditional binomial draws.

Two things the whole-array formulation buys, both of which the per-hop CUDA kernel
cannot easily have:

* ``F(n)``, ``F(n + dxi_s)`` and ``F(n - dxi_s)`` are computed **once per species
  per step** as whole-lattice arrays. Every direction is then a roll of arrays
  already in hand, rather than a fresh free-energy evaluation per candidate hop.
* Because the free energy comes from an exact integer table, those three arrays
  are three gathers.

The field term is precomputed once at construction: ``psi`` is static, so
``phi_s(v + e_d) - phi_s(v)`` never changes.
"""

from __future__ import annotations

import numpy as np

from .guards import check_cfl, check_hop_probability_sum, check_xi3_saturation


def bernoulli(u, small=1e-6):
    """``B(u) = u / (exp(u) - 1)``, the Scharfetter-Gummel factor.

    ``B(0) == 1`` exactly, via a series branch. ``B`` is *unbounded* as
    ``u -> -inf`` (a strongly downhill move), which is why the baseline CFL bound on
    ``q`` is necessary but not sufficient and the realised probability sum must be
    checked every step.

    Large positive ``u`` (strongly uphill) is clipped before the exponential so the
    result decays to zero rather than overflowing to NaN. The clip is at a work of
    700 kT, far outside any physical regime, and it only ever makes an already
    negligible probability exactly zero.
    """
    u = np.asarray(u, dtype=np.float64)
    out = np.empty_like(u)
    near = np.abs(u) < small
    # Series: u/(e^u - 1) = 1 - u/2 + u^2/12 + O(u^4)
    un = u[near]
    out[near] = 1.0 - 0.5 * un + un * un / 12.0
    uf = np.clip(u[~near], -700.0, 700.0)
    out[~near] = uf / np.expm1(uf)
    return out


class Hop:
    """Per-species transport with drift and volume exclusion.

    Parameters
    ----------
    lattice : Lattice
    state : State
    exclusion : ExclusionModel or None
        ``None`` disables volume exclusion entirely (every exclusion work is zero).
    psi : array, optional
        Static basis fields, shape ``(n_bases,) + lattice.shape`` or
        ``(n_bases, n_voxels)``. Never modified.
    D_um2_s : float or sequence
        Diffusion coefficient per species, in um^2/s.
    tau_s : float
        Timestep in seconds.
    report : GuardReport
    """

    def __init__(self, lattice, state, exclusion, psi, D_um2_s, tau_s, report):
        self.lattice = lattice
        self.state = state
        self.exclusion = exclusion
        self.tau_s = float(tau_s)
        self.report = report
        self.n_species = state.n_species

        if self.tau_s <= 0 or not np.isfinite(self.tau_s):
            raise ValueError(f"tau_s must be positive and finite; got {tau_s}")

        self._setup_fields(psi)
        self._setup_diffusion(D_um2_s)
        self._precompute_field_work()

    # ------------------------------------------------------------------ setup

    def _setup_fields(self, psi):
        n_bases = self.state.n_bases
        if psi is None:
            if n_bases != 0:
                raise ValueError(
                    f"species declare {n_bases} basis coupling(s) but no psi was "
                    "given; pass psi with that many fields or declare gamma of "
                    "length 0"
                )
            self.psi = np.zeros((0, self.lattice.n_voxels))
        else:
            arr = np.asarray(psi, dtype=np.float64)
            if arr.ndim == self.lattice.dim:          # a single unwrapped field
                arr = arr[None, ...]
            if arr.shape[1:] == self.lattice.shape:
                arr = arr.reshape(arr.shape[0], -1)
            if arr.shape != (n_bases, self.lattice.n_voxels):
                raise ValueError(
                    f"psi must have shape ({n_bases},) + {self.lattice.shape} or "
                    f"({n_bases}, {self.lattice.n_voxels}); got {np.shape(psi)}"
                )
            if not np.all(np.isfinite(arr)):
                raise ValueError("psi contains non-finite values")
            # Own a copy and freeze it: the fields are inputs, not state, and a
            # caller mutating them mid-run would silently invalidate the
            # precomputed field work below.
            self.psi = arr.copy()
            self.psi.flags.writeable = False

        self.gamma = np.stack(
            [np.asarray(sp.gamma, dtype=np.float64) for sp in self.state.species]
        ) if n_bases else np.zeros((self.n_species, 0))

    def _setup_diffusion(self, D_um2_s):
        D = np.atleast_1d(np.asarray(D_um2_s, dtype=np.float64))
        if D.size == 1:
            D = np.full(self.n_species, float(D[0]))
        if D.shape != (self.n_species,):
            raise ValueError(
                f"D_um2_s must be a scalar or one value per species "
                f"({self.n_species}); got shape {D.shape}"
            )
        if np.any(D < 0) or not np.all(np.isfinite(D)):
            raise ValueError(f"D_um2_s must be finite and non-negative; got {D}")
        self.D_um2_s = D

        # q = D tau / h^2, with D in um^2/s -> nm^2/s and h in nm.
        h_nm = self.lattice.voxel_nm
        self.q = D * 1.0e6 * self.tau_s / (h_nm ** 2)

        for s, sp in enumerate(self.state.species):
            check_cfl(
                float(self.q[s]),
                self.lattice.dim,
                self.report,
                D_um2_s=float(D[s]),
                tau_s=self.tau_s,
                voxel_nm=h_nm,
            )

    def _precompute_field_work(self):
        """``phi_s(v + e_d) - phi_s(v)`` for every species and direction.

        ``psi`` is static, so this is computed once. Shape
        ``(n_species, n_dirs) + lattice.shape``.
        """
        shape = (self.n_species, self.lattice.n_dirs) + self.lattice.shape
        self.dphi = np.zeros(shape, dtype=np.float64)
        if self.state.n_bases == 0:
            return
        for s in range(self.n_species):
            phi = np.tensordot(self.gamma[s], self.psi, axes=(0, 0))
            phi = phi.reshape(self.lattice.shape)
            for d in range(self.lattice.n_dirs):
                self.dphi[s, d] = self.lattice.neighbour(phi, d) - phi

    # ------------------------------------------------------- exclusion arrays

    def _exclusion_arrays(self, counts, step=None):
        """Insertion and removal works per species, as whole-lattice arrays.

        Returns ``(insert, remove)`` each shaped ``(n_species, n_voxels)``:

        * ``insert[s][v] = F(n(v) + dxi_s) - F(n(v))``: cost of adding one
          particle of ``s`` at ``v``.
        * ``remove[s][v] = F(n(v)) - F(n(v) - dxi_s)``: cost already paid by a
          particle of ``s`` sitting at ``v``, i.e. the self-exclusion term.

        A hop's exclusion work is then ``insert[s]`` read at the destination minus
        ``remove[s]`` read at the source.
        """
        V = self.lattice.n_voxels
        if self.exclusion is None or self.exclusion.n_hard_core == 0:
            zeros = np.zeros((self.n_species, V))
            return zeros, zeros.copy()

        xi = self.exclusion.xi(counts)
        check_xi3_saturation(xi[3], self.report, step=step, lattice=self.lattice)

        F0 = self.exclusion.free_energy(counts)
        insert = np.zeros((self.n_species, V))
        remove = np.zeros((self.n_species, V))
        for s in range(self.n_species):
            if not self.state.species[s].is_hard_core:
                continue
            dnu_plus = np.zeros(self.n_species, dtype=np.int64)
            dnu_plus[s] = 1
            dnu_minus = -dnu_plus
            off = self.exclusion.stoichiometry_offset(dnu_plus) if self.exclusion.uses_table else 0
            insert[s] = self.exclusion.shifted_free_energy(counts, off, dnu_plus) - F0
            remove[s] = F0 - self.exclusion.shifted_free_energy(counts, -off, dnu_minus)
        return insert, remove

    # ------------------------------------------------------------- one step

    def probabilities(self, species_index, insert, remove, headroom=None):
        """Per-direction hop probability for one species, shape ``(n_dirs,) + shape``.

        Two sets of probabilities are zeroed here, which is what makes
        :meth:`Lattice.deposit` and the free-energy table lookups safe afterwards:

        * boundary directions, so nothing leaves the lattice;
        * directions whose destination is already at the occupancy cap, so a hard
          constraint is expressed as a zero transition probability rather than
          being repaired after the fact.

        ``headroom`` is the pre-step ``cap - occupancy`` per voxel. Because it is
        evaluated before any particle moves, arrivals from several directions (or
        several species) can still collide; :meth:`step` checks the cap afterwards
        and fails loudly rather than quietly discarding the overflow.
        """
        s = species_index
        lat = self.lattice
        rem = remove[s].reshape(lat.shape)
        ins = insert[s].reshape(lat.shape)
        room = None if headroom is None else headroom.reshape(lat.shape)
        p = np.empty((lat.n_dirs,) + lat.shape, dtype=np.float64)
        for d in range(lat.n_dirs):
            u = self.dphi[s, d] + (lat.neighbour(ins, d) - rem)
            p[d] = self.q[s] * bernoulli(u)
            p[d][~lat.can_leave(d)] = 0.0
            if room is not None:
                p[d][lat.neighbour(room, d) <= 0] = 0.0
        return p

    def step(self, rng, step=None):
        """Advance transport by one timestep, in place on ``state.counts``.

        Mass is conserved exactly: each voxel's particles are partitioned across
        directions plus "stay", and every departure is deposited.
        """
        lat = self.lattice
        counts = self.state.counts
        insert, remove = self._exclusion_arrays(counts, step=step)
        headroom = self.state.headroom()

        new_counts = np.empty_like(counts)
        for s in range(self.n_species):
            p = self.probabilities(s, insert, remove, headroom=headroom)

            # The guard the production kernel replaces with a clip. Must run every
            # step: the exclusion work grows as crowding develops, so a run that
            # satisfies the construction-time CFL bound can violate this later.
            check_hop_probability_sum(
                p.sum(axis=0).reshape(-1),
                self.report,
                step=step,
                species_name=self.state.species[s].name,
                lattice=lat,
            )

            remaining = counts[s].reshape(lat.shape).copy()
            departures = []
            p_left = np.ones(lat.shape, dtype=np.float64)
            for d in range(lat.n_dirs):
                # Conditional probability: this direction's share of whatever
                # probability the earlier directions have not already claimed.
                p_cond = np.where(p_left > 0.0, p[d] / np.maximum(p_left, 1e-300), 0.0)
                # The sum guard above establishes p_cond <= 1; clipping here would
                # be the silent flux loss it exists to prevent, so assert instead.
                if np.any(p_cond > 1.0 + 1e-12):
                    worst = float(p_cond.max())
                    raise AssertionError(
                        f"conditional hop probability {worst:.6g} > 1 for species "
                        f"{self.state.species[s].name!r} direction {d}; the "
                        f"probability-sum guard should have caught this first"
                    )
                moved = rng.binomial(remaining, np.clip(p_cond, 0.0, 1.0))
                departures.append((d, moved))
                remaining = remaining - moved
                p_left = p_left - p[d]

            acc = remaining
            for d, moved in departures:
                lat.assert_no_wrap(moved, d)
                acc = acc + lat.deposit(moved, d)
            new_counts[s] = acc.reshape(-1)

        self.state.counts = new_counts

        # Arrivals from different directions or different species can collide even
        # though every departure targeted a voxel with room. Discarding the excess
        # here would silently break both mass conservation and detailed balance, so
        # it is a failure with an actionable message instead.
        occ = self.state.occupancy()
        if np.any(occ > self.state.occupancy_cap):
            worst = int(occ.max())
            where = int(np.argmax(occ))
            n_over = int(np.count_nonzero(occ > self.state.occupancy_cap))
            self.report.fail(
                "occupancy-cap",
                f"Transport exceeded the occupancy cap "
                f"{self.state.occupancy_cap}"
                f"{'' if step is None else f' at step {step}'}.\n"
                f"Voxel {where} {lat.coord(where)} holds {worst} particles.\n"
                f"{n_over} voxel(s) are over the cap.\n"
                f"Each particle moved to a voxel that had room. "
                f"They then collided on arrival.\n"
                f"Do one of these steps:\n"
                f"  - Reduce tau, so that fewer particles move in each step.\n"
                f"  - Raise occupancy_cap.\n"
                f"  - Reduce the particle count.\n"
                f"If exclusion must prevent this, examine the cap-vs-exclusion "
                f"guard.",
            )
        return self.state

    def __repr__(self) -> str:
        return (
            f"Hop(n_species={self.n_species}, tau_s={self.tau_s:g}, "
            f"q={np.array2string(self.q, precision=4)}, "
            f"exclusion={'on' if self.exclusion is not None else 'off'})"
        )
