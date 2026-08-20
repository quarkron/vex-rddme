"""Reactions: reversible pairs of order <= 2 with Fröhner-Noé acceptance.

A reaction's propensity is multiplied by an acceptance factor
``pi = min(1, exp(-dPhi))``, where ``dPhi`` is the total work of the reaction: the
basis-field part ``sum_s dnu_s phi_s(v)`` plus the volume-exclusion part.

The exclusion part is one table shift. A reaction changes the count vector by
``dnu``, so ``didx = sum_s dnu_s * stride_s`` is fixed at declaration time and the
work is a single pair of gathers, the same mechanism transport uses. That is why
this module is short.

**The reverse channel is evaluated from the post-reaction state.** For a pair with
forward change ``dnu``:

    forward, from n:        dbF_f = F(n + dnu) - F(n)
    reverse, from n + dnu:  dbF_r = F(n)       - F(n + dnu)   =  -dbF_f

so ``pi_f / pi_r = exp(-dPhi)`` identically and detailed balance holds by
construction. Evaluating the reverse from ``n`` instead would give
``F(n - dnu) - F(n)``, which is not ``-dbF_f`` unless F is linear. It is not, so
detailed balance would break silently. Hence
:meth:`ReactionSet.detailed_balance_residual` and the test that the pre-state
formulation fails it.

Forward and reverse firings drawn in the same voxel in the same step are **netted**,
and the feasibility cap is applied to the net. Capping each direction separately
would consume reactants that the other direction was about to replace.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .guards import check_reaction_saturation


@dataclass
class Reaction:
    """One reversible reaction channel.

    ``k_reverse = 0`` makes it irreversible.
    """

    name: str
    reactants: tuple
    products: tuple
    k_forward: float
    k_reverse: float = 0.0
    dnu: np.ndarray = field(default=None, repr=False)
    didx_forward: int = field(default=0, repr=False)

    @property
    def order_forward(self) -> int:
        return len(self.reactants)

    @property
    def order_reverse(self) -> int:
        return len(self.products)

    @property
    def is_reversible(self) -> bool:
        return self.k_reverse > 0.0


class ReactionSet:
    """Declared reactions plus the machinery to fire them.

    Parameters
    ----------
    state : State
    exclusion : ExclusionModel or None
    hop : Hop
        Supplies the static per-species potential ``phi_s(v)`` used for the field
        part of the reaction work.
    tau_s : float
    report : GuardReport
    """

    def __init__(self, state, exclusion, hop, tau_s, report):
        self.state = state
        self.exclusion = exclusion
        self.hop = hop
        self.tau_s = float(tau_s)
        self.report = report
        self.reactions = []
        self._pi_sum = []
        self._pi_count = []
        self._phi = self._species_potentials()

    # ------------------------------------------------------------------ setup

    def _species_potentials(self):
        """``phi_s(v) = sum_k gamma[s,k] psi_k(v)``, shape ``(n_species, n_voxels)``."""
        n_sp = self.state.n_species
        V = self.state.lattice.n_voxels
        phi = np.zeros((n_sp, V), dtype=np.float64)
        if self.state.n_bases == 0:
            return phi
        for s in range(n_sp):
            phi[s] = np.tensordot(self.hop.gamma[s], self.hop.psi, axes=(0, 0))
        return phi

    def add(self, name, reactants, products, k_forward, k_reverse=0.0,
            typical_reactant_product=1.0):
        """Declare a reaction, optionally reversible.

        ``reactants`` and ``products`` are sequences of species names or indices.
        Total order on each side must be at most two.
        """
        r_idx = tuple(self.state.index(x) for x in reactants)
        p_idx = tuple(self.state.index(x) for x in products)

        for side, idx in (("reactants", r_idx), ("products", p_idx)):
            if len(idx) > 2:
                raise ValueError(
                    f"reaction {name!r} has {len(idx)} {side} (order {len(idx)}); "
                    "vex-rddme supports total order at most 2 on each side. Split it "
                    "into elementary steps."
                )

        for group in (r_idx, p_idx):
            for s in group:
                if self.state.species[s].inert:
                    raise ValueError(
                        f"reaction {name!r} involves species "
                        f"{self.state.species[s].name!r}, which is declared inert. An "
                        "inert crowder must appear in no reaction so that crowding "
                        "can be varied independently of the reacting species."
                    )

        if k_forward < 0 or k_reverse < 0:
            raise ValueError(
                f"reaction {name!r}: rate constants must be non-negative; got "
                f"k_forward={k_forward}, k_reverse={k_reverse}"
            )

        dnu = np.zeros(self.state.n_species, dtype=np.int64)
        for s in r_idx:
            dnu[s] -= 1
        for s in p_idx:
            dnu[s] += 1

        didx = (
            self.exclusion.stoichiometry_offset(dnu)
            if self.exclusion is not None and self.exclusion.uses_table
            else 0
        )

        rxn = Reaction(
            name=str(name),
            reactants=r_idx,
            products=p_idx,
            k_forward=float(k_forward),
            k_reverse=float(k_reverse),
            dnu=dnu,
            didx_forward=int(didx),
        )

        check_reaction_saturation(
            rxn.name, rxn.k_forward, self.tau_s, typical_reactant_product, self.report
        )
        if rxn.is_reversible:
            check_reaction_saturation(
                f"{rxn.name} (reverse)", rxn.k_reverse, self.tau_s,
                typical_reactant_product, self.report,
            )

        self.reactions.append(rxn)
        self._pi_sum.append([0.0, 0.0])
        self._pi_count.append([0, 0])
        return len(self.reactions) - 1

    # ------------------------------------------------------------------- work

    def field_work(self, dnu, vox=None):
        """``sum_s dnu_s phi_s(v)``: the basis-field part of the reaction work.

        ``vox`` selects a subset of voxels, so the same expression serves both the
        per-voxel step (all voxels) and the detailed-balance check (synthetic count
        vectors attached to sampled voxels).
        """
        phi = self._phi if vox is None else self._phi[:, vox]
        return np.tensordot(np.asarray(dnu, dtype=np.float64), phi, axes=(0, 0))

    def feasible_mask(self, counts, dnu):
        """Where applying ``dnu`` is representable: reactants present, room for products.

        The table's removal lookup ``idx - stride_s`` only addresses the intended count
        vector when ``n_s >= 1``; with ``n_s == 0`` the subtraction borrows from a
        higher digit and lands on an unrelated composition. The step path never
        notices, because the propensity is zero wherever a reactant is missing. But
        anything that *averages* the work over voxels would silently average garbage.
        """
        counts = np.asarray(counts)
        mask = np.ones(counts.shape[1], dtype=bool)
        radix = getattr(self.exclusion, "radix", None)
        for s in range(counts.shape[0]):
            d = int(dnu[s])
            if d < 0:
                mask &= counts[s] >= -d
            elif d > 0 and radix is not None:
                mask &= counts[s] + d <= radix - 1
        return mask

    def exclusion_work(self, counts, dnu, didx):
        """``F(n + dnu) - F(n)`` per voxel, or zeros when exclusion is off.

        Returns exactly zero where the change is not representable (a reactant is
        absent, or a product would exceed the table radix). Zero is the honest value
        there: no reaction can occur, so no work is done. It also keeps the array safe
        to average, which a garbage lookup would not be.
        """
        if self.exclusion is None or self.exclusion.n_hard_core == 0:
            # Shape follows the supplied counts, not the lattice: the
            # detailed-balance check passes synthetic count vectors whose number of
            # columns is unrelated to the voxel count.
            return np.zeros(np.shape(counts)[1])

        work = self.exclusion.shifted_free_energy(counts, didx, dnu) - \
            self.exclusion.free_energy(counts)

        if self.exclusion.uses_table and np.issubdtype(np.asarray(counts).dtype, np.integer):
            work = np.where(self.feasible_mask(counts, dnu), work, 0.0)
        return work

    def reaction_work_at(self, composition, ridx=0):
        """Reaction work at one (possibly fractional) composition.

        The prediction for a crowding-shifted equilibrium is
        ``d ln K = -d(beta F_ex)`` evaluated at the *mean* composition. A mean is
        fractional, so it takes the elementwise path and the feasibility caveat above
        does not arise.
        """
        comp = np.asarray(composition, dtype=np.float64)
        if comp.ndim == 1:
            comp = comp[:, None]
        rxn = self.reactions[ridx]
        return float(self.exclusion_work(comp, rxn.dnu, rxn.didx_forward).mean()) \
            if self.exclusion is not None else 0.0

    def total_work(self, counts, dnu, didx, vox=None):
        """``dPhi`` for applying ``dnu``, field plus exclusion."""
        return self.field_work(dnu, vox=vox) + self.exclusion_work(counts, dnu, didx)

    @staticmethod
    def acceptance(dphi):
        """``pi = min(1, exp(-dPhi))``: the Metropolis rule Fröhner-Noé prescribes."""
        return np.minimum(1.0, np.exp(np.minimum(-np.asarray(dphi), 700.0)))

    # ------------------------------------------------------- detailed balance

    def detailed_balance_residual(self, rng=None, n_samples=2000, from_pre_state=False):
        """Max ``|pi_f/pi_r - exp(-dPhi_f)|`` over randomised admissible states.

        ``from_pre_state=True`` evaluates the reverse work from ``n`` instead of
        ``n + dnu``, the incorrect formulation, so a test can confirm it fails.
        """
        rng = np.random.default_rng(0) if rng is None else rng
        worst = 0.0
        cap = self.state.occupancy_cap
        n_sp = self.state.n_species

        for rxn in self.reactions:
            if not rxn.is_reversible:
                continue
            # Sample states with room to move in both directions.
            counts = rng.integers(1, max(2, cap // max(2, n_sp)), size=(n_sp, n_samples))
            counts = counts.astype(np.int64)
            # Synthetic count vectors are attached to sampled voxels so the field
            # part of the work is evaluated with real psi values.
            vox = rng.integers(0, self.state.lattice.n_voxels, size=n_samples)
            over = counts.sum(axis=0) > cap - 2
            counts, vox = counts[:, ~over], vox[~over]
            if counts.shape[1] == 0:
                continue

            post = counts + rxn.dnu[:, None]
            keep = (post >= 0).all(axis=0) & (post.sum(axis=0) <= cap)
            counts, post, vox = counts[:, keep], post[:, keep], vox[keep]
            if counts.shape[1] == 0:
                continue

            dphi_f = self.total_work(counts, rxn.dnu, rxn.didx_forward, vox=vox)
            if from_pre_state:
                dphi_r = self.total_work(counts, -rxn.dnu, -rxn.didx_forward, vox=vox)
            else:
                dphi_r = self.total_work(post, -rxn.dnu, -rxn.didx_forward, vox=vox)

            pi_f = self.acceptance(dphi_f)
            pi_r = self.acceptance(dphi_r)
            # pi_f/pi_r == exp(-dPhi_f) for the Metropolis rule, which is what makes
            # the equilibrium ratio k_F/k_R * exp(-dPhi).
            lhs = pi_f / np.maximum(pi_r, 1e-300)
            rhs = np.exp(np.clip(-dphi_f, -700, 700))
            worst = max(worst, float(np.max(np.abs(lhs - rhs) / np.maximum(rhs, 1e-300))))
        return worst

    def work_antisymmetry_residual(self, counts):
        """Max ``|dbF_forward + dbF_reverse|`` with the reverse taken post-state.

        Zero to roundoff is what makes detailed balance exact rather than approximate.

        Evaluated on the **feasible set** only. Where the forward change is not
        representable the work is defined as zero (see :meth:`exclusion_work`), and
        zero is not the negative of the reverse work. Including those voxels would
        report a spurious residual for a state in which no reaction can occur anyway.
        Forward feasibility implies the reverse is feasible from the post-state, since
        the forward change is what created the products the reverse consumes.
        """
        worst = 0.0
        for rxn in self.reactions:
            post = counts + rxn.dnu[:, None]
            mask = self.feasible_mask(counts, rxn.dnu) if self.exclusion is not None \
                else np.ones(np.shape(counts)[1], dtype=bool)
            if not np.any(mask):
                continue
            fwd = self.exclusion_work(counts, rxn.dnu, rxn.didx_forward)
            rev = self.exclusion_work(post, -rxn.dnu, -rxn.didx_forward)
            worst = max(worst, float(np.max(np.abs((fwd + rev)[mask]))))
        return worst

    # -------------------------------------------------------------- one step

    def step(self, rng, step=None):
        """Fire every reaction once, netting forward and reverse.

        Order: works are computed from the state at entry, so all reactions in a
        step see the same configuration. This is the reaction half of the
        operator splitting; transport runs separately.
        """
        state = self.state
        counts = state.counts

        for ridx, rxn in enumerate(self.reactions):
            dphi_f = self.total_work(counts, rxn.dnu, rxn.didx_forward)
            pi_f = self.acceptance(dphi_f)

            lam_f = rxn.k_forward * self._reactant_product(counts, rxn.reactants) \
                * pi_f * self.tau_s
            n_f = rng.poisson(np.maximum(lam_f, 0.0))

            if rxn.is_reversible:
                post = counts + rxn.dnu[:, None]
                dphi_r = self.total_work(post, -rxn.dnu, -rxn.didx_forward)
                pi_r = self.acceptance(dphi_r)
                lam_r = rxn.k_reverse * self._reactant_product(counts, rxn.products) \
                    * pi_r * self.tau_s
                n_r = rng.poisson(np.maximum(lam_r, 0.0))
                self._accumulate(ridx, 1, pi_r, lam_r)
            else:
                n_r = np.zeros_like(n_f)

            self._accumulate(ridx, 0, pi_f, lam_f)

            # Net first, then cap. Capping each direction separately would consume
            # reactants the other direction is about to restore.
            net = n_f.astype(np.int64) - n_r.astype(np.int64)
            net = self._apply_feasibility(counts, rxn, net)

            counts = counts + rxn.dnu[:, None] * net[None, :]
            state.note_reaction(rxn.dnu * int(net.sum()))

        state.counts = counts
        return state

    @staticmethod
    def _reactant_product(counts, side):
        """Propensity factor: 1, ``n_R``, or ``n_A n_B``."""
        if len(side) == 0:
            return np.ones(counts.shape[1], dtype=np.float64)
        if len(side) == 1:
            return counts[side[0]].astype(np.float64)
        a, b = side
        if a == b:
            # A + A: ordered distinct pairs, so n(n-1).
            n = counts[a].astype(np.float64)
            return n * np.maximum(n - 1.0, 0.0)
        return counts[a].astype(np.float64) * counts[b].astype(np.float64)

    def _apply_feasibility(self, counts, rxn, net):
        """Clamp the net firing count so no species goes negative and the cap holds.

        Solved per voxel as an interval on ``net``: for each species,
        ``counts_s + dnu_s * net >= 0``, and ``sum_s counts_s + sum(dnu) * net <= cap``.
        """
        lo = np.full(net.shape, -np.inf)
        hi = np.full(net.shape, np.inf)

        for s in range(self.state.n_species):
            d = int(rxn.dnu[s])
            if d == 0:
                continue
            bound = counts[s].astype(np.float64) / abs(d)
            if d > 0:
                lo = np.maximum(lo, -bound)      # net >= -n_s/d
            else:
                hi = np.minimum(hi, bound)       # net <=  n_s/|d|

        dtot = int(rxn.dnu.sum())
        if dtot != 0:
            room = (self.state.occupancy_cap - counts.sum(axis=0)).astype(np.float64)
            if dtot > 0:
                hi = np.minimum(hi, room / dtot)
            else:
                lo = np.maximum(lo, room / dtot)

        return np.clip(net, np.ceil(lo), np.floor(hi)).astype(np.int64)

    # ------------------------------------------------------------ diagnostics

    def _accumulate(self, ridx, direction, pi, lam):
        active = lam > 0.0
        if np.any(active):
            self._pi_sum[ridx][direction] += float(np.sum(pi[active]))
            self._pi_count[ridx][direction] += int(np.count_nonzero(active))

    def mean_acceptance(self, ridx, direction=0):
        """Running mean acceptance factor for one reaction direction."""
        total, count = self._pi_sum[ridx][direction], self._pi_count[ridx][direction]
        return total / count if count else 1.0

    def __repr__(self) -> str:
        return f"ReactionSet({[r.name for r in self.reactions]})"
