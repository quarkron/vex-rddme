"""The loud-failure contract.

Every condition under which this package cannot deliver the requested physics
either raises or reports. Nothing is clamped, clipped, or substituted silently.

That is a deliberate inversion of what the production CUDA kernel does in two
places, and both inversions exist because the silent version costs real time to
diagnose:

* ``xi3`` at or above 1 is clamped there. Past that point the free energy is capped
  and exclusion stops repelling — the physics switches off in exactly the crowded
  regime the method exists for, and nothing says so.
* A per-direction hop probability sum above 1 is absorbed by clipping in the
  conditional-binomial split. Flux is lost, silently, and only in crowded runs.

Guards are grouped by when they can fire:

* **Construction** — sigma/h consistency, table radix, baseline CFL. Cheap, once.
* **Per step** — hop-probability sum, ``xi3`` saturation. The exclusion work grows
  as crowding develops, so a run can start valid and become invalid; these cannot
  be construction-time checks.
* **On demand** — mass conservation, detailed-balance residual. Used by tests and
  available to notebooks.

A library configuring logging is normally rude. It is done here because a guard
that does not reach the user is the failure mode this module exists to prevent;
pass ``attach_handler=False`` to opt out and route the ``vex_rddme`` logger yourself.
"""

from __future__ import annotations

import logging
import sys

import numpy as np

LOGGER_NAME = "vex_rddme"


class GuardViolation(AssertionError):
    """A guard failed. Subclasses AssertionError so tests read naturally."""


def get_logger(attach_handler=True):
    logger = logging.getLogger(LOGGER_NAME)
    if attach_handler and not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
        logger.addHandler(handler)
    # Set the level whether or not a handler was attached: an unset level inherits
    # root's WARNING, which would drop every info-level guard record on the floor
    # even for a caller who has wired up their own handler.
    if logger.level == logging.NOTSET:
        logger.setLevel(logging.INFO)
    return logger


class GuardReport:
    """Collects guard messages and makes sure they reach the log."""

    def __init__(self, attach_handler=True):
        self.logger = get_logger(attach_handler=attach_handler)
        self.records = []

    def info(self, tag, message):
        self.records.append(("INFO", tag, message))
        self.logger.info("[%s] %s", tag, message)

    def warn(self, tag, message):
        self.records.append(("WARNING", tag, message))
        self.logger.warning("[%s] %s", tag, message)

    def fail(self, tag, message):
        """Log, then raise. Logging first means a caught exception still left a trace."""
        self.records.append(("ERROR", tag, message))
        self.logger.error("[%s] %s", tag, message)
        raise GuardViolation(f"[{tag}] {message}")

    # ------------------------------------------------------------- inspection

    @property
    def warnings(self):
        return [(t, m) for lvl, t, m in self.records if lvl == "WARNING"]

    @property
    def infos(self):
        return [(t, m) for lvl, t, m in self.records if lvl == "INFO"]

    def has_warning(self, tag):
        return any(t == tag for t, _ in self.warnings)

    def summary(self):
        if not self.records:
            return "no guard messages"
        return "\n".join(f"{lvl:7s} [{tag}] {msg}" for lvl, tag, msg in self.records)

    def __repr__(self):
        n_warn = len(self.warnings)
        return f"GuardReport({len(self.records)} records, {n_warn} warning(s))"


# ---------------------------------------------------------------------------
# Construction-time guards
# ---------------------------------------------------------------------------


def check_sigma_voxel_consistency(species, voxel_nm, occupancy_cap, report):
    """Verify the diameters, voxel size, and cap admit a meaningful packing fraction.

    Each particle of species ``s`` adds ``dxi3 = (pi/6) sigma_s**3 / h**3`` to the
    local packing fraction. At ``h = 10 nm`` that is 0.0042 for a 2 nm sphere
    (``xi3 = 1`` at about 239 particles), 0.065 for 5 nm (about 15), and 0.524 for
    10 nm — a sphere as wide as the voxel is strictly one per voxel, and no
    crowding study is possible at that combination.

    Fails when ``cap`` particles of any single species would reach ``xi3 >= 1``,
    since that state is representable but physically meaningless.
    """
    h3 = float(voxel_nm) ** 3
    pi6 = np.pi / 6.0

    for sp in species:
        if sp.sigma_nm <= 0.0:
            continue
        dxi3 = pi6 * sp.sigma_nm ** 3 / h3
        max_n = int(np.floor(1.0 / dxi3)) if dxi3 > 0 else 0
        if occupancy_cap * dxi3 >= 1.0:
            report.fail(
                "sigma-voxel",
                f"species {sp.name!r} has diameter {sp.sigma_nm:g} nm in a "
                f"{voxel_nm:g} nm voxel, contributing dxi3 = {dxi3:.4g} per "
                f"particle. With occupancy_cap = {occupancy_cap} the packing "
                f"fraction reaches {occupancy_cap * dxi3:.3g} >= 1, which is not a "
                f"physical state. This diameter admits at most {max_n} particle(s) "
                f"per voxel. Either lower occupancy_cap to {max_n} or below, "
                f"enlarge the voxel (try voxel_nm >= "
                f"{sp.sigma_nm * (pi6 * occupancy_cap) ** (1 / 3):.1f}), or reduce "
                f"the diameter.",
            )

    max_xi3 = occupancy_cap * max(
        (pi6 * sp.sigma_nm ** 3 / h3 for sp in species if sp.sigma_nm > 0), default=0.0
    )
    report.info(
        "sigma-voxel",
        f"maximum attainable packing fraction is {max_xi3:.4g} "
        f"(occupancy_cap {occupancy_cap} of the largest species)",
    )
    return max_xi3


def report_occupancy_at_half_packing(exclusion, report):
    """Record the occupancy at which each species alone reaches ``xi3 = 0.5``."""
    table = exclusion.occupancy_at_half_packing()
    parts = []
    for name, n in table.items():
        parts.append(f"{name}: n/a (ideal)" if not np.isfinite(n) else f"{name}: {n:.1f}")
    report.info(
        "packing", "occupancy at xi3 = 0.5 per species — " + ", ".join(parts)
    )
    return table


def check_cap_binds_after_exclusion(max_xi3, occupancy_cap, report, threshold=0.3):
    """Warn when the occupancy cap, not the free energy, is limiting the density.

    There is a window in which a volume-exclusion study is meaningful:

    * ``max_xi3 >= 1`` — rejected by :func:`check_sigma_voxel_consistency`, because
      the packing fraction cannot reach one.
    * ``max_xi3`` small — a *full* voxel is still barely excluded, so the BMCSL
      insertion cost never becomes large and density is limited by the integer cap
      instead of by the physics. Occupancy then piles up against the cap, transport
      starts being rejected mechanically, and the measured behaviour is an artifact
      of the cap rather than of the free energy.

    The second case is a warning rather than a failure: it is a legitimate
    configuration for a transport-only study with ideal species, and only misleading
    when the point of the run is exclusion.
    """
    if max_xi3 <= 0.0:
        return max_xi3        # no hard-core species: the cap is pure bookkeeping

    # Upper end of the window. This duplicates the intent of
    # check_sigma_voxel_consistency but is reachable from the exclusion model,
    # which knows the increments and the cap without needing the voxel edge —
    # so an ExclusionModel built directly can never accept xi3 >= 1.
    if max_xi3 >= 1.0:
        report.fail(
            "cap-vs-exclusion",
            f"a voxel at the occupancy cap ({occupancy_cap}) would reach packing "
            f"fraction {max_xi3:.4g} >= 1, which is not a physical state: the BMCSL "
            f"free energy diverges there. Lower occupancy_cap to at most "
            f"{int(np.floor(occupancy_cap / max_xi3))}, reduce the diameters, or "
            f"enlarge the voxel.",
        )

    if max_xi3 < threshold:
        report.warn(
            "cap-vs-exclusion",
            f"a voxel at the occupancy cap ({occupancy_cap}) reaches a packing "
            f"fraction of only {max_xi3:.3g}, below {threshold}. Volume exclusion is "
            f"weak there, so the integer cap rather than the BMCSL free energy will "
            f"limit local density, and results will reflect the cap. Raise "
            f"occupancy_cap, increase the diameters, or reduce the voxel size so "
            f"that a full voxel is genuinely crowded.",
        )
    else:
        report.info(
            "cap-vs-exclusion",
            f"a voxel at the cap reaches packing fraction {max_xi3:.3g}: exclusion "
            f"is the binding constraint, as intended",
        )
    return max_xi3


def check_table_radix(radix, occupancy_cap, report):
    """Require ``radix > occupancy_cap + 1``.

    Insertion looks up ``idx + stride_s``, so a voxel already holding ``cap``
    particles of one species needs the index ``cap + 1`` to exist. With
    ``radix = cap + 2`` both that and the removal lookup stay in range.
    """
    if radix <= occupancy_cap + 1:
        report.fail(
            "table-radix",
            f"table radix {radix} is too small for occupancy_cap "
            f"{occupancy_cap}: the insertion lookup needs count "
            f"{occupancy_cap + 1} to be representable, so radix must exceed "
            f"occupancy_cap + 1 = {occupancy_cap + 1}. Use radix >= "
            f"{occupancy_cap + 2}.",
        )
    return True


def check_cfl(q, dim, report, D_um2_s=None, tau_s=None, voxel_nm=None):
    """Baseline per-direction hop probability against ``q <= 1/(2*dim)``.

    Necessary, not sufficient: the Bernoulli factor exceeds 1 for downhill moves,
    so the realised sum must also be checked every step. See
    :func:`check_hop_probability_sum`.
    """
    limit = 1.0 / (2.0 * dim)
    if not np.isfinite(q) or q < 0:
        report.fail("cfl", f"baseline hop probability q must be finite and non-negative; got {q}")
    if q > limit:
        detail = ""
        if None not in (D_um2_s, tau_s, voxel_nm):
            detail = (
                f" This came from D = {D_um2_s:g} um^2/s, tau = {tau_s:g} s, "
                f"h = {voxel_nm:g} nm. To satisfy the bound, use tau <= "
                f"{limit * (voxel_nm * 1e-3) ** 2 / D_um2_s:.3g} s "
                f"or a larger voxel."
            )
        report.fail(
            "cfl",
            f"baseline hop probability q = {q:.4g} exceeds the {dim}D limit "
            f"1/(2*dim) = {limit:.4g}. The per-direction probabilities plus the "
            f"stay probability could not sum to one.{detail}",
        )
    report.info("cfl", f"baseline hop probability q = {q:.4g} (limit {limit:.4g}, {dim}D)")
    return True


# ---------------------------------------------------------------------------
# Per-step guards
# ---------------------------------------------------------------------------


def check_xi3_saturation(xi3, report, threshold=0.999, step=None, lattice=None):
    """Report voxels whose packing fraction approaches 1.

    Not clamped-and-continued: past saturation the free energy stops responding to
    added particles, so exclusion is inactive while the run appears healthy.
    """
    xi3 = np.asarray(xi3)
    if xi3.size == 0:
        return 0.0
    worst = float(np.max(xi3))
    if worst >= threshold:
        n_over = int(np.count_nonzero(xi3 >= threshold))
        where = int(np.argmax(xi3))
        loc = f" at voxel {where}"
        if lattice is not None:
            loc += f" {lattice.coord(where)}"
        when = "" if step is None else f" at step {step}"
        report.fail(
            "xi3-saturation",
            f"packing fraction reached {worst:.6g} (threshold {threshold})"
            f"{loc}{when}; {n_over} voxel(s) are saturated. Above this the BMCSL "
            f"free energy stops responding to added particles, so volume exclusion "
            f"is effectively switched off. Lower the occupancy cap, reduce the "
            f"particle count, enlarge the voxel, or reduce diameters.",
        )
    return worst


def check_hop_probability_sum(p_sum, report, step=None, species_name=None, lattice=None):
    """Require the summed per-direction hop probability to stay at or below 1.

    ``B(u) = u/(exp(u) - 1)`` is unbounded as ``u -> -inf``, so a strongly downhill
    move can carry a per-direction probability above the baseline ``q``. When the
    sum exceeds 1 the conditional-binomial split has no probability left for later
    directions and clipping would silently discard flux.

    This must run **every step**: the exclusion contribution grows as crowding
    develops, so a run that satisfies the construction-time CFL bound can violate
    this one thousands of steps later.
    """
    p_sum = np.asarray(p_sum)
    if p_sum.size == 0:
        return 0.0
    worst = float(np.max(p_sum))
    if worst > 1.0:
        where = int(np.argmax(p_sum))
        loc = f" at voxel {where}"
        if lattice is not None:
            loc += f" {lattice.coord(where)}"
        who = "" if species_name is None else f" for species {species_name!r}"
        when = "" if step is None else f" at step {step}"
        report.fail(
            "hop-probability-sum",
            f"summed per-direction hop probability reached {worst:.6g} > 1"
            f"{who}{loc}{when}. The Bernoulli factor exceeds 1 for downhill moves, "
            f"so the baseline CFL bound is not sufficient; the work has grown until "
            f"the multinomial split has no probability left. Clipping here would "
            f"silently discard flux, so this is a failure. Reduce tau, reduce the "
            f"field gradient, or reduce crowding.",
        )
    return worst


def suggest_tau(D_um2_s, voxel_nm, dim, eta_max, margin=4.0):
    """Largest timestep that keeps the hop-probability budget under 1.

    The binding constraint with volume exclusion is not the bare CFL bound but the
    downhill Bernoulli factor. For a strongly downhill move ``B(u) -> |u|``, and the
    largest work a particle can shed is the excess chemical potential at the highest
    packing fraction it will meet, so::

        2 * dim * q * mu_ex(eta_max)  <=  1,     q = D tau / h^2

    ``margin`` divides the result, because ``eta_max`` is a *mean* expectation and
    fluctuations push local occupancy above it. A factor of 4 has been enough in
    practice; raise it if the probability-sum guard still fires.

    This exists because the constraint is easy to underestimate: at eta = 0.5,
    mu_ex is about 17 kT, so the admissible tau is an order of magnitude below what
    the bare CFL bound suggests.
    """
    eta_max = float(eta_max)
    if not 0.0 < eta_max < 1.0:
        raise ValueError(f"eta_max must lie in (0, 1); got {eta_max}")
    mu = float((8 * eta_max - 9 * eta_max ** 2 + 3 * eta_max ** 3) / (1 - eta_max) ** 3)
    worst = max(mu, 1.0)          # never suggest more than the bare CFL bound allows
    q_max = 1.0 / (2.0 * dim * worst)
    tau = q_max * voxel_nm ** 2 / (D_um2_s * 1.0e6)
    return tau / margin


def check_reaction_saturation(name, k, tau_s, typical_reactant_product, report, threshold=0.1):
    """Require the characteristic firing probability per step to stay small.

    ``k * <n_reactants> * tau`` is the expected firings per voxel per step. Once it
    approaches 1 the propensity saturates: raising ``k`` further stops raising the
    realised rate, so a rate rescale silently does nothing.
    """
    p = float(k) * float(typical_reactant_product) * float(tau_s)
    if p > threshold:
        report.warn(
            "reaction-saturation",
            f"reaction {name!r} has a characteristic firing probability of "
            f"{p:.4g} per voxel per step (k = {k:g}, tau = {tau_s:g} s, typical "
            f"reactant product {typical_reactant_product:g}), above the threshold "
            f"{threshold}. Near saturation the realised rate stops tracking k, so "
            f"rate changes have little effect. Reduce tau or reduce k.",
        )
    return p
