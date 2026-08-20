"""Reversible reactions: declaration, propensities, acceptance, detailed balance.

The load-bearing test here is ``test_pre_state_reverse_breaks_detailed_balance``. The
post-state formulation makes the forward and reverse works exact negatives, so
detailed balance holds identically; the pre-state formulation is a plausible-looking
alternative that fails by a couple of percent. Without a test that the *wrong*
version fails, the right version passing proves nothing.
"""

import numpy as np
import pytest

from vex_rddme import Simulation, Species
from vex_rddme.guards import GuardReport
from vex_rddme.react import ReactionSet


def build(
    shape=(12, 12),
    sigmas=(5.0, 5.0, 6.3),
    gammas=None,
    cap=24,
    voxel_nm=20.0,
    tau=1e-5,
    psi=None,
    exclusion=True,
    seed=0,
    names=("A", "B", "C"),
):
    n_bases = 1 if gammas is None else len(gammas[0])
    gammas = gammas if gammas is not None else tuple((0.0,) for _ in sigmas)
    species = [
        Species(n, sigma_nm=s, gamma=np.asarray(g, dtype=float))
        for n, s, g in zip(names, sigmas, gammas)
    ]
    if psi is None:
        psi = np.zeros((n_bases,) + shape)
    return Simulation(
        shape=shape, voxel_nm=voxel_nm, species=species, occupancy_cap=cap,
        psi=psi, D_um2_s=1.0, tau_s=tau, exclusion=exclusion, seed=seed,
        attach_log_handler=False,
    )


# ------------------------------------------------------------- declaration


def test_first_and_second_order_are_accepted():
    sim = build()
    sim.add_reaction("decay", ["A"], ["B"], 1.0)
    sim.add_reaction("assoc", ["A", "B"], ["C"], 1.0, 1.0)
    assert len(sim.reactions.reactions) == 2
    assert sim.reactions.reactions[0].order_forward == 1
    assert sim.reactions.reactions[1].order_forward == 2


def test_order_three_is_rejected_with_the_name_and_order():
    sim = build(sigmas=(5.0, 5.0, 6.3, 5.0), names=("A", "B", "C", "D"))
    with pytest.raises(ValueError) as exc:
        sim.add_reaction("ternary", ["A", "B", "C"], ["D"], 1.0)
    msg = str(exc.value)
    assert "'ternary'" in msg and "3 reactants" in msg and "order 3" in msg


def test_order_three_on_the_product_side_is_rejected():
    sim = build(sigmas=(5.0, 5.0, 6.3, 5.0), names=("A", "B", "C", "D"))
    with pytest.raises(ValueError, match="3 products"):
        sim.add_reaction("split", ["A"], ["B", "C", "D"], 1.0)


def test_negative_rate_constants_are_rejected():
    sim = build()
    with pytest.raises(ValueError, match="must be non-negative"):
        sim.add_reaction("bad", ["A"], ["B"], -1.0)
    with pytest.raises(ValueError, match="must be non-negative"):
        sim.add_reaction("bad", ["A"], ["B"], 1.0, -1.0)


def test_inert_crowder_cannot_appear_in_a_reaction():
    species = [
        Species("A", 5.0, np.zeros(1)),
        Species("B", 5.0, np.zeros(1)),
        Species("X", 6.0, np.zeros(1), inert=True),
    ]
    sim = Simulation(
        shape=(8, 8), voxel_nm=20.0, species=species, occupancy_cap=20,
        psi=np.zeros((1, 8, 8)), tau_s=1e-5, seed=0, attach_log_handler=False,
    )
    with pytest.raises(ValueError, match="declared inert"):
        sim.add_reaction("bad", ["A", "X"], ["B"], 1.0)


def test_reversibility_flag_follows_the_reverse_rate():
    sim = build()
    i = sim.add_reaction("one_way", ["A"], ["B"], 1.0)
    j = sim.add_reaction("two_way", ["A"], ["B"], 1.0, 0.5)
    assert not sim.reactions.reactions[i].is_reversible
    assert sim.reactions.reactions[j].is_reversible


def test_stoichiometry_vector_is_correct():
    sim = build()
    sim.add_reaction("assoc", ["A", "B"], ["C"], 1.0, 1.0)
    assert sim.reactions.reactions[0].dnu.tolist() == [-1, -1, 1]


# ------------------------------------------------------------- propensities


def test_propensity_forms():
    sim = build()
    counts = np.array([[3, 0], [4, 5], [1, 2]], dtype=np.int64)
    rs = sim.reactions
    assert rs._reactant_product(counts, ()).tolist() == [1.0, 1.0]
    assert rs._reactant_product(counts, (0,)).tolist() == [3.0, 0.0]
    assert rs._reactant_product(counts, (0, 1)).tolist() == [12.0, 0.0]
    # A + A uses ordered distinct pairs: n(n-1)
    assert rs._reactant_product(counts, (1, 1)).tolist() == [12.0, 20.0]


def test_same_species_dimerisation_cannot_go_negative():
    sim = build()
    counts = np.array([[0], [1], [0]], dtype=np.int64)
    assert sim.reactions._reactant_product(counts, (1, 1)).tolist() == [0.0]


# --------------------------------------------------------------- acceptance


def test_acceptance_is_one_for_zero_or_negative_work():
    assert ReactionSet.acceptance(np.array([0.0]))[0] == 1.0
    assert ReactionSet.acceptance(np.array([-3.0]))[0] == 1.0


def test_acceptance_decays_for_positive_work():
    pi = ReactionSet.acceptance(np.array([1.0, 5.0]))
    assert pi[0] == pytest.approx(np.exp(-1.0))
    assert pi[1] == pytest.approx(np.exp(-5.0))


def test_acceptance_is_finite_for_extreme_work():
    pi = ReactionSet.acceptance(np.array([1e6, -1e6]))
    assert np.all(np.isfinite(pi))
    assert pi[0] == pytest.approx(0.0)
    assert pi[1] == 1.0


def test_mean_acceptance_is_retrievable():
    sim = build()
    sim.add_reaction("assoc", ["A", "B"], ["C"], 0.05, 0.05)
    sim.seed_uniform("A", 300)
    sim.seed_uniform("B", 300)
    sim.record_initial()
    for _ in range(60):
        sim.step()
    fwd = sim.reactions.mean_acceptance(0, 0)
    rev = sim.reactions.mean_acceptance(0, 1)
    assert 0.0 < fwd <= 1.0 and 0.0 < rev <= 1.0


# ----------------------------------------------------------- detailed balance


def test_forward_and_reverse_exclusion_works_are_exact_negatives():
    sim = build()
    sim.add_reaction("assoc", ["A", "B"], ["C"], 1.0, 1.0)
    sim.seed_uniform("A", 400)
    sim.seed_uniform("B", 400)
    residual = sim.reactions.work_antisymmetry_residual(sim.state.counts)
    assert residual < 1e-12, f"antisymmetry residual {residual:.3e}"


def test_detailed_balance_holds_with_post_state_reverse():
    sim = build()
    sim.add_reaction("assoc", ["A", "B"], ["C"], 1.0, 1.0)
    residual = sim.reactions.detailed_balance_residual(np.random.default_rng(0))
    assert residual < 1e-12, f"detailed-balance residual {residual:.3e}"


def test_pre_state_reverse_breaks_detailed_balance():
    """The incorrect formulation must fail, or the correct one proves nothing.

    Evaluating the reverse work from ``n`` rather than ``n + dnu`` gives
    ``F(n - dnu) - F(n)``, which is not the negative of the forward work because
    the free energy is not linear in the counts.
    """
    sim = build()
    sim.add_reaction("assoc", ["A", "B"], ["C"], 1.0, 1.0)
    good = sim.reactions.detailed_balance_residual(np.random.default_rng(0))
    bad = sim.reactions.detailed_balance_residual(
        np.random.default_rng(0), from_pre_state=True
    )
    assert good < 1e-12
    assert bad > 1e-6, (
        f"the pre-state formulation should visibly break detailed balance, but its "
        f"residual was only {bad:.3e}"
    )
    assert bad > good * 1e6


def test_detailed_balance_holds_without_exclusion_too():
    sim = build(exclusion=False)
    sim.add_reaction("assoc", ["A", "B"], ["C"], 1.0, 1.0)
    assert sim.reactions.detailed_balance_residual(np.random.default_rng(1)) < 1e-12


# ------------------------------------------------------------- equilibrium


def test_equilibrium_ratio_equals_rate_ratio_without_exclusion():
    """With exclusion and field off, ``<n_C> / <n_A n_B> == k_F / k_R`` exactly.

    The ratio uses ``<n_A n_B>`` rather than ``<n_A><n_B>``: A and B are correlated
    through the conservation law, so the product of means is not the mean of the
    product and only the latter satisfies the detailed-balance relation.
    """
    # k must be large enough that the reaction actually equilibrates in the run:
    # at k_f = 0.4 the per-voxel firing rate is 7e-5/step and equilibration takes
    # ~25,000 steps. Scaling both rates by 100 keeps the ratio (hence Q) identical
    # and brings equilibration down to ~250 steps.
    k_f, k_r = 40.0, 200.0
    sim = build(shape=(16, 16), exclusion=False, cap=200, tau=2e-5, seed=3)
    sim.add_reaction("assoc", ["A", "B"], ["C"], k_f, k_r)
    sim.seed_uniform("A", 1200)
    sim.seed_uniform("B", 1200)
    sim.record_initial()

    num = den = 0.0
    n = 0
    for i in range(6000):
        sim.step()
        if i >= 2500 and i % 5 == 0:
            c = sim.state.counts
            num += c[2].mean()
            den += (c[0].astype(float) * c[1].astype(float)).mean()
            n += 1
    Q = (num / n) / (den / n)
    assert Q == pytest.approx(k_f / k_r, rel=0.08), (
        f"measured Q = {Q:.4f}, expected k_F/k_R = {k_f / k_r:.4f}"
    )


def test_local_equilibrium_ratio_follows_the_field():
    """Fröhner-Noé acceptance shifts the local equilibrium by exp(-dPhi).

    A scaled-down version of the demonstration-1 notebook check, kept here so a regression is
    caught by the test suite rather than only by running a notebook.
    """
    k_f, k_r, g = 1.0, 1.0, 1.5
    shape = (24, 6)
    ramp = np.arange(shape[-1], dtype=float) / shape[-1]
    psi = np.broadcast_to(ramp, shape).copy()[None, ...]
    sim = build(
        shape=shape, exclusion=False, cap=400, tau=2e-5, psi=psi, seed=4,
        gammas=((0.0,), (0.0,), (g,)),
    )
    sim.add_reaction("assoc", ["A", "B"], ["C"], k_f, k_r)
    sim.seed_uniform("A", 3000)
    sim.seed_uniform("B", 3000)
    sim.record_initial()

    num = np.zeros(shape[-1])
    den = np.zeros(shape[-1])
    n = 0
    for i in range(5000):
        sim.step()
        if i >= 2000 and i % 5 == 0:
            c = sim.state.counts.reshape((3,) + shape)
            num += c[2].mean(axis=0)
            den += (c[0].astype(float) * c[1].astype(float)).mean(axis=0)
            n += 1
    Q = (num / n) / (den / n)
    predicted = (k_f / k_r) * np.exp(-g * ramp)

    # Compare shapes rather than absolute values: normalise both to their means.
    rel = np.abs(Q / Q.mean() - predicted / predicted.mean())
    assert rel.max() < 0.12, (
        f"local equilibrium ratio does not follow exp(-dPhi)\n"
        f"measured  {np.array2string(Q / Q.mean(), precision=3)}\n"
        f"predicted {np.array2string(predicted / predicted.mean(), precision=3)}"
    )


# ---------------------------------------------------- feasibility and capping


def test_reactions_never_drive_a_count_negative():
    sim = build(shape=(8, 8), exclusion=False, cap=100, tau=5e-5, seed=5)
    sim.add_reaction("fast", ["A", "B"], ["C"], 5.0e3, 5.0e3)
    sim.seed_uniform("A", 200)
    sim.seed_uniform("B", 200)
    sim.record_initial()
    for _ in range(80):
        sim.step()
        assert np.all(sim.state.counts >= 0), "a reaction drove a count negative"
    sim.state.check_mass()


def test_net_capping_when_both_directions_fire():
    """The cap applies to the net, not to each direction separately."""
    sim = build(shape=(6, 6), exclusion=False, cap=60, tau=5e-5, seed=6)
    sim.add_reaction("assoc", ["A", "B"], ["C"], 3.0e3, 3.0e3)
    # A single voxel holding exactly one A and one B: at most one net forward firing
    # is possible, however many the two Poisson draws propose.
    counts = np.zeros((3, sim.lattice.n_voxels), dtype=np.int64)
    counts[0, 0] = 1
    counts[1, 0] = 1
    sim.state.counts = counts
    sim.state.record_initial()
    rs = sim.reactions
    rxn = rs.reactions[0]
    for proposed in (-5, -1, 0, 1, 5):
        net = rs._apply_feasibility(counts, rxn, np.full(sim.lattice.n_voxels, proposed))
        after = counts + rxn.dnu[:, None] * net[None, :]
        assert np.all(after >= 0), f"proposed {proposed} produced negative counts"
        assert np.all(after.sum(axis=0) <= sim.state.occupancy_cap)
        assert net[0] <= 1, f"proposed {proposed} allowed more than one net firing"


def test_reactions_respect_the_occupancy_cap():
    """A product-increasing net firing is limited by the room in the voxel."""
    sim = build(shape=(4, 4), exclusion=False, cap=4, tau=5e-5, seed=7)
    # 2 A -> 3 C would raise occupancy; cap must bound the net.
    sim.add_reaction("expand", ["A"], ["B", "C"], 10.0)
    counts = np.zeros((3, sim.lattice.n_voxels), dtype=np.int64)
    counts[0, 0] = 4              # voxel full already
    sim.state.counts = counts
    sim.state.record_initial()
    rxn = sim.reactions.reactions[0]
    net = sim.reactions._apply_feasibility(
        counts, rxn, np.full(sim.lattice.n_voxels, 4)
    )
    after = counts + rxn.dnu[:, None] * net[None, :]
    assert np.all(after.sum(axis=0) <= 4)
    assert np.all(after >= 0)


def test_mass_changes_only_by_stoichiometry():
    sim = build(shape=(10, 10), exclusion=False, cap=60, tau=1e-4, seed=8)
    sim.add_reaction("assoc", ["A", "B"], ["C"], 1.0, 1.0)
    sim.seed_uniform("A", 500)
    sim.seed_uniform("B", 500)
    sim.record_initial()
    for _ in range(150):
        sim.step()
        sim.state.check_mass()
    # A + C and B + C are both conserved by this stoichiometry
    a, b, c = sim.state.totals()
    assert a + c == 500
    assert b + c == 500


def test_irreversible_reaction_runs_to_completion_one_way():
    sim = build(shape=(8, 8), exclusion=False, cap=100, tau=5e-5, seed=9)
    sim.add_reaction("decay", ["A"], ["B"], 2.0e3)
    sim.seed_uniform("A", 400)
    sim.record_initial()
    for _ in range(200):
        sim.step()
    a, b, _ = sim.state.totals()
    assert a + b == 400
    assert b > a, "an irreversible reaction should deplete its reactant"


# ------------------------------------------------------------- guard wiring


def test_saturation_warning_covers_both_directions():
    report = GuardReport(attach_handler=False)
    sim = build(tau=5e-5)
    sim.reactions.report = report
    sim.add_reaction("hot", ["A", "B"], ["C"], 1e4, 1e4, typical_reactant_product=4.0)
    msgs = [m for t, m in report.warnings if t == "reaction-saturation"]
    assert len(msgs) == 2, f"expected forward and reverse warnings, got {msgs}"
    assert any("reverse" in m for m in msgs)


def test_sigma_voxel_guard_is_wired_into_simulation():
    """The guard that needs both the lattice and the species must actually run."""
    from vex_rddme.guards import GuardViolation

    with pytest.raises(GuardViolation, match="not a physical state"):
        Simulation(
            shape=(8, 8), voxel_nm=10.0,
            species=[Species("big", 10.0, np.zeros(1))],
            occupancy_cap=4, psi=np.zeros((1, 8, 8)),
            seed=0, attach_log_handler=False,
        )


# ------------------------------------------------------- inert crowder (9.3)


def test_inert_crowder_contributes_to_exclusion():
    """A crowder must be felt by the reacting species even though it never reacts."""
    species = [
        Species("A", 5.0, np.zeros(1)),
        Species("B", 5.0, np.zeros(1)),
        Species("C", 6.3, np.zeros(1)),
        Species("X", 8.0, np.zeros(1), inert=True),
    ]
    sim = Simulation(
        shape=(10, 10), voxel_nm=20.0, species=species, occupancy_cap=20,
        psi=np.zeros((1, 10, 10)), tau_s=1e-5, seed=0, attach_log_handler=False,
    )
    sim.add_reaction("assoc", ["A", "B"], ["C"], 1.0, 1.0)
    sim.set_counts("A", np.full(sim.lattice.shape, 2, dtype=np.int64))
    sim.set_counts("B", np.full(sim.lattice.shape, 2, dtype=np.int64))

    bare = sim.reactions.exclusion_work(sim.state.counts, sim.reactions.reactions[0].dnu,
                                        sim.reactions.reactions[0].didx_forward).copy()
    sim.set_counts("X", np.full(sim.lattice.shape, 6, dtype=np.int64))
    crowded = sim.reactions.exclusion_work(sim.state.counts, sim.reactions.reactions[0].dnu,
                                           sim.reactions.reactions[0].didx_forward)
    assert not np.allclose(bare, crowded), (
        "adding an inert crowder must change the reaction's exclusion work"
    )


def test_crowder_packing_fraction_is_separable():
    species = [
        Species("A", 5.0, np.zeros(1)),
        Species("X", 8.0, np.zeros(1), inert=True),
    ]
    sim = Simulation(
        shape=(8, 8), voxel_nm=20.0, species=species, occupancy_cap=20,
        psi=np.zeros((1, 8, 8)), tau_s=1e-5, seed=0, attach_log_handler=False,
    )
    sim.set_counts("A", np.full(sim.lattice.shape, 3, dtype=np.int64))
    sim.set_counts("X", np.full(sim.lattice.shape, 5, dtype=np.int64))

    eta_x = sim.crowder_packing_fraction()
    eta_a = sim.packing_fraction(["A"])
    eta_all = sim.packing_fraction()
    expected_x = 5 * (np.pi / 6) * 8.0 ** 3 / 8000.0
    assert eta_x == pytest.approx(expected_x)
    assert eta_all == pytest.approx(eta_a + eta_x)
    assert sim.inert_indices == [1]


def test_crowder_packing_fraction_scales_with_crowder_count():
    species = [Species("A", 5.0, np.zeros(1)), Species("X", 8.0, np.zeros(1), inert=True)]
    etas = []
    for n_x in (2, 4, 8):
        sim = Simulation(
            shape=(8, 8), voxel_nm=20.0, species=species, occupancy_cap=20,
            psi=np.zeros((1, 8, 8)), tau_s=1e-5, seed=0, attach_log_handler=False,
        )
        sim.set_counts("X", np.full(sim.lattice.shape, n_x, dtype=np.int64))
        etas.append(sim.crowder_packing_fraction())
    assert etas[1] == pytest.approx(2 * etas[0])
    assert etas[2] == pytest.approx(4 * etas[0])


def test_crowder_packing_fraction_requires_a_crowder():
    sim = build()
    with pytest.raises(RuntimeError, match="no species is declared inert"):
        sim.crowder_packing_fraction()


# ------------------------- infeasible-voxel work must not pollute diagnostics


def test_exclusion_work_is_zero_where_the_reaction_is_infeasible():
    """The table's removal lookup is only valid where the reactant is present.

    With n_s == 0 the index subtraction borrows from a higher digit and lands on an
    unrelated composition. The step path never notices, because the propensity is zero
    there. But averaging the work over voxels would silently average garbage, which
    is exactly how a crowding-shift prediction came out as -5290 instead of +0.3.
    """
    sim = build(shape=(6, 6), cap=24)
    sim.add_reaction("assoc", ["A", "B"], ["C"], 1.0, 1.0)
    rxn = sim.reactions.reactions[0]

    counts = np.zeros((3, sim.lattice.n_voxels), dtype=np.int64)
    counts[0, :10] = 3          # A present only in the first ten voxels
    counts[1, :10] = 3
    counts[2, :] = 1

    work = sim.reactions.exclusion_work(counts, rxn.dnu, rxn.didx_forward)
    assert np.all(np.isfinite(work))
    assert np.all(work[10:] == 0.0), "infeasible voxels must contribute zero work"
    assert np.all(work[:10] != 0.0), "feasible voxels must carry a real work"
    # and the mean is now a usable number rather than dominated by garbage
    assert abs(float(work.mean())) < 10.0


def test_feasible_mask_requires_reactants_and_product_room():
    sim = build(shape=(4, 4), cap=24)
    sim.add_reaction("assoc", ["A", "B"], ["C"], 1.0, 1.0)
    rxn = sim.reactions.reactions[0]
    counts = np.zeros((3, sim.lattice.n_voxels), dtype=np.int64)
    counts[0, 0] = 1
    counts[1, 0] = 1            # feasible
    counts[0, 1] = 1
    counts[1, 1] = 0            # B missing -> infeasible
    counts[0, 2] = 0            # A missing -> infeasible
    mask = sim.reactions.feasible_mask(counts, rxn.dnu)
    assert bool(mask[0]) and not bool(mask[1]) and not bool(mask[2])


def test_reaction_work_at_a_fractional_composition_is_finite():
    sim = build(shape=(6, 6), cap=24)
    sim.add_reaction("assoc", ["A", "B"], ["C"], 1.0, 1.0)
    w = sim.reactions.reaction_work_at(np.array([3.2, 3.1, 1.4]))
    assert np.isfinite(w)
    assert abs(w) < 20.0


def test_crowding_raises_the_association_work_penalty_monotonically():
    """Adding crowder must make the reaction's exclusion work more favourable.

    A + B -> C at conserved volume removes one exclusion centre, so more crowding
    means a more negative work, hence a larger equilibrium constant. This is the
    Minton mechanism, checked on the work directly rather than through a long run.
    """
    sA = sB = 6.0
    sC = (sA ** 3 + sB ** 3) ** (1 / 3)
    species = [
        Species("A", sA, np.zeros(1)),
        Species("B", sB, np.zeros(1)),
        Species("C", sC, np.zeros(1)),
        Species("X", 7.0, np.zeros(1), inert=True),
    ]
    works = []
    for n_x in (0.0, 2.0, 4.0, 6.0):
        sim = Simulation(
            shape=(6, 6), voxel_nm=20.0, species=species, occupancy_cap=24,
            psi=np.zeros((1, 6, 6)), tau_s=1e-6, seed=0, attach_log_handler=False,
        )
        sim.add_reaction("assoc", ["A", "B"], ["C"], 1.0, 1.0)
        works.append(sim.reactions.reaction_work_at(np.array([3.0, 3.0, 1.0, n_x])))
    assert all(np.isfinite(works))
    assert works == sorted(works, reverse=True), (
        f"the association work should fall monotonically with crowding; got {works}"
    )
    assert works[-1] < works[0], "crowding must favour association"


def test_antisymmetry_is_claimed_only_on_the_feasible_set():
    """Outside the feasible set the work is defined as zero, not as a negation.

    Documented as a test so the scope of the antisymmetry claim is explicit: it holds
    exactly where a reaction can actually occur, which is everywhere the dynamics
    looks.
    """
    sim = build(shape=(6, 6), cap=24)
    sim.add_reaction("assoc", ["A", "B"], ["C"], 1.0, 1.0)
    rxn = sim.reactions.reactions[0]

    counts = np.zeros((3, sim.lattice.n_voxels), dtype=np.int64)
    counts[0, :8] = 3
    counts[1, :8] = 3            # feasible only in the first eight voxels
    counts[2, :] = 1

    mask = sim.reactions.feasible_mask(counts, rxn.dnu)
    assert mask[:8].all() and not mask[8:].any()

    # exact on the feasible set...
    assert sim.reactions.work_antisymmetry_residual(counts) < 1e-12
    # ...and the raw arrays do not cancel outside it, which is why the mask is needed
    post = counts + rxn.dnu[:, None]
    fwd = sim.reactions.exclusion_work(counts, rxn.dnu, rxn.didx_forward)
    rev = sim.reactions.exclusion_work(post, -rxn.dnu, -rxn.didx_forward)
    assert np.abs((fwd + rev)[8:]).max() > 0.0
