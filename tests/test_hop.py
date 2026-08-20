"""Transport: the Bernoulli factor, the multinomial split, and the two behaviours
that are validated here rather than demonstrated in a notebook.

``test_bare_hop_reproduces_boltzmann_profile`` is the layer-isolation test: without
it, a demonstration-1 notebook failure cannot be attributed to transport versus acceptance.
``test_removal_term_is_self_excluded`` pins the self-exclusion, whose absence would
otherwise only show up as a quantitative shift in demonstration 2.
"""

import numpy as np
import pytest

from vex_rddme import Lattice, Species, State
from vex_rddme.guards import GuardReport, GuardViolation
from vex_rddme.hop import Hop, bernoulli
from vex_rddme.vex import ExclusionModel


def quiet():
    return GuardReport(attach_handler=False)


def linear_ramp(lattice, axis=-1):
    """psi rising linearly from 0 to 1 along one axis."""
    n = lattice.shape[axis]
    ramp = np.arange(n, dtype=np.float64) / n
    return np.broadcast_to(ramp, lattice.shape).copy()


def build(
    shape=(32, 8),
    sigmas=(0.0,),
    gammas=((0.0,),),
    cap=10_000,
    voxel_nm=20.0,
    D=1.0,
    tau=1e-5,
    psi=None,
    exclusion=True,
):
    lat = Lattice(shape=shape, voxel_nm=voxel_nm)
    species = [
        Species(f"S{i}", sigma_nm=sig, gamma=np.asarray(g, dtype=float))
        for i, (sig, g) in enumerate(zip(sigmas, gammas))
    ]
    st = State(lat, species, occupancy_cap=cap)
    report = quiet()
    exc = None
    if exclusion and any(s > 0 for s in sigmas):
        exc = ExclusionModel(
            species, lat.voxel_volume_nm3, cap, report=report
        )
    if psi is None:
        psi = linear_ramp(lat)[None, ...] if len(gammas[0]) else None
    hop = Hop(lat, st, exc, psi, D_um2_s=D, tau_s=tau, report=report)
    return lat, st, hop, report


# ------------------------------------------------------------------ bernoulli


def test_bernoulli_is_exactly_one_at_zero():
    assert bernoulli(np.array([0.0]))[0] == 1.0


def test_bernoulli_series_is_continuous_across_the_threshold():
    # The two sample points straddle the branch and differ in u by 2e-9, so the
    # values legitimately differ by ~1e-9; the tolerance bounds the *branch*
    # discontinuity, which must be far smaller than that.
    eps = 1e-6
    below = float(bernoulli(np.array([eps * 0.999]))[0])
    above = float(bernoulli(np.array([eps * 1.001]))[0])
    assert below == pytest.approx(above, rel=1e-7)


def test_bernoulli_series_agrees_with_closed_form_at_the_branch():
    """Evaluate both branches at the *same* u to bound the discontinuity itself."""
    u = 1e-6
    series = 1.0 - 0.5 * u + u * u / 12.0
    closed = u / np.expm1(u)
    assert series == pytest.approx(closed, rel=1e-12)


def test_bernoulli_matches_the_closed_form_away_from_zero():
    u = np.array([-5.0, -1.0, -0.1, 0.1, 1.0, 5.0, 20.0])
    assert np.allclose(bernoulli(u), u / np.expm1(u))


def test_bernoulli_suppresses_uphill_and_amplifies_downhill():
    assert float(bernoulli(np.array([2.0]))[0]) < 1.0
    assert float(bernoulli(np.array([-2.0]))[0]) > 1.0


def test_bernoulli_is_finite_for_extreme_arguments():
    u = np.array([-1e4, -800.0, 800.0, 1e4, 1e300])
    out = bernoulli(u)
    assert np.all(np.isfinite(out))
    assert out[-1] == pytest.approx(0.0)
    assert out[2] == pytest.approx(0.0)


def test_bernoulli_ratio_gives_boltzmann():
    """B(u)/B(-u) == exp(-u): the identity that makes the stationary state e^{-phi}."""
    u = np.array([-3.0, -0.5, 0.5, 3.0])
    assert np.allclose(bernoulli(u) / bernoulli(-u), np.exp(-u))


# ---------------------------------------------------------------- setup / CFL


def test_cfl_violation_is_caught_at_construction():
    with pytest.raises(GuardViolation, match="exceeds the 2D limit"):
        build(D=100.0, tau=1e-3)


def test_per_species_diffusion_is_accepted():
    _, _, hop, _ = build(
        sigmas=(0.0, 0.0), gammas=((0.0,), (0.0,)), D=[1.0, 2.0]
    )
    assert hop.q[1] == pytest.approx(2.0 * hop.q[0])


def test_mismatched_diffusion_length_is_rejected():
    with pytest.raises(ValueError, match="scalar or one value per species"):
        build(sigmas=(0.0, 0.0), gammas=((0.0,), (0.0,)), D=[1.0, 2.0, 3.0])


def test_bad_psi_shape_is_rejected():
    lat = Lattice(shape=(8, 8), voxel_nm=20.0)
    st = State(lat, [Species("A", 0.0, np.zeros(1))], occupancy_cap=100)
    with pytest.raises(ValueError, match="psi must have shape"):
        Hop(lat, st, None, np.zeros((1, 5, 5)), 1.0, 1e-5, quiet())


def test_nonfinite_psi_is_rejected():
    lat = Lattice(shape=(8, 8), voxel_nm=20.0)
    st = State(lat, [Species("A", 0.0, np.zeros(1))], occupancy_cap=100)
    psi = np.zeros((1,) + lat.shape)
    psi[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        Hop(lat, st, None, psi, 1.0, 1e-5, quiet())


def test_psi_is_read_only_and_bit_identical_after_stepping():
    lat, st, hop, _ = build(gammas=((2.0,),))
    rng = np.random.default_rng(0)
    st.seed_uniform("S0", 2000, rng)
    st.record_initial()
    before = np.array(hop.psi, copy=True)
    for i in range(50):
        hop.step(rng, step=i)
    assert np.array_equal(hop.psi, before), "psi is an input and must never change"
    assert not hop.psi.flags.writeable


def test_caller_cannot_mutate_psi_through_the_original_array():
    lat = Lattice(shape=(8, 8), voxel_nm=20.0)
    st = State(lat, [Species("A", 0.0, np.zeros(1))], occupancy_cap=100)
    psi = np.zeros((1,) + lat.shape)
    hop = Hop(lat, st, None, psi, 1.0, 1e-5, quiet())
    psi[0, 0, 0] = 99.0                      # mutate the caller's array
    assert hop.psi[0, 0] == 0.0, "Hop must own a copy of psi"


# -------------------------------------------------------- zero-work behaviour


def test_uniform_occupancy_gives_isotropic_positive_work():
    """On a uniform lattice the exclusion work is the second difference of F.

    It is *not* zero: moving a particle between two equally-occupied voxels takes
    ``F(n+1) - 2F(n) + F(n-1) > 0`` because F is convex in the local density. That
    convexity is exactly why the system resists density fluctuations. What makes
    the uniform state stationary is that the work is the same in every direction,
    so there is no net flux.
    """
    lat, st, hop, _ = build(
        shape=(8, 8), sigmas=(5.0,), gammas=((0.0,),), cap=12, voxel_nm=20.0
    )
    n = 4
    st.set_counts("S0", np.full(lat.shape, n, dtype=np.int64))
    insert, remove = hop._exclusion_arrays(st.counts)

    exc = hop.exclusion
    F = lambda k: float(exc.free_energy(np.array([[k]]))[0])
    expected = (F(n + 1) - F(n)) - (F(n) - F(n - 1))
    assert expected > 0.0

    works = []
    for d in range(lat.n_dirs):
        u = lat.neighbour(insert[0].reshape(lat.shape), d) - remove[0].reshape(lat.shape)
        interior = u[lat.can_leave(d)]
        assert np.allclose(interior, expected, atol=1e-12)
        works.append(float(interior.flat[0]))
    # isotropic: identical in every direction, hence no net drift
    assert np.allclose(works, works[0], atol=1e-12)


def test_field_free_work_is_zero_for_an_ideal_species():
    """The genuinely-zero case: no diameter means no exclusion work at all."""
    lat, st, hop, _ = build(
        shape=(8, 8), sigmas=(0.0,), gammas=((0.0,),), cap=1000, voxel_nm=20.0
    )
    st.set_counts("S0", np.full(lat.shape, 4, dtype=np.int64))
    insert, remove = hop._exclusion_arrays(st.counts)
    assert np.allclose(insert, 0.0) and np.allclose(remove, 0.0)
    p = hop.probabilities(0, insert, remove)
    for d in range(lat.n_dirs):
        interior = p[d][lat.can_leave(d)]
        assert np.allclose(interior, hop.q[0], atol=1e-12)


def test_uniform_start_stays_uniform_without_field_or_gradient():
    """No field and no initial gradient: the mean profile must stay flat.

    Flatness is judged on the axis-projected profile, not per voxel. Per-voxel
    occupancy fluctuates (that is real physics, and exclusion makes it sub-Poisson
    rather than absent); projecting over the other axis averages those fluctuations
    down by sqrt(12) and leaves any *systematic* drift visible, which is what this
    test is actually about. Samples are taken every 25 steps because consecutive
    configurations are strongly correlated.
    """
    lat, st, hop, _ = build(
        shape=(12, 12), sigmas=(8.0,), gammas=((0.0,),), cap=20, voxel_nm=20.0
    )
    st.set_counts("S0", np.full(lat.shape, 4, dtype=np.int64))
    st.record_initial()
    rng = np.random.default_rng(1)
    acc = np.zeros(lat.shape[-1])
    n_avg = 0
    for i in range(3000):
        hop.step(rng, step=i)
        if i >= 500 and i % 25 == 0:
            acc += st.lattice_view("S0").sum(axis=0)
            n_avg += 1
    prof = acc / n_avg
    assert prof.mean() == pytest.approx(4.0 * lat.shape[0], rel=1e-9)
    assert prof.std() / prof.mean() < 0.02, (
        f"projected profile is not flat: {np.array2string(prof, precision=2)}"
    )
    st.check_mass()


# ------------------------------------------------------- the multinomial split


def test_partition_conserves_mass_exactly_over_many_steps():
    lat, st, hop, _ = build(shape=(16, 16), gammas=((2.0,),))
    rng = np.random.default_rng(2)
    st.seed_uniform("S0", 5000, rng)
    st.record_initial()
    for i in range(200):
        hop.step(rng, step=i)
        st.check_mass()
    assert int(st.totals()[0]) == 5000


def test_no_mass_crosses_a_boundary_in_3d():
    lat, st, hop, _ = build(shape=(8, 8, 8), gammas=((1.0,),), D=1.0, tau=1e-5)
    rng = np.random.default_rng(3)
    st.seed_uniform("S0", 3000, rng)
    st.record_initial()
    for i in range(150):
        hop.step(rng, step=i)
    st.check_mass()
    assert int(st.totals()[0]) == 3000


def test_boundary_probabilities_are_zero():
    lat, st, hop, _ = build(shape=(10, 6), gammas=((3.0,),))
    st.set_counts("S0", np.full(lat.shape, 5, dtype=np.int64))
    insert, remove = hop._exclusion_arrays(st.counts)
    p = hop.probabilities(0, insert, remove)
    for d in range(lat.n_dirs):
        assert np.all(p[d][~lat.can_leave(d)] == 0.0)


def test_probability_sum_guard_fires_when_work_is_too_steep():
    """A steep enough field pushes the downhill Bernoulli factor over the budget."""
    lat = Lattice(shape=(16, 4), voxel_nm=20.0)
    sp = [Species("A", 0.0, np.array([1.0]))]
    st = State(lat, sp, occupancy_cap=1000)
    psi = (np.arange(lat.shape[-1], dtype=float) * 40.0)
    psi = np.broadcast_to(psi, lat.shape).copy()[None, ...]
    hop = Hop(lat, st, None, psi, D_um2_s=1.0, tau_s=1e-5, report=quiet())
    st.set_counts("A", np.full(lat.shape, 3, dtype=np.int64))
    st.record_initial()
    with pytest.raises(GuardViolation, match="silently discard flux"):
        hop.step(np.random.default_rng(4), step=0)


# ------------------------------------------------- 5.8 barometric distribution


def test_bare_hop_reproduces_boltzmann_profile():
    """rho ~ exp(-phi) with exclusion and reactions off.

    Isolates transport. If this passes and demonstration 1 fails, the fault is in the
    acceptance layer, not here.
    """
    gamma = 2.0
    lat, st, hop, _ = build(
        shape=(32, 8), sigmas=(0.0,), gammas=((gamma,),), cap=100_000,
        D=1.0, tau=1e-5,
    )
    rng = np.random.default_rng(5)
    st.seed_uniform("S0", 40_000, rng)
    st.record_initial()

    burn, total = 6000, 14000
    acc = np.zeros(lat.shape[-1])
    n_avg = 0
    for i in range(total):
        hop.step(rng, step=i)
        if i >= burn:
            acc += st.lattice_view("S0").sum(axis=0)
            n_avg += 1
    measured = acc / n_avg
    measured = measured / measured.sum()

    psi_line = linear_ramp(lat)[0]
    predicted = np.exp(-gamma * psi_line)
    predicted = predicted / predicted.sum()

    rel = np.abs(measured - predicted) / predicted
    assert rel.max() < 0.05, (
        f"barometric profile off by up to {rel.max() * 100:.1f}%\n"
        f"measured  {np.array2string(measured, precision=4)}\n"
        f"predicted {np.array2string(predicted, precision=4)}"
    )
    st.check_mass()


def test_zero_coupling_stays_uniform_in_the_same_field():
    """The control for the test above: gamma = 0 must see no gradient."""
    lat, st, hop, _ = build(
        shape=(32, 8), sigmas=(0.0,), gammas=((0.0,),), cap=100_000, D=1.0, tau=1e-5
    )
    rng = np.random.default_rng(6)
    st.seed_uniform("S0", 20_000, rng)
    st.record_initial()
    acc = np.zeros(lat.shape[-1])
    n_avg = 0
    for i in range(6000):
        hop.step(rng, step=i)
        if i >= 2500:
            acc += st.lattice_view("S0").sum(axis=0)
            n_avg += 1
    prof = acc / n_avg
    assert prof.std() / prof.mean() < 0.03


# ------------------------------------------------------ 5.9 self-exclusion


def test_removal_term_is_self_excluded():
    """``remove[s][v] == F(n(v)) - F(n(v) - dxi_s)``, evaluated one particle down.

    The alternative, evaluating the removal at ``F(n(v))`` itself and so omitting
    the self-exclusion, would let a particle feel its own volume at the voxel it is
    leaving, and biases every hop out of an occupied voxel.
    """
    lat, st, hop, _ = build(
        shape=(6, 6), sigmas=(5.0, 3.0), gammas=((0.0,), (0.0,)), cap=10, voxel_nm=20.0
    )
    rng = np.random.default_rng(7)
    st.set_counts("S0", rng.integers(1, 4, size=lat.shape))
    st.set_counts("S1", rng.integers(1, 4, size=lat.shape))

    insert, remove = hop._exclusion_arrays(st.counts)
    exc = hop.exclusion
    F0 = exc.free_energy(st.counts)

    for s in (0, 1):
        dnu = np.zeros(2, dtype=np.int64)
        dnu[s] = -1
        F_minus = exc.shifted_free_energy(
            st.counts, exc.stoichiometry_offset(dnu), dnu
        )
        assert np.allclose(remove[s], F0 - F_minus, atol=1e-12)
        # and it is genuinely different from the no-self-exclusion alternative
        assert not np.allclose(remove[s], 0.0)


def test_insertion_term_is_the_destination_cost():
    lat, st, hop, _ = build(
        shape=(6, 6), sigmas=(5.0,), gammas=((0.0,),), cap=10, voxel_nm=20.0
    )
    st.set_counts("S0", np.full(lat.shape, 3, dtype=np.int64))
    insert, _ = hop._exclusion_arrays(st.counts)
    exc = hop.exclusion
    dnu = np.array([1], dtype=np.int64)
    expected = exc.shifted_free_energy(
        st.counts, exc.stoichiometry_offset(dnu), dnu
    ) - exc.free_energy(st.counts)
    assert np.allclose(insert[0], expected, atol=1e-12)


def test_ideal_species_has_no_exclusion_work():
    lat, st, hop, _ = build(
        shape=(6, 6), sigmas=(0.0, 5.0), gammas=((0.0,), (0.0,)), cap=10, voxel_nm=20.0
    )
    st.set_counts("S0", np.full(lat.shape, 2, dtype=np.int64))
    st.set_counts("S1", np.full(lat.shape, 2, dtype=np.int64))
    insert, remove = hop._exclusion_arrays(st.counts)
    assert np.allclose(insert[0], 0.0)
    assert np.allclose(remove[0], 0.0)
    assert not np.allclose(insert[1], 0.0)


def test_exclusion_pushes_particles_out_of_crowded_voxels():
    """A density spike must relax faster with exclusion on than off."""
    def spread(exclusion_on):
        lat = Lattice(shape=(16, 16), voxel_nm=20.0)
        sp = [Species("A", 9.0 if exclusion_on else 0.0, np.zeros(1))]
        st = State(lat, sp, occupancy_cap=15)
        report = quiet()
        exc = (
            ExclusionModel(sp, lat.voxel_volume_nm3, 15, report=report)
            if exclusion_on
            else None
        )
        hop = Hop(lat, st, exc, np.zeros((1,) + lat.shape), 1.0, 2e-6, report)
        blob = np.zeros(lat.shape, dtype=np.int64)
        blob[6:10, 6:10] = 10
        st.set_counts("A", blob)
        st.record_initial()
        rng = np.random.default_rng(8)
        for i in range(200):
            hop.step(rng, step=i)
        st.check_mass()
        return int(st.lattice_view("A")[6:10, 6:10].sum())

    assert spread(True) < spread(False), (
        "volume exclusion must accelerate the escape from a crowded region"
    )


# ------------------------------- absorbed from driftRDME_standalone.ipynb (10.1)
#
# The retired standalone notebook's first validated section ran three non-reacting
# species at gamma in {+2, 0, -2} simultaneously in one linear ramp. Reproduced here
# as a test so the content survives the notebook's retirement; the negative-gamma
# case in particular was not covered by the single-species tests above.


def test_three_couplings_in_one_field_each_follow_boltzmann():
    gammas = (+2.0, 0.0, -2.0)
    lat = Lattice(shape=(32, 8), voxel_nm=20.0)
    species = [
        Species(f"g{g:+.0f}", sigma_nm=0.0, gamma=np.array([g])) for g in gammas
    ]
    st = State(lat, species, occupancy_cap=200_000)
    psi = linear_ramp(lat)[None, ...]
    hop = Hop(lat, st, None, psi, D_um2_s=1.0, tau_s=1e-5, report=quiet())

    rng = np.random.default_rng(11)
    for sp in species:
        st.seed_uniform(sp.name, 20_000, rng)
    st.record_initial()

    burn, total = 6000, 14000
    acc = np.zeros((len(gammas), lat.shape[-1]))
    n_avg = 0
    for i in range(total):
        hop.step(rng, step=i)
        if i >= burn:
            for s in range(len(gammas)):
                acc[s] += st.lattice_view(s).sum(axis=0)
            n_avg += 1

    psi_line = linear_ramp(lat)[0]
    for s, g in enumerate(gammas):
        measured = acc[s] / n_avg
        measured = measured / measured.sum()
        predicted = np.exp(-g * psi_line)
        predicted = predicted / predicted.sum()
        rel = np.abs(measured - predicted) / predicted
        assert rel.max() < 0.06, (
            f"gamma={g:+.0f} profile off by up to {rel.max()*100:.1f}%\n"
            f"measured  {np.array2string(measured, precision=4)}\n"
            f"predicted {np.array2string(predicted, precision=4)}"
        )

    # The three species must be ordered: gamma>0 depleted from high psi, gamma<0
    # enriched there, gamma=0 flat.
    high = [float((acc[s] / n_avg)[-4:].sum()) for s in range(len(gammas))]
    low = [float((acc[s] / n_avg)[:4].sum()) for s in range(len(gammas))]
    ratio = [h / l for h, l in zip(high, low)]
    assert ratio[0] < ratio[1] < ratio[2], (
        f"high/low occupancy ratios should increase with decreasing gamma; got {ratio}"
    )
    st.check_mass()
