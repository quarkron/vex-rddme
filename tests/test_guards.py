"""The loud-failure contract.

Each guard corresponds to a way the physics dies quietly. The tests check not only
that a violation is caught, but that the message names the value, the limit, and
the knob that fixes it. A guard the reader cannot act on is barely better than
silence.
"""

import logging

import numpy as np
import pytest

from vex_rddme import Lattice, Species
from vex_rddme.guards import (
    GuardReport,
    GuardViolation,
    check_cfl,
    check_hop_probability_sum,
    check_reaction_saturation,
    check_sigma_voxel_consistency,
    check_table_radix,
    check_xi3_saturation,
    report_occupancy_at_half_packing,
)
from vex_rddme.vex import ExclusionModel


def quiet():
    return GuardReport(attach_handler=False)


# --------------------------------------------------------------- GuardReport


def test_guard_violation_is_an_assertion_error():
    assert issubclass(GuardViolation, AssertionError)


def test_failures_reach_the_log_before_raising(caplog):
    """A caught exception must still have left a trace."""
    report = quiet()
    with caplog.at_level(logging.INFO, logger="vex_rddme"):
        with pytest.raises(GuardViolation):
            report.fail("demo", "something specific went wrong")
    assert any(
        "something specific went wrong" in rec.getMessage() for rec in caplog.records
    )
    assert any(rec.levelno == logging.ERROR for rec in caplog.records)


def test_warnings_reach_the_log_and_are_recorded(caplog):
    report = quiet()
    with caplog.at_level(logging.INFO, logger="vex_rddme"):
        report.warn("demo", "a degraded but continuing condition")
    assert report.has_warning("demo")
    assert any(r.levelno == logging.WARNING for r in caplog.records)


def test_summary_lists_every_record():
    report = quiet()
    report.info("a", "one")
    report.warn("b", "two")
    text = report.summary()
    assert "one" in text and "two" in text
    assert "INFO" in text and "WARNING" in text


def test_empty_summary_is_explicit():
    assert quiet().summary() == "no guard messages"


# ------------------------------------------------------- sigma / voxel size


def test_sphere_as_wide_as_the_voxel_is_rejected():
    """At h = sigma, one sphere is 0.524 of the voxel: no crowding is possible."""
    report = quiet()
    species = [Species("big", 10.0, np.zeros(1))]
    with pytest.raises(GuardViolation) as exc:
        check_sigma_voxel_consistency(species, voxel_nm=10.0, occupancy_cap=4, report=report)
    msg = str(exc.value)
    assert "'big'" in msg
    assert "10 nm voxel" in msg
    assert "diameter 10 nm" in msg
    assert "at most 1 particle" in msg
    assert "voxel_nm >=" in msg          # actionable suggestion


def test_admissible_configuration_passes_and_reports_max_packing():
    report = quiet()
    species = [Species("small", 4.0, np.zeros(1))]
    max_xi3 = check_sigma_voxel_consistency(
        species, voxel_nm=20.0, occupancy_cap=8, report=report
    )
    expected = 8 * (np.pi / 6.0) * 4.0 ** 3 / 20.0 ** 3
    assert max_xi3 == pytest.approx(expected)
    assert max_xi3 < 1.0
    assert any(t == "sigma-voxel" for t, _ in report.infos)


def test_ideal_species_never_trips_the_sigma_guard():
    report = quiet()
    species = [Species("ideal", 0.0, np.zeros(1))]
    assert check_sigma_voxel_consistency(
        species, voxel_nm=1.0, occupancy_cap=1000, report=report
    ) == 0.0


def test_cap_reaching_unit_packing_is_rejected():
    """The comparison is `>= 1`: xi3 == 1 is not a physical state."""
    V = 1000.0
    # sigma sized so one particle occupies half the voxel; cap 3 reaches 1.5.
    sigma = (3.0 * V / np.pi) ** (1 / 3)
    dxi3 = (np.pi / 6.0) * sigma ** 3 / V
    assert dxi3 == pytest.approx(0.5)
    with pytest.raises(GuardViolation, match="not a physical state"):
        check_sigma_voxel_consistency(
            [Species("s", sigma, np.zeros(1))], voxel_nm=10.0, occupancy_cap=3, report=quiet()
        )


def test_configuration_just_below_unit_packing_passes_construction():
    """Construction only rejects `xi3 >= 1`; a hair under is legal by spec.

    Such a configuration is physically useless, but it is the per-step saturation
    guard's job to catch it. Construction cannot know how full voxels will get, so
    drawing the line anywhere above the representable maximum would reject valid
    setups. Documented here so the division of labour is deliberate rather than an
    accident of the comparison operator.
    """
    V = 1000.0
    sigma = (3.0 * V / np.pi) ** (1 / 3)          # dxi3 = 0.5
    max_xi3 = check_sigma_voxel_consistency(
        [Species("s", sigma, np.zeros(1))], voxel_nm=10.0, occupancy_cap=1, report=quiet()
    )
    assert max_xi3 == pytest.approx(0.5)

    # ...and the per-step guard is what stops it once voxels actually fill.
    with pytest.raises(GuardViolation, match="stops working"):
        check_xi3_saturation(np.array([0.9999]), quiet())


def test_half_packing_occupancy_is_reported_per_species():
    report = quiet()
    model = ExclusionModel(
        [Species("a", 4.0, np.zeros(1)), Species("ideal", 0.0, np.zeros(1))],
        voxel_volume_nm3=8000.0,
        occupancy_cap=8,
        report=quiet(),
    )
    table = report_occupancy_at_half_packing(model, report)
    assert np.isfinite(table["a"])
    assert table["ideal"] == float("inf")
    msg = dict(report.infos)["packing"]
    assert "a:" in msg and "ideal: n/a (ideal)" in msg


# -------------------------------------------------------------- table radix


def test_radix_must_exceed_cap_plus_one():
    report = quiet()
    with pytest.raises(GuardViolation) as exc:
        check_table_radix(radix=9, occupancy_cap=8, report=report)
    msg = str(exc.value)
    assert "radix 9" in msg
    assert "occupancy_cap 8" in msg
    assert "radix >= 10" in msg


def test_radix_equal_to_cap_plus_two_is_accepted():
    assert check_table_radix(radix=10, occupancy_cap=8, report=quiet()) is True


def test_exclusion_model_chooses_a_valid_radix():
    for cap in (1, 4, 8, 30):
        m = ExclusionModel(
            [Species("a", 2.0, np.zeros(1))],
            voxel_volume_nm3=8000.0,
            occupancy_cap=cap,
            report=quiet(),
        )
        assert check_table_radix(m.radix, cap, quiet()) is True


def test_insertion_index_stays_in_range_at_full_occupancy():
    """A voxel already holding `cap` particles must still admit the +1 lookup."""
    cap = 6
    m = ExclusionModel(
        [Species("a", 3.0, np.zeros(1)), Species("b", 3.0, np.zeros(1))],
        voxel_volume_nm3=8000.0,
        occupancy_cap=cap,
        report=quiet(),
    )
    counts = np.array([[cap], [0]], dtype=np.int64)
    idx = m.index(counts)
    for s in (0, 1):
        assert 0 <= int(idx[0]) + m.stride(s) < m.table_entries


# ---------------------------------------------------------------------- CFL


@pytest.mark.parametrize("dim,limit", [(2, 0.25), (3, 1.0 / 6.0)])
def test_cfl_limit_depends_on_dimension(dim, limit):
    check_cfl(limit, dim, quiet())                      # exactly at the limit is fine
    with pytest.raises(GuardViolation, match="exceeds the"):
        check_cfl(limit * 1.01, dim, quiet())


def test_cfl_failure_reports_the_physical_inputs():
    report = quiet()
    with pytest.raises(GuardViolation) as exc:
        check_cfl(0.45, dim=2, report=report, D_um2_s=0.9, tau_s=50e-6, voxel_nm=10.0)
    msg = str(exc.value)
    assert "0.45" in msg
    assert "D = 0.9" in msg
    assert "tau = 5e-05" in msg
    assert "h = 10 nm" in msg
    assert "use tau <=" in msg


def test_cfl_matches_the_lattice_limit():
    for shape in [(8, 8), (8, 8, 8)]:
        lat = Lattice(shape=shape, voxel_nm=10.0)
        check_cfl(lat.cfl_limit(), lat.dim, quiet())


def test_cfl_rejects_nonsense_q():
    for bad in (-0.1, np.nan, np.inf):
        with pytest.raises(GuardViolation, match="finite and non-negative"):
            check_cfl(bad, 2, quiet())


# ------------------------------------------------------- per-step: xi3, sums


def test_xi3_saturation_fails_loudly_with_location():
    lat = Lattice(shape=(4, 4), voxel_nm=10.0)
    xi3 = np.zeros(lat.n_voxels)
    xi3[lat.flat((2, 3))] = 0.9995
    with pytest.raises(GuardViolation) as exc:
        check_xi3_saturation(xi3, quiet(), step=1234, lattice=lat)
    msg = str(exc.value)
    assert "0.9995" in msg
    assert "(2, 3)" in msg
    assert "step 1234" in msg
    assert "stops working" in msg


def test_xi3_below_threshold_returns_the_worst_value():
    xi3 = np.array([0.1, 0.4, 0.2])
    assert check_xi3_saturation(xi3, quiet()) == pytest.approx(0.4)


def test_hop_probability_sum_above_one_is_a_failure_not_a_clip():
    lat = Lattice(shape=(4, 4), voxel_nm=10.0)
    p = np.full(lat.n_voxels, 0.6)
    p[5] = 1.4
    with pytest.raises(GuardViolation) as exc:
        check_hop_probability_sum(p, quiet(), step=77, species_name="A", lattice=lat)
    msg = str(exc.value)
    assert "1.4" in msg
    assert "'A'" in msg
    assert "step 77" in msg
    assert "silently discard flux" in msg


def test_hop_probability_sum_exactly_one_is_allowed():
    assert check_hop_probability_sum(np.array([1.0, 0.5]), quiet()) == pytest.approx(1.0)


def test_hop_probability_guard_explains_why_cfl_is_insufficient():
    with pytest.raises(GuardViolation) as exc:
        check_hop_probability_sum(np.array([1.2]), quiet())
    assert "Bernoulli factor exceeds 1" in str(exc.value)


# ------------------------------------------------------- reaction saturation


def test_reaction_saturation_warns_rather_than_failing():
    report = quiet()
    p = check_reaction_saturation("assoc", k=1000.0, tau_s=1e-3,
                                  typical_reactant_product=4.0, report=report)
    assert p == pytest.approx(4.0)
    assert report.has_warning("reaction-saturation")
    msg = dict(report.warnings)["reaction-saturation"]
    assert "'assoc'" in msg
    assert "stops tracking k" in msg


def test_reaction_below_threshold_is_silent():
    report = quiet()
    check_reaction_saturation("slow", k=1.0, tau_s=1e-5,
                              typical_reactant_product=2.0, report=report)
    assert not report.has_warning("reaction-saturation")


# ------------------------------------------------- timestep advisory


def test_suggest_tau_is_tighter_than_cfl_at_high_packing():
    """With exclusion the downhill Bernoulli factor, not CFL, sets the timestep."""
    from vex_rddme.guards import suggest_tau

    voxel, D, dim = 20.0, 1.0, 2
    cfl_tau = (1.0 / (2 * dim)) * voxel ** 2 / (D * 1e6)
    assert suggest_tau(D, voxel, dim, eta_max=0.5) < cfl_tau / 10, (
        "at eta = 0.5 the excess chemical potential is ~17 kT, so the admissible "
        "timestep should be an order of magnitude below the CFL bound"
    )


def test_suggest_tau_never_exceeds_the_cfl_bound():
    from vex_rddme.guards import suggest_tau

    voxel, D = 20.0, 1.0
    for dim in (2, 3):
        cfl_tau = (1.0 / (2 * dim)) * voxel ** 2 / (D * 1e6)
        assert suggest_tau(D, voxel, dim, eta_max=1e-6) <= cfl_tau + 1e-30


def test_suggest_tau_scaling():
    from vex_rddme.guards import suggest_tau

    base = suggest_tau(1.0, 20.0, 2, 0.3)
    assert suggest_tau(1.0, 40.0, 2, 0.3) == pytest.approx(4.0 * base)   # h^2
    assert suggest_tau(2.0, 20.0, 2, 0.3) == pytest.approx(base / 2.0)   # 1/D
    assert suggest_tau(1.0, 20.0, 2, 0.5) < base                          # 1/mu_ex


def test_suggest_tau_rejects_impossible_packing():
    from vex_rddme.guards import suggest_tau

    for bad in (0.0, 1.0, 1.5, -0.2):
        with pytest.raises(ValueError, match=r"eta_max must lie in \(0, 1\)"):
            suggest_tau(1.0, 20.0, 2, bad)


def test_suggested_tau_actually_keeps_a_run_inside_the_budget():
    """End-to-end: a run at the suggested timestep must not trip the guard."""
    from vex_rddme import Simulation
    from vex_rddme.guards import suggest_tau

    voxel, cap, sigma = 20.0, 20, 8.0
    dxi3 = (np.pi / 6) * sigma ** 3 / voxel ** 3
    tau = suggest_tau(1.0, voxel, 2, eta_max=3.0 * 3 * dxi3)
    ramp = np.arange(16, dtype=float) / 16
    psi = np.broadcast_to(ramp, (16, 16)).copy()[None, ...]
    sim = Simulation(
        shape=(16, 16), voxel_nm=voxel,
        species=[Species("A", sigma, np.array([3.0]))],
        occupancy_cap=cap, psi=psi, D_um2_s=1.0, tau_s=tau, seed=0,
        attach_log_handler=False,
    )
    sim.set_counts("A", np.full((16, 16), 3, dtype=np.int64))
    sim.record_initial()
    for _ in range(2000):
        sim.step()          # would raise GuardViolation if the budget were exceeded
    sim.state.check_mass()
