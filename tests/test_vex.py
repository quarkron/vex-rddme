"""Weighted densities, the BMCSL free energy, and the exact integer table.

Two tests here exist because the corresponding mistake is silent:

* ``test_bfex_scales_with_voxel_volume`` and
  ``test_insertion_work_matches_carnahan_starling`` pin the *scale*. The bracketed
  BMCSL expression is a free-energy density; forgetting the voxel-volume factor
  makes every channel work ``1/V`` too small, 1000x at a 10 nm voxel, which
  switches exclusion off while every density profile still looks plausible.
* ``test_f_order_strides_are_detected`` pins the *index order*. C-order strides are
  required to match a table built by ``np.indices``; the transposed assignment
  yields a table that is wrong but still runs.
"""

import numpy as np
import pytest

from vex_rddme import Species
from vex_rddme.guards import GuardReport, GuardViolation
from vex_rddme.vex import (
    ExclusionModel,
    bfex,
    dxi_increments,
    mu_ex_carnahan_starling,
)

V_VOXEL = 8000.0  # 20 nm voxel


def quiet_report():
    return GuardReport(attach_handler=False)


def make_model(sigmas=(4.0, 6.0), cap=8, max_entries=8_388_608, voxel_volume=V_VOXEL):
    species = [
        Species(name=f"S{i}", sigma_nm=s, gamma=np.zeros(1)) for i, s in enumerate(sigmas)
    ]
    return ExclusionModel(
        species,
        voxel_volume_nm3=voxel_volume,
        occupancy_cap=cap,
        max_table_entries=max_entries,
        report=quiet_report(),
    )


# ------------------------------------------------------------ dxi increments


def test_dxi_increments_are_pi_over_six_sigma_to_the_n():
    sp = [Species("a", 4.0, np.zeros(1))]
    dxi = dxi_increments(sp, V_VOXEL)
    pi6 = np.pi / 6.0
    for n in range(4):
        assert dxi[0, n] == pytest.approx(pi6 * 4.0 ** n / V_VOXEL)


def test_ideal_species_gets_an_all_zero_row():
    sp = [Species("ideal", 0.0, np.zeros(1)), Species("hard", 5.0, np.zeros(1))]
    dxi = dxi_increments(sp, V_VOXEL)
    assert np.all(dxi[0] == 0.0)
    assert np.any(dxi[1] != 0.0)


def test_dxi_rejects_bad_volume():
    sp = [Species("a", 4.0, np.zeros(1))]
    for bad in (0.0, -1.0, np.inf, np.nan):
        with pytest.raises(ValueError, match="volume must be positive"):
            dxi_increments(sp, bad)


def test_xi3_is_the_packing_fraction():
    sigma = 5.0
    sp = [Species("a", sigma, np.zeros(1))]
    dxi = dxi_increments(sp, V_VOXEL)
    # one sphere's volume fraction of the voxel
    assert dxi[0, 3] == pytest.approx((np.pi / 6.0) * sigma ** 3 / V_VOXEL)


# -------------------------------------------------------------------- bfex


def test_bfex_requires_four_components():
    with pytest.raises(ValueError, match="4 leading components"):
        bfex(np.zeros((3, 5)), V_VOXEL)


def test_bfex_rejects_bad_volume():
    for bad in (0.0, -1.0, np.nan):
        with pytest.raises(ValueError, match="voxel_volume_nm3 must be positive"):
            bfex(np.zeros((4, 2)), bad)


def test_empty_voxel_has_exactly_zero_free_energy():
    assert bfex(np.zeros((4, 3)), V_VOXEL).tolist() == [0.0, 0.0, 0.0]


def test_bfex_scales_with_voxel_volume():
    """The bracketed expression is a density; the result must be extensive in V."""
    xi = np.array([[0.001], [0.004], [0.02], [0.15]])
    a = float(bfex(xi, 1000.0)[0])
    b = float(bfex(xi, 2000.0)[0])
    assert b == pytest.approx(2.0 * a)
    assert a != 0.0


def test_bfex_is_positive_and_grows_with_packing():
    dxi = dxi_increments([Species("a", 2.0, np.zeros(1))], 1000.0)[0]
    vals = [float(bfex((dxi * n)[:, None], 1000.0)[0]) for n in (10, 40, 80, 120)]
    assert all(v > 0 for v in vals)
    assert vals == sorted(vals)


def test_insertion_work_matches_carnahan_starling():
    """Single-species reduction: dF/dN from BMCSL is the Carnahan-Starling mu_ex.

    The discrete insertion work F(n+1) - F(n) is a centred finite difference of the
    continuum derivative at n + 1/2, so agreement is second order in dxi3. A small
    diameter keeps dxi3 small and the residual at the 1e-4 level; a missing
    voxel-volume factor would show up as a flat factor of 1/V instead.
    """
    V = 1000.0
    dxi = dxi_increments([Species("a", 2.0, np.zeros(1))], V)[0]
    for n in (10, 40, 80, 120, 160):
        eta_mid = (n + 0.5) * dxi[3]
        insertion = float(bfex((dxi * (n + 1))[:, None], V)[0]) - float(
            bfex((dxi * n)[:, None], V)[0]
        )
        analytic = float(mu_ex_carnahan_starling(eta_mid))
        assert insertion == pytest.approx(analytic, rel=1e-3), (
            f"n={n}, eta={eta_mid:.4f}: insertion work {insertion:.6g} vs "
            f"Carnahan-Starling {analytic:.6g}. A constant ratio of ~{1/V:g} means "
            f"the voxel-volume factor is missing from bfex."
        )


def test_carnahan_starling_reference_shape():
    # mu_ex -> 0 as eta -> 0, and diverges as eta -> 1
    assert mu_ex_carnahan_starling(0.0) == pytest.approx(0.0)
    assert mu_ex_carnahan_starling(0.4) > mu_ex_carnahan_starling(0.2) > 0
    assert mu_ex_carnahan_starling(0.9) > 100


# ------------------------------------------------------------------- table


def test_table_is_built_for_a_modest_configuration():
    m = make_model(sigmas=(4.0, 6.0), cap=8)
    assert m.uses_table
    assert m.radix == 10  # cap + 2
    assert m.table_entries == 10 ** 2


def test_ideal_species_do_not_enter_the_index():
    m = make_model(sigmas=(0.0, 5.0), cap=8)
    assert m.n_hard_core == 1
    assert m.table_entries == 10
    assert m.stride(0) == 0     # ideal
    assert m.stride(1) == 1


def test_no_hard_core_species_means_no_table_and_zero_work():
    m = make_model(sigmas=(0.0, 0.0), cap=4)
    assert not m.uses_table
    counts = np.array([[2, 1], [3, 0]])
    assert np.allclose(m.channel_work(counts, [-1, +1]), 0.0)


def test_table_agrees_with_elementwise_evaluation():
    """The table is exact, not an approximation."""
    tabled = make_model(sigmas=(4.0, 6.0), cap=8)
    direct = make_model(sigmas=(4.0, 6.0), cap=8, max_entries=4)
    assert tabled.uses_table and not direct.uses_table

    rng = np.random.default_rng(0)
    counts = rng.integers(0, 5, size=(2, 500))
    counts[1] = np.minimum(counts[1], 8 - counts[0])  # respect the cap

    assert np.allclose(
        tabled.free_energy(counts), direct.free_energy(counts), rtol=0, atol=1e-12
    )


def test_channel_work_agrees_on_both_paths():
    tabled = make_model(sigmas=(4.0, 6.0), cap=8)
    direct = make_model(sigmas=(4.0, 6.0), cap=8, max_entries=4)
    rng = np.random.default_rng(1)
    counts = rng.integers(1, 4, size=(2, 400))
    for dnu in ([-1, +1], [+1, -1], [-1, 0], [0, +1]):
        a = tabled.channel_work(counts, dnu)
        b = direct.channel_work(counts, dnu)
        assert np.allclose(a, b, rtol=0, atol=1e-12), f"dnu={dnu}"


def test_f_order_strides_are_detected():
    """Transposed strides give a table that runs but is wrong.

    The table is built by ``np.indices``, whose first axis varies slowest, so
    strides must be ``radix**(S-1-s)``. Substituting ``radix**s`` still produces
    in-range indices and plausible numbers, so only a comparison against direct
    evaluation catches it.
    """
    m = make_model(sigmas=(4.0, 6.0), cap=8)
    direct = make_model(sigmas=(4.0, 6.0), cap=8, max_entries=4)

    rng = np.random.default_rng(2)
    counts = rng.integers(0, 5, size=(2, 300))
    counts[1] = np.minimum(counts[1], 8 - counts[0])

    assert np.allclose(m.free_energy(counts), direct.free_energy(counts), atol=1e-12)

    # Now transpose the stride order and confirm the agreement breaks.
    m._strides = m._strides[::-1].copy()
    assert not np.allclose(
        m.free_energy(counts), direct.free_energy(counts), atol=1e-9
    ), "F-order strides must not agree with direct evaluation"


def test_removal_lookup_is_valid_whenever_the_particle_is_present():
    """``idx - stride_s`` addresses the intended count vector when n_s >= 1.

    When ``n_s == 0`` the subtraction borrows from a higher digit and addresses an
    unrelated vector. But a hop or reaction can only remove a particle that is
    there, so that case never arises in the solver.
    """
    m = make_model(sigmas=(4.0, 6.0), cap=8)
    direct = make_model(sigmas=(4.0, 6.0), cap=8, max_entries=4)
    rng = np.random.default_rng(3)
    counts = rng.integers(1, 5, size=(2, 300))     # every count >= 1
    counts[1] = np.minimum(counts[1], 8 - counts[0])
    counts = counts[:, counts.min(axis=0) >= 1]

    for s in (0, 1):
        dnu = [0, 0]
        dnu[s] = -1
        assert np.allclose(
            m.channel_work(counts, dnu), direct.channel_work(counts, dnu), atol=1e-12
        ), f"removal of species {s}"


def test_stoichiometry_offset_composes_species_strides():
    m = make_model(sigmas=(4.0, 5.0, 6.0), cap=6)
    # A + B -> C over three hard-core species
    dnu = [-1, -1, +1]
    expected = -m.stride(0) - m.stride(1) + m.stride(2)
    assert m.stoichiometry_offset(dnu) == expected


def test_index_and_stride_require_the_table_path():
    direct = make_model(sigmas=(4.0, 6.0), cap=8, max_entries=4)
    with pytest.raises(RuntimeError, match="requires the table path"):
        direct.index(np.zeros((2, 3), dtype=np.int64))
    with pytest.raises(RuntimeError, match="requires the table path"):
        direct.stride(0)


def test_xi_rejects_wrong_species_count():
    m = make_model()
    with pytest.raises(ValueError, match="must have 2 species rows"):
        m.xi(np.zeros((3, 10)))


# ------------------------------------------------------- announced fallback


def test_oversized_table_falls_back_and_says_so():
    report = quiet_report()
    species = [Species(f"S{i}", 3.0, np.zeros(1)) for i in range(4)]
    m = ExclusionModel(
        species,
        voxel_volume_nm3=V_VOXEL,
        occupancy_cap=30,
        max_table_entries=1000,
        report=report,
    )
    assert not m.uses_table
    assert report.has_warning("exclusion-fallback")
    msg = dict(report.warnings)["exclusion-fallback"]
    assert "32**4" in msg                 # radix = cap + 2
    assert "1,000" in msg                 # the envelope
    assert "elementwise" in msg


def test_table_path_reports_what_it_built():
    report = quiet_report()
    ExclusionModel(
        [Species("a", 4.0, np.zeros(1))],
        voxel_volume_nm3=V_VOXEL,
        occupancy_cap=8,
        report=report,
    )
    tags = [t for t, _ in report.infos]
    assert "exclusion" in tags


def test_elementwise_path_needs_dnu():
    direct = make_model(sigmas=(4.0, 6.0), cap=8, max_entries=4)
    with pytest.raises(ValueError, match="needs dnu as well as offset"):
        direct.shifted_free_energy(np.ones((2, 4), dtype=np.int64), offset=0, dnu=None)


# ------------------------------------------------------------- packing info


def test_max_attainable_xi3_uses_the_largest_species():
    m = make_model(sigmas=(4.0, 6.0), cap=8)
    expected = 8 * (np.pi / 6.0) * 6.0 ** 3 / V_VOXEL
    assert m.max_attainable_xi3() == pytest.approx(expected)


def test_occupancy_at_half_packing_is_infinite_for_ideal_species():
    m = make_model(sigmas=(0.0, 6.0), cap=8)
    table = m.occupancy_at_half_packing()
    assert table["S0"] == float("inf")
    assert np.isfinite(table["S1"])
    assert table["S1"] == pytest.approx(0.5 / ((np.pi / 6.0) * 6.0 ** 3 / V_VOXEL))


# ------------------------------------------- fractional (mean) compositions


def test_fractional_counts_take_the_elementwise_path():
    """A time-averaged composition is a valid free-energy argument, not an error.

    It cannot index the integer table, so it falls through to elementwise evaluation.
    That is the only correct option, and it is what the analytic comparisons in the
    notebooks need.
    """
    m = make_model(sigmas=(4.0, 6.0), cap=8)
    assert m.uses_table
    frac = np.array([[2.4, 3.1], [1.7, 0.5]])
    F = m.free_energy(frac)
    assert F.shape == (2,)
    assert np.all(np.isfinite(F))


def test_fractional_and_integer_paths_agree_on_whole_numbers():
    m = make_model(sigmas=(4.0, 6.0), cap=8)
    ints = np.array([[2, 3], [1, 0]], dtype=np.int64)
    floats = ints.astype(np.float64)
    assert np.allclose(m.free_energy(ints), m.free_energy(floats), atol=1e-12)


def test_channel_work_at_a_fractional_composition():
    m = make_model(sigmas=(4.0, 6.0), cap=8)
    frac = np.array([[2.4], [1.7]])
    w_frac = m.channel_work(frac, [-1, 1])
    w_int = m.channel_work(np.array([[2], [2]], dtype=np.int64), [-1, 1])
    assert np.all(np.isfinite(w_frac))
    # nearby composition, so a comparable magnitude
    assert abs(float(w_frac[0])) < 10 * abs(float(w_int[0])) + 1.0
