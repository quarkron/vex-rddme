"""Observables: projection, error bars, and the analytic comparison helpers."""

import numpy as np
import pytest

from vex_rddme import Lattice, Species, State, mu_ex_carnahan_starling
from vex_rddme.observe import (
    QuotientAccumulator,
    Series,
    align_additive_constant,
    mu_ex_from_profile,
    project,
    reaction_quotient,
    relative_discrepancy,
    report_comparison,
)
from vex_rddme.vex import dxi_increments


# ------------------------------------------------------------------- project


@pytest.mark.parametrize("shape", [(8, 4), (4, 5, 6)])
def test_project_conserves_the_total(shape):
    lat = Lattice(shape=shape, voxel_nm=10.0)
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 5, size=shape)
    for axis in range(lat.dim):
        prof = project(arr, lat, axis=axis)
        assert prof.shape == (shape[axis],)
        assert prof.sum() == arr.sum()


def test_project_accepts_flat_and_shaped_input():
    lat = Lattice(shape=(6, 4), voxel_nm=10.0)
    arr = np.arange(24).reshape(6, 4)
    assert np.array_equal(project(arr, lat), project(arr.reshape(-1), lat))


def test_project_picks_the_right_axis():
    lat = Lattice(shape=(3, 4), voxel_nm=10.0)
    arr = np.zeros((3, 4))
    arr[2, :] = 1.0                       # a full row at x = 2
    assert project(arr, lat, axis=0).tolist() == [0.0, 0.0, 4.0]
    assert project(arr, lat, axis=1).tolist() == [1.0, 1.0, 1.0, 1.0]


def test_project_rejects_wrong_shape():
    lat = Lattice(shape=(6, 4), voxel_nm=10.0)
    with pytest.raises(ValueError, match="expected shape"):
        project(np.zeros((5, 5)), lat)


# -------------------------------------------------------------------- Series


def test_series_mean_and_error():
    s = Series("x")
    for v in ([1.0, 2.0], [3.0, 4.0], [5.0, 6.0]):
        s.add(v)
    assert s.n == 3
    assert s.mean.tolist() == [3.0, 4.0]
    # population std of (1,3,5) is sqrt(8/3)
    assert s.std[0] == pytest.approx(np.sqrt(8.0 / 3.0))
    assert s.sem[0] == pytest.approx(np.sqrt(8.0 / 3.0) / np.sqrt(3))


def test_series_with_one_sample_has_zero_error():
    s = Series().add([2.0, 3.0])
    assert s.sem.tolist() == [0.0, 0.0]


def test_empty_series_raises():
    with pytest.raises(RuntimeError, match="no samples"):
        _ = Series("empty").mean


def test_series_error_shrinks_with_more_samples():
    rng = np.random.default_rng(1)
    sems = []
    for n in (50, 500, 5000):
        s = Series()
        for _ in range(n):
            s.add(rng.standard_normal(3))
        sems.append(float(s.sem.mean()))
    assert sems[0] > sems[1] > sems[2]
    assert sems[2] < sems[0] / 5


# ------------------------------------------------------- mu_ex extraction


def test_mu_ex_extraction_recovers_carnahan_starling():
    """Construct a profile that satisfies equilibrium, then invert it.

    This checks the extraction algebra without a simulation: build rho(x) from a
    chosen phi(x) by solving ln rho + mu_ex(rho) + phi = const, then confirm the
    extraction returns mu_ex.
    """
    dxi3 = 0.004
    eta = np.linspace(0.02, 0.35, 40)
    density = eta / dxi3
    mu_ex = mu_ex_carnahan_starling(eta)
    # phi that makes this profile stationary, up to a constant
    phi = -np.log(density) - mu_ex
    recovered = mu_ex_from_profile(density, phi)
    aligned = align_additive_constant(recovered, mu_ex)
    assert np.allclose(aligned, mu_ex, atol=1e-10)


def test_mu_ex_extraction_is_shape_sensitive():
    """A wrong excess term changes the curve's shape, not just its offset."""
    dxi3 = 0.004
    eta = np.linspace(0.02, 0.35, 40)
    density = eta / dxi3
    truth = mu_ex_carnahan_starling(eta)
    phi = -np.log(density) - truth
    recovered = mu_ex_from_profile(density, phi)

    # Pretend the model gave an ideal (zero) excess term instead.
    wrong = np.zeros_like(truth)
    aligned_wrong = align_additive_constant(recovered, wrong)
    assert not np.allclose(aligned_wrong, wrong, atol=0.05)


def test_mu_ex_extraction_rejects_empty_bins():
    with pytest.raises(ValueError, match="strictly positive density"):
        mu_ex_from_profile(np.array([1.0, 0.0, 2.0]), np.zeros(3))


def test_align_additive_constant_matches_means():
    a = np.array([1.0, 2.0, 3.0])
    b = a + 7.5
    assert np.allclose(align_additive_constant(a, b), b)


# ------------------------------------------------------------ discrepancies


def test_relative_discrepancy_uses_the_predicted_range():
    predicted = np.array([0.0, 1.0, 2.0])
    measured = predicted + 0.1
    d = relative_discrepancy(measured, predicted)
    assert d["max"] == pytest.approx(0.05)      # 0.1 / range 2.0
    assert d["rms"] == pytest.approx(0.05)


def test_relative_discrepancy_rejects_a_flat_prediction():
    with pytest.raises(ValueError, match="predicted curve is flat"):
        relative_discrepancy(np.ones(3), np.ones(3))


def test_relative_discrepancy_accepts_an_explicit_scale():
    d = relative_discrepancy(np.ones(3) * 1.1, np.ones(3), scale=1.0)
    assert d["max"] == pytest.approx(0.1)


# ------------------------------------------------------------ quotients


def test_reaction_quotient_forms():
    counts = np.array([[2, 3], [4, 0], [6, 5]], dtype=np.int64)
    num, den = reaction_quotient(counts, reactants=(0, 1), products=(2,))
    assert num.tolist() == [6.0, 5.0]
    assert den.tolist() == [8.0, 0.0]


def test_quotient_accumulator_scalar_mode():
    lat = Lattice(shape=(4, 4), voxel_nm=10.0)
    counts = np.zeros((3, lat.n_voxels), dtype=np.int64)
    counts[0] = 2
    counts[1] = 3
    counts[2] = 12
    acc = QuotientAccumulator(reactants=(0, 1), products=(2,))
    for _ in range(5):
        acc.add(counts)
    assert acc.n == 5
    assert float(acc.quotient[0]) == pytest.approx(2.0)     # 12 / (2*3)
    assert float(acc.sem[0]) == pytest.approx(0.0)          # identical samples


def test_quotient_accumulator_axis_resolved():
    lat = Lattice(shape=(4, 2), voxel_nm=10.0)
    counts = np.zeros((3, lat.n_voxels), dtype=np.int64)
    c = counts.reshape(3, 4, 2)
    c[0, :, :] = 1
    c[1, :, :] = 1
    c[2, 0, :] = 4
    c[2, 1:, :] = 2
    acc = QuotientAccumulator((0, 1), (2,), lattice=lat, axis=0)
    acc.add(counts)
    q = acc.quotient
    assert q.shape == (4,)
    assert q[0] == pytest.approx(4.0)      # 8 / 2
    assert q[1] == pytest.approx(2.0)      # 4 / 2


def test_quotient_accumulator_reports_error_from_scatter():
    lat = Lattice(shape=(4, 4), voxel_nm=10.0)
    acc = QuotientAccumulator((0, 1), (2,))
    rng = np.random.default_rng(2)
    for _ in range(200):
        counts = np.zeros((3, lat.n_voxels), dtype=np.int64)
        counts[0] = 2
        counts[1] = 2
        counts[2] = rng.integers(2, 10, size=lat.n_voxels)
        acc.add(counts)
    assert float(acc.sem[0]) > 0.0
    assert float(acc.quotient[0]) == pytest.approx(5.5 / 4.0, rel=0.05)


def test_quotient_undefined_when_denominator_is_zero():
    lat = Lattice(shape=(2, 2), voxel_nm=10.0)
    counts = np.zeros((3, lat.n_voxels), dtype=np.int64)
    counts[2] = 1
    acc = QuotientAccumulator((0, 1), (2,)).add(counts)
    with pytest.raises(RuntimeError, match="undefined"):
        _ = acc.quotient


def test_axis_resolved_accumulation_needs_a_lattice():
    counts = np.ones((3, 4), dtype=np.int64)
    with pytest.raises(ValueError, match="needs a lattice"):
        QuotientAccumulator((0, 1), (2,), axis=0).add(counts)


# -------------------------------------------------------------- reporting


def test_report_scalar_comparison_states_all_three_numbers():
    text = report_comparison("K_eq", 0.198, 0.200, sem=0.004)
    assert "measured" in text and "0.198" in text
    assert "predicted" in text and "0.2" in text
    assert "relative discrepancy" in text and "1.00%" in text


def test_report_profile_comparison_states_max_and_rms():
    predicted = np.linspace(1.0, 3.0, 10)
    measured = predicted + 0.02
    text = report_comparison("profile", measured, predicted, sem=np.full(10, 0.01))
    assert "bins" in text and "10" in text
    assert "max discrepancy" in text
    assert "rms discrepancy" in text
    assert "standard error" in text


def test_series_variance_survives_a_large_mean():
    """Welford, not E[x^2]-E[x]^2.

    The textbook form cancels catastrophically when the mean dwarfs the spread: at a
    mean of 1e8 with true sd 1 it reported ~4.4, a 300% error, with no indication
    anything was wrong. Error bars back every demonstration's stated claim, so a silently wrong
    one is the failure mode this package refuses elsewhere.
    """
    rng = np.random.default_rng(0)
    for mean, sd in [(1e4, 30.0), (1e6, 5.0), (1e8, 1.0)]:
        x = rng.normal(mean, sd, size=(2000, 4))
        s = Series()
        for row in x:
            s.add(row)
        assert float(s.std.mean()) == pytest.approx(float(x.std(axis=0).mean()), rel=1e-6)
        assert float(s.mean.mean()) == pytest.approx(float(x.mean()), rel=1e-12)


def test_series_std_matches_numpy_population_convention():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((500, 3)) * 2.5 + 7.0
    s = Series()
    for row in x:
        s.add(row)
    assert np.allclose(s.std, x.std(axis=0), rtol=1e-12)
    assert np.allclose(s.mean, x.mean(axis=0), rtol=1e-12)


def test_series_mean_is_not_aliased_to_internal_state():
    s = Series().add([1.0, 2.0]).add([3.0, 4.0])
    m = s.mean
    m[0] = 999.0
    assert s.mean[0] == 2.0, "mean must return a copy, not internal state"


def test_series_std_requires_samples():
    with pytest.raises(RuntimeError, match="no samples"):
        _ = Series("empty").std
