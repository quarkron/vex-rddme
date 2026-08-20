"""Occupancy-count state: exact seeding, the occupancy cap, and mass bookkeeping."""

import numpy as np
import pytest

from vex_rddme import Lattice, Species, State


def make_state(shape=(8, 8), cap=8, n_bases=1, sigmas=(4.0, 6.0)):
    lat = Lattice(shape=shape, voxel_nm=20.0)
    sp = [
        Species(name=f"S{i}", sigma_nm=s, gamma=np.zeros(n_bases))
        for i, s in enumerate(sigmas)
    ]
    return lat, State(lat, sp, occupancy_cap=cap)


# ------------------------------------------------------------------ construction


def test_requires_at_least_one_species():
    lat = Lattice(shape=(4, 4), voxel_nm=10.0)
    with pytest.raises(ValueError, match="at least one species"):
        State(lat, [], occupancy_cap=4)


def test_rejects_duplicate_species_names():
    lat = Lattice(shape=(4, 4), voxel_nm=10.0)
    sp = [Species("A", 1.0, np.zeros(1)), Species("A", 2.0, np.zeros(1))]
    with pytest.raises(ValueError, match="duplicate species names"):
        State(lat, sp, occupancy_cap=4)


def test_rejects_inconsistent_basis_count():
    lat = Lattice(shape=(4, 4), voxel_nm=10.0)
    sp = [Species("A", 1.0, np.zeros(1)), Species("B", 1.0, np.zeros(2))]
    with pytest.raises(ValueError, match="same number of basis"):
        State(lat, sp, occupancy_cap=4)


def test_rejects_nonpositive_cap():
    lat = Lattice(shape=(4, 4), voxel_nm=10.0)
    with pytest.raises(ValueError, match="occupancy_cap must be at least 1"):
        State(lat, [Species("A", 1.0, np.zeros(1))], occupancy_cap=0)


def test_negative_diameter_rejected():
    with pytest.raises(ValueError, match="sigma_nm must be finite and non-negative"):
        Species("A", -1.0, np.zeros(1))


def test_inert_species_must_not_drift():
    with pytest.raises(ValueError, match="declared inert but has non-zero basis"):
        Species("C", 5.0, np.array([1.5]), inert=True)
    # zero coupling is fine
    Species("C", 5.0, np.zeros(1), inert=True)


def test_hard_core_indices_exclude_ideal_species():
    lat = Lattice(shape=(4, 4), voxel_nm=10.0)
    sp = [
        Species("ideal", 0.0, np.zeros(1)),
        Species("hard", 5.0, np.zeros(1)),
        Species("also_hard", 3.0, np.zeros(1)),
    ]
    st = State(lat, sp, occupancy_cap=4)
    assert st.hard_core_indices == [1, 2]


# ------------------------------------------------------------------- lookup


def test_index_by_name_and_by_integer():
    _, st = make_state()
    assert st.index("S0") == 0
    assert st.index("S1") == 1
    assert st.index(1) == 1
    with pytest.raises(KeyError, match="unknown species"):
        st.index("nope")
    with pytest.raises(KeyError, match="out of range"):
        st.index(5)


# ------------------------------------------------------------------- seeding


def test_seed_uniform_places_the_exact_total():
    _, st = make_state(shape=(16, 16), cap=8)
    rng = np.random.default_rng(0)
    st.seed_uniform("S0", 500, rng)
    assert int(st.totals()[0]) == 500
    assert int(st.totals()[1]) == 0


def test_seed_uniform_respects_the_cap_across_species():
    _, st = make_state(shape=(8, 8), cap=3)
    rng = np.random.default_rng(1)
    st.seed_uniform("S0", 100, rng)
    st.seed_uniform("S1", 80, rng)
    st.check_occupancy_cap()
    assert st.occupancy().max() <= 3


def test_seed_uniform_refuses_to_truncate():
    _, st = make_state(shape=(4, 4), cap=2)  # capacity 32
    rng = np.random.default_rng(2)
    with pytest.raises(ValueError, match="only 32 slots remain"):
        st.seed_uniform("S0", 33, rng)


def test_seed_uniform_fills_to_capacity_exactly():
    _, st = make_state(shape=(4, 4), cap=2)
    rng = np.random.default_rng(3)
    st.seed_uniform("S0", 32, rng)
    assert int(st.totals()[0]) == 32
    assert np.all(st.occupancy() == 2)


def test_set_counts_accepts_lattice_and_flat_shapes():
    lat, st = make_state(shape=(4, 4), cap=8)
    st.set_counts("S0", np.ones(lat.shape, dtype=np.int64))
    assert int(st.totals()[0]) == 16
    st.set_counts("S0", np.full(lat.n_voxels, 2, dtype=np.int64))
    assert int(st.totals()[0]) == 32


def test_set_counts_rejects_cap_violation_with_a_useful_message():
    lat, st = make_state(shape=(4, 4), cap=3)
    st.set_counts("S0", np.full(lat.shape, 2, dtype=np.int64))
    with pytest.raises(ValueError, match="above the occupancy cap 3"):
        st.set_counts("S1", np.full(lat.shape, 2, dtype=np.int64))


def test_set_counts_rejects_negative_and_wrong_shape():
    lat, st = make_state(shape=(4, 4))
    with pytest.raises(ValueError, match="must be non-negative"):
        st.set_counts("S0", -np.ones(lat.shape, dtype=np.int64))
    with pytest.raises(ValueError, match="must have shape"):
        st.set_counts("S0", np.ones((3, 3), dtype=np.int64))


# --------------------------------------------------------- mass bookkeeping


def test_initial_totals_are_recorded_exactly():
    _, st = make_state(shape=(16, 16), cap=8)
    rng = np.random.default_rng(4)
    st.seed_uniform("S0", 417, rng)
    st.seed_uniform("S1", 233, rng)
    st.record_initial()
    assert st.expected_totals().tolist() == [417, 233]
    assert st.mass_residual().tolist() == [0, 0]
    st.check_mass()


def test_mass_check_requires_a_recorded_reference():
    _, st = make_state()
    with pytest.raises(RuntimeError, match="record_initial\\(\\) has not been called"):
        st.check_mass()


def test_mass_check_detects_a_leak_and_names_the_species():
    _, st = make_state(shape=(8, 8), cap=8)
    rng = np.random.default_rng(5)
    st.seed_uniform("S1", 60, rng)
    st.record_initial()
    st.counts[1, 0] -= 3  # simulate a transport bug
    with pytest.raises(AssertionError, match=r"mass not conserved"):
        st.check_mass()
    try:
        st.check_mass()
    except AssertionError as exc:
        assert "S1" in str(exc)
        assert "-3" in str(exc)


def test_reaction_delta_is_accounted_for():
    _, st = make_state(shape=(8, 8), cap=8)
    rng = np.random.default_rng(6)
    st.seed_uniform("S0", 40, rng)
    st.seed_uniform("S1", 40, rng)
    st.record_initial()
    # 7 firings of S0 -> S1
    st.counts[0, 0] -= 7
    st.counts[1, 0] += 7
    st.note_reaction([-7, +7])
    st.check_mass()
    assert st.expected_totals().tolist() == [33, 47]


def test_reaction_delta_does_not_excuse_a_transport_leak():
    _, st = make_state(shape=(8, 8), cap=8)
    rng = np.random.default_rng(7)
    st.seed_uniform("S0", 40, rng)
    st.record_initial()
    st.note_reaction([-5, 0])
    st.counts[0, 0] -= 6  # one more than the reaction accounts for
    with pytest.raises(AssertionError, match="mass not conserved"):
        st.check_mass()


# ------------------------------------------------------------ cap enforcement


def test_check_occupancy_cap_reports_the_worst_voxel():
    lat, st = make_state(shape=(4, 4), cap=2)
    st.counts[0, lat.flat((1, 2))] = 5
    with pytest.raises(AssertionError, match="occupancy cap 2 exceeded"):
        st.check_occupancy_cap()
    try:
        st.check_occupancy_cap()
    except AssertionError as exc:
        assert "(1, 2)" in str(exc)
        assert "holds 5" in str(exc)


def test_headroom_matches_cap_minus_occupancy():
    _, st = make_state(shape=(8, 8), cap=6)
    rng = np.random.default_rng(8)
    st.seed_uniform("S0", 100, rng)
    assert np.array_equal(st.headroom(), 6 - st.occupancy())
    assert st.headroom().min() >= 0


def test_lattice_view_is_shaped_and_consistent():
    lat, st = make_state(shape=(4, 6), cap=8)
    st.counts[0, lat.flat((2, 3))] = 4
    view = st.lattice_view("S0")
    assert view.shape == lat.shape
    assert view[2, 3] == 4
