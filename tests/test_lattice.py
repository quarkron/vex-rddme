"""Lattice geometry, with the two roll directions pinned explicitly.

``deposit`` and ``neighbour`` roll opposite ways. Confusing them is a silent
one-voxel shift in the drift that no aggregate observable would obviously reveal,
so both are tested against a single hand-placed particle in every direction, in
both dimensions.
"""

import numpy as np
import pytest

from vex_rddme import Lattice


# ------------------------------------------------------------------ construction


def test_rejects_unsupported_dimension():
    with pytest.raises(ValueError, match="dim=2 and dim=3"):
        Lattice(shape=(8,), voxel_nm=10.0)
    with pytest.raises(ValueError, match="dim=2 and dim=3"):
        Lattice(shape=(4, 4, 4, 4), voxel_nm=10.0)


def test_explicit_dim_must_agree_with_shape():
    Lattice(shape=(8, 8), voxel_nm=10.0, dim=2)
    Lattice(shape=(4, 4, 4), voxel_nm=10.0, dim=3)
    with pytest.raises(ValueError, match="disagrees with shape"):
        Lattice(shape=(8, 8), voxel_nm=10.0, dim=3)


def test_rejects_bad_voxel_size():
    for bad in (0.0, -5.0, np.inf, np.nan):
        with pytest.raises(ValueError, match="voxel_nm must be positive"):
            Lattice(shape=(8, 8), voxel_nm=bad)


@pytest.mark.parametrize("shape,n_dirs", [((8, 8), 4), ((4, 5, 6), 6)])
def test_direction_count_follows_dimension(shape, n_dirs):
    lat = Lattice(shape=shape, voxel_nm=10.0)
    assert lat.n_dirs == n_dirs
    assert lat.offsets.shape == (n_dirs, lat.dim)


def test_voxel_volume_is_cubic_even_in_2d():
    # A 2D lattice is a slab one voxel thick. Weighted densities are volume
    # fractions, so the reference volume must stay h**3 in both dimensions.
    assert Lattice(shape=(8, 8), voxel_nm=20.0).voxel_volume_nm3 == pytest.approx(8000.0)
    assert Lattice(shape=(8, 8, 8), voxel_nm=20.0).voxel_volume_nm3 == pytest.approx(8000.0)


def test_cfl_limit_depends_on_dimension():
    assert Lattice(shape=(8, 8), voxel_nm=10.0).cfl_limit() == pytest.approx(0.25)
    assert Lattice(shape=(8, 8, 8), voxel_nm=10.0).cfl_limit() == pytest.approx(1.0 / 6.0)


# ------------------------------------------------------------- offsets / opposite


@pytest.mark.parametrize("shape", [(8, 8), (4, 5, 6)])
def test_offsets_are_unit_axial_vectors(shape):
    lat = Lattice(shape=shape, voxel_nm=10.0)
    for d in range(lat.n_dirs):
        off = lat.offsets[d]
        assert np.count_nonzero(off) == 1
        assert abs(int(off[lat.axis(d)])) == 1
        assert int(off[lat.axis(d)]) == lat.sign(d)


@pytest.mark.parametrize("shape", [(8, 8), (4, 5, 6)])
def test_opposite_directions_cancel(shape):
    lat = Lattice(shape=shape, voxel_nm=10.0)
    for d in range(lat.n_dirs):
        o = int(lat.opposite[d])
        assert o != d
        assert int(lat.opposite[o]) == d
        assert np.array_equal(lat.offsets[d] + lat.offsets[o], np.zeros(lat.dim, int))


@pytest.mark.parametrize("shape", [(8, 8), (4, 5, 6)])
def test_every_direction_is_distinct(shape):
    lat = Lattice(shape=shape, voxel_nm=10.0)
    seen = {tuple(off) for off in lat.offsets}
    assert len(seen) == lat.n_dirs


# ------------------------------------------------- the two roll conventions


@pytest.mark.parametrize("shape", [(8, 8), (4, 5, 6)])
def test_deposit_moves_a_particle_to_the_expected_neighbour(shape):
    """deposit(moved, d) places departures at src + offset(d)."""
    lat = Lattice(shape=shape, voxel_nm=10.0)
    src = tuple(s // 2 for s in shape)
    for d in range(lat.n_dirs):
        moved = np.zeros(shape, dtype=np.int64)
        moved[src] = 1
        landed = lat.deposit(moved, d)
        expected = tuple(int(src[a] + lat.offsets[d][a]) for a in range(lat.dim))
        assert landed.sum() == 1, "deposit must not create or destroy particles"
        assert landed[expected] == 1, (
            f"d={d} (axis {lat.axis(d)}, sign {lat.sign(d):+d}): particle from {src} "
            f"landed at {np.unravel_index(int(np.argmax(landed)), shape)}, "
            f"expected {expected}"
        )


@pytest.mark.parametrize("shape", [(8, 8), (4, 5, 6)])
def test_neighbour_reads_the_expected_voxel(shape):
    """neighbour(values, d) at v returns values at v + offset(d)."""
    lat = Lattice(shape=shape, voxel_nm=10.0)
    here = tuple(s // 2 for s in shape)
    for d in range(lat.n_dirs):
        there = tuple(int(here[a] + lat.offsets[d][a]) for a in range(lat.dim))
        values = np.zeros(shape, dtype=np.float64)
        values[there] = 7.0
        got = lat.neighbour(values, d)
        assert got[here] == pytest.approx(7.0), (
            f"d={d}: reading the direction-{d} neighbour of {here} should see the "
            f"marker planted at {there}"
        )


@pytest.mark.parametrize("shape", [(8, 8), (4, 5, 6)])
def test_deposit_and_neighbour_roll_opposite_ways(shape):
    """The two operations are adjoint: depositing then reading back is identity."""
    lat = Lattice(shape=shape, voxel_nm=10.0)
    rng = np.random.default_rng(0)
    a = rng.standard_normal(shape)
    for d in range(lat.n_dirs):
        assert np.allclose(lat.neighbour(lat.deposit(a, d), d), a)


# ------------------------------------------------------------------ boundaries


@pytest.mark.parametrize("shape", [(8, 8), (4, 5, 6)])
def test_can_leave_excludes_exactly_the_exit_face(shape):
    lat = Lattice(shape=shape, voxel_nm=10.0)
    for d in range(lat.n_dirs):
        mask = lat.can_leave(d)
        assert mask.shape == shape
        face_size = lat.n_voxels // shape[lat.axis(d)]
        assert np.count_nonzero(~mask) == face_size, (
            f"d={d} should block exactly one face of {face_size} voxels"
        )
        sl = [slice(None)] * lat.dim
        sl[lat.axis(d)] = -1 if lat.sign(d) > 0 else 0
        assert not mask[tuple(sl)].any()


@pytest.mark.parametrize("shape", [(8, 8), (4, 5, 6)])
def test_masked_deposit_conserves_mass_at_the_boundary(shape):
    """With departures zeroed outside can_leave, no mass crosses a face."""
    lat = Lattice(shape=shape, voxel_nm=10.0)
    rng = np.random.default_rng(1)
    for d in range(lat.n_dirs):
        moved = rng.integers(0, 4, size=shape).astype(np.int64)
        moved[~lat.can_leave(d)] = 0
        lat.assert_no_wrap(moved, d)
        landed = lat.deposit(moved, d)
        assert landed.sum() == moved.sum()
        # Everything that landed came from inside, so nothing arrived on the
        # face opposite the exit face (which is where a wrap would show up).
        sl = [slice(None)] * lat.dim
        sl[lat.axis(d)] = 0 if lat.sign(d) > 0 else -1
        assert landed[tuple(sl)].sum() == 0


@pytest.mark.parametrize("shape", [(8, 8), (4, 5, 6)])
def test_assert_no_wrap_catches_unmasked_departures(shape):
    lat = Lattice(shape=shape, voxel_nm=10.0)
    for d in range(lat.n_dirs):
        moved = np.zeros(shape, dtype=np.int64)
        exit_face = ~lat.can_leave(d)
        moved[exit_face] = 1
        with pytest.raises(AssertionError, match="would leave the lattice"):
            lat.assert_no_wrap(moved, d)


# ------------------------------------------------------------------- indexing


@pytest.mark.parametrize("shape", [(8, 8), (4, 5, 6)])
def test_flat_and_coord_roundtrip(shape):
    lat = Lattice(shape=shape, voxel_nm=10.0)
    rng = np.random.default_rng(2)
    for _ in range(20):
        c = tuple(int(rng.integers(0, s)) for s in shape)
        assert lat.coord(lat.flat(c)) == c
    for v in (0, lat.n_voxels - 1, lat.n_voxels // 3):
        assert lat.flat(lat.coord(v)) == v
