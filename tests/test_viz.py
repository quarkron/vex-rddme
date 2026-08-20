"""Plotting helpers, exercised headlessly.

These tests check that figures are produced and reductions are correct, not that
they look right. The Agg backend is forced so the suite needs no display.
"""

import numpy as np
import pytest

import matplotlib

matplotlib.use("Agg")

from vex_rddme import Lattice, Species, State          # noqa: E402
from vex_rddme import viz                              # noqa: E402


@pytest.fixture(autouse=True)
def close_figures():
    yield
    import matplotlib.pyplot as plt

    plt.close("all")


def make_state(shape=(8, 6), cap=20):
    lat = Lattice(shape=shape, voxel_nm=20.0)
    sp = [Species("A", 4.0, np.zeros(1)), Species("B", 6.0, np.zeros(1))]
    st = State(lat, sp, occupancy_cap=cap)
    rng = np.random.default_rng(0)
    st.seed_uniform("A", 100, rng)
    st.seed_uniform("B", 60, rng)
    return lat, st


# -------------------------------------------------------------- reductions


def test_2d_image_is_the_lattice_itself():
    lat, st = make_state()
    img = viz._as_image(st.counts[0], lat)
    assert img.shape == lat.shape
    assert img.sum() == st.counts[0].sum()


def test_3d_sum_reduction_preserves_the_total():
    lat = Lattice(shape=(4, 5, 6), voxel_nm=20.0)
    arr = np.arange(lat.n_voxels, dtype=float)
    img = viz._as_image(arr, lat, reduce="sum")
    assert img.shape == (5, 6)
    assert img.sum() == pytest.approx(arr.sum())


def test_3d_slice_reduction_picks_one_plane():
    lat = Lattice(shape=(4, 5, 6), voxel_nm=20.0)
    arr = np.zeros(lat.shape)
    arr[2] = 7.0
    img = viz._as_image(arr, lat, reduce="slice", index=2)
    assert img.shape == (5, 6)
    assert np.all(img == 7.0)


def test_slice_defaults_to_the_middle_plane():
    lat = Lattice(shape=(5, 4, 4), voxel_nm=20.0)
    arr = np.zeros(lat.shape)
    arr[2] = 3.0
    assert np.all(viz._as_image(arr, lat, reduce="slice") == 3.0)


def test_unknown_reduction_is_rejected():
    lat = Lattice(shape=(4, 4, 4), voxel_nm=20.0)
    with pytest.raises(ValueError, match="reduce must be"):
        viz._as_image(np.zeros(lat.shape), lat, reduce="median")


def test_wrong_shape_is_rejected():
    lat = Lattice(shape=(4, 4), voxel_nm=20.0)
    with pytest.raises(ValueError, match="expected shape"):
        viz._as_image(np.zeros((3, 3)), lat)


# ------------------------------------------------------------------ drawing


def test_show_lattice_returns_axes_in_2d():
    lat, st = make_state()
    ax = viz.show_lattice(st.counts[0], lat, title="A")
    assert ax.get_title() == "A"
    assert len(ax.images) == 1


def test_show_lattice_works_in_3d():
    lat = Lattice(shape=(4, 5, 6), voxel_nm=20.0)
    ax = viz.show_lattice(np.arange(lat.n_voxels, dtype=float), lat, reduce="sum")
    assert len(ax.images) == 1


def test_show_species_draws_one_panel_per_species():
    lat, st = make_state()
    axes = viz.show_species(st)
    assert len(axes) == st.n_species
    for ax, sp in zip(axes, st.species):
        assert sp.name in ax.get_title()


def test_plot_profile_with_and_without_prediction():
    measured = np.array([1.0, 2.0, 3.0, 2.0])
    ax = viz.plot_profile(measured, predicted=measured * 1.02, sem=np.full(4, 0.05))
    assert len(ax.lines) >= 1
    assert ax.get_legend() is not None

    ax2 = viz.plot_profile(measured)
    assert len(ax2.lines) >= 1


def test_plot_mu_ex_sorts_by_packing_fraction():
    eta = np.array([0.3, 0.1, 0.2])
    measured = np.array([3.0, 1.0, 2.0])
    ax = viz.plot_mu_ex(eta, measured, measured * 1.01, sem=np.full(3, 0.02))
    line = ax.lines[-1]
    assert np.all(np.diff(line.get_xdata()) > 0), "x data should be sorted"


# ---------------------------------------------------------------- animation


def test_animate_embeds_every_frame():
    """Frames are base64 PNGs, so titles are drawn *into* the images, not emitted
    as HTML text. The check is that one image per frame is embedded."""
    lat, st = make_state(shape=(6, 6))
    n_frames = 4
    frames = [st.counts[0] + i for i in range(n_frames)]
    html = viz.animate(frames, lat, titles=[f"step {i}" for i in range(n_frames)])
    assert isinstance(html, str)
    assert len(html) > 1000
    assert html.count("base64") >= n_frames, (
        f"expected at least {n_frames} embedded images, found {html.count('base64')}"
    )


def test_animate_needs_no_writer_binary():
    """to_jshtml must be used, not a writer-backed format like mp4."""
    lat, st = make_state(shape=(4, 4))
    html = viz.animate([st.counts[0], st.counts[0] + 1], lat)
    assert "<script" in html.lower()


def test_animate_rejects_an_empty_sequence():
    lat, _ = make_state()
    with pytest.raises(ValueError, match="no frames"):
        viz.animate([], lat)


# ------------------------------------------- trajectory and time-series helpers


def test_plot_timeseries_draws_one_line_per_label():
    steps = np.arange(10)
    ax = viz.plot_timeseries(steps, {"A": np.arange(10), "B": np.arange(10) * 2},
                             title="counts")
    assert len(ax.lines) == 2
    assert ax.get_title() == "counts"


def test_plot_timeseries_rejects_a_length_mismatch():
    with pytest.raises(ValueError, match="has 5 points but x has 10"):
        viz.plot_timeseries(np.arange(10), {"A": np.arange(5)})


def test_plot_timeseries_log_scale():
    ax = viz.plot_timeseries(np.arange(5), {"A": np.arange(1, 6)}, logy=True)
    assert ax.get_yscale() == "log"


def test_profile_with_field_has_two_shared_panels():
    profile = np.array([4.0, 3.0, 2.0, 1.0])
    psi = np.linspace(0, 1, 4)
    fig, (top, bot) = viz.plot_profile_with_field(profile, psi, gamma=2.0)
    assert len(fig.axes) == 2
    assert top.get_xlim() == bot.get_xlim(), "panels must share the x axis"
    assert "2" in bot.get_ylabel()          # gamma is annotated


# ------------------------------------------------------- particle scatter


def test_scatter_particles_draws_one_point_per_particle_in_2d():
    lat = Lattice(shape=(6, 6), voxel_nm=20.0)
    counts = np.zeros(lat.shape, dtype=np.int64)
    counts[2, 3] = 5
    counts[4, 1] = 2
    ax = viz.scatter_particles(counts, lat)
    assert sum(c.get_offsets().shape[0] for c in ax.collections) == 7
    assert "7 particles" in ax.get_title()


def test_scatter_particles_jitters_inside_the_voxel():
    """Points must land within the voxel they belong to, not on its centre."""
    lat = Lattice(shape=(4, 4), voxel_nm=20.0)
    counts = np.zeros(lat.shape, dtype=np.int64)
    counts[1, 2] = 200
    ax = viz.scatter_particles(counts, lat, jitter=0.4)
    pts = ax.collections[0].get_offsets()
    assert np.all(np.abs(pts[:, 0] - 1) <= 0.4 + 1e-9)
    assert np.all(np.abs(pts[:, 1] - 2) <= 0.4 + 1e-9)
    assert pts[:, 0].std() > 0.1, "points should be spread, not stacked"


def test_scatter_particles_works_in_3d():
    lat = Lattice(shape=(4, 4, 4), voxel_nm=20.0)
    counts = np.zeros(lat.shape, dtype=np.int64)
    counts[1, 2, 3] = 6
    ax = viz.scatter_particles(counts, lat)
    assert hasattr(ax, "get_zlim"), "should be a 3D axes"
    assert "6 particles" in ax.get_title()


def test_scatter_particles_announces_subsampling():
    """A thinned picture must not pass as a complete one."""
    lat = Lattice(shape=(8, 8), voxel_nm=20.0)
    counts = np.full(lat.shape, 50, dtype=np.int64)      # 3200 particles
    ax = viz.scatter_particles(counts, lat, max_points=500)
    assert ax.collections[0].get_offsets().shape[0] == 500
    title = ax.get_title()
    assert "3200" in title and "%" in title, title


def test_scatter_particles_does_not_annotate_when_nothing_is_dropped():
    lat = Lattice(shape=(4, 4), voxel_nm=20.0)
    counts = np.full(lat.shape, 2, dtype=np.int64)
    ax = viz.scatter_particles(counts, lat, max_points=10_000)
    assert "%" not in ax.get_title()


def test_scatter_particles_rejects_an_empty_lattice():
    lat = Lattice(shape=(4, 4), voxel_nm=20.0)
    with pytest.raises(ValueError, match="every voxel is empty"):
        viz.scatter_particles(np.zeros(lat.shape, dtype=np.int64), lat)


def test_scatter_particles_rejects_a_wrong_shape():
    lat = Lattice(shape=(4, 4), voxel_nm=20.0)
    with pytest.raises(ValueError, match="pass one species'"):
        viz.scatter_particles(np.zeros((3, 3), dtype=np.int64), lat)


def test_scatter_particles_is_reproducible():
    lat = Lattice(shape=(5, 5), voxel_nm=20.0)
    counts = np.full(lat.shape, 3, dtype=np.int64)
    a = viz.scatter_particles(counts, lat, seed=7).collections[0].get_offsets()
    b = viz.scatter_particles(counts, lat, seed=7).collections[0].get_offsets()
    assert np.allclose(a, b)


def test_scatter_species_gives_each_species_its_own_colour():
    lat, st = make_state()
    ax = viz.scatter_species(st)
    colours = {tuple(np.ravel(c.get_facecolor())[:3]) for c in ax.collections}
    assert len(colours) == st.n_species
    assert ax.get_legend() is not None


def test_scatter_species_skips_empty_species():
    lat = Lattice(shape=(6, 6), voxel_nm=20.0)
    sp = [Species("A", 4.0, np.zeros(1)), Species("empty", 4.0, np.zeros(1))]
    st = State(lat, sp, occupancy_cap=20)
    st.set_counts("A", np.full(lat.shape, 2, dtype=np.int64))
    ax = viz.scatter_species(st)
    assert len(ax.collections) == 1, "the empty species should not be drawn"
