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
    as HTML text — the check is that one image per frame is embedded."""
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
