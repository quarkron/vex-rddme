"""Plotting helpers. Imports matplotlib, so it is not imported by ``vex_rddme``.

Import it explicitly when you want figures::

    from vex_rddme import viz

2D is the default for figures because a lattice renders directly as an image and is
legible at a glance. A 3D lattice has no such rendering, so :func:`show_lattice`
reduces it, by summing along an axis (the default, which keeps every particle
visible) or by slicing (which shows a true cross-section).
"""

from __future__ import annotations

import numpy as np

import matplotlib.pyplot as plt

# A small qualitative palette, reused so a species keeps its colour across figures.
_SERIES_COLOURS = ("#1f4e79", "#c1440e", "#2e7d32", "#6a3d9a", "#b8860b", "#00838f")


def _as_image(values, lattice, reduce="sum", index=None):
    """Reduce a lattice-shaped array to something imshow can draw."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.shape == (lattice.n_voxels,):
        arr = arr.reshape(lattice.shape)
    if arr.shape != lattice.shape:
        raise ValueError(
            f"expected shape {lattice.shape} or ({lattice.n_voxels},); got {arr.shape}"
        )
    if lattice.dim == 2:
        return arr
    if reduce == "sum":
        return arr.sum(axis=0)
    if reduce == "slice":
        k = lattice.shape[0] // 2 if index is None else int(index)
        return arr[k]
    raise ValueError(f"reduce must be 'sum' or 'slice'; got {reduce!r}")


def show_lattice(
    values,
    lattice,
    ax=None,
    reduce="sum",
    index=None,
    title=None,
    cmap="viridis",
    colorbar=True,
    vmin=None,
    vmax=None,
):
    """Draw one lattice field as an image. Returns the axes."""
    img = _as_image(values, lattice, reduce=reduce, index=index)
    if ax is None:
        _, ax = plt.subplots(figsize=(4.2, 3.6))
    im = ax.imshow(
        img.T, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest"
    )
    ax.set_xlabel("x (voxels)")
    ax.set_ylabel("y (voxels)" if lattice.dim == 2 else "y (voxels)")
    if title:
        ax.set_title(title, fontsize=10)
    if colorbar:
        ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return ax


def show_species(state, ax=None, **kwargs):
    """Draw every species side by side, on a shared colour scale per species."""
    n = state.n_species
    if ax is None:
        _, ax = plt.subplots(1, n, figsize=(3.6 * n, 3.2), squeeze=False)
        ax = ax[0]
    for i, sp in enumerate(state.species):
        show_lattice(
            state.counts[i], state.lattice, ax=ax[i],
            title=f"{sp.name}  (sigma {sp.sigma_nm:g} nm, N={int(state.counts[i].sum())})",
            **kwargs,
        )
    return ax


def plot_profile(
    measured,
    predicted=None,
    sem=None,
    ax=None,
    xlabel="voxel along the field axis",
    ylabel="",
    label_measured="measured",
    label_predicted="predicted",
    title=None,
):
    """Measured profile with error bars against an analytic prediction.

    The standard figure for every validation demonstration: the reader should be able to see
    agreement, and the accompanying printed numbers make it quantitative.
    """
    measured = np.asarray(measured, dtype=np.float64)
    x = np.arange(measured.size)
    if ax is None:
        _, ax = plt.subplots(figsize=(5.2, 3.6))
    if sem is not None:
        ax.errorbar(
            x, measured, yerr=np.asarray(sem, dtype=np.float64), fmt="o", ms=3.5,
            lw=1.0, capsize=2, label=label_measured, color="#1f4e79", zorder=3,
        )
    else:
        ax.plot(x, measured, "o", ms=3.5, label=label_measured, color="#1f4e79", zorder=3)
    if predicted is not None:
        ax.plot(
            x, np.asarray(predicted, dtype=np.float64), "-", lw=1.8,
            label=label_predicted, color="#c1440e", zorder=2,
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, fontsize=10)
    ax.legend(frameon=False, fontsize=9)
    ax.margins(x=0.02)
    return ax


def plot_mu_ex(eta, measured, analytic, sem=None, ax=None):
    """Excess chemical potential against packing fraction (demonstration 1, part 1)."""
    order = np.argsort(np.asarray(eta))
    eta = np.asarray(eta)[order]
    measured = np.asarray(measured)[order]
    analytic = np.asarray(analytic)[order]
    if ax is None:
        _, ax = plt.subplots(figsize=(5.2, 3.6))
    if sem is not None:
        ax.errorbar(
            eta, measured, yerr=np.asarray(sem)[order], fmt="o", ms=3.5, lw=1.0,
            capsize=2, color="#1f4e79", label="measured (from the profile)", zorder=3,
        )
    else:
        ax.plot(eta, measured, "o", ms=3.5, color="#1f4e79",
                label="measured (from the profile)", zorder=3)
    ax.plot(eta, analytic, "-", lw=1.8, color="#c1440e",
            label="Carnahan-Starling", zorder=2)
    ax.set_xlabel(r"packing fraction $\xi_3$")
    ax.set_ylabel(r"$\mu_{\rm ex}$  ($k_BT$)")
    ax.legend(frameon=False, fontsize=9)
    return ax


def animate(frames, lattice, reduce="sum", interval=120, cmap="viridis", titles=None):
    """Animate a sequence of lattice fields. Returns an HTML string for a notebook.

    Uses ``to_jshtml``, so every frame is embedded as a base64 PNG and no writer
    binary is needed. ffmpeg is not a dependency of this package.

    One caveat: matplotlib's ``to_jshtml`` output links a font-awesome stylesheet
    from a CDN for the playback-button icons. The animation itself is fully embedded
    and plays offline; only those icons are missing without a network connection.
    """
    from matplotlib import animation

    imgs = [_as_image(f, lattice, reduce=reduce) for f in frames]
    if not imgs:
        raise ValueError("no frames to animate")
    vmin = min(float(i.min()) for i in imgs)
    vmax = max(float(i.max()) for i in imgs)

    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    im = ax.imshow(
        imgs[0].T, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax,
        interpolation="nearest",
    )
    ax.set_xlabel("x (voxels)")
    ax.set_ylabel("y (voxels)")
    ttl = ax.set_title(titles[0] if titles else "", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    def update(k):
        im.set_data(imgs[k].T)
        if titles:
            ttl.set_text(titles[k])
        return (im, ttl)

    anim = animation.FuncAnimation(
        fig, update, frames=len(imgs), interval=interval, blit=False
    )
    html = anim.to_jshtml()
    plt.close(fig)
    return html


def plot_timeseries(x, series, ax=None, xlabel="step", ylabel="count",
                    title=None, logy=False):
    """Totals against time, one line per label.

    ``series`` is a mapping from label to a sequence of the same length as ``x``.
    Use it to watch a reaction approach equilibrium, or to confirm that a total that
    should be conserved is flat.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(5.4, 3.6))
    x = np.asarray(x)
    for i, (label, y) in enumerate(series.items()):
        y = np.asarray(y, dtype=np.float64)
        if y.shape != x.shape:
            raise ValueError(
                f"series {label!r} has {y.shape[0]} points but x has {x.shape[0]}"
            )
        ax.plot(x, y, "-", lw=1.6, label=label,
                color=_SERIES_COLOURS[i % len(_SERIES_COLOURS)])
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, fontsize=10)
    ax.legend(frameon=False, fontsize=9)
    ax.margins(x=0.02)
    return ax


def plot_profile_with_field(profile, psi_line, sem=None, predicted=None,
                            gamma=None, figsize=(5.4, 4.4)):
    """A density profile above a strip showing the field that produced it.

    Two panels on a shared x axis: the profile on top, ``psi`` beneath. Seeing the
    field next to the response is the quickest way to tell a real gradient from a
    statistical one.
    """
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=figsize, sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08},
    )
    plot_profile(profile, predicted=predicted, sem=sem, ax=ax_top,
                 xlabel="", ylabel="mean occupancy")
    psi_line = np.asarray(psi_line, dtype=np.float64)
    ax_bot.plot(np.arange(psi_line.size), psi_line, "-", lw=1.6, color="#333")
    ax_bot.fill_between(np.arange(psi_line.size), psi_line, alpha=0.15, color="#333")
    label = r"$\psi$" if gamma is None else rf"$\psi$   ($\gamma$ = {gamma:g})"
    ax_bot.set_ylabel(label)
    ax_bot.set_xlabel("voxel along the field axis")
    ax_bot.margins(x=0.02)
    return fig, (ax_top, ax_bot)


def scatter_particles(counts, lattice, ax=None, max_points=40_000, seed=0,
                      jitter=0.4, size=4.0, colour="#1f4e79", alpha=0.55,
                      title=None, label=None):
    """Draw the lattice as particles, by scattering counts inside their voxels.

    The state is occupancy counts, not positions, so there are no particle
    trajectories to plot. Placing ``n`` points at uniformly random offsets inside a
    voxel that holds ``n`` particles gives a picture that reads like a particle
    system while representing exactly the same information. Works in 2D and 3D.

    ``jitter`` is the half-width of the offset in voxel units, so 0.5 fills the voxel
    and the default 0.4 leaves a faint lattice texture visible.

    Above ``max_points`` the points are subsampled, and the fraction kept is written
    into the axes title. Nothing is dropped silently.
    """
    arr = np.asarray(counts)
    if arr.ndim > 1 and arr.shape != lattice.shape:
        raise ValueError(
            f"pass one species' counts, shaped {lattice.shape} or "
            f"({lattice.n_voxels},); got {arr.shape}"
        )
    flat = arr.reshape(-1)
    if flat.size != lattice.n_voxels:
        raise ValueError(
            f"expected {lattice.n_voxels} voxels; got {flat.size}"
        )

    occupied = np.flatnonzero(flat)
    if occupied.size == 0:
        raise ValueError("no particles to draw: every voxel is empty")
    coords = np.stack(np.unravel_index(occupied, lattice.shape), axis=1)
    pts = np.repeat(coords.astype(np.float64), flat[occupied], axis=0)

    total = pts.shape[0]
    rng = np.random.default_rng(seed)
    kept = 1.0
    if total > max_points:
        pick = rng.choice(total, size=max_points, replace=False)
        pts = pts[pick]
        kept = max_points / total
    pts = pts + rng.uniform(-jitter, jitter, size=pts.shape)

    note = "" if kept == 1.0 else f"  [{kept*100:.0f}% of {total} shown]"
    if lattice.dim == 2:
        if ax is None:
            _, ax = plt.subplots(figsize=(4.4, 4.0))
        ax.scatter(pts[:, 0], pts[:, 1], s=size, c=colour, alpha=alpha,
                   linewidths=0, label=label)
        ax.set_xlim(-1, lattice.shape[0])
        ax.set_ylim(-1, lattice.shape[1])
        ax.set_aspect("equal")
        ax.set_xlabel("x (voxels)")
        ax.set_ylabel("y (voxels)")
    else:
        if ax is None:
            fig = plt.figure(figsize=(4.8, 4.4))
            ax = fig.add_subplot(projection="3d")
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=size, c=colour,
                   alpha=alpha, linewidths=0, label=label)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
    ax.set_title((title or f"{total} particles") + note, fontsize=10)
    return ax


def scatter_species(state, ax=None, max_points=40_000, seed=0, **kwargs):
    """Overlay every species as particles, one colour each.

    Ideal species are drawn too. Only the diameters differ in the physics, not in
    how the counts are stored, so the picture treats them alike.
    """
    if ax is None:
        if state.lattice.dim == 2:
            _, ax = plt.subplots(figsize=(4.8, 4.4))
        else:
            fig = plt.figure(figsize=(5.2, 4.8))
            ax = fig.add_subplot(projection="3d")
    for i, sp in enumerate(state.species):
        if state.counts[i].sum() == 0:
            continue
        scatter_particles(
            state.counts[i], state.lattice, ax=ax, max_points=max_points,
            seed=seed + i, colour=_SERIES_COLOURS[i % len(_SERIES_COLOURS)],
            label=f"{sp.name} (sigma {sp.sigma_nm:g} nm)", **kwargs,
        )
    ax.set_title(f"{state.n_species} species, "
                 f"{int(state.counts.sum())} particles", fontsize=10)
    ax.legend(frameon=False, fontsize=8.5, markerscale=2.5)
    return ax
