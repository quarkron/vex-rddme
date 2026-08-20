"""Plotting helpers. Imports matplotlib, so it is not imported by ``vex_rddme``.

Import it explicitly when you want figures::

    from vex_rddme import viz

2D is the default for figures because a lattice renders directly as an image and is
legible at a glance. A 3D lattice has no such rendering, so :func:`show_lattice`
reduces it — by summing along an axis (the default, which keeps every particle
visible) or by slicing (which shows a true cross-section).
"""

from __future__ import annotations

import numpy as np

import matplotlib.pyplot as plt


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

    The standard figure for every validation rung: the reader should be able to see
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
    """Excess chemical potential against packing fraction (rung 2)."""
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
    binary is needed — ffmpeg is not a dependency of this package.

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
