"""Observables: profiles with error bars, and the analytic comparisons.

Every validation rung compares a measured quantity against an analytic prediction,
so measurements carry a standard error rather than a bare number — a discrepancy is
only meaningful next to the noise on it.

Samples drawn from a running simulation are strongly correlated, so the standard
error over samples underestimates the true uncertainty unless samples are spaced by
more than a correlation time. :class:`Series` reports the naive standard error and
leaves the spacing to the caller; the notebooks sample every 25 steps for that
reason.
"""

from __future__ import annotations

import numpy as np


def project(values, lattice, axis=-1):
    """Sum a lattice-shaped (or flat) array over every axis except ``axis``.

    Reduces a 2D or 3D field to a 1D profile. Summing rather than averaging keeps
    the result a particle count, which is what the barometric and equilibrium
    comparisons are stated in.
    """
    arr = np.asarray(values)
    if arr.shape == (lattice.n_voxels,):
        arr = arr.reshape(lattice.shape)
    if arr.shape != lattice.shape:
        raise ValueError(
            f"expected shape {lattice.shape} or ({lattice.n_voxels},); got {arr.shape}"
        )
    keep = axis % lattice.dim
    other = tuple(a for a in range(lattice.dim) if a != keep)
    return arr.sum(axis=other) if other else arr


class Series:
    """Accumulates samples of a vector observable and reports mean and error.

    Uses Welford's online algorithm rather than accumulating ``sum`` and ``sum of
    squares``. The textbook ``E[x^2] - E[x]^2`` form loses catastrophically to
    cancellation when the mean is large relative to the spread: for a mean of 1e8 with
    a true standard deviation of 1, it reports about 4.4 — a 300% error, silently.
    Every validation rung states its claim next to an error bar, so a silently wrong
    error bar is precisely the failure this package refuses elsewhere.

    ``sem`` is the naive standard error of the mean over samples. It is a lower bound
    on the true uncertainty when samples are correlated, which they are when drawn from
    a running simulation; space them by more than a correlation time.
    """

    def __init__(self, name=""):
        self.name = name
        self._mean = None
        self._m2 = None
        self.n = 0

    def add(self, value):
        v = np.asarray(value, dtype=np.float64)
        if self._mean is None:
            self._mean = np.zeros_like(v)
            self._m2 = np.zeros_like(v)
        self.n += 1
        delta = v - self._mean
        self._mean = self._mean + delta / self.n
        self._m2 = self._m2 + delta * (v - self._mean)
        return self

    @property
    def mean(self):
        if self.n == 0:
            raise RuntimeError(f"Series {self.name!r} has no samples")
        return self._mean.copy()

    @property
    def std(self):
        """Population standard deviation (matches ``np.std`` with default ddof=0)."""
        if self.n == 0:
            raise RuntimeError(f"Series {self.name!r} has no samples")
        if self.n < 2:
            return np.zeros_like(self._mean)
        return np.sqrt(np.maximum(self._m2 / self.n, 0.0))

    @property
    def sem(self):
        if self.n < 2:
            return np.zeros_like(self._mean)
        return self.std / np.sqrt(self.n)

    def __repr__(self):
        return f"Series({self.name!r}, n={self.n})"


# ---------------------------------------------------------------------------
# Rung 2: excess chemical potential from a stationary profile
# ---------------------------------------------------------------------------


def mu_ex_from_profile(density, phi):
    """Excess chemical potential, up to an additive constant, from a profile.

    At equilibrium the total chemical potential is uniform in space::

        mu_ideal(rho) + mu_ex(rho) + phi  =  const
        ln(rho)      + mu_ex(rho) + phi  =  const

    so ``mu_ex(rho(x)) = const - ln rho(x) - phi(x)``. The constant is not
    recoverable from a single profile, which is why the comparison against
    Carnahan-Starling is done on the *shape*: see :func:`align_additive_constant`.

    A wrong self-exclusion term changes the shape, not just the offset, so this is a
    genuine test of it.
    """
    density = np.asarray(density, dtype=np.float64)
    phi = np.asarray(phi, dtype=np.float64)
    if np.any(density <= 0):
        raise ValueError(
            "mu_ex extraction needs a strictly positive density everywhere; "
            f"{int(np.count_nonzero(density <= 0))} bin(s) are empty. Run longer, "
            "use more particles, or narrow the field range."
        )
    return -np.log(density) - phi


def align_additive_constant(measured, analytic):
    """Shift ``measured`` by the single constant that best matches ``analytic``.

    One free parameter for the whole curve, fixed by least squares (i.e. matching
    means). Everything else about the comparison is parameter-free.
    """
    measured = np.asarray(measured, dtype=np.float64)
    analytic = np.asarray(analytic, dtype=np.float64)
    return measured + float(np.mean(analytic - measured))


def relative_discrepancy(measured, predicted, scale=None):
    """Max and RMS relative discrepancy, reported against a common scale.

    ``scale`` defaults to the peak-to-peak range of ``predicted``. Using a range
    rather than pointwise magnitudes avoids dividing by values near zero, which
    would report a huge relative error on a curve that happens to cross zero.
    """
    measured = np.asarray(measured, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    if scale is None:
        scale = np.ptp(predicted)
    scale = float(scale)
    if scale <= 0:
        raise ValueError(
            "cannot form a relative discrepancy: the predicted curve is flat, so "
            "there is no scale to compare against"
        )
    err = np.abs(measured - predicted) / scale
    return {"max": float(err.max()), "rms": float(np.sqrt(np.mean(err ** 2)))}


# ---------------------------------------------------------------------------
# Rungs 1 and 4: reaction quotients
# ---------------------------------------------------------------------------


def reaction_quotient(counts, reactants, products):
    """``n_products / prod(n_reactants)`` per voxel, as numerator and denominator.

    Returned unreduced so a caller can average numerator and denominator separately.
    The equilibrium relation is ``<n_C> / <n_A n_B> = k_F/k_R``, with the mean of the
    *product* in the denominator: A and B are correlated through the conservation
    law, so the product of means does not satisfy the relation.
    """
    counts = np.asarray(counts)
    num = np.ones(counts.shape[1], dtype=np.float64)
    for s in products:
        num = num * counts[s].astype(np.float64)
    den = np.ones(counts.shape[1], dtype=np.float64)
    for s in reactants:
        den = den * counts[s].astype(np.float64)
    return num, den


class QuotientAccumulator:
    """Accumulates numerator and denominator of a reaction quotient.

    Averages the two separately and divides at the end, which is what the
    detailed-balance relation is stated in. Also tracks the per-sample ratio so a
    standard error can be reported.
    """

    def __init__(self, reactants, products, lattice=None, axis=None):
        self.reactants = tuple(reactants)
        self.products = tuple(products)
        self.lattice = lattice
        self.axis = axis
        self._num = Series("numerator")
        self._den = Series("denominator")

    def add(self, counts):
        num, den = reaction_quotient(counts, self.reactants, self.products)
        if self.axis is not None:
            if self.lattice is None:
                raise ValueError("axis-resolved accumulation needs a lattice")
            num = project(num, self.lattice, self.axis)
            den = project(den, self.lattice, self.axis)
        else:
            num, den = np.array([num.mean()]), np.array([den.mean()])
        self._num.add(num)
        self._den.add(den)
        return self

    @property
    def n(self):
        return self._num.n

    @property
    def quotient(self):
        den = self._den.mean
        if np.any(den <= 0):
            raise RuntimeError(
                "reaction quotient is undefined: the mean reactant product is zero "
                "in at least one bin. Run longer or use more particles."
            )
        return self._num.mean / den

    @property
    def sem(self):
        """Propagated standard error, treating numerator and denominator errors
        as independent — an approximation, since they are anticorrelated through
        the conservation law, so this errs on the conservative side."""
        num, den = self._num.mean, self._den.mean
        rel = np.sqrt(
            (self._num.sem / np.maximum(num, 1e-300)) ** 2
            + (self._den.sem / np.maximum(den, 1e-300)) ** 2
        )
        return np.abs(self.quotient) * rel


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def report_comparison(label, measured, predicted, sem=None, scale=None):
    """Format a measured-versus-predicted comparison as text.

    Every notebook prints one of these, so a reader does not have to judge
    agreement from a plot.
    """
    measured = np.atleast_1d(np.asarray(measured, dtype=np.float64))
    predicted = np.atleast_1d(np.asarray(predicted, dtype=np.float64))
    lines = [f"{label}"]
    if measured.size == 1:
        err = f" +/- {float(np.atleast_1d(sem)[0]):.4g}" if sem is not None else ""
        rel = abs(measured[0] - predicted[0]) / max(abs(predicted[0]), 1e-300)
        lines += [
            f"  measured   {measured[0]:.6g}{err}",
            f"  predicted  {predicted[0]:.6g}",
            f"  relative discrepancy  {rel * 100:.2f}%",
        ]
    else:
        d = relative_discrepancy(measured, predicted, scale=scale)
        lines += [
            f"  bins           {measured.size}",
            f"  max discrepancy  {d['max'] * 100:.2f}% of the predicted range",
            f"  rms discrepancy  {d['rms'] * 100:.2f}% of the predicted range",
        ]
        if sem is not None:
            sem = np.atleast_1d(np.asarray(sem, dtype=np.float64))
            scale_used = float(np.ptp(predicted)) if scale is None else float(scale)
            lines.append(
                f"  typical standard error  "
                f"{float(np.mean(sem)) / max(scale_used, 1e-300) * 100:.2f}% of range"
            )
    return "\n".join(lines)
