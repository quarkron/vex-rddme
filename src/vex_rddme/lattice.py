"""Lattice geometry: dimension, the axial neighbour alphabet, and boundaries.

Directions are indexed ``d in [0, 2*dim)`` with ``axis = d // 2`` and
``sign = +1`` for even ``d``, ``-1`` for odd. So in 2D: d=0 is +x, d=1 is -x,
d=2 is +y, d=3 is -y.

Two array operations carry the geometry, and they roll in *opposite* directions
because they answer opposite questions:

  ``deposit(m, d)``: m particles left each voxel heading in direction d;
                        place them at their destinations.
  ``neighbour(a, d)``: at each voxel, read the value a holds at that voxel's
                        direction-d neighbour.

Getting these two confused is a silent one-voxel shift in the drift, which is
why ``tests/test_lattice.py`` pins both against a single hand-placed particle.

Boundaries are reflecting in the sense that a particle may not leave the lattice:
``can_leave(d)`` is False on the face a direction-d hop would exit, and the hop
layer zeroes those probabilities. Nothing wraps, so no mass crosses a face.
"""

from __future__ import annotations

import numpy as np


class Lattice:
    """A rectangular lattice in 2 or 3 dimensions with axial neighbours.

    Parameters
    ----------
    shape : tuple of int
        Voxel counts per axis. Length fixes the dimension (2 or 3).
    voxel_nm : float
        Edge length of a cubic voxel, in nanometres.
    dim : int, optional
        Redundant with ``len(shape)``; if given it must agree. Present so calling
        code can be explicit, per the ``dim=2``/``dim=3`` interface.
    """

    def __init__(self, shape, voxel_nm, dim=None):
        shape = tuple(int(s) for s in shape)
        if len(shape) not in (2, 3):
            raise ValueError(
                f"vex_rddme supports dim=2 and dim=3; got shape {shape} "
                f"({len(shape)} axes)"
            )
        if any(s < 1 for s in shape):
            raise ValueError(f"every axis must have at least one voxel; got {shape}")
        if dim is not None and int(dim) != len(shape):
            raise ValueError(
                f"dim={dim} disagrees with shape {shape} ({len(shape)} axes). "
                "Pass a shape with dim axes, or omit dim."
            )
        if not np.isfinite(voxel_nm) or voxel_nm <= 0:
            raise ValueError(f"voxel_nm must be positive and finite; got {voxel_nm}")

        self.shape = shape
        self.dim = len(shape)
        self.voxel_nm = float(voxel_nm)
        self.n_voxels = int(np.prod(shape))
        self.n_dirs = 2 * self.dim

        # Unit offset vector per direction, and the index of the reverse direction.
        self._axis = np.array([d // 2 for d in range(self.n_dirs)], dtype=np.int64)
        self._sign = np.array(
            [1 if d % 2 == 0 else -1 for d in range(self.n_dirs)], dtype=np.int64
        )
        offsets = np.zeros((self.n_dirs, self.dim), dtype=np.int64)
        for d in range(self.n_dirs):
            offsets[d, self._axis[d]] = self._sign[d]
        self.offsets = offsets

        # Reverse of d is its partner in the pair (even <-> odd).
        self.opposite = np.array(
            [d + 1 if d % 2 == 0 else d - 1 for d in range(self.n_dirs)],
            dtype=np.int64,
        )

        self._can_leave = tuple(self._build_can_leave(d) for d in range(self.n_dirs))

    # ---------------------------------------------------------------- geometry

    @property
    def voxel_volume_nm3(self) -> float:
        """Volume of one voxel in nm^3.

        In 2D the lattice is a slab one voxel thick, so the volume is still
        ``h**3``. Weighted densities are volume fractions, so they must be
        referred to a real volume regardless of the lattice dimension,
        using ``h**2`` in 2D would make the packing fraction dimensionless in
        the wrong way and every free energy would be wrong by a factor of h.
        """
        return self.voxel_nm ** 3

    def axis(self, d: int) -> int:
        return int(self._axis[d])

    def sign(self, d: int) -> int:
        return int(self._sign[d])

    def _build_can_leave(self, d: int) -> np.ndarray:
        """Boolean lattice mask: True where a direction-d hop stays on-lattice."""
        mask = np.ones(self.shape, dtype=bool)
        sl = [slice(None)] * self.dim
        # Moving +1 along an axis is impossible from the last index on that axis;
        # moving -1 is impossible from index 0.
        sl[self.axis(d)] = -1 if self.sign(d) > 0 else 0
        mask[tuple(sl)] = False
        return mask

    def can_leave(self, d: int) -> np.ndarray:
        """Mask of voxels from which a direction-d hop is possible."""
        return self._can_leave[d]

    # ------------------------------------------------------- array operations

    def deposit(self, moved: np.ndarray, d: int) -> np.ndarray:
        """Place particles that left in direction d at their destinations.

        ``moved`` is a lattice-shaped count array: ``moved[v]`` particles departed
        voxel ``v`` heading in direction ``d``. The result is lattice-shaped and
        holds those particles at their arrival voxels.

        Callers must have zeroed ``moved`` outside ``can_leave(d)``; otherwise the
        wrap in ``np.roll`` would carry mass across a face. ``Hop`` does this by
        construction, and ``assert_no_wrap`` checks it in tests.
        """
        return np.roll(moved, self.sign(d), axis=self.axis(d))

    def neighbour(self, values: np.ndarray, d: int) -> np.ndarray:
        """Read each voxel's direction-d neighbour's value.

        Result at voxel ``v`` is ``values`` evaluated at ``v + offset(d)``. Rolls
        by ``-sign``, the opposite of :meth:`deposit`: to bring a neighbour's value
        *here*, the array must shift toward us.

        At the face where the neighbour is off-lattice the returned value wraps and
        is meaningless; it is always multiplied by a zero probability from
        ``can_leave(d)`` before use.
        """
        return np.roll(values, -self.sign(d), axis=self.axis(d))

    def assert_no_wrap(self, moved: np.ndarray, d: int) -> None:
        """Raise if ``moved`` is non-zero where a direction-d hop cannot happen."""
        offending = moved[~self.can_leave(d)]
        if np.any(offending != 0):
            raise AssertionError(
                f"{int(np.count_nonzero(offending))} particle(s) would leave the "
                f"lattice in direction d={d} (axis {self.axis(d)}, "
                f"sign {self.sign(d):+d}). Boundary probabilities were not zeroed."
            )

    # ------------------------------------------------------------------ misc

    def flat(self, coord) -> int:
        """Flat C-order index of a coordinate tuple."""
        return int(np.ravel_multi_index(tuple(coord), self.shape))

    def coord(self, index: int):
        """Coordinate tuple of a flat C-order index."""
        return tuple(int(c) for c in np.unravel_index(int(index), self.shape))

    def cfl_limit(self) -> float:
        """Largest baseline per-direction hop probability the split admits.

        The per-direction probabilities plus the stay probability must sum to one,
        and there are ``2*dim`` directions, so ``q <= 1/(2*dim)`` at zero work.
        This is necessary but *not* sufficient: the Bernoulli factor exceeds one
        for downhill moves, so the realised sum must also be checked every step
        (see :mod:`vex_rddme.guards`).
        """
        return 1.0 / (2.0 * self.dim)

    def __repr__(self) -> str:
        return (
            f"Lattice(shape={self.shape}, voxel_nm={self.voxel_nm:g}, "
            f"dim={self.dim}, n_voxels={self.n_voxels}, n_dirs={self.n_dirs})"
        )
