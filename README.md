# vex-rddme

**V**olume-**EX**cluded **R**eaction–**D**rift–**D**iffusion **M**aster **E**quation on a
2D or 3D lattice, in numpy.

- **Wang–Peskin–Elston / Scharfetter–Gummel** lattice transport in a static potential
- **White-Bear / BMCSL** multi-species hard-sphere exclusion, with self-exclusion
- **Fröhner–Noé** acceptance for reversible reactions of order ≤ 2, declared in Python

Two runtime dependencies: numpy and matplotlib. No CUDA, no HDF5, no Protobuf, no
SWIG, no numba, no compile step. It runs in a notebook on a laptop.

```python
import numpy as np
from vex_rddme import Simulation, Species
from vex_rddme.guards import suggest_tau

# Hard spheres in a linear potential, on a 64x64 lattice.
ramp = np.arange(64) / 64
psi  = np.broadcast_to(ramp, (64, 64)).copy()[None, ...]

sim = Simulation(
    shape=(64, 64), voxel_nm=20.0,
    species=[Species("A", sigma_nm=8.0, gamma=np.array([2.0]))],
    occupancy_cap=20, psi=psi,
    D_um2_s=1.0,
    tau_s=suggest_tau(D_um2_s=1.0, voxel_nm=20.0, dim=2, eta_max=0.3),
)
sim.set_counts("A", np.full((64, 64), 3, dtype=np.int64))
sim.record_initial()

for _ in range(2000):
    sim.step()

sim.state.check_mass()          # exact, not approximate
print(sim.packing_fraction())   # mean xi_3
```

Construction prints what the guards found: the σ/h consistency check, the maximum
attainable packing fraction, the free-energy table that was built, whether exclusion or
the occupancy cap is the binding constraint, and the CFL margin.

Add reactions with `sim.add_reaction("assoc", ["A", "B"], ["C"], k_forward, k_reverse)`
— order ≤ 2 on each side, reversible pairs supported, detailed balance exact by
construction.

## What this is

A teaching and prototyping instrument. It exists so that the volume-excluded drift
RDME method can be read in an afternoon, demonstrated in a meeting, and used to try a
mechanism without a GPU allocation.

Four notebooks validate it against results with analytic answers — reaction acceptance
against `Q(x)`, extracted excess chemical potential against Carnahan–Starling,
two-species depletion, and crowding-shifted equilibrium against BMCSL. Each prints the
measured value beside the prediction, so you can check the implementation rather than
trust it.

## What this is not

**Not a production solver.** The production implementation is the CUDA drift-RDME
solver in Lattice Microbes 2.6. This package is a separate, independent
implementation of the same method.

**No numerical agreement with Lattice Microbes 2.6 is claimed or tested.** Results
from this package are not a substitute for results from the production solver, and
should not be reported as if they were. Its correctness is pinned by the four analytic
checks above, not by comparison against the kernel.

**Out of scope**, deliberately: filaments and the multi-structure lattice handler;
self-consistent field feedback (the particles here do not source the field they feel —
`ψ` is input, not state); smeared multi-voxel bodies; trajectory I/O and
checkpoint/restart; exact hop integration for steep potentials.

## A 2D result refutes cheaply but only suggests positively

This asymmetry is worth stating plainly, because it is easy to over-read a
good-looking 2D figure:

- A mechanism that **fails** in 2D almost certainly fails in 3D. Refutations travel.
- A mechanism that **works** in 2D may still fail in 3D. Positive results do not travel.

The reason is that several things the method depends on are genuinely
dimension-dependent: the bond-angle spectrum available to a chain, non-crossing
constraints, and the rate at which parallel moves conflict at a given density. A 2D
demonstration is evidence that a mechanism is *worth testing* in 3D, not evidence that
it holds there.

Set `dim=3` to run in three dimensions. The notebook figures are 2D because 2D
renders legibly and instantly; the physics is identical either way.

## Guards

Every condition under which the package cannot deliver the requested physics either
aborts or reports. Nothing is clamped, clipped, or substituted silently:

| guard | catches |
|---|---|
| σ/h consistency | sphere diameters that admit no crowding at the chosen voxel size |
| ξ₃ saturation | packing fraction reaching 1, where exclusion silently stops repelling |
| table radix bound | free-energy table too small for the occupancy cap |
| hop-probability sum | flux silently lost to clipping — checked **every step**, since exclusion work grows as crowding develops |
| reaction saturation | rate constants so large that propensities go inert |
| mass conservation | any transport or reaction channel leaking particles |
| detailed balance | forward/reverse reaction works that are not exact negatives |

### Choosing the occupancy cap

The cap is a single integer, so it is a blunt instrument when species differ in size:
twenty 8 nm spheres in a 20 nm voxel is a packing fraction of 0.67, while twenty 5 nm
spheres is 0.16. Pick the cap so that the **largest** species at the cap reaches a
sensible packing fraction — roughly 0.4 to 0.7 — and leave headroom above the peak
occupancy you expect:

- too high, and the largest species can reach ξ₃ ≥ 1 (rejected at construction);
- too low, and the cap rather than the free energy limits density (warned about);
- too close to the peak occupancy, and arrivals colliding within one step will trip
  the `occupancy-cap` guard mid-run.

That last case is a failure rather than a silent repair, because discarding the excess
would break both mass conservation and detailed balance. Reduce τ, raise the cap, or
lower the particle count.

## Performance

Measured per-step cost (2 species, exclusion on, static ψ):

| lattice | ms/step |
|---|---|
| 64² | 0.87 |
| 32³ | 7.86 |

At 4 species and occupancy 2: 64² 2.67 ms/step, 128² 9.45 ms/step.

### Relaxation, and which lattice to use

The number that matters is not seconds of simulated time but **steps to stationary**:
every validation rung is an equilibrium measurement. Measured by releasing particles
from the high-field half of the box and tracking the profile's centre of mass
(`bench/relaxation.py`):

| lattice | N | τ (s) | ms/step | COM moved | steps to stationary | wall | ×10 for averaging |
|---|---|---|---|---|---|---|---|
| **64²** | 4 096 | 2.0e−5 | 1.01 | 21.9 vox | 18 900 | 19 s | **3.2 min** |
| **32³** | 32 768 | 1.0e−5 | 9.56 | 9.9 vox | 6 325 | 61 s | **10.1 min** |
| 128² | 16 384 | 1.0e−5 | 2.79 | 29.6 vox | 52 250 | 146 s | 24.3 min |

- **64² is the notebook default.** Equilibrated and averaged in a few minutes.
- **32³ works for 3D.** Ten minutes for a full measurement — slower, but not a token
  parity mode.
- **128² is for figures, not for a notebook.** Twenty-five minutes is a
  go-and-do-something-else run.

### The timestep is set by exclusion, not by the CFL bound

With volume exclusion the binding constraint on τ is the *downhill* Bernoulli factor,
not the bare `q ≤ 1/(2·dim)`. For a strongly downhill move `B(u) → |u|`, and the
largest work a particle can shed is `μ_ex` at the highest packing fraction it meets:

```
    2 · dim · q · μ_ex(η_max)  ≤  1
```

At η = 0.5, `μ_ex ≈ 17 kT`, so the admissible τ is an order of magnitude below what
the CFL bound alone suggests. `suggest_tau()` computes it:

```python
from vex_rddme.guards import suggest_tau
tau = suggest_tau(D_um2_s=1.0, voxel_nm=20.0, dim=2, eta_max=0.3)
```

The 128² row above needed a τ backoff for exactly this reason. If the
`hop-probability-sum` guard fires, the timestep is too large for the crowding that
developed — not a bug.

## Layout

```
src/vex_rddme/
  lattice.py   dimension, neighbour offsets, boundaries
  state.py     n[species, voxel] counts, occupancy cap, mass bookkeeping
  vex.py       weighted densities, BMCSL free energy, the integer table
  hop.py       Bernoulli hop, field coupling, self-excluded exclusion work
  react.py     reversible order ≤ 2 reactions, Fröhner–Noé acceptance
  guards.py    the loud-failure contract
  observe.py   density profiles, μ_ex extraction, Q(x), K_eq
  viz.py       2D rendering and animation, 1D profiles
notebooks/
  01_reaction_acceptance.ipynb          Q(x) vs exp(-dPhi)
  02_excess_chemical_potential.ipynb    mu_ex vs Carnahan-Starling
  03_depletion.ipynb                    multi-species BMCSL insertion work
  04_crowding_shifts_equilibrium.ipynb  flux balance + the Minton shift
docs/
  porting_to_lattice_microbes.md        where each piece lands in the CUDA solver
  supersedes_driftRDME_standalone.md    what was absorbed from the old notebook
bench/
  relaxation.py                         steps-to-stationary measurement
tests/
```

## References

- Wang, Peskin & Elston — Scharfetter–Gummel discretisation of the lattice
  Fokker–Planck equation
- Roth, Evans, Lang & Kahl 2002; Yu & Wu 2002 — White-Bear / BMCSL fundamental-measure
  excess free energy
- Fröhner & Noé 2018 — Metropolis acceptance for reactive lattice schemes
- Minton — macromolecular crowding shifting association equilibria
