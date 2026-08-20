# vex-rddme

**V**olume-**EX**cluded **R**eaction–**D**rift–**D**iffusion **M**aster **E**quation on a
2D or 3D lattice, in numpy.

- **Wang–Peskin–Elston / Scharfetter–Gummel** transport in a static potential
- **White-Bear / BMCSL** multi-species hard-sphere exclusion, with self-exclusion
- **Fröhner–Noé** acceptance for reversible reactions of order ≤ 2, declared in Python

Two dependencies: numpy and matplotlib. No CUDA, no HDF5, no SWIG, no numba, no
compile step. It runs in a notebook on a laptop.

## Quickstart

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

Construction reports what the guards found: σ/h consistency, the maximum attainable
packing fraction, the free-energy table, the binding constraint, and the CFL margin.

Add reactions with `sim.add_reaction("assoc", ["A","B"], ["C"], k_forward, k_reverse)`.
Order ≤ 2 per side. Reversible pairs hold detailed balance exactly.

Set `dim=3` for three dimensions. The physics is identical.

## What this is

A teaching and prototyping instrument. Read the method in an afternoon. Show it in a meeting. Try a mechanism
without a GPU.

Three notebooks check it against analytic results. Each prints the measured value beside
the prediction, so you verify the implementation instead of trusting it.

| notebook | checks | against |
|---|---|---|
| `01_volume_excluded_lattice_gas` | exclusion alone, in three parts | Carnahan-Starling, Poisson and the compressibility relation, BMCSL insertion work |
| `02_macromolecular_crowding` | exclusion × a reversible bimolecular reaction | flux balance, then the Minton shift |
| `03_reversible_reaction_ramp_potential` | transport × acceptance, exclusion off | `Q(x) ∝ e^{−Δφ}` |

Demonstration 1 has three parts, and each closes a gap the previous one leaves open.
Part 1 fixes the excess chemical potential at the mean density. Part 2 fixes its
curvature, which the first moment cannot see. Part 3 fixes the cross-species terms,
which no single-species measurement constrains.

## What this is not

**Not a production solver.** The production implementation is the CUDA drift-RDME
solver in Lattice Microbes 2.6. This package is independent of it.

**This package does not claim or test numerical agreement with Lattice Microbes
2.6.** Do not report results from here as if the production solver produced them. The
four analytic checks above pin correctness. Comparison against the kernel does not.

**Out of scope, deliberately:** filaments and the multi-structure lattice handler;
self-consistent fields (ψ is input, not state); smeared multi-voxel bodies; trajectory
I/O and checkpointing; exact hop integration for steep potentials.

## A 2D result refutes cheaply. It only suggests positively

- A mechanism that **fails** in 2D almost certainly fails in 3D. Refutations travel.
- A mechanism that **works** in 2D may still fail in 3D. Positive results do not travel.

Three things the method depends on are dimension-dependent: the bond-angle spectrum, the
non-crossing constraints, and the conflict rate between parallel moves at density. So a
2D demonstration says a mechanism is worth testing in 3D. It does not say the mechanism
holds there.

## Guards

Every condition that stops the package delivering the requested physics either aborts
or reports. The package never clamps, clips, or substitutes a value silently.

| guard | catches |
|---|---|
| σ/h consistency | diameters that admit no crowding at this voxel size |
| ξ₃ saturation | packing fraction reaching 1, where exclusion stops repelling |
| table radix | free-energy table too small for the occupancy cap |
| cap vs exclusion | the integer cap, not the free energy, limiting density |
| hop-probability sum | flux lost to clipping, checked **every step** |
| reaction saturation | rate constants so large that propensities go inert |
| mass conservation | any channel leaking particles |
| detailed balance | forward/reverse works that are not exact negatives |

### Choosing the occupancy cap

The cap is one integer, so it is blunt across species sizes. Twenty 8 nm spheres in a
20 nm voxel give ξ₃ = 0.67. Twenty 5 nm spheres give 0.16.

Pick the cap from the **largest** species. Aim for ξ₃ between 0.4 and 0.7 at the cap,
and leave headroom above the peak occupancy you expect.

- Too high: the largest species can reach ξ₃ ≥ 1. Construction refuses it.
- Too low: the cap limits density instead of the free energy. You get a warning.
- Too near the peak: arrivals collide in one step and trip the `occupancy-cap` guard.

That last case fails rather than repairs itself. Discarding the excess would break both
mass conservation and detailed balance. Do one of these steps:

- Reduce τ.
- Raise the cap.
- Use fewer particles.

## Performance

Per-step cost, 2 species, exclusion on, static ψ: **64² 0.87 ms**, **32³ 7.86 ms**. At
4 species and occupancy 2: 64² 2.67 ms, 128² 9.45 ms.

### Relaxation, and which lattice to use

Steps to stationary matters more than seconds of simulated time, because every demonstration is
an equilibrium measurement. Measured by releasing particles from the high-field half and
tracking the profile's centre of mass (`bench/relaxation.py`):

| lattice | N | τ (s) | ms/step | COM moved | steps | wall | ×10 for averaging |
|---|---|---|---|---|---|---|---|
| **64²** | 4 096 | 2.0e−5 | 1.01 | 21.9 vox | 18 900 | 19 s | **3.2 min** |
| **32³** | 32 768 | 1.0e−5 | 9.56 | 9.9 vox | 6 325 | 61 s | **10.1 min** |
| 128² | 16 384 | 1.0e−5 | 2.79 | 29.6 vox | 52 250 | 146 s | 24.3 min |

- **64² is the notebook default.** A few minutes per measurement.
- **32³ works for 3D.** Ten minutes. Slower, but not a token mode.
- **128² is for figures.** Twenty-five minutes is a go-away-and-come-back run.

### Exclusion sets the timestep, not the CFL bound

With exclusion active, the downhill Bernoulli factor binds τ, not `q ≤ 1/(2·dim)`. For a
strongly downhill move `B(u) → |u|`. The largest work a particle sheds is `μ_ex` at the
highest packing fraction it meets:

```
    2 · dim · q · μ_ex(η_max)  ≤  1
```

At η = 0.5, `μ_ex ≈ 17 kT`. The admissible τ is then an order of magnitude below the CFL
bound. Use `suggest_tau()` to compute it.

If the `hop-probability-sum` guard fires, τ is too large for the crowding that
developed. That is not a bug. The 128² row above needed a τ backoff for this reason.

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
  sim.py       assembly, and the construction guards
notebooks/     the three validation demonstrations
docs/          porting_to_lattice_microbes, supersedes_driftRDME_standalone
bench/         relaxation.py
tests/         242 tests
```

## References

- Wang, Peskin & Elston: Scharfetter–Gummel discretisation of the lattice
  Fokker–Planck equation
- Roth, Evans, Lang & Kahl 2002; Yu & Wu 2002: White-Bear / BMCSL fundamental-measure
  excess free energy
- Fröhner & Noé 2018: Metropolis acceptance for reactive lattice schemes
- Minton: macromolecular crowding shifting association equilibria
