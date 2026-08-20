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

## Demo Jupyter notebooks

| notebook | checks | against |
|---|---|---|
| `01_volume_excluded_lattice_gas` | exclusion alone, in three parts | Carnahan-Starling, Poisson and the compressibility relation, BMCSL insertion work |
| `02_macromolecular_crowding` | exclusion × a reversible bimolecular reaction | flux balance, then the Minton shift |
| `03_reversible_reaction_ramp_potential` | transport × acceptance, exclusion off | `Q(x) ∝ e^{−Δφ}` |

Demonstration 1 has three parts, and each closes a gap the previous one leaves open.
Part 1 fixes the excess chemical potential at the mean density. Part 2 fixes its
curvature, which the first moment cannot see. Part 3 fixes the cross-species terms,
which no single-species measurement constrains.

## Tutorials: launching simulations

Each step below is runnable on its own and builds on the one before. Step counts are
small so they finish in seconds; a real measurement needs the numbers in
[Relaxation, and which lattice to use](#relaxation-and-which-lattice-to-use).

### 1. The smallest useful run

An ideal lattice gas: no potential, no exclusion, no reactions. `sigma_nm=0` makes a
species ideal, so it occupies no volume.

```python
import numpy as np
from vex_rddme import Simulation, Species

sim = Simulation(
    shape=(32, 32),          # 2D. Use a 3-tuple for 3D.
    voxel_nm=20.0,           # voxel edge h
    species=[Species("A", sigma_nm=0.0, gamma=np.zeros(0))],
    occupancy_cap=200,       # most particles one voxel may hold
    D_um2_s=1.0,             # diffusion coefficient
    tau_s=2.0e-5,            # timestep
    seed=0,
)
sim.seed_uniform("A", 2000)  # scatter 2000 particles at random
sim.record_initial()         # freeze the reference for the mass check

for _ in range(500):
    sim.step()

sim.state.check_mass()       # exact: integers in, integers out
print(sim.state.totals())    # [2000]
```

`gamma=np.zeros(0)` declares no basis fields, so `psi` is not needed. Two things are
worth doing on every run: `record_initial()` once at the start, and `check_mass()`
whenever you want proof that nothing leaked.

### 2. Add a one-body potential

A potential is any per-voxel array you supply. It is an **input**: the package never
modifies it. Each species declares how strongly it feels each field through `gamma`.

```python
import numpy as np
from vex_rddme import Simulation, Species

N = 32
ramp = np.arange(N) / N                              # psi rising 0 to 1
psi = np.broadcast_to(ramp, (N, N)).copy()[None, ...]  # shape (n_fields, N, N)

sim = Simulation(
    shape=(N, N), voxel_nm=20.0,
    species=[Species("A", sigma_nm=0.0, gamma=np.array([2.0]))],  # gamma = 2
    occupancy_cap=200, psi=psi, D_um2_s=1.0, tau_s=2.0e-5, seed=0,
)
sim.seed_uniform("A", 4000)
sim.record_initial()
for _ in range(3000):
    sim.step()

# rho ~ exp(-gamma * psi), so the low-psi edge fills up
profile = sim.state.lattice_view("A").sum(axis=0)
print(f"low-psi edge {profile[0]:.0f}, high-psi edge {profile[-1]:.0f}")
```

The sign convention: positive `gamma` means the species is **repelled** from high `psi`.
At equilibrium the density follows `exp(-phi)` with `phi = sum_k gamma[k] * psi[k]`.

### 3. Add volume exclusion

Give a species a diameter and it becomes a hard sphere. Now the timestep is set by
exclusion rather than by the CFL bound, so ask `suggest_tau` for it.

```python
import numpy as np
from vex_rddme import Simulation, Species
from vex_rddme.guards import suggest_tau

VOXEL, SIGMA, CAP, N_PER_VOXEL = 20.0, 8.0, 20, 3
dxi3 = (np.pi / 6) * SIGMA**3 / VOXEL**3      # packing fraction per particle
eta_expected = 3.0 * N_PER_VOXEL * dxi3       # allow the field to triple the peak

tau = suggest_tau(D_um2_s=1.0, voxel_nm=VOXEL, dim=2, eta_max=eta_expected)
print(f"eta at the mean density {N_PER_VOXEL * dxi3:.3f}, suggested tau {tau:.2e} s")

N = 32
ramp = np.arange(N) / N
psi = np.broadcast_to(ramp, (N, N)).copy()[None, ...]

sim = Simulation(
    shape=(N, N), voxel_nm=VOXEL,
    species=[Species("A", sigma_nm=SIGMA, gamma=np.array([2.0]))],
    occupancy_cap=CAP, psi=psi, D_um2_s=1.0, tau_s=tau, seed=0,
)
# seed an exact uniform count, not a random one: random seeding creates
# high-occupancy outliers whose exclusion work is large
sim.set_counts("A", np.full((N, N), N_PER_VOXEL, dtype=np.int64))
sim.record_initial()
for _ in range(2000):
    sim.step()

sim.state.check_mass()
print(f"mean packing fraction {sim.packing_fraction():.4f}")
```

### 4. Add a reversible reaction

You declare reactions in Python. Order at most 2 on each side. A non-zero
`k_reverse` makes the pair reversible, and detailed balance then holds exactly.

```python
import numpy as np
from vex_rddme import Simulation, Species

N = 24
sim = Simulation(
    shape=(N, N), voxel_nm=20.0,
    species=[
        Species("A", sigma_nm=0.0, gamma=np.zeros(0)),
        Species("B", sigma_nm=0.0, gamma=np.zeros(0)),
        Species("C", sigma_nm=0.0, gamma=np.zeros(0)),
    ],
    occupancy_cap=400, D_um2_s=1.0, tau_s=2.0e-5, exclusion=False, seed=0,
)
sim.add_reaction("assoc", ["A", "B"], ["C"],
                 k_forward=40.0, k_reverse=200.0,
                 typical_reactant_product=9.0)   # only used by the saturation guard
sim.seed_uniform("A", 1500)
sim.seed_uniform("B", 1500)
sim.record_initial()

for _ in range(4000):
    sim.step()

a, b, c = sim.state.counts
Q = c.mean() / (a.astype(float) * b.astype(float)).mean()
print(f"Q = {Q:.3f}, k_F/k_R = {40.0/200.0:.3f}")
print("detailed-balance residual:",
      f"{sim.reactions.detailed_balance_residual(np.random.default_rng(0)):.2e}")
```

Use the mean of the **product** `<n_A n_B>` in the denominator, not the product of the
means. The reactants are correlated through the conservation law.

### 5. Add an inert crowder

An inert species excludes volume but never reacts and feels no field, so you can vary
crowding without touching anything else. It is the sweep variable for a crowding study.

```python
import numpy as np
from vex_rddme import Simulation, Species
from vex_rddme.guards import suggest_tau

sA = sB = 6.0
sC = (sA**3 + sB**3) ** (1/3)     # volume-conserving product: xi_3 unchanged
N, CAP = 24, 24
tau = suggest_tau(D_um2_s=1.0, voxel_nm=20.0, dim=2, eta_max=0.45)

for n_crowder in (0, 4):
    sim = Simulation(
        shape=(N, N), voxel_nm=20.0,
        species=[
            Species("A", sA, np.zeros(0)),
            Species("B", sB, np.zeros(0)),
            Species("C", sC, np.zeros(0)),
            Species("X", 7.0, np.zeros(0), inert=True),   # the crowder
        ],
        occupancy_cap=CAP, D_um2_s=1.0, tau_s=tau, seed=11,
        attach_log_handler=False,
    )
    sim.add_reaction("assoc", ["A", "B"], ["C"], 60.0, 300.0,
                     typical_reactant_product=9.0)
    sim.set_counts("A", np.full((N, N), 3, dtype=np.int64))
    sim.set_counts("B", np.full((N, N), 3, dtype=np.int64))
    if n_crowder:
        sim.set_counts("X", np.full((N, N), n_crowder, dtype=np.int64))
    sim.record_initial()
    for _ in range(1500):
        sim.step()
    print(f"crowder {n_crowder}: xi_3^X = {sim.crowder_packing_fraction():.4f}")
```

Construction refuses a crowder that has non-zero `gamma`, and refuses to put one in a
reaction. That is what keeps it a clean independent variable.

### 6. Run in three dimensions

Pass a 3-tuple. Nothing else changes, except that the CFL bound tightens from
`1/(2*2)` to `1/(2*3)`.

```python
import numpy as np
from vex_rddme import Simulation, Species
from vex_rddme.guards import suggest_tau

tau = suggest_tau(D_um2_s=1.0, voxel_nm=20.0, dim=3, eta_max=0.3)
sim = Simulation(
    shape=(16, 16, 16), voxel_nm=20.0,             # <- 3-tuple
    species=[Species("A", sigma_nm=8.0, gamma=np.zeros(0))],
    occupancy_cap=20, D_um2_s=1.0, tau_s=tau, seed=0,
)
sim.set_counts("A", np.full((16, 16, 16), 3, dtype=np.int64))
sim.record_initial()
for _ in range(300):
    sim.step()
sim.state.check_mass()
print(f"dim {sim.lattice.dim}, CFL limit {sim.lattice.cfl_limit():.4f}")
```

3D costs about ten times more per step than 2D at a comparable voxel count. See the
relaxation table for what that means in wall time.

### 7. Sample correctly

Every measurement here is an equilibrium average, so two things matter: discard the
approach to equilibrium, and space the samples. Consecutive configurations are
strongly correlated, so sampling every step inflates the apparent sample count without
adding information.

```python
import numpy as np
from vex_rddme import Simulation, Species
from vex_rddme.observe import Series, project

N, STEPS = 24, 3000
BURN_IN, SAMPLE_EVERY = STEPS // 3, 25      # discard a third, then space the samples

ramp = np.arange(N) / N
psi = np.broadcast_to(ramp, (N, N)).copy()[None, ...]
sim = Simulation(
    shape=(N, N), voxel_nm=20.0,
    species=[Species("A", 0.0, np.array([2.0]))],
    occupancy_cap=400, psi=psi, D_um2_s=1.0, tau_s=2.0e-5, seed=0,
)
sim.seed_uniform("A", 4000)
sim.record_initial()

rho = Series("density")
for i in range(STEPS):
    sim.step()
    if i >= BURN_IN and i % SAMPLE_EVERY == 0:
        rho.add(project(sim.state.lattice_view("A"), sim.lattice) / N)

print(f"{rho.n} samples, mean {rho.mean.mean():.3f}, typical error {rho.sem.mean():.4f}")
```

### 8. Read the guard output

Construction reports what it found. A normal run looks like this:

```
INFO [sigma-voxel] maximum attainable packing fraction is 0.6702 ...
INFO [exclusion] free-energy table built: 1 hard-core species, radix 22 ...
INFO [packing] occupancy at xi3 = 0.5 per species: A: 14.9
INFO [cap-vs-exclusion] A voxel at the cap reaches packing fraction 0.67.
                        Exclusion is the binding constraint, as intended.
INFO [cfl] baseline hop probability q = 0.0128 (limit 0.25, 2D)
```

Read the last two. `cap-vs-exclusion` tells you whether the physics or the integer cap
is limiting density. `cfl` tells you how much timestep headroom you have.

Pass `attach_log_handler=False` to silence the stream and inspect
`sim.report.records` instead. A failure raises `GuardViolation`, and every message
names the values, the limit, and the actions that would fix it.

```python
from vex_rddme.guards import GuardViolation
import numpy as np
from vex_rddme import Simulation, Species

try:
    Simulation(shape=(8, 8), voxel_nm=10.0,
               species=[Species("big", sigma_nm=10.0, gamma=np.zeros(0))],
               occupancy_cap=4, seed=0, attach_log_handler=False)
except GuardViolation as exc:
    print(exc)
```

### 9. Choose the parameters

The three that interact:

**Timestep.** Use `suggest_tau`. If the `hop-probability-sum` guard fires mid-run, the
crowding grew past what that timestep supports: halve it and rerun. That is the guard
working, not a bug.

**Occupancy cap.** Pick it from the *largest* species, aiming for a packing fraction
between 0.4 and 0.7 at the cap, and leave headroom above the peak occupancy you
expect. See [Choosing the occupancy cap](#choosing-the-occupancy-cap).

**Lattice.** 64 squared for a notebook, 32 cubed for 3D, 128 squared for figures. The
relaxation table gives the wall time for each.

A useful sanity check before a long run: do a short one and read the guard lines. If
`cap-vs-exclusion` warns, or `cfl` shows almost no headroom, fix that first.

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

## References

to add. see manuscript draft for vex-rddme.
