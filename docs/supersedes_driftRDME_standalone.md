# Supersession of `driftRDME_standalone.ipynb`

`~/ftsz_tests/driftRDME_standalone.ipynb` (28 May 2026) was the first numpy port of
the drift RDME: a ~150-line `DriftRDME` class covering the WPE/Scharfetter–Gummel hop
and Fröhner–Noé reaction acceptance, with three validated sections and a plotly viewer.
This package supersedes it.

## What was absorbed, and where it went

| notebook section | now lives in |
|---|---|
| §1 pure WPE diffusion, three species at γ ∈ {+2, 0, −2} | `tests/test_hop.py::test_three_couplings_in_one_field_each_follow_boltzmann` |
| §2 pure FN reaction, `Q(x)` diagnostic | `tests/test_react.py::test_local_equilibrium_ratio_follows_the_field`, and `notebooks/03_reversible_reaction_ramp_potential.ipynb` |
| §3 combined drift + reaction | covered by the two above plus `notebooks/01_volume_excluded_lattice_gas.ipynb` |
| optional 3D plotly viewer | `vex_rddme.viz` (matplotlib; `dim=3` reduces by sum or slice) |

The three-γ section is called out because the **negative**-coupling case was not
otherwise covered: the single-species tests only exercised γ ≥ 0.

## What changed, and why

**Count-based state was kept, but for a different reason.** The notebook used
per-species occupancy counts because that is the natural numpy expression. This package
uses them because they are the RDME's own state variable *and* because the measured
crossover against a particle representation sits at about one particle per voxel per
species. Volume-exclusion studies live above it, where count cost is nearly flat in
occupancy and particle cost is linear.

**`p_cond` is no longer precomputed once.** The notebook precomputed the conditional
hop probabilities per species at construction, which is correct only for a static
potential with no exclusion. With volume exclusion the work depends on the current
occupancy, so the exclusion arrays are rebuilt every step. The static field
contribution *is* still precomputed once, since ψ never changes.

**The CFL assertion was too weak.** The notebook asserted `q ≤ 1/6`. That is necessary
but not sufficient once exclusion is on: the Bernoulli factor exceeds one for downhill
moves, so the realised probability sum must be checked every step. See the
`hop-probability-sum` guard and `suggest_tau()`.

**The free energy gained its voxel-volume factor.** The notebook had no exclusion term
at all, so the issue did not arise; it is recorded here because the first version of
`bfex` in this package omitted the factor and every channel work came out `1/V` too
small.

## The file itself

The original notebook has been left in place at
`~/ftsz_tests/driftRDME_standalone.ipynb`. Nothing in this package depends on it, and
everything it validated is reproduced above, so it can be deleted whenever its owner
wants to. That deletion is deliberately not automated.
