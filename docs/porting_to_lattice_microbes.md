# Porting a mechanism to the Lattice Microbes CUDA solver

This package exists partly so that a mechanism can be tried cheaply before it is
written into `Lattice_Microbes_2.6/src/rdme/`. That only pays off if there is an
obvious landing site for whatever you validated here. This document is that map.

**What transfers is the physics, not the code.** The two implementations deliberately
differ in data layout and in numerics, and no numerical agreement between them is
claimed or tested. What is shared is the free-energy functional, the acceptance rules,
and the decomposition of dynamics into channels. So a mechanism expressed here has a
one-to-one statement there.

## The free energy

| here | Lattice Microbes 2.6 |
|---|---|
| `vex.bfex(xi, voxel_volume_nm3)` | `bFex_wb(x0,x1,x2,x3)` in `src/rdme/dev/byte_diffusion_drift_dev.cu` |
| `vex.dxi_increments(species, V)` | `dxiG[t*4 + n]` table, declared in `src/cuda/constant.cuh` |
| `ExclusionModel.xi(counts)` | `xiAt(inLattice, v, xi, includeSmear)` |
| `ExclusionModel.channel_work(counts, dnu)` | `whiteBearHopDelta(type, xi_src, xi_dst)` for hops |

The functional is identical, including the `log(1 - xi3)` term that makes it BMCSL
rather than Rosenfeld-1989. Both multiply by the voxel volume at the end. That factor
is what turns a free-energy *density* into a per-voxel energy in kT, and dropping it
makes every work `1/V` too small while leaving profiles looking plausible.

**Difference to be aware of.** The kernel evaluates `xiAt` and `bFex_wb` per candidate
hop; it is described in its own comments as "THE hot primitive". Here the free energy is
evaluated once per species per step as whole-lattice arrays, and every direction is
obtained by rolling arrays already in hand. On top of that, `ExclusionModel` tabulates
the functional exactly over integer count vectors, so a channel's work is a pair of
gathers. Neither optimisation transfers directly: the table depends on lattice
quantisation of `xi`, which the kernel breaks by supporting real-valued smeared-body
contributions (`structSmearXiG`).

## Transport

| here | Lattice Microbes 2.6 |
|---|---|
| `hop.bernoulli(u)` | `bernoulli(float u)` in `byte_diffusion_drift_dev.cu` |
| `Hop._precompute_field_work()` → `dphi` | `computeSpeciesPotential(type, latticeIndex)` |
| `Species.gamma`, `Hop.psi` | `gammaG[s*N_MAX_BASES + k]`, `psi_fieldG[k*N_xyz + v]` |
| `Hop._exclusion_arrays()` → `insert`, `remove` | the two bracketed terms of `whiteBearHopDelta` |
| `Hop.probabilities()` | the drift hop probability in `mpd_x_drift_kernel` and friends |

Both use the same sign convention: the Bernoulli argument is `phi_dst - phi_src`, so
`rho ∝ exp(-phi)` at equilibrium and species with `gamma > 0` are repelled from high
`psi`.

**Not implemented here:** the charge-hole / self-pair correction
(`psiSelfLookupG`, `chargeHoleFlagsG`). This package has no self-consistent fields, so
a particle never sources the field it feels and the correction has nothing to correct.
If you prototype anything where particles source their own field, that term must come
with it. Without it, every particle is repelled by itself.

## Reactions

| here | Lattice Microbes 2.6 |
|---|---|
| `ReactionSet.acceptance(dphi)` | the Fröhner–Noé `pi_r` factor in `byte_reaction_drift_dev.cu` |
| `ReactionSet.mean_acceptance(r, dir)` | `piSumG[r] / piCountG[r]`, exposed as `getMeanAcceptance()` |
| `Reaction.dnu`, `Reaction.didx_forward` | per-reaction stoichiometry; the kernel has no index-offset analogue |
| `ReactionSet._apply_feasibility` | `correct_overflows` plus the per-reaction feasibility checks |

**The one thing to carry over carefully.** For a reversible pair, the reverse work must
be evaluated from the **post-reaction** count vector, not the pre-reaction one:

```
    forward, from n:        dbF_f = F(n + dnu) - F(n)
    reverse, from n + dnu:  dbF_r = F(n)       - F(n + dnu)  =  -dbF_f
```

That makes `pi_f / pi_r = exp(-dPhi)` identically, so detailed balance holds by
construction rather than approximately. Evaluating the reverse from `n` gives
`F(n - dnu) - F(n)`, which is not the negative of the forward work, and detailed
balance breaks *silently*. It is measurable, but only if you look. Here that is pinned by
`ReactionSet.detailed_balance_residual()` and by a test asserting the wrong
formulation fails it. Any port should carry an equivalent check.

## State layout: where the two genuinely diverge

| here | Lattice Microbes 2.6 |
|---|---|
| `State.counts[species, voxel]`, integer | packed byte lattice, `MPD_WORDS_PER_SITE` words/site |
| `State.occupancy_cap` | `MAX_OCCUPANCY`, with `correct_overflows` |
| none | `structComplexIdG` (L1), `structBondDirG` (L2), `structRoleFillG` (L3) |
| none | `structRegistryG[cid-1]`, the per-complex registry |

This package has no structure overlay at all, so nothing involving bonded assemblies
can be prototyped here. That is the sharpest scope limit: filaments, tips, severing,
annealing, anchors and links have no representation.

## What cannot be prototyped here

- Anything needing bonded structures (the whole `byte_structure_drift_dev.cu` surface).
- Self-consistent fields: the reduce/diff/scatter/swap pipeline, configurable stencils,
  the Yukawa kernel, and the charge-hole correction.
- Smeared multi-voxel bodies. They contribute real-valued weighted densities, which
  breaks the integer quantisation the free-energy table relies on.
- Anything whose answer depends on three dimensions in an essential way. `dim=3` runs,
  but see the refutes-cheaply / suggests-only caveat in the README.
- Steep-potential regimes needing exact hop integration. Here the
  `hop-probability-sum` guard refuses them rather than integrating them.

## Suggested workflow

1. Express the mechanism as a **channel**: a stoichiometry-like change with a work.
   Both implementations decompose dynamics that way, so this is the portable form.
2. Validate it here against something analytic, or at least against a control with the
   mechanism switched off.
3. Check whether the result is a refutation or a positive. Refutations travel to 3D and
   to the kernel; positives do not.
4. Write the kernel version against the row of the tables above, and reproduce the
   corresponding guard, above all the detailed-balance residual if the channel is
   reversible.
