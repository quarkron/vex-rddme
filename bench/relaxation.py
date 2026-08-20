"""Measure steps-to-stationary, which fixes the notebook lattice size.

Three things the exploration probes got wrong, all fixed here:

1. **A gradient must actually form.** Comparing a profile against its own final state
   proves nothing if the profile never developed structure; the earlier probe reported
   "converged at step 0" for exactly that reason. Asserted as a precondition.
2. **Mass must be exactly conserved**, checked against a recorded initial total, so a
   "relaxed" profile cannot be an artifact of leaking particles.
3. **The convergence metric needs a noise floor below its threshold.** The summed
   per-bin profile deviation has a Poisson floor of order ``sqrt(N/bins) * bins / N``,
   about 12% for 4096 particles in 64 bins. A 2% threshold on it is therefore
   unreachable in principle. The profile's *centre of mass* is one scalar with noise
   ``sigma_x / sqrt(N)``, roughly 0.2 voxels here, against a dynamic range of ~20
   voxels. That is the observable with the headroom to measure a relaxation time.

Particles start in the high-field half, so the field drives them the full width of
the box: a genuinely non-equilibrium start with a large dynamic range.
"""

import argparse
import time

import numpy as np

from vex_rddme import Simulation, Species
from vex_rddme.guards import GuardViolation
from vex_rddme.observe import project

CONV_FRACTION = 0.05      # within 5% of the total relaxation amplitude
BLOCK = 5                 # consecutive samples averaged before testing


def centre_of_mass(profile):
    p = np.asarray(profile, dtype=np.float64)
    x = np.arange(p.size, dtype=np.float64)
    return float((x * p).sum() / p.sum())


def run_case(shape, voxel_nm, sigma_nm, gamma, cap, n_particles, tau_s, n_steps,
             sample_every, seed=0):
    dim = len(shape)
    n_axis = shape[-1]
    ramp = np.arange(n_axis, dtype=float) / n_axis
    psi = np.broadcast_to(ramp, shape).copy()[None, ...]

    sim = Simulation(
        shape=shape, voxel_nm=voxel_nm,
        species=[Species("A", sigma_nm, np.array([gamma]))],
        occupancy_cap=cap, psi=psi, D_um2_s=1.0, tau_s=tau_s,
        seed=seed, attach_log_handler=False,
    )

    # All particles in the HIGH-psi half: the field then drives them across the
    # whole box, giving the centre of mass its full dynamic range.
    sl = [slice(None)] * dim
    sl[-1] = slice(n_axis // 2, n_axis)
    room = np.zeros(shape, dtype=np.int64)
    room[tuple(sl)] = 1
    n_room = int(room.sum())
    per_voxel = n_particles // n_room
    if per_voxel < 1 or per_voxel > cap:
        raise SystemExit(
            f"{n_particles} particles over {n_room} voxels gives {per_voxel}/voxel, "
            f"outside [1, {cap}]. Use n_particles between {n_room} and {n_room * cap}."
        )
    sim.set_counts("A", room * per_voxel)
    sim.record_initial()
    initial_totals = sim.state.totals().copy()
    com_initial = centre_of_mass(project(sim.state.lattice_view("A"), sim.lattice))

    coms, steps = [], []
    t0 = time.perf_counter()
    for i in range(n_steps):
        sim.step()
        if i % sample_every == 0:
            coms.append(centre_of_mass(project(sim.state.lattice_view("A"), sim.lattice)))
            steps.append(i)
    wall = time.perf_counter() - t0

    sim.state.check_mass()
    sim.state.check_occupancy_cap()
    assert np.array_equal(sim.state.totals(), initial_totals), "mass changed"

    coms = np.array(coms)
    com_ref = float(coms[-max(5, len(coms) // 10):].mean())
    amplitude = abs(com_initial - com_ref)

    # PRECONDITION: the field must actually have moved the distribution, or a
    # "relaxation time" is meaningless.
    if amplitude < 2.0:
        raise SystemExit(
            f"no relaxation to measure: the centre of mass moved only "
            f"{amplitude:.2f} voxels ({com_initial:.2f} -> {com_ref:.2f}). Increase "
            f"gamma or start further from equilibrium."
        )

    # Block-average before testing, so sample noise does not trigger convergence.
    blocked, blocked_steps = [], []
    for k in range(0, len(coms) - BLOCK + 1):
        blocked.append(coms[k:k + BLOCK].mean())
        blocked_steps.append(steps[k + BLOCK - 1])
    blocked = np.array(blocked)

    dev = np.abs(blocked - com_ref) / amplitude
    conv = next((s for s, d in zip(blocked_steps, dev) if d < CONV_FRACTION), None)
    noise = float(coms[-max(10, len(coms) // 5):].std())

    return {
        "shape": shape, "n_particles": int(initial_totals[0]), "tau_s": tau_s,
        "ms_per_step": wall / n_steps * 1e3, "wall_s": wall, "n_steps": n_steps,
        "com_initial": com_initial, "com_ref": com_ref, "amplitude": amplitude,
        "noise_voxels": noise, "conv_step": conv,
        "conv_wall_s": None if conv is None else conv * wall / n_steps,
        "final_dev": float(dev[-1]),
        "max_xi3": float(cap * (np.pi / 6) * sigma_nm ** 3 / voxel_nm ** 3),
    }


def run_with_tau_backoff(max_halvings=6, **case):
    """Retry with a halved timestep when the probability-sum guard fires.

    Strong exclusion plus a strong field caps the admissible timestep: particles pile
    up against the low-field wall, the local removal work grows, and the downhill
    Bernoulli factor eventually exhausts the probability budget. The largest workable
    tau is a property of the configuration rather than something to guess, so the
    bench discovers it and reports it.
    """
    tau = case.pop("tau_s")
    n_steps = case.pop("n_steps")
    for k in range(max_halvings + 1):
        try:
            result = run_case(tau_s=tau, n_steps=n_steps, **case)
            result["tau_halvings"] = k
            return result
        except GuardViolation as exc:
            if "hop-probability-sum" not in str(exc):
                raise
            tau /= 2.0
            n_steps *= 2
            print(f"    (probability budget exceeded; retrying at tau = {tau:.2e} s)")
    raise SystemExit(
        f"still exceeding the probability budget after {max_halvings} halvings "
        f"(tau down to {tau:g} s); the field gradient or the crowding is too strong"
    )


CASES = [
    dict(name="64^2", shape=(64, 64), voxel_nm=20.0, sigma_nm=8.0, gamma=1.5,
         cap=20, n_particles=4096, tau_s=2e-5, n_steps=30000, sample_every=50),
    dict(name="32^3", shape=(32, 32, 32), voxel_nm=20.0, sigma_nm=8.0, gamma=1.5,
         cap=20, n_particles=32768, tau_s=1e-5, n_steps=8000, sample_every=25),
    dict(name="128^2", shape=(128, 128), voxel_nm=20.0, sigma_nm=8.0, gamma=1.5,
         cap=20, n_particles=16384, tau_s=2e-5, n_steps=30000, sample_every=50),
]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    args = ap.parse_args()

    hdr = (f"{'case':>6} {'N':>6} {'tau':>9} {'ms/step':>8} {'xi3max':>7} "
           f"{'com move':>9} {'noise':>7} {'conv step':>10} {'conv wall':>10} "
           f"{'+10x avg':>9}")
    print(hdr)
    print("-" * len(hdr))
    for case in CASES:
        if args.only and case["name"] != args.only:
            continue
        c = dict(case)
        name = c.pop("name")
        r = run_with_tau_backoff(**c)
        cs = "not reached" if r["conv_step"] is None else str(r["conv_step"])
        cw = "n/a" if r["conv_wall_s"] is None else f"{r['conv_wall_s']:.1f}s"
        x10 = ("n/a" if r["conv_step"] is None
               else f"{10 * r['conv_step'] * r['ms_per_step'] / 1e3 / 60:.1f} min")
        print(f"{name:>6} {r['n_particles']:>6} {r['tau_s']:>9.2e} "
              f"{r['ms_per_step']:>8.3f} {r['max_xi3']:>7.3f} "
              f"{r['amplitude']:>8.1f}v {r['noise_voxels']:>6.2f}v {cs:>10} "
              f"{cw:>10} {x10:>9}")
