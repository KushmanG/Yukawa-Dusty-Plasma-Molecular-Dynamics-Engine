'''
heuristics.py - Empirical calibration of every [HEURISTICS] parameter in testing_parameters.txt
================================================================================================

test.py gates run.py on a ladder of PASS/FAIL checks. Each check contains knobs
(sampling lengths, bin counts, tolerances, thresholds). None of them are hard
physical constraints -- they are OPERATING POINTS, and the standard way to pick
an operating point in molecular dynamics is to MEASURE it, not assert it.

This script runs those measurements. Every method used here is a standard,
textbook procedure in computational physics:

  1. TIMESTEP CONVERGENCE (energy-drift scaling)
       Velocity-Verlet is a 2nd-order integrator: NVE energy error ~ dt^2.
       Sweep dt over a decade+, confirm the slope on a log-log plot, and place
       the working dt / failure threshold on the measured curve.
       (Allen & Tildesley "Computer Simulation of Liquids"; Frenkel & Smit
        "Understanding Molecular Simulation" -- the canonical dt test.)

  2. EQUILIBRATION / SETTLING TIME
       Start the system deliberately far from the target (hot), watch T(t) and
       the potential energy U(t) relax, and measure the step after which both
       stay inside their steady-state noise band. Equilibration cutoffs must
       exceed this measured settling time -- not a guess.

  3. AUTOCORRELATION ANALYSIS (integrated autocorrelation time)
       Frames used for averages are only worth full statistical weight if they
       are decorrelated. Compute the velocity autocorrelation function and the
       potential-energy autocorrelation function, extract the 1/e and
       integrated correlation times, and report the effective number of
       independent frames N_eff = N_frames * min(1, interval / (2 tau_int)).
       (Sokal's lecture notes on Monte Carlo error analysis; standard MD
        practice.)

  4. BLOCK AVERAGING (Flyvbjerg & Petersen, J. Chem. Phys. 91, 461 (1989))
       The standard single-run estimator of the statistical error of a time
       average over correlated data: repeatedly pair-and-average the series;
       the error estimate grows to a plateau, which is the true error bar.

  5. SEED-REPLICA ERROR ESTIMATION (noise floor)
       Repeat an entire measurement with independent RNG seeds; the
       seed-to-seed standard deviation is the statistical noise floor. A
       PASS/FAIL tolerance is defensible when it sits several times above the
       floor (never trips on luck) while still catching real bias.

  6. CONVERGENCE ("KNEE") STUDIES
       For resolution knobs (number of frames, number of bins): sweep the knob,
       measure the observable, and take the value where the answer stops
       changing. Identical in spirit to a mesh-refinement study.

  7. SENSITIVITY / INVARIANCE CHECKS
       For knobs that should NOT matter (Langevin friction -- any friction
       samples the same canonical ensemble; the number of skipped small-r
       bins): sweep them and show the verdict/observable is flat across the
       range, so the chosen value is not doing hidden work.

USAGE
-----
    python3 heuristics.py --sweep s1            # one sweep
    python3 heuristics.py --sweep s6 --case crystal
    python3 heuristics.py --sweep s8 --friction 0.5
    python3 heuristics.py --sweep s6-plot       # combine s6 shards into figure
    python3 heuristics.py --sweep s8-plot       # combine s8 shards into figure
    python3 heuristics.py --all [--quick]       # everything (quick = smoke test)

Sweeps:
    s1  timestep dt         : NVE drift vs dt, slope check      -> dt, drift threshold
    s2  equilibration       : T(t) and U(t) settling time       -> step>1000 cutoff, steps//2
    s3  decorrelation       : VACF + U-ACF correlation times    -> every=20
    s4  g(r) resolution     : frames & bins convergence         -> frames=50, bins=80
    s5  noise floors        : seed spread + block averaging     -> 5% T tol, 0.05 g(r) tol
    s6  peak separation     : crystal/liquid peak distributions -> peak>2.5, ratio>1.5
    s7  small-r bins        : per-bin noise + skip sensitivity  -> skip 5 bins
    s8  friction invariance : ensemble independence of friction -> friction=1.0

Each sweep writes heuristics_out/<name>.json (raw numbers) and .png (figure).
The JSON is the source of truth quoted in testing_parameters.txt.
'''

import matplotlib
matplotlib.use("Agg")           # must be set BEFORE sim.py pulls in pyplot

from b_sim import *               # the exact physics functions test.py uses
import argparse
import json
import os
import time

# =============================================================================
# BASELINE CONFIGURATION -- mirrors test.py exactly. Every sweep varies ONE
# knob off this baseline so results are not cross-contaminated.
# =============================================================================
N        = 256
l        = L(N)
K        = 1.0
DT       = 0.01
FRICTION = 1.0
G_CRYS   = 400.0            # test.py crystal state point
G_LIQ    = 2.0              # test.py liquid  state point
T_HOT    = 2.0              # test.py hot start (T1)
T_COLD   = 1.0 / G_CRYS     # test.py thermostat target (T2)
STEPS    = 6000             # test.py run length
CUT      = 1000             # test.py "step > 1000" thermostat cutoff
EVERY    = 20               # test.py frame sampling interval
BINS     = 80               # test.py g(r) bins
SKIP     = 5                # test.py "tail = g[5:]"
FRAMES   = 50               # test.py ideal-gas frame count

# test.py tolerances under calibration
TOL_T     = 0.05            # thermostat: |T_mean - T2| < 5% of T2
TOL_G     = 0.05            # g(r) norm:  |g_mean - 1| < 0.05
TOL_DRIFT = 1e-2            # NVE: relative energy drift PASS bound
PEAK_ABS  = 2.5             # structure: peak_crystal > 2.5
PEAK_RAT  = 1.5             # structure: peak_crystal > 1.5 * peak_liquid

SEEDS  = [1000 + i for i in range(10)]   # replica seeds (independent streams)
OUTDIR = "heuristics_out"

# Sweep-internal lengths (full mode). --quick shrinks these for a smoke test.
EQUIL_STEPS = 3000          # thermostatted settling before any measurement
PROD_STEPS  = 4000          # production window for frames / ACFs
NVE_TIME    = 20.0          # physical time of each NVE drift run (t.u.)
DT_LIST     = [0.002, 0.005, 0.01, 0.02, 0.04, 0.08]
FRAMES_LIST = [10, 25, 50, 100, 200]
BINS_LIST   = [40, 60, 80, 120, 160, 240]
FRIC_LIST   = [0.1, 0.5, 1.0, 2.0, 5.0]
MAX_LAG     = 600           # ACF lags (steps)
S2_SEEDS    = 8             # replicas for the settling-time sweep
S8_SEEDS    = 3             # replicas per friction value

# Figure palette -- entity->color is FIXED across every figure in this file
# (validated for CVD separation & contrast: crystal blue / liquid red).
C_CRYS  = "#2a62ad"
C_LIQ   = "#c0392b"
C_GUIDE = "#8a8a8a"


def apply_quick():
    """Shrink everything for a fast end-to-end smoke test (NOT for real numbers)."""
    global STEPS, CUT, EQUIL_STEPS, PROD_STEPS, NVE_TIME, DT_LIST
    global FRAMES_LIST, BINS_LIST, FRIC_LIST, MAX_LAG, SEEDS, S2_SEEDS, S8_SEEDS, FRAMES
    STEPS, CUT = 1200, 300
    EQUIL_STEPS, PROD_STEPS = 500, 600
    NVE_TIME = 4.0
    DT_LIST = [0.005, 0.02, 0.06]
    FRAMES_LIST = [5, 10, 30]
    BINS_LIST = [40, 80, 160]
    FRIC_LIST = [0.5, 1.0]
    MAX_LAG = 150
    SEEDS = SEEDS[:2]
    S2_SEEDS, S8_SEEDS = 2, 2
    FRAMES = 15


# =============================================================================
# SMALL SHARED UTILITIES
# =============================================================================

def save_json(name, payload):
    """Write a sweep's raw numbers to heuristics_out/<name>.json."""
    os.makedirs(OUTDIR, exist_ok=True)
    path = os.path.join(OUTDIR, name + ".json")

    def convert(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(f"not JSON-serializable: {type(o)}")

    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=convert)
    print(f"[saved] {path}")


def load_json(name):
    with open(os.path.join(OUTDIR, name + ".json")) as f:
        return json.load(f)


def style(ax):
    """Recessive grid/axes so the data, not the furniture, is what you see."""
    ax.grid(alpha=0.25, linewidth=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def savefig(fig, name):
    os.makedirs(OUTDIR, exist_ok=True)
    path = os.path.join(OUTDIR, name + ".png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {path}")


def fresh_state(T, rng):
    """Lattice + Maxwell-Boltzmann start, exactly like every test.py check."""
    x = initial_positions(N, l, rng)
    v = initial_velocities(N, T, rng)
    a = net_force(x, l, K)
    return x, v, a


def moving_average(y, w):
    """Plain boxcar smoother; output index i covers raw samples [i, i+w)."""
    w = max(1, int(w))
    return np.convolve(y, np.ones(w) / w, mode="valid")


def settling_step(series, target, band, window):
    """LAST step at which the smoothed series sits outside target +- band.

    'Settled' means: from here on, the smoothed signal never leaves the band
    again. Returned in raw-sample units, conservatively including the smoothing
    window. 0 means it never left the band at all.
    """
    sm = moving_average(series, window)
    outside = np.where(np.abs(sm - target) > band)[0]
    if len(outside) == 0:
        return 0
    return int(outside[-1]) + window   # conservative: end of the offending window


def integrated_tau(c, cut=0.05):
    """Integrated autocorrelation time tau_int = 1/2 + sum C(k), summed until
    the ACF first drops below `cut` (a standard automatic windowing rule --
    beyond that point the ACF is noise and would corrupt the sum)."""
    M = len(c) - 1
    for k in range(1, len(c)):
        if c[k] < cut:
            M = k
            break
    return float(0.5 + np.sum(c[1:M + 1])), M


def one_over_e_time(c):
    """First lag at which the ACF crosses 1/e."""
    below = np.where(c < 1.0 / np.e)[0]
    return int(below[0]) if len(below) else len(c) - 1


def acf_1d(y, max_lag):
    """Normalized autocorrelation of a scalar time series."""
    y = np.asarray(y, dtype=float) - np.mean(y)
    var = np.mean(y * y)
    n = len(y)
    max_lag = min(max_lag, n - 2)
    c = np.array([np.mean(y[: n - k] * y[k:]) for k in range(max_lag + 1)])
    return c / var


def vacf(vels, max_lag):
    """Normalized velocity autocorrelation function.
    vels: (n_steps, N, 2) array of velocity snapshots (every step)."""
    n = len(vels)
    max_lag = min(max_lag, n - 2)
    c = np.empty(max_lag + 1)
    for k in range(max_lag + 1):
        c[k] = np.mean(np.sum(vels[: n - k] * vels[k:], axis=(1, 2)))
    return c / c[0]


def block_error(y):
    """Flyvbjerg-Petersen blocking analysis of a correlated series.

    Returns [(n_blocks, sigma_of_mean_estimate), ...] per blocking level.
    The estimate rises with blocking and plateaus at the true statistical
    error of the mean; the plateau value is what we report."""
    y = np.asarray(y, dtype=float)
    levels = []
    while len(y) >= 4:
        n = len(y)
        levels.append((n, float(np.sqrt(np.var(y, ddof=1) / (n - 1)))))
        m = (n // 2) * 2
        y = 0.5 * (y[0:m:2] + y[1:m:2])
    return levels


def sample_frames_like_test(Gamma, steps, every, friction, rng):
    """EXACT replica of test.py's sample_frames(): lattice start at T=1/Gamma,
    thermostat at that T, keep frames for step > steps//2, step % every == 0."""
    T = 1.0 / Gamma
    x, v, a = fresh_state(T, rng)
    frames = []
    for step in range(steps):
        x, v, a = thermostat(x, v, a, T, friction, DT, l, K, rng)
        if step > steps // 2 and step % every == 0:
            frames.append(x.copy())
    return frames


def production_run(Gamma, equil_steps, prod_steps, rng,
                   keep_frames_every=0, keep_vels=False, keep_pe=False):
    """Equilibrate, then run a production window collecting what's asked for."""
    T = 1.0 / Gamma
    x, v, a = fresh_state(T, rng)
    for _ in range(equil_steps):
        x, v, a = thermostat(x, v, a, T, FRICTION, DT, l, K, rng)

    frames, vels, pes = [], [], []
    for step in range(prod_steps):
        x, v, a = thermostat(x, v, a, T, FRICTION, DT, l, K, rng)
        if keep_frames_every and step % keep_frames_every == 0:
            frames.append(x.copy())
        if keep_vels:
            vels.append(v.copy())
        if keep_pe:
            pes.append(potential_energy(x, l, K))
    return frames, (np.array(vels) if keep_vels else None), np.array(pes)


# =============================================================================
# S1 -- TIMESTEP: NVE energy drift vs dt  (2nd-order scaling of velocity-Verlet)
#
# Calibrates: dt = 0.01, energy_drift PASS bound 1e-2, MARGINAL band 1e-1.
# Method: from ONE equilibrated state per case, integrate NVE for a FIXED
# physical time at each dt and measure drift = (E_max - E_min)/|E_mean|.
# On a log-log plot the points must fall on a slope-2 line (integrator
# correct); the thresholds are then placed on the measured curve.
# =============================================================================

def s1_dt():
    results = {"method": "timestep convergence via NVE energy-drift scaling",
               "nve_time": NVE_TIME, "dts": DT_LIST, "cases": {}}

    for label, Gamma in (("crystal", G_CRYS), ("liquid", G_LIQ)):
        rng = np.random.default_rng(SEEDS[0])
        T = 1.0 / Gamma
        x0, v0, a0 = fresh_state(T, rng)
        for _ in range(EQUIL_STEPS):
            x0, v0, a0 = thermostat(x0, v0, a0, T, FRICTION, DT, l, K, rng)

        drifts = []
        for dt in DT_LIST:
            x, v, a = x0.copy(), v0.copy(), net_force(x0, l, K)
            steps = max(2, int(round(NVE_TIME / dt)))
            E = np.empty(steps)
            with np.errstate(over="ignore", invalid="ignore"):
                for s in range(steps):
                    x, v, a = dynamize(x, v, a, l, K, dt)
                    E[s] = 0.5 * np.sum(v * v) + potential_energy(x, l, K)
            if np.all(np.isfinite(E)):
                drifts.append(float((E.max() - E.min()) / abs(E.mean())))
            else:
                drifts.append(None)          # integrator blew up: unstable dt
            print(f"  s1 {label}: dt={dt}  drift={drifts[-1]}")

        # slope of log(drift) vs log(dt) over the stable points
        ok = [(d, dr) for d, dr in zip(DT_LIST, drifts)
              if dr is not None and dr < 0.5]
        slope = None
        if len(ok) >= 2:
            xs = np.log10([d for d, _ in ok])
            ys = np.log10([dr for _, dr in ok])
            slope = float(np.polyfit(xs, ys, 1)[0])

        drift_at_base = drifts[DT_LIST.index(DT)] if DT in DT_LIST else None
        results["cases"][label] = {
            "Gamma": Gamma, "drifts": drifts, "slope": slope,
            "drift_at_dt0.01": drift_at_base,
            "headroom_to_PASS_bound":
                (TOL_DRIFT / drift_at_base) if drift_at_base else None,
        }

    # ---- figure ----
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    for label, color in (("crystal", C_CRYS), ("liquid", C_LIQ)):
        case = results["cases"][label]
        pts = [(d, dr) for d, dr in zip(DT_LIST, case["drifts"]) if dr]
        ax.loglog([p[0] for p in pts], [p[1] for p in pts], "o-", color=color,
                  lw=2, ms=8, label=f"{label} (slope {case['slope']:.2f})"
                  if case["slope"] else label)
    # slope-2 guide anchored at the crystal dt=0.01 point
    ref = results["cases"]["crystal"]["drift_at_dt0.01"]
    if ref:
        gx = np.array([DT_LIST[0], DT_LIST[-1]])
        ax.loglog(gx, ref * (gx / DT) ** 2, "--", color=C_GUIDE, lw=1.2,
                  label="slope-2 guide (2nd-order integrator)")
    ax.axhline(TOL_DRIFT, color=C_GUIDE, ls=":", lw=1.2)
    ax.text(DT_LIST[0], TOL_DRIFT * 1.3, "PASS bound 1e-2", fontsize=8,
            color=C_GUIDE)
    ax.axvline(DT, color=C_GUIDE, ls=":", lw=1.2)
    ax.text(DT * 1.05, ax.get_ylim()[0] * 2, "working dt", fontsize=8,
            color=C_GUIDE, rotation=90)
    ax.set_xlabel("timestep dt")
    ax.set_ylabel("relative NVE energy drift")
    ax.set_title(f"S1  Energy drift vs dt over t = {NVE_TIME} (fixed physical time)")
    ax.legend(fontsize=8)
    style(ax)
    savefig(fig, "s1_dt")
    save_json("s1_dt", results)


# =============================================================================
# S2 -- EQUILIBRATION TIME: how long until a hot start settles at the target?
#
# Calibrates: the "step > 1000" thermostat cutoff and the "step > steps//2"
# structure-sampling cutoff. Method: replicate test.py's worst case (start at
# T1 = 2.0 = 800x the target T2 = 1/400), watch smoothed T(t) AND smoothed
# U(t), and record the LAST step each leaves its steady-state band. The
# configurational settling (U) is the slow, binding one.
# =============================================================================

def s2_equilibration():
    PE_EVERY = 10
    results = {"method": "settling time of T(t) and U(t) from a hot start "
                         "(last exit from steady-state band)",
               "protocol": {"T_start": T_HOT, "T_target": T_COLD,
                            "steps": STEPS, "friction": FRICTION,
                            "seeds": S2_SEEDS},
               "per_seed": []}
    traces = []

    for seed in SEEDS[:S2_SEEDS]:
        rng = np.random.default_rng(seed)
        x, v, a = fresh_state(T_HOT, rng)      # HOT start, exactly like test.py
        T_hist = np.empty(STEPS)
        pe_hist = []
        for step in range(STEPS):
            x, v, a = thermostat(x, v, a, T_COLD, FRICTION, DT, l, K, rng)
            T_hist[step] = temp(v)
            if step % PE_EVERY == 0:
                pe_hist.append(potential_energy(x, l, K))
        pe_hist = np.array(pe_hist)

        # T settling: 15% band around target ~ 5x the smoothed noise (see s5)
        w_T = 200
        t_eq_T = settling_step(T_hist, T_COLD, 0.15 * T_COLD, w_T)

        # U settling: band = 4 sigma of the smoothed steady-state fluctuation
        w_U = 20                                    # x PE_EVERY = 200 raw steps
        pe_sm = moving_average(pe_hist, w_U)
        tail = pe_sm[3 * len(pe_sm) // 4:]
        band = max(4.0 * float(np.std(tail)), 1e-9)
        t_eq_U = settling_step(pe_hist, float(np.mean(tail)), band, w_U) * PE_EVERY

        results["per_seed"].append({"seed": seed,
                                    "t_eq_T_steps": int(t_eq_T),
                                    "t_eq_U_steps": int(t_eq_U)})
        traces.append((T_hist, pe_hist))
        print(f"  s2 seed {seed}: t_eq(T) = {t_eq_T} steps, "
              f"t_eq(U) = {t_eq_U} steps")

    worst = max(max(r["t_eq_T_steps"], r["t_eq_U_steps"])
                for r in results["per_seed"])
    results["worst_settling_steps"] = int(worst)
    results["cutoff_in_test_py"] = CUT
    results["safety_factor_of_cutoff"] = float(CUT / worst) if worst else None
    results["structure_cutoff_steps_over_2"] = STEPS // 2

    # ---- figure ----
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    for T_hist, _ in traces:
        sm = moving_average(np.abs(T_hist - T_COLD) / T_COLD, 200)
        ax.semilogy(np.arange(len(sm)), sm, color=C_CRYS, alpha=0.45, lw=1.0)
    ax.axhline(0.15, color=C_GUIDE, ls=":", lw=1.2)
    ax.text(STEPS * 0.55, 0.17, "settling band (15%)", fontsize=8, color=C_GUIDE)
    ax.axvline(CUT, color=C_GUIDE, ls="--", lw=1.2)
    ax.text(CUT * 1.05, ax.get_ylim()[1] * 0.3, "test.py cutoff (1000)",
            fontsize=8, color=C_GUIDE, rotation=90)
    ax.set_xlabel("step")
    ax.set_ylabel("|T - T_target| / T_target  (smoothed)")
    ax.set_title(f"S2  Temperature settling, {S2_SEEDS} seeds (hot start T1={T_HOT})")
    style(ax)

    ax = axes[1]
    for _, pe_hist in traces:
        sm = moving_average(pe_hist, 20)
        ax.plot(np.arange(len(sm)) * PE_EVERY, sm, color=C_CRYS,
                alpha=0.45, lw=1.0)
    ax.axvline(CUT, color=C_GUIDE, ls="--", lw=1.2)
    ax.axvline(worst, color=C_LIQ, ls="--", lw=1.2)
    ax.text(worst * 1.03, np.mean(traces[0][1]), f"worst t_eq = {worst}",
            fontsize=8, color=C_LIQ, rotation=90)
    ax.set_xlabel("step")
    ax.set_ylabel("potential energy U (smoothed)")
    ax.set_title("S2  Configurational settling (the slow one)")
    style(ax)
    savefig(fig, "s2_equilibration")
    save_json("s2_equilibration", results)


# =============================================================================
# S3 -- DECORRELATION: how many steps until frames are independent?
#
# Calibrates: every = 20 (frame sampling interval). Method: velocity ACF and
# potential-energy ACF from an equilibrated production run; report 1/e and
# integrated correlation times; convert to the effective independent frame
# count behind test.py's g(r) averages.
# =============================================================================

def s3_decorrelation():
    results = {"method": "velocity & potential-energy autocorrelation times "
                         "(Sokal windowed tau_int)",
               "every_in_test_py": EVERY, "cases": {}}

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, (label, Gamma, color) in zip(
            axes, (("crystal", G_CRYS, C_CRYS), ("liquid", G_LIQ, C_LIQ))):
        rng = np.random.default_rng(SEEDS[0])
        _, vels, pes = production_run(Gamma, EQUIL_STEPS, PROD_STEPS, rng,
                                      keep_vels=True, keep_pe=True)
        cv = vacf(vels, MAX_LAG)
        cu = acf_1d(pes, MAX_LAG)

        tau_v_int, Mv = integrated_tau(cv)
        tau_u_int, Mu = integrated_tau(cu)
        tau_v_e = one_over_e_time(cv)
        tau_u_e = one_over_e_time(cu)

        # frames test.py actually collects: step>steps//2, every 20 -> ~150
        n_frames = len([s for s in range(STEPS)
                        if s > STEPS // 2 and s % EVERY == 0])
        # standard effective-sample-size correction for correlated samples
        n_eff_u = n_frames * min(1.0, EVERY / (2.0 * tau_u_int))
        n_eff_v = n_frames * min(1.0, EVERY / (2.0 * tau_v_int))

        results["cases"][label] = {
            "Gamma": Gamma,
            "tau_vacf_1e_steps": tau_v_e, "tau_vacf_int_steps": tau_v_int,
            "tau_pe_1e_steps": tau_u_e, "tau_pe_int_steps": tau_u_int,
            "acf_window_used_v": Mv, "acf_window_used_u": Mu,
            "frames_collected_by_test": n_frames,
            "n_eff_frames_by_pe_acf": n_eff_u,
            "n_eff_frames_by_vacf": n_eff_v,
        }
        print(f"  s3 {label}: tau_int(VACF) = {tau_v_int:.1f}, "
              f"tau_int(U) = {tau_u_int:.1f} steps; "
              f"N_eff(U) = {n_eff_u:.0f}/{n_frames}")

        lags = np.arange(len(cv))
        ax.plot(lags, cv, color=color, lw=2, label="velocity ACF")
        ax.plot(np.arange(len(cu)), cu, color=color, lw=2, ls="--",
                label="potential-energy ACF")
        ax.axhline(1 / np.e, color=C_GUIDE, ls=":", lw=1)
        ax.axhline(0.0, color=C_GUIDE, lw=0.8)
        ax.axvline(EVERY, color=C_GUIDE, ls="--", lw=1.2)
        ax.text(EVERY * 1.1, 0.9, f"every = {EVERY}", fontsize=8, color=C_GUIDE)
        ax.set_xlabel("lag (steps)")
        ax.set_title(f"S3  {label} (Gamma = {Gamma:.0f})")
        ax.legend(fontsize=8)
        style(ax)
    axes[0].set_ylabel("autocorrelation")
    savefig(fig, "s3_decorrelation")
    save_json("s3_decorrelation", results)


# =============================================================================
# S4 -- g(r) RESOLUTION: convergence in number of frames and number of bins
#
# Calibrates: frames = 50, bins = 80. Method: knee study. Frames: self-
# convergence of g(r) against the longest average (RMS relative difference).
# Bins: saturation of the crystal peak height (resolution) traded against the
# per-bin Poisson noise of the ideal-gas normalization test (statistics).
# =============================================================================

def s4_gr_convergence():
    results = {"method": "convergence (knee) study in frames and bins",
               "frames_list": FRAMES_LIST, "bins_list": BINS_LIST, "cases": {}}

    all_frames = {}
    for label, Gamma in (("crystal", G_CRYS), ("liquid", G_LIQ)):
        rng = np.random.default_rng(SEEDS[0])
        frames, _, _ = production_run(Gamma, EQUIL_STEPS, PROD_STEPS, rng,
                                      keep_frames_every=EVERY)
        all_frames[label] = frames
        n_max = len(frames)

        # ---- frames sweep at fixed bins = BINS ----
        _, g_ref = radial_distribution_function(frames, N, l, BINS)
        conv, peaks_f = [], []
        for k in [f for f in FRAMES_LIST if f <= n_max]:
            _, gk = radial_distribution_function(frames[:k], N, l, BINS)
            num = np.sqrt(np.mean((gk[SKIP:] - g_ref[SKIP:]) ** 2))
            den = np.sqrt(np.mean(g_ref[SKIP:] ** 2))
            conv.append(float(num / den))
            peaks_f.append(float(gk.max()))

        # ---- bins sweep using ALL frames ----
        peaks_b = []
        for nb in BINS_LIST:
            _, gb = radial_distribution_function(frames, N, l, nb)
            peaks_b.append(float(gb.max()))

        results["cases"][label] = {
            "Gamma": Gamma, "n_frames_available": n_max,
            "frames_used": [f for f in FRAMES_LIST if f <= n_max],
            "rms_rel_diff_vs_full_average": conv,
            "peak_vs_frames": peaks_f,
            "peak_vs_bins": peaks_b,
        }
        print(f"  s4 {label}: RMS conv {conv}; peak vs bins {peaks_b}")

    # per-bin Poisson noise of the IDEAL-GAS normalization test vs bin count:
    # expected pair counts in bin i are FRAMES*N*n*annulus_i, so the relative
    # noise is 1/sqrt(counts). Report the median over the trusted region r > 1.
    n_density = N / (l * l)
    med_noise = []
    for nb in BINS_LIST:
        edges = np.linspace(0.0, l / 2.0, nb + 1)
        annulus = np.pi * (edges[1:] ** 2 - edges[:-1] ** 2)
        counts = FRAMES * N * n_density * annulus
        centers = 0.5 * (edges[1:] + edges[:-1])
        med_noise.append(float(np.median(1.0 / np.sqrt(counts[centers > 1.0]))))
    results["ideal_gas_median_per_bin_rel_noise_vs_bins"] = med_noise

    # ---- figure ----
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.0))
    ax = axes[0]
    for label, color in (("crystal", C_CRYS), ("liquid", C_LIQ)):
        case = results["cases"][label]
        ax.loglog(case["frames_used"], case["rms_rel_diff_vs_full_average"],
                  "o-", color=color, lw=2, ms=7, label=label)
    ax.axvline(FRAMES, color=C_GUIDE, ls="--", lw=1.2)
    ax.text(FRAMES * 1.08, ax.get_ylim()[1] * 0.5, f"frames = {FRAMES}",
            fontsize=8, color=C_GUIDE, rotation=90)
    ax.set_xlabel("number of frames averaged")
    ax.set_ylabel("RMS rel. difference vs full average")
    ax.set_title("S4a  g(r) self-convergence vs frames")
    ax.legend(fontsize=8)
    style(ax)

    ax = axes[1]
    for label, color in (("crystal", C_CRYS), ("liquid", C_LIQ)):
        ax.plot(BINS_LIST, results["cases"][label]["peak_vs_bins"], "o-",
                color=color, lw=2, ms=7, label=label)
    ax.axvline(BINS, color=C_GUIDE, ls="--", lw=1.2)
    ax.text(BINS * 1.03, ax.get_ylim()[0] * 1.05, f"bins = {BINS}",
            fontsize=8, color=C_GUIDE, rotation=90)
    ax.set_xlabel("number of bins")
    ax.set_ylabel("g(r) first-peak height")
    ax.set_title("S4b  Peak resolution vs bins")
    ax.legend(fontsize=8)
    style(ax)

    ax = axes[2]
    ax.plot(BINS_LIST, med_noise, "o-", color=C_CRYS, lw=2, ms=7)
    ax.axvline(BINS, color=C_GUIDE, ls="--", lw=1.2)
    ax.axhline(TOL_G, color=C_GUIDE, ls=":", lw=1.2)
    ax.text(BINS_LIST[0], TOL_G * 1.08, "0.05 tolerance", fontsize=8,
            color=C_GUIDE)
    ax.set_xlabel("number of bins")
    ax.set_ylabel("median per-bin rel. noise (ideal gas)")
    ax.set_title(f"S4c  Statistics cost of resolution ({FRAMES} frames)")
    style(ax)
    savefig(fig, "s4_gr_convergence")
    save_json("s4_gr_convergence", results)


# =============================================================================
# S5 -- NOISE FLOORS for the two tolerances (5% on T_mean, 0.05 on g_mean)
#
# Method: (a) seed replicas -- run the EXACT test.py measurement 10x with
# independent seeds; the seed-to-seed std IS the statistical noise floor.
# (b) Flyvbjerg-Petersen block averaging on one run as the independent,
# single-run estimate of the same error bar. A tolerance is justified when it
# is several times the floor.
# =============================================================================

def s5_noise_floors():
    results = {"method": "seed-replica noise floor + Flyvbjerg-Petersen "
                         "block averaging",
               "seeds": SEEDS}

    # ---- (a) thermostat T_mean, exact test.py protocol ----
    T_means, first_series = [], None
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        x, v, a = fresh_state(T_HOT, rng)          # hot start, like test.py
        samples = []
        for step in range(STEPS):
            x, v, a = thermostat(x, v, a, T_COLD, FRICTION, DT, l, K, rng)
            if step > CUT:
                samples.append(temp(v))
        T_means.append(float(np.mean(samples)))
        if first_series is None:
            first_series = np.array(samples)
        print(f"  s5 T_mean seed {seed}: {T_means[-1]:.6f} "
              f"(target {T_COLD:.6f})")

    T_means = np.array(T_means)
    rel_dev = np.abs(T_means - T_COLD) / T_COLD
    blocks = block_error(first_series)
    # plateau = largest error estimate among levels that still have >= 16 blocks
    plateau = max(se for n, se in blocks if n >= 16)

    results["T_mean"] = {
        "target": T_COLD,
        "per_seed": T_means.tolist(),
        "mean_of_means": float(T_means.mean()),
        "seed_std_rel": float(T_means.std(ddof=1) / T_COLD),
        "max_rel_dev": float(rel_dev.max()),
        "tolerance_rel": TOL_T,
        "tolerance_over_seed_std": float(TOL_T / (T_means.std(ddof=1) / T_COLD)),
        "block_averaging_levels": blocks,
        "block_plateau_rel_error": float(plateau / T_COLD),
        "instantaneous_rel_fluct_theory_1_over_sqrtN": float(1 / np.sqrt(N)),
    }

    # ---- (b) ideal-gas g_mean, exact test.py protocol ----
    g_means = []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        gas = [rng.uniform(0, l, size=(N, 2)) for _ in range(FRAMES)]
        _, g = radial_distribution_function(gas, N, l, BINS)
        g_means.append(float(g[SKIP:].mean()))
        print(f"  s5 g_mean seed {seed}: {g_means[-1]:.5f}")
    g_means = np.array(g_means)

    results["g_mean"] = {
        "per_seed": g_means.tolist(),
        "mean": float(g_means.mean()),
        "seed_std": float(g_means.std(ddof=1)),
        "max_abs_dev_from_1": float(np.abs(g_means - 1.0).max()),
        "tolerance": TOL_G,
        "tolerance_over_seed_std": float(TOL_G / g_means.std(ddof=1)),
    }

    # ---- figure ----
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.0))
    ax = axes[0]
    ax.plot(range(len(T_means)), T_means / T_COLD, "o", color=C_CRYS, ms=8)
    ax.axhline(1.0, color=C_GUIDE, lw=1)
    ax.axhline(1 + TOL_T, color=C_GUIDE, ls="--", lw=1.2)
    ax.axhline(1 - TOL_T, color=C_GUIDE, ls="--", lw=1.2)
    ax.text(0, 1 + TOL_T * 1.1, "5% tolerance band", fontsize=8, color=C_GUIDE)
    ax.set_xlabel("seed replica")
    ax.set_ylabel("T_mean / T_target")
    ax.set_title(f"S5a  Thermostat noise floor ({len(SEEDS)} seeds)")
    style(ax)

    ax = axes[1]
    ns = [n for n, _ in blocks]
    ses = [se / T_COLD for _, se in blocks]
    ax.semilogx(ns, ses, "o-", color=C_CRYS, lw=2, ms=7)
    ax.axhline(plateau / T_COLD, color=C_GUIDE, ls="--", lw=1.2)
    ax.text(ns[-1], plateau / T_COLD * 1.1,
            f"plateau = {plateau / T_COLD:.2%}", fontsize=8, color=C_GUIDE)
    ax.invert_xaxis()                       # blocking proceeds right-to-left
    ax.set_xlabel("number of blocks (halves each level)")
    ax.set_ylabel("rel. error estimate of T_mean")
    ax.set_title("S5b  Flyvbjerg-Petersen blocking (one run)")
    style(ax)

    ax = axes[2]
    ax.plot(range(len(g_means)), g_means, "o", color=C_LIQ, ms=8)
    ax.axhline(1.0, color=C_GUIDE, lw=1)
    ax.axhline(1 + TOL_G, color=C_GUIDE, ls="--", lw=1.2)
    ax.axhline(1 - TOL_G, color=C_GUIDE, ls="--", lw=1.2)
    ax.text(0, 1 + TOL_G * 1.02, "0.05 tolerance band", fontsize=8,
            color=C_GUIDE)
    ax.set_xlabel("seed replica")
    ax.set_ylabel("ideal-gas g_mean")
    ax.set_title(f"S5c  g(r) normalization noise floor ({len(SEEDS)} seeds)")
    style(ax)
    savefig(fig, "s5_noise_floors")
    save_json("s5_noise_floors", results)


# =============================================================================
# S6 -- PEAK SEPARATION: distributions of peak_crystal and peak_liquid
#
# Calibrates: peak_crystal > 2.5 and peak_crystal > 1.5 * peak_liquid.
# Method: run the EXACT test.py structure protocol across seed replicas for
# one case (shardable) and report the peak distribution; the thresholds are
# justified by the measured gap between the two distributions.
# =============================================================================

def s6_peaks(case):
    Gamma = G_CRYS if case == "crystal" else G_LIQ
    peaks = []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        frames = sample_frames_like_test(Gamma, STEPS, EVERY, FRICTION, rng)
        _, g = radial_distribution_function(frames, N, l, BINS)
        peaks.append(float(g.max()))
        print(f"  s6 {case} seed {seed}: peak = {peaks[-1]:.3f}")

    peaks = np.array(peaks)
    save_json(f"s6_{case}", {
        "method": "seed-replica distribution of the g(r) first-peak height "
                  "(exact test.py structure protocol)",
        "case": case, "Gamma": Gamma, "seeds": SEEDS,
        "n_frames_per_run": len([s for s in range(STEPS)
                                 if s > STEPS // 2 and s % EVERY == 0]),
        "peaks": peaks.tolist(),
        "min": float(peaks.min()), "mean": float(peaks.mean()),
        "max": float(peaks.max()), "std": float(peaks.std(ddof=1)),
    })


def s6_plot():
    c = load_json("s6_crystal")
    q = load_json("s6_liquid")
    pc, pl = np.array(c["peaks"]), np.array(q["peaks"])
    ratios = pc / pl                            # paired by seed index

    summary = {
        "crystal": {k: c[k] for k in ("min", "mean", "max", "std")},
        "liquid": {k: q[k] for k in ("min", "mean", "max", "std")},
        "gap_between_distributions": float(pc.min() - pl.max()),
        "abs_threshold": PEAK_ABS,
        "margin_min_crystal_over_abs_threshold": float(pc.min() / PEAK_ABS),
        "margin_max_liquid_under_abs_threshold": float(pl.max() / PEAK_ABS),
        "ratio_threshold": PEAK_RAT,
        "paired_ratios": ratios.tolist(),
        "min_paired_ratio": float(ratios.min()),
    }

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    jitterx = np.linspace(-0.08, 0.08, len(pc))
    ax.plot(0 + jitterx, pc, "o", color=C_CRYS, ms=9, label="crystal")
    ax.plot(1 + jitterx, pl, "o", color=C_LIQ, ms=9, label="liquid")
    ax.axhline(PEAK_ABS, color=C_GUIDE, ls="--", lw=1.4)
    ax.text(0.4, PEAK_ABS * 1.04, f"threshold {PEAK_ABS}", fontsize=9,
            color=C_GUIDE)
    ax.set_xticks([0, 1], ["crystal\n(Gamma=400)", "liquid\n(Gamma=2)"])
    ax.set_ylabel("g(r) first-peak height")
    ax.set_title(f"S6  Peak separation, {len(pc)} seeds per case")
    ax.legend(fontsize=8)
    style(ax)

    ax = axes[1]
    ax.plot(range(len(ratios)), ratios, "o", color=C_CRYS, ms=9)
    ax.axhline(PEAK_RAT, color=C_GUIDE, ls="--", lw=1.4)
    ax.text(0, PEAK_RAT * 1.05, f"ratio threshold {PEAK_RAT}", fontsize=9,
            color=C_GUIDE)
    ax.set_xlabel("seed replica (paired)")
    ax.set_ylabel("peak_crystal / peak_liquid")
    ax.set_title("S6  Paired peak ratio per seed")
    style(ax)
    savefig(fig, "s6_peaks")
    save_json("s6_summary", summary)


# =============================================================================
# S7 -- SMALL-r BINS: which bins are statistically unreliable, and does the
#        verdict care exactly how many we skip?
#
# Calibrates: tail = g[5:]. Method: (a) per-bin seed-to-seed scatter of the
# ideal-gas g(r) compared against the Poisson prediction 1/sqrt(expected
# counts) -- shows the innermost bins are the noisy ones; (b) sensitivity:
# recompute the normalization verdict for skip = 0..12 and show the PASS is
# flat across the whole range, i.e. 5 is a convention inside a plateau, not a
# tuned value.
# =============================================================================

def s7_bin_skip():
    gs = []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        gas = [rng.uniform(0, l, size=(N, 2)) for _ in range(FRAMES)]
        _, g = radial_distribution_function(gas, N, l, BINS)
        gs.append(g)
    gs = np.array(gs)                       # (n_seeds, BINS)

    per_bin_std = gs.std(axis=0, ddof=1)
    per_bin_mean = gs.mean(axis=0)

    # Poisson prediction: expected pair counts per bin -> rel. error
    n_density = N / (l * l)
    edges = np.linspace(0.0, l / 2.0, BINS + 1)
    annulus = np.pi * (edges[1:] ** 2 - edges[:-1] ** 2)
    exp_counts = FRAMES * N * n_density * annulus
    poisson_rel = 1.0 / np.sqrt(exp_counts)

    # sensitivity of the verdict to the skip count
    max_skip = 13
    worst_dev, verdicts = [], []
    for skip in range(max_skip):
        devs = np.abs(gs[:, skip:].mean(axis=1) - 1.0)
        worst_dev.append(float(devs.max()))
        verdicts.append(bool(devs.max() < TOL_G))

    results = {
        "method": "per-bin noise vs Poisson prediction + verdict sensitivity "
                  "to the skip count",
        "seeds": SEEDS, "frames": FRAMES, "bins": BINS,
        "first_12_bins": [{
            "bin": i,
            "r_center": float(0.5 * (edges[i] + edges[i + 1])),
            "expected_pair_counts": float(exp_counts[i]),
            "poisson_rel_error": float(poisson_rel[i]),
            "measured_seed_std": float(per_bin_std[i]),
            "measured_mean": float(per_bin_mean[i]),
        } for i in range(12)],
        "verdict_vs_skip": [{"skip": s, "worst_seed_dev": d, "pass": v}
                            for s, (d, v) in
                            enumerate(zip(worst_dev, verdicts))],
        "skip_in_test_py": SKIP,
    }

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    bins_idx = np.arange(BINS)
    ax.plot(bins_idx[:30], per_bin_std[:30], "o-", color=C_CRYS, lw=1.6,
            ms=5, label="measured seed-to-seed std")
    ax.plot(bins_idx[:30], poisson_rel[:30], "--", color=C_GUIDE, lw=1.4,
            label="Poisson prediction 1/sqrt(counts)")
    ax.axvline(SKIP - 0.5, color=C_GUIDE, ls=":", lw=1.2)
    ax.text(SKIP, ax.get_ylim()[1] * 0.8, f"skip = {SKIP}", fontsize=8,
            color=C_GUIDE)
    ax.set_xlabel("bin index")
    ax.set_ylabel("relative noise of g in bin")
    ax.set_title("S7a  Innermost bins are the noisy ones")
    ax.legend(fontsize=8)
    style(ax)

    ax = axes[1]
    ax.plot(range(max_skip), worst_dev, "o-", color=C_LIQ, lw=2, ms=7)
    ax.axhline(TOL_G, color=C_GUIDE, ls="--", lw=1.4)
    ax.text(0, TOL_G * 1.05, f"tolerance {TOL_G}", fontsize=9, color=C_GUIDE)
    ax.axvline(SKIP, color=C_GUIDE, ls=":", lw=1.2)
    ax.set_xlabel("number of skipped small-r bins")
    ax.set_ylabel("worst |g_mean - 1| across seeds")
    ax.set_title("S7b  Verdict is flat across skip = 0..12")
    style(ax)
    savefig(fig, "s7_bin_skip")
    save_json("s7_bin_skip", results)


# =============================================================================
# S8 -- FRICTION INVARIANCE: equilibrium observables must not depend on gamma
#
# Calibrates: friction = 1.0. Method: any (correct) Langevin friction samples
# the same canonical ensemble -- friction only sets HOW FAST you get there and
# how damped the dynamics look, never WHERE you end up. Sweep gamma over
# 0.1..5 and show T_mean and the g(r) peak are flat within the noise floor.
# =============================================================================

def s8_friction(gamma):
    results = {"method": "ensemble-invariance check: equilibrium observables "
                         "vs Langevin friction",
               "friction": gamma, "seeds": SEEDS[:S8_SEEDS], "cases": {}}

    for label, Gamma in (("crystal", G_CRYS), ("liquid", G_LIQ)):
        T = 1.0 / Gamma
        T_means, peaks = [], []
        for seed in SEEDS[:S8_SEEDS]:
            rng = np.random.default_rng(seed)
            x, v, a = fresh_state(T, rng)
            samples, frames = [], []
            for step in range(STEPS):
                x, v, a = thermostat(x, v, a, T, gamma, DT, l, K, rng)
                if step > CUT:
                    samples.append(temp(v))
                if step > STEPS // 2 and step % EVERY == 0:
                    frames.append(x.copy())
            _, g = radial_distribution_function(frames, N, l, BINS)
            T_means.append(float(np.mean(samples)))
            peaks.append(float(g.max()))
            print(f"  s8 gamma={gamma} {label} seed {seed}: "
                  f"T_mean/T = {T_means[-1] / T:.4f}, peak = {peaks[-1]:.3f}")
        results["cases"][label] = {"Gamma": Gamma, "T_target": T,
                                   "T_means": T_means, "peaks": peaks}

    save_json(f"s8_f{gamma}", results)


def s8_plot():
    gammas, data = [], []
    for gamma in FRIC_LIST:
        try:
            data.append(load_json(f"s8_f{gamma}"))
            gammas.append(gamma)
        except FileNotFoundError:
            print(f"  s8-plot: missing shard for gamma={gamma}, skipping")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    summary = {"gammas": gammas, "cases": {}}
    for label, color in (("crystal", C_CRYS), ("liquid", C_LIQ)):
        Tm = np.array([[t / d["cases"][label]["T_target"]
                        for t in d["cases"][label]["T_means"]] for d in data])
        Pk = np.array([d["cases"][label]["peaks"] for d in data])
        axes[0].errorbar(gammas, Tm.mean(axis=1), yerr=Tm.std(axis=1, ddof=1),
                         fmt="o-", color=color, lw=2, ms=7, capsize=4,
                         label=label)
        axes[1].errorbar(gammas, Pk.mean(axis=1), yerr=Pk.std(axis=1, ddof=1),
                         fmt="o-", color=color, lw=2, ms=7, capsize=4,
                         label=label)
        summary["cases"][label] = {
            "T_mean_over_target_by_gamma": Tm.mean(axis=1).tolist(),
            "T_spread_by_gamma": Tm.std(axis=1, ddof=1).tolist(),
            "peak_by_gamma": Pk.mean(axis=1).tolist(),
            "peak_spread_by_gamma": Pk.std(axis=1, ddof=1).tolist(),
            "max_rel_variation_of_peak_across_gamma":
                float((Pk.mean(axis=1).max() - Pk.mean(axis=1).min())
                      / Pk.mean(axis=1).mean()),
        }

    axes[0].axhline(1.0, color=C_GUIDE, lw=1)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Langevin friction gamma")
    axes[0].set_ylabel("T_mean / T_target")
    axes[0].set_title("S8  Thermostat accuracy vs friction")
    axes[0].legend(fontsize=8)
    style(axes[0])

    axes[1].set_xscale("log")
    axes[1].axvline(FRICTION, color=C_GUIDE, ls="--", lw=1.2)
    axes[1].text(FRICTION * 1.05, axes[1].get_ylim()[0] * 1.05,
                 "working value", fontsize=8, color=C_GUIDE, rotation=90)
    axes[1].set_xlabel("Langevin friction gamma")
    axes[1].set_ylabel("g(r) first-peak height")
    axes[1].set_title("S8  Structure vs friction (must be flat)")
    axes[1].legend(fontsize=8)
    style(axes[1])
    savefig(fig, "s8_friction")
    save_json("s8_summary", summary)


# =============================================================================
# CLI
# =============================================================================

SWEEPS = {
    "s1": lambda a: s1_dt(),
    "s2": lambda a: s2_equilibration(),
    "s3": lambda a: s3_decorrelation(),
    "s4": lambda a: s4_gr_convergence(),
    "s5": lambda a: s5_noise_floors(),
    "s6": lambda a: s6_peaks(a.case),
    "s6-plot": lambda a: s6_plot(),
    "s7": lambda a: s7_bin_skip(),
    "s8": lambda a: s8_friction(a.friction),
    "s8-plot": lambda a: s8_plot(),
}


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--sweep", choices=sorted(SWEEPS), help="which sweep to run")
    p.add_argument("--case", choices=["crystal", "liquid"], default="crystal",
                   help="s6 shard: which state point")
    p.add_argument("--friction", type=float, default=FRICTION,
                   help="s8 shard: which friction value")
    p.add_argument("--all", action="store_true", help="run every sweep serially")
    p.add_argument("--quick", action="store_true",
                   help="smoke test: tiny runs, NOT for real numbers")
    args = p.parse_args()

    if args.quick:
        apply_quick()
        print("[quick mode] shrunk parameters -- results are a smoke test only")

    t0 = time.time()
    if args.all:
        for name in ("s1", "s2", "s3", "s4", "s5", "s7"):
            print(f"=== {name} ===")
            SWEEPS[name](args)
        for case in ("crystal", "liquid"):
            print(f"=== s6 {case} ===")
            s6_peaks(case)
        s6_plot()
        for gamma in FRIC_LIST:
            print(f"=== s8 gamma={gamma} ===")
            s8_friction(gamma)
        s8_plot()
    elif args.sweep:
        SWEEPS[args.sweep](args)
    else:
        p.error("pick --sweep <name> or --all")
    print(f"[done] {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
