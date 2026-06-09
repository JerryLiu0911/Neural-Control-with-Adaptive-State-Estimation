"""Week 4 — hand-position control for the BMI_and_Hand plant.

Drives the hand to a target (x, y) using only the official interface
(``next_state(u)``, neural obs from ``_brain.measure()``, and ``hand_pos`` /
``current_pos``). The neural-net weights are NOT used by the controller — it is
black-box; the only model knowledge is the arm *geometry* (link lengths), which
the evaluation harness also assumes.

Structure (see design notes):
  * the muscle head selects a muscle by the FREQUENCY of x1 and powers it by a
    slow x2 rhythm -> each "primitive" is a two-tone brain input;
  * shoulder is driven by muscles 0/1, elbow by muscles 2/3 -> control decouples
    into two independent joint servos working in IK joint-space;
  * select frequencies + which joint each drives are brain-INVARIANT
    (calibrate once, offline); effective gains vary per brain (self-cal online).
"""
from __future__ import annotations

import numpy as np

ARM_LINK = 30.0          # cm, both links (matches the eval harness / arm geometry)
F_POWER = 0.028          # slow power-rhythm tone (brain-invariant, from the head)


# --------------------------------------------------------------------------- #
#  arm geometry (known; not the neural-net weights)
# --------------------------------------------------------------------------- #
def fk(sh: float, el: float) -> np.ndarray:
    x = ARM_LINK * np.cos(sh) + ARM_LINK * np.cos(sh + el)
    y = ARM_LINK * np.sin(sh) + ARM_LINK * np.sin(sh + el)
    return np.array([x, y])


def ik(x, y, prev_sh=0.0, prev_el=0.0):
    """Inverse kinematics with elbow-config choice nearest the previous pose
    (identical convention to the evaluation harness)."""
    r2 = x ** 2 + y ** 2
    cos_el = np.clip((r2 - 2 * ARM_LINK ** 2) / (2 * ARM_LINK ** 2), -1.0, 1.0)
    el_pos = float(np.arccos(cos_el))
    el_neg = -el_pos

    def _sh(el):
        s = np.arctan2(y, x) - np.arctan2(ARM_LINK * np.sin(el), ARM_LINK + ARM_LINK * np.cos(el))
        return float(np.arctan2(np.sin(s), np.cos(s)))

    def _wrap(a, b):
        return float(np.arctan2(np.sin(a - b), np.cos(a - b)))

    sh_p, sh_n = _sh(el_pos), _sh(el_neg)
    err_p = _wrap(sh_p, prev_sh) ** 2 + _wrap(el_pos, prev_el) ** 2
    err_n = _wrap(sh_n, prev_sh) ** 2 + _wrap(el_neg, prev_el) ** 2
    return (sh_p, el_pos) if err_p <= err_n else (sh_n, el_neg)


def _wrap(a):
    return float(np.arctan2(np.sin(a), np.cos(a)))


# --------------------------------------------------------------------------- #
#  low-level: synthesise a two-tone (or multi-tone) brain input sample
# --------------------------------------------------------------------------- #
def tone_sample(t, tones, *, bias=0.5, channels=(1.0, 1.0)) -> np.ndarray:
    """u(t) = bias + sum_i a_i sin(2 pi f_i t), broadcast on both channels and
    clipped to [0, 1]. ``tones`` is a list of (freq, amplitude)."""
    s = sum(a * np.sin(2 * np.pi * f * t) for f, a in tones)
    return np.clip(bias + s * np.asarray(channels, float), 0.0, 1.0)


def _arm_joints(plant):
    """Read true joint angles (shoulder, elbow). OFFLINE STRUCTURE DISCOVERY
    ONLY — the controller never calls this; online it uses hand_pos + IK. Reading
    the angles directly avoids IK branch-flips over the large open-loop swings of
    calibration bursts."""
    a = plant._arm
    if hasattr(a, "_shoulder_angle"):
        return float(a._shoulder_angle), float(a._elbow_angle)   # real BMI_and_Hand
    return float(a.sh), float(a.el)                               # mock plant


def _drive(plant, tones, T, *, channels=(1.0, 1.0)):
    """Apply a multi-tone burst for T steps; return the true joint-angle
    trajectory (calibration use)."""
    traj = np.zeros((T, 2))
    for t in range(T):
        plant.next_state(tone_sample(t, tones, channels=channels).tolist())
        traj[t] = _arm_joints(plant)
    return traj


# --------------------------------------------------------------------------- #
#  offline calibration (brain-invariant): discover select freqs + joint map
# --------------------------------------------------------------------------- #
def calibrate_frequencies(plant_factory, *, f_lo=0.045, f_hi=0.34, n_scan=20,
                          a_sel=0.4, a_pow=0.20, burst=90, win=(45, 75), verbose=False):
    """Scan select-frequencies on fresh plants; for each, drive a two-tone burst
    and measure the steady joint angular velocity (slope over ``win``, after the
    power ramp). Returns 4 primitives ``[{f, joint, sign}, ...]`` — the
    brain-invariant structure — plus the raw scan rows.
    
    parameters:
        * f_lo, f_hi, n_scan: scan this many frequencies uniformly in [f_lo, f_hi]
        * a_sel: tone amplitude for the select-tone (the one that should move a joint)
        * a_pow: tone amplitude for the power-tone (same for all frequencies; just needs to be strong enough to get above stiction)
        * burst: how many steps to drive each tone for (should be long enough to get a clean velocity estimate)
        * win: time window (steps) to measure the velocity over (should be after the initial ramp-up transient, but
                before any non-linearities kick in from hitting joint limits)
    returns:
        * primitives: list of 4 dicts with keys 'f', 'joint' (0=shoulder, 1=elbow), 'sign' (1=positive-tone moves
                        joint positive), and 'scan_vel' (the measured velocity during the scan, for debugging)
        * rows: the raw scan data (freq, shoulder_vel, elbow_vel) for all frequencies, for debugging/plotting
        
    """
    freqs = np.linspace(f_lo, f_hi, n_scan)
    rows = []
    for f in freqs:
        traj = _drive(plant_factory(), [(f, a_sel), (F_POWER, a_pow)], burst)
        seg = traj[win[0]:win[1]] # segments observation windowto reach steady velocity, avoiding initial transient and late nonlinearities
        tt = np.arange(seg.shape[0]) # time vector for velocity estimation
        vsh = np.polyfit(tt, seg[:, 0], 1)[0]      # steady joint velocity for shoulder (rad/step)
        vel = np.polyfit(tt, seg[:, 1], 1)[0]   # steady joint velocity for elbow (rad/step)
        rows.append((f, vsh, vel))
        if verbose:
            print(f"  f={f:.3f}  v_sh={vsh:+.4f}  v_el={vel:+.4f}")

    rows = np.array(rows)
    best = {}   # (joint, sign) -> (|velocity|, freq)
    for f, vsh, vel in rows:
        joint = 0 if abs(vsh) >= abs(vel) else 1            # 0=shoulder, 1=elbow
        v = vsh if joint == 0 else vel
        key = (joint, 1 if v > 0 else -1)
        if abs(v) > best.get(key, (0.0,))[0]:
            best[key] = (abs(v), float(f))
    primitives = [{"f": f, "joint": j, "sign": s, "scan_vel": mag}
                  for (j, s), (mag, f) in best.items()]
    primitives.sort(key=lambda p: (p["joint"], -p["sign"]))
    return primitives, rows


# --------------------------------------------------------------------------- #
#  neural observer (V2): estimate the hand WITHOUT hand feedback
# --------------------------------------------------------------------------- #
class HandObserver:
    """Forward observer for V2: replays the KNOWN, brain-invariant downstream
    (decoder -> muscle head -> arm-velocity model) on the observed neural to
    integrate a joint-angle estimate, using no hand feedback.

    Per the design decision to treat the fixed BMI apparatus as known, this loads
    the ANN weights (the *fixed* decoder + head, identical across trials); the
    per-trial *brain* is never used. The limitation is noise: the controller only
    receives the single noisy ``measure()`` each step (noise ~2.6), whereas the
    real arm is driven by the 100-sample average (noise ~0.25), so the observer
    runs the nonlinear head on a ~10x noisier signal -> its estimate drifts. That
    drift is exactly the V1-vs-V2 gap (the value of the hand measurement).
    """

    SH_DRY, SH_WET = 0.05, 7.5
    EL_DRY, EL_WET = 0.05, 5.0

    def __init__(self, smooth=0.0):
        from mock_plant import MuscleHead, _find_weights
        w = np.load(_find_weights())
        self.D = np.asarray(w["output_weight"], float) @ np.asarray(w["encoder_weight"], float)
        self.bias = np.asarray(w["output_bias"], float)
        self.head = MuscleHead(w)
        self.smooth = smooth          # optional EMA on neural to fight measurement noise
        self.reset()

    def reset(self):
        self.head.reset(); self.sh = 0.0; self.el = 0.0; self._y = None

    def step(self, neural):
        y = np.asarray(neural, float)
        if self.smooth > 0.0:
            self._y = y if self._y is None else self.smooth * self._y + (1 - self.smooth) * y
            y = self._y
        m = self.head.step(self.D @ y + self.bias)          # known head -> 4 muscle activations
        d_sh, d_el = m[0] - m[1], m[2] - m[3]               # antagonist differences
        self.sh += (d_sh - np.sign(d_sh) * self.SH_DRY) / self.SH_WET if abs(d_sh) > self.SH_DRY else 0.0
        self.el += (d_el - np.sign(d_el) * self.EL_DRY) / self.EL_WET if abs(d_el) > self.EL_DRY else 0.0
        return self.sh, self.el


# --------------------------------------------------------------------------- #
#  closed-loop hand controller (matches the official control_policy interface)
# --------------------------------------------------------------------------- #
class HandController:
    """Two IK joint-servos driven by select-tone primitives.

    Each joint error -> a select tone at that muscle's frequency, amplitude
    proportional to the error; the raised bias keeps x2 above the power
    threshold (power is "free"); dropping the tone lets stiction hold.

    mode='hand'   : V1 — joint estimate from the measured hand position.
    mode='neural' : V2 — joint estimate from a neural observer (added later).
    """

    def __init__(self, primitives, *, bias=0.5, mode="hand", k_gain=1.0,
                 a_min=0.14, a_max=0.34, tol=0.05, a_total_cap=0.48,
                 elbow_boost=1.7, obs_smooth=0.0):
        self.prim = {(p["joint"], p["sign"]): p["f"] for p in primitives}
        self.bias = bias; self.mode = mode; self.k = k_gain
        self.a_min = a_min; self.a_max = a_max; self.tol = tol; self.cap = a_total_cap
        self.elbow_boost = elbow_boost   # elbow muscle is higher-freq/weaker -> needs more drive
        self.observer = HandObserver(smooth=obs_smooth) if mode == "neural" else None
        self.reset()

    def reset(self):
        self.prev_sh = 0.0; self.prev_el = 0.0; self.tgt = None
        if self.observer is not None:
            self.observer.reset()

    def __call__(self, observations, target, current_pos):
        t = len(observations)
        if self.tgt is None:
            self.tgt = ik(target[0], target[1], 0.0, 0.0)
        # joint estimate
        if self.mode == "neural":                       # V2: neural observer, no hand feedback
            if observations:
                sh, el = self.observer.step(observations[-1])
            else:
                sh, el = 0.0, 0.0
        else:                                           # V1: from measured hand position
            sh, el = ik(current_pos[0], current_pos[1], self.prev_sh, self.prev_el)
            self.prev_sh, self.prev_el = sh, el
        errs = [(_wrap(self.tgt[0] - sh), 0), (_wrap(self.tgt[1] - el), 1)]

        tones = []
        for e, joint in errs:
            if abs(e) > self.tol:
                f = self.prim.get((joint, 1 if e > 0 else -1))
                if f is not None:
                    a = float(np.clip(self.k * abs(e), self.a_min, self.a_max))
                    if joint == 1:
                        a *= self.elbow_boost
                    tones.append((f, a))
        tot = sum(a for _, a in tones)
        if tot > self.cap:                      # share the [0,1] budget when both joints active
            tones = [(f, a * self.cap / tot) for f, a in tones]
        s = sum(a * np.sin(2 * np.pi * f * t) for f, a in tones)
        return np.clip(self.bias + np.array([s, s]), 0.0, 1.0)


# --------------------------------------------------------------------------- #
#  evaluation harness (mirrors eval/evaluation_notebook_template.ipynb)
# --------------------------------------------------------------------------- #
def run_trial(plant, controller, target, T=400):
    """One closed-loop trial; returns (hand_traj (T,2), dist-to-target (T,))."""
    controller.reset()
    obs = []
    hand = np.zeros((T, 2))
    for t in range(T):
        cp = np.array(plant.hand_pos)
        y = np.array(plant._brain.measure())
        u = controller(obs, target, cp)
        obs.append(y)
        plant.next_state(np.asarray(u, float).tolist())
        hand[t] = plant.hand_pos
    return hand, np.linalg.norm(hand - np.asarray(target, float), axis=1)


def make_targets(n_trials=10, seed=42, r_min=20.0, r_max=55.0, th_min=0.1, th_max=3.04):
    rng = np.random.default_rng(seed)
    r = rng.uniform(r_min, r_max, n_trials)
    th = rng.uniform(th_min, th_max, n_trials)
    return [(float(ri * np.cos(t)), float(ri * np.sin(t))) for ri, t in zip(r, th)]
