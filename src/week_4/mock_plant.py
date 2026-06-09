"""Week 4 — pure-numpy stand-in for ``BMI_and_Hand`` (offline development only).

Faithfully reimplements the decoder -> muscle head -> arm chain from
``week_3/BMI_and_Hand.py`` using the provided ``...weights.npz``, driven by a
tunable stand-in linear-Gaussian brain. This lets the controller be developed
and tested without ``torch`` / the ``GG4`` wheel; the real ``BMI_and_Hand`` is a
drop-in replacement (same ``next_state`` / ``measure`` / ``hand_pos`` surface).

The weights are used ONLY by this plant (replicating the simulator). The
controller in ``hand_control.py`` never touches them — it stays black-box.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np

_WEIGHTS = "neural_activity_to_muscle_weights.npz"


def _find_weights() -> Path:
    here = Path(__file__).resolve().parent
    for cand in (here / _WEIGHTS, here.parent / "week_3" / _WEIGHTS, Path(_WEIGHTS)):
        if cand.exists():
            return cand
    raise FileNotFoundError(f"{_WEIGHTS} not found next to mock_plant.py or in week_3/")


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# =========================================================================== #
#  Muscle head — numpy port of _LatentToMuscleHead._compute_from_cache
# =========================================================================== #
class MuscleHead:
    """Stateful 2 -> 4 muscle head: x1 freq selects muscle, x2 rhythm powers it."""

    def __init__(self, w):
        self.cos = np.asarray(w["muscle_filter_cos"], float)      # (4, K)
        self.sin = np.asarray(w["muscle_filter_sin"], float)      # (4, K)
        self.K = self.cos.shape[1]
        self.sel_thr = np.asarray(w["selector_threshold"], float) # (4,)
        self.sel_scale = float(w["selector_scale"].mean()) if np.ndim(w["selector_scale"]) else float(w["selector_scale"])
        self.sel_gain = float(w["selector_gain"])
        self.pos_thr = float(w["power_positive_threshold"])
        self.neg_thr = float(w["power_negative_threshold"])
        self.sharp = float(w["power_wake_sharpness"])
        self.peak_thr = float(w["power_peak_threshold"])
        self.half = int(w["power_half_period_steps"])
        self.full = int(w["power_full_period_steps"])
        self.hw = float(w["power_half_pair_weight"])
        self.fw = float(w["power_full_pair_weight"])
        self.ev_thr = float(w["power_evidence_threshold"])
        self.hit_decay = float(w["power_evidence_hit_decay"])
        self.miss_decay = float(w["power_evidence_miss_decay"])
        self.add_rate = float(w["power_evidence_add_rate"])
        self.drive_thr = float(w["power_drive_threshold"])
        self.drive_scale = float(w["power_drive_scale"])
        self.enter = float(w["power_enter_rate"])
        self.exit = float(w["power_exit_rate"])
        self.reset()

    def reset(self):
        self.cache = np.zeros((self.K, 2))   # rolling latent history (oldest..newest)
        self.power = 0.0
        self.evidence = 0.0

    def _wake(self, v):
        return _sigmoid(self.sharp * (v - self.pos_thr)), _sigmoid(self.sharp * (-v - self.neg_thr))

    def step(self, latent_xy) -> np.ndarray:
        # roll new latent into the cache
        self.cache = np.vstack([self.cache[1:], np.asarray(latent_xy, float)[None, :]])
        x1_hist = self.cache[:, 0]
        x2_hist = self.cache[:, 1]

        # x1 -> band energy -> selector (conv1d with full-length kernel = dot)
        cos_r = self.cos @ x1_hist
        sin_r = self.sin @ x1_hist
        band = np.sqrt(cos_r ** 2 + sin_r ** 2 + 1e-12)
        selector = np.clip(self.sel_gain * (band - self.sel_thr) / (self.sel_scale + 1e-12), 0.0, 1.0)

        # x2 -> periodicity evidence -> power
        cur, halfv, fullv = x2_hist[-1], x2_hist[-1 - self.half], x2_hist[-1 - self.full]
        pn, nn = self._wake(cur)
        ph, nh = self._wake(halfv)
        pf, nf = self._wake(fullv)
        half_pair = pn * nh + nn * ph
        full_pair = pn * pf + nn * nf
        period_evidence = np.clip(self.hw * half_pair + self.fw * full_pair, 0.0, 1.0)
        peak = max(pn, nn)
        hit = (period_evidence > self.ev_thr) and (peak > self.peak_thr)
        strengthened = self.hit_decay * self.evidence + self.add_rate * period_evidence
        faded = self.miss_decay * self.evidence
        self.evidence = float(np.clip(strengthened if hit else faded, 0.0, 1.0))
        drive = np.clip((self.evidence - self.drive_thr) / (self.drive_scale + 1e-12), 0.0, 1.0)
        rate = self.enter if drive > self.power else self.exit
        self.power = float(np.clip((1.0 - rate) * self.power + rate * drive, 0.0, 1.0))

        return np.clip(self.power * selector, 0.0, 1.0)   # (4,) muscle activations


# =========================================================================== #
#  Arm — numpy port of _Arm
# =========================================================================== #
class Arm:
    UPPER = 30.0
    FORE = 30.0
    SH_DRY, SH_WET = 0.05, 7.5
    EL_DRY, EL_WET = 0.05, 5.0

    def __init__(self):
        self.sh = 0.0
        self.el = 0.0

    @property
    def hand_pos(self) -> np.ndarray:
        x = self.UPPER * np.cos(self.sh) + self.FORE * np.cos(self.sh + self.el)
        y = self.UPPER * np.sin(self.sh) + self.FORE * np.sin(self.sh + self.el)
        return np.array([x, y])

    def move(self, m):
        sp, sn, ep, en = (float(v) for v in m)   # shoulder+/-, elbow+/-
        d_sh = sp - sn
        d_el = ep - en
        self.sh += (d_sh - np.sign(d_sh) * self.SH_DRY) / self.SH_WET if abs(d_sh) > self.SH_DRY else 0.0
        self.el += (d_el - np.sign(d_el) * self.EL_DRY) / self.EL_WET if abs(d_el) > self.EL_DRY else 0.0


# =========================================================================== #
#  Stand-in linear brain  (u in [0,1]^2  ->  16-D neural activity)
# =========================================================================== #
class StandInBrain:
    def __init__(self, seed=0, n=6, p=16, m=2, rho=0.8, obs_noise=0.02, out_gain=1.0):
        rng = np.random.default_rng(seed)
        M = 0.5 * (rng.standard_normal((n, n)) - rng.standard_normal((n, n)).T)
        self.A = rho * M / (np.max(np.abs(np.linalg.eigvals(M))) + 1e-9)
        self.B = rng.standard_normal((n, m)) / np.sqrt(n)
        self.C = out_gain * rng.standard_normal((p, n))
        self.R = obs_noise ** 2 * np.eye(p)
        self.rng, self.p, self.input_dim = rng, p, m
        self.x = rng.standard_normal(n) * 0.1
        self.current_time_stamp = 0

    def measure(self):
        v = self.rng.multivariate_normal(np.zeros(self.p), self.R)
        return (self.C @ self.x + v).tolist()

    def next_state(self, u=None):
        u = np.zeros(self.input_dim) if u is None else np.clip(np.asarray(u, float), 0.0, 1.0)
        self.x = self.A @ self.x + self.B @ u
        self.current_time_stamp += 1


# =========================================================================== #
#  MockBMI — same surface as BMI_and_Hand
# =========================================================================== #
class MockBMI:
    """Drop-in stand-in for ``BMI_and_Hand``: same ``_brain`` / ``_arm`` /
    ``hand_pos`` / ``next_state`` surface the official eval harness uses
    (``bmi._brain.measure()``, ``bmi.hand_pos``, ``bmi.next_state(u)``)."""

    repeat_measurement = 20   # denoising average that drives the arm (cheaper than 100)

    def __init__(self, brain, *, auto_gain=True):
        w = np.load(_find_weights())
        self.D = np.asarray(w["output_weight"], float) @ np.asarray(w["encoder_weight"], float)  # (2,16)
        self.bias = np.asarray(w["output_bias"], float)
        self._head = MuscleHead(w)
        self._arm = Arm()
        self._brain = brain
        self.input_dim = brain.input_dim
        if auto_gain:
            self._auto_gain()

    def _decode(self, y):
        return self.D @ np.asarray(y, float) + self.bias

    def _auto_gain(self, target_x2_amp=1.6):
        """Scale the stand-in brain so a reference oscillation drives x2 past the
        power wake threshold (keeps the mock 'live').

        TRADE-OFF: this normalises every mock brain to a similar latent gain, so
        per-trial gain variation on the mock is *smaller* than the real GG4
        brains. Robustness to gain spread must therefore be probed deliberately
        (vary ``target_x2_amp`` across seeds). The real BMI needs no scaling.
        """
        import copy
        probe = copy.deepcopy(self._brain)
        f = 0.028
        xs = []
        for t in range(400):
            u = 0.5 + 0.45 * np.sin(2 * np.pi * f * t) * np.ones(self.input_dim)
            probe.next_state(u)
            xs.append(self._decode(probe.measure())[1])
        amp = (np.max(xs[100:]) - np.min(xs[100:])) / 2 + 1e-9
        if hasattr(self._brain, "C"):
            self._brain.C *= target_x2_amp / amp

    @property
    def hand_pos(self):
        return self._arm.hand_pos

    def next_state(self, u=None):
        self._brain.next_state(u)
        ys = np.array([self._brain.measure() for _ in range(self.repeat_measurement)])
        self._arm.move(self._head.step(self._decode(ys.mean(0))))   # denoised latent drives arm
        return np.array(self._brain.measure())      # noisy obs returned to the controller


def make_mock(seed=0, **brain_kw) -> "MockBMI":
    """Factory mirroring ``BMI_and_Hand(Brain(random_seed=seed))``."""
    return MockBMI(StandInBrain(seed=seed, **brain_kw))
