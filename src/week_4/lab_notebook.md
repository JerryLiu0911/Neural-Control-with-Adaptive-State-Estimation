# Lab notebook — Week 4: closed-loop hand control

This notebook records the engineering reasoning behind the Week-4 controller: the
plant analysis that had to precede any control design, the ladder of designs we
worked through (each discarded for a *specific, identifiable* reason), the
structure of the controller we settled on, and the evaluation metrics used to
judge it. As in Week 2, the intent is to justify the final design constructively —
each decision is tied to an observation about the plant rather than asserted.

All control logic lives in a single module, `hand_control.py`, and is exercised
through the official `control_policy(observations, target, current_pos)` interface
in `template.ipynb`.

---

## 1. Motivation and the shape of the problem

The task is to command a two-dimensional brain stimulation `u(t) ∈ [0,1]²` so that
a simulated hand reaches a target Cartesian position `(x, y)`. The only signals
available online are the brain's 16-dimensional neural measurement
`bmi._brain.measure()` and the current hand position `bmi.hand_pos`; the input is
applied with `bmi.next_state(u)`.

The end-to-end map from input to hand position is the composition

```
u → brain (LGSSM, per-trial random) → 16-D neural → ANN decoder → (x1, x2)
   → muscle head (nonlinear, stateful) → 4 muscle activations → 2-link arm → hand
```

This composition is **nonlinear**, **partially observable**, **non-invertible**,
and **stochastic**. Crucially, no single off-the-shelf controller applies to it
directly, because the difficulty is not in any one stage but in the *interface*
between a linear, identifiable front end (the brain) and a nonlinear, rhythm-coded
back end (the BMI decoder and arm). The first half of the work was therefore not
control design at all but **reverse-engineering the plant** to the point where a
control abstraction became visible.

---

## 2. Anatomy of the plant (what had to be discovered first)

We treat the per-trial **brain** as unknown but identifiable, and the **decoder +
muscle head + arm** as a *fixed, brain-invariant* apparatus that is the same on
every trial. This split is the single most important structural fact we exploit:
the random part of the plant (the brain) is the easy, linear part, and the hard,
nonlinear part (the BMI) never changes. The following properties were established
empirically, several of them by building an exact NumPy replica of the muscle head
(verified to match the provided Torch network to `3·10⁻⁷`) and using it as a
glass-box to read the head's hidden state.

**The decoder is linear, with two latents of distinct roles.** `(x1, x2) = D·y +
b`, where `D` is fixed. `x1` is the *muscle-selection / frequency* latent; `x2` is
the *power* latent.

**Muscle selection is frequency-addressed.** A bank of cos/sin band-pass filters
acts on the recent history of `x1`; each of the four muscles responds to a distinct
frequency (≈ 0.066, 0.131, 0.213, 0.311 cycles/sample). Driving `x1` to oscillate
near one of these frequencies selects the corresponding muscle. The selector
saturates easily, so *selection is the easy half*.

**Power is governed by `x2`, and — decisively — power is "free".** The head turns a
muscle on only when `x2` clears a wake threshold (≈ ±0.72) with persistent
"evidence". We found that `x2` sits at a large *positive DC offset* (≈ +1.8) under
neutral input, which already exceeds the threshold; a *constant-high* `x2` sustains
power indefinitely (glass-box: power held at 0.65 ± 0.00). This was the pivotal
discovery — it removed the need to synthesise a clean slow "power rhythm", which we
had initially assumed was required and which proved nearly impossible to produce.

**Muscle activation drives joint velocity, not position.** Each muscle's activation
is `power × selector ∈ [0,1]`; the arm converts antagonist differences into joint
*angular velocity* through a dry-(stiction)/wet-(viscous) friction model. Hand
position is therefore the *time-integral* of a velocity we can modulate, and below
the stiction threshold the joint *holds*. This makes feedback-plus-integration the
natural control idiom, and gives a free "brake".

**The brain transports frequency linearly, but is noisy.** Because the brain is
(approximately) LTI, a sinusoidal input at frequency `f` produces neural — and
hence latent — oscillation at the same `f`; the brain only changes the *amplitude
and phase*. However, a single `measure()` is extremely noisy (≈ 2.6 per sample);
the arm is driven by a 100-sample average (≈ 0.25 residual), but the controller
only *sees* the single noisy sample. This noise is large enough that the nonlinear
head occasionally stalls, which makes open-loop drive unreliable and forces both a
closed loop and multi-trial evaluation.

---

## 3. Why standard linear control does not transfer

Before committing to a bespoke design we considered, and rejected, the obvious
model-based controllers, each for a concrete reason rooted in §2.

**LQG / LQR.** LQG regulates a *linear* state to a *static* setpoint using a Kalman
estimate. Two mismatches make it the wrong instrument here. First, the controlled
variable — hand position — is not in the brain's state or observations (there is no
arm→brain feedback), so a Kalman filter on the brain is blind to the hand. Second,
the muscle head needs the latents to *oscillate* to produce any motion; holding a
static setpoint produces zero movement. LQG's defining behaviour (settle to a
constant) is the opposite of what the plant requires.

**Feed-forward inversion / a learned input basis.** For the *linear* inner channel
(`u → latent`) one can, in principle, build an orthogonal Fourier/SVD basis and
invert the brain's frequency response to synthesise any reachable latent
trajectory. This is elegant and we use its spirit, but it cannot cross the
nonlinearity: a *position* reference is not a linear functional of the latents, and
pure feed-forward has no mechanism to correct the integrator under brain noise.

**MPC.** Linear MPC is the constrained, receding-horizon generalisation of the
above and would handle the `[0,1]` input limit natively, but it still only reaches
the latent; targeting the hand needs *nonlinear* MPC over the full (unknown,
stochastic) stack — a non-convex problem whose cost would not spontaneously
discover that it must oscillate at specific frequencies.

The conclusion is structural: the linear tools belong on the *inner* channel
(produce the right latent oscillation), and the *outer* problem (position, through
a nonlinear integrator) must be closed with feedback and a velocity abstraction.
This hierarchy is the backbone of the final design.

---

## 4. Design approach: a ladder to the select-tone servo

As in Week 2 we present the design as a sequence of models, each adding one
assumption and each failing a specific, observable requirement until the last
satisfies them all.

### 4.1 Naive two-tone primitive

The first idea followed the literal description of the head: drive `x1` with a
*select tone* (to pick the muscle) and `x2` with a slow *power tone* (to power it),
summed on both input channels. This fails. The select tone, transported through the
brain into both latents, **contaminates `x2`**: `x2` dips through the power
threshold, and because the head's power *exits* fast (rate 0.5) but *enters* slowly
(rate 0.055), each dip crashes the power and it recovers only slowly. The observed
symptom was a single large "fire-once" slew followed by a latch-off that could not
be re-triggered. *Requirement failed: sustained, repeatable, graded motion.*

### 4.2 Clean separation by input-direction nulling

The principled fix is to use the two input channels to *separate* the latents: at
the select frequency choose the input that drives `x1` while nulling `x2` (and vice
versa at the power frequency), and centre `x2`'s DC. Using the known decoder and an
online estimate of the brain's frequency response, this is two well-conditioned
2×2 solves. It works in the glass-box, but on the real plant it runs into the
`[0,1]` **input budget**: centring the large `x2` offset consumes nearly all the
headroom, the select tone then clips, and clipping (a nonlinearity) re-introduces
the very contamination we removed. *Requirement failed: realisability within the
input constraint.* The attempt was not wasted — it located the true obstacle (the
`x2` offset and the budget), which the next step turns from a liability into an
asset.

### 4.3 Power from the offset (the simplification)

The breakthrough of §2 — that constant-high `x2` sustains power — collapses the
problem. We do **not** synthesise a power rhythm, do **not** centre `x2`, and do
**not** need the power tone at all. The natural `x2` offset is left in place to keep
power on "for free"; we only inject a *single select tone*, kept at an amplitude low
enough that (a) it does not clip and (b) it does not perturb `x2` below threshold.
This removes the budget fight entirely and yields sustained, graded motion whose
speed is set by the select amplitude. *All §4.1–4.2 requirements now satisfied for a
single muscle.*

### 4.4 Joint decoupling and IK control

The four muscles are two antagonist pairs: shoulder is driven by muscles 0/1, elbow
by 2/3, and the arm's velocity for each joint depends only on its own pair. Control
therefore **decouples into two independent one-dimensional joint servos**. Because
the muscle→joint map is fixed (brain-invariant) and the arm geometry (link lengths)
is known — it is the same geometry the evaluation harness assumes, and is *not* the
neural-network weights — we control in **IK joint space**: invert the target hand
position to target joint angles, invert the current hand position to current joint
angles, and run a proportional servo per joint. This is far more robust than
hand-space Jacobian inversion because from the home pose shoulder and elbow both
move the hand in the same direction and are otherwise hard to disambiguate.

### 4.5 Closing the loop

A static drive is unreliable under the measurement noise of §2, so the servo must be
closed. Each step the joint errors set the select-tone amplitudes; inside a small
deadzone the tones are dropped and joint stiction holds the pose. Feedback both
rejects the stochastic stalls (a stalled muscle leaves a persistent error that keeps
driving it) and absorbs the per-trial variation in brain gain (a low-gain brain
simply holds the error longer at higher amplitude). The select **frequencies and the
joint map are calibrated once offline** (they are brain-invariant); only the
*effective gain* varies per trial, and that is handled by the loop rather than by
re-identification.

---

## 5. The controller in detail

### 5.1 Offline calibration (`calibrate_frequencies`)

On one probe plant we sweep the select frequency, drive a short burst at each, and
measure the steady joint angular velocity (read, for this offline structure-discovery
step only, from the true joint angles, since IK is unreliable over the large
open-loop swings of a calibration burst). The four peaks give the brain-invariant
`(frequency → joint, direction)` primitives: shoulder± at ≈ 0.073/0.133, elbow± at
≈ 0.207/0.312. Online the controller never reads joint angles; it uses only
`hand_pos` and IK.

### 5.2 Online policy (`HandController`)

At each step, with `t = len(observations)`:

1. **Estimate joints.** V1 inverts the measured `current_pos`; V2 uses the observer
   of §5.3.
2. **Errors.** Wrap the target-minus-current shoulder and elbow errors.
3. **Select tones.** For each joint whose error exceeds the deadzone, emit a tone at
   that muscle's frequency with amplitude proportional to the error, clipped to a
   working range. The elbow receives an **amplitude boost** (it is the weaker,
   higher-frequency muscle — see §6/§7), and the total amplitude is capped to share
   the `[0,1]` budget when both joints are active.
4. **Compose `u`.** Sum the tones around a raised bias (which lifts `x2`'s margin
   above the power threshold) on both channels, clipped to `[0,1]`.

Because the policy is stateful (tone phase, IK history, observer), it resets at the
start of each trial, detected by an empty `observations` list.

### 5.3 Two versions and the role of feedback

The two versions differ *only* in where the joint estimate comes from, which makes
their comparison a clean measurement of the value of hand feedback.

- **V1 — hand feedback.** The joint estimate is `IK(current_pos)`. This is the
  strong baseline.
- **V2 — neural observer only.** No hand signal is used. Instead a forward observer
  *replays the known, fixed downstream* (decoder → muscle head → arm-velocity model)
  on the observed neural and integrates a joint estimate (`HandObserver`). This is
  legitimate under our standing decision to treat the fixed BMI apparatus as known
  while keeping the per-trial *brain* black-box. Its fundamental difficulty is noise:
  the observer is fed the *single* noisy `measure()` (≈ 2.6), whereas the real arm is
  driven by the 100-sample average (≈ 0.25), so the nonlinear head is replayed on a
  ~10× noisier signal and the estimate drifts. The V1–V2 gap therefore *quantifies*
  how much the direct hand measurement is worth.

The conceptual link to Week 2 is direct: V1 is a "predict + measurement-update"
estimator and V2 is "predict only" — a Kalman filter with and without its correction
step, applied to the hand instead of the latent state.

---

## 6. Evaluation metrics and their justification

Control quality is judged on the official polar grid of targets (radii × angles),
which probes the whole reachable workspace rather than a few hand-picked points. We
report a small battery of metrics, each chosen to expose a *distinct* aspect of
behaviour; together they separate "did it get there", "how fast", "how precisely",
"why it failed", and "at what cost".

- **Final distance to target (cm).** The headline task metric: Euclidean hand-to-
  target distance at the last step. Reported as a distribution (mean and std), because
  outcomes are bimodal (see §7) and the mean alone is misleading.

- **Time to threshold (steps to < 10 cm).** Responsiveness — how quickly the hand
  first enters a tolerance ball. Reported with the fraction of trials that ever reach
  it, since a controller can be fast on the ones it solves yet fail others outright.

- **Steady-state distance (mean over the final third).** Precision and *settling*: it
  penalises a controller that brushes the target then drifts or hunts, which a single
  final-step reading would miss. This is where the overshoot/limit-cycle behaviour
  shows up.

- **Final shoulder and elbow angle error (deg).** A *diagnostic decomposition* of the
  Cartesian error into the two actuated degrees of freedom. This is the metric that
  isolates the dominant failure mode: a large elbow error against a small shoulder
  error localises the problem to the weak, high-frequency elbow muscle rather than to
  the control law in general.

- **Control effort (sum of squared inputs).** Efficiency, and a guard against
  "cheating" by saturating the inputs; broken out by target radius to show how effort
  scales with reach demand.

- **Spatial final-distance heatmap.** A map of final error over the workspace. This is
  the most informative single panel for our plant: it reveals the *structure* of the
  failures — a reachable near-field versus a far/behind shortfall — which a scalar
  average hides, and directly visualises the reach limitation of §7.

- **Convergence curves and an example trajectory.** The time course (distance vs.
  step, trial-averaged with a spread band) exposes the ~150-step power-ramp latency
  and any hunting; the example trajectory shows the characteristic "shoulder sweeps to
  the angle, elbow bends in" path qualitatively.

The grid is run on a fixed brain seed so that differences across targets reflect
target difficulty rather than per-brain variation; per-brain robustness is a separate
axis, examined by varying the seed.

---

## 7. Limitations and honest trade-offs

The design is deliberately simple and its failures are interpretable; we record them
because they are as informative as the successes.

1. **Elbow under-actuation (dominant).** The elbow muscles sit at higher select
   frequencies, where the brain's response rolls off (low-pass): a given input
   produces less `x1` oscillation, so the elbow muscle is weaker and slower. Interior
   targets that need a large elbow bend are therefore *reach-limited*, producing a
   large radial residual and the elbow-dominated angle error seen in the metrics. This
   is a genuine property of the frequency-multiplexed code, not a tuning oversight: the
   same mechanism that makes selection clean penalises the high-frequency muscles.

2. **Shared input budget.** With `u ∈ [0,1]` and both joint tones summed on two
   channels, amplitude is capped to avoid clipping; when both joints must move, each is
   under-driven, and the already-weak elbow suffers most.

3. **Overshoot / minimum-burst granularity.** Power ramps over tens of steps and exits
   quickly, so the smallest *sustained* burst that moves a joint can be larger than the
   final correction required — the hand approaches, overshoots, and hunts. This is the
   same "can't stop precisely" wall reached by the primitive-based controller, from a
   different direction.

4. **Stochasticity.** The heavy single-sample measurement noise occasionally stalls a
   muscle; the closed loop recovers most stalls but at a cost in time and final
   accuracy, and it is the reason evaluation must be multi-trial.

5. **IK branch flips.** Near singular poses the IK joint estimate can jump between
   elbow-up/down branches; the hand-space final-distance metric is immune to this, but
   the joint-error panels can be momentarily distorted.

The clear directions for improvement follow from these causes: **sequential per-joint
phases** (give each joint the full input budget in turn) to relieve (2); a short
**online elbow-amplitude self-calibration** to relieve (1) per brain; **input-
direction (FRF) design** to extract more elbow `x1` per unit input; and a
**graded-stop / anti-overshoot** rule to relieve (3). The neural-observer variant
(§5.3) is the natural next study, turning the V1–V2 gap into a quantitative result on
the value of feedback.
