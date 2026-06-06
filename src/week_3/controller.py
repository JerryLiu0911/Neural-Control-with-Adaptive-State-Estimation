"""Week 3 — Closed-loop control of the GG4 ``Brain`` (individual).

Everything for the control task lives in this single module:

  1. ``LinearModel``        — identified state-space model container.
  2. ``identify_model``     — EM system-ID from *known* probing inputs
                              (we command ``u``, so this is well-posed, unlike
                              the blind estimator of Week 2).
  3. design routines        — steady-state Kalman gain, LQR, and LQI (servo
                              with integral action), plus the input setpoint.
  4. ``Controller``         — online output-feedback policy with modes
                              ``'p'`` (baseline), ``'lqg'``, ``'lqi'``.
  5. ``BrainLike``          — drop-in stand-in for ``GG4.Brain`` (same
                              interface + ``[0, 1]`` input clipping) so the
                              notebook runs before the Linux wheel is in place.
  6. experiment + metrics   — ``run_open_loop`` / ``run_closed_loop`` and
                              evaluation helpers used from the notebook.

Model convention (matches the Week-1 ``l4b`` Simulator and the documented
brain): the input affects the *next* state and there is no direct feedthrough,

    x_{t+1} = A x_t + B u_t + w_t,   w_t ~ N(0, Q)
    y_t     = C x_t + v_t,           v_t ~ N(0, R)

with u_t clipped element-wise to [0, 1] inside the plant.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np
from scipy.linalg import solve_discrete_are

# ---- frozen numerics ------------------------------------------------------- #
_JITTER = 1e-6        # diagonal regularisation on covariance / inverse updates
_EM_ITERS = 150       # EM iterations cap (convergence is usually earlier)
_EM_TOL = 1e-5        # relative log-likelihood convergence tolerance
_INPUT_LO, _INPUT_HI = 0.0, 1.0   # plant input clipping range


# =========================================================================== #
#  Model container
# =========================================================================== #
@dataclass
class LinearModel:
    """Identified discrete-time LG-SSM ``(A, B, C, Q, R)`` with no feedthrough."""

    A: np.ndarray
    B: np.ndarray
    C: np.ndarray
    Q: np.ndarray
    R: np.ndarray

    @property
    def n(self) -> int:  # latent state dim
        return self.A.shape[0]

    @property
    def m(self) -> int:  # input dim
        return self.B.shape[1]

    @property
    def p(self) -> int:  # observation dim
        return self.C.shape[0]

    # -- structural diagnostics --------------------------------------------- #
    def spectral_radius(self) -> float:
        return float(np.max(np.abs(np.linalg.eigvals(self.A))))

    def dc_gain(self) -> np.ndarray:
        """Steady-state output gain ``G = C (I - A)^{-1} B``  (p x m)."""
        I = np.eye(self.n)
        return self.C @ np.linalg.solve(I - self.A, self.B)

    def controllability_rank(self) -> int:
        blocks = [self.B]
        for _ in range(self.n - 1):
            blocks.append(self.A @ blocks[-1])
        return int(np.linalg.matrix_rank(np.hstack(blocks)))

    def observability_rank(self) -> int:
        blocks = [self.C]
        for _ in range(self.n - 1):
            blocks.append(blocks[-1] @ self.A)
        return int(np.linalg.matrix_rank(np.vstack(blocks)))


# =========================================================================== #
#  Kalman filter / RTS smoother for an LG-SSM with KNOWN inputs
# =========================================================================== #
def _kalman_filter(Y, U, A, B, C, Q, R, x0, P0):
    """Forward filter. ``Y, U`` are ``(T, p)`` and ``(T, m)``; ``U[t]`` is the
    input applied at step ``t`` (driving the transition into ``t + 1``)."""
    T, p = Y.shape
    n = A.shape[0]
    xf = np.zeros((T, n)); Pf = np.zeros((T, n, n))
    xp = np.zeros((T, n)); Pp = np.zeros((T, n, n))
    I = np.eye(n)
    ll = 0.0
    x_prev, P_prev = x0.copy(), P0.copy()
    for t in range(T):
        if t == 0:
            x_pred, P_pred = x0.copy(), P0.copy()
        else:
            x_pred = A @ x_prev + B @ U[t - 1]
            P_pred = A @ P_prev @ A.T + Q
        xp[t], Pp[t] = x_pred, P_pred

        S = C @ P_pred @ C.T + R
        e = Y[t] - C @ x_pred
        K = np.linalg.solve(S, C @ P_pred).T
        x_filt = x_pred + K @ e
        P_filt = (I - K @ C) @ P_pred
        P_filt = 0.5 * (P_filt + P_filt.T)
        xf[t], Pf[t] = x_filt, P_filt
        x_prev, P_prev = x_filt, P_filt

        _, logdet = np.linalg.slogdet(S)
        ll += -0.5 * (logdet + e @ np.linalg.solve(S, e) + p * np.log(2 * np.pi))
    return xf, Pf, xp, Pp, ll


def _rts_smoother(A, xf, Pf, xp, Pp):
    T, n = xf.shape
    xs = xf.copy(); Ps = Pf.copy()
    Plag = np.zeros((T, n, n))   # Plag[t] = Cov(x_t, x_{t-1} | Y)
    for t in range(T - 2, -1, -1):
        J = np.linalg.solve(Pp[t + 1], A @ Pf[t]).T
        xs[t] = xf[t] + J @ (xs[t + 1] - xp[t + 1])
        Ps[t] = Pf[t] + J @ (Ps[t + 1] - Pp[t + 1]) @ J.T
        Plag[t + 1] = J @ Ps[t + 1]
    return xs, Ps, Plag



#  System identification (improved from Week 2 code)

def _stabilize(A, rho_max=0.98):
    """Reflect any unstable modes back inside the unit circle (warm start only).

    The raw subspace realisation is not guaranteed stable; a wildly unstable A
    can blow up the first EM E-step, so we cap the spectral radius. This only
    seeds EM, which then refines A freely.
    """
    try:
        w, V = np.linalg.eig(A)
        mag = np.abs(w)
        if np.any(mag > rho_max):
            w = np.where(mag > rho_max, w / mag * rho_max, w)
            A = np.real(V @ np.diag(w) @ np.linalg.inv(V))
    except np.linalg.LinAlgError:
        pass                                          # defective A: leave as-is
    return A


def _ls_B_x0(U, Y, A, C):
    """Least-squares (B, x0) given (A, C) and the known input-output data.

    The output is linear in (x0, vec(B)): simulate the response to each basis
    parameter (unit x0 directions with B=0, then unit B entries with x0=0) to
    build the design matrix, and regress the observations onto it.
    """
    T, p = Y.shape
    n, m = A.shape[0], U.shape[1]

    def sim_output(x0, B):
        x = x0.copy(); out = np.empty((T, p))
        for t in range(T):
            out[t] = C @ x
            x = A @ x + B @ U[t]
        return out.reshape(-1)

    cols = [sim_output(np.eye(n)[j], np.zeros((n, m))) for j in range(n)]
    for r in range(n):
        for c in range(m):
            Bb = np.zeros((n, m)); Bb[r, c] = 1.0
            cols.append(sim_output(np.zeros(n), Bb))
    Phi = np.stack(cols, axis=1)                      # (T*p, n + n*m)
    theta, *_ = np.linalg.lstsq(Phi, Y.reshape(-1), rcond=None)
    return theta[n:].reshape(n, m), theta[:n]         # B, x0


def _warm_start(U, Y, n):
    """Input-aware subspace (MOESP) warm start returning (C, A, B, x0).

    Future output and input block-Hankel matrices are formed; the future-input
    row space is projected out of the future outputs (so the deterministic input
    response is removed), and the column space of the residual is the extended
    observability matrix -> (C, A) via shift-invariance. With (A, C) fixed,
    (B, x0) follow from a linear least-squares against the known data. Valid for
    n > p, and seeds a real B (unlike the output-only / random start).
    """
    U = np.asarray(U, float); Y = np.asarray(Y, float)
    T, p = Y.shape
    m = U.shape[1]
    i = max(2, int(np.ceil((n + p) / p)) + 1)         # block rows; (i-1)*p >= n
    N = T - i + 1                                      # columns (time index)

    Yf = np.empty((i * p, N)); Uf = np.empty((i * m, N))
    for r in range(i):
        Yf[r * p:(r + 1) * p] = Y[r:r + N].T          # future output block-Hankel
        Uf[r * m:(r + 1) * m] = U[r:r + N].T          # future input  block-Hankel

    # project the future-input row space out of the future outputs (MOESP)
    coef = np.linalg.lstsq(Uf.T, Yf.T, rcond=None)[0]  # Uf^T coef ~ Yf^T
    Yf_perp = Yf - (Uf.T @ coef).T                     # (i*p, N)

    # column space of the residual = extended observability matrix [C; CA; ...]
    Ug, _, _ = np.linalg.svd(Yf_perp, full_matrices=False)
    O = Ug[:, :n]                                      # (i*p, n)
    C = O[:p]                                          # first block row -> (p, n)
    A = np.linalg.lstsq(O[:-p], O[p:], rcond=None)[0]  # shift-invariance
    A = _stabilize(A)

    B, x0 = _ls_B_x0(U, Y, A, C)
    return C, A, B, x0


def _warm_start_pca(Y, n, m, seed=0):
    """Baseline warm start (pre-improvement): C, A from PCA of the outputs and a
    *random* B. Ignores the inputs and is only valid for n <= p. Kept for the
    before/after comparison against the input-aware subspace start.
    """
    mu = Y.mean(0, keepdims=True)
    _, _, Vt = np.linalg.svd(Y - mu, full_matrices=False)
    C = Vt[:n].T                                       # leading PCA directions
    X = (Y - mu) @ C                                   # crude state proxy
    A = np.linalg.lstsq(X[:-1], X[1:], rcond=None)[0].T
    B = 0.01 * np.random.default_rng(seed).standard_normal((n, m))
    x0 = np.zeros(n)
    return C, A, B, x0


def identify_model(
    U: np.ndarray,
    Y: np.ndarray,
    n: int,
    *,
    iters: int = _EM_ITERS,
    tol: float = _EM_TOL,
    warm_start: str = "subspace",
    verbose: bool = False,
) -> Tuple[LinearModel, np.ndarray]:
    """Identify ``(A, B, C, Q, R)`` from a probing run with known inputs.

    Parameters
    ----------
    U : (T, m) inputs actually applied (``U[t]`` drives ``x_t -> x_{t+1}``).
    Y : (T, p) noisy observations recorded during the same run.
    n : latent state dimension to fit.

    Returns
    -------
    model    : identified ``LinearModel``.
    ll_trace : log-likelihood per EM iteration (for convergence plots).
    """
    U = np.asarray(U, float); Y = np.asarray(Y, float)
    T, p = Y.shape
    m = U.shape[1]

    if warm_start == "subspace":
        C, A, B, x0 = _warm_start(U, Y, n)
    elif warm_start == "pca":
        C, A, B, x0 = _warm_start_pca(Y, n, m)
    else:
        raise ValueError(f"unknown warm_start {warm_start!r}")
    Q = 0.1 * np.eye(n)
    R = 0.1 * np.eye(p)
    P0 = np.eye(n)

    ll_trace = []
    prev = -np.inf
    for it in range(iters):
        # -- E-step ---------------------------------------------------------- #
        xf, Pf, xp, Pp, ll = _kalman_filter(Y, U, A, B, C, Q, R, x0, P0)
        xs, Ps, Plag = _rts_smoother(A, xf, Pf, xp, Pp)
        ll_trace.append(ll)

        # -- sufficient statistics ------------------------------------------ #
        # dynamics regression  x_t ~ [x_{t-1}; u_{t-1}]
        Sxx_t   = np.zeros((n, n))         # sum E[x_t x_t^T],          t>=1
        Sx_phi  = np.zeros((n, n + m))     # sum E[x_t [x_{t-1};u]^T]
        Sphiphi = np.zeros((n + m, n + m)) # sum E[[x_{t-1};u][..]^T]
        # observation regression  y_t ~ x_t  (all t)
        Exx_all = np.zeros((n, n))
        Syx     = np.zeros((p, n))
        Syy     = np.zeros((p, p))
        for t in range(T):
            Exx = Ps[t] + np.outer(xs[t], xs[t])
            Exx_all += Exx
            Syx += np.outer(Y[t], xs[t])
            Syy += np.outer(Y[t], Y[t])
            if t >= 1:
                Exx_prev = Ps[t - 1] + np.outer(xs[t - 1], xs[t - 1])
                Ex_xprev = Plag[t] + np.outer(xs[t], xs[t - 1])
                u = U[t - 1]
                Sxx_t += Exx
                Sx_phi[:, :n] += Ex_xprev
                Sx_phi[:, n:] += np.outer(xs[t], u)
                Sphiphi[:n, :n] += Exx_prev
                Sphiphi[:n, n:] += np.outer(xs[t - 1], u)
                Sphiphi[n:, :n] += np.outer(u, xs[t - 1])
                Sphiphi[n:, n:] += np.outer(u, u)
        Ndyn = T - 1

        # -- M-step ---------------------------------------------------------- #
        AB = np.linalg.solve(Sphiphi + _JITTER * np.eye(n + m), Sx_phi.T).T
        A, B = AB[:, :n], AB[:, n:]
        Q = (Sxx_t - AB @ Sx_phi.T) / Ndyn
        Q = 0.5 * (Q + Q.T) + _JITTER * np.eye(n)

        C = np.linalg.solve(Exx_all + _JITTER * np.eye(n), Syx.T).T
        R = (Syy - C @ Syx.T) / T
        R = 0.5 * (R + R.T) + _JITTER * np.eye(p)

        x0 = xs[0].copy()

        if verbose and (it % 10 == 0 or it == iters - 1):
            print(f"  EM iter {it:3d}  ll = {ll:.3f}")
        if abs(ll - prev) < tol * max(1.0, abs(prev)):
            break
        prev = ll

    return LinearModel(A, B, C, Q, R), np.asarray(ll_trace)


def select_latent_dim(
    U: np.ndarray, Y: np.ndarray, candidates, **kw
) -> list:
    """Fit a model for each candidate ``n`` and report the converged
    log-likelihood — used to pick the latent dimension via an elbow."""
    out = []
    for n in candidates:
        model, ll = identify_model(U, Y, n, **kw)
        out.append((n, float(ll[-1]), model))
    return out


def subspace_singular_values(U: np.ndarray, Y: np.ndarray,
                             horizon: int = 10) -> np.ndarray:
    """Singular-value spectrum of the input-projected output block-Hankel.

    These are exactly the singular values whose leading vectors the subspace
    warm start (`_warm_start`) keeps as the observability subspace. A gap in
    the spectrum marks the system order (number of latent variables): the
    singular values above the gap are state modes, those below are noise. With
    measurement noise the gap is usually *soft*, so it is a guide rather than a
    hard cut-off — corroborate it with the closed-loop performance-vs-n sweep.

    Parameters
    ----------
    U, Y : (T, m) / (T, p) probing inputs and recorded observations.
    horizon : number of block rows i; the spectrum can resolve up to i*p modes.
    """
    U = np.asarray(U, float); Y = np.asarray(Y, float)
    T, p = Y.shape
    m = U.shape[1]
    i = max(2, int(horizon))
    N = T - i + 1
    Yf = np.empty((i * p, N)); Uf = np.empty((i * m, N))
    for r in range(i):
        Yf[r * p:(r + 1) * p] = Y[r:r + N].T
        Uf[r * m:(r + 1) * m] = U[r:r + N].T
    coef = np.linalg.lstsq(Uf.T, Yf.T, rcond=None)[0]   # project inputs out
    Yf_perp = Yf - (Uf.T @ coef).T
    return np.linalg.svd(Yf_perp, compute_uv=False)


# =========================================================================== #
#  Estimator + controller design
# =========================================================================== #
def design_kalman_gain(model: LinearModel) -> np.ndarray:
    """Steady-state Kalman gain ``L`` (n x p) from the predictive DARE."""
    A, C, Q, R = model.A, model.C, model.Q, model.R
    P = solve_discrete_are(A.T, C.T, Q, R)             # predicted covariance
    S = C @ P @ C.T + R
    L = np.linalg.solve(S, C @ P).T                    # = P C^T S^{-1}
    return L


def design_lqr(model: LinearModel, Qx: np.ndarray, Ru: np.ndarray) -> np.ndarray:
    """Infinite-horizon LQR gain ``K`` (m x n): u = -K x minimises
    sum x^T Qx x + u^T Ru u."""
    A, B = model.A, model.B
    P = solve_discrete_are(A, B, Qx, Ru)
    K = np.linalg.solve(Ru + B.T @ P @ B, B.T @ P @ A)
    return K


def compute_setpoint(model: LinearModel, y_star: np.ndarray):
    """Feasible input/state setpoint for output target ``y_star``.

    Output reachable set is ``im(G)`` (rank <= m), so for the typical
    over-actuated-output case we take the least-squares input that gets the
    output as close to ``y_star`` as possible.

    Returns ``(u_ss, x_ss, y_ss)`` with ``y_ss`` the achievable output.
    """
    G = model.dc_gain()                       # p x m
    u_ss, *_ = np.linalg.lstsq(G, y_star, rcond=None)
    x_ss = np.linalg.solve(np.eye(model.n) - model.A, model.B @ u_ss)
    y_ss = model.C @ x_ss
    return u_ss, x_ss, y_ss


def design_lqi(
    model: LinearModel, Qx: np.ndarray, Qi: np.ndarray, Ru: np.ndarray,
    H: np.ndarray,
):
    """LQI / servo gain with integral action on the tracked output ``z = H y``.

    Augmented state ``[x; xi]`` with ``xi_{t+1} = xi_t + (H C x_t - z*)``.
    Returns ``(Kx, Ki)`` so that ``u = u_ss - Kx (x - x_ss) - Ki xi``.
    """
    A, B, C = model.A, model.B, model.C
    n, m = model.n, model.m
    q = H.shape[0]                            # number of tracked outputs
    HC = H @ C
    Aa = np.block([[A, np.zeros((n, q))],
                   [HC, np.eye(q)]])
    Ba = np.vstack([B, np.zeros((q, m))])
    Qa = np.block([[Qx, np.zeros((n, q))],
                   [np.zeros((q, n)), Qi]])
    Pa = solve_discrete_are(Aa, Ba, Qa, Ru)
    Ka = np.linalg.solve(Ru + Ba.T @ Pa @ Ba, Ba.T @ Pa @ Aa)
    return Ka[:, :n], Ka[:, n:]


# =========================================================================== #
#  Online controller
# =========================================================================== #
class Controller:
    """Closed-loop output-feedback controller.

    Modes
    -----
    ``'p'``   : model-free proportional output feedback (baseline).
    ``'lqg'`` : steady-state Kalman estimator + LQR regulation to the setpoint.
    ``'lqi'`` : LQG with integral action on the tracked output (zero
                steady-state error despite input clipping / model mismatch).

    Usage (matches the brain's measure-then-act loop)::

        ctrl.reset()
        for t in range(T):
            y = np.array(brain.measure())
            u = ctrl.step(y)
            brain.next_state(u)
    """

    def __init__(
        self,
        model: LinearModel,
        y_star: np.ndarray,
        mode: str = "lqg",
        *,
        u0: float = 0.5,
        q_state: float = 1.0,
        q_int: float = 1.0,
        r_input: float = 1.0,
        kp: float = 0.1,
        track: Optional[np.ndarray] = None,
        anti_windup: bool = True,
    ):
        self.model = model
        self.mode = mode
        self.y_star = np.asarray(y_star, float)
        self.u0 = u0
        self.anti_windup = anti_windup

        n, m, p = model.n, model.m, model.p

        # tracked-output selector H (defaults to the achievable output dirs).
        if track is None:
            # track the m left-singular directions of the DC gain — exactly the
            # output subspace the m inputs can actually steer.
            Ug, _, _ = np.linalg.svd(model.dc_gain(), full_matrices=False)
            H = Ug.T                          # (m x p)
        else:
            H = np.atleast_2d(track)
        self.H = H
        self.z_star = H @ self.y_star

        # setpoint
        self.u_ss, self.x_ss, self.y_ss = compute_setpoint(model, self.y_star)

        # estimator + control gains
        if mode in ("lqg", "lqi"):
            self.L = design_kalman_gain(model)
            Qx = q_state * np.eye(n)
            Ru = r_input * np.eye(m)
            if mode == "lqg":
                self.K = design_lqr(model, Qx, Ru)
            else:
                Qi = q_int * np.eye(H.shape[0])
                self.Kx, self.Ki = design_lqi(model, Qx, Qi, Ru, H)
        elif mode == "p":
            self.kp = kp
        else:
            raise ValueError(f"unknown mode {mode!r}")

        self.reset()

    # ----------------------------------------------------------------------- #
    def reset(self):
        n = self.model.n
        self.x_hat = self.x_ss.copy() if hasattr(self, "x_ss") else np.zeros(n)
        self.xi = np.zeros(self.H.shape[0])    # integral state
        self.u_prev = np.full(self.model.m, self.u0)
        self._initialised = False

    # ----------------------------------------------------------------------- #
    def step(self, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, float)

        if self.mode == "p":
            # proportional feedback: tracked-output error (dim m) maps directly
            # to the m inputs. H has m rows, so err already lives in input space.
            err = self.H @ (self.y_star - y)
            u = self.u0 + self.kp * err
            return self._finish(u)

        # ----- estimator update (steady-state Kalman, known input) --------- #
        A, B, C = self.model.A, self.model.B, self.model.C
        if not self._initialised:
            x_pred = self.x_hat                       # use setpoint prior at t=0
            self._initialised = True
        else:
            x_pred = A @ self.x_hat + B @ self.u_prev
        self.x_hat = x_pred + self.L @ (y - C @ x_pred)

        if self.mode == "lqg":
            dx = self.x_hat - self.x_ss
            u = self.u_ss - self.K @ dx
        else:  # lqi — pure servo: integrator finds the input bias itself, and
               # integrates the MEASURED output error so plant/model mismatch is
               # cancelled at DC (the point of integral action).
            u_unsat = -self.Kx @ self.x_hat - self.Ki @ self.xi
            u = u_unsat
            saturated = np.any((u_unsat < _INPUT_LO) | (u_unsat > _INPUT_HI))
            if not (self.anti_windup and saturated):
                self.xi = self.xi + (self.H @ y - self.z_star)
        return self._finish(u)

    # ----------------------------------------------------------------------- #
    def _finish(self, u: np.ndarray) -> np.ndarray:
        u = np.clip(u, _INPUT_LO, _INPUT_HI)
        self.u_prev = u
        return u


def probe_inputs(T: int, m: int, seed: int = 0, hold: int = 3,
                 lo: float = 0.0, hi: float = 1.0) -> np.ndarray:
    """Bounded piecewise-constant random probe in [lo, hi] (a PRBS-like signal
    that holds each level for ``hold`` steps to excite slow modes)."""
    rng = np.random.default_rng(seed)
    n_holds = int(np.ceil(T / hold))
    levels = rng.uniform(lo, hi, size=(n_holds, m))
    U = np.repeat(levels, hold, axis=0)[:T]
    return U


def run_open_loop(brain, U: np.ndarray) -> np.ndarray:
    """Apply input sequence ``U`` (T x m) and record observations ``Y`` (T x p).
    ``Y[t]`` is measured *before* applying ``U[t]`` — the convention the
    estimator and ``Controller`` assume."""
    T = U.shape[0]
    Y = []
    for t in range(T):
        Y.append(np.asarray(brain.measure(), float))
        brain.next_state(U[t])
    return np.asarray(Y)


def run_closed_loop(brain, controller: Controller, T: int) -> dict:
    """Run the controller in closed loop for ``T`` steps. Returns a log dict
    with ``Y`` (T x p), ``U`` (T x m), and the controller's ``Xhat`` (T x n)."""
    controller.reset()
    Y, U, Xhat = [], [], []
    for t in range(T):
        y = np.asarray(brain.measure(), float)
        u = controller.step(y)
        Y.append(y); U.append(u.copy()); Xhat.append(controller.x_hat.copy())
        brain.next_state(u)
    return {"Y": np.asarray(Y), "U": np.asarray(U), "Xhat": np.asarray(Xhat)}


# =========================================================================== #
#  Evaluation metrics
# =========================================================================== #
def tracked_output(log_or_Y, H: np.ndarray) -> np.ndarray:
    """Project observations onto the tracked-output subspace: ``z_t = H y_t``."""
    Y = log_or_Y["Y"] if isinstance(log_or_Y, dict) else log_or_Y
    return Y @ H.T


def evaluate(log: dict, controller: Controller, settle_frac: float = 0.25) -> dict:
    """Closed-loop performance summary.

    Metrics: steady-state tracked-output error, total/mean control effort,
    input saturation duty cycle, and observation variance over the last
    ``settle_frac`` of the run (a steadiness proxy).
    """
    Y, U = log["Y"], log["U"]
    T = Y.shape[0]
    tail = slice(int((1 - settle_frac) * T), T)

    z = Y @ controller.H.T
    z_err = z - controller.z_star
    ss_err = float(np.sqrt(np.mean(z_err[tail] ** 2)))

    effort_total = float(np.sum(np.linalg.norm(U, axis=1)))
    effort_mean = float(np.mean(np.linalg.norm(U, axis=1)))
    sat = float(np.mean((U <= _INPUT_LO + 1e-9) | (U >= _INPUT_HI - 1e-9)))
    tail_var = float(np.mean(np.var(Y[tail], axis=0)))

    return {
        "steady_state_error": ss_err,
        "control_effort_total": effort_total,
        "control_effort_mean": effort_mean,
        "saturation_duty": sat,
        "tail_output_variance": tail_var,
    }
