from __future__ import annotations

from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import solve_discrete_are


class KalmanResult(NamedTuple):
    states: NDArray        # (T, state_dim)  posterior state estimates x̂_{t|t}
    covariances: NDArray   # (T, state_dim, state_dim)  posterior covariances P_{t|t}
    gains: NDArray         # (T, state_dim, obs_dim)  Kalman gains K_t
    innovations: NDArray   # (T, obs_dim)  y_t - C x̂_{t|t-1}


class KalmanFilter:
    """Discrete-time Kalman filter for the linear Gaussian state-space model

        x_{t+1} = A x_t + B u_t + w_t,   w_t ~ N(0, Q)
        y_t     = C x_t + v_t,            v_t ~ N(0, R)

    Parameters
    ----------
    model : dict
        Same format as ``Simulator``: must contain ``A``, ``B``, ``C``, ``Q``,
        ``R``, ``state_dim``, ``input_dim``, ``obs_dim``.
    """

    def __init__(self, model: dict) -> None:
        self.state_dim: int = model["state_dim"]
        self.input_dim: int = model["input_dim"]
        self.obs_dim: int = model["obs_dim"]

        self.A: NDArray = np.asarray(model["A"])
        self.B: NDArray = np.asarray(model["B"])
        self.C: NDArray = np.asarray(model["C"])
        self.Q: NDArray = np.asarray(model["Q"])
        self.R: NDArray = np.asarray(model["R"])

    # ------------------------------------------------------------------
    # Time-varying Kalman filter
    # ------------------------------------------------------------------

    def run(
        self,
        observations: NDArray,
        control_inputs: NDArray,
        initial_state: NDArray,
        initial_covariance: NDArray | None = None,
    ) -> KalmanResult:
        """Run the time-varying Kalman filter over a single trial.

        The filter treats *initial_state* as the prior mean x̂_{0|-1} (before
        seeing any observations).  ``initial_covariance`` defaults to Q if not
        provided, which is a reasonable uninformative prior.

        :param observations: Array of shape ``(T, obs_dim)``.
        :param control_inputs: Array of shape ``(T, input_dim)``.  The input
            ``u[t]`` is used in the predict step *after* updating with ``y[t]``,
            so it drives the prediction for time ``t+1``.
        :param initial_state: Prior mean, shape ``(state_dim,)``.
        :param initial_covariance: Prior covariance, shape
            ``(state_dim, state_dim)``.  Defaults to ``Q``.
        :returns: :class:`KalmanResult` named tuple.
        """
        T = observations.shape[0]
        n, m = self.state_dim, self.obs_dim

        if initial_covariance is None:
            initial_covariance = self.Q.copy()

        states = np.zeros((T, n))
        covariances = np.zeros((T, n, n))
        gains = np.zeros((T, n, m))
        innovations = np.zeros((T, m))

        I = np.eye(n)
        x = initial_state.copy()
        P = initial_covariance.copy()

        for t in range(T):
            # --- Update step (incorporate y_t) ---
            S = self.C @ P @ self.C.T + self.R          # innovation covariance
            K = P @ self.C.T @ np.linalg.inv(S)         # Kalman gain
            inn = observations[t] - self.C @ x           # innovation
            x = x + K @ inn                              # posterior mean
            P = (I - K @ self.C) @ P                    # posterior covariance (Joseph form below for symmetry)
            P = (P + P.T) / 2

            states[t] = x
            covariances[t] = P
            gains[t] = K
            innovations[t] = inn

            # --- Predict step (propagate to t+1) ---
            if t < T - 1:
                x = self.A @ x + self.B @ control_inputs[t]
                P = self.A @ P @ self.A.T + self.Q
                P = (P + P.T) / 2

        return KalmanResult(states, covariances, gains, innovations)

    # ------------------------------------------------------------------
    # Steady-state Kalman filter (constant gain from DARE)
    # ------------------------------------------------------------------

    def steady_state_gain(self) -> tuple[NDArray, NDArray]:
        """Compute the steady-state Kalman gain via the discrete algebraic
        Riccati equation (DARE).

        Solves for the steady-state *prior* covariance P∞ satisfying:

            P∞ = A P∞ A^T + Q - A P∞ C^T (C P∞ C^T + R)^{-1} C P∞ A^T

        then derives the gain and the corresponding *posterior* covariance.

        :returns: ``(K_inf, P_post)`` where ``K_inf`` has shape
            ``(state_dim, obs_dim)`` and ``P_post`` has shape
            ``(state_dim, state_dim)``.
        :raises numpy.linalg.LinAlgError: if the DARE has no solution (system
            not detectable/stabilisable).
        """
        # solve_discrete_are solves: A^T P A - P - (A^T P B)(R + B^T P B)^{-1}(B^T P A) + Q = 0
        # For the Kalman filter formulation we swap roles: state ← A^T, input ← C^T, noise ← Q, R
        P_prior = solve_discrete_are(self.A.T, self.C.T, self.Q, self.R)
        S = self.C @ P_prior @ self.C.T + self.R
        K_inf = P_prior @ self.C.T @ np.linalg.inv(S)
        P_post = (np.eye(self.state_dim) - K_inf @ self.C) @ P_prior
        P_post = (P_post + P_post.T) / 2
        return K_inf, P_post

    def run_steady_state(
        self,
        observations: NDArray,
        control_inputs: NDArray,
        initial_state: NDArray,
    ) -> KalmanResult:
        """Run the Kalman filter with a fixed (steady-state) gain K∞.

        Faster than :meth:`run` because the covariance propagation is skipped
        after the first step.  The returned ``covariances`` array is filled with
        the constant posterior covariance P_post everywhere.

        :param observations: Array of shape ``(T, obs_dim)``.
        :param control_inputs: Array of shape ``(T, input_dim)``.
        :param initial_state: Prior mean, shape ``(state_dim,)``.
        :returns: :class:`KalmanResult` named tuple.
        """
        T = observations.shape[0]
        n, m = self.state_dim, self.obs_dim

        K_inf, P_post = self.steady_state_gain()

        states = np.zeros((T, n))
        covariances = np.full((T, n, n), P_post)
        gains = np.full((T, n, m), K_inf)
        innovations = np.zeros((T, m))

        x = initial_state.copy()
        I_KC = np.eye(n) - K_inf @ self.C

        for t in range(T):
            inn = observations[t] - self.C @ x
            x = x + K_inf @ inn

            states[t] = x
            innovations[t] = inn

            if t < T - 1:
                x = self.A @ x + self.B @ control_inputs[t]

        return KalmanResult(states, covariances, gains, innovations)
    
    def smooth_rts(
        self,
        observations: NDArray,
        control_inputs: NDArray,
        initial_state: NDArray,
        initial_covariance: NDArray | None = None,
    ) -> KalmanResult:
        """
        Run Kalman filter forward, then RTS smoother backward.

        Filter gives x_hat[t|t]: estimate using observations up to time t.
        Smoother gives x_hat[t|T]: estimate using all observations 0...T-1.
        """
        T = observations.shape[0]
        n, m = self.state_dim, self.obs_dim

        if initial_covariance is None:
            initial_covariance = self.Q.copy()

        # Storage for forward pass
        x_filt = np.zeros((T, n))
        P_filt = np.zeros((T, n, n))
        x_pred = np.zeros((T, n))
        P_pred = np.zeros((T, n, n))
        gains = np.zeros((T, n, m))
        innovations = np.zeros((T, m))

        I = np.eye(n)
        x = initial_state.copy()
        P = initial_covariance.copy()

        # forward Kalman filter
        for t in range(T):
            x_pred[t] = x
            P_pred[t] = P

            S = self.C @ P @ self.C.T + self.R
            K = P @ self.C.T @ np.linalg.inv(S)
            inn = observations[t] - self.C @ x

            x = x + K @ inn
            P = (I - K @ self.C) @ P
            P = (P + P.T) / 2

            x_filt[t] = x
            P_filt[t] = P
            gains[t] = K
            innovations[t] = inn

            if t < T - 1:
                x = self.A @ x + self.B @ control_inputs[t]
                P = self.A @ P @ self.A.T + self.Q
                P = (P + P.T) / 2

        # Backward RTS smoother
        x_smooth = x_filt.copy()
        P_smooth = P_filt.copy()

        for t in range(T - 2, -1, -1):
            # Smoother gain
            J = P_filt[t] @ self.A.T @ np.linalg.inv(P_pred[t + 1])

            # Correct filtered estimate using future-smoothed estimate
            x_smooth[t] = x_filt[t] + J @ (
                x_smooth[t + 1] - x_pred[t + 1]
            )

            P_smooth[t] = P_filt[t] + J @ (
                P_smooth[t + 1] - P_pred[t + 1]
            ) @ J.T
            P_smooth[t] = (P_smooth[t] + P_smooth[t].T) / 2

        return KalmanResult(x_smooth, P_smooth, gains, innovations)
    
