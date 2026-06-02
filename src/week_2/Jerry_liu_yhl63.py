
from typing import Tuple

import numpy as np

# ---- frozen hyperparameters ----------------------------------------------- #
_N_ITER = 100          # EM iterations per restart (cap; convergence is earlier)
_N_RESTARTS = 4        # restarts from perturbed warm start; best likelihood kept
_LL_TOL = 1e-4         # relative log-likelihood convergence tolerance
_JITTER = 1e-6         # diagonal regularisation on covariance updates

def _warm_start(Y, n):
    '''Compute the warm-start initialisation for EM from l4b tools.
    
    Parameters
    ----------
    Y : (T, p) array of observations for ONE trial
    n : latent state dimension
    
    Returns
    -------
    C_init : (p, n) initial observation map
    A_init : (n, n) initial state transition matrix (diagonal)
    '''

    #In case the imports don't work : 
    T, p = Y.shape
    mu = Y.mean(0, keepdims=True)
    _, _, Vt = np.linalg.svd(Y - mu, full_matrices=False)
    C_init = Vt[:n].T
    Xpx = (Y - mu) @ C_init                                   # crude state proxy
    A_init = np.linalg.lstsq(Xpx[:-1], Xpx[1:], rcond=None)[0].T

    return C_init, A_init

def _kalman_filter(Y, A, C, Q, R, z0, P0):
    '''Kalman filter for the augmented (no-external-input) LG-SSM.
    
    Parameters
    ----------
    Y : (T, p) array of observations for ONE trial
    A : (n, n) state transition matrix
    C : (p, n) observation matrix
    Q : (n, n) process noise covariance
    R : (p, p) observation noise covariance
    z0 : (n,) initial state mean
    P0 : (n, n) initial state covariance
    
    Returns
    -------
    xf : (T, n) filtered state means
    Pf : (T, n, n) filtered state covariances
    xp : (T, n) predicted state means
    Pp : (T, n, n) predicted state covariances
    ll : marginal log-likelihood
    '''
    T, p = Y.shape
    na = A.shape[0] # 
    xf = np.zeros((T, na));    Pf = np.zeros((T, na, na))   # filtered (post-update)
    xp = np.zeros((T, na));    Pp = np.zeros((T, na, na))   # predicted (pre-update)
    I = np.eye(na)
    x_prev, P_prev = z0.copy(), P0.copy()
    ll = 0.0
    for t in range(T):
        # ----- predict ------------------------------------------------------ #
        if t == 0:
            x_pred, P_pred = z0.copy(), P0.copy()        # prior at t=0
        else:
            x_pred = A @ x_prev
            P_pred = A @ P_prev @ A.T + Q
        xp[t], Pp[t] = x_pred, P_pred

        # ----- update (innovation form) ------------------------------------- #
        S = C @ P_pred @ C.T + R                          # innovation covariance
        e = Y[t] - C @ x_pred                             # innovation
        K = np.linalg.solve(S, C @ P_pred).T              # Kalman gain via solve
        x_filt = x_pred + K @ e
        P_filt = (I - K @ C) @ P_pred
        P_filt = 0.5 * (P_filt + P_filt.T)                # symmetrise (numerics)
        xf[t], Pf[t] = x_filt, P_filt
        x_prev, P_prev = x_filt, P_filt

        # ----- accumulate marginal log-likelihood --------------------------- #
        _, logdet = np.linalg.slogdet(S)
        ll += -0.5 * (logdet + e @ np.linalg.solve(S, e) + p * np.log(2 * np.pi))
    return xf, Pf, xp, Pp, ll

def _rts_smoother(A, xf, Pf, xp, Pp):
    T, na = xf.shape
    xs = xf.copy(); Ps = Pf.copy()
    Plag = np.zeros((T, na, na))                          # Plag[t] = Cov(z_t, z_{t-1})
    for t in range(T - 2, -1, -1):
        # smoother gain (the analogue of the Kalman gain, running backwards)
        J = np.linalg.solve(Pp[t + 1], A @ Pf[t]).T       # = Pf[t] A^T Pp[t+1]^{-1}
        xs[t] = xf[t] + J @ (xs[t + 1] - xp[t + 1])
        Ps[t] = Pf[t] + J @ (Ps[t + 1] - Pp[t + 1]) @ J.T
        Plag[t + 1] = J @ Ps[t + 1]
    return xs, Ps, Plag


def _assemble(A, B, C, D, Q, R, F, Sigma_u):
    n, m = A.shape[0], B.shape[1]
    A_aug = np.block([[A, B], [np.zeros((m, n)), F]])     # [[A, B], [0, F]]
    C_aug = np.hstack([C, D])                             # [C, D]
    Q_aug = np.block([[Q, np.zeros((n, m))],
                      [np.zeros((m, n)), Sigma_u]])       # blkdiag(Q, Sigma_u)
    return A_aug, C_aug, Q_aug, R


def _fit_once(Y, n, m, F, Sigma_u, seed, C_init, A_init):
    T, p = Y.shape
    print(p, n, m)
    na = n + m
    rng = np.random.default_rng(seed)
    perturb = 0.05 * seed                                  # 0 for seed 0

    # ----- initialise ------------------------------------------------------- #
    C = C_init + perturb * rng.standard_normal((p, n))
    A = A_init + perturb * rng.standard_normal((n, n))
    D = 0.1 * rng.standard_normal((p, m))
    B = 0.1 * rng.standard_normal((n, m))
    Q = 0.1 * np.eye(n); R = 0.1 * np.eye(p)

    # diffuse prior on the input block of z0 (we know nothing about u_0)
    z0 = np.zeros(na); P0 = np.eye(na); P0[n:, n:] = Sigma_u * 10.0

    prev = -np.inf
    for _ in range(_N_ITER):
        # ----- E-step ------------------------------------------------------- #
        A_aug, C_aug, Q_aug, R_aug = _assemble(A, B, C, D, Q, R, F, Sigma_u)
        xf, Pf, xp, Pp, ll = _kalman_filter(Y, A_aug, C_aug, Q_aug, R_aug, z0, P0)
        xs, Ps, Plag = _rts_smoother(A_aug, xf, Pf, xp, Pp)

        # ----- accumulate sufficient statistics ----------------------------- #
        Szz = np.zeros((na, na)); Syz = np.zeros((p, na)); Syy = np.zeros((p, p))
        S11 = np.zeros((na, na)); S00 = np.zeros((na, na)); S10 = np.zeros((na, na))
        for t in range(T):
            Ezz = Ps[t] + np.outer(xs[t], xs[t])           # E[z_t z_t^T | Y]
            Szz += Ezz
            Syz += np.outer(Y[t], xs[t])
            Syy += np.outer(Y[t], Y[t])
            if t >= 1:
                S11 += Ezz
                S00 += Ps[t - 1] + np.outer(xs[t - 1], xs[t - 1])
                S10 += Plag[t] + np.outer(xs[t], xs[t - 1])
        Ndyn = T - 1

        # ----- M-step: regression updates for free blocks ------------------- #
        AB = np.linalg.solve(S00, S10.T).T
        A, B = AB[:n, :n], AB[:n, n:]

        CD = np.linalg.solve(Szz, Syz.T).T
        C, D = CD[:, :n], CD[:, n:]

        AB_top = np.hstack([A, B])
        Q = (S11[:n, :n] - AB_top @ S10[:n].T) / Ndyn
        Q = 0.5 * (Q + Q.T) + _JITTER * np.eye(n)

        R = (Syy - CD @ Syz.T) / T
        R = 0.5 * (R + R.T) + _JITTER * np.eye(p)

        if abs(ll - prev) < _LL_TOL * max(1.0, abs(prev)):
            break
        prev = ll
    return (A, B, C, D, Q, R), prev


def _identify(Y, n, m, F, Sigma_u):
    C_init, A_init = _warm_start(Y, n)
    best, best_ll = None, -np.inf
    for s in range(_N_RESTARTS):
        params, ll = _fit_once(Y, n, m, F, Sigma_u, seed=s,
                               C_init=C_init, A_init=A_init)
        if ll > best_ll:
            best, best_ll = params, ll
    return best


def _smooth(Y, params, n, m, F, Sigma_u):
    A, B, C, D, Q, R = params
    A_aug, C_aug, Q_aug, R_aug = _assemble(A, B, C, D, Q, R, F, Sigma_u)
    z0 = np.zeros(n + m); P0 = np.eye(n + m); P0[n:, n:] = Sigma_u * 10.0
    xf, Pf, xp, Pp, _ = _kalman_filter(Y, A_aug, C_aug, Q_aug, R_aug, z0, P0)
    zs, _, _ = _rts_smoother(A_aug, xf, Pf, xp, Pp)
    return zs[:, :n], zs[:, n:]


def estimate_latent_and_input(
    observation: np.ndarray, LatentDim: int, InputDim: int
) -> Tuple[np.ndarray, np.ndarray]:
    Y = np.asarray(observation, dtype=float)
    if Y.ndim != 2:
        raise ValueError(f"expected observation of shape (T, p); got {Y.shape}")
    T, p = Y.shape
    n, m = int(LatentDim), int(InputDim)

    F = np.eye(m)
    scale = float(np.var(np.diff(Y, axis=0))) if T > 1 else 1.0
    Sigma_u = np.eye(m) * max(scale, _JITTER)

    params = _identify(Y, n, m, F, Sigma_u)
    x_hat, u_hat = _smooth(Y, params, n, m, F, Sigma_u)

    assert x_hat.shape == (T, n) and u_hat.shape == (T, m)
    return x_hat, u_hat
