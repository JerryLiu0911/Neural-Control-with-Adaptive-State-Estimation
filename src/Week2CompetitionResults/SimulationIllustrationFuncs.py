"""
SimulationIllustrationFuncs.py

Helpers for generating and illustrating the Week 2 competition test cases.

The model is a linear time-invariant state-space system

    x[t]   = A x[t-1] + B u[t-1] + process_noise[t]
    y[t]   = C x[t]              + observation_noise[t]

with the convention that the input applied at step ``t-1`` drives the state at
step ``t`` (strictly causal), and the very first state is just the process noise
injection ``x[0] = process_noise[0]``.

Two generators are provided:

* :func:`simulate`
    Forward-simulate the system given an explicit input sequence.

* :func:`simulate_from_latents`
    Solve for the minimum-energy input sequence that drives the *noiseless*
    latent trajectory through a set of waypoints (subject to a box constraint
    on the input), then forward-simulate with noise.

Plotting helpers:

* :func:`plot_input_latent_observation`
    Three stacked panels: input, latent states, observations.

* :func:`plot_input_latent_comparison_interactive`
    Overlay the ground-truth input / latents against every participant's
    estimate, with per-participant checkboxes to toggle visibility.

* :func:`fit_transform_latents`
    Least-squares linear alignment of one trajectory onto a reference, used to
    account for the linear identifiability of latent states.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Tuple

import numpy as np
import matplotlib.pyplot as plt


ArrayLike = Any


# --------------------------------------------------------------------------- #
# Core simulation
# --------------------------------------------------------------------------- #
def _forward_states(
    A: np.ndarray,
    B: np.ndarray,
    inputs: np.ndarray,
    process_noise: np.ndarray,
) -> np.ndarray:
    """
    Roll the state recursion forward.

        x[0] = process_noise[0]
        x[t] = A x[t-1] + B u[t-1] + process_noise[t]

    Parameters
    ----------
    A:
        State transition matrix, shape (n, n).
    B:
        Input matrix, shape (n, p).
    inputs:
        Input sequence, shape (T, p).
    process_noise:
        Per-step process noise, shape (T, n).

    Returns
    -------
    states:
        Latent trajectory, shape (T, n).
    """
    T = inputs.shape[0]
    n = A.shape[0]

    states = np.zeros((T, n))
    states[0] = process_noise[0]

    for t in range(1, T):
        states[t] = A @ states[t - 1] + B @ inputs[t - 1] + process_noise[t]

    return states


def simulate(
    A: ArrayLike,
    B: ArrayLike,
    C: ArrayLike,
    inputs: ArrayLike,
    x_noise: ArrayLike,
    y_noise: ArrayLike,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Forward-simulate the linear state-space system.

    Parameters
    ----------
    A:
        State transition matrix, shape (n, n).
    B:
        Input matrix, shape (n, p).
    C:
        Observation matrix, shape (m, n).
    inputs:
        Input sequence, shape (T, p).
    x_noise:
        Process noise, shape (T, n).
    y_noise:
        Observation noise, shape (T, m).

    Returns
    -------
    states:
        Latent trajectory, shape (T, n).
    observations:
        Observed trajectory, shape (T, m).
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    C = np.asarray(C, dtype=float)
    inputs = np.asarray(inputs, dtype=float)
    x_noise = np.asarray(x_noise, dtype=float)
    y_noise = np.asarray(y_noise, dtype=float)

    states = _forward_states(A, B, inputs, x_noise)
    observations = states @ C.T + y_noise

    return states, observations


# --------------------------------------------------------------------------- #
# Minimum-energy input that hits latent waypoints
# --------------------------------------------------------------------------- #
def _controllability_matrix(A: np.ndarray, B: np.ndarray, T: int) -> np.ndarray:
    """
    Build the block lower-triangular matrix ``G`` mapping a flattened input
    sequence ``u`` (length T*p) to the flattened *noiseless* state trajectory
    (length T*n), assuming a zero initial state.

        x[t] = sum_{k=0}^{t-1} A^{t-1-k} B u[k]

    Returns
    -------
    G:
        Array of shape (T*n, T*p).
    """
    n = A.shape[0]
    p = B.shape[1]

    # Precompute A^j up to A^{T-1}.
    powers = [np.eye(n)]
    for _ in range(T):
        powers.append(A @ powers[-1])

    G = np.zeros((T * n, T * p))
    for t in range(T):
        for k in range(t):
            G[t * n:(t + 1) * n, k * p:(k + 1) * p] = powers[t - 1 - k] @ B

    return G


def simulate_from_latents(
    A: ArrayLike,
    B: ArrayLike,
    C: ArrayLike,
    latents: Mapping[Tuple[int, int], float],
    x_noise: ArrayLike,
    y_noise: ArrayLike,
    maxu: float = 1.0,
    energy_reg: float = 1e-3,
    waypoint_weight: float = 1e3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Find the (approximately) minimum-energy input that drives the noiseless
    latent trajectory through the requested waypoints, then forward-simulate.

    The input is obtained by solving the bound-constrained least-squares problem

        minimise   energy_reg * ||u||^2  +  waypoint_weight * ||M u - b||^2
        subject to -maxu <= u <= maxu

    where each row of ``M`` selects one latent coordinate at one time step and
    ``b`` holds the corresponding target value. The soft waypoint penalty keeps
    the problem feasible even when the box constraint cannot hit a waypoint
    exactly.

    Parameters
    ----------
    A, B, C:
        System matrices, shapes (n, n), (n, p), (m, n).
    latents:
        Mapping ``(t, dim) -> target_value``. ``t`` is the time index and
        ``dim`` the latent coordinate that should reach ``target_value`` in the
        noiseless trajectory.
    x_noise:
        Process noise, shape (T, n). Its length defines the horizon T.
    y_noise:
        Observation noise, shape (T, m).
    maxu:
        Box bound on the input magnitude.
    energy_reg:
        Weight on input energy (kept small so the waypoints dominate).
    waypoint_weight:
        Weight on the waypoint-tracking penalty.

    Returns
    -------
    states:
        Noisy latent trajectory, shape (T, n).
    observations:
        Noisy observations, shape (T, m).
    inputs:
        Recovered input sequence, shape (T, p).
    noiseless_states:
        Deterministic latent trajectory under ``inputs`` with zero initial
        state and no noise, shape (T, n).
    """
    from scipy.optimize import lsq_linear

    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    C = np.asarray(C, dtype=float)
    x_noise = np.asarray(x_noise, dtype=float)
    y_noise = np.asarray(y_noise, dtype=float)

    T, n = x_noise.shape
    p = B.shape[1]

    G = _controllability_matrix(A, B, T)

    # Assemble the waypoint constraints  M u = b.
    rows = []
    targets = []
    for (t, dim), value in latents.items():
        if not (0 <= t < T):
            raise ValueError(f"Waypoint time {t} out of range [0, {T}).")
        if not (0 <= dim < n):
            raise ValueError(f"Waypoint latent dim {dim} out of range [0, {n}).")
        rows.append(G[t * n + dim])
        targets.append(value)

    if rows:
        M = np.vstack(rows)
        b = np.asarray(targets, dtype=float)
    else:
        M = np.zeros((0, T * p))
        b = np.zeros(0)

    # Stack the energy regulariser and the (weighted) waypoint penalty.
    design = np.vstack([
        np.sqrt(energy_reg) * np.eye(T * p),
        np.sqrt(waypoint_weight) * M,
    ])
    rhs = np.concatenate([
        np.zeros(T * p),
        np.sqrt(waypoint_weight) * b,
    ])

    solution = lsq_linear(design, rhs, bounds=(-maxu, maxu))
    u_flat = solution.x

    inputs = u_flat.reshape(T, p)
    noiseless_states = (G @ u_flat).reshape(T, n)

    states = _forward_states(A, B, inputs, x_noise)
    observations = states @ C.T + y_noise

    return states, observations, inputs, noiseless_states


# --------------------------------------------------------------------------- #
# Alignment (linear identifiability of latents)
# --------------------------------------------------------------------------- #
def fit_transform_latents(reference: ArrayLike, source: ArrayLike) -> np.ndarray:
    """
    Linearly align ``source`` onto ``reference`` by least squares.

    Latent states are only identifiable up to a linear transform, so before
    overlaying an estimate on the ground truth we find the matrix ``W`` that
    minimises ``||source @ W - reference||`` and return ``source @ W``. Handles a
    mismatch between the number of columns of ``source`` and ``reference``.

    Parameters
    ----------
    reference:
        Target trajectory, shape (T, k).
    source:
        Trajectory to align, shape (T, j).

    Returns
    -------
    aligned:
        ``source`` mapped into the reference space, shape (T, k).
    """
    reference = np.atleast_2d(np.asarray(reference, dtype=float))
    source = np.atleast_2d(np.asarray(source, dtype=float))

    if reference.shape[0] == 1 and reference.shape[1] != source.shape[0]:
        reference = reference.T
    if source.shape[0] == 1 and source.shape[1] != reference.shape[0]:
        source = source.T

    W, *_ = np.linalg.lstsq(source, reference, rcond=None)
    return source @ W


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #
def plot_input_latent_observation(
    inputs: ArrayLike,
    states: ArrayLike,
    observations: ArrayLike,
):
    """
    Three stacked panels showing the input, latent states and observations.

    Returns
    -------
    fig, axes:
        The created figure and its three axes.
    """
    inputs = np.atleast_2d(np.asarray(inputs, dtype=float))
    states = np.atleast_2d(np.asarray(states, dtype=float))
    observations = np.atleast_2d(np.asarray(observations, dtype=float))

    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)

    for d in range(inputs.shape[1]):
        axes[0].plot(inputs[:, d], label=f"u[{d}]")
    axes[0].set_title("Input")
    axes[0].set_ylabel("u")
    axes[0].grid(True, alpha=0.3)
    if inputs.shape[1] > 1:
        axes[0].legend(fontsize=8, ncol=2)

    for d in range(states.shape[1]):
        axes[1].plot(states[:, d], label=f"x[{d}]")
    axes[1].set_title("Latent states")
    axes[1].set_ylabel("x")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=8, ncol=2)

    for d in range(observations.shape[1]):
        axes[2].plot(observations[:, d], lw=0.9, alpha=0.7)
    axes[2].set_title(f"Observations ({observations.shape[1]} channels)")
    axes[2].set_xlabel("time step")
    axes[2].set_ylabel("y")
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    return fig, axes


def _as_result_dict(value: Any) -> Dict[str, np.ndarray]:
    """
    Normalise a single participant's stored result to a plain dict.

    Accepts either ``{"latent_states":..., "inputs":...}`` (live results) or a
    0-d object array wrapping such a dict (as returned when loading an ``.npz``).
    """
    if isinstance(value, np.ndarray) and value.dtype == object and value.shape == ():
        value = value.item()
    return value


def plot_input_latent_comparison_interactive(
    inputs: ArrayLike,
    states: ArrayLike,
    results: Mapping[str, Any],
    align_inputs: bool = False,
    align_latents: bool = True,
    active: Mapping[str, bool] = None,
):
    """
    Interactive overlay of the ground truth against every participant estimate.

    A column of checkboxes (one per participant) toggles the visibility of that
    participant's traces. The ground-truth traces are always shown in black.

    Parameters
    ----------
    inputs:
        Ground-truth input, shape (T, p).
    states:
        Ground-truth latent states, shape (T, n).
    results:
        Mapping ``name -> {"latent_states": ..., "inputs": ...}``. Values may
        also be 0-d object arrays (as loaded from ``.npz``).
    align_inputs:
        If True, linearly align each participant's input onto the true input
        before plotting (accounts for input sign/scale ambiguity).
    align_latents:
        If True, linearly align each participant's latents onto the true latent
        states before plotting (accounts for linear identifiability).
    active:
        Optional mapping ``name -> bool`` of initial checkbox states. If omitted
        or empty, every participant starts visible.

    Notes
    -----
    Uses the ``inline`` backend together with :func:`ipywidgets.interactive_output`,
    which manages the redraw on every toggle. It relies only on core ipywidgets and
    deliberately avoids the ``ipympl`` ("widget") canvas, whose frontend frequently
    fails to load in VS Code. Use ``%matplotlib inline`` in the notebook.
    """
    import ipywidgets as widgets
    from IPython.display import display

    inputs = np.atleast_2d(np.asarray(inputs, dtype=float))
    states = np.atleast_2d(np.asarray(states, dtype=float))
    results = {name: _as_result_dict(v) for name, v in results.items()}

    names = list(results)
    if active:
        visible0 = {name: bool(active.get(name, False)) for name in names}
    else:
        visible0 = {name: True for name in names}

    T = states.shape[0]
    n_latent = states.shape[1]
    n_input = inputs.shape[1]

    cmap = plt.get_cmap("tab20")
    colours = {name: cmap(i % 20) for i, name in enumerate(names)}

    # Pre-align each participant once (alignment is independent of visibility).
    prepared: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for name in names:
        res = results[name]
        u = np.atleast_2d(np.asarray(res["inputs"], dtype=float))
        x = np.atleast_2d(np.asarray(res["latent_states"], dtype=float))
        if align_inputs and u.shape[0] == T:
            u = fit_transform_latents(inputs, u)
        if align_latents and x.shape[0] == T:
            x = fit_transform_latents(states, x)
        prepared[name] = (u, x)

    checkboxes = {
        name: widgets.Checkbox(value=visible0[name], description=name, indent=False)
        for name in names
    }

    def _draw(**selected):
        n_rows = n_input + n_latent
        fig, axes = plt.subplots(
            n_rows, 1, figsize=(12, 2.2 * n_rows), sharex=True, squeeze=False,
        )
        axes = axes[:, 0]

        # Ground truth.
        for d in range(n_input):
            axes[d].plot(inputs[:, d], color="black", lw=2.5, label="true", zorder=5)
            axes[d].set_ylabel(f"u[{d}]")
            axes[d].set_title("Input" if d == 0 else "")
        for d in range(n_latent):
            ax = axes[n_input + d]
            ax.plot(states[:, d], color="black", lw=2.5, label="true", zorder=5)
            ax.set_ylabel(f"x[{d}]")
            ax.set_title("Latent states" if d == 0 else "")

        # Selected participants overlaid.
        for name in names:
            if not selected.get(name, False):
                continue
            u, x = prepared[name]
            for d in range(min(n_input, u.shape[1])):
                axes[d].plot(u[:, d], color=colours[name], lw=1.3, alpha=0.85, label=name)
            for d in range(min(n_latent, x.shape[1])):
                axes[n_input + d].plot(
                    x[:, d], color=colours[name], lw=1.3, alpha=0.85, label=name,
                )

        for ax in axes:
            ax.grid(True, alpha=0.3)
        axes[0].legend(fontsize=7, ncol=4, loc="upper right")
        axes[-1].set_xlabel("time step")
        fig.tight_layout()
        plt.show()

    # interactive_output handles the (re)draw on initial display and every toggle.
    out = widgets.interactive_output(_draw, checkboxes)

    def _set_all(value):
        for cb in checkboxes.values():
            cb.value = value

    show_all = widgets.Button(description="show all")
    hide_all = widgets.Button(description="hide all")
    show_all.on_click(lambda _btn: _set_all(True))
    hide_all.on_click(lambda _btn: _set_all(False))

    # Checkboxes wrap horizontally so the plot can use the full width below them.
    checkbox_grid = widgets.Box(
        [checkboxes[name] for name in names],
        layout=widgets.Layout(flex_flow="row wrap", display="flex"),
    )
    controls = widgets.VBox([widgets.HBox([show_all, hide_all]), checkbox_grid])
    display(widgets.VBox([controls, out]))
    return None
