import numpy as np
import matplotlib.pyplot as plt

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # points to /src
sys.path.insert(0, str(ROOT))

from l4b import illustrator as ill, simulator as sim
from kalman_filter import KalmanFilter

model = {
    "state_dim": 2,
    "input_dim": 2,
    "obs_dim": 2,

    "A": np.array([
        [0.9, 0.1],
        [0.2, 0.8]
    ]),

    "B": np.array([
        [0.1, 0.3],
        [0.3, 0.1]
    ]),

    "Q": np.array([
        [1.0, 0.0],
        [0.0, 1.0]
    ]),

    "C": np.array([
        [1.0, 0.0],
        [0.0, 1.0]
    ]),

    "R": np.array([
        [1.0, 0.0],
        [0.0, 1.0]
    ])
}

test = sim.Simulator(model=model)

T = 50
u = np.zeros((T, 2))
x0 = np.array([1.0, 1.0])

states, observations = test.run(
    initial_state=x0,
    control_inputs=u,
    time_steps=T,
    trials=2
)

# Use one trial for Kalman filter / RTS smoother
x_true = states[0]
y = observations[0]

kf = KalmanFilter(model)

kf_result = kf.run(
    observations=y,
    control_inputs=u,
    initial_state=x0
)

rts_result = kf.smooth_rts(
    observations=y,
    control_inputs=u,
    initial_state=x0
)

# Compare MSE
kf_mse = np.mean((x_true - kf_result.states) ** 2)
rts_mse = np.mean((x_true - rts_result.states) ** 2)

print("Kalman MSE:", kf_mse)
print("RTS MSE:", rts_mse)

# Plot both state dimensions
for dim in range(model["state_dim"]):
    plt.figure(figsize=(10, 4))

    plt.plot(x_true[:, dim], label="True state", linewidth=3)
    plt.plot(y[:, dim], label="Noisy observation", alpha=0.4)
    plt.plot(kf_result.states[:, dim], label="Kalman estimate", linewidth=2)
    plt.plot(rts_result.states[:, dim], label="RTS smoothed estimate", linewidth=2)

    plt.xlabel("Time")
    plt.ylabel(f"State {dim}")
    plt.title(f"State {dim}: Kalman vs RTS smoother")
    plt.legend()
    plt.tight_layout()
    plt.show()

illustrator = ill.Illustrator(observations)
illustrator.plot_all()
