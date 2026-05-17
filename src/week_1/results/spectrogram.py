from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from matplotlib.axes import Axes
from numpy.typing import NDArray

from _helpers import _ensure_axes, _finalize, _resolve_indices


@dataclass(frozen=True, eq=False)
class SpectrogramResult:
    times: NDArray
    frequencies: NDArray
    power: NDArray
    feature_labels: list[str]

    def plot(
        self,
        feature: int | str,
        ax: Axes | None = None,
        log_scale: bool = True,
        add_colorbar: bool = True,
        figsize: tuple[float, float] = (10, 4),
    ) -> Axes:
        fi = _resolve_indices(
            feature, len(self.feature_labels), self.feature_labels, "feature"
        )[0]
        s = self.power[fi]
        s = np.log10(s + 1e-20) if log_scale else s
        _, ax = _ensure_axes(ax, figsize)
        im = ax.pcolormesh(
            self.times, self.frequencies, s, shading="auto", cmap="magma"
        )
        if add_colorbar:
            label = r"$\log_{10}$ Power" if log_scale else "Power"
            ax.figure.colorbar(im, ax=ax, label=label)
        return _finalize(
            ax,
            xlabel="Time",
            ylabel="Frequency",
            title=f"Spectrogram: {self.feature_labels[fi]}",
            legend=False,
        )
