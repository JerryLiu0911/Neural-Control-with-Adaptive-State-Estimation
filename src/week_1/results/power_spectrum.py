from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from matplotlib.axes import Axes
from numpy.typing import NDArray

from _helpers import IndexLike, _ensure_axes, _finalize, _resolve_indices


@dataclass(frozen=True, eq=False)
class PowerSpectrumResult:
    frequencies: NDArray
    psd: NDArray
    feature_labels: list[str]

    def peak_frequencies(self) -> dict[str, float]:
        return {
            lbl: float(self.frequencies[int(np.argmax(self.psd[k]))])
            for k, lbl in enumerate(self.feature_labels)
        }

    def plot(
        self,
        features: IndexLike = None,
        ax: Axes | None = None,
        figsize: tuple[float, float] = (10, 4),
    ) -> Axes:
        idx = _resolve_indices(
            features, len(self.feature_labels), self.feature_labels, "feature"
        )
        _, ax = _ensure_axes(ax, figsize)
        for fi in idx:
            ax.semilogy(
                self.frequencies,
                self.psd[fi],
                lw=2,
                label=self.feature_labels[fi],
            )
        return _finalize(
            ax,
            xlabel="Frequency",
            ylabel="PSD (log scale)",
            title="Power spectral density (Welch)",
        )
