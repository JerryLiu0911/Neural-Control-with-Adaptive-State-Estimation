from __future__ import annotations

from dataclasses import dataclass

from matplotlib.axes import Axes
from numpy.typing import NDArray

from _helpers import _ensure_axes, _finalize, _resolve_indices


@dataclass(frozen=True, eq=False)
class CoherenceResult:
    frequencies: NDArray
    coherence: NDArray
    feature_labels: list[str]

    def plot_pair(
        self,
        i: int | str,
        j: int | str,
        ax: Axes | None = None,
        figsize: tuple[float, float] = (8, 4),
    ) -> Axes:
        ii = _resolve_indices(
            i, len(self.feature_labels), self.feature_labels, "feature"
        )[0]
        jj = _resolve_indices(
            j, len(self.feature_labels), self.feature_labels, "feature"
        )[0]
        _, ax = _ensure_axes(ax, figsize)
        ax.plot(self.frequencies, self.coherence[ii, jj], lw=2)
        ax.set_ylim(0, 1)
        return _finalize(
            ax,
            xlabel="Frequency",
            ylabel="Coherence",
            title=f"Coherence: {self.feature_labels[ii]} vs {self.feature_labels[jj]}",
            legend=False,
        )
