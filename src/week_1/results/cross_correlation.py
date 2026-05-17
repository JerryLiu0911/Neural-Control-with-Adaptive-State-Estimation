from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from matplotlib.axes import Axes
from numpy.typing import NDArray

from _helpers import _ensure_axes, _finalize, _resolve_indices


@dataclass(frozen=True, eq=False)
class CrossCorrelationResult:
    lags: NDArray
    values: NDArray
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
        ax.plot(self.lags, self.values[ii, jj], lw=2)
        ax.axvline(0, color="black", lw=0.5, ls="--")
        ax.axhline(0, color="black", lw=0.5, ls="--")
        return _finalize(
            ax,
            xlabel="Lag",
            ylabel="Cross-correlation",
            title=f"{self.feature_labels[ii]} $\\rightarrow$ {self.feature_labels[jj]}",
            legend=False,
        )

    def plot_matrix(
        self,
        lag: float = 0.0,
        ax: Axes | None = None,
        add_colorbar: bool = True,
        figsize: tuple[float, float] = (6, 5),
    ) -> Axes:
        k = int(np.argmin(np.abs(self.lags - lag)))
        mat = self.values[:, :, k]
        _, ax = _ensure_axes(ax, figsize)
        im = ax.imshow(mat, cmap="RdBu_r", vmin=-1, vmax=1)
        n = len(self.feature_labels)
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(self.feature_labels, rotation=90, fontsize=7)
        ax.set_yticklabels(self.feature_labels, fontsize=7)
        if add_colorbar:
            ax.figure.colorbar(im, ax=ax, label="Cross-correlation")
        return _finalize(
            ax, title=f"Cross-correlation at lag = {self.lags[k]:.2f}", legend=False
        )
