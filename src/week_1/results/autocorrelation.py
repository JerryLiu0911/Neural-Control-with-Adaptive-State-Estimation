from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from matplotlib.axes import Axes
from numpy.typing import NDArray

from _helpers import IndexLike, _ensure_axes, _finalize, _resolve_indices


@dataclass(frozen=True, eq=False)
class AutocorrelationResult:
    lags: NDArray
    values: NDArray
    feature_labels: list[str]
    n_effective: int

    @property
    def significance_band(self) -> float:
        return 1.96 / np.sqrt(self.n_effective)

    def peak_lags(self, exclude_zero: bool = True) -> dict[str, float]:
        out: dict[str, float] = {}
        for k, lbl in enumerate(self.feature_labels):
            v = self.values[k]
            start = 1 if exclude_zero else 0
            if len(v) > start:
                i = int(np.argmax(v[start:])) + start
                out[lbl] = float(self.lags[i])
            else:
                out[lbl] = float("nan")
        return out

    def plot(
        self,
        features: IndexLike = None,
        ax: Axes | None = None,
        show_significance: bool = True,
        figsize: tuple[float, float] = (10, 4),
    ) -> Axes:
        idx = _resolve_indices(
            features, len(self.feature_labels), self.feature_labels, "feature"
        )
        _, ax = _ensure_axes(ax, figsize)
        for fi in idx:
            ax.plot(
                self.lags,
                self.values[fi],
                lw=2,
                label=self.feature_labels[fi],
            )
        ax.axhline(0, color="black", lw=0.8, ls="--")
        if show_significance:
            band = self.significance_band
            ax.axhspan(
                -band, band, alpha=0.15, color="grey", label="95% CI (white noise)"
            )
        return _finalize(
            ax, xlabel="Lag", ylabel="Autocorrelation", title="Autocorrelation function"
        )
