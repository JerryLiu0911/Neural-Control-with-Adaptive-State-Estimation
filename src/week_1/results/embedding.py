from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from numpy.typing import NDArray

from _helpers import _ensure_axes, _finalize


@dataclass(frozen=True, eq=False)
class EmbeddingResult:
    embedding: NDArray
    method: str
    feature_labels: list[str]
    point_labels: NDArray | None = None
    point_values: NDArray | None = None

    def plot(
        self,
        ax: Axes | None = None,
        cmap: str = "viridis",
        add_colorbar: bool = True,
        figsize: tuple[float, float] = (6, 5),
    ) -> Axes:
        _, ax = _ensure_axes(ax, figsize)
        sc = None
        if self.point_labels is not None:
            uniq = list(pd.unique(self.point_labels))
            for k, lbl in enumerate(uniq):
                mask = self.point_labels == lbl
                ax.scatter(
                    self.embedding[mask, 0],
                    self.embedding[mask, 1],
                    color=f"C{k % 10}",
                    s=20,
                    label=str(lbl),
                )
            legend = True
        else:
            c = (
                self.point_values
                if self.point_values is not None
                else np.arange(len(self.embedding))
            )
            sc = ax.scatter(
                self.embedding[:, 0], self.embedding[:, 1], c=c, cmap=cmap, s=20
            )
            legend = False
        if sc is not None and add_colorbar:
            ax.figure.colorbar(sc, ax=ax, label="Index")
        return _finalize(
            ax,
            xlabel=f"{self.method.upper()} 1",
            ylabel=f"{self.method.upper()} 2",
            title=f"{self.method.upper()} embedding",
            legend=legend,
        )
