from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from matplotlib.axes import Axes
from numpy.typing import NDArray

from _helpers import _ensure_axes, _finalize


@dataclass(frozen=True, eq=False)
class PCAResult:
    scores: NDArray
    components: NDArray
    explained_variance: NDArray
    explained_variance_ratio: NDArray
    centre: NDArray
    time: NDArray
    feature_labels: list[str]

    @property
    def n_components(self) -> int:
        return self.scores.shape[1]

    @property
    def participation_ratio(self) -> float:
        ev = self.explained_variance_ratio
        return float(np.sum(ev) ** 2 / np.sum(ev**2))

    def plot_timeseries(
        self,
        n_components: int = 3,
        ax: Axes | None = None,
        figsize: tuple[float, float] = (8, 4),
    ) -> Axes:
        n = min(n_components, self.n_components)
        _, ax = _ensure_axes(ax, figsize)
        for i in range(n):
            ax.plot(
                self.time,
                self.scores[:, i],
                lw=2,
                label=f"PC{i + 1} ({self.explained_variance_ratio[i] * 100:.1f}%)",
            )
        return _finalize(
            ax, xlabel="Time", ylabel="PC score", title="Principal components over time"
        )

    def plot_trajectory(
        self,
        components: tuple[int, int] = (0, 1),
        ax: Axes | None = None,
        colour_by: NDArray | None = None,
        cmap: str = "viridis",
        add_colorbar: bool = True,
        figsize: tuple[float, float] = (5, 4),
    ) -> Axes:
        i, j = components
        if max(i, j) >= self.n_components:
            raise IndexError(
                f"components {components} exceed n_components={self.n_components}."
            )
        c = self.time if colour_by is None else np.asarray(colour_by)
        cbar_label = "Time" if colour_by is None else "Value"

        _, ax = _ensure_axes(ax, figsize)
        ax.plot(self.scores[:, i], self.scores[:, j], color="grey", lw=0.8, alpha=0.5)
        sc = ax.scatter(
            self.scores[:, i], self.scores[:, j], c=c, cmap=cmap, s=20, zorder=3
        )
        if add_colorbar:
            ax.figure.colorbar(sc, ax=ax, label=cbar_label)
        return _finalize(
            ax,
            xlabel=f"PC{i + 1}",
            ylabel=f"PC{j + 1}",
            title="State-space trajectory",
            legend=False,
        )

    def plot_variance_explained(
        self,
        threshold: float | None = 90.0,
        ax: Axes | None = None,
        figsize: tuple[float, float] = (6, 4),
    ) -> Axes:
        ev = self.explained_variance_ratio
        ks = np.arange(1, len(ev) + 1)
        _, ax1 = _ensure_axes(ax, figsize)
        ax2 = ax1.twinx()
        ax1.bar(ks, ev * 100, color="C0", alpha=0.7, label="Individual")
        ax2.plot(
            ks,
            np.cumsum(ev) * 100,
            "o-",
            color="C1",
            lw=2,
            label="Cumulative",
        )
        if threshold is not None:
            ax2.axhline(
                threshold,
                color="grey",
                ls="--",
                lw=1,
                label=f"{threshold:.0f}% threshold",
            )
        ax1.set(
            xlabel="Principal component",
            ylabel="Variance explained (%)",
            title=f"PCA variance explained ($d_{{\\mathrm{{eff}}}}$ = "
            f"{self.participation_ratio:.2f})",
        )
        ax2.set_ylabel("Cumulative variance (%)")
        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1 + h2, l1 + l2, fontsize=8, loc="center right", frameon=False)
        return ax1

    def plot_loadings(
        self,
        n_components: int = 3,
        ax: Axes | None = None,
        figsize: tuple[float, float] = (8, 4),
    ) -> Axes:
        n = min(n_components, self.n_components)
        _, ax = _ensure_axes(ax, figsize)
        x = np.arange(len(self.feature_labels))
        width = 0.8 / n
        for i in range(n):
            ax.bar(
                x + i * width - 0.4 + width / 2,
                self.components[i],
                width=width,
                label=f"PC{i + 1}",
            )
        ax.set_xticks(x)
        ax.set_xticklabels(self.feature_labels, rotation=90, fontsize=7)
        ax.axhline(0, color="black", lw=0.5)
        return _finalize(ax, xlabel="Feature", ylabel="Loading", title="PCA loadings")
