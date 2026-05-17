from __future__ import annotations

from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray

IndexLike = int | str | Sequence[int | str] | slice | np.ndarray | None


def _ensure_axes(
    ax: Axes | None,
    figsize: tuple[float, float],
) -> tuple[Figure, Axes]:
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
        assert isinstance(fig, Figure)
    return fig, ax


def _finalize(
    ax: Axes,
    *,
    xlabel: str | None = None,
    ylabel: str | None = None,
    title: str | None = None,
    legend: bool = True,
    legend_kwargs: dict[str, Any] | None = None,
) -> Axes:
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)
    if legend:
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            kw: dict[str, Any] = {
                "fontsize": 8,
                "ncol": min(4, len(handles)),
                "loc": "best",
                "frameon": False,
            }
            if legend_kwargs:
                kw.update(legend_kwargs)
            ax.legend(**kw)
    return ax


def _resolve_indices(
    idx: IndexLike,
    n: int,
    labels: Sequence[str] | None = None,
    name: str = "item",
) -> list[int]:
    if idx is None:
        return list(range(n))
    if isinstance(idx, (int, np.integer)):
        out = [int(idx)]
    elif isinstance(idx, str):
        if labels is None or idx not in labels:
            raise KeyError(f"label {idx!r} not found among {name}s.")
        return [list(labels).index(idx)]
    elif isinstance(idx, slice):
        return list(range(*idx.indices(n)))
    elif isinstance(idx, np.ndarray) and idx.dtype == bool:
        if len(idx) != n:
            raise ValueError(f"boolean mask length {len(idx)} != {n}.")
        return np.flatnonzero(idx).tolist()
    else:
        out = []
        for item in idx:
            if isinstance(item, (int, np.integer)):
                out.append(int(item))
            elif isinstance(item, str):
                if labels is None or item not in labels:
                    raise KeyError(f"label {item!r} not found among {name}s.")
                out.append(list(labels).index(item))
            else:
                raise TypeError(
                    f"unsupported index type for {name}: {type(item).__name__}"
                )

    for i in out:
        if not 0 <= i < n:
            raise IndexError(f"{name} index {i} out of range [0, {n}).")
    return out


def _signal_noise_decomp(data: NDArray) -> tuple[NDArray, NDArray]:
    trial_mean = data.mean(axis=0)
    residuals = data - trial_mean[np.newaxis, :, :]
    return trial_mean, residuals
