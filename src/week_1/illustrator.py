from __future__ import annotations

import warnings
from functools import cached_property
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from _helpers import (
    IndexLike,
    _ensure_axes,
    _finalize,
    _resolve_indices,
    _signal_noise_decomp,
)
from matplotlib.axes import Axes
from matplotlib.colors import TwoSlopeNorm
from matplotlib.figure import Figure
from numpy.typing import ArrayLike, NDArray
from results import (
    AutocorrelationResult,
    CoherenceResult,
    CrossCorrelationResult,
    EmbeddingResult,
    PCAResult,
    PowerSpectrumResult,
    SpectrogramResult,
)
from scipy.ndimage import gaussian_filter1d, uniform_filter1d
from scipy.signal import butter, filtfilt, spectrogram, welch
from scipy.signal import coherence as _coherence
from scipy.stats import kurtosis, skew
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_regression
from sklearn.manifold import TSNE


class Illustrator:
    def __init__(
        self,
        data: ArrayLike,
        dt: float = 1.0,
        feature_labels: Sequence[str] | None = None,
        trial_labels: ArrayLike | None = None,
    ) -> None:
        arr = np.asarray(data, dtype=float)
        if arr.ndim == 2:
            arr = arr[np.newaxis, :, :]
        elif arr.ndim != 3:
            raise ValueError(
                f"data must be 2-D (T, N) or 3-D (R, T, N); got {arr.ndim}-D."
            )
        if dt <= 0:
            raise ValueError(f"dt must be strictly positive; got {dt}.")
        if not np.all(np.isfinite(arr)):
            n_bad = int(np.sum(~np.isfinite(arr)))
            warnings.warn(
                f"data contains {n_bad} non-finite values; downstream "
                f"analyses may produce NaNs.",
                RuntimeWarning,
                stacklevel=2,
            )

        self._data: NDArray[np.floating] = arr
        self.dt: float = float(dt)
        self.n_trials, self.n_time, self.n_features = arr.shape
        self.time: NDArray[np.floating] = np.arange(self.n_time) * dt

        self.feature_labels: list[str] = self._validate_labels(
            feature_labels, self.n_features, "feature"
        )

        if trial_labels is None:
            self.trial_labels: NDArray | None = None
        else:
            tl = np.asarray(trial_labels)
            if tl.shape != (self.n_trials,):
                raise ValueError(
                    f"trial_labels shape {tl.shape} does not match "
                    f"n_trials={self.n_trials}."
                )
            self.trial_labels = tl

    def __repr__(self) -> str:
        cond = (
            f", conditions={len(self.conditions)}"
            if self.trial_labels is not None
            else ""
        )
        return (
            f"Illustrator(trials={self.n_trials}, time={self.n_time}, "
            f"features={self.n_features}, dt={self.dt}{cond})"
        )

    @staticmethod
    def _validate_labels(labels: Sequence[str] | None, n: int, name: str) -> list[str]:
        if labels is None:
            return [f"{name.capitalize()} {i}" for i in range(n)]
        if len(labels) != n:
            raise ValueError(f"len({name}_labels)={len(labels)} does not match n={n}.")
        if len(set(labels)) != n:
            warnings.warn(f"duplicate {name} labels detected.", stacklevel=3)
        return [str(s) for s in labels]

    @property
    def data(self) -> NDArray[np.floating]:
        return self._data

    @property
    def shape(self) -> tuple[int, int, int]:
        return self._data.shape

    @cached_property
    def trial_mean(self) -> NDArray[np.floating]:
        return self._data.mean(axis=0)

    @cached_property
    def trial_sem(self) -> NDArray[np.floating]:
        if self.n_trials < 2:
            return np.zeros((self.n_time, self.n_features))
        return self._data.std(axis=0, ddof=1) / np.sqrt(self.n_trials)

    @cached_property
    def population_mean(self) -> NDArray[np.floating]:
        return self._data.mean(axis=2)

    @cached_property
    def conditions(self) -> list[Any]:
        if self.trial_labels is None:
            return []
        return list(pd.unique(self.trial_labels))

    def _trial_idx_for(self, condition: Any) -> NDArray:
        if self.trial_labels is None:
            raise ValueError("no trial labels were provided.")
        return np.where(self.trial_labels == condition)[0]

    def select(
        self,
        trials: IndexLike = None,
        features: IndexLike = None,
        conditions: Sequence[Any] | None = None,
    ) -> Illustrator:
        trial_idx = _resolve_indices(trials, self.n_trials, name="trial")
        if conditions is not None:
            if self.trial_labels is None:
                raise ValueError("conditions requested but no trial labels.")
            cond_mask = np.isin(self.trial_labels, list(conditions))
            cond_idx = np.flatnonzero(cond_mask).tolist()
            trial_idx = sorted(set(trial_idx) & set(cond_idx))
        feat_idx = _resolve_indices(
            features, self.n_features, self.feature_labels, "feature"
        )
        new_data = self._data[np.ix_(trial_idx, np.arange(self.n_time), feat_idx)]
        new_feature_labels = [self.feature_labels[i] for i in feat_idx]
        new_trial_labels = (
            self.trial_labels[trial_idx] if self.trial_labels is not None else None
        )
        return type(self)(
            new_data,
            dt=self.dt,
            feature_labels=new_feature_labels,
            trial_labels=new_trial_labels,
        )

    def smooth(
        self,
        window: float = 5.0,
        kind: str = "gaussian",
    ) -> Illustrator:
        if kind == "gaussian":
            sigma = window / 2.355
            smoothed = gaussian_filter1d(self._data, sigma=sigma, axis=1)
        elif kind == "boxcar":
            smoothed = uniform_filter1d(self._data, size=max(1, int(window)), axis=1)
        else:
            raise ValueError(f"unknown kind {kind!r}; choose 'gaussian' or 'boxcar'.")
        return type(self)(
            smoothed,
            dt=self.dt,
            feature_labels=self.feature_labels,
            trial_labels=self.trial_labels,
        )

    def bandpass(
        self,
        low: float | None = None,
        high: float | None = None,
        order: int = 4,
    ) -> Illustrator:
        fs = 1.0 / self.dt
        nyq = fs / 2
        if low is None and high is None:
            raise ValueError("specify at least one of low or high cutoff.")
        if low is not None and high is not None:
            b, a = butter(order, [low / nyq, high / nyq], btype="band")
        elif high is not None:
            b, a = butter(order, high / nyq, btype="low")
        else:
            assert low is not None
            b, a = butter(order, low / nyq, btype="high")
        filtered = filtfilt(b, a, self._data, axis=1)
        return type(self)(
            filtered,
            dt=self.dt,
            feature_labels=self.feature_labels,
            trial_labels=self.trial_labels,
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        np.savez(
            path,
            data=self._data,
            dt=np.array(self.dt),
            feature_labels=np.array(self.feature_labels, dtype=object),
            trial_labels=(
                np.array(self.trial_labels, dtype=object)
                if self.trial_labels is not None
                else np.array([], dtype=object)
            ),
        )

    @classmethod
    def load(cls, path: str | Path) -> Illustrator:
        f = np.load(path, allow_pickle=True)
        tl = f["trial_labels"]
        return cls(
            data=f["data"],
            dt=float(f["dt"]),
            feature_labels=list(f["feature_labels"]),
            trial_labels=tl if tl.size > 0 else None,
        )

    def summary(self) -> pd.DataFrame:
        flat = self._data.reshape(-1, self.n_features)
        n_samp = flat.shape[0]
        mean = flat.mean(axis=0)
        std = flat.std(axis=0, ddof=1) if n_samp > 1 else flat.std(axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            snr = np.where(std > 0, np.abs(mean) / std, np.nan)
            cv = np.where(np.abs(mean) > 0, std / np.abs(mean), np.nan)
        df = pd.DataFrame(
            {
                "mean": mean,
                "std": std,
                "min": flat.min(axis=0),
                "q25": np.percentile(flat, 25, axis=0),
                "median": np.median(flat, axis=0),
                "q75": np.percentile(flat, 75, axis=0),
                "max": flat.max(axis=0),
                "skew": skew(flat, axis=0),
                "kurtosis": kurtosis(flat, axis=0),
                "snr": snr,
                "cv": cv,
            },
            index=pd.Index(self.feature_labels, name="feature"),
        )
        if self.n_trials > 1:
            df["across_trial_std"] = self._data.mean(axis=1).std(axis=0, ddof=1)
        return df

    def summary_by_condition(self) -> pd.DataFrame:
        if self.trial_labels is None:
            raise ValueError("no trial labels were provided.")
        frames = []
        for cond in self.conditions:
            sub = self.select(conditions=[cond])
            s = sub.summary().assign(condition=cond)
            frames.append(s)
        out = pd.concat(frames).reset_index()
        return out.set_index(["condition", "feature"])

    def correlation_matrix(
        self,
        trial: int | None = None,
        kind: str = "pooled",
    ) -> pd.DataFrame:
        if kind == "pooled":
            mat = (
                self._data.reshape(-1, self.n_features)
                if trial is None
                else self._data[trial]
            )
            corr = np.corrcoef(mat.T)
        elif kind == "signal":
            signal, _ = _signal_noise_decomp(self._data)
            corr = np.corrcoef(signal.T)
        elif kind == "noise":
            _, noise = _signal_noise_decomp(self._data)
            corr = np.corrcoef(noise.reshape(-1, self.n_features).T)
        else:
            raise ValueError(
                f"unknown kind {kind!r}; choose 'pooled', 'signal', or 'noise'."
            )
        return pd.DataFrame(
            corr, index=self.feature_labels, columns=self.feature_labels
        )

    def signal_noise_variance(self) -> pd.DataFrame:
        if self.n_trials < 2:
            raise ValueError("need at least 2 trials to estimate signal/noise.")
        signal, noise = _signal_noise_decomp(self._data)
        sig_var = signal.var(axis=0, ddof=1)
        noise_var = noise.reshape(-1, self.n_features).var(axis=0, ddof=1)
        total = sig_var + noise_var
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(total > 0, sig_var / total, np.nan)
        return pd.DataFrame(
            {
                "signal_variance": sig_var,
                "noise_variance": noise_var,
                "signal_ratio": ratio,
            },
            index=pd.Index(self.feature_labels, name="feature"),
        )

    def mutual_information(self, n_neighbors: int = 3) -> pd.DataFrame:
        flat = self._data.reshape(-1, self.n_features)
        mi = np.zeros((self.n_features, self.n_features))
        for i in range(self.n_features):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                mi[i] = mutual_info_regression(
                    flat,
                    flat[:, i],
                    discrete_features=False,
                    n_neighbors=n_neighbors,
                )
        mi = 0.5 * (mi + mi.T)
        return pd.DataFrame(mi, index=self.feature_labels, columns=self.feature_labels)

    def to_long_dataframe(self) -> pd.DataFrame:
        r_idx, t_idx, f_idx = np.indices(self._data.shape)
        df = pd.DataFrame(
            {
                "trial": r_idx.ravel(),
                "time": self.time[t_idx.ravel()],
                "feature": np.array(self.feature_labels)[f_idx.ravel()],
                "value": self._data.ravel(),
            }
        )
        if self.trial_labels is not None:
            df["condition"] = np.asarray(self.trial_labels)[r_idx.ravel()]
        return df

    def pca(self, n_components: int | None = None) -> PCAResult:
        n_max = min(self.n_features, self.n_time)
        n = n_max if n_components is None else min(n_components, n_max)
        centred = self.trial_mean - self.trial_mean.mean(axis=0, keepdims=True)
        pca = PCA(n_components=n)
        scores = pca.fit_transform(centred)
        return PCAResult(
            scores=scores,
            components=pca.components_,
            explained_variance=pca.explained_variance_,
            explained_variance_ratio=pca.explained_variance_ratio_,
            centre=self.trial_mean.mean(axis=0),
            time=self.time,
            feature_labels=list(self.feature_labels),
        )

    def autocorrelation(
        self,
        max_lag: int | None = None,
    ) -> AutocorrelationResult:
        max_lag = self.n_time // 2 if max_lag is None else max_lag
        lags = np.arange(0, max_lag + 1) * self.dt
        values = np.empty((self.n_features, max_lag + 1))
        for ni in range(self.n_features):
            acfs = np.empty((self.n_trials, max_lag + 1))
            for t in range(self.n_trials):
                x = self._data[t, :, ni] - self._data[t, :, ni].mean()
                ac = np.correlate(x, x, mode="full")[self.n_time - 1 :]
                denom = ac[0] if ac[0] != 0 else 1.0
                acfs[t] = ac[: max_lag + 1] / denom
            values[ni] = acfs.mean(axis=0)
        return AutocorrelationResult(
            lags=lags,
            values=values,
            feature_labels=list(self.feature_labels),
            n_effective=self.n_time * self.n_trials,
        )

    def power_spectrum(
        self,
        nperseg: int | None = None,
    ) -> PowerSpectrumResult:
        fs = 1.0 / self.dt
        nperseg = min(32, self.n_time) if nperseg is None else nperseg
        freqs, _ = welch(self._data[0, :, 0], fs=fs, nperseg=nperseg)
        psd = np.zeros((self.n_features, len(freqs)))
        for t in range(self.n_trials):
            for ni in range(self.n_features):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    _, p = welch(self._data[t, :, ni], fs=fs, nperseg=nperseg)
                psd[ni] += p
        psd /= self.n_trials
        return PowerSpectrumResult(
            frequencies=freqs,
            psd=psd,
            feature_labels=list(self.feature_labels),
        )

    def cross_correlation(
        self,
        max_lag: int | None = None,
    ) -> CrossCorrelationResult:
        max_lag = self.n_time // 4 if max_lag is None else max_lag
        lags = np.arange(-max_lag, max_lag + 1) * self.dt
        n = self.n_features
        xcorr = np.zeros((n, n, 2 * max_lag + 1))
        for t in range(self.n_trials):
            centred = self._data[t] - self._data[t].mean(axis=0, keepdims=True)
            norms = np.sqrt((centred**2).sum(axis=0))
            for i in range(n):
                for j in range(n):
                    if norms[i] == 0 or norms[j] == 0:
                        continue
                    c = np.correlate(centred[:, i], centred[:, j], mode="full")
                    centre = len(c) // 2
                    xcorr[i, j] += c[centre - max_lag : centre + max_lag + 1] / (
                        norms[i] * norms[j]
                    )
        xcorr /= self.n_trials
        return CrossCorrelationResult(
            lags=lags,
            values=xcorr,
            feature_labels=list(self.feature_labels),
        )

    def coherence(
        self,
        nperseg: int | None = None,
    ) -> CoherenceResult:
        fs = 1.0 / self.dt
        nperseg = min(64, self.n_time) if nperseg is None else nperseg
        freqs, _ = _coherence(
            self._data[0, :, 0], self._data[0, :, 0], fs=fs, nperseg=nperseg
        )
        n = self.n_features
        coh = np.zeros((n, n, len(freqs)))
        for t in range(self.n_trials):
            for i in range(n):
                for j in range(i, n):
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        _, cxy = _coherence(
                            self._data[t, :, i],
                            self._data[t, :, j],
                            fs=fs,
                            nperseg=nperseg,
                        )
                    coh[i, j] += cxy
                    if i != j:
                        coh[j, i] += cxy
        coh /= self.n_trials
        return CoherenceResult(
            frequencies=freqs,
            coherence=coh,
            feature_labels=list(self.feature_labels),
        )

    def spectrogram(
        self,
        nperseg: int | None = None,
    ) -> SpectrogramResult:
        fs = 1.0 / self.dt
        nperseg = min(32, self.n_time // 4) if nperseg is None else nperseg
        freqs, times, _ = spectrogram(self._data[0, :, 0], fs=fs, nperseg=nperseg)
        power = np.zeros((self.n_features, len(freqs), len(times)))
        for t in range(self.n_trials):
            for ni in range(self.n_features):
                _, _, sxx = spectrogram(self._data[t, :, ni], fs=fs, nperseg=nperseg)
                power[ni] += sxx
        power /= self.n_trials
        return SpectrogramResult(
            times=times,
            frequencies=freqs,
            power=power,
            feature_labels=list(self.feature_labels),
        )

    def embed(
        self,
        method: str = "tsne",
        source: str = "trial_mean",
        n_components: int = 2,
        **kwargs: Any,
    ) -> EmbeddingResult:
        if source == "trial_mean":
            X = self.trial_mean
            labels = None
        elif source == "all_trials":
            X = self._data.reshape(-1, self.n_features)
            labels = (
                np.repeat(self.trial_labels, self.n_time)
                if self.trial_labels is not None
                else None
            )
        else:
            raise ValueError(
                f"unknown source {source!r}; choose 'trial_mean' or 'all_trials'."
            )

        if method == "tsne":
            kwargs.setdefault("init", "pca")
            kwargs.setdefault("learning_rate", "auto")
            kwargs.setdefault(
                "perplexity",
                min(30.0, max(5.0, X.shape[0] / 4)),
            )
            emb = TSNE(n_components=n_components, **kwargs).fit_transform(X)
        elif method == "umap":
            try:
                import umap
            except ImportError as e:
                raise ImportError("UMAP requires the `umap-learn` package.") from e
            emb = umap.UMAP(n_components=n_components, **kwargs).fit_transform(X)
        else:
            raise ValueError(f"unknown method {method!r}; choose 'tsne' or 'umap'.")

        return EmbeddingResult(
            embedding=emb,
            method=method,
            feature_labels=list(self.feature_labels),
            point_labels=labels,
        )

    def plot_traces(
        self,
        trial: int = 0,
        features: IndexLike = None,
        ax: Axes | None = None,
        show_population_mean: bool = False,
        show_population_std: bool = False,
        figsize: tuple[float, float] = (10, 4),
    ) -> Axes:
        feats = _resolve_indices(
            features, self.n_features, self.feature_labels, "feature"
        )
        if not 0 <= trial < self.n_trials:
            raise IndexError(f"trial {trial} out of range [0, {self.n_trials}).")
        _, ax = _ensure_axes(ax, figsize)
        for fi in feats:
            ax.plot(
                self.time,
                self._data[trial, :, fi],
                lw=1.5,
                label=self.feature_labels[fi],
            )
        if show_population_mean:
            ax.plot(
                self.time,
                self.population_mean[trial],
                color="black",
                lw=2.5,
                label="Population mean",
            )
        if show_population_std:
            ax.plot(
                self.time,
                self._data[trial, :, feats].std(axis=1),
                color="red",
                lw=2,
                label="Population std",
            )
        return _finalize(ax, xlabel="Time", ylabel="Activity", title=f"Trial {trial}")

    def plot_trial_average(
        self,
        features: IndexLike = None,
        ax: Axes | None = None,
        show_sem: bool = True,
        figsize: tuple[float, float] = (10, 4),
    ) -> Axes:
        feats = _resolve_indices(
            features, self.n_features, self.feature_labels, "feature"
        )
        _, ax = _ensure_axes(ax, figsize)
        for fi in feats:
            mean = self.trial_mean[:, fi]
            (line,) = ax.plot(self.time, mean, lw=2, label=self.feature_labels[fi])
            if show_sem and self.n_trials > 1:
                sem = self.trial_sem[:, fi]
                ax.fill_between(
                    self.time,
                    mean - sem,
                    mean + sem,
                    color=line.get_color(),
                    alpha=0.3,
                    linewidth=0,
                )
        sem_text = r" $\pm$ SEM" if show_sem and self.n_trials > 1 else ""
        return _finalize(
            ax,
            xlabel="Time",
            ylabel="Activity",
            title=f"Trial-averaged activity{sem_text}",
        )

    def plot_condition_average(
        self,
        feature: int | str,
        conditions: Sequence[Any] | None = None,
        ax: Axes | None = None,
        show_sem: bool = True,
        figsize: tuple[float, float] = (10, 4),
    ) -> Axes:
        if self.trial_labels is None:
            raise ValueError("no trial labels were provided.")
        fi = _resolve_indices(feature, self.n_features, self.feature_labels, "feature")[
            0
        ]
        conds = list(self.conditions) if conditions is None else list(conditions)
        _, ax = _ensure_axes(ax, figsize)
        for cond in conds:
            idx = self._trial_idx_for(cond)
            if len(idx) == 0:
                continue
            sub = self._data[idx, :, fi]
            mean = sub.mean(axis=0)
            (line,) = ax.plot(self.time, mean, lw=2, label=str(cond))
            if show_sem and len(idx) > 1:
                sem = sub.std(axis=0, ddof=1) / np.sqrt(len(idx))
                ax.fill_between(
                    self.time,
                    mean - sem,
                    mean + sem,
                    color=line.get_color(),
                    alpha=0.3,
                    linewidth=0,
                )
        return _finalize(
            ax,
            xlabel="Time",
            ylabel="Activity",
            title=f"{self.feature_labels[fi]} by condition",
        )

    def plot_heatmap(
        self,
        trial: int | None = None,
        ax: Axes | None = None,
        cmap: str = "RdBu_r",
        centre_zero: bool = True,
        add_colorbar: bool = True,
        figsize: tuple[float, float] = (10, 5),
    ) -> Axes:
        if trial is None:
            mat = self.trial_mean.T
            title = "Trial-averaged activity"
        else:
            if not 0 <= trial < self.n_trials:
                raise IndexError(f"trial {trial} out of range [0, {self.n_trials}).")
            mat = self._data[trial].T
            title = f"Trial {trial}"
        _, ax = _ensure_axes(ax, figsize)
        norm = None
        if centre_zero and np.any(mat):
            vmax = float(np.abs(mat).max())
            norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
        im = ax.imshow(
            mat,
            aspect="auto",
            cmap=cmap,
            norm=norm,
            extent=(self.time[0], self.time[-1], self.n_features - 0.5, -0.5),
        )
        ax.set_yticks(range(self.n_features))
        ax.set_yticklabels(self.feature_labels, fontsize=7)
        if add_colorbar:
            ax.figure.colorbar(im, ax=ax, label="Activity")
        return _finalize(ax, xlabel="Time", ylabel="Feature", title=title, legend=False)

    def plot_correlation(
        self,
        kind: str = "pooled",
        ax: Axes | None = None,
        add_colorbar: bool = True,
        figsize: tuple[float, float] = (6, 5),
    ) -> Axes:
        corr = self.correlation_matrix(kind=kind).to_numpy()
        _, ax = _ensure_axes(ax, figsize)
        im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(self.n_features))
        ax.set_yticks(range(self.n_features))
        ax.set_xticklabels(self.feature_labels, rotation=90, fontsize=7)
        ax.set_yticklabels(self.feature_labels, fontsize=7)
        if add_colorbar:
            ax.figure.colorbar(im, ax=ax, label="Pearson $r$")
        return _finalize(ax, title=f"{kind.capitalize()} correlation", legend=False)

    def plot_distribution(
        self,
        features: IndexLike = None,
        ax: Axes | None = None,
        bins: int = 40,
        figsize: tuple[float, float] = (8, 4),
    ) -> Axes:
        feats = _resolve_indices(
            features, self.n_features, self.feature_labels, "feature"
        )
        _, ax = _ensure_axes(ax, figsize)
        flat = self._data.reshape(-1, self.n_features)
        for fi in feats:
            ax.hist(
                flat[:, fi],
                bins=bins,
                alpha=0.5,
                label=self.feature_labels[fi],
                density=True,
            )
        return _finalize(
            ax, xlabel="Value", ylabel="Density", title="Per-feature distributions"
        )

    def plot_overview(
        self,
        figsize: tuple[float, float] = (15, 11),
    ) -> Figure:
        fig, axes = plt.subplots(3, 2, figsize=figsize)
        self.plot_trial_average(ax=axes[0, 0])
        self.plot_heatmap(ax=axes[0, 1])
        self.plot_correlation(ax=axes[1, 0])
        self.pca().plot_variance_explained(ax=axes[1, 1])
        self.autocorrelation().plot(ax=axes[2, 0])
        self.power_spectrum().plot(ax=axes[2, 1])
        fig.tight_layout()
        return fig
