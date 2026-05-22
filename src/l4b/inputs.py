from __future__ import annotations

from typing import Optional

import numpy as np
from numpy.random import Generator
from numpy.typing import NDArray


def _resolve_channels(m: int, channels: Optional[list[int]]) -> list[int]:
    """Return selected channels, checking they are valid."""
    if channels is None:
        return list(range(m))

    for ch in channels:
        if ch < 0 or ch >= m:
            raise ValueError(f"channel {ch} out of range [0, {m})")

    return channels


class InputSignal:
    """Immutable control input signal with numpy array protocol support."""

    def __init__(
        self,
        T: int,
        m: int,
        array: NDArray,
        pattern_name: Optional[str] = None,
        dtype: type = np.float64,
    ) -> None:
        array = np.asarray(array, dtype=dtype)

        if array.shape != (T, m):
            raise ValueError(
                f"Array shape {array.shape} does not match expected shape ({T}, {m})"
            )

        self._signal = array.copy()
        self.T = T
        self.m = m
        self.pattern_name = pattern_name
        self.dtype = dtype

    def __array__(self, dtype=None) -> NDArray:
        arr = self._signal.copy()
        if dtype is not None:
            arr = arr.astype(dtype)
        return arr

    @property
    def shape(self) -> tuple[int, int]:
        return self._signal.shape

    def __getitem__(self, key):
        return self._signal[key]

    def __repr__(self) -> str:
        pattern = f", pattern='{self.pattern_name}'" if self.pattern_name else ""
        return f"InputSignal(T={self.T}, m={self.m}{pattern})"

    def to_array(self) -> NDArray:
        return self._signal.copy()

    def plot(self, ax=None, title: Optional[str] = None):
        """Plot every input channel."""
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(figsize=(10, 3))

        for ch in range(self.m):
            ax.plot(self._signal[:, ch], label=f"input {ch}")

        ax.set_xlabel("Timestep")
        ax.set_ylabel("u(t)")
        ax.set_title(title or self.pattern_name or "InputSignal")
        ax.legend(frameon=False)
        return ax

    @classmethod
    def from_array(
        cls,
        array: NDArray,
        pattern_name: str = "custom",
        dtype: type = np.float64,
    ) -> InputSignal:
        array = np.asarray(array, dtype=dtype)

        if array.ndim != 2:
            raise ValueError(f"array must be 2-D with shape (T, m), got {array.ndim}-D")

        T, m = array.shape
        return cls(T, m, array, pattern_name=pattern_name, dtype=dtype)

    @classmethod
    def zero(cls, T: int, m: int, dtype: type = np.float64) -> InputSignal:
        return cls(T, m, np.zeros((T, m), dtype=dtype), pattern_name="zero", dtype=dtype)

    @classmethod
    def random_gaussian(
        cls,
        T: int,
        m: int,
        scale: float = 1.0,
        rng: Optional[Generator] = None,
        channels: Optional[list[int]] = None,
        dtype: type = np.float64,
    ) -> InputSignal:
        rng = rng or np.random.default_rng()
        cols = _resolve_channels(m, channels)

        array = np.zeros((T, m), dtype=dtype)
        array[:, cols] = rng.normal(0.0, scale, (T, len(cols))).astype(dtype)

        return cls(T, m, array, pattern_name="random_gaussian", dtype=dtype)

    @classmethod
    def pulse(
        cls,
        T: int,
        m: int,
        onset: int,
        duration: int = 1,
        channels: Optional[list[int]] = None,
        amplitude: float = 1.0,
        dtype: type = np.float64,
    ) -> InputSignal:
        if onset < 0 or onset >= T:
            raise ValueError(f"onset={onset} must be in range [0, {T})")
        if duration < 1:
            raise ValueError(f"duration={duration} must be at least 1")
        if onset + duration > T:
            raise ValueError(
                f"Pulse extends beyond T: onset={onset}, duration={duration}, T={T}"
            )

        cols = _resolve_channels(m, channels)
        array = np.zeros((T, m), dtype=dtype)
        array[onset : onset + duration, cols] = amplitude

        return cls(T, m, array, pattern_name="pulse", dtype=dtype)

    @classmethod
    def impulse(
        cls,
        T: int,
        m: int,
        onset: int,
        channels: Optional[list[int]] = None,
        amplitude: float = 1.0,
        dtype: type = np.float64,
    ) -> InputSignal:
        return cls.pulse(
            T=T,
            m=m,
            onset=onset,
            duration=1,
            channels=channels,
            amplitude=amplitude,
            dtype=dtype,
        )

    @classmethod
    def step(
        cls,
        T: int,
        m: int,
        onset: int = 0,
        channels: Optional[list[int]] = None,
        amplitude: float = 1.0,
        dtype: type = np.float64,
    ) -> InputSignal:
        if onset < 0 or onset >= T:
            raise ValueError(f"onset={onset} must be in range [0, {T})")

        cols = _resolve_channels(m, channels)
        array = np.zeros((T, m), dtype=dtype)
        array[onset:, cols] = amplitude

        return cls(T, m, array, pattern_name="step", dtype=dtype)

    @classmethod
    def pulse_train(
        cls,
        T: int,
        m: int,
        onset: int,
        period: int,
        duration: int = 1,
        channels: Optional[list[int]] = None,
        amplitude: float = 1.0,
        dtype: type = np.float64,
    ) -> InputSignal:
        if onset < 0 or onset >= T:
            raise ValueError(f"onset={onset} must be in range [0, {T})")
        if period < 1:
            raise ValueError(f"period={period} must be at least 1")
        if duration < 1:
            raise ValueError(f"duration={duration} must be at least 1")
        if duration > period:
            raise ValueError("duration should not exceed period for a pulse train")

        cols = _resolve_channels(m, channels)
        array = np.zeros((T, m), dtype=dtype)

        for start in range(onset, T, period):
            end = min(start + duration, T)
            array[start:end, cols] = amplitude

        return cls(T, m, array, pattern_name="pulse_train", dtype=dtype)

    @classmethod
    def oscillatory(
        cls,
        T: int,
        m: int,
        frequency: float,
        amplitude: float = 1.0,
        phase: float = 0.0,
        channels: Optional[list[int]] = None,
        dtype: type = np.float64,
    ) -> InputSignal:
        cols = _resolve_channels(m, channels)

        t = np.arange(T, dtype=dtype)
        signal = amplitude * np.sin(2 * np.pi * frequency * t + phase)

        array = np.zeros((T, m), dtype=dtype)
        array[:, cols] = signal[:, np.newaxis]

        return cls(T, m, array, pattern_name="oscillatory", dtype=dtype)

    @classmethod
    def chirp(
        cls,
        T: int,
        m: int,
        f0: float,
        f1: float,
        amplitude: float = 1.0,
        phase: float = 0.0,
        channels: Optional[list[int]] = None,
        dtype: type = np.float64,
    ) -> InputSignal:
        cols = _resolve_channels(m, channels)

        freqs = np.linspace(f0, f1, T, dtype=dtype)
        phase_t = 2 * np.pi * np.cumsum(freqs) + phase
        signal = amplitude * np.sin(phase_t)

        array = np.zeros((T, m), dtype=dtype)
        array[:, cols] = signal[:, np.newaxis]

        return cls(T, m, array, pattern_name="chirp", dtype=dtype)

    @classmethod
    def single_channel(
        cls,
        T: int,
        m: int,
        channel: int,
        pattern: NDArray,
        dtype: type = np.float64,
    ) -> InputSignal:
        _resolve_channels(m, [channel])

        pattern = np.asarray(pattern, dtype=dtype)
        if pattern.shape != (T,):
            raise ValueError(f"pattern must have shape ({T},), got {pattern.shape}")

        array = np.zeros((T, m), dtype=dtype)
        array[:, channel] = pattern

        return cls(T, m, array, pattern_name="single_channel", dtype=dtype)

    @classmethod
    def ramp(
        cls,
        T: int,
        m: int,
        channel: int,
        start: float = 0.0,
        end: float = 1.0,
        dtype: type = np.float64,
    ) -> InputSignal:
        _resolve_channels(m, [channel])

        array = np.zeros((T, m), dtype=dtype)
        array[:, channel] = np.linspace(start, end, T, dtype=dtype)

        return cls(T, m, array, pattern_name="ramp", dtype=dtype)


class InputBuilder:
    """Fluent interface for composing multiple input patterns."""

    def __init__(self, T: int, m: int, dtype: type = np.float64) -> None:
        self.T = T
        self.m = m
        self.dtype = dtype
        self._signal = np.zeros((T, m), dtype=dtype)

    def _add(self, signal: InputSignal) -> InputBuilder:
        """Add an InputSignal into the builder."""
        if signal.shape != (self.T, self.m):
            raise ValueError(
                f"signal shape {signal.shape} does not match builder shape {(self.T, self.m)}"
            )

        self._signal += signal.to_array()
        return self

    def zero(self) -> InputBuilder:
        return self

    def random_gaussian(
        self,
        channels: Optional[list[int]] = None,
        scale: float = 1.0,
        rng: Optional[Generator] = None,
    ) -> InputBuilder:
        return self._add(
            InputSignal.random_gaussian(
                T=self.T,
                m=self.m,
                scale=scale,
                rng=rng,
                channels=channels,
                dtype=self.dtype,
            )
        )

    def pulse(
        self,
        onset: int,
        duration: int = 1,
        channels: Optional[list[int]] = None,
        amplitude: float = 1.0,
    ) -> InputBuilder:
        return self._add(
            InputSignal.pulse(
                T=self.T,
                m=self.m,
                onset=onset,
                duration=duration,
                channels=channels,
                amplitude=amplitude,
                dtype=self.dtype,
            )
        )

    def impulse(
        self,
        onset: int,
        channels: Optional[list[int]] = None,
        amplitude: float = 1.0,
    ) -> InputBuilder:
        return self._add(
            InputSignal.impulse(
                T=self.T,
                m=self.m,
                onset=onset,
                channels=channels,
                amplitude=amplitude,
                dtype=self.dtype,
            )
        )

    def step(
        self,
        onset: int = 0,
        channels: Optional[list[int]] = None,
        amplitude: float = 1.0,
    ) -> InputBuilder:
        return self._add(
            InputSignal.step(
                T=self.T,
                m=self.m,
                onset=onset,
                channels=channels,
                amplitude=amplitude,
                dtype=self.dtype,
            )
        )

    def pulse_train(
        self,
        onset: int,
        period: int,
        duration: int = 1,
        channels: Optional[list[int]] = None,
        amplitude: float = 1.0,
    ) -> InputBuilder:
        return self._add(
            InputSignal.pulse_train(
                T=self.T,
                m=self.m,
                onset=onset,
                period=period,
                duration=duration,
                channels=channels,
                amplitude=amplitude,
                dtype=self.dtype,
            )
        )

    def oscillatory(
        self,
        frequency: float,
        channels: Optional[list[int]] = None,
        amplitude: float = 1.0,
        phase: float = 0.0,
    ) -> InputBuilder:
        return self._add(
            InputSignal.oscillatory(
                T=self.T,
                m=self.m,
                frequency=frequency,
                amplitude=amplitude,
                phase=phase,
                channels=channels,
                dtype=self.dtype,
            )
        )

    def chirp(
        self,
        f0: float,
        f1: float,
        amplitude: float = 1.0,
        phase: float = 0.0,
        channels: Optional[list[int]] = None,
    ) -> InputBuilder:
        return self._add(
            InputSignal.chirp(
                T=self.T,
                m=self.m,
                f0=f0,
                f1=f1,
                amplitude=amplitude,
                phase=phase,
                channels=channels,
                dtype=self.dtype,
            )
        )

    def single_channel(
        self,
        channel: int,
        pattern: NDArray,
    ) -> InputBuilder:
        return self._add(
            InputSignal.single_channel(
                T=self.T,
                m=self.m,
                channel=channel,
                pattern=pattern,
                dtype=self.dtype,
            )
        )

    def ramp(
        self,
        channel: int,
        start: float = 0.0,
        end: float = 1.0,
    ) -> InputBuilder:
        return self._add(
            InputSignal.ramp(
                T=self.T,
                m=self.m,
                channel=channel,
                start=start,
                end=end,
                dtype=self.dtype,
            )
        )

    def build(self) -> InputSignal:
        return InputSignal(
            T=self.T,
            m=self.m,
            array=self._signal.copy(),
            pattern_name="composed",
            dtype=self.dtype,
        )