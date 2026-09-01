"""Shared vectorised helpers - geo maths, ID minting, timestamps, logging."""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager

import numpy as np
import pandas as pd

EARTH_RADIUS_KM = 6371.0088

__all__ = [
    "get_logger", "timed", "haversine_km", "jitter_coords", "make_ids",
    "random_timestamps", "bearing_deg", "offset_coords", "clip_series",
]


def get_logger(name: str = "logisticsgen") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%H:%M:%S")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


@contextmanager
def timed(label: str, logger: logging.Logger | None = None):
    """Log wall-clock duration of a block - generation stages are slow enough
    that per-stage timings are genuinely useful when tuning a 1M-row run."""
    log = logger or get_logger()
    start = time.perf_counter()
    log.info("→ %s", label)
    yield
    log.info("✓ %s (%.2fs)", label, time.perf_counter() - start)


def haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Great-circle distance in km between two arrays of coordinates."""
    lat1, lon1, lat2, lon2 = map(np.asarray, (lat1, lon1, lat2, lon2))
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlam / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def bearing_deg(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Initial compass bearing from point 1 to point 2, in degrees [0, 360)."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dlam = np.radians(np.asarray(lon2) - np.asarray(lon1))
    x = np.sin(dlam) * np.cos(p2)
    y = np.cos(p1) * np.sin(p2) - np.sin(p1) * np.cos(p2) * np.cos(dlam)
    return (np.degrees(np.arctan2(x, y)) + 360.0) % 360.0


def jitter_coords(
    rng: np.random.Generator, lat, lon, radius_km: float
) -> tuple[np.ndarray, np.ndarray]:
    """Scatter points uniformly within ``radius_km`` of the given centres."""
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    n = lat.size
    theta = rng.uniform(0, 2 * np.pi, n)
    r = radius_km * np.sqrt(rng.uniform(0, 1, n))          # uniform over the disc
    dlat = (r / 111.32) * np.sin(theta)
    dlon = (r / (111.32 * np.cos(np.radians(lat)) + 1e-9)) * np.cos(theta)
    return lat + dlat, lon + dlon


def offset_coords(lat, lon, distance_km, bearing) -> tuple[np.ndarray, np.ndarray]:
    """Move points ``distance_km`` along ``bearing`` degrees."""
    d = np.asarray(distance_km, dtype=float) / EARTH_RADIUS_KM
    b = np.radians(np.asarray(bearing, dtype=float))
    p1, l1 = np.radians(lat), np.radians(lon)
    p2 = np.arcsin(np.sin(p1) * np.cos(d) + np.cos(p1) * np.sin(d) * np.cos(b))
    l2 = l1 + np.arctan2(
        np.sin(b) * np.sin(d) * np.cos(p1), np.cos(d) - np.sin(p1) * np.sin(p2)
    )
    return np.degrees(p2), (np.degrees(l2) + 540) % 360 - 180


def make_ids(prefix: str, n: int, width: int = 9, start: int = 1) -> np.ndarray:
    """Zero-padded surrogate keys, e.g. ``ORD-000000001``."""
    idx = np.arange(start, start + n)
    return np.char.add(f"{prefix}-", np.char.zfill(idx.astype(str), width))


def random_timestamps(
    rng: np.random.Generator,
    start: pd.Timestamp,
    end: pd.Timestamp,
    n: int,
    weekday_weights: list[float] | None = None,
    month_weights: list[float] | None = None,
    hour_weights: list[float] | None = None,
) -> pd.DatetimeIndex:
    """Draw ``n`` timestamps in ``[start, end)`` shaped by seasonality weights.

    Implemented as: uniform day draw → accept/reject by day weight → independent
    hour draw from the hourly profile → uniform minute/second. Fast enough for
    millions of rows and produces the weekly/annual rhythm real order books show.
    """
    days = pd.date_range(start, end, freq="D", inclusive="left")
    n_days = len(days)
    w = np.ones(n_days, dtype=float)
    if weekday_weights:
        w *= np.asarray(weekday_weights, dtype=float)[days.dayofweek.to_numpy()]
    if month_weights:
        w *= np.asarray(month_weights, dtype=float)[days.month.to_numpy() - 1]
    w /= w.sum()

    day_idx = rng.choice(n_days, size=n, p=w)
    if hour_weights:
        hw = np.asarray(hour_weights, dtype=float)
        hours = rng.choice(24, size=n, p=hw / hw.sum())
    else:
        hours = rng.integers(0, 24, n)
    minutes = rng.integers(0, 60, n)
    seconds = rng.integers(0, 60, n)

    base = days.to_numpy()[day_idx].astype("datetime64[s]")
    offsets = (hours * 3600 + minutes * 60 + seconds).astype("timedelta64[s]")
    return pd.DatetimeIndex(base + offsets)


def clip_series(s: pd.Series, quantile: float) -> pd.Series:
    """Two-sided quantile clip; no-op for non-numeric or degenerate input."""
    if not pd.api.types.is_numeric_dtype(s) or s.notna().sum() < 10:
        return s
    lo, hi = s.quantile(1 - quantile), s.quantile(quantile)
    return s.clip(lower=lo, upper=hi)
