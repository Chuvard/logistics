"""Deterministic random number plumbing.

Every generator draws from its *own* named stream derived from the master seed.
That means adding, removing or reordering a table never perturbs the values of
the others - runs stay diffable across code changes.
"""

from __future__ import annotations

import hashlib

import numpy as np

__all__ = ["RandomStreams", "truncated_normal", "lognormal_between", "weighted_choice"]


class RandomStreams:
    """Factory for named, reproducible ``numpy.random.Generator`` instances."""

    def __init__(self, master_seed: int) -> None:
        self.master_seed = int(master_seed)
        self._cache: dict[str, np.random.Generator] = {}

    def _derive(self, name: str) -> int:
        digest = hashlib.blake2b(
            name.encode("utf-8"), digest_size=8, key=str(self.master_seed).encode()
        ).digest()
        return int.from_bytes(digest, "big") % (2**63)

    def get(self, name: str) -> np.random.Generator:
        """Return the generator for stream ``name`` (created on first use)."""
        if name not in self._cache:
            self._cache[name] = np.random.default_rng(self._derive(name))
        return self._cache[name]

    def spawn(self, name: str) -> np.random.Generator:
        """Return a *fresh* generator for ``name`` regardless of cache state."""
        return np.random.default_rng(self._derive(name))


def truncated_normal(
    rng: np.random.Generator,
    mean: float,
    sd: float,
    low: float,
    high: float,
    size: int,
) -> np.ndarray:
    """Normal draws clipped to ``[low, high]`` - cheap and adequate for sim data."""
    return np.clip(rng.normal(mean, sd, size), low, high)


def lognormal_between(
    rng: np.random.Generator, low: float, high: float, size: int, skew: float = 0.85
) -> np.ndarray:
    """Right-skewed draws spanning ``[low, high]``.

    Package weights, order values and repair costs are all long-tailed in real
    logistics data; a plain uniform makes downstream models look implausibly easy.
    """
    low = float(low)
    high = float(max(high, low + 1e-9))
    raw = rng.lognormal(mean=0.0, sigma=skew, size=size)
    # Map the bulk of the distribution into range, then clip the extreme tail.
    scaled = (raw - raw.min()) / max(raw.max() - raw.min(), 1e-9)
    return low + scaled * (high - low)


def weighted_choice(
    rng: np.random.Generator, labels: list[str], weights: list[float], size: int
) -> np.ndarray:
    """Vectorised categorical draw with automatic weight normalisation."""
    w = np.asarray(weights, dtype=float)
    total = w.sum()
    if total <= 0:
        w = np.ones_like(w)
        total = w.sum()
    return rng.choice(np.asarray(labels, dtype=object), size=size, p=w / total)
