"""Configuration loading and access.

The generator is driven entirely by YAML. Multiple config files are merged
left-to-right (later wins), then CLI ``--set`` overrides are applied on top.
Access is dotted-path based so call sites read like the YAML itself::

    cfg.get("operations.avg_speed_kmh.urban")
    cfg.volume("deliveries")          # already multiplied by project.scale
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

__all__ = ["Config", "load_config"]


def _deep_merge(base: dict, override: Mapping) -> dict:
    """Recursively merge ``override`` into ``base`` and return a new dict."""
    out = copy.deepcopy(base)
    for key, value in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, Mapping):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _coerce(text: str) -> Any:
    """Best-effort scalar parse for ``--set key=value`` CLI overrides."""
    try:
        return json.loads(text)
    except (TypeError, ValueError):
        return text


@dataclass
class Config:
    """Immutable-ish view over the merged configuration tree."""

    data: dict = field(default_factory=dict)
    sources: list[str] = field(default_factory=list)

    # -- access ---------------------------------------------------------------
    def get(self, path: str, default: Any = None) -> Any:
        """Fetch a value by dotted path, e.g. ``"export.formats"``."""
        node: Any = self.data
        for part in path.split("."):
            if not isinstance(node, Mapping) or part not in node:
                return default
            node = node[part]
        return node

    def require(self, path: str) -> Any:
        sentinel = object()
        value = self.get(path, sentinel)
        if value is sentinel:
            raise KeyError(f"Missing required config key: {path!r}")
        return value

    def set(self, path: str, value: Any) -> None:
        parts = path.split(".")
        node = self.data
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value

    # -- derived --------------------------------------------------------------
    @property
    def scale(self) -> float:
        return float(self.get("project.scale", 1.0))

    def volume(self, name: str) -> int:
        """Entity count for ``name`` after applying ``project.scale``.

        Always returns at least 1 so tiny smoke-test scales stay valid.
        """
        base = int(self.require(f"volumes.{name}"))
        return max(1, int(round(base * self.scale)))

    @property
    def seed(self) -> int:
        return int(self.get("project.seed", 0))

    @property
    def output_dir(self) -> Path:
        return Path(self.get("project.output_dir", "./data")).expanduser()

    def to_dict(self) -> dict:
        return copy.deepcopy(self.data)

    def dump(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(self.data, sort_keys=False), encoding="utf-8")


def load_config(paths: Iterable[str | Path], overrides: Iterable[str] = ()) -> Config:
    """Load and merge YAML config files, then apply ``key=value`` overrides."""
    merged: dict = {}
    sources: list[str] = []
    for p in paths:
        p = Path(p)
        if not p.exists():
            raise FileNotFoundError(f"Config file not found: {p}")
        with p.open("r", encoding="utf-8") as fh:
            loaded = yaml.safe_load(fh) or {}
        merged = _deep_merge(merged, loaded)
        sources.append(str(p))

    cfg = Config(data=merged, sources=sources)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"--set expects key.path=value, got {item!r}")
        key, _, raw = item.partition("=")
        cfg.set(key.strip(), _coerce(raw.strip()))
    return cfg
