"""Configuration loading with dotted-path access.

Mirrors the generator's config module so both halves of the project behave the
same way: YAML files merge left-to-right, then ``--set`` overrides land on top.
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
    out = copy.deepcopy(base)
    for key, value in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, Mapping):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _coerce(text: str) -> Any:
    try:
        return json.loads(text)
    except (TypeError, ValueError):
        return text


@dataclass
class Config:
    data: dict = field(default_factory=dict)
    sources: list[str] = field(default_factory=list)
    base_dir: Path = field(default_factory=Path.cwd)

    def get(self, path: str, default: Any = None) -> Any:
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
    def seed(self) -> int:
        return int(self.get("project.seed", 0))

    @property
    def output_dir(self) -> Path:
        return self._resolve(self.get("project.output_dir", "./artifacts"))

    def _resolve(self, path: str | Path) -> Path:
        """Resolve a config path relative to the *config file*, not the cwd."""
        p = Path(path).expanduser()
        return p if p.is_absolute() else (self.base_dir / p).resolve()

    def task(self, name: str | None = None) -> dict:
        """Return the task spec, with its name attached."""
        name = name or self.get("default_task", "late_delivery")
        spec = self.get(f"tasks.{name}")
        if spec is None:
            available = ", ".join(sorted((self.get("tasks") or {}).keys()))
            raise KeyError(f"Unknown task {name!r}. Available: {available}")
        return {"name": name, **spec}

    def to_dict(self) -> dict:
        return copy.deepcopy(self.data)

    def dump(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(self.data, sort_keys=False), encoding="utf-8")


def load_config(paths: Iterable[str | Path], overrides: Iterable[str] = ()) -> Config:
    merged: dict = {}
    sources: list[str] = []
    base_dir = Path.cwd()
    for i, p in enumerate(paths):
        p = Path(p)
        if not p.exists():
            raise FileNotFoundError(f"Config file not found: {p}")
        if i == 0:
            base_dir = p.resolve().parent
        with p.open("r", encoding="utf-8") as fh:
            merged = _deep_merge(merged, yaml.safe_load(fh) or {})
        sources.append(str(p))

    cfg = Config(data=merged, sources=sources, base_dir=base_dir)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"--set expects key.path=value, got {item!r}")
        key, _, raw = item.partition("=")
        cfg.set(key.strip(), _coerce(raw.strip()))
    return cfg
