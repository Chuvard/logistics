"""Model registry.

A versioned, file-backed store: every training run writes a new immutable
version with its estimator, its preprocessor, the exact feature order, its
metrics and its provenance. Promotion to ``production`` is an explicit act, and
the pointer can be rolled back to any retained version.

Deliberately dependency-free (joblib + JSON on disk). It gives you versioning,
lineage, promotion and rollback without asking anyone to run a tracking server.
"""

from __future__ import annotations

import json
import platform
import shutil
from datetime import datetime, timezone
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from ..utils import get_logger, save_json

__all__ = ["ModelRegistry", "RegisteredModel"]

logger = get_logger()


@dataclass
class RegisteredModel:
    name: str
    version: str
    path: Path
    metadata: dict = field(default_factory=dict)

    @property
    def metrics(self) -> dict:
        return self.metadata.get("metrics", {})

    def load(self):
        import joblib
        return joblib.load(self.path / "model.joblib")

    def load_preprocessor(self):
        import joblib
        p = self.path / "preprocessor.joblib"
        return joblib.load(p) if p.exists() else None


class ModelRegistry:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.index_path = self.root / "index.json"
        self.index = self._load_index()

    # -- index ---------------------------------------------------------------
    def _load_index(self) -> dict:
        if self.index_path.exists():
            return json.loads(self.index_path.read_text(encoding="utf-8"))
        return {"models": {}}

    def _save_index(self) -> None:
        save_json(self.index, self.index_path)

    # -- write ---------------------------------------------------------------
    def register(self, name: str, estimator, metrics: dict, task: dict,
                 feature_names: list[str], preprocessor=None,
                 extra: dict | None = None, keep_last_n: int = 5) -> RegisteredModel:
        import joblib

        version = datetime.now(timezone.utc).strftime("v%Y%m%d_%H%M%S")
        path = self.root / name / version
        path.mkdir(parents=True, exist_ok=True)

        joblib.dump(estimator, path / "model.joblib")
        if preprocessor is not None:
            joblib.dump(preprocessor, path / "preprocessor.joblib")

        metadata = {
            "name": name,
            "version": version,
            "registered_at": datetime.now(timezone.utc).isoformat(),
            "task": task,
            "metrics": metrics,
            "estimator_class": type(estimator).__name__,
            # Feature order is part of the contract - a model served with columns
            # in a different order silently returns garbage.
            "feature_names": feature_names,
            "n_features": len(feature_names),
            "environment": {
                "python": platform.python_version(),
                "platform": platform.platform(),
            },
            **(extra or {}),
        }
        save_json(metadata, path / "metadata.json")

        entry = self.index["models"].setdefault(
            name, {"versions": [], "production": None})
        entry["versions"].append({"version": version, "metrics": metrics,
                                  "registered_at": metadata["registered_at"]})
        self._save_index()
        self._prune(name, keep_last_n)

        logger.info("Registered %s %s (%d features)", name, version, len(feature_names))
        return RegisteredModel(name, version, path, metadata)

    def _prune(self, name: str, keep_last_n: int) -> None:
        """Delete old versions, never the one currently in production."""
        entry = self.index["models"].get(name)
        if not entry or keep_last_n <= 0:
            return
        production = entry.get("production")
        versions = entry["versions"]
        if len(versions) <= keep_last_n:
            return
        keep = {v["version"] for v in versions[-keep_last_n:]}
        if production:
            keep.add(production)
        for v in list(versions):
            if v["version"] not in keep:
                shutil.rmtree(self.root / name / v["version"], ignore_errors=True)
                versions.remove(v)
        self._save_index()

    # -- read ----------------------------------------------------------------
    def list_models(self) -> pd.DataFrame:
        rows = []
        for name, entry in self.index["models"].items():
            for v in entry["versions"]:
                rows.append({
                    "name": name, "version": v["version"],
                    "registered_at": v["registered_at"],
                    "is_production": v["version"] == entry.get("production"),
                    **{k: val for k, val in (v.get("metrics") or {}).items()
                       if isinstance(val, (int, float))},
                })
        return pd.DataFrame(rows)

    def get(self, name: str, version: str | None = None) -> RegisteredModel:
        entry = self.index["models"].get(name)
        if not entry:
            raise KeyError(f"No model registered under {name!r}")
        version = version or entry.get("production") or entry["versions"][-1]["version"]
        path = self.root / name / version
        if not path.exists():
            raise FileNotFoundError(f"Version {version} of {name} is no longer on disk")
        metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
        return RegisteredModel(name, version, path, metadata)

    def promote(self, name: str, version: str | None = None) -> str:
        entry = self.index["models"].get(name)
        if not entry:
            raise KeyError(f"No model registered under {name!r}")
        version = version or entry["versions"][-1]["version"]
        entry["production"] = version
        self._save_index()
        logger.info("Promoted %s %s to production", name, version)
        return version

    def rollback(self, name: str) -> str | None:
        """Point production at the previous retained version."""
        entry = self.index["models"].get(name)
        if not entry or len(entry["versions"]) < 2:
            return None
        current = entry.get("production")
        versions = [v["version"] for v in entry["versions"]]
        if current in versions:
            i = versions.index(current)
            if i == 0:
                return None
            target = versions[i - 1]
        else:
            target = versions[-2]
        entry["production"] = target
        self._save_index()
        logger.info("Rolled %s back to %s", name, target)
        return target
