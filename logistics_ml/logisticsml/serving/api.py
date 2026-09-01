"""Prediction service.

``PredictionService`` is the core: it loads a version from the registry, applies
the *same* preprocessor that was fitted at training time, enforces the feature
contract, and returns predictions with probabilities.

Two front-ends wrap it:

* a FastAPI app when FastAPI is installed (``uvicorn`` ready), and
* a dependency-free ``http.server`` fallback exposing the same JSON routes,

so the service is runnable in any environment. Both share one request-handling
path, so behaviour cannot drift between them.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..utils import get_logger, optional_import
from .registry import ModelRegistry

__all__ = ["PredictionService", "create_app", "serve"]

logger = get_logger()


@dataclass
class PredictionService:
    registry: ModelRegistry
    model_name: str
    version: str | None = None
    batch_limit: int = 5000

    def __post_init__(self) -> None:
        self.registered = self.registry.get(self.model_name, self.version)
        self.model = self.registered.load()
        self.preprocessor = self.registered.load_preprocessor()
        self.feature_names: list[str] = self.registered.metadata.get("feature_names", [])
        self.task = self.registered.metadata.get("task", {})
        logger.info("Serving %s %s (%d features)", self.model_name,
                    self.registered.version, len(self.feature_names))

    # -- contract ------------------------------------------------------------
    def _prepare(self, records: list[dict]) -> pd.DataFrame:
        """Turn raw JSON records into the exact matrix the model was trained on.

        Raw input goes through the fitted preprocessor. Input that is already
        preprocessed is accepted too, but must satisfy the feature contract:
        missing columns are filled with 0.0 and reported, and column order is
        restored - silently accepting a different order is how serving skew
        starts.
        """
        frame = pd.DataFrame(records)
        if frame.empty:
            raise ValueError("No records supplied")

        if self.preprocessor is not None and not set(self.feature_names).issubset(frame.columns):
            try:
                arr = self.preprocessor.transform(frame)
                names = list(self.preprocessor.get_feature_names_out())
                frame = pd.DataFrame(np.asarray(arr, dtype=float), columns=names)
            except Exception as exc:
                raise ValueError(f"Preprocessing failed: {exc}") from exc

        missing = [c for c in self.feature_names if c not in frame.columns]
        for c in missing:
            frame[c] = 0.0
        if missing:
            logger.warning("Request missing %d feature(s), defaulted to 0: %s",
                           len(missing), ", ".join(missing[:5]))

        return frame[self.feature_names].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # -- inference -----------------------------------------------------------
    def predict(self, records: list[dict]) -> dict:
        if len(records) > self.batch_limit:
            raise ValueError(f"Batch of {len(records)} exceeds limit {self.batch_limit}")

        X = self._prepare(records)
        preds = self.model.predict(X)
        out: dict = {
            "model": self.model_name,
            "version": self.registered.version,
            "task": self.task.get("name"),
            "n_records": len(X),
            "predictions": [p.item() if hasattr(p, "item") else p for p in preds],
        }
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(X)
            classes = [str(c) for c in getattr(self.model, "classes_", range(proba.shape[1]))]
            out["probabilities"] = [dict(zip(classes, np.round(row, 6).tolist()))
                                    for row in proba]
            if proba.shape[1] == 2:
                out["positive_probability"] = np.round(proba[:, 1], 6).tolist()
        return out

    def health(self) -> dict:
        return {
            "status": "ok", "model": self.model_name,
            "version": self.registered.version,
            "n_features": len(self.feature_names),
            "task": self.task.get("name"),
            "metrics": self.registered.metrics,
        }

    def schema(self) -> dict:
        return {
            "model": self.model_name, "version": self.registered.version,
            "task": self.task,
            "features": self.feature_names,
            "example_request": {"records": [
                {name: 0.0 for name in self.feature_names[:8]}]},
        }


# --------------------------------------------------------------------------- #
def create_app(service: PredictionService):
    """Build a FastAPI app, or return ``None`` if FastAPI is unavailable."""
    fastapi = optional_import("fastapi")
    if fastapi is None:
        return None
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel

    class PredictRequest(BaseModel):
        records: list[dict]

    app = FastAPI(title="Logistics Optimization ML Platform",
                  description="Prediction service backed by the model registry",
                  version="1.0.0")

    @app.get("/health")
    def health():
        return service.health()

    @app.get("/schema")
    def schema():
        return service.schema()

    @app.post("/predict")
    def predict(request: PredictRequest):
        try:
            return service.predict(request.records)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return app


def serve(service: PredictionService, host: str = "127.0.0.1", port: int = 8000) -> None:
    """Run the service, preferring FastAPI/uvicorn and falling back to stdlib."""
    app = create_app(service)
    uvicorn = optional_import("uvicorn")
    if app is not None and uvicorn is not None:
        logger.info("Serving with FastAPI on http://%s:%d", host, port)
        uvicorn.run(app, host=host, port=port, log_level="info")
        return

    from http.server import BaseHTTPRequestHandler, HTTPServer

    class Handler(BaseHTTPRequestHandler):
        def _send(self, payload: dict, code: int = 200) -> None:
            body = json.dumps(payload, default=str).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/health":
                self._send(service.health())
            elif self.path == "/schema":
                self._send(service.schema())
            else:
                self._send({"error": "not found"}, 404)

        def do_POST(self):
            if self.path != "/predict":
                self._send({"error": "not found"}, 404)
                return
            length = int(self.headers.get("Content-Length", 0))
            try:
                payload = json.loads(self.rfile.read(length) or b"{}")
                self._send(service.predict(payload.get("records", [])))
            except Exception as exc:
                self._send({"error": str(exc)}, 400)

        def log_message(self, *args):     # keep stdout clean
            pass

    logger.info("FastAPI unavailable - serving with the standard library on http://%s:%d",
                host, port)
    HTTPServer((host, port), Handler).serve_forever()
