"""Command line interface.

Examples::

    # full pipeline on the default task
    python -m logisticsml.cli run

    # a different prediction task
    python -m logisticsml.cli run --task fraud_detection

    # quick pass: cap rows, skip the slow stages
    python -m logisticsml.cli run --max-rows 20000 --skip unsupervised,deep_learning

    # inspect and serve the registry
    python -m logisticsml.cli registry
    python -m logisticsml.cli serve --task late_delivery
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import load_config
from .utils import dependency_report, get_logger

logger = get_logger()
DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "configs" / "default.yaml"

_STAGE_FLAGS = {
    "eda": "eda.enabled",
    "supervised": "supervised.enabled",
    "unsupervised": "unsupervised.enabled",
    "deep_learning": "deep_learning.enabled",
    "optimization": "optimization.enabled",
    "explainability": "explainability.enabled",
    "reports": "reports.enabled",
}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="logisticsml",
        description="Enterprise ML platform for logistics optimization.")
    sub = p.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", "-c", action="append", default=None,
                        help="YAML config; repeatable, later files win.")
    common.add_argument("--set", dest="overrides", action="append", default=[],
                        metavar="KEY=VALUE", help="Dotted-path override.")
    common.add_argument("--quiet", "-q", action="store_true")

    run = sub.add_parser("run", parents=[common], help="Run the full pipeline")
    run.add_argument("--task", "-t", help="Prediction task name")
    run.add_argument("--output-dir", "-o")
    run.add_argument("--max-rows", type=int, help="Cap rows for a fast pass")
    run.add_argument("--seed", type=int)
    run.add_argument("--skip", help=f"Comma-separated stages: {', '.join(_STAGE_FLAGS)}")
    run.add_argument("--only", help="Run only these stages (comma-separated)")

    sub.add_parser("tasks", parents=[common], help="List available prediction tasks")
    sub.add_parser("deps", parents=[common], help="Show optional dependency status")

    reg = sub.add_parser("registry", parents=[common], help="Inspect the model registry")
    reg.add_argument("--promote", nargs=2, metavar=("NAME", "VERSION"))
    reg.add_argument("--rollback", metavar="NAME")

    srv = sub.add_parser("serve", parents=[common], help="Start the prediction API")
    srv.add_argument("--task", "-t", help="Registered model name (defaults to default_task)")
    srv.add_argument("--version", "-v")
    srv.add_argument("--host")
    srv.add_argument("--port", type=int)
    return p


def _load(args):
    cfg = load_config(args.config or [str(DEFAULT_CONFIG)], args.overrides)
    if getattr(args, "quiet", False):
        logger.setLevel("WARNING")
    for attr, key in [("output_dir", "project.output_dir"), ("seed", "project.seed"),
                      ("max_rows", "data.max_rows")]:
        value = getattr(args, attr, None)
        if value is not None:
            cfg.set(key, value)

    if getattr(args, "only", None):
        wanted = {s.strip() for s in args.only.split(",")}
        for stage, key in _STAGE_FLAGS.items():
            cfg.set(key, stage in wanted or stage == "reports")
    if getattr(args, "skip", None):
        for stage in (s.strip() for s in args.skip.split(",")):
            if stage in _STAGE_FLAGS:
                cfg.set(_STAGE_FLAGS[stage], False)
            else:
                logger.warning("Unknown stage %r in --skip", stage)
    return cfg


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = _load(args)

    if args.command == "deps":
        print(dependency_report().to_string(index=False))
        return 0

    if args.command == "tasks":
        rows = []
        for name, spec in (cfg.get("tasks") or {}).items():
            rows.append({"task": name, "type": spec.get("type"),
                         "target": spec.get("target"),
                         "primary_metric": spec.get("primary_metric"),
                         "description": spec.get("description", "")})
        import pandas as pd
        print(pd.DataFrame(rows).to_string(index=False))
        return 0

    if args.command == "registry":
        from .serving.registry import ModelRegistry
        registry = ModelRegistry(cfg._resolve(cfg.get("registry.path", "./registry")))
        if args.promote:
            registry.promote(args.promote[0], args.promote[1])
        if args.rollback:
            target = registry.rollback(args.rollback)
            print(f"Rolled back to {target}" if target else "Nothing to roll back to")
        table = registry.list_models()
        print(table.to_string(index=False) if not table.empty else "Registry is empty")
        return 0

    if args.command == "serve":
        from .serving.api import PredictionService, serve
        from .serving.registry import ModelRegistry
        registry = ModelRegistry(cfg._resolve(cfg.get("registry.path", "./registry")))
        name = args.task or cfg.get("default_task")
        service = PredictionService(
            registry, name, args.version,
            batch_limit=int(cfg.get("serving.batch_size_limit", 5000)))
        serve(service,
              args.host or cfg.get("serving.api_host", "127.0.0.1"),
              args.port or int(cfg.get("serving.api_port", 8000)))
        return 0

    # ---- run ---------------------------------------------------------------
    from .pipeline import run_pipeline

    logger.info("Config: %s", ", ".join(cfg.sources))
    result = run_pipeline(cfg, args.task)

    print("\n" + "=" * 78)
    print(f"TASK: {result.task.get('name')} ({result.task.get('type')})")
    print(f"      {result.task.get('description', '')}")
    print("=" * 78)

    if result.supervised is not None:
        lb = result.supervised.leaderboard()
        if not lb.empty:
            wanted = ["model", result.supervised.primary_metric, "accuracy",
                      "precision", "recall", "f1", "roc_auc", "average_precision",
                      "rmse", "mae", "r2", "train_seconds"]
            # dict.fromkeys preserves order while dropping repeats - the primary
            # metric is usually also in the generic list.
            cols = [c for c in dict.fromkeys(wanted) if c in lb.columns]
            print("\nSUPERVISED LEADERBOARD")
            print(lb[cols].to_string(index=False))
    if result.deep is not None and not result.deep.leaderboard().empty:
        print(f"\nDEEP LEARNING (backend: {result.deep.backend})")
        dl = result.deep.leaderboard()
        cols = [c for c in ["model", "roc_auc", "f1", "rmse", "r2",
                            "reconstruction_mse", "train_seconds", "note"] if c in dl.columns]
        print(dl[cols].to_string(index=False))
    if result.optimization is not None and not result.optimization.leaderboard().empty:
        print("\nOPTIMIZATION")
        print(result.optimization.leaderboard().to_string(index=False))
    if result.registered is not None:
        print(f"\nRegistered: {result.registered.name} {result.registered.version}")
    if result.reports:
        print(f"Reports:    {result.reports[0].parent}")
    if result.errors:
        print(f"\nStage errors: {result.errors}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
