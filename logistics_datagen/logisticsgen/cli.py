"""Command line interface.

Examples::

    # 50k-delivery sample
    python -m logisticsgen.cli --config configs/default.yaml --config configs/sample.yaml

    # full 1M enterprise run
    python -m logisticsgen.cli --config configs/default.yaml --config configs/full.yaml

    # ad-hoc override without editing YAML
    python -m logisticsgen.cli --scale 0.01 --formats parquet --seed 7
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import load_config
from .generate import generate_dataset
from .utils import get_logger

logger = get_logger()
DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "configs" / "default.yaml"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="logisticsgen",
        description="Generate an enterprise-scale synthetic logistics dataset.")
    p.add_argument("--config", "-c", action="append", default=None,
                   help="YAML config file; repeatable, later files override earlier ones.")
    p.add_argument("--set", dest="overrides", action="append", default=[],
                   metavar="KEY=VALUE", help="Dotted-path override, e.g. --set project.seed=42")
    p.add_argument("--scale", type=float, help="Shortcut for project.scale")
    p.add_argument("--seed", type=int, help="Shortcut for project.seed")
    p.add_argument("--output-dir", "-o", help="Shortcut for project.output_dir")
    p.add_argument("--formats", help="Comma-separated export formats "
                                     "(csv,parquet,sql,sqlite,duckdb)")
    p.add_argument("--no-export", action="store_true", help="Generate in memory only")
    p.add_argument("--no-reports", action="store_true", help="Skip EDA reporting")
    p.add_argument("--no-missingness", action="store_true", help="Disable missingness injection")
    p.add_argument("--no-anomalies", action="store_true", help="Disable anomaly injection")
    p.add_argument("--quiet", "-q", action="store_true", help="Warnings and errors only")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.quiet:
        logger.setLevel("WARNING")

    configs = args.config or [str(DEFAULT_CONFIG)]
    cfg = load_config(configs, args.overrides)

    if args.scale is not None:
        cfg.set("project.scale", args.scale)
    if args.seed is not None:
        cfg.set("project.seed", args.seed)
    if args.output_dir:
        cfg.set("project.output_dir", args.output_dir)
    if args.formats:
        cfg.set("export.formats", [f.strip() for f in args.formats.split(",") if f.strip()])
    if args.no_missingness:
        cfg.set("missingness.enabled", False)
    if args.no_anomalies:
        cfg.set("anomalies.enabled", False)

    logger.info("Config: %s", ", ".join(cfg.sources))
    logger.info("Scale %.4g → %s deliveries | seed %d | output %s",
                cfg.scale, f"{cfg.volume('deliveries'):,}", cfg.seed, cfg.output_dir)

    result = generate_dataset(cfg, export=not args.no_export, reports=not args.no_reports)

    print("\n" + result.summary().to_string(index=False))
    if result.exported:
        print("\nExports:")
        for fmt, paths in result.exported.items():
            print(f"  {fmt:<8} {len(paths)} file(s)")
    if result.reports:
        print(f"\nReports: {result.reports[0].parent}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
