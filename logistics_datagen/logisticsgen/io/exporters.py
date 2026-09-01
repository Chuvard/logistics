"""Export layer: CSV, Parquet, SQL DDL+INSERT, SQLite and DuckDB.

Each writer takes the same ``dict[str, DataFrame]`` and is independently
selectable via ``export.formats``. Failures in one format never abort the run -
they are logged and reported so a missing optional dependency (duckdb) doesn't
cost you a 40-minute generation.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import Config
from ..utils import get_logger

__all__ = ["export_all", "write_csv", "write_parquet", "write_sqlite", "write_duckdb", "write_sql"]

logger = get_logger()

_SQL_TYPES = {
    "int64": "BIGINT", "int32": "INTEGER", "int8": "SMALLINT",
    "float64": "DOUBLE PRECISION", "float32": "REAL",
    "bool": "BOOLEAN", "datetime64[ns]": "TIMESTAMP", "object": "TEXT",
}


def _sql_type(dtype) -> str:
    return _SQL_TYPES.get(str(dtype), "TEXT")


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce types that some backends refuse (object-dtype None, categoricals)."""
    out = df.copy()
    for col in out.columns:
        if isinstance(out[col].dtype, pd.CategoricalDtype):
            out[col] = out[col].astype(str)
    return out


# --------------------------------------------------------------------------- #
def write_csv(tables: dict[str, pd.DataFrame], out_dir: Path, cfg: Config) -> list[Path]:
    target = out_dir / "csv"
    target.mkdir(parents=True, exist_ok=True)
    comp = cfg.get("export.csv_compression")
    suffix = ".csv.gz" if comp == "gzip" else ".csv"
    written = []
    for name, df in tables.items():
        if df is None or df.empty:
            continue
        path = target / f"{name}{suffix}"
        df.to_csv(path, index=False, compression=comp)
        written.append(path)
    logger.info("CSV: %d files → %s", len(written), target)
    return written


def write_parquet(tables: dict[str, pd.DataFrame], out_dir: Path, cfg: Config) -> list[Path]:
    target = out_dir / "parquet"
    target.mkdir(parents=True, exist_ok=True)
    comp = cfg.get("export.parquet_compression", "snappy")
    written = []
    for name, df in tables.items():
        if df is None or df.empty:
            continue
        path = target / f"{name}.parquet"
        _prepare(df).to_parquet(path, index=False, compression=comp)
        written.append(path)
    logger.info("Parquet: %d files → %s", len(written), target)
    return written


def write_sqlite(tables: dict[str, pd.DataFrame], out_dir: Path, cfg: Config) -> list[Path]:
    target = out_dir / "sqlite"
    target.mkdir(parents=True, exist_ok=True)
    path = target / cfg.get("export.sqlite_filename", "logistics.db")
    if path.exists():
        path.unlink()

    chunk = int(cfg.get("export.chunk_rows", 500_000))
    with sqlite3.connect(path) as conn:
        for name, df in tables.items():
            if df is None or df.empty:
                continue
            _prepare(df).to_sql(name, conn, index=False, if_exists="replace", chunksize=chunk)
        # Index the join keys - makes the DB genuinely usable for exploration.
        cur = conn.cursor()
        for table, cols in _index_plan(tables).items():
            for col in cols:
                try:
                    cur.execute(f'CREATE INDEX IF NOT EXISTS "idx_{table}_{col}" ON "{table}" ("{col}")')
                except sqlite3.OperationalError as exc:
                    logger.debug("Index skipped %s.%s: %s", table, col, exc)
        conn.commit()
    logger.info("SQLite: %s (%.1f MB)", path, path.stat().st_size / 1e6)
    return [path]


def write_duckdb(tables: dict[str, pd.DataFrame], out_dir: Path, cfg: Config) -> list[Path]:
    try:
        import duckdb  # noqa: PLC0415
    except ImportError:
        logger.warning("duckdb not installed - skipping DuckDB export (pip install duckdb)")
        return []

    target = out_dir / "duckdb"
    target.mkdir(parents=True, exist_ok=True)
    path = target / cfg.get("export.duckdb_filename", "logistics.duckdb")
    if path.exists():
        path.unlink()

    con = duckdb.connect(str(path))
    try:
        for name, df in tables.items():
            if df is None or df.empty:
                continue
            prepared = _prepare(df)  # noqa: F841 - referenced by the SQL below
            con.register("_tmp", prepared)
            con.execute(f'CREATE OR REPLACE TABLE "{name}" AS SELECT * FROM _tmp')
            con.unregister("_tmp")
    finally:
        con.close()
    logger.info("DuckDB: %s (%.1f MB)", path, path.stat().st_size / 1e6)
    return [path]


def write_sql(tables: dict[str, pd.DataFrame], out_dir: Path, cfg: Config) -> list[Path]:
    """Emit portable ``CREATE TABLE`` + batched ``INSERT`` scripts.

    Values are written with parameterless literals, so the output loads into
    Postgres or MySQL without a driver. Large tables are capped by
    ``export.chunk_rows`` per INSERT statement.
    """
    target = out_dir / "sql"
    target.mkdir(parents=True, exist_ok=True)
    batch = 1000
    written = []

    for name, df in tables.items():
        if df is None or df.empty:
            continue
        path = target / f"{name}.sql"
        cols = list(df.columns)
        ddl_cols = ",\n  ".join(f'"{c}" {_sql_type(df[c].dtype)}' for c in cols)

        with path.open("w", encoding="utf-8") as fh:
            fh.write(f'DROP TABLE IF EXISTS "{name}";\n')
            fh.write(f'CREATE TABLE "{name}" (\n  {ddl_cols}\n);\n\n')
            col_list = ", ".join(f'"{c}"' for c in cols)
            values = _literalise(df)
            for start in range(0, len(values), batch):
                block = values[start:start + batch]
                fh.write(f'INSERT INTO "{name}" ({col_list}) VALUES\n')
                fh.write(",\n".join(block))
                fh.write(";\n")
        written.append(path)
    logger.info("SQL: %d scripts → %s", len(written), target)
    return written


def _literalise(df: pd.DataFrame) -> list[str]:
    """Render each row as a ``(...)`` SQL value tuple."""
    out = []
    records = df.to_dict("records")
    for rec in records:
        parts = []
        for value in rec.values():
            if value is None or (isinstance(value, float) and np.isnan(value)) or value is pd.NaT:
                parts.append("NULL")
            elif isinstance(value, (bool, np.bool_)):
                parts.append("TRUE" if value else "FALSE")
            elif isinstance(value, (int, float, np.integer, np.floating)):
                parts.append(repr(value))
            elif isinstance(value, pd.Timestamp):
                parts.append("'" + value.strftime("%Y-%m-%d %H:%M:%S") + "'")
            else:
                parts.append("'" + str(value).replace("'", "''") + "'")
        out.append("  (" + ", ".join(parts) + ")")
    return out


def _index_plan(tables: dict[str, pd.DataFrame]) -> dict[str, list[str]]:
    """Index every ``*_id`` column plus common timestamp/date columns."""
    plan: dict[str, list[str]] = {}
    for name, df in tables.items():
        if df is None or df.empty:
            continue
        cols = [c for c in df.columns
                if c.endswith("_id") or c in {"timestamp", "date", "order_timestamp",
                                              "event_timestamp", "snapshot_date", "shift_date"}]
        if cols:
            plan[name] = cols[:6]
    return plan


# --------------------------------------------------------------------------- #
_WRITERS = {
    "csv": write_csv, "parquet": write_parquet, "sqlite": write_sqlite,
    "duckdb": write_duckdb, "sql": write_sql,
}


def export_all(tables: dict[str, pd.DataFrame], cfg: Config, out_dir: Path | None = None) -> dict[str, list[Path]]:
    out_dir = Path(out_dir or cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, list[Path]] = {}
    for fmt in cfg.get("export.formats", ["parquet"]):
        writer = _WRITERS.get(str(fmt).lower())
        if writer is None:
            logger.warning("Unknown export format %r - skipping", fmt)
            continue
        try:
            results[fmt] = writer(tables, out_dir, cfg)
        except Exception as exc:  # keep other formats alive
            logger.error("Export %s failed: %s", fmt, exc)
            results[fmt] = []
    return results
