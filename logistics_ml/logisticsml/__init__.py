"""logisticsml - enterprise ML platform for logistics optimization.

Quick start::

    from logisticsml import load_config, run_pipeline

    cfg = load_config(["configs/default.yaml"])
    result = run_pipeline(cfg, task_name="late_delivery")
"""

from .config import Config, load_config
from .data import Dataset, build_task, load_table, load_tables
from .pipeline import PipelineResult, run_pipeline

__version__ = "1.0.0"
__all__ = [
    "Config", "load_config", "Dataset", "build_task", "load_table", "load_tables",
    "run_pipeline", "PipelineResult",
]
