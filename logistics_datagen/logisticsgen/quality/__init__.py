"""Data-quality simulation: missingness mechanisms and anomaly injection."""

from .anomalies import AnomalyReport, inject_anomalies
from .missingness import MissingnessReport, apply_missingness, sweep_missingness

__all__ = [
    "AnomalyReport", "inject_anomalies",
    "MissingnessReport", "apply_missingness", "sweep_missingness",
]
