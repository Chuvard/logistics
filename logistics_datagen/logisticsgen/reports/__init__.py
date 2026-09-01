"""EDA, missingness and data-dictionary reporting."""

from .eda import build_reports, data_dictionary, profile_tables

__all__ = ["build_reports", "profile_tables", "data_dictionary"]
