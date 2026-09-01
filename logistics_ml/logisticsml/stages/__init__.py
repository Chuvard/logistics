"""Pipeline stages: preprocessing/splitting and exploratory data analysis."""

from .eda import EDAResult, run_eda
from .preprocessing import SplitData, build_preprocessor, preprocess, split_dataset

__all__ = ["preprocess", "SplitData", "split_dataset", "build_preprocessor",
           "run_eda", "EDAResult"]
