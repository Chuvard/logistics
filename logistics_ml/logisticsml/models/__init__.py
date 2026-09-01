"""Model families: supervised, unsupervised and deep learning."""

from .deep import DeepResults, run_deep_learning
from .supervised import ModelResult, SupervisedResults, run_supervised
from .unsupervised import UnsupervisedResults, run_unsupervised

__all__ = ["run_supervised", "SupervisedResults", "ModelResult",
           "run_unsupervised", "UnsupervisedResults",
           "run_deep_learning", "DeepResults"]
