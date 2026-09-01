"""logisticsgen - enterprise-scale synthetic logistics dataset generator.

Quick start::

    from logisticsgen import load_config, generate_dataset

    cfg = load_config(["configs/default.yaml", "configs/sample.yaml"])
    result = generate_dataset(cfg)
    orders = result.tables["orders"]
"""

from .config import Config, load_config
from .generate import GenerationResult, generate_dataset
from .rng import RandomStreams

__version__ = "1.0.0"
__all__ = ["Config", "load_config", "generate_dataset", "GenerationResult", "RandomStreams"]
