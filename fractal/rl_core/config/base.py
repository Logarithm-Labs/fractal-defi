from dataclasses import asdict, dataclass
from typing import Any, Dict, List

from fractal.rl_core.features.uniswap_feature_extractor import \
    UniswapFeatureExtractor


@dataclass
class BaseModelConfig:
    """Base configuration for all RL models."""

    learning_rate: float = 0.0003
    batch_size: int = 64
    seed: int = 42
    net_arch: List[int] = None

    def __post_init__(self):
        if self.net_arch is None:
            self.net_arch = [128, 128]

    def to_model_kwargs(self) -> Dict[str, Any]:
        """Convert config to model initialization kwargs."""
        kwargs = asdict(self)
        # Add common policy kwargs
        kwargs["policy_kwargs"] = {
            "net_arch": kwargs.pop("net_arch"),
            "features_extractor_class": UniswapFeatureExtractor,
            "features_extractor_kwargs": {
                "features_dim": 22,
                "price_history_length": 168,
                "alpha": 0.05,
            },
        }
        # Add common training kwargs
        kwargs.update(
            {
                "verbose": 0,
                "tensorboard_log": "./model_logs/",
            }
        )
        return kwargs
