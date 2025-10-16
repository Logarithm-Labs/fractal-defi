from dataclasses import asdict, dataclass
from typing import Any, Dict, List


@dataclass
class BaseModelConfig:
    """Base configuration for all RL models."""

    learning_rate: float = 0.0003
    batch_size: int = 64
    seed: int = 44
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
        }
        # Add common training kwargs
        kwargs.update(
            {
                "verbose": 0,
                "tensorboard_log": "./model_logs/",
            }
        )
        return kwargs
