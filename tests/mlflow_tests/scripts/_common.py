"""Shared helpers for the e2e MLflow scripts.

Path resolution, observation loading from the example CSV fixtures, and
output-directory bookkeeping. Importable from sibling scripts via:

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _common import (...)

The fixtures live in ``examples/`` and are reused by the slow real-data
tests (``tests/core/test_hyperliquid_basis_real_data.py``,
``tests/core/test_tau_reset_real_data.py``) — same shape, same loaders.

The :class:`HyperliquidBasisParams` dataclass below mirrors the one in
``examples/basis/backtest.py``. ``HyperliquidBasis``
looks for ``params.EXECUTION_COST`` in ``set_up``, so the strategy-level
hyperparams must carry it.
"""
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import pandas as pd

# Make the repo root importable so the scripts can be run from anywhere.
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fractal.core.base import Observation  # noqa: E402
from fractal.core.entities import (  # noqa: E402
    HyperliquidGlobalState,
    UniswapV3LPGlobalState,
    UniswapV3SpotGlobalState,
)
from fractal.strategies.hyperliquid_basis import (  # noqa: E402,F401  pylint: disable=unused-import
    HyperliquidBasisParams,
)

# -------------------------------------------------------------- paths
TESTS_DIR = REPO_ROOT / "tests" / "mlflow_tests"
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", TESTS_DIR / "output"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EXAMPLES_DIR = REPO_ROOT / "examples"
MANAGED_BASIS_FIXTURES = EXAMPLES_DIR / "basis"
TAU_FIXTURE = EXAMPLES_DIR / "tau_reset" / "tau_strategy_result.csv"

DEFAULT_MLFLOW_URI = os.getenv("MLFLOW_URI", "http://localhost:5500")


# -------------------------------------------------------------- experiment names
EXP_MB_SINGLE = "e2e_managed_basis_single"
EXP_MB_PIPELINE = "e2e_managed_basis_pipeline"
EXP_TAU_SINGLE = "e2e_tau_reset_single"
EXP_TAU_PIPELINE = "e2e_tau_reset_pipeline"


# -------------------------------------------------------------- observations
def load_managed_basis_observations(
    ticker: str = "BTC", n: int = 168,
) -> List[Observation]:
    """Load HyperliquidBasis observations from the example CSV fixtures.

    Each fixture has ``HEDGE_mark_price``, ``HEDGE_funding_rate``,
    ``SPOT_price`` columns alongside the strategy's own state columns;
    we only consume the observation inputs.
    """
    path = MANAGED_BASIS_FIXTURES / f"{ticker}_hyperliquid.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"managed_basis fixture missing: {path}. "
            "Run the example pipeline first to populate it."
        )
    df = pd.read_csv(path).head(n)
    obs: List[Observation] = []
    for _, row in df.iterrows():
        ts = pd.Timestamp(row["timestamp"]).to_pydatetime()
        obs.append(Observation(
            timestamp=ts,
            states={
                "SPOT": UniswapV3SpotGlobalState(price=float(row["SPOT_price"])),
                "HEDGE": HyperliquidGlobalState(
                    mark_price=float(row["HEDGE_mark_price"]),
                    funding_rate=float(row["HEDGE_funding_rate"]),
                ),
            },
        ))
    return obs


def load_tau_observations(n: int = 168) -> List[Observation]:
    """Load TauResetStrategy observations from the example CSV fixture."""
    if not TAU_FIXTURE.exists():
        raise FileNotFoundError(
            f"tau fixture missing: {TAU_FIXTURE}. "
            "Run the tau example first to populate it."
        )
    df = pd.read_csv(TAU_FIXTURE).head(n)
    obs: List[Observation] = []
    for _, row in df.iterrows():
        ts = pd.Timestamp(row["timestamp"]).to_pydatetime()
        obs.append(Observation(timestamp=ts, states={
            "UNISWAP_V3": UniswapV3LPGlobalState(
                price=float(row["UNISWAP_V3_price"]),
                tvl=float(row["UNISWAP_V3_tvl"]),
                volume=float(row["UNISWAP_V3_volume"]),
                fees=float(row["UNISWAP_V3_fees"]),
                liquidity=float(row["UNISWAP_V3_liquidity"]),
            ),
        }))
    return obs


# -------------------------------------------------------------- mlflow helpers
def make_mlflow_config(
    experiment_name: str,
    uri: Optional[str] = None,
):
    """Build an :class:`MLflowConfig` pointing at the local docker server."""
    from fractal.core.pipeline import MLflowConfig
    return MLflowConfig(
        mlflow_uri=uri or DEFAULT_MLFLOW_URI,
        experiment_name=experiment_name,
    )


def stamp() -> str:
    """Compact UTC timestamp for output-file naming."""
    return datetime.utcnow().strftime("%Y%m%dT%H%M%S")
