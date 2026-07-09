"""Helpers for replicating real Uniswap V3 LP positions in Fractal:
the positions.json registry, on-chain ground truth (un-floored
``feeGrowthInside`` accrual via web3.py, head-state reads only), and
observation building with the mint-block correction of the first bar.

RPC: ``BASE_RPC_URL`` (default https://mainnet.base.org); an archive
node is only needed for the exact mint-edge correction, otherwise it
degrades to time-proration. Requires ``pip install web3``.
"""
import json
import os
from pathlib import Path
from typing import List, Optional

import pandas as pd
from web3 import Web3

from fractal.core.base import Observation
from fractal.core.entities import UniswapV3LPGlobalState

U256 = 1 << 256

POOL_ABI = json.loads("""[
 {"name":"slot0","outputs":[{"type":"uint160","name":"sqrtPriceX96"},{"type":"int24","name":"tick"},
  {"type":"uint16","name":"observationIndex"},{"type":"uint16","name":"observationCardinality"},
  {"type":"uint16","name":"observationCardinalityNext"},{"type":"uint8","name":"feeProtocol"},
  {"type":"bool","name":"unlocked"}],"inputs":[],"stateMutability":"view","type":"function"},
 {"name":"feeGrowthGlobal0X128","outputs":[{"type":"uint256"}],"inputs":[],"stateMutability":"view","type":"function"},
 {"name":"feeGrowthGlobal1X128","outputs":[{"type":"uint256"}],"inputs":[],"stateMutability":"view","type":"function"},
 {"name":"ticks","outputs":[{"type":"uint128","name":"liquidityGross"},{"type":"int128","name":"liquidityNet"},
  {"type":"uint256","name":"feeGrowthOutside0X128"},{"type":"uint256","name":"feeGrowthOutside1X128"},
  {"type":"int56","name":"tickCumulativeOutside"},{"type":"uint160","name":"secondsPerLiquidityOutsideX128"},
  {"type":"uint32","name":"secondsOutside"},{"type":"bool","name":"initialized"}],
  "inputs":[{"type":"int24","name":"tick"}],"stateMutability":"view","type":"function"}
]""")
NFPM_ABI = json.loads("""[
 {"name":"positions","outputs":[{"type":"uint96","name":"nonce"},{"type":"address","name":"operator"},
  {"type":"address","name":"token0"},{"type":"address","name":"token1"},{"type":"uint24","name":"fee"},
  {"type":"int24","name":"tickLower"},{"type":"int24","name":"tickUpper"},{"type":"uint128","name":"liquidity"},
  {"type":"uint256","name":"feeGrowthInside0LastX128"},{"type":"uint256","name":"feeGrowthInside1LastX128"},
  {"type":"uint128","name":"tokensOwed0"},{"type":"uint128","name":"tokensOwed1"}],
  "inputs":[{"type":"uint256","name":"tokenId"}],"stateMutability":"view","type":"function"}
]""")


def get_w3() -> Web3:
    """Web3 over BASE_RPC_URL with retries — public endpoints rate-limit."""
    from requests.exceptions import ConnectionError as RequestsConnectionError
    from requests.exceptions import HTTPError, Timeout
    from web3.providers.rpc.utils import ExceptionRetryConfiguration

    provider = Web3.HTTPProvider(
        os.getenv("BASE_RPC_URL", "https://mainnet.base.org"),
        exception_retry_configuration=ExceptionRetryConfiguration(
            errors=(RequestsConnectionError, HTTPError, Timeout),
            retries=8,
            backoff_factor=0.5,
        ),
    )
    return Web3(provider)


def load_positions(path: Optional[str] = None) -> List[dict]:
    """Read the positions registry shipped next to this module."""
    path = path or Path(__file__).with_name("positions.json")
    return json.loads(Path(path).read_text())


# ------------------------------------------------------------ ground truth
def _fee_growth_inside(pool, tick_lower: int, tick_upper: int) -> tuple:
    """Current feeGrowthInside{0,1}X128 (mirrors V3 Tick.getFeeGrowthInside):
    ``global − below(lower) − above(upper)``; halves flip with the current
    tick's side, underflow wraps mod 2^256 by design.
    """
    fg0 = pool.functions.feeGrowthGlobal0X128().call()
    fg1 = pool.functions.feeGrowthGlobal1X128().call()
    tick = pool.functions.slot0().call()[1]
    lower = pool.functions.ticks(tick_lower).call()
    upper = pool.functions.ticks(tick_upper).call()
    lo0, lo1 = lower[2], lower[3]
    up0, up1 = upper[2], upper[3]
    below0 = lo0 if tick >= tick_lower else (fg0 - lo0) % U256
    below1 = lo1 if tick >= tick_lower else (fg1 - lo1) % U256
    above0 = up0 if tick < tick_upper else (fg0 - up0) % U256
    above1 = up1 if tick < tick_upper else (fg1 - up1) % U256
    return (fg0 - below0 - above0) % U256, (fg1 - below1 - above1) % U256


def position_ground_truth(position: dict, w3: Optional[Web3] = None) -> dict:
    """Un-floored on-chain fees since mint, per token leg:
    ``(feeGrowthInside_now − feeGrowthInsideLast_mint) × L / 2^128``.
    Head-state reads only — the mint-side checkpoint is stored in the
    position manager.
    """
    w3 = w3 or get_w3()
    nfpm = w3.eth.contract(Web3.to_checksum_address(position["manager"]), abi=NFPM_ABI)
    pool = w3.eth.contract(Web3.to_checksum_address(position["pool_address"]), abi=POOL_ABI)

    p = nfpm.functions.positions(position["token_id"]).call()
    liquidity, fg_in0_last, fg_in1_last, owed0, owed1 = p[7], p[8], p[9], p[10], p[11]

    in0, in1 = _fee_growth_inside(pool, position["tick_lower"], position["tick_upper"])
    fees_token0 = (((in0 - fg_in0_last) % U256) * liquidity / 2 ** 128 + owed0) / 10 ** position["decimals0"]
    fees_token1 = (((in1 - fg_in1_last) % U256) * liquidity / 2 ** 128 + owed1) / 10 ** position["decimals1"]
    return {
        "fees_token0": fees_token0,
        "fees_token1": fees_token1,
        "liquidity_onchain": liquidity,
    }


# ------------------------------------------------------- observation build
def to_notional_frame(
    pool_history: pd.DataFrame,
    notional_side: str,
    notional_usd: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """Convert a raw pool-history frame to the entity's price convention:
    invert ``price`` when the notional token is slot0, convert USD
    ``fees`` to notional via ``notional_usd``; ``fee_growth0/1`` are
    token-denominated and pass through unchanged.
    """
    df = pool_history.copy()
    if notional_side == "token0":
        df["price"] = 1.0 / df["price"]
    if notional_usd is not None:
        usd = notional_usd.reindex(df.index.union(notional_usd.index)).sort_index().ffill()
        df["fees"] = df["fees"] / usd.reindex(df.index).values
    return df.dropna(subset=["price", "fees", "liquidity"])


def anchor_first_bar(loader, frame: pd.DataFrame, position: dict,
                     w3: Optional[Web3] = None) -> pd.DataFrame:
    """Correct the first post-mint bar's feeGrowth delta: the loader's
    delta spans back to the previous pool row, crediting pre-mint fees.
    Exact mode reads the counters at the mint block (archive
    ``eth_call``); on a non-archive node the delta is time-prorated —
    fine on dense pools (±1%), rough on bursty sparse ones.
    """
    if frame.empty:
        return frame
    frame = frame.copy()
    first_ts = int(frame.index[0].timestamp())
    w3 = w3 or get_w3()
    try:
        pool = w3.eth.contract(Web3.to_checksum_address(position["pool_address"]), abi=POOL_ABI)
        block = position["mint_block"]
        anchor0 = pool.functions.feeGrowthGlobal0X128().call(block_identifier=block)
        anchor1 = pool.functions.feeGrowthGlobal1X128().call(block_identifier=block)
    except Exception:  # noqa: BLE001 - non-archive node: prorate instead
        anchor0 = anchor1 = None
    if anchor0 is not None:
        query = ('{ poolHourDatas(first: 1, where: {pool: "%s", periodStartUnix: %d}) '
                 '{ feeGrowthGlobal0X128 feeGrowthGlobal1X128 } }'
                 ) % (position["pool_address"], first_ts)
        rows = loader._make_request(query)["poolHourDatas"]  # noqa: SLF001 - example-level reuse
        if rows:
            cum0, cum1 = int(rows[0]["feeGrowthGlobal0X128"]), int(rows[0]["feeGrowthGlobal1X128"])
            frame.iloc[0, frame.columns.get_loc("fee_growth0")] = max(cum0 - anchor0, 0) / 2 ** 128
            frame.iloc[0, frame.columns.get_loc("fee_growth1")] = max(cum1 - anchor1, 0) / 2 ** 128
        return frame
    # Fallback: find where the first bar's delta actually starts (end of
    # the previous pool row) and keep the post-mint time share of it.
    query = ('{ poolHourDatas(first: 1, orderBy: periodStartUnix, orderDirection: desc, '
             'where: {pool: "%s", periodStartUnix_lt: %d}) { periodStartUnix } }'
             ) % (position["pool_address"], first_ts)
    rows = loader._make_request(query)["poolHourDatas"]  # noqa: SLF001
    span_start = int(rows[0]["periodStartUnix"]) + 3600 if rows else first_ts
    span_end = first_ts + 3600
    weight = min(max((span_end - position["mint_ts"]) / max(span_end - span_start, 1), 0.0), 1.0)
    for col in ("fee_growth0", "fee_growth1"):
        frame.iloc[0, frame.columns.get_loc(col)] *= weight
    return frame


def append_head_bar(loader, frame: pd.DataFrame, position: dict,
                    w3: Optional[Web3] = None) -> pd.DataFrame:
    """Append a synthetic bar carrying the counter growth the subgraph
    has not snapshotted yet. Hourly rows copy ``feeGrowthGlobal`` BEFORE
    the triggering swap's fee is applied, so the newest row lags the
    chain by up to one swap — invisible until the pool's next swap, and
    a visible share of a young position's fees on sparse pools.
    """
    if frame.empty:
        return frame
    w3 = w3 or get_w3()
    first_ts = int(frame.index[-1].timestamp())
    query = ('{ poolHourDatas(first: 1, orderBy: periodStartUnix, orderDirection: desc, '
             'where: {pool: "%s"}) { feeGrowthGlobal0X128 feeGrowthGlobal1X128 } }'
             ) % position["pool_address"]
    rows = loader._make_request(query)["poolHourDatas"]  # noqa: SLF001 - example-level reuse
    if not rows:
        return frame
    pool = w3.eth.contract(Web3.to_checksum_address(position["pool_address"]), abi=POOL_ABI)
    tail0 = max(pool.functions.feeGrowthGlobal0X128().call() - int(rows[0]["feeGrowthGlobal0X128"]), 0)
    tail1 = max(pool.functions.feeGrowthGlobal1X128().call() - int(rows[0]["feeGrowthGlobal1X128"]), 0)
    if not (tail0 or tail1):
        return frame
    head = frame.iloc[[-1]].copy()
    head.index = [max(pd.Timestamp.now(tz="UTC"), frame.index[-1] + pd.Timedelta(seconds=1))]
    head["volume"] = 0.0
    head["fees"] = 0.0
    head["fee_growth0"] = tail0 / 2 ** 128
    head["fee_growth1"] = tail1 / 2 ** 128
    return pd.concat([frame, head])


def build_observations(df: pd.DataFrame) -> List[Observation]:
    """Assemble LP-entity observations. Bars are not filtered on ``tvl``
    (per-bar feeGrowth deltas are non-recoverable); ``None`` marks bars
    without feeGrowth data so the ``auto`` fee model falls back
    explicitly.
    """
    df = df.sort_index()
    has_growth = {"fee_growth0", "fee_growth1"}.issubset(df.columns)
    return [
        Observation(
            timestamp=timestamp,
            states={
                'UNISWAP_V3': UniswapV3LPGlobalState(
                    price=row.price, tvl=row.tvl if pd.notna(row.tvl) else 0.0,
                    volume=row.volume,
                    fees=row.fees, liquidity=row.liquidity,
                    fee_growth0=row.fee_growth0 if has_growth else None,
                    fee_growth1=row.fee_growth1 if has_growth else None,
                ),
            }
        ) for timestamp, row in df.iterrows()
    ]
