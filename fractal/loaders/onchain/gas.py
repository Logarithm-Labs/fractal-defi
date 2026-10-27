"""Native gas paid per transaction, and per protocol leg, from receipts.

A strategy that moves through a protocol (a supply, a withdraw, a bridge
burn or mint) pays the gas of the transaction that carried the action.
:func:`transaction_gas_costs` reads each transaction's receipt and returns
the gas in native units; when a leg filter is given it also counts the
matching logs in the receipt, so a transaction that bundles several
actions can be split per leg (or left out by keeping single-leg rows).

USD values are produced only from a caller-supplied price function
``native_price(timestamp) -> float | None`` (for example an hourly series
the caller already holds); nothing here fetches prices.
"""
from collections.abc import Callable, Iterable, Sequence
from typing import Any

import numpy as np
import pandas as pd

from fractal.loaders.rpc import JsonRpcClient, gas_cost_wei, hex_to_int

__all__ = ["GAS_COLUMNS", "count_legs", "transaction_gas_costs"]

GAS_COLUMNS = [
    "tx", "block", "time", "gas_used", "gas_wei", "gas_native", "n_legs", "gas_native_per_leg", "gas_usd",
    "gas_usd_per_leg",
]


def count_legs(receipt: dict[str, Any], address: str, topics: Sequence[str]) -> int:
    """Logs in ``receipt`` emitted by ``address`` whose ``topics[0]`` is one of ``topics``."""
    want_address, want_topics = address.lower(), {t.lower() for t in topics}
    return sum(
        1 for log in receipt.get("logs") or []
        if str(log.get("address", "")).lower() == want_address
        and log.get("topics") and str(log["topics"][0]).lower() in want_topics
    )


def transaction_gas_costs(
    rpc: JsonRpcClient,
    tx_hashes: Iterable[str],
    *,
    leg_address: str | None = None,
    leg_topics: Sequence[str] = (),
    native_price: Callable[[int], float | None] | None = None,
    native_decimals: int = 18,
) -> pd.DataFrame:
    """Gas of each distinct transaction, in native units and optionally USD.

    Args:
        rpc: client of the chain the transactions ran on.
        tx_hashes: transaction hashes (duplicates are read once).
        leg_address / leg_topics: count the receipt's logs from this
            contract with these event topics as the transaction's legs;
            without a filter every transaction is one leg.
        native_price: USD per native token at a block timestamp; ``None``
            or a ``None`` answer leaves the USD columns ``NaN``.
        native_decimals: decimals of the native token.

    Returns:
        One row per transaction with :data:`GAS_COLUMNS`; ``time`` is the
        block's UNIX timestamp. ``*_per_leg`` is ``NaN`` when no leg log
        matched.
    """
    rows, times = [], {}
    for tx in dict.fromkeys(h.lower() for h in tx_hashes):
        receipt = rpc.get_transaction_receipt(tx)
        price = None if receipt.get("effectiveGasPrice") else rpc.get_transaction(tx).get("gasPrice")
        wei = gas_cost_wei(receipt, gas_price=price)
        block = hex_to_int(receipt.get("blockNumber"))
        if block not in times:
            times[block] = rpc.block_timestamp(block)
        legs = count_legs(receipt, leg_address, leg_topics) if leg_address else 1
        native = wei / 10 ** native_decimals
        usd_price = native_price(times[block]) if native_price is not None else None
        usd = native * usd_price if usd_price is not None else np.nan
        rows.append({
            "tx": tx, "block": block, "time": times[block], "gas_used": hex_to_int(receipt.get("gasUsed")),
            "gas_wei": wei, "gas_native": native, "n_legs": legs,
            "gas_native_per_leg": native / legs if legs else np.nan, "gas_usd": usd,
            "gas_usd_per_leg": usd / legs if legs else np.nan,
        })
    return pd.DataFrame(rows, columns=GAS_COLUMNS)
