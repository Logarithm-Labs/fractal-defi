"""Borrow/repay flows of a lending debt token from its on-chain events.

A variable-debt token (Aave V3, Spark and other forks) is minted on
borrow and burned on repay; it cannot be transferred between users. So
its ``Transfer`` logs from or to the zero address are the debt flows of
every borrower, readable with plain ``eth_getLogs`` (no archive state,
no explorer key). The token address is the reserve's
``variableDebtTokenAddress`` (see
:func:`fractal.loaders.onchain.aave_reserve_rates.decode_reserve_data`).

Aave V3 mints and burns the *balance change* at each interaction, which
folds in the interest accrued since the borrower's previous interaction
(a repay smaller than that interest even shows up as a mint). With
``split_interest=True`` the loader reads the token's V3 ``Mint`` / ``Burn``
events instead, whose ``balanceIncrease`` separates that interest from
the principal actually borrowed or repaid.
"""
import hashlib
from collections.abc import Sequence
from typing import Any, ClassVar

import pandas as pd

from fractal.loaders._http import HttpClient
from fractal.loaders.base_loader import Loader, LoaderType
from fractal.loaders.rpc import (
    JsonRpcClient,
    decode_words,
    encode_address,
    event_topic,
    hex_to_int,
    topic_to_address,
)
from fractal.loaders.thegraph.base_graph_loader import validate_evm_address

__all__ = ["DebtTokenFlowsLoader"]

TRANSFER_TOPIC = event_topic("Transfer(address,address,uint256)")
MINT_TOPIC = event_topic("Mint(address,address,uint256,uint256,uint256)")
BURN_TOPIC = event_topic("Burn(address,address,uint256,uint256,uint256)")
ZERO_TOPIC = "0x" + "0" * 64


class DebtTokenFlowsLoader(Loader):
    """Per-event mint/burn flows of one debt token over a block range.

    ``read`` returns a ``DataFrame`` indexed by UTC block time with
    ``block``, ``log_index``, ``tx``, ``user`` and ``amount`` — the signed
    change of the user's debt in token units (``+`` mint, ``-`` burn).
    With ``split_interest`` it adds ``interest`` (accrued since the user's
    previous interaction) and ``principal = amount - interest`` (``+``
    borrowed, ``-`` repaid).

    Args:
        rpc_url: JSON-RPC endpoint (``eth_getLogs`` capable; never cached).
        debt_token: the debt token address.
        decimals: token decimals (those of the underlying asset).
        start_block / end_block: inclusive range; latest when omitted.
        users: optional borrowers to restrict to (filtered on the node).
        split_interest: read Aave V3 ``Mint``/``Burn`` events to split
            accrued interest from principal.
        chunk_size: blocks per ``eth_getLogs``; halved on rejection.
        loader_type: cache format.
        http: injectable transport (offline tests pass a fake).
    """

    _BASE: ClassVar[list[str]] = ["time", "block", "log_index", "tx", "user", "amount"]
    _SPLIT: ClassVar[list[str]] = ["interest", "principal"]

    def __init__(
        self,
        rpc_url: str,
        debt_token: str,
        decimals: int,
        start_block: int,
        end_block: int | None = None,
        users: Sequence[str] | None = None,
        split_interest: bool = False,
        chunk_size: int = 10_000,
        loader_type: LoaderType = LoaderType.CSV,
        http: HttpClient | None = None,
    ) -> None:
        super().__init__(loader_type=loader_type)
        self.debt_token = validate_evm_address(debt_token, field="debt_token")
        self.decimals = int(decimals)
        self.start_block = int(start_block)
        self.end_block = end_block
        self.users = sorted({validate_evm_address(u, field="users") for u in users}) if users else None
        self.split_interest = bool(split_interest)
        self.chunk_size = int(chunk_size)
        self._rpc = JsonRpcClient(rpc_url, http=http)
        self._times: dict[int, int] = {}

    def _columns(self) -> list[str]:
        return self._BASE + (self._SPLIT if self.split_interest else [])

    def _cache_key(self) -> str:
        end = self.end_block if self.end_block is not None else "latest"
        digest = hashlib.sha256(",".join(self.users).encode()).hexdigest()[:12] if self.users else ""
        users = f"-{len(self.users)}u{digest}" if self.users else ""
        mode = "-split" if self.split_interest else ""
        return f"{self.debt_token}-flows-{self.start_block}-{end}{users}{mode}"

    # -------------------------------------------------------------- logs
    def _logs(self, topics: list[Any], end_block: int) -> list[dict[str, Any]]:
        while topics and topics[-1] is None:
            topics = topics[:-1]
        return list(self._rpc.iter_logs(self.debt_token, topics, self.start_block, end_block,
                                        chunk_size=self.chunk_size))

    def _time(self, log: dict[str, Any]) -> int:
        block = hex_to_int(log["blockNumber"])
        if block not in self._times:
            stamp = log.get("blockTimestamp")
            self._times[block] = hex_to_int(stamp) if stamp else self._rpc.block_timestamp(block)
        return self._times[block]

    def _row(self, log: dict[str, Any], user_topic: str, sign: int) -> dict[str, Any]:
        return {
            "time": self._time(log),
            "block": hex_to_int(log["blockNumber"]),
            "log_index": hex_to_int(log["logIndex"]),
            "tx": str(log["transactionHash"]).lower(),
            "user": topic_to_address(user_topic),
            "sign": sign,
        }

    def extract(self) -> None:
        end_block = self.end_block if self.end_block is not None else self._rpc.block_number()
        who = ["0x" + encode_address(u) for u in self.users] if self.users else None
        rows: list[dict[str, Any]] = []
        if self.split_interest:
            # Mint(caller, onBehalfOf, value, balanceIncrease, index); Burn(from, target, ...)
            for log in self._logs([MINT_TOPIC, None, who], end_block):
                value, increase = decode_words(log["data"], 2)
                rows.append({**self._row(log, log["topics"][2], 1), "value": value, "increase": increase})
            for log in self._logs([BURN_TOPIC, who], end_block):
                value, increase = decode_words(log["data"], 2)
                rows.append({**self._row(log, log["topics"][1], -1), "value": value, "increase": increase})
        else:
            for topics, side, sign in (([TRANSFER_TOPIC, ZERO_TOPIC, who], 2, 1),
                                       ([TRANSFER_TOPIC, who, ZERO_TOPIC], 1, -1)):
                for log in self._logs(topics, end_block):
                    if int(log["topics"][side], 16) == 0:
                        continue  # zero to zero is neither a mint nor a burn
                    rows.append({**self._row(log, log["topics"][side], sign),
                                 "value": decode_words(log["data"], 1)[0]})
        self._data = pd.DataFrame(rows)

    def transform(self) -> None:
        if self._data is None or self._data.empty:
            self._data = pd.DataFrame(columns=self._columns())
            return
        df = self._data.sort_values(["block", "log_index"]).reset_index(drop=True)
        scale = 10 ** self.decimals
        # Raw amounts can exceed int64: scale as Python ints, then store floats.
        df["amount"] = [s * v / scale for s, v in zip(df["sign"], df["value"], strict=True)]
        if self.split_interest:
            df["interest"] = [v / scale for v in df["increase"]]
            df["principal"] = df["amount"] - df["interest"]
        self._data = df[self._columns()]

    def read(self, with_run: bool = False) -> pd.DataFrame:
        if with_run:
            self.run()
        else:
            self._read(self._cache_key())
        cols = self._columns()
        if self._data is None or self._data.empty:
            empty = pd.DataFrame(columns=cols[1:])
            empty.index = pd.DatetimeIndex([], tz="UTC", name="time")
            return empty
        out = self._data[cols[1:]].copy()
        out.index = self._utc_index("time")
        out["tx"] = out["tx"].astype(str)
        out["user"] = out["user"].astype(str)
        return out
