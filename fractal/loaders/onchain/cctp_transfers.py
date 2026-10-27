"""Realised CCTP V2 transfers: source burns joined to destination mints on chain.

A CCTP V2 transfer burns USDC on the source chain (``TokenMessengerV2``
``DepositForBurn``) and mints it on the destination chain
(``MessageTransmitterV2`` ``MessageReceived`` plus the ``TokenMessengerV2``
``MintAndWithdraw`` of the same transaction). Reading both sides gives what
a transfer actually cost and how long it took: the fee collected on the
mint (``feeCollected``, else the burn message's ``feeExecuted``), the mode
(``minFinalityThreshold`` <= 1000 is Fast, >= 2000 Standard) and the
latency between the two block times.

The source event carries no nonce (CCTP V2 assigns it off chain), so the
join uses, in order:

1. Iris nonces — either ``iris_messages`` (rows of
   :class:`~fractal.loaders.bridges.cctp.CctpMessagesLoader`, one frame per
   source transaction or concatenated), assigned to the burns of each
   transaction in log order by amount and destination
   (:func:`assign_iris_nonces`), or an explicit ``(burn tx hash, log index)
   -> nonce`` map, which wins; matched against ``MessageReceived``'s
   indexed nonce;
2. a fingerprint of the burn message — mint recipient, amount, sender and
   ``maxFee`` — taking the first unused receipt at or after the burn time.

Burns left unmatched (still in flight, or minted after the grace window)
are kept as rows with ``join == "unmatched"``.

With ``targeted=True`` the destination is not scanned in full: only the
``MessageReceived`` logs whose indexed nonce is one of the burns' Iris
nonces are requested (topic filter), and the ``MintAndWithdraw`` of each
hit is read from its transaction receipt. Burns without an Iris nonce then
stay unmatched.

Events are decoded with :mod:`fractal.loaders.bridges.decoders`.
"""
import hashlib
from collections import defaultdict
from collections.abc import Iterable, Mapping
from datetime import datetime
from typing import Any, ClassVar

import numpy as np
import pandas as pd

from fractal.loaders._dt import to_seconds, to_utc
from fractal.loaders._http import HttpClient
from fractal.loaders.base_loader import Loader, LoaderType
from fractal.loaders.bridges.decoders import (
    TOPIC_CCTP_V2_DEPOSIT_FOR_BURN,
    TOPIC_CCTP_V2_MESSAGE_RECEIVED,
    TOPIC_CCTP_V2_MINT_AND_WITHDRAW,
    decode_cctp_v2_deposit_for_burn,
    decode_cctp_v2_message_received,
    decode_cctp_v2_mint_and_withdraw,
)
from fractal.loaders.rpc import JsonRpcClient, hex_to_int

__all__ = [
    "MESSAGE_TRANSMITTER_V2",
    "TOKEN_MESSENGER_V2",
    "CctpTransfersLoader",
    "assign_iris_nonces",
    "cctp_mode",
    "join_transfers",
]

# Same CREATE2 addresses on every EVM chain CCTP V2 supports.
TOKEN_MESSENGER_V2 = "0x28b5a0e9c621a5badaa536219b3a228c8168cf5d"
MESSAGE_TRANSMITTER_V2 = "0x81d40f21f12a8f0e3252bccb954d722d4c464b64"


def _b32(value: int) -> str:
    return "0x" + format(value, "064x")


def _with_meta(decoded: dict[str, Any], log: Mapping[str, Any]) -> dict[str, Any]:
    return {**decoded, "tx": str(log["transactionHash"]).lower(), "log_index": hex_to_int(log["logIndex"]),
            "block": hex_to_int(log["blockNumber"])}


def cctp_mode(min_finality: int) -> str:
    """``"fast"`` for a finality threshold <= 1000, ``"standard"`` for >= 2000, else ``"other"``."""
    if min_finality <= 1000:
        return "fast"
    return "standard" if min_finality >= 2000 else "other"


def _burn(log: Mapping[str, Any]) -> dict[str, Any]:
    return _with_meta(decode_cctp_v2_deposit_for_burn(log["topics"], log["data"]), log)


def _received(log: Mapping[str, Any]) -> dict[str, Any]:
    return _with_meta(decode_cctp_v2_message_received(log["topics"], log["data"]), log)


def _mint(log: Mapping[str, Any]) -> dict[str, Any]:
    return _with_meta(decode_cctp_v2_mint_and_withdraw(log["topics"], log["data"]), log)


def _pair_mints(received: list[dict], mints: list[dict]) -> None:
    """Attach to each receipt the nearest earlier unused ``MintAndWithdraw`` of its transaction."""
    by_tx: dict[str, list[dict]] = defaultdict(list)
    for mint in mints:
        by_tx[mint["tx"]].append(mint)
    used: set[int] = set()
    for rec in sorted(received, key=lambda r: (r["tx"], r["log_index"])):
        pool = [m for m in by_tx.get(rec["tx"], []) if m["log_index"] < rec["log_index"] and id(m) not in used]
        rec["mint"] = max(pool, key=lambda m: m["log_index"]) if pool else None
        if rec["mint"] is not None:
            used.add(id(rec["mint"]))


def _fingerprint(recipient: str, amount: int, sender: str, max_fee: int) -> tuple:
    return recipient.lower(), int(amount), sender.lower()[-40:], int(max_fee)


def join_transfers(
    burns: list[dict],
    received: list[dict],
    nonces: Mapping[tuple[str, int], str] | None = None,
) -> tuple[list[tuple[dict, dict, str]], list[dict]]:
    """Join burns to receipts: by caller-supplied nonce first, then by burn-message fingerprint.

    ``burns`` / ``received`` carry a block time ``ts``. Returns
    ``([(burn, receipt, "nonce" | "fingerprint"), ...], unmatched burns)``.
    """
    known = {(str(tx).lower(), int(i)): _b32(hex_to_int(n)) for (tx, i), n in (nonces or {}).items()}
    by_nonce = {r["nonce"]: r for r in received}
    by_fp: dict[tuple, list[dict]] = defaultdict(list)
    for rec in sorted(received, key=lambda r: (r["ts"], r["log_index"])):
        if rec.get("body"):
            body = rec["body"]
            by_fp[_fingerprint(body["mint_recipient"], body["amount"], body["message_sender"],
                               body["max_fee"])].append(rec)
    matched, pending, used = [], [], set()
    ordered = sorted(burns, key=lambda b: (b["ts"], b["block"], b["log_index"]))
    for burn in ordered:  # nonce joins claim their receipts first
        rec = by_nonce.get(known.get((burn["tx"], burn["log_index"]), ""))
        if rec is not None and id(rec) not in used:
            used.add(id(rec))
            matched.append((burn, rec, "nonce"))
        else:
            pending.append(burn)
    unmatched = []
    for burn in pending:
        key = _fingerprint(burn["mint_recipient"], burn["amount"], burn["depositor"], burn["max_fee"])
        rec = next((r for r in by_fp.get(key, []) if id(r) not in used and r["ts"] >= burn["ts"]), None)
        if rec is None:
            unmatched.append(burn)
            continue
        used.add(id(rec))
        matched.append((burn, rec, "fingerprint"))
    return matched, unmatched


def _missing(value: Any) -> bool:
    return value is None or (not isinstance(value, str) and bool(pd.isna(value)))


def assign_iris_nonces(
    burns: Iterable[Mapping[str, Any]],
    messages: pd.DataFrame | Iterable[Mapping[str, Any]],
) -> dict[tuple[str, int], str]:
    """Assign Iris message nonces to on-chain burns: ``{(burn tx, log index): nonce}``.

    ``messages`` are :class:`~fractal.loaders.bridges.cctp.CctpMessagesLoader`
    rows (``tx_hash``, ``nonce``, ``amount_raw``, ``dst_domain``) in the order
    Iris returns them. Within each transaction, burns are taken in log order
    and each gets the first unused message whose amount and destination
    agree (a missing field matches anything). Messages without a nonce yet
    are skipped. Pass every burn of a transaction, whatever its destination,
    so the log-order assignment stays aligned.
    """
    rows = messages.to_dict("records") if isinstance(messages, pd.DataFrame) else list(messages)
    by_tx: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if not _missing(row.get("nonce")) and not _missing(row.get("tx_hash")):
            by_tx[str(row["tx_hash"]).lower()].append(row)
    out: dict[tuple[str, int], str] = {}
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for burn in burns:
        grouped[str(burn["tx"]).lower()].append(burn)
    for tx, group in grouped.items():
        free = list(by_tx.get(tx, []))
        for burn in sorted(group, key=lambda b: int(b["log_index"])):
            hit = next((m for m in free
                        if (_missing(m.get("amount_raw")) or int(m["amount_raw"]) == int(burn["amount"]))
                        and (_missing(m.get("dst_domain")) or int(m["dst_domain"]) == int(burn["dst_domain"]))),
                       None)
            if hit is not None:
                free.remove(hit)
                out[(tx, int(burn["log_index"]))] = _b32(hex_to_int(hit["nonce"]))
    return out


# ------------------------------------------------------------------ loader
class CctpTransfersLoader(Loader):
    """Realised CCTP V2 transfers on one route with burns in ``[start_time, end_time)``.

    Destination mints are scanned until ``end_time + grace_seconds`` (capped
    at the destination tip). ``read`` returns one row per burn with
    ``burn_time`` / ``mint_time`` (UTC), ``nonce``, ``amount`` (burned) and
    ``amount_received`` (minted to the recipient) and ``fee`` / ``max_fee``
    in token units, ``min_finality``, ``mode``, ``finality_executed``,
    ``latency_s``, ``join`` (``nonce`` / ``fingerprint`` / ``unmatched``),
    ``burn_tx``, ``burn_log_index``, ``mint_tx`` and, with ``with_gas``,
    ``burn_gas_native`` / ``mint_gas_native`` (whole native tokens; price
    them with a caller-supplied series).

    Args:
        src_rpc_url / dst_rpc_url: JSON-RPC endpoints of the two chains
            (``eth_getLogs`` capable; keys in the URL are redacted, never cached).
        src_domain / dst_domain: CCTP domain ids of the two chains.
        start_time / end_time: burn window (naive = UTC).
        grace_seconds: how long after ``end_time`` to look for mints.
        nonces: optional ``(burn tx hash, log index) -> nonce`` map.
        iris_messages: optional ``CctpMessagesLoader`` rows for the window's
            burn transactions (see :func:`assign_iris_nonces`).
        targeted: look up destination receipts by Iris nonce only.
        token_decimals: decimals of the burned token (USDC: 6).
        with_gas: read both legs' receipts for native gas.
    """

    COLUMNS: ClassVar[list[str]] = [
        "burn_time", "mint_time", "src_domain", "dst_domain", "nonce", "amount", "amount_received", "fee",
        "max_fee", "min_finality", "mode", "finality_executed", "latency_s", "join", "burn_tx", "burn_log_index",
        "mint_tx",
    ]
    GAS_COLUMNS: ClassVar[list[str]] = ["burn_gas_native", "mint_gas_native"]

    def __init__(
        self,
        src_rpc_url: str,
        src_domain: int,
        dst_rpc_url: str,
        dst_domain: int,
        start_time: datetime,
        end_time: datetime,
        grace_seconds: int = 6 * 3_600,
        nonces: Mapping[tuple[str, int], str] | None = None,
        iris_messages: pd.DataFrame | Iterable[Mapping[str, Any]] | None = None,
        targeted: bool = False,
        token_decimals: int = 6,
        with_gas: bool = False,
        chunk_size: int = 10_000,
        loader_type: LoaderType = LoaderType.CSV,
        src_http: HttpClient | None = None,
        dst_http: HttpClient | None = None,
    ) -> None:
        super().__init__(loader_type=loader_type)
        self.src_domain, self.dst_domain = int(src_domain), int(dst_domain)
        self.start_time, self.end_time = to_utc(start_time), to_utc(end_time)
        if self.end_time <= self.start_time:
            raise ValueError("end_time must be after start_time")
        self.grace_seconds = int(grace_seconds)
        self.nonces = {(str(tx).lower(), int(i)): _b32(hex_to_int(n)) for (tx, i), n in (nonces or {}).items()}
        if iris_messages is None:
            self.iris_messages: list[dict[str, Any]] = []
        elif isinstance(iris_messages, pd.DataFrame):
            self.iris_messages = iris_messages.to_dict("records")
        else:
            self.iris_messages = [dict(m) for m in iris_messages]
        self.targeted = bool(targeted)
        self.token_decimals = int(token_decimals)
        self.with_gas = bool(with_gas)
        self.chunk_size = int(chunk_size)
        self._src = JsonRpcClient(src_rpc_url, http=src_http)
        self._dst = JsonRpcClient(dst_rpc_url, http=dst_http)

    def _columns(self) -> list[str]:
        return self.COLUMNS + (self.GAS_COLUMNS if self.with_gas else [])

    def _cache_key(self) -> str:
        nonce_tag = ""
        if self.nonces or self.iris_messages:
            items = sorted(f"{tx}:{i}:{n}" for (tx, i), n in self.nonces.items())
            items += sorted("iris:" + ":".join(str(m.get(k)) for k in ("tx_hash", "nonce", "amount_raw", "dst_domain"))
                            for m in self.iris_messages)
            nonce_tag = "-n" + hashlib.sha256(",".join(items).encode()).hexdigest()[:12]
        gas = "-gas" if self.with_gas else ""
        mode = "-targeted" if self.targeted else ""
        return (f"cctp-{self.src_domain}-{self.dst_domain}-{to_seconds(self.start_time)}-"
                f"{to_seconds(self.end_time)}-{self.grace_seconds}{nonce_tag}{gas}{mode}")

    @staticmethod
    def _range(rpc: JsonRpcClient, start: int, end: int) -> tuple[int, int] | None:
        """Blocks with time in ``[start, end)``, capped at the tip; ``None`` when empty."""
        tip = rpc.block_number()
        if rpc.block_timestamp(tip) < start:
            return None
        lo = rpc.block_at_timestamp(start, closest="after", high=tip)
        hi = rpc.block_at_timestamp(end - 1, closest="before", high=tip) if end - 1 >= start else lo - 1
        return (lo, hi) if hi >= lo else None

    @staticmethod
    def _timed(rpc: JsonRpcClient, logs: list[dict], times: dict[int, int]) -> list[dict]:
        for log in logs:
            if log["block"] not in times:
                times[log["block"]] = rpc.block_timestamp(log["block"])
            log["ts"] = times[log["block"]]
        return logs

    def _logs(self, rpc: JsonRpcClient, address: str, topic: str, blocks: tuple[int, int]) -> list[dict]:
        return list(rpc.iter_logs(address, [topic], blocks[0], blocks[1], chunk_size=self.chunk_size))

    def _received_targeted(self, nonces: list[str], blocks: tuple[int, int]) -> list[dict]:
        """``MessageReceived`` logs with one of ``nonces``; mints from each hit's receipt."""
        received = []
        for i in range(0, len(nonces), 100):  # OR list per topic position, in batches
            topics = [TOPIC_CCTP_V2_MESSAGE_RECEIVED, None, nonces[i:i + 100]]
            received += [_received(log) for log in self._dst.iter_logs(
                MESSAGE_TRANSMITTER_V2, topics, blocks[0], blocks[1], chunk_size=self.chunk_size)]
        mints = []
        for tx in dict.fromkeys(r["tx"] for r in received):
            logs = self._dst.get_transaction_receipt(tx).get("logs") or []
            mints += [_mint(log) for log in logs
                      if str(log.get("address", "")).lower() == TOKEN_MESSENGER_V2 and log.get("topics")
                      and str(log["topics"][0]).lower() == TOPIC_CCTP_V2_MINT_AND_WITHDRAW]
        _pair_mints(received, mints)
        return received

    def extract(self) -> None:
        start, end = to_seconds(self.start_time), to_seconds(self.end_time)
        burns: list[dict] = []
        nonces = dict(self.nonces)
        src_blocks = self._range(self._src, start, end)
        if src_blocks is not None:
            decoded = [_burn(log) for log in
                       self._logs(self._src, TOKEN_MESSENGER_V2, TOPIC_CCTP_V2_DEPOSIT_FOR_BURN, src_blocks)]
            # every burn of a transaction takes part in the log-order assignment, whatever its destination
            nonces = {**assign_iris_nonces(decoded, self.iris_messages), **nonces}
            burns = self._timed(self._src, [b for b in decoded if b["dst_domain"] == self.dst_domain], {})
        received: list[dict] = []
        dst_blocks = self._range(self._dst, start, end + self.grace_seconds) if burns else None
        if dst_blocks is not None and self.targeted:
            wanted = sorted({nonces[(b["tx"], b["log_index"])] for b in burns if (b["tx"], b["log_index"]) in nonces})
            received = self._received_targeted(wanted, dst_blocks) if wanted else []
        elif dst_blocks is not None:
            received = [_received(log) for log in
                        self._logs(self._dst, MESSAGE_TRANSMITTER_V2, TOPIC_CCTP_V2_MESSAGE_RECEIVED, dst_blocks)]
            mints = [_mint(log) for log in
                     self._logs(self._dst, TOKEN_MESSENGER_V2, TOPIC_CCTP_V2_MINT_AND_WITHDRAW, dst_blocks)]
            _pair_mints(received, mints)
        received = self._timed(self._dst, [r for r in received if r["src_domain"] == self.src_domain], {})
        matched, unmatched = join_transfers(burns, received, nonces)
        rows = [self._row(burn, rec, how) for burn, rec, how in matched]
        rows += [self._row(burn, None, "unmatched") for burn in unmatched]
        self._data = pd.DataFrame(rows, columns=self._columns())

    def _row(self, burn: dict, rec: dict | None, how: str) -> dict[str, Any]:
        scale = 10 ** self.token_decimals
        row: dict[str, Any] = {
            "burn_time": burn["ts"], "src_domain": self.src_domain, "dst_domain": self.dst_domain,
            "amount": burn["amount"] / scale, "max_fee": burn["max_fee"] / scale,
            "min_finality": burn["min_finality"], "mode": cctp_mode(burn["min_finality"]), "join": how,
            "burn_tx": burn["tx"], "burn_log_index": burn["log_index"],
            "mint_time": np.nan, "nonce": None, "amount_received": np.nan, "fee": np.nan,
            "finality_executed": np.nan, "latency_s": np.nan, "mint_tx": None,
        }
        if rec is not None:
            mint = rec.get("mint")
            fee = mint["fee_collected"] if mint else (rec["body"] or {}).get("fee_executed")
            row.update({
                "mint_time": rec["ts"], "nonce": rec["nonce"], "finality_executed": rec["finality_executed"],
                "latency_s": rec["ts"] - burn["ts"], "mint_tx": rec["tx"],
                "amount_received": mint["amount"] / scale if mint else np.nan,
                "fee": fee / scale if fee is not None else np.nan,
            })
        if self.with_gas:
            row["burn_gas_native"] = self._src.transaction_gas_cost(burn["tx"]) / 10 ** 18
            row["mint_gas_native"] = self._dst.transaction_gas_cost(rec["tx"]) / 10 ** 18 if rec else np.nan
        return row

    def transform(self) -> None:
        if self._data is None or self._data.empty:
            self._data = pd.DataFrame(columns=self._columns())
            return
        self._data = self._data.sort_values(["burn_time", "burn_tx", "burn_log_index"]).reset_index(drop=True)

    def read(self, with_run: bool = False) -> pd.DataFrame:
        if with_run:
            self.run()
        else:
            self._read(self._cache_key())
        df = (self._data if self._data is not None else pd.DataFrame(columns=self._columns()))[self._columns()]
        df = df.copy()
        for col in ("burn_time", "mint_time"):
            df[col] = pd.to_datetime(pd.to_numeric(df[col], errors="coerce"), unit="s", utc=True)
        for col in ("nonce", "mint_tx", "burn_tx", "mode", "join"):
            df[col] = df[col].astype(object).where(df[col].notna(), None)
        return df.reset_index(drop=True)
