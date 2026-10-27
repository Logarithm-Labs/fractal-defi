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

1. ``nonces`` — ``(burn tx hash, log index) -> nonce`` supplied by the
   caller, for example from Circle's Iris messages API; matched against
   ``MessageReceived``'s indexed nonce;
2. a fingerprint of the burn message — mint recipient, amount, sender and
   ``maxFee`` — taking the first unused receipt at or after the burn time.

Burns left unmatched (still in flight, or minted after the grace window)
are kept as rows with ``join == "unmatched"``.

The decoders here cover the V2 layouts locally; the bridge decoders in
``fractal.loaders.bridges`` (V1 ``MintAndWithdraw`` and other bridges) live
in a separate change and can be consolidated with these later.
"""
import hashlib
from collections import defaultdict
from collections.abc import Mapping
from datetime import datetime
from typing import Any, ClassVar

import numpy as np
import pandas as pd

from fractal.loaders._dt import to_seconds, to_utc
from fractal.loaders._http import HttpClient
from fractal.loaders.base_loader import Loader, LoaderType
from fractal.loaders.rpc import JsonRpcClient, event_topic, hex_to_int, topic_to_address

__all__ = [
    "MESSAGE_TRANSMITTER_V2",
    "TOKEN_MESSENGER_V2",
    "CctpTransfersLoader",
    "cctp_mode",
    "decode_burn_message_body",
    "decode_deposit_for_burn",
    "decode_message_received",
    "decode_mint_and_withdraw",
    "join_transfers",
]

# Same CREATE2 addresses on every EVM chain CCTP V2 supports.
TOKEN_MESSENGER_V2 = "0x28b5a0e9c621a5badaa536219b3a228c8168cf5d"
MESSAGE_TRANSMITTER_V2 = "0x81d40f21f12a8f0e3252bccb954d722d4c464b64"
TOPIC_DEPOSIT_FOR_BURN = event_topic(
    "DepositForBurn(address,uint256,address,bytes32,uint32,bytes32,bytes32,uint256,uint32,bytes)")
TOPIC_MESSAGE_RECEIVED = event_topic("MessageReceived(address,uint32,bytes32,bytes32,uint32,bytes)")
TOPIC_MINT_AND_WITHDRAW = event_topic("MintAndWithdraw(address,uint256,address,uint256)")
_BODY_MIN_LEN = 228  # BurnMessageV2 up to and including expirationBlock


# ---------------------------------------------------------------- decoding
def _data(log: Mapping[str, Any]) -> bytes:
    raw = str(log.get("data") or "0x")
    return bytes.fromhex(raw[2:] if raw.startswith("0x") else raw)


def _word(data: bytes, index: int) -> int:
    chunk = data[32 * index: 32 * (index + 1)]
    if len(chunk) < 32:
        raise ValueError(f"ABI data too short for word {index}")
    return int.from_bytes(chunk, "big")


def _bytes_at(data: bytes, head_index: int) -> bytes:
    """A dynamic ``bytes`` argument whose offset sits in head word ``head_index``."""
    offset = _word(data, head_index)
    length = int.from_bytes(data[offset: offset + 32], "big")
    if offset + 32 + length > len(data):
        raise ValueError("ABI bytes value runs past the end of data")
    return data[offset + 32: offset + 32 + length]


def _b32(value: int) -> str:
    return "0x" + format(value, "064x")


def _meta(log: Mapping[str, Any]) -> dict[str, Any]:
    return {"tx": str(log["transactionHash"]).lower(), "log_index": hex_to_int(log["logIndex"]),
            "block": hex_to_int(log["blockNumber"])}


def cctp_mode(min_finality: int) -> str:
    """``"fast"`` for a finality threshold <= 1000, ``"standard"`` for >= 2000, else ``"other"``."""
    if min_finality <= 1000:
        return "fast"
    return "standard" if min_finality >= 2000 else "other"


def decode_deposit_for_burn(log: Mapping[str, Any]) -> dict[str, Any]:
    """``DepositForBurn(burnToken idx, amount, depositor idx, mintRecipient, destinationDomain,
    destinationTokenMessenger, destinationCaller, maxFee, minFinalityThreshold idx, hookData)``."""
    topics, data = log["topics"], _data(log)
    return {
        "burn_token": topic_to_address(topics[1]),
        "depositor": topic_to_address(topics[2]),
        "min_finality": hex_to_int(topics[3]),
        "amount": _word(data, 0),
        "mint_recipient": _b32(_word(data, 1)),
        "dst_domain": _word(data, 2),
        "max_fee": _word(data, 5),
        "hook_data": "0x" + _bytes_at(data, 6).hex(),
        **_meta(log),
    }


def decode_burn_message_body(body: bytes) -> dict[str, Any]:
    """BurnMessageV2: version, burnToken, mintRecipient, amount, messageSender, maxFee,
    feeExecuted, expirationBlock, hookData (packed, 32-byte fields after a 4-byte version)."""
    if len(body) < _BODY_MIN_LEN:
        raise ValueError(f"burn message body has {len(body)} bytes, expected at least {_BODY_MIN_LEN}")

    def field(offset: int) -> int:
        return int.from_bytes(body[offset: offset + 32], "big")

    return {
        "burn_token": "0x" + body[16:36].hex(),
        "mint_recipient": _b32(field(36)),
        "amount": field(68),
        "message_sender": "0x" + body[112:132].hex(),
        "max_fee": field(132),
        "fee_executed": field(164),
    }


def decode_message_received(log: Mapping[str, Any]) -> dict[str, Any]:
    """``MessageReceived(caller idx, sourceDomain, nonce idx, sender, finalityThresholdExecuted idx,
    messageBody)``; ``body`` is the decoded burn message, or ``None`` for other messages."""
    topics, data = log["topics"], _data(log)
    body = _bytes_at(data, 2)
    return {
        "caller": topic_to_address(topics[1]),
        "nonce": _b32(hex_to_int(topics[2])),
        "finality_executed": hex_to_int(topics[3]),
        "src_domain": _word(data, 0),
        "sender": _b32(_word(data, 1)),
        "body": decode_burn_message_body(body) if len(body) >= _BODY_MIN_LEN else None,
        **_meta(log),
    }


def decode_mint_and_withdraw(log: Mapping[str, Any]) -> dict[str, Any]:
    """V2 ``MintAndWithdraw(mintRecipient idx, amount, mintToken idx, feeCollected)``."""
    topics, data = log["topics"], _data(log)
    return {"mint_recipient": topic_to_address(topics[1]), "mint_token": topic_to_address(topics[2]),
            "amount": _word(data, 0), "fee_collected": _word(data, 1), **_meta(log)}


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
        self.nonces = dict(nonces or {})
        self.token_decimals = int(token_decimals)
        self.with_gas = bool(with_gas)
        self.chunk_size = int(chunk_size)
        self._src = JsonRpcClient(src_rpc_url, http=src_http)
        self._dst = JsonRpcClient(dst_rpc_url, http=dst_http)

    def _columns(self) -> list[str]:
        return self.COLUMNS + (self.GAS_COLUMNS if self.with_gas else [])

    def _cache_key(self) -> str:
        nonce_tag = ""
        if self.nonces:
            items = sorted(f"{tx}:{i}:{n}" for (tx, i), n in self.nonces.items())
            nonce_tag = "-n" + hashlib.sha256(",".join(items).encode()).hexdigest()[:12]
        gas = "-gas" if self.with_gas else ""
        return (f"cctp-{self.src_domain}-{self.dst_domain}-{to_seconds(self.start_time)}-"
                f"{to_seconds(self.end_time)}-{self.grace_seconds}{nonce_tag}{gas}")

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

    def extract(self) -> None:
        start, end = to_seconds(self.start_time), to_seconds(self.end_time)
        burns: list[dict] = []
        src_blocks = self._range(self._src, start, end)
        if src_blocks is not None:
            decoded = [decode_deposit_for_burn(log)
                       for log in self._logs(self._src, TOKEN_MESSENGER_V2, TOPIC_DEPOSIT_FOR_BURN, src_blocks)]
            burns = self._timed(self._src, [b for b in decoded if b["dst_domain"] == self.dst_domain], {})
        received: list[dict] = []
        dst_blocks = self._range(self._dst, start, end + self.grace_seconds) if burns else None
        if dst_blocks is not None:
            received = [r for r in (decode_message_received(log) for log in self._logs(
                self._dst, MESSAGE_TRANSMITTER_V2, TOPIC_MESSAGE_RECEIVED, dst_blocks))
                if r["src_domain"] == self.src_domain]
            mints = [decode_mint_and_withdraw(log)
                     for log in self._logs(self._dst, TOKEN_MESSENGER_V2, TOPIC_MINT_AND_WITHDRAW, dst_blocks)]
            _pair_mints(received, mints)
            self._timed(self._dst, received, {})
        matched, unmatched = join_transfers(burns, received, self.nonces)
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
