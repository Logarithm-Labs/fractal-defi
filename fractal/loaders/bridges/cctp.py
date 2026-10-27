"""Circle Iris V2 keyless CCTP endpoints: fees, Fast allowance and messages.

``CCTP_DOMAINS`` is the documented chain-name to CCTP domain-ID map for
the chains covered by the bridge-cost reference collector.
"""

import re
from typing import Any

import pandas as pd

from fractal.loaders._http import HttpClient, LoaderHttpError
from fractal.loaders.base_loader import Loader, LoaderType

IRIS_API = "https://iris-api.circle.com"
CCTP_DOMAINS = {
    "ethereum": 0,
    "optimism": 2,
    "arbitrum": 3,
    "base": 6,
    "polygon": 7,
    "unichain": 10,
    "monad": 15,
    "hyperevm": 19,
    "arc": 26,
    "plasma": 33,
}
CCTP_DOMAIN_CHAINS = {domain: chain for chain, domain in CCTP_DOMAINS.items()}
_CACHE_SCHEMA = "v1"
_ZERO_NONCE = "0x" + "0" * 64
_TX_HASH = re.compile(r"^(0x)?[0-9A-Za-z]+$")


def _is_404(error: LoaderHttpError) -> bool:
    return "HTTP 404" in str(error)


def _int(value: Any, field: str) -> int | None:
    """Int from an int or a hex/decimal string; ``None`` when absent."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    text = str(value).strip()
    if text in ("", "0x"):
        return None
    try:
        return int(text, 16) if text.lower().startswith("0x") else int(text)
    except ValueError as exc:
        raise ValueError(f"invalid Circle Iris integer field {field!r}: {value!r}") from exc


def _text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text.lower() if text.lower().startswith("0x") else (text or None)


def _mode(finality_threshold: int) -> str | None:
    if finality_threshold <= 1000:
        return "fast"
    if finality_threshold >= 2000:
        return "standard"
    return None


class CctpFeeLoader(Loader):
    """Published Fast and Standard USDC transfer fees for one CCTP route.

    ``minimum_fee_bps`` is Iris ``minimumFee`` in basis points. Entries with
    ``finalityThreshold <= 1000`` are Fast and ``>= 2000`` Standard; entries
    strictly between 1000 and 2000 match neither documented mode and are
    dropped.

    Args:
        source_domain: Circle CCTP source domain ID.
        destination_domain: Circle CCTP destination domain ID.
        loader_type: Cache format.
        http: Injectable HTTP client for offline tests.
    """

    _columns = (
        "src_domain",
        "dst_domain",
        "mode",
        "minimum_fee_bps",
        "finality_threshold",
    )

    def __init__(
        self,
        source_domain: int,
        destination_domain: int,
        *,
        loader_type: LoaderType = LoaderType.CSV,
        http: HttpClient | None = None,
    ) -> None:
        super().__init__(loader_type=loader_type)
        if int(source_domain) < 0 or int(destination_domain) < 0:
            raise ValueError("source_domain and destination_domain must be non-negative")
        self.source_domain = int(source_domain)
        self.destination_domain = int(destination_domain)
        self._http = http or HttpClient()

    def _cache_key(self) -> str:
        return f"{_CACHE_SCHEMA}-{self.source_domain}-{self.destination_domain}"

    def extract(self) -> None:
        """Fetch the current fee schedule; an unknown 404 route is empty."""
        url = f"{IRIS_API}/v2/burn/USDC/fees/{self.source_domain}/{self.destination_domain}"
        try:
            self._data = self._http.get(url)
        except LoaderHttpError as exc:
            if not _is_404(exc):
                raise
            self._data = []

    @classmethod
    def _frame(cls, rows: Any) -> pd.DataFrame:
        frame = pd.DataFrame(rows, columns=cls._columns)
        if frame.empty:
            return frame
        frame["src_domain"] = pd.to_numeric(frame["src_domain"], errors="raise").astype("int64")
        frame["dst_domain"] = pd.to_numeric(frame["dst_domain"], errors="raise").astype("int64")
        frame["minimum_fee_bps"] = pd.to_numeric(frame["minimum_fee_bps"], errors="raise").astype(float)
        frame["finality_threshold"] = pd.to_numeric(frame["finality_threshold"], errors="raise").astype("int64")
        return frame

    def transform(self) -> None:
        """Parse Iris fee entries into one row per supported transfer mode."""
        payload = self._data
        rows = payload if isinstance(payload, list) else (payload or {}).get("data", [])
        if not isinstance(rows, list):
            raise ValueError("Circle Iris fee response must be a list or an object with a 'data' list")
        parsed = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            try:
                finality_threshold = int(row["finalityThreshold"])
                mode = _mode(finality_threshold)
                minimum_fee_bps = float(row["minimumFee"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"invalid Circle Iris fee entry: {row!r}") from exc
            if mode is not None:
                parsed.append({
                    "src_domain": self.source_domain,
                    "dst_domain": self.destination_domain,
                    "mode": mode,
                    "minimum_fee_bps": minimum_fee_bps,
                    "finality_threshold": finality_threshold,
                })
        parsed.sort(key=lambda row: row["finality_threshold"])
        self._data = self._frame(parsed)

    def read(self, with_run: bool = False) -> pd.DataFrame:
        """Return the route's published Fast and Standard fee schedule."""
        if with_run:
            self.run()
        else:
            self._read(self._cache_key())
        return self._frame(self._data).copy()


class CctpFastAllowanceLoader(Loader):
    """Circle Iris V2 remaining USDC Fast Transfer allowance (a snapshot).

    The allowance is the USDC amount that can still be sent with Fast
    Transfer before Circle's cap; one row with ``allowance_usdc`` and the
    provider's ``last_updated`` time (UTC, missing if not reported). A cached
    read returns the snapshot taken by the last run.

    Args:
        loader_type: Cache format.
        http: Injectable HTTP client for offline tests.
    """

    _columns = ("allowance_usdc", "last_updated")

    def __init__(self, *, loader_type: LoaderType = LoaderType.CSV, http: HttpClient | None = None) -> None:
        super().__init__(loader_type=loader_type)
        self._http = http or HttpClient()

    def _cache_key(self) -> str:
        return f"{_CACHE_SCHEMA}-usdc"

    def extract(self) -> None:
        """Fetch the current Fast Transfer allowance."""
        self._data = self._http.get(f"{IRIS_API}/v2/fastBurn/USDC/allowance")

    @classmethod
    def _frame(cls, rows: Any) -> pd.DataFrame:
        frame = pd.DataFrame(rows, columns=cls._columns)
        frame["allowance_usdc"] = pd.to_numeric(frame["allowance_usdc"], errors="raise").astype(float)
        frame["last_updated"] = pd.to_datetime(frame["last_updated"], utc=True, errors="coerce")
        return frame

    def transform(self) -> None:
        """Parse the allowance object into one row."""
        payload = self._data
        if not isinstance(payload, dict) or payload.get("allowance") is None:
            raise ValueError("Circle Iris allowance response must be an object with an 'allowance' field")
        try:
            allowance = float(payload["allowance"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid Circle Iris allowance: {payload['allowance']!r}") from exc
        self._data = self._frame([{"allowance_usdc": allowance, "last_updated": payload.get("lastUpdated")}])

    def read(self, with_run: bool = False) -> pd.DataFrame:
        """Return the Fast Transfer allowance snapshot."""
        if with_run:
            self.run()
        else:
            self._read(self._cache_key())
        return self._frame(self._data).copy()


class CctpMessagesLoader(Loader):
    """Circle Iris V2 CCTP messages emitted by one source transaction.

    One row per message in the order Iris returns them (source log order),
    without the raw message or attestation bytes. ``nonce`` is the 32-byte
    hex nonce Iris assigns (missing until assigned; the on-chain V2 burn
    message carries a zero nonce), which identifies the destination
    ``MessageReceived`` event. Integer amounts are in the burn token's base
    units (USDC: 6 decimals). Status reflects the time of the fetch, so a
    cached read can be stale for pending messages. A transaction Iris does
    not know (HTTP 404) yields an empty frame.

    Args:
        source_domain: Circle CCTP source domain ID.
        transaction_hash: Source transaction hash or signature.
        loader_type: Cache format.
        http: Injectable HTTP client for offline tests.
    """

    _columns = (
        "src_domain",
        "dst_domain",
        "tx_hash",
        "nonce",
        "status",
        "cctp_version",
        "amount_raw",
        "max_fee_raw",
        "fee_executed_raw",
        "min_finality_threshold",
        "finality_threshold_executed",
        "sender",
        "recipient",
        "mint_recipient",
        "message_sender",
        "burn_token",
        "delay_reason",
    )
    _int_columns = (
        "src_domain",
        "dst_domain",
        "cctp_version",
        "amount_raw",
        "max_fee_raw",
        "fee_executed_raw",
        "min_finality_threshold",
        "finality_threshold_executed",
    )

    def __init__(
        self,
        source_domain: int,
        transaction_hash: str,
        *,
        loader_type: LoaderType = LoaderType.CSV,
        http: HttpClient | None = None,
    ) -> None:
        super().__init__(loader_type=loader_type)
        if int(source_domain) < 0:
            raise ValueError("source_domain must be non-negative")
        tx_hash = str(transaction_hash).strip()
        if not _TX_HASH.match(tx_hash):
            raise ValueError(f"transaction_hash must be a hex hash or an alphanumeric signature, got {tx_hash!r}")
        self.source_domain = int(source_domain)
        self.transaction_hash = tx_hash.lower() if tx_hash.lower().startswith("0x") else tx_hash
        self._http = http or HttpClient()

    def _cache_key(self) -> str:
        return f"{_CACHE_SCHEMA}-{self.source_domain}-{self.transaction_hash}"

    def extract(self) -> None:
        """Fetch the transaction's messages; an unknown transaction (404) is empty."""
        url = f"{IRIS_API}/v2/messages/{self.source_domain}"
        try:
            self._data = self._http.get(url, params={"transactionHash": self.transaction_hash})
        except LoaderHttpError as exc:
            if not _is_404(exc):
                raise
            self._data = {"messages": []}

    def _parse(self, message: dict[str, Any]) -> dict[str, Any]:
        decoded = message.get("decodedMessage") or {}
        body = decoded.get("decodedMessageBody") or {}
        raw_nonce = decoded.get("nonce")
        if raw_nonce in (None, "", "0x"):
            raw_nonce = message.get("eventNonce")
        nonce = _int(raw_nonce, "nonce")
        nonce_hex = None if nonce is None else "0x" + format(nonce, "064x")
        source_domain = _int(decoded.get("sourceDomain"), "sourceDomain")
        return {
            "src_domain": self.source_domain if source_domain is None else source_domain,
            "dst_domain": _int(decoded.get("destinationDomain"), "destinationDomain"),
            "tx_hash": self.transaction_hash,
            "nonce": None if nonce_hex == _ZERO_NONCE else nonce_hex,
            "status": message.get("status"),
            "cctp_version": _int(message.get("cctpVersion"), "cctpVersion"),
            "amount_raw": _int(body.get("amount"), "amount"),
            "max_fee_raw": _int(body.get("maxFee"), "maxFee"),
            "fee_executed_raw": _int(body.get("feeExecuted"), "feeExecuted"),
            "min_finality_threshold": _int(decoded.get("minFinalityThreshold"), "minFinalityThreshold"),
            "finality_threshold_executed": _int(
                decoded.get("finalityThresholdExecuted"), "finalityThresholdExecuted"
            ),
            "sender": _text(decoded.get("sender")),
            "recipient": _text(decoded.get("recipient")),
            "mint_recipient": _text(body.get("mintRecipient")),
            "message_sender": _text(body.get("messageSender")),
            "burn_token": _text(body.get("burnToken")),
            "delay_reason": message.get("delayReason"),
        }

    @classmethod
    def _frame(cls, rows: Any) -> pd.DataFrame:
        frame = pd.DataFrame(rows, columns=cls._columns)
        for column in cls._int_columns:
            frame[column] = pd.array(pd.to_numeric(frame[column], errors="raise"), dtype="Int64")
        for column in set(cls._columns) - set(cls._int_columns):
            frame[column] = frame[column].astype(object).where(frame[column].notna(), None)
        return frame

    def transform(self) -> None:
        """Parse Iris messages into one row each, without attestation bytes."""
        payload = self._data
        messages = payload.get("messages", []) if isinstance(payload, dict) else payload
        if not isinstance(messages, list):
            raise ValueError("Circle Iris messages response must carry a 'messages' list")
        rows = [self._parse(message) for message in messages if isinstance(message, dict)]
        self._data = self._frame(rows)

    def read(self, with_run: bool = False) -> pd.DataFrame:
        """Return the source transaction's CCTP messages."""
        if with_run:
            self.run()
        else:
            self._read(self._cache_key())
        return self._frame(self._data).copy()
