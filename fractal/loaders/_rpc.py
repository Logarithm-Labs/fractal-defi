"""Minimal JSON-RPC ``eth_call`` helper shared by loaders that need one
on-chain read (a Pendle market's ``readState``, a Morpho oracle's
``baseDiscountPerYear``). Goes through :class:`HttpClient` like every
other transport in the package; no web3 dependency.
"""
from typing import Any, List, Optional

from fractal.loaders._http import HttpClient

__all__ = ["RpcCallError", "eth_call", "decode_words", "as_signed"]


class RpcCallError(RuntimeError):
    """JSON-RPC returned an error or a malformed payload."""


def eth_call(
    rpc_url: str,
    to: str,
    data: str,
    *,
    block: str = "latest",
    http: Optional[HttpClient] = None,
) -> str:
    """``eth_call`` and return the raw ``0x…`` hex result."""
    client = http or HttpClient()
    payload = client.post(rpc_url, json={
        "jsonrpc": "2.0", "id": 1, "method": "eth_call",
        "params": [{"to": to, "data": data}, block],
    })
    if not isinstance(payload, dict) or "error" in payload or "result" not in payload:
        raise RpcCallError(f"eth_call to {to} failed: {payload!r}")
    result: Any = payload["result"]
    if not isinstance(result, str) or not result.startswith("0x"):
        raise RpcCallError(f"eth_call to {to} returned a non-hex result: {result!r}")
    return result


def decode_words(hex_result: str, count: int) -> List[int]:
    """Split an ABI-encoded static return into ``count`` unsigned 256-bit words."""
    body = hex_result[2:]
    if len(body) < 64 * count:
        raise RpcCallError(f"expected {count} return words, got {len(body) // 64}")
    return [int(body[64 * i: 64 * (i + 1)], 16) for i in range(count)]


def as_signed(word: int, bits: int = 256) -> int:
    """Two's-complement reinterpretation of an unsigned ABI word."""
    return word - (1 << bits) if word >= (1 << (bits - 1)) else word
