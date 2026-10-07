"""JSON-RPC client and ABI helpers shared by every on-chain loader.

One transport for the three things a backtest reads from a node:

* a static call (``eth_call``: a Pendle market's ``readState``, a Morpho
  oracle's ``baseDiscountPerYear``, any view function);
* event logs over a block range (``eth_getLogs`` with adaptive chunking:
  Uniswap ``Swap``, Aave ``Supply`` / ``Borrow``, Morpho ``Borrow``, ...);
* block headers for timestamps.

No web3 dependency: requests go through :class:`HttpClient` like every
other transport in the package, and the ABI helpers cover the static
types a loader needs (words, signed ints, addresses, event topics and
function selectors from their signatures via a built-in keccak-256).

Example — every ``Supply`` of one Aave V3 pool::

    rpc = JsonRpcClient("https://mainnet.example/rpc")
    topic = event_topic("Supply(address,address,address,uint256,uint16)")
    for log in rpc.iter_logs(POOL, [topic], from_block=21_000_000, to_block=21_100_000):
        reserve, on_behalf_of = topic_to_address(log["topics"][1]), topic_to_address(log["topics"][3])
        amount, referral = decode_words(log["data"], 2)
"""
from typing import Any, Dict, Iterator, List, Optional, Sequence, Union

from fractal.loaders._http import HttpClient, LoaderHttpError

__all__ = [
    "RpcCallError",
    "JsonRpcClient",
    "eth_call",
    "decode_words",
    "as_signed",
    "decode_address",
    "topic_to_address",
    "topic_to_int",
    "keccak256",
    "event_topic",
    "function_selector",
    "encode_address",
]

BlockTag = Union[int, str]


class RpcCallError(RuntimeError):
    """JSON-RPC returned an error or a malformed payload."""


# ------------------------------------------------------------------ keccak
_MASK = (1 << 64) - 1
_ROUND_CONSTANTS = (
    0x0000000000000001, 0x0000000000008082, 0x800000000000808A, 0x8000000080008000,
    0x000000000000808B, 0x0000000080000001, 0x8000000080008081, 0x8000000000008009,
    0x000000000000008A, 0x0000000000000088, 0x0000000080008009, 0x000000008000000A,
    0x000000008000808B, 0x800000000000008B, 0x8000000000008089, 0x8000000000008003,
    0x8000000000008002, 0x8000000000000080, 0x000000000000800A, 0x800000008000000A,
    0x8000000080008081, 0x8000000000008080, 0x0000000080000001, 0x8000000080008008,
)
_ROTATIONS = (
    (0, 36, 3, 41, 18), (1, 44, 10, 45, 2), (62, 6, 43, 15, 61), (28, 55, 25, 21, 56), (27, 20, 39, 8, 14),
)
_RATE_BYTES = 136  # keccak-256: 1600 − 2·256 bits


def _rotl(value: int, shift: int) -> int:
    shift %= 64
    return ((value << shift) | (value >> (64 - shift))) & _MASK if shift else value


def _keccak_f(state: List[List[int]]) -> None:
    for constant in _ROUND_CONSTANTS:
        parity = [state[x][0] ^ state[x][1] ^ state[x][2] ^ state[x][3] ^ state[x][4] for x in range(5)]
        delta = [parity[(x - 1) % 5] ^ _rotl(parity[(x + 1) % 5], 1) for x in range(5)]
        for x in range(5):
            for y in range(5):
                state[x][y] ^= delta[x]
        rotated = [[0] * 5 for _ in range(5)]
        for x in range(5):
            for y in range(5):
                rotated[y][(2 * x + 3 * y) % 5] = _rotl(state[x][y], _ROTATIONS[x][y])
        for x in range(5):
            for y in range(5):
                state[x][y] = rotated[x][y] ^ ((~rotated[(x + 1) % 5][y]) & rotated[(x + 2) % 5][y] & _MASK)
        state[0][0] ^= constant


def keccak256(data: bytes) -> bytes:
    """Keccak-256 (the pre-NIST padding Ethereum uses), pure Python."""
    state = [[0] * 5 for _ in range(5)]
    padded = bytearray(data)
    padded.append(0x01)
    while len(padded) % _RATE_BYTES:
        padded.append(0x00)
    padded[-1] |= 0x80
    for offset in range(0, len(padded), _RATE_BYTES):
        block = padded[offset:offset + _RATE_BYTES]
        for i in range(_RATE_BYTES // 8):
            state[i % 5][i // 5] ^= int.from_bytes(block[8 * i: 8 * i + 8], "little")
        _keccak_f(state)
    out = bytearray()
    for i in range(4):  # 32 bytes = four lanes
        out += state[i % 5][i // 5].to_bytes(8, "little")
    return bytes(out)


def event_topic(signature: str) -> str:
    """``topics[0]`` of an event from its canonical signature, e.g. ``"Transfer(address,address,uint256)"``."""
    return "0x" + keccak256(signature.encode()).hex()


def function_selector(signature: str) -> str:
    """Four-byte selector of a function from its canonical signature, e.g. ``"readState(address)"``."""
    return "0x" + keccak256(signature.encode())[:4].hex()


# --------------------------------------------------------------- decoding
def decode_words(hex_result: str, count: int) -> List[int]:
    """Split ABI-encoded static data (``0x…``) into ``count`` unsigned 256-bit words."""
    body = hex_result[2:] if hex_result.startswith("0x") else hex_result
    if len(body) < 64 * count:
        raise RpcCallError(f"expected {count} return words, got {len(body) // 64}")
    return [int(body[64 * i: 64 * (i + 1)], 16) for i in range(count)]


def as_signed(word: int, bits: int = 256) -> int:
    """Two's-complement reinterpretation of an unsigned ABI word."""
    return word - (1 << bits) if word >= (1 << (bits - 1)) else word


def decode_address(word: int) -> str:
    """Lower-case ``0x``-address from a 256-bit word."""
    return "0x" + format(word & ((1 << 160) - 1), "040x")


def topic_to_address(topic: str) -> str:
    """Indexed ``address`` argument from a 32-byte topic."""
    return decode_address(int(topic, 16))


def topic_to_int(topic: str, signed: bool = False, bits: int = 256) -> int:
    """Indexed integer argument from a 32-byte topic."""
    value = int(topic, 16)
    return as_signed(value, bits) if signed else value


def encode_address(address: str) -> str:
    """ABI-encode an address as one 32-byte word (no ``0x``)."""
    return address.lower().replace("0x", "").rjust(64, "0")


def _hex_block(block: BlockTag) -> str:
    return hex(block) if isinstance(block, int) else block


# ----------------------------------------------------------------- client
class JsonRpcClient:
    """Thin JSON-RPC client over :class:`HttpClient`.

    Args:
        rpc_url: JSON-RPC endpoint.
        http: injectable transport (offline tests pass a fake).
    """

    def __init__(self, rpc_url: str, http: Optional[HttpClient] = None) -> None:
        self.rpc_url = rpc_url
        self._http = http or HttpClient()
        self._id = 0

    def call(self, method: str, params: Sequence[Any]) -> Any:
        """Raw JSON-RPC call; raises :class:`RpcCallError` on an error body."""
        self._id += 1
        payload = self._http.post(self.rpc_url, json={
            "jsonrpc": "2.0", "id": self._id, "method": method, "params": list(params),
        })
        if not isinstance(payload, dict) or "error" in payload or "result" not in payload:
            raise RpcCallError(f"RPC {method} failed: {payload!r}")
        return payload["result"]

    # ------------------------------------------------------------ blocks
    def block_number(self) -> int:
        return int(self.call("eth_blockNumber", []), 16)

    def get_block(self, block: BlockTag = "latest", full_transactions: bool = False) -> Dict[str, Any]:
        result = self.call("eth_getBlockByNumber", [_hex_block(block), full_transactions])
        if not isinstance(result, dict):
            raise RpcCallError(f"eth_getBlockByNumber({block!r}) returned {result!r}")
        return result

    def block_timestamp(self, block: BlockTag) -> int:
        return int(self.get_block(block)["timestamp"], 16)

    # ------------------------------------------------------------- calls
    def eth_call(self, to: str, data: str, block: BlockTag = "latest") -> str:
        """``eth_call`` and return the raw ``0x…`` hex result."""
        result = self.call("eth_call", [{"to": to, "data": data}, _hex_block(block)])
        if not isinstance(result, str) or not result.startswith("0x"):
            raise RpcCallError(f"eth_call to {to} returned a non-hex result: {result!r}")
        return result

    # -------------------------------------------------------------- logs
    def get_logs(
        self,
        address: Optional[Union[str, Sequence[str]]],
        topics: Optional[Sequence[Optional[Union[str, Sequence[str]]]]],
        from_block: BlockTag,
        to_block: BlockTag,
    ) -> List[Dict[str, Any]]:
        """One ``eth_getLogs`` over ``[from_block, to_block]``."""
        query: Dict[str, Any] = {"fromBlock": _hex_block(from_block), "toBlock": _hex_block(to_block)}
        if address is not None:
            query["address"] = address
        if topics is not None:
            query["topics"] = list(topics)
        result = self.call("eth_getLogs", [query])
        if not isinstance(result, list):
            raise RpcCallError(f"eth_getLogs returned {result!r}")
        return result

    def iter_logs(
        self,
        address: Optional[Union[str, Sequence[str]]],
        topics: Optional[Sequence[Optional[Union[str, Sequence[str]]]]],
        from_block: int,
        to_block: Optional[int] = None,
        *,
        chunk_size: int = 10_000,
        min_chunk: int = 500,
    ) -> Iterator[Dict[str, Any]]:
        """Logs over a block range, one ``eth_getLogs`` per chunk.

        Nodes reject oversized ranges either as a JSON-RPC error or at
        the HTTP level; both halve the chunk and retry the same range.
        After a success the chunk grows back toward ``chunk_size`` so one
        bad range does not multiply the number of calls. Below
        ``min_chunk`` the rejection propagates.
        """
        end_block = self.block_number() if to_block is None else to_block
        start, chunk = from_block, chunk_size
        while start <= end_block:
            end = min(start + chunk - 1, end_block)
            try:
                logs = self.get_logs(address, topics, start, end)
            except (RpcCallError, LoaderHttpError):
                if chunk <= min_chunk:
                    raise
                chunk //= 2
                continue
            yield from logs
            start = end + 1
            chunk = min(chunk * 2, chunk_size)


def eth_call(
    rpc_url: str,
    to: str,
    data: str,
    *,
    block: BlockTag = "latest",
    http: Optional[HttpClient] = None,
) -> str:
    """One-shot ``eth_call`` (see :meth:`JsonRpcClient.eth_call`)."""
    return JsonRpcClient(rpc_url, http=http).eth_call(to, data, block=block)
