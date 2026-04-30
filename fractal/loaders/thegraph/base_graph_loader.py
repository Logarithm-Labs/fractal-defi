"""Base class for TheGraph subgraph loaders.

All TheGraph loaders go through ``gateway-arbitrum.network.thegraph.com``
(the public unified gateway, used for both Ethereum and Arbitrum
subgraphs). Concrete loaders supply a subgraph id; this class handles
URL construction, transport, error wrapping.
"""
import re
from typing import Any, Dict, Optional

from fractal.loaders._http import HttpClient
from fractal.loaders.base_loader import Loader, LoaderType


class GraphLoaderException(RuntimeError):
    pass


# 0x-prefixed 40-hex-char EVM address. Concrete loaders interpolate pool
# / token addresses into GraphQL queries — validating shape up front
# stops bogus or malicious strings reaching the subgraph.
_EVM_ADDRESS_RE = re.compile(r"^0x[a-fA-F0-9]{40}$")


def validate_evm_address(value: str, field: str = "address") -> str:
    """Reject anything that does not look like a 0x-prefixed EVM address.

    Returns the lower-cased canonical form so caches/keys stay stable.
    """
    if not isinstance(value, str) or not _EVM_ADDRESS_RE.match(value):
        raise GraphLoaderException(
            f"{field} must match ^0x[a-fA-F0-9]{{40}}$, got {value!r}"
        )
    return value.lower()


class BaseGraphLoader(Loader):
    """Generic Graph subgraph loader."""

    def __init__(
        self,
        root_url: str,
        api_key: str,
        subgraph_id: str,
        loader_type: LoaderType,
        http: Optional[HttpClient] = None,
    ) -> None:
        super().__init__(loader_type=loader_type)
        if not api_key:
            raise GraphLoaderException("api_key is required")
        self._api_key: str = api_key
        self._subgraph_id: str = subgraph_id
        self._url: str = f"{root_url}/{api_key}/subgraphs/id/{subgraph_id}"
        self._http: HttpClient = http or HttpClient()

    def _make_request(self, query: str, *args, **kwargs) -> Dict[str, Any]:
        """Run a GraphQL query and return the ``data`` payload."""
        payload = self._http.post(self._url, json={"query": query})
        if not isinstance(payload, dict):
            raise GraphLoaderException(f"Unexpected response from subgraph: {payload!r}")
        if "errors" in payload:
            raise GraphLoaderException(f"GraphQL errors: {payload['errors']}")
        if "data" not in payload:
            raise GraphLoaderException(f"GraphQL response missing 'data': {payload!r}")
        return payload["data"]


class ArbitrumGraphLoader(BaseGraphLoader):
    """Convenience: subgraph hosted on the Arbitrum gateway (default for
    both Ethereum and Arbitrum subgraphs in the unified Graph network)."""

    ROOT_URL = "https://gateway-arbitrum.network.thegraph.com/api"

    def __init__(
        self,
        api_key: str,
        subgraph_id: str,
        loader_type: LoaderType,
        http: Optional[HttpClient] = None,
    ) -> None:
        super().__init__(
            root_url=self.ROOT_URL,
            api_key=api_key,
            subgraph_id=subgraph_id,
            loader_type=loader_type,
            http=http,
        )
