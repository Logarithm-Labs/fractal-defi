from typing import Tuple

from fractal.loaders.base_loader import LoaderType
from fractal.loaders.thegraph.base_graph_loader import ArbitrumGraphLoader, GraphLoaderException, validate_evm_address


class UniswapV3Loader(ArbitrumGraphLoader):
    """
    Loader for Uniswap V3
    """
    def __init__(self, api_key: str, subgraph_id: str, loader_type: LoaderType = LoaderType.CSV) -> None:
        """
        Args:
            api_key (str): The Graph API key
            loader_type (LoaderType): loader type
        """
        super().__init__(api_key=api_key, subgraph_id=subgraph_id, loader_type=loader_type)

    def get_pool_decimals(self, address: str) -> Tuple[int, int]:
        """
        Get pool input tokens decimals.

        Delegates to :meth:`get_pool_info`, so loaders backed by a
        different schema only override that one method.

        Args:
            address (str): Pool address

        Returns:
            Tuple[int, int]: Decimals of input tokens (token0, token1)
        """
        info = self.get_pool_info(address)
        return info["decimals0"], info["decimals1"]

    def get_pool_info(self, address: str) -> dict:
        """Pool static config from the subgraph (standard uniswap-v3 schema).

        Loaders backed by a different schema (e.g. the Messari Arbitrum
        subgraph) must override this.

        Args:
            address (str): Pool address

        Returns:
            dict: ``{address, fee_tier, token0, token1, decimals0,
            decimals1}`` where ``token0``/``token1`` are symbols and
            ``fee_tier`` is in Uniswap pips (1e6 = 100%). Note: what the
            subgraph cannot know stays out — e.g. the protocol-fee split
            (``slot0.feeProtocol``, on-chain only) and which side to
            treat as notional (a modeling choice).
        """
        address = validate_evm_address(address, field="pool")
        query = """
        {
            pool(id: "%s") {
                feeTier
                token0 { symbol decimals }
                token1 { symbol decimals }
            }
        }
        """ % address
        data = self._make_request(query)["pool"]
        if data is None:
            raise GraphLoaderException(
                f"pool {address} not found in subgraph {self._subgraph_id}"
            )
        return {
            "address": address,
            "fee_tier": int(data["feeTier"]),
            "token0": data["token0"]["symbol"],
            "token1": data["token1"]["symbol"],
            "decimals0": int(data["token0"]["decimals"]),
            "decimals1": int(data["token1"]["decimals"]),
        }

    # Lifecycle methods are required by the ``Loader`` ABC but have no
    # meaningful implementation at this layer — concrete pool/spot
    # loaders below override them. Keep them as ``NotImplementedError``
    # so a misconfigured subclass that forgets to override fails loudly.
    def extract(self):
        raise NotImplementedError(
            f"{type(self).__name__} must override extract()."
        )

    def transform(self):
        raise NotImplementedError(
            f"{type(self).__name__} must override transform()."
        )

    def load(self):
        raise NotImplementedError(
            f"{type(self).__name__} must override load()."
        )

    def read(self, with_run: bool = False):
        raise NotImplementedError(
            f"{type(self).__name__} must override read()."
        )
