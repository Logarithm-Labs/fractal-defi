from typing import Tuple

from fractal.loaders.base_loader import LoaderType
from fractal.loaders.thegraph.uniswap_v3.uniswap_loader import UniswapV3Loader


class EthereumUniswapV3Loader(UniswapV3Loader):
    """
    Loader for Uniswap V3 Ethereum.
    The Graph:
    https://thegraph.com/explorer/subgraphs/5zvR82QoaXYFyDEKLZ9t6v9adgnptxYpKpSbxtgVENFV?view=Query&chain=arbitrum-one
    SUBGRAPH_ID = "5zvR82QoaXYFyDEKLZ9t6v9adgnptxYpKpSbxtgVENFV"
    """

    SUBGRAPH_ID = "5zvR82QoaXYFyDEKLZ9t6v9adgnptxYpKpSbxtgVENFV"

    def __init__(self, api_key: str, loader_type: LoaderType) -> None:
        """
        Args:
            api_key (str): The Graph API key
            loader_type (LoaderType): loader type
        """
        super().__init__(api_key=api_key, subgraph_id=self.SUBGRAPH_ID, loader_type=loader_type)

    def get_pool_decimals(self, address: str) -> Tuple[int, int]:
        """
        Get pool input tokens decimals

        Args:
            address (str): Pool address

        Returns:
            Tuple[int, int]: Decimals of input tokens (token0, token1)
        """
        query = """
        {
            pool(id: "%s") {
                token0 {
                    decimals
                }
                token1 {
                    decimals
                }
            }
        }
        """ % address.lower()
        data = self._make_request(query)
        decimals0 = data["pool"]["token0"]["decimals"]
        decimals1 = data["pool"]["token1"]["decimals"]
        return float(decimals0), float(decimals1)
