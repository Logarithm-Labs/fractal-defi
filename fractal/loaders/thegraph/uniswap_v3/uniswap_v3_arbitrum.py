from typing import Tuple

from fractal.loaders.base_loader import LoaderType
from fractal.loaders.thegraph.uniswap_v3.uniswap_loader import UniswapV3Loader


class ArbitrumUniswapV3Loader(UniswapV3Loader):
    """
    Loader for Uniswap V3 Arbitrum.
    The Graph:
    https://thegraph.com/explorer/subgraphs/FQ6JYszEKApsBpAmiHesRsd9Ygc6mzmpNRANeVQFYoVX?view=Query&chain=arbitrum-one
    SUBGRAPH_ID = "FQ6JYszEKApsBpAmiHesRsd9Ygc6mzmpNRANeVQFYoVX"
    """

    SUBGRAPH_ID = "FQ6JYszEKApsBpAmiHesRsd9Ygc6mzmpNRANeVQFYoVX"

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
            liquidityPools(where: {id:"%s"}) {
                id
                inputTokens {
                decimals
                }
            }
        }
        """ % address.lower()
        data = self._make_request(query)
        decimals0 = data["liquidityPools"][0]["inputTokens"][0]["decimals"]
        decimals1 = data["liquidityPools"][0]["inputTokens"][1]["decimals"]
        return float(decimals0), float(decimals1)
