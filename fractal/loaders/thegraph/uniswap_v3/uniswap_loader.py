from abc import abstractmethod
from typing import Tuple

from fractal.loaders.base_loader import LoaderType
from fractal.loaders.thegraph.base_graph_loader import ArbitrumGraphLoader


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

    @abstractmethod
    def get_pool_decimals(self, address: str) -> Tuple[int, int]:
        """
        Get pool input tokens decimals

        Args:
            address (str): Pool address

        Returns:
            Tuple[int, int]: Decimals of input tokens (token0, token1)
        """

    def extract(self):
        pass

    def transform(self):
        pass

    def load(self):
        pass

    def read(self):
        pass
