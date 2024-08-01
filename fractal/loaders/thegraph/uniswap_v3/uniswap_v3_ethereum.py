from fractal.loaders.base_loader import LoaderType
from fractal.loaders.thegraph.base_graph_loader import ArbitrumGraphLoader


class EthereumUniswapV3Loader(ArbitrumGraphLoader):
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
