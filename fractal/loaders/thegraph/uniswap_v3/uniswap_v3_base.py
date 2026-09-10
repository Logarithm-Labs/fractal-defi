from fractal.loaders.base_loader import LoaderType
from fractal.loaders.thegraph.uniswap_v3.uniswap_loader import UniswapV3Loader


class BaseUniswapV3Loader(UniswapV3Loader):
    """
    Loader for Uniswap V3 on Base (the network). Uses the standard
    ``uniswap-v3`` subgraph schema (``pools`` / ``poolDayDatas`` /
    ``poolHourDatas``).
    The Graph:
    https://thegraph.com/explorer/subgraphs/HMuAwufqZ1YCRmzL2SfHTVkzZovC9VL2UAKhjvRqKiR1?view=Query&chain=arbitrum-one
    SUBGRAPH_ID = "HMuAwufqZ1YCRmzL2SfHTVkzZovC9VL2UAKhjvRqKiR1"
    """

    SUBGRAPH_ID = "HMuAwufqZ1YCRmzL2SfHTVkzZovC9VL2UAKhjvRqKiR1"

    def __init__(self, api_key: str, loader_type: LoaderType = LoaderType.CSV) -> None:
        """
        Args:
            api_key (str): The Graph API key
            loader_type (LoaderType): loader type
        """
        super().__init__(api_key=api_key, subgraph_id=self.SUBGRAPH_ID, loader_type=loader_type)
