"""Aerodrome (Base) subgraph loaders.

Aerodrome Slipstream is a Uniswap V3-style concentrated-liquidity AMM and
its subgraph is a fork of the ``uniswap-v3`` schema (``pools`` /
``poolHourDatas`` / ``poolDayDatas``), so the loaders reuse the shared
UniswapV3 pool machinery.
"""
from fractal.loaders.base_loader import LoaderType
from fractal.loaders.thegraph.uniswap_v3.uniswap_loader import UniswapV3Loader
from fractal.loaders.thegraph.uniswap_v3.uniswap_v3_pool import _UniswapV3PoolHourBase


class AerodromeLoader(UniswapV3Loader):
    """
    Loader for Aerodrome on Base (covers Slipstream CL pools).
    The Graph ("Aerodrome Base Full"):
    https://thegraph.com/explorer/subgraphs/GENunSHWLBXm59mBSgPzQ8metBEp9YDfdqwFr91Av1UM?view=Query&chain=arbitrum-one
    SUBGRAPH_ID = "GENunSHWLBXm59mBSgPzQ8metBEp9YDfdqwFr91Av1UM"
    """

    SUBGRAPH_ID = "GENunSHWLBXm59mBSgPzQ8metBEp9YDfdqwFr91Av1UM"

    def __init__(self, api_key: str, loader_type: LoaderType = LoaderType.CSV) -> None:
        """
        Args:
            api_key (str): The Graph API key
            loader_type (LoaderType): loader type
        """
        super().__init__(api_key=api_key, subgraph_id=self.SUBGRAPH_ID, loader_type=loader_type)


class AerodromeSlipstreamPoolHourDataLoader(_UniswapV3PoolHourBase, AerodromeLoader):
    """Native hourly pool snapshots for an Aerodrome Slipstream pool on Base."""
