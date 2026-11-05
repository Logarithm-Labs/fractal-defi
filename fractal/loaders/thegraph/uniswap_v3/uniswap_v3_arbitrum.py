from fractal.loaders.base_loader import LoaderType
from fractal.loaders.thegraph.base_graph_loader import GraphLoaderException, validate_evm_address
from fractal.loaders.thegraph.uniswap_v3.uniswap_loader import UniswapV3Loader


class ArbitrumUniswapV3Loader(UniswapV3Loader):
    """
    Loader for Uniswap V3 Arbitrum.
    The Graph:
    https://thegraph.com/explorer/subgraphs/FQ6JYszEKApsBpAmiHesRsd9Ygc6mzmpNRANeVQFYoVX?view=Query&chain=arbitrum-one
    SUBGRAPH_ID = "FQ6JYszEKApsBpAmiHesRsd9Ygc6mzmpNRANeVQFYoVX"
    """

    SUBGRAPH_ID = "FQ6JYszEKApsBpAmiHesRsd9Ygc6mzmpNRANeVQFYoVX"

    def __init__(self, api_key: str, loader_type: LoaderType = LoaderType.CSV) -> None:
        """
        Args:
            api_key (str): The Graph API key
            loader_type (LoaderType): loader type
        """
        super().__init__(api_key=api_key, subgraph_id=self.SUBGRAPH_ID, loader_type=loader_type)

    def get_pool_info(self, address: str) -> dict:
        """Messari-schema variant of the standard uniswap-v3 pool lookup."""
        address = validate_evm_address(address, field="pool")
        query = """
        {
            liquidityPools(where: {id:"%s"}) {
                fees { feePercentage feeType }
                inputTokens { symbol decimals }
            }
        }
        """ % address
        pools = self._make_request(query)["liquidityPools"]
        if not pools:
            raise GraphLoaderException(
                f"pool {address} not found in subgraph {self.SUBGRAPH_ID}"
            )
        data = pools[0]
        fee_tier = next(
            (round(float(f["feePercentage"]) * 10_000) for f in data["fees"]
             if f["feeType"] == "FIXED_TRADING_FEE"),
            None,
        )
        return {
            "address": address,
            "fee_tier": fee_tier,
            "token0": data["inputTokens"][0]["symbol"],
            "token1": data["inputTokens"][1]["symbol"],
            "decimals0": int(data["inputTokens"][0]["decimals"]),
            "decimals1": int(data["inputTokens"][1]["decimals"]),
        }
