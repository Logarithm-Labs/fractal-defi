fractal.loaders.bridges.cctp module
===================================

``CctpFeeLoader`` reads Circle Iris V2's keyless, published USDC transfer-fee
schedule for one source-domain and destination-domain route. It returns Fast
and Standard rows with ``minimum_fee_bps`` and the corresponding
``finality_threshold``. An unknown route reported as HTTP 404 returns an empty,
well-shaped frame.

The module exposes the following static CCTP domain-ID map:

.. list-table:: CCTP domains
   :header-rows: 1

   * - Chain
     - Domain ID
   * - Ethereum
     - 0
   * - Optimism
     - 2
   * - Arbitrum
     - 3
   * - Base
     - 6
   * - Polygon
     - 7
   * - Unichain
     - 10
   * - Monad
     - 15
   * - HyperEVM
     - 19
   * - Arc
     - 26
   * - Plasma
     - 33

.. automodule:: fractal.loaders.bridges.cctp
   :members:
   :show-inheritance:
   :undoc-members:
