fractal.loaders.bridges.cctp module
===================================

Three loaders over Circle Iris V2's keyless API.

``CctpFeeLoader`` reads the published USDC transfer-fee schedule for one
source-domain and destination-domain route. It returns Fast and Standard rows
with ``minimum_fee_bps`` and the corresponding ``finality_threshold``: entries
at or below 1000 are Fast, at or above 2000 Standard, and entries strictly
between 1000 and 2000 match neither documented mode and are dropped. An
unknown route reported as HTTP 404 returns an empty, well-shaped frame.

``CctpFastAllowanceLoader`` returns one row with the remaining USDC Fast
Transfer allowance (``allowance_usdc``) and the provider's ``last_updated``
time. It is a point-in-time snapshot.

``CctpMessagesLoader`` returns the CCTP messages of one source transaction
(source domain plus transaction hash): the Iris-assigned ``nonce`` (32-byte
hex, missing until assigned, since the on-chain V2 burn message carries a zero
nonce), ``status``, destination domain, amount, maximum and executed fee in
base units, finality thresholds and the sender and recipient fields. Raw
message and attestation bytes are not returned. An unknown transaction (HTTP
404) returns an empty frame; status reflects the time of the fetch.

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
