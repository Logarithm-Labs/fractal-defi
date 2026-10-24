fractal.loaders.cex\_stable\_mid module
=========================================

``StablecoinMidLoader`` builds an hourly or daily stablecoin series from the
median close available across Binance, Kraken, Coinbase Exchange and Bitstamp.
Reciprocal venue symbols are inverted before aggregation. The result is a CEX
last-trade proxy, not an executable on-chain price or swap quote. Missing bars
are never forward-filled.

The public APIs impose history limits per request: Binance and Bitstamp return
at most 1,000 candles, Coinbase Exchange returns at most 300, and the loader
pages those endpoints. Kraken's OHLC endpoint exposes only the most recent 720
bars even when an earlier ``since`` value is requested, so long windows can
have an absent Kraken prefix.

.. automodule:: fractal.loaders.cex_stable_mid
   :members:
   :show-inheritance:
   :undoc-members:
