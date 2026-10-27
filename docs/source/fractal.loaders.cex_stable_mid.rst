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

A bar belongs to the window when it opens in ``[start_time, end_time)`` and
has closed by the time of the request, so an unaligned ``end_time`` keeps the
bar it falls inside once that bar has closed. After a fetch,
``venue_report`` gives each venue's status, expected and present bars,
missing ``[start, end)`` ranges and the number of zero-volume candles dropped.
``include_volume=True`` adds per-venue volume columns in units of the pair's
base asset.

.. automodule:: fractal.loaders.cex_stable_mid
   :members:
   :show-inheritance:
   :undoc-members:
