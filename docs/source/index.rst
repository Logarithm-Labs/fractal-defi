Fractal — DeFi research library
===============================

**Fractal** is an open-source Python research library for DeFi
strategies. Compose protocol-agnostic entities (lending, perps, DEX
and LP) into typed strategies; backtest, simulate, track experiments.

Library-shaped, not product-shaped — small primitives with a big
composition surface. Strategies written against the
``BasePerpEntity`` / ``BaseLendingEntity`` / ``BasePoolEntity`` /
``BaseSpotEntity`` contracts work across protocols; swap concrete
implementations (Hyperliquid, Aave, Uniswap V3, GMX, your own)
without touching strategy code.

The codebase is organized around three layers:

* :mod:`fractal.core.entities` — protocol primitives (lending pools,
  perpetual exchanges, AMM LPs, spot exchanges, liquid-staking
  tokens), each with a typed ``InternalState`` + ``GlobalState``.
* :mod:`fractal.core.base` — the ``BaseStrategy`` / ``BaseEntity``
  contracts plus the per-step driver loop (``predict`` →
  ``ActionToTake`` → ``execute``).
* :mod:`fractal.loaders` — ETL adapters for Binance, Hyperliquid,
  Aave V3, GMX, TheGraph (Uniswap V2/V3, Lido), and offline
  Monte-Carlo / bootstrap simulators.

Ready-made strategies live under :mod:`fractal.strategies` (basis
trading, Hyperliquid basis, TauReset Uniswap V3 LP).
:mod:`fractal.core.pipeline` wires the framework to MLflow for
grid-search experiments.

.. toctree::
   :maxdepth: 2
   :caption: Getting started

   introduction
   installation

.. toctree::
   :maxdepth: 2
   :caption: API reference

   fractal.core
   fractal.core.base
   fractal.core.entities
   fractal.loaders
   fractal.strategies

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
