Introduction
============

**Fractal** is an open-source Python research library for DeFi
strategies. Compose protocol-agnostic entities (lending, perps, DEX
and LP) into typed strategies; backtest, simulate, track experiments.

The framework is library-shaped — small primitives with a big
composition surface. A strategy written against the generic
``BasePerpEntity`` / ``BaseLendingEntity`` / ``BasePoolEntity`` /
``BaseSpotEntity`` contracts works across protocols; swap concrete
implementations without touching the strategy code.

Core concepts
-------------

The framework is built on three primitives that compose:

**Entity** (:class:`fractal.core.base.entity.BaseEntity`)
    A model of a single on-chain protocol position. Each entity owns:

    * an ``InternalState`` — the user's position (collateral, debt,
      LP tokens, perp size, …);
    * a ``GlobalState`` — the market context the entity reads at every
      step (prices, funding rates, lending rates, pool TVL, …);
    * a fixed set of ``action_*`` methods (``deposit``, ``borrow``,
      ``open_position``, …) and an ``update_state(state)`` hook that
      the framework calls every observation.

    Concrete entities live under :mod:`fractal.core.entities` —
    ``AaveEntity``, ``HyperliquidEntity``, ``UniswapV3LPEntity``,
    ``UniswapV3SpotEntity``, ``SimpleLendingEntity``, etc. Each has a
    paradigm-agnostic ``Base*`` parent and one or more concrete
    protocol-specific subclasses.

**Strategy** (:class:`fractal.core.base.strategy.BaseStrategy`)
    Coordinator over one or more entities. A subclass overrides
    ``set_up`` (register entities) and ``predict`` (return a list of
    ``ActionToTake`` based on current entity states). The framework
    drives the loop:

    .. code-block:: text

        for observation in observations:
            validate observation
            for entity_name, state in observation.states.items():
                entity.update_state(state)
            actions = strategy.predict()
            for action in actions:
                entity.execute(action)
            snapshot internal_state + global_state for the result

    Strategy hyperparameters live on a typed
    :class:`~fractal.core.base.strategy.BaseStrategyParams` dataclass
    referenced through the ``BaseStrategy[Params]`` generic — both
    instance objects and dict-shaped grid cells are accepted and
    coerced into that dataclass.

**Loader** (:class:`fractal.loaders.base_loader.Loader`)
    ETL adapter for an external data source. Lifecycle is
    ``extract → transform → load`` driven by ``run()``;
    ``read(with_run=False)`` reads the on-disk cache. Concrete loaders
    return one of the typed structures from
    :mod:`fractal.loaders.structs` (``PriceHistory``,
    ``FundingHistory``, ``PoolHistory``, ``LendingHistory``,
    ``KlinesHistory``, ``TrajectoryBundle``).

End-to-end shape
----------------

A typical Fractal program has four sections:

.. code-block:: python

    # 1. Build observations from one or more loaders.
    prices = BinancePriceLoader('BTCUSDT', interval='1h',
                                start_time=..., end_time=...).read(with_run=True)
    observations = [
        Observation(timestamp=ts, states={'SPOT': SimpleSpotExchangeGlobalState(close=p)})
        for ts, p in zip(prices.index, prices['price'])
    ]

    # 2. Configure strategy hyperparameters.
    params = MyParams(INITIAL_BALANCE=1_000_000, ...)

    # 3. Run the strategy. ``set_up`` registers entities; ``predict``
    #    fires every observation; ``run`` drives the loop and returns
    #    a StrategyResult.
    strategy = MyStrategy(params=params)
    result = strategy.run(observations)

    # 4. Inspect: per-step DataFrame, scalar metrics.
    print(result.get_default_metrics())
    result.to_dataframe().to_csv('result.csv')

For grid search and MLflow logging, wrap the same strategy and
observations in :class:`fractal.core.pipeline.DefaultPipeline`.

Conventions
-----------

* **Pricing.** Entities are unit-agnostic: ``GlobalState.price`` /
  ``mark_price`` / ``collateral_price`` are whatever scale you supply.
  Mixing units across entities in one strategy produces a wrong
  ``total_balance``; pick a single accounting unit (typically USD)
  and feed prices in that unit.
* **Rate sign.** ``lending_rate > 0`` ⇒ collateral grows;
  ``borrowing_rate > 0`` ⇒ debt grows; ``funding_rate > 0`` ⇒ longs
  pay shorts; ``trading_fee > 0`` ⇒ execution cost charged on traded
  notional. Loaders in :mod:`fractal.loaders` follow this convention
  uniformly.
* **Action atomicity.** Each ``action_*`` method validates inputs and
  either fully applies or raises. Risk-increasing perp trades that
  would breach ``max_leverage`` are rejected with an atomic rollback;
  lending borrows are checked against cumulative LTV.
* **Observation ordering.** Timestamps inside a strategy run must be
  monotonically increasing. ``StrategyResult.get_metrics`` returns
  zero metrics on degenerate inputs (empty / single-timestamp /
  zero-initial-balance) instead of raising.

Next steps
----------

* :doc:`installation` — install from PyPI or from source.
* :mod:`fractal.core` — full API reference, starting with the base
  contracts.
* `Examples on GitHub <https://github.com/Logarithm-Labs/fractal-defi/tree/main/examples>`_
  — ``quick_start.py`` (passive lending), ``basis/`` (Hyperliquid
  basis trade), ``tau_reset/`` (active Uniswap V3 LP),
  ``agentic_trader/`` (LLM-driven trading), ``holder/`` (toy hodler).
