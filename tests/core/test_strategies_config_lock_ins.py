"""Lock-in tests for strategy-configuration changes (A-F batch).

Covers:

* **HB-1**: ``HyperliquidBasis.MAX_LEVERAGE`` has a sensible default — a
  caller who forgets to override no longer trips ``AttributeError``
  inside ``set_up``.
* **T-1**: ``TauResetStrategy`` accepts ``token0_decimals`` /
  ``token1_decimals`` / ``tick_spacing`` as constructor kwargs (instance
  state, parallel-safe) while still falling back to class-level
  attributes when callers haven't migrated.
* **T-2**: ``TAU`` semantics — TAU is in *buckets*, each bucket of
  ``tick_spacing`` ticks — pinned numerically against the
  ``1.0001^tick`` Uniswap V3 formula.
"""
import pytest


# ============================================================ HB-1


@pytest.mark.core
def test_hyperliquid_basis_max_leverage_has_safe_default():
    """A caller who forgets to override the class-level
    ``MAX_LEVERAGE`` no longer hits ``AttributeError`` deep inside
    ``set_up`` — a default of 10 keeps construction working.
    """
    from fractal.strategies.hyperliquid_basis import HyperliquidBasis
    # The annotation alone (``MAX_LEVERAGE: float``) wouldn't expose a
    # value at the class level; only an explicit assignment does.
    assert HyperliquidBasis.MAX_LEVERAGE == pytest.approx(10.0)


@pytest.mark.core
def test_hyperliquid_basis_constructs_without_overriding_max_leverage():
    """End-to-end check: with the default in place, the strategy can
    be instantiated without any class-level mutation."""
    from dataclasses import dataclass

    from fractal.strategies.basis_trading_strategy import \
        BasisTradingStrategyHyperparams
    from fractal.strategies.hyperliquid_basis import HyperliquidBasis

    @dataclass
    class _P(BasisTradingStrategyHyperparams):
        EXECUTION_COST: float = 0.0

    s = HyperliquidBasis(params=_P(
        MIN_LEVERAGE=1.0, TARGET_LEVERAGE=3.0, MAX_LEVERAGE=5.0,
        INITIAL_BALANCE=100_000.0, EXECUTION_COST=0.0,
    ))
    hedge = s.get_entity('HEDGE')
    assert hedge.max_leverage == pytest.approx(10.0)


@pytest.mark.core
def test_hyperliquid_basis_class_level_override_still_takes_effect():
    """Backwards compatibility: the legacy ``cls.MAX_LEVERAGE = X``
    pattern continues to flow through to the entity."""
    from dataclasses import dataclass

    from fractal.strategies.basis_trading_strategy import \
        BasisTradingStrategyHyperparams
    from fractal.strategies.hyperliquid_basis import HyperliquidBasis

    @dataclass
    class _P(BasisTradingStrategyHyperparams):
        EXECUTION_COST: float = 0.0

    # Use a temporary subclass so we don't pollute the canonical class
    # for parallel tests.
    class _Probe(HyperliquidBasis):
        MAX_LEVERAGE = 25.0

    s = _Probe(params=_P(
        MIN_LEVERAGE=1.0, TARGET_LEVERAGE=3.0, MAX_LEVERAGE=5.0,
        INITIAL_BALANCE=100_000.0, EXECUTION_COST=0.0,
    ))
    assert s.get_entity('HEDGE').max_leverage == 25.0


# ============================================================ T-1 — constructor kwargs


@pytest.mark.core
def test_tau_reset_accepts_constructor_kwargs():
    """Pool config flows through constructor kwargs into the entity
    config — no class-level mutation required."""
    from fractal.strategies.tau_reset_strategy import (TauResetParams,
                                                       TauResetStrategy)
    s = TauResetStrategy(
        params=TauResetParams(TAU=15, INITIAL_BALANCE=100_000.0),
        token0_decimals=6, token1_decimals=18, tick_spacing=60,
    )
    assert s._token0_decimals == 6
    assert s._token1_decimals == 18
    assert s._tick_spacing == 60


@pytest.mark.core
def test_tau_reset_constructor_kwargs_take_precedence_over_class_level():
    """If a caller supplies a kwarg, it wins over a stale class-level value."""
    from fractal.strategies.tau_reset_strategy import (TauResetParams,
                                                       TauResetStrategy)

    class _Probe(TauResetStrategy):
        token0_decimals = 18  # would-be conflict
        token1_decimals = 18
        tick_spacing = 200

    s = _Probe(
        params=TauResetParams(TAU=15, INITIAL_BALANCE=100_000.0),
        token0_decimals=6, tick_spacing=60,
    )
    assert s._token0_decimals == 6
    assert s._token1_decimals == 18  # falls back to class
    assert s._tick_spacing == 60


@pytest.mark.core
def test_tau_reset_raises_clear_error_when_pool_config_missing():
    """Without kwargs AND without class-level setup, construction
    raises a clear ``TauResetException`` instead of an opaque
    ``AssertionError`` (the prior raw ``assert`` behaviour)."""
    from fractal.strategies.tau_reset_strategy import (TauResetException,
                                                       TauResetParams,
                                                       TauResetStrategy)

    class _Probe(TauResetStrategy):
        # No class-level overrides — sentinels remain.
        pass

    with pytest.raises(TauResetException, match="token0_decimals"):
        _Probe(params=TauResetParams(TAU=15, INITIAL_BALANCE=100_000.0))


@pytest.mark.core
def test_tau_reset_class_level_pattern_still_works():
    """Backwards compatibility — the legacy pattern of mutating class
    attributes before construction continues to work."""
    from fractal.strategies.tau_reset_strategy import (TauResetParams,
                                                       TauResetStrategy)

    class _Probe(TauResetStrategy):
        token0_decimals = 6
        token1_decimals = 18
        tick_spacing = 60

    s = _Probe(params=TauResetParams(TAU=15, INITIAL_BALANCE=100_000.0))
    assert s._token0_decimals == 6
    assert s._token1_decimals == 18
    assert s._tick_spacing == 60


# ============================================================ T-2 — TAU semantics


@pytest.mark.core
def test_tau_reset_range_matches_uniswap_v3_tick_formula_for_tau_one():
    """**Lock-in for T-2.** TAU is in *buckets*, each of ``tick_spacing``
    ticks. For ``TAU=1, tick_spacing=60`` the active range is
    ``[P · 1.0001^-60, P · 1.0001^60]`` ≈ ``[P / 1.00601, P · 1.00601]``.
    """
    from datetime import datetime

    from fractal.core.base import Observation
    from fractal.core.entities import UniswapV3LPGlobalState
    from fractal.strategies.tau_reset_strategy import (TauResetParams,
                                                       TauResetStrategy)

    s = TauResetStrategy(
        params=TauResetParams(TAU=1, INITIAL_BALANCE=100_000.0),
        token0_decimals=6, token1_decimals=18, tick_spacing=60,
    )
    s.run([
        Observation(
            timestamp=datetime(2024, 1, 1),
            states={
                'UNISWAP_V3': UniswapV3LPGlobalState(
                    price=1.0, tvl=1_000_000.0, volume=0.0,
                    fees=0.0, liquidity=1_000_000.0,
                ),
            },
        ),
        # Second tick triggers _rebalance and pins the bounds.
        Observation(
            timestamp=datetime(2024, 1, 2),
            states={
                'UNISWAP_V3': UniswapV3LPGlobalState(
                    price=2.0, tvl=1_000_000.0, volume=0.0,
                    fees=0.0, liquidity=1_000_000.0,
                ),
            },
        ),
    ])
    entity = s.get_entity('UNISWAP_V3')
    expected_upper = 2.0 * (1.0001 ** 60)
    expected_lower = 2.0 * (1.0001 ** -60)
    assert entity.internal_state.price_upper == pytest.approx(expected_upper, rel=1e-9)
    assert entity.internal_state.price_lower == pytest.approx(expected_lower, rel=1e-9)
    # Sanity: both within ±0.7% of reference price.
    assert abs(entity.internal_state.price_upper / 2.0 - 1.0) < 0.007
    assert abs(entity.internal_state.price_lower / 2.0 - 1.0) < 0.007
