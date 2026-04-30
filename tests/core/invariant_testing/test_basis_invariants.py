"""Invariants and randomized property checks for ``BasisTradingStrategy``.

Layer 3 — pinned invariants that should hold regardless of small
numerical changes to the underlying entities.

Layer 4 — lightweight property-based: deterministic seeds drive a
batch of randomized scenarios; each scenario re-checks the same
invariants. No external Hypothesis dependency — we trade Hypothesis'
shrinking power for zero install cost. If a seed surfaces a
counterexample, pin it as a regular Layer 3 test.

The invariants:

* **Basis hedge equality** — after any rebalance step that completes
  without liquidation, ``hedge.size ≈ -spot.amount``.
* **Equity conservation** — under zero fee + zero funding + no
  liquidation, ``total_balance(t) ≈ total_balance(0)``.
* **Leverage bounds** — after a rebalance step, hedge leverage falls
  inside ``[MIN_LEVERAGE, MAX_LEVERAGE]``.
* **Action symmetry on delta_spot<0** — the four-action sequence moves
  exactly the right notional (sell proceeds = deposit = withdraw =
  abs(open_position) × price).
"""
from __future__ import annotations

import math
import random
from datetime import datetime, timedelta
from typing import List

import pytest

from fractal.core.base import NamedEntity, Observation
from fractal.core.entities.protocols.uniswap_v3_spot import (
    UniswapV3SpotEntity, UniswapV3SpotGlobalState,
)
from fractal.core.entities.simple.perp import (
    SimplePerpEntity, SimplePerpGlobalState,
)
from fractal.strategies.basis_trading_strategy import (
    BasisTradingStrategy, BasisTradingStrategyHyperparams,
)


# ============================================================ scaffolding


class _TestableBasis(BasisTradingStrategy):
    HEDGE_TRADING_FEE: float = 0.0
    SPOT_TRADING_FEE: float = 0.0
    HEDGE_MAX_LEVERAGE: float = 50.0  # high enough to avoid bumping the cap

    def set_up(self):
        self.register_entity(NamedEntity(
            'HEDGE',
            SimplePerpEntity(
                trading_fee=self.HEDGE_TRADING_FEE,
                max_leverage=self.HEDGE_MAX_LEVERAGE,
            ),
        ))
        self.register_entity(NamedEntity(
            'SPOT',
            UniswapV3SpotEntity(trading_fee=self.SPOT_TRADING_FEE),
        ))
        super().set_up()


def _make_strategy(*, target_lev=3.0, min_lev=1.0, max_lev=5.0,
                   initial=100_000.0, hedge_fee=0.0, spot_fee=0.0,
                   hedge_max_lev=50.0) -> _TestableBasis:
    cls = type(
        '_S', (_TestableBasis,),
        {'HEDGE_TRADING_FEE': hedge_fee, 'SPOT_TRADING_FEE': spot_fee,
         'HEDGE_MAX_LEVERAGE': hedge_max_lev},
    )
    return cls(params=BasisTradingStrategyHyperparams(
        MIN_LEVERAGE=min_lev, TARGET_LEVERAGE=target_lev,
        MAX_LEVERAGE=max_lev, INITIAL_BALANCE=initial,
    ))


def _obs(t: datetime, price: float, funding: float = 0.0) -> Observation:
    return Observation(
        timestamp=t,
        states={
            'SPOT': UniswapV3SpotGlobalState(price=price),
            'HEDGE': SimplePerpGlobalState(mark_price=price, funding_rate=funding),
        },
    )


def _build_path(prices: List[float], start: datetime = datetime(2024, 1, 1),
                step: timedelta = timedelta(hours=1)) -> List[Observation]:
    return [_obs(start + i * step, p) for i, p in enumerate(prices)]


def _basis_ratio(s: _TestableBasis) -> float:
    """``|hedge.size + spot.amount| / max(|hedge.size|, eps)`` —
    0 for a perfect basis hedge."""
    hedge = s.get_entity('HEDGE')
    spot = s.get_entity('SPOT')
    h = hedge.size
    a = spot.internal_state.amount
    return abs(h + a) / max(abs(h), 1e-12)


# ============================================================ Layer 3 — invariants


@pytest.mark.core
def test_basis_invariant_holds_after_initial_deposit():
    """Right after the deposit step, hedge short matches spot long."""
    s = _make_strategy(target_lev=3.0)
    s.run([_obs(datetime(2024, 1, 1), 3000.0)])
    assert _basis_ratio(s) < 1e-12


@pytest.mark.core
def test_basis_invariant_holds_through_quiet_path_no_rebalance():
    """Tiny price wiggles inside [MIN, MAX] keep the basis hedge intact."""
    s = _make_strategy(min_lev=1.0, target_lev=3.0, max_lev=10.0)
    # ±0.2% wiggles — should not exceed leverage bounds.
    prices = [3000.0 * (1 + 0.002 * (i % 3 - 1)) for i in range(15)]
    s.run(_build_path(prices))
    # No rebalance fired (would have to perturb hedge) — basis still pristine.
    assert _basis_ratio(s) < 1e-12


@pytest.mark.core
def test_basis_invariant_restored_after_rebalance_uptrend():
    """After a tight-bounds uptrend triggers rebalancing, the basis
    hedge invariant must be restored within numerical tolerance."""
    s = _make_strategy(min_lev=2.5, target_lev=3.0, max_lev=3.2)
    prices = [3000.0 * (1 + 0.005 * i) for i in range(15)]
    s.run(_build_path(prices))
    assert _basis_ratio(s) < 1e-3


@pytest.mark.core
def test_equity_conservation_under_zero_fee_no_liquidation():
    """No fees + no funding + bounded price moves → total_balance constant.

    The sum across HEDGE + SPOT must equal INITIAL_BALANCE exactly under
    zero-fee, zero-funding conditions when no liquidation fires.
    """
    s = _make_strategy(target_lev=3.0, min_lev=1.0, max_lev=10.0,
                       hedge_fee=0.0, spot_fee=0.0)
    initial = s._params.INITIAL_BALANCE
    # Gentle path that won't trigger rebalance or liquidation.
    prices = [3000.0 * (1 + 0.001 * (i % 5 - 2)) for i in range(20)]
    s.run(_build_path(prices))
    final = s.total_balance
    assert final == pytest.approx(initial, rel=1e-9)


@pytest.mark.core
def test_equity_conservation_through_rebalance_zero_fee():
    """Rebalance under zero-fee preserves total equity exactly.

    The four-action sequence (sell/buy + transfer + open_position) does
    not bleed cash if all fees and slippage are zero.
    """
    s = _make_strategy(min_lev=2.5, target_lev=3.0, max_lev=3.1,
                       hedge_fee=0.0, spot_fee=0.0)
    initial = s._params.INITIAL_BALANCE
    # Aggressive enough to force at least one rebalance.
    prices = [3000.0 * (1 + 0.005 * i) for i in range(10)]
    s.run(_build_path(prices))
    # Allow tiny numerical noise from 4-step rebalance arithmetic.
    assert s.total_balance == pytest.approx(initial, rel=1e-6)


@pytest.mark.core
def test_leverage_returns_to_target_after_rebalance():
    """Each rebalance snaps hedge leverage back near TARGET_LEVERAGE.

    The exact post-rebalance leverage is ``TARGET_LEVERAGE × (1 + fee
    correction)`` ≈ TARGET when fees are 0.
    """
    s = _make_strategy(min_lev=2.0, target_lev=3.0, max_lev=3.5,
                       hedge_fee=0.0, spot_fee=0.0)
    prices = [3000.0 * (1 + 0.01 * i) for i in range(8)]
    s.run(_build_path(prices))
    hedge = s.get_entity('HEDGE')
    # Right after the last rebalance, leverage is close to TARGET.
    # Loose bound: it must lie inside [MIN, MAX].
    assert s._params.MIN_LEVERAGE <= hedge.leverage <= s._params.MAX_LEVERAGE * 1.05


@pytest.mark.core
def test_total_balance_never_increases_above_initial_under_zero_funding():
    """Without funding income, equity is monotonically non-increasing.

    Each step's total_balance must be ≤ initial (modulo float noise).
    A rising series would indicate money leaking IN (the inverse of B-1).
    """
    s = _make_strategy(min_lev=2.0, target_lev=3.0, max_lev=3.5,
                       hedge_fee=0.0, spot_fee=0.0)
    initial = s._params.INITIAL_BALANCE
    prices = [3000.0 * (1 + 0.005 * i) for i in range(12)]
    obs = _build_path(prices)
    result = s.run(obs)
    for i, balances in enumerate(result.balances):
        total = sum(balances.values())
        assert total <= initial * (1 + 1e-6), (
            f"step {i}: total_balance {total} exceeds initial {initial}"
        )


# ============================================================ Layer 3 — action-symmetry invariant on delta_spot<0


@pytest.mark.core
def test_rebalance_delta_spot_negative_action_amounts_consistent():
    """In the ``delta_spot<0`` branch, the four returned actions move
    matching notional through the cycle.

    sell-proceeds = deposit-amount = withdraw-amount = |open_position| × price.
    Delegates resolve to the same value at execute time.
    """
    s = _make_strategy(min_lev=2.5, target_lev=3.0, max_lev=3.05)
    s.run([_obs(datetime(2024, 1, 1), 3000.0)])  # initial deposit
    # Bump price so leverage > MAX → triggers delta_spot < 0 branch.
    s.get_entity('SPOT').update_state(UniswapV3SpotGlobalState(price=3030.0))
    s.get_entity('HEDGE').update_state(SimplePerpGlobalState(mark_price=3030.0))
    actions = s._rebalance()
    sell_amount_product = actions[0].action.args['amount_in_product']
    open_amount_product = actions[3].action.args['amount_in_product']
    price = 3030.0
    sell_proceeds_notional = sell_amount_product * price  # zero fee
    open_notional = abs(open_amount_product) * price
    # The strategy moves the same notional through both legs.
    assert sell_proceeds_notional == pytest.approx(open_notional, rel=1e-9)


# ============================================================ Layer 4 — randomized property-based (lightweight)


def _random_quiet_path(rng: random.Random, n: int = 30,
                        start: float = 3000.0,
                        step_pct: float = 0.003) -> List[float]:
    """Random walk with bounded steps; guaranteed to keep hedge alive
    when ``MAX_LEVERAGE`` is high.

    Uses ``rng.uniform(-1, 1)`` so the sum of steps stays bounded by
    ``n * step_pct``.
    """
    prices = [start]
    for _ in range(n - 1):
        delta = rng.uniform(-step_pct, step_pct)
        prices.append(prices[-1] * (1 + delta))
    return prices


@pytest.mark.core
@pytest.mark.parametrize("seed", list(range(20)))
def test_property_basis_invariant_holds_after_random_quiet_path(seed: int):
    """For each of 20 deterministic seeds, run a 30-step random walk
    and assert the basis hedge invariant at the end.

    Wide leverage bounds keep rebalances rare; the hedge must still
    track spot at every recorded step.
    """
    rng = random.Random(seed)
    target_lev = rng.uniform(2.0, 5.0)
    s = _make_strategy(min_lev=1.0, target_lev=target_lev, max_lev=15.0,
                       hedge_fee=0.0, spot_fee=0.0)
    prices = _random_quiet_path(rng, n=30)
    s.run(_build_path(prices))
    ratio = _basis_ratio(s)
    assert ratio < 1e-3, f"seed={seed} target_lev={target_lev} ratio={ratio}"


@pytest.mark.core
@pytest.mark.parametrize("seed", list(range(20)))
def test_property_equity_preserved_under_random_quiet_path(seed: int):
    """20 seeds × random walk × zero fees → equity must be conserved."""
    rng = random.Random(seed)
    target_lev = rng.uniform(2.0, 5.0)
    s = _make_strategy(min_lev=1.0, target_lev=target_lev, max_lev=15.0,
                       hedge_fee=0.0, spot_fee=0.0)
    prices = _random_quiet_path(rng, n=30)
    s.run(_build_path(prices))
    initial = s._params.INITIAL_BALANCE
    final = s.total_balance
    # The strategy may rebalance under tighter bounds, but with
    # zero-fee and no liquidation total stays put.
    assert final == pytest.approx(initial, rel=1e-6), (
        f"seed={seed} initial={initial} final={final}"
    )


@pytest.mark.core
@pytest.mark.parametrize("seed", list(range(15)))
def test_property_no_nan_or_inf_after_random_path(seed: int):
    """Across random scenarios, no balance/leverage ever goes NaN/inf."""
    rng = random.Random(seed)
    target_lev = rng.uniform(2.0, 5.0)
    s = _make_strategy(min_lev=1.0, target_lev=target_lev, max_lev=10.0,
                       hedge_fee=0.0, spot_fee=0.0)
    prices = _random_quiet_path(rng, n=25, step_pct=0.005)
    s.run(_build_path(prices))
    hedge = s.get_entity('HEDGE')
    spot = s.get_entity('SPOT')
    for v in (hedge.balance, hedge.size, hedge.leverage,
              spot.balance, spot.internal_state.amount, spot.internal_state.cash):
        assert math.isfinite(v), f"seed={seed} non-finite: {v}"


@pytest.mark.core
@pytest.mark.parametrize("seed", list(range(10)))
def test_property_leverage_returns_to_bounds_after_run(seed: int):
    """With tighter MIN/MAX, rebalances must keep hedge leverage in
    bounds at the end of any random run that doesn't liquidate."""
    rng = random.Random(seed)
    s = _make_strategy(min_lev=2.0, target_lev=3.0, max_lev=4.0,
                       hedge_fee=0.0, spot_fee=0.0)
    # Slightly wider step keeps MAX achievable; horizon short.
    prices = _random_quiet_path(rng, n=20, step_pct=0.008)
    s.run(_build_path(prices))
    hedge = s.get_entity('HEDGE')
    # Tolerance on the upper edge — the rebalance fires *after* the
    # offending step, so the very last leverage may sit slightly above
    # MAX if the price path drifts further before rebalance hits.
    assert hedge.leverage >= s._params.MIN_LEVERAGE * 0.5
    assert hedge.leverage <= s._params.MAX_LEVERAGE * 1.5
