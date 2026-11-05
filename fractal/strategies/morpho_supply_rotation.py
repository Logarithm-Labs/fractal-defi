"""Morpho Blue supply rotation with lagged signals and liquidity-gated exits.

Ranks eligible markets on a trailing mean of either their supply APY or the
projected APY at the portfolio's size. Signals end SIGNAL_LAG_BARS observed
bars before execution. Transfers are bounded by the source's available
liquidity and costs; stranded balances are retried and reported as deferrals.

MAX_LLTV, ORACLE_TYPES, LOAN_SYMBOLS and MIN_POOL_SIZE_USD control eligibility.
TOP_K defines an acceptable-holding band; outside it, MIN_EDGE_BPS controls
switching. MAX_HOLD_BARS can force re-selection, not a profitable exit.

Entities record per-leg costs in the execution ledger. Pass a market registry
and cash entity at construction; set_up runs inside BaseStrategy.__init__.

Sparse markets do not advance their signal buffers. A lag is measured in
observed bars, not wall-clock hours. Missing observations retain the previous
entity state, including its liquidity; this is not an executable-price model.
Cross-chain transfers assume instantaneous USD-notional settlement: bridging,
depegs, oracle failures and bad debt are not simulated. Defaults are examples,
not investment recommendations.
"""
import math
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass

from fractal.core.base.entity import BaseEntity
from fractal.core.base.strategy import ActionToTake, BaseStrategy, NamedEntity
from fractal.loaders.morpho_markets import MorphoMarketListing
from fractal.strategies.morpho_supply_common import (
    MorphoSupplyBaseParams,
    build_supply_entity,
    is_selectable,
    market_entity_name,
    validate_common_params,
)

__all__ = [
    "MorphoSupplyRotationParams",
    "MorphoSupplyRotationStrategy",
    "market_entity_name",
]


@dataclass
class MorphoSupplyRotationParams(MorphoSupplyBaseParams):
    """Illustrative hyperparameters; calibrate explicitly for each experiment.

    The shared eligibility, signal and entity fields live on
    :class:`MorphoSupplyBaseParams`; this class adds the rotation's own
    band and switching parameters.

    Attributes:
        TOP_K: size of the acceptable-holding band. The best-signal market is
            the target; the current market is kept if it is still inside the
            top ``TOP_K`` (avoids churn between near-ties).
        MIN_EDGE_BPS: how much better (in basis points of APY) the target must
            be before a market-to-market rotation pays two legs of cost.
            Applies to every such rotation; leaving cash is always allowed.
        MAX_HOLD_BARS: force a re-selection once a position has been held this
            many bars (``>= 0``; ``0`` disables). It never forces an exit to cash.
        RANK_ON_PROJECTED_APY: rank on the APY the portfolio's own size would
            earn in each market (self-dilution included) instead of the
            headline rate. No effect when ``OWN_IMPACT`` is off.
        SIGNAL_WINDOW_BARS: bars averaged to form the signal level; ``24`` by
            default here, while the allocation overrides it with ``1``.
    """
    TOP_K: int = 1
    MIN_EDGE_BPS: float = 10.0
    MAX_HOLD_BARS: int = 0
    RANK_ON_PROJECTED_APY: bool = True
    SIGNAL_WINDOW_BARS: int = 24


class MorphoSupplyRotationStrategy(BaseStrategy[MorphoSupplyRotationParams]):
    """Rotate one supply position across the registered Morpho markets."""

    #: Markets can drop out of an observation (no data on a chain for that
    #: bar); a missing state must not abort the run, it just means no fresh
    #: signal for that market.
    STRICT_OBSERVATIONS: bool = False

    def __init__(
        self,
        markets: Sequence[MorphoMarketListing],
        cash_entity: BaseEntity,
        **kwargs,
    ) -> None:
        # Config must exist before ``super().__init__`` because that calls
        # ``set_up`` (see :class:`MorphoSupplyEntity`'s use of the same ordering).
        if not markets:
            raise ValueError("markets must not be empty")
        self._markets = tuple(markets)
        self._cash_entity = cash_entity
        self._history: dict[str, deque[float]] = {market_entity_name(listing): deque() for listing in self._markets}
        self._metadata: dict[str, MorphoMarketListing] = {}
        self._holder: str | None = None
        self._bars_held: int = 0
        self._bar_index: int = 0
        self._deferrals: list[dict] = []
        self._deferred_since: dict[str, int] = {}
        self._fresh: set[str] = set()
        super().__init__(**kwargs)

    # ------------------------------------------------------------- registry
    def set_up(self) -> None:
        """Register one entity per market plus the cash entity."""
        params = self._params
        validate_common_params(params)
        for field_name in ("TOP_K",):
            value = getattr(params, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{field_name} must be an integer >= 1, got {value!r}")
        max_hold = params.MAX_HOLD_BARS
        if isinstance(max_hold, bool) or not isinstance(max_hold, int) or max_hold < 0:
            raise ValueError(f"MAX_HOLD_BARS must be an integer >= 0, got {max_hold!r}")
        if not math.isfinite(params.MIN_EDGE_BPS):
            raise ValueError(f"MIN_EDGE_BPS must be finite, got {params.MIN_EDGE_BPS!r}")
        for listing in self._markets:
            self.register_entity(NamedEntity(
                entity_name=market_entity_name(listing),
                entity=build_supply_entity(params, listing),
            ))
        self.register_entity(NamedEntity(entity_name=params.CASH_ENTITY_NAME, entity=self._cash_entity))
        self._metadata = {market_entity_name(listing): listing for listing in self._markets}
        self._holder = params.CASH_ENTITY_NAME

    # ------------------------------------------------------------ telemetry
    @property
    def deferrals(self) -> list[dict]:
        """Deferral events recorded so far.

        ``kind == "exit"``: one event per blocked-exit episode with the blocked
        ``notional`` and, once the market drained, its duration in ``bars``.
        ``kind == "entry"``: one event per episode of consecutive bars in which
        the best market was skipped because it could not be exited, with the
        skipped ``notional`` and the episode length in ``bars``.
        """
        return [dict(event) for event in self._deferrals]

    def deferral_report(self) -> dict:
        """Summary of how often, how much, and how long exits were blocked.

        ``events`` / ``deferred_notional`` / ``*_bars_blocked`` describe exit
        deferrals only; entry refusals are reported separately as episodes so
        a position skipped for 48 bars counts once, not 48 times.
        """
        exits = [event for event in self._deferrals if event["kind"] == "exit"]
        entries = [event for event in self._deferrals if event["kind"] == "entry"]
        bars = [event["bars"] for event in exits if "bars" in event]
        return {
            "events": len(exits),
            "deferred_notional": float(sum(event["notional"] for event in exits)),
            "max_bars_blocked": int(max(bars)) if bars else 0,
            "mean_bars_blocked": float(sum(bars) / len(bars)) if bars else 0.0,
            "open_exit_deferrals": len(self._deferred_since),
            "entry_episodes": len(entries),
            "entry_blocked_bars": int(sum(event["bars"] for event in entries)),
        }

    # ---------------------------------------------------------------- step
    def _validate_observation(self, observation) -> None:
        super()._validate_observation(observation)
        self._fresh = set(observation.states)

    def predict(self) -> list[ActionToTake]:
        """Rank eligible markets on the lagged signal and rotate if it pays."""
        params = self._params
        self._bar_index += 1
        self._record_history()
        signals = self._signals()
        if not signals:
            return []

        # A blocked rotation leaves supply behind in a market we no longer
        # hold: retry it every bar, before choosing a new target.
        cleanup = self._cleanup()
        if cleanup:
            return cleanup

        candidates = self._eligible(signals)
        source = self._holder or params.CASH_ENTITY_NAME
        if not candidates:
            # forced exit: no market is eligible
            return self._rotate(source, params.CASH_ENTITY_NAME) if self._must_exit(source) else []
        ranking = sorted(candidates, key=lambda name: candidates[name], reverse=True)
        self._bars_held += 1
        desired = self._first_enterable(ranking, source)
        if desired is None:
            # Every candidate is too illiquid for this position (refusals recorded):
            # stay put, unless the holder itself must be left.
            return self._rotate(source, params.CASH_ENTITY_NAME) if self._must_exit(source) else []

        if not self._should_rotate(desired, ranking, signals):
            self._clear_deferrals()
            return []
        if source == desired:
            return []
        return self._rotate(source, desired)

    def _must_exit(self, source: str) -> bool:
        """``source`` is a fresh market holding that no longer passes the eligibility filters."""
        return (source != self._params.CASH_ENTITY_NAME and source in self._fresh
                and not self._selectable(source))

    def _first_enterable(self, ranking: list[str], source: str) -> str | None:
        """Best-ranked market whose withdrawable liquidity can take the position.

        Walks down the ranking instead of giving up on the top market: a
        position too large for the best market is not a reason to sit in cash
        when the second-best could hold it. Every market skipped on the way is
        recorded as an entry refusal. The current holder always
        passes (we are already in it).
        """
        amount = self._leg_amount(source)
        for name in ranking:
            if name == source:
                return name
            if self._destination_room(name) >= amount * (1.0 - 1e-12):
                return name
            self._defer_entry(name, amount)
        return None

    def _cleanup(self) -> list[ActionToTake]:
        """Withdraw capital stranded in a market that is no longer the holder."""
        holder = self._holder or self._params.CASH_ENTITY_NAME
        for name in self._history:
            if name == holder:
                continue
            entity = self.get_entity(name)
            if entity.balance <= 1e-9:
                continue
            source_amount = self._leg_amount(name)
            amount = self._transfer_amount(name, holder)
            self._defer_exit(name, self._blocked_notional(name))
            if holder != self._params.CASH_ENTITY_NAME and amount < source_amount * (1.0 - 1e-12):
                self._defer_entry(holder, source_amount)
            self._clear_deferrals()
            if amount <= 0.0:
                continue
            return self.transfer(from_entity=name, to_entity=holder, amount_in_notional=amount)
        return []

    def _rotate(self, source: str, desired: str) -> list[ActionToTake]:
        """Emit the two-leg transfer, sizing it to what the market can pay out."""
        params = self._params
        source_amount = self._leg_amount(source)
        # Entry gate (structural re-check; ``_first_enterable`` already applied it).
        if self._destination_room(desired) < source_amount * (1.0 - 1e-12):
            self._defer_entry(desired, source_amount)
            return []
        amount = self._transfer_amount(source, desired)
        # Record the blocked part before any early return: a fully gated exit is
        # a deferral even when no transfer can execute.
        self._defer_exit(source, self._blocked_notional(source))
        if amount <= 0.0:
            return []
        if source != params.CASH_ENTITY_NAME and self._blocked_notional(source) <= 1e-9:
            self._clear_deferrals()
        self._holder = desired
        self._bars_held = 0
        return self.transfer(from_entity=source, to_entity=desired, amount_in_notional=amount)

    # ------------------------------------------------------------- helpers
    def _transfer_amount(self, source: str, dest: str) -> float:
        """Transferable source amount capped by the destination's remaining capacity."""
        amount = min(self._leg_amount(source), self._destination_room(dest))
        if dest != self._params.CASH_ENTITY_NAME:
            gas, _ = self.get_entity(dest).leg_cost_model
            if amount <= 2.0 * gas:
                return 0.0
        return amount

    def _destination_room(self, dest: str) -> float:
        """Additional notional ``dest`` can accept while satisfying the entry gate."""
        if dest == self._params.CASH_ENTITY_NAME:
            return float("inf")
        entity = self.get_entity(dest)
        capacity = entity.liquidity_capacity(self._params.MIN_LIQUIDITY_MULTIPLE)
        return max(capacity - entity.balance, 0.0)

    def _leg_amount(self, source: str) -> float:
        """What the source can hand to a destination right now (gated for markets).

        The cash entity has no withdrawal constraint; a market entity is capped
        by its own withdrawable liquidity and pays its leg fee out of the
        position (:attr:`MorphoSupplyEntity.max_withdrawable_notional`).
        """
        entity = self.get_entity(source)
        if source == self._params.CASH_ENTITY_NAME:
            return float(entity.balance)
        return float(entity.max_withdrawable_notional)

    def _blocked_notional(self, market: str) -> float:
        """Part of a market position the gate will not release this bar (cash is never gated)."""
        if market == self._params.CASH_ENTITY_NAME:
            return 0.0
        entity = self.get_entity(market)
        return max(entity.balance - entity.withdrawable, 0.0)

    def _record_history(self) -> None:
        """Append this bar's signal value to each market's ring buffer."""
        params = self._params
        cap = params.SIGNAL_LAG_BARS + params.SIGNAL_WINDOW_BARS
        projected = params.OWN_IMPACT and params.RANK_ON_PROJECTED_APY
        size = self.total_balance if projected else 0.0
        for name, series in self._history.items():
            if name not in self._fresh:
                continue
            entity = self.get_entity(name)
            value = entity.projected_supply_apy(size) if projected else entity.global_state.supply_apy
            series.append(float(value))
            while len(series) > cap:
                series.popleft()

    def _signals(self) -> dict[str, float]:
        """Mean rate over the window ending ``SIGNAL_LAG_BARS`` bars before now."""
        lag, window = self._params.SIGNAL_LAG_BARS, self._params.SIGNAL_WINDOW_BARS
        signals = {}
        for name, series in self._history.items():
            values = list(series)
            if len(values) < lag + window:
                continue
            window_values = values[len(values) - lag - window:len(values) - lag]
            signals[name] = float(sum(window_values) / len(window_values))
        return signals

    def _eligible(self, signals: dict[str, float]) -> dict[str, float]:
        """Candidates for new capital: fresh on this bar and passing the eligibility filters."""
        return {
            name: signal for name, signal in signals.items()
            if name in self._fresh and self._selectable(name)
        }

    def _selectable(self, name: str) -> bool:
        """Static (LLTV, oracle, loan asset) and current-bar (pool size) eligibility of a market."""
        return is_selectable(self._params, self._metadata[name], self.get_entity(name).global_state)

    def _should_rotate(self, desired: str, ranking: list[str], signals: dict[str, float]) -> bool:
        params = self._params
        source = self._holder or params.CASH_ENTITY_NAME
        if source == params.CASH_ENTITY_NAME:
            return True  # idle capital should be earning
        if source not in self._fresh:
            return False  # the holder's feed stopped: hold, never trade on stale data
        if not self._selectable(source):
            return True  # the holder left the eligible universe: rotate out, spike or not
        if source not in signals:
            return True  # the holder lost its signal: leave on the best available
        if params.MAX_HOLD_BARS and self._bars_held >= params.MAX_HOLD_BARS:
            return desired != source
        if source in ranking[:params.TOP_K]:
            return False  # inside the acceptable band: hold, no churn on near-ties
        return (signals[desired] - signals[source]) * 1e4 > params.MIN_EDGE_BPS

    def _defer_exit(self, market: str, notional: float) -> None:
        """Record that part of a position could not be withdrawn this bar.

        One event per blocked episode (not per bar), carrying the blocked
        notional; the duration is stamped when the market finally empties.
        """
        if notional <= 1e-9 or market in self._deferred_since:
            return
        self._deferred_since[market] = self._bar_index
        self._deferrals.append({"market": market, "notional": notional, "bar": self._bar_index,
                                "kind": "exit"})

    def _defer_entry(self, market: str, notional: float) -> None:
        """Record that a market was skipped because it could not be exited.

        Consecutive bars of the same refusal extend one episode instead of
        adding an event per bar.
        """
        for event in reversed(self._deferrals):
            if event["kind"] != "entry" or event["market"] != market:
                continue
            if event["bar"] + event["bars"] == self._bar_index:
                event["bars"] += 1
                event["notional"] = max(event["notional"], notional)
                return
            break  # the last episode for this market ended earlier: start a new one
        self._deferrals.append({"market": market, "notional": notional, "bar": self._bar_index,
                                "kind": "entry", "bars": 1})

    def _clear_deferrals(self) -> None:
        """Close deferrals whose market can now be fully drained, stamping the duration."""
        for market in list(self._deferred_since):
            if self._blocked_notional(market) > 1e-9:
                continue
            started = self._deferred_since.pop(market)
            for event in reversed(self._deferrals):
                if event["market"] == market and event["kind"] == "exit":
                    event["bars"] = self._bar_index - started
                    break
