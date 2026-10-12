"""Morpho Blue LP rotation: hold the highest-yielding eligible market.

One position, rotated between isolated Morpho markets, with the exit gate
respected at every step. The design follows what the research actually found
rather than the naive formulation:

* **Rank on a smoothed level, not the entry snapshot.** RQ2 (CONTEXT §8 M1b)
  falsified the timing edge: entering a market whose *current* rate is at the
  top decile realised 56.8 bp less over 7 days, statistically the same as the
  market's own trailing 30-day average. What did persist is the cross-sectional
  level (52.2% of top-decile markets are still top-decile a week later versus
  10% by chance). So the signal is the mean rate over ``SIGNAL_WINDOW_BARS``
  bars ending ``SIGNAL_LAG_BARS`` bars before now — ``SIGNAL_WINDOW_BARS = 1``
  reproduces the falsified snapshot chase.
* **The signal may never see the bar it trades on.** ``SIGNAL_LAG_BARS`` models
  the API's indexing lag (RQ8): with ``k`` bars of lag the decision at ``t``
  uses only bars at ``t−k`` and older.
* **The exit gate is respected before entering, not after.** A market is only
  entered when its withdrawable liquidity covers ``MIN_LIQUIDITY_MULTIPLE``
  times the position, and the transfer is sized to ``min(position,
  withdrawable)`` at execute time. Anything that cannot be withdrawn stays
  supplied, accrues, is retried on the next bar, and is recorded as a deferral
  event — the RQ3 metric.
* **Risk filters are parameters, not registry decisions** (A4): ``MAX_LLTV`` and
  ``ORACLE_TYPES`` select among everything registered, so one registry serves
  every run and RQ7 is a sweep over them.

Costs are charged by the entity on each leg (A3), so nothing here touches the
cash entity for fees and ``StrategyResult.fees_paid`` reflects the real drag.

The strategy is given its market registry and a cash entity at construction
time, because ``set_up`` runs inside ``BaseStrategy.__init__``::

    strategy = MorphoLpRotationStrategy(
        markets=registry,                       # list[MorphoMarketListing]
        cash_entity=SimpleSpotExchange(trading_fee=0.0),
        params=MorphoLpRotationParams(TOP_K=3, MAX_LLTV=0.915),
    )
    result = strategy.run(observations)
"""
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass

from fractal.core.base.entity import BaseEntity
from fractal.core.base.strategy import ActionToTake, BaseStrategy, BaseStrategyParams, NamedEntity
from fractal.core.entities.protocols.morpho_lp import MorphoLpEntity
from fractal.loaders.morpho_markets import MorphoMarketListing

__all__ = [
    "MorphoLpRotationParams",
    "MorphoLpRotationStrategy",
    "market_entity_name",
]


def market_entity_name(listing: MorphoMarketListing) -> str:
    """Registry name of the entity backing ``listing``.

    Public because observation builders must produce states under exactly these
    names; deriving them separately in an example would be a silent mismatch.
    """
    return f"LP_{listing.chain_id}_{listing.market_id[:10]}"


@dataclass
class MorphoLpRotationParams(BaseStrategyParams):
    """Hyperparameters. Every default is a research-stated choice, not a tuned value.

    Attributes:
        TOP_K: size of the acceptable-holding band. The best-signal market is
            the target; the current market is kept if it is still inside the
            top ``TOP_K`` (avoids churn between near-ties).
        MAX_LLTV: inclusive LLTV ceiling for eligibility (RQ7 sweeps it).
        ORACLE_TYPES: allowed oracle types; anything else is not selectable.
        MIN_LIQUIDITY_MULTIPLE: required ratio of a target's withdrawable
            liquidity to the position being entered. ``1.0`` means "we could
            get out immediately"; ``0`` disables the check.
        MIN_EDGE_BPS: how much better (in basis points of APY) the target must
            be before paying two legs of cost is worth it.
        MAX_HOLD_BARS: force a re-selection once a position has been held this
            many bars (``0`` disables). It never forces an exit to cash.
        SIGNAL_LAG_BARS: bars of observation lag in the signal (``>= 1``).
        SIGNAL_WINDOW_BARS: bars averaged to form the signal level.
        CASH_ENTITY_NAME: registry name of the entity holding idle capital.
        OWN_IMPACT / ABSORPTION / COST_BPS_PER_LEG / BARS_PER_YEAR: passed
            straight to each :class:`MorphoLpEntity` — whether the position's
            own size moves the rate (RQ4), the share of it borrowers take up,
            the per-leg cost in basis points, and the bar length convention.
    """
    TOP_K: int = 1
    MAX_LLTV: float = 0.915
    ORACLE_TYPES: tuple[str, ...] = ("ChainlinkOracle", "ChainlinkOracleV2")
    MIN_LIQUIDITY_MULTIPLE: float = 1.0
    MIN_EDGE_BPS: float = 10.0
    MAX_HOLD_BARS: int = 0
    SIGNAL_LAG_BARS: int = 1
    SIGNAL_WINDOW_BARS: int = 24
    CASH_ENTITY_NAME: str = "CASH"
    OWN_IMPACT: bool = True
    ABSORPTION: float = 0.0
    COST_BPS_PER_LEG: float = 0.0
    BARS_PER_YEAR: float = 8760.0


class MorphoLpRotationStrategy(BaseStrategy[MorphoLpRotationParams]):
    """Rotate one LP position across the registered Morpho markets."""

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
        # ``set_up`` (see :class:`MorphoLpEntity`'s use of the same ordering).
        if not markets:
            raise ValueError("markets must not be empty")
        self._markets = tuple(markets)
        self._cash_entity = cash_entity
        self._history: dict[str, deque[float]] = {self._name(listing): deque() for listing in self._markets}
        self._holder: str | None = None
        self._bars_held: int = 0
        self._bar_index: int = 0
        self._deferrals: list[dict] = []
        self._deferred_since: dict[str, int] = {}
        self._fresh: set[str] = set()
        super().__init__(**kwargs)

    # ------------------------------------------------------------- registry
    @staticmethod
    def _name(listing: MorphoMarketListing) -> str:
        return market_entity_name(listing)

    def set_up(self) -> None:
        """Register one entity per market plus the cash entity."""
        for listing in self._markets:
            self.register_entity(NamedEntity(
                entity_name=self._name(listing),
                entity=MorphoLpEntity(
                    own_impact=self._params.OWN_IMPACT,
                    absorption=self._params.ABSORPTION,
                    cost_bps_per_leg=self._params.COST_BPS_PER_LEG,
                    bars_per_year=self._params.BARS_PER_YEAR,
                ),
            ))
        self.register_entity(NamedEntity(entity_name=self._params.CASH_ENTITY_NAME, entity=self._cash_entity))
        self._metadata = {self._name(listing): listing for listing in self._markets}
        self._holder = self._params.CASH_ENTITY_NAME

    # ------------------------------------------------------------ telemetry
    @property
    def deferrals(self) -> list[dict]:
        """Deferral events recorded so far (the RQ3 metric's raw material)."""
        return list(self._deferrals)

    def deferral_report(self) -> dict:
        """RQ3 summary: how often, how much, and how long exits were blocked."""
        notional = [event["notional"] for event in self._deferrals]
        bars = [event["bars"] for event in self._deferrals if "bars" in event]
        return {
            "events": len(self._deferrals),
            "deferred_notional": float(sum(notional)),
            "max_bars_blocked": int(max(bars)) if bars else 0,
            "mean_bars_blocked": float(sum(bars) / len(bars)) if bars else 0.0,
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
        if not candidates:
            return []
        desired = max(candidates, key=lambda name: candidates[name])
        ranking = sorted(candidates, key=lambda name: candidates[name], reverse=True)
        self._bars_held += 1

        if not self._should_rotate(desired, ranking, signals):
            self._clear_deferrals()
            return []
        source = self._holder or params.CASH_ENTITY_NAME
        if source == desired:
            return []
        return self._rotate(source, desired)

    def _cleanup(self) -> list[ActionToTake]:
        """Withdraw capital stranded in a market that is no longer the holder."""
        holder = self._holder or self._params.CASH_ENTITY_NAME
        for name in self._history:
            if name == holder:
                continue
            entity = self.get_entity(name)
            if entity.balance <= 1e-9:
                continue
            amount = self._leg_amount(name)
            self._defer_exit(name, entity.balance - amount)
            self._clear_deferrals()
            if amount <= 0.0:
                return []
            return self.transfer(
                from_entity=name, to_entity=holder,
                amount_in_notional=lambda strategy, source=name: self._leg_amount(source),
            )
        return []

    def _rotate(self, source: str, desired: str) -> list[ActionToTake]:
        """Emit the two-leg transfer, sizing it to what the market can pay out."""
        params = self._params
        source_entity = self.get_entity(source)
        amount = self._leg_amount(source)
        # Record the blocked part before any early return: a fully gated exit is
        # exactly the case RQ3 must count.
        self._defer_exit(source, source_entity.balance - amount)
        if amount <= 0.0:
            return []
        target_entity = self.get_entity(desired)
        # Entry gate: never supply into a market we could not leave.
        if target_entity.withdrawable < params.MIN_LIQUIDITY_MULTIPLE * amount:
            self._defer_entry(desired, amount)
            return []
        if source != params.CASH_ENTITY_NAME and amount >= source_entity.balance - 1e-9:
            self._clear_deferrals()
        self._holder = desired
        self._bars_held = 0
        return self.transfer(
            from_entity=source,
            to_entity=desired,
            amount_in_notional=lambda strategy: self._leg_amount(source),
        )

    # ------------------------------------------------------------- helpers
    def _leg_amount(self, source: str) -> float:
        """What the source can actually hand over right now (gated for markets).

        The cash entity has no withdrawal constraint; a market entity is capped
        by its own withdrawable liquidity.
        """
        entity = self.get_entity(source)
        if source == self._params.CASH_ENTITY_NAME:
            return float(entity.balance)
        return float(min(entity.balance, entity.withdrawable))

    def _record_history(self) -> None:
        """Append this bar's rate to each market's ring buffer."""
        cap = self._params.SIGNAL_LAG_BARS + self._params.SIGNAL_WINDOW_BARS
        for name, series in self._history.items():
            if name not in self._fresh:
                continue
            series.append(float(self.get_entity(name).global_state.supply_apy))
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
        params = self._params
        return {
            name: signal for name, signal in signals.items()
            if self._metadata[name].lltv <= params.MAX_LLTV
            and self._metadata[name].oracle_type in params.ORACLE_TYPES
        }

    def _should_rotate(self, desired: str, ranking: list[str], signals: dict[str, float]) -> bool:
        params = self._params
        source = self._holder or params.CASH_ENTITY_NAME
        if source == params.CASH_ENTITY_NAME:
            return True  # idle capital should be earning
        if source not in ranking[:params.TOP_K]:
            return True
        if params.MAX_HOLD_BARS and self._bars_held >= params.MAX_HOLD_BARS:
            return desired != source
        return (signals[desired] - signals[source]) * 1e4 > params.MIN_EDGE_BPS

    def _defer_exit(self, market: str, notional: float) -> None:
        """Record that part of a position could not be withdrawn this bar.

        One event per blocked episode (not per bar), carrying the blocked
        notional; the duration is stamped when the market finally empties.
        """
        if notional <= 0.0 or market in self._deferred_since:
            return
        self._deferred_since[market] = self._bar_index
        self._deferrals.append({"market": market, "notional": notional, "bar": self._bar_index,
                                "kind": "exit"})

    def _defer_entry(self, market: str, notional: float) -> None:
        """Record that a market was skipped because it could not be exited."""
        self._deferrals.append({"market": market, "notional": notional, "bar": self._bar_index,
                                "kind": "entry"})

    def _clear_deferrals(self) -> None:
        """Close deferrals whose market can now be fully drained, stamping the duration."""
        for market in list(self._deferred_since):
            entity = self.get_entity(market)
            if entity.balance - self._leg_amount(market) > 1e-9:
                continue
            started = self._deferred_since.pop(market)
            for event in reversed(self._deferrals):
                if event["market"] == market and event["kind"] == "exit":
                    event["bars"] = self._bar_index - started
                    break
