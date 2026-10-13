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
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, field

from fractal.core.base.entity import BaseEntity
from fractal.core.base.strategy import ActionToTake, BaseStrategy, BaseStrategyParams, NamedEntity
from fractal.core.entities.protocols.morpho_supply import MorphoSupplyEntity
from fractal.loaders.morpho_markets import MorphoMarketListing

__all__ = [
    "MorphoSupplyRotationParams",
    "MorphoSupplyRotationStrategy",
    "market_entity_name",
]


def market_entity_name(listing: MorphoMarketListing) -> str:
    """Registry name of the entity backing ``listing``.

    Public because observation builders must produce states under exactly these
    names; deriving them separately in an example would be a silent mismatch.
    """
    return f"SUPPLY_{listing.chain_id}_{listing.market_id[:10]}"


@dataclass
class MorphoSupplyRotationParams(BaseStrategyParams):
    """Illustrative hyperparameters; calibrate explicitly for each experiment.

    Attributes:
        TOP_K: size of the acceptable-holding band. The best-signal market is
            the target; the current market is kept if it is still inside the
            top ``TOP_K`` (avoids churn between near-ties).
        MAX_LLTV: inclusive LLTV ceiling for eligibility.
        ORACLE_TYPES: allowed oracle types; anything else is not selectable.
        LOAN_SYMBOLS: loan assets the strategy may supply (exact symbols).
        MIN_POOL_SIZE_USD: a market must have at least this ``supply_assets_usd``
            on the current bar to be selectable (size is read at execution
            time, like the liquidity gate).
        MIN_LIQUIDITY_MULTIPLE: required ratio of a target's withdrawable
            liquidity to the position being entered. ``1.0`` means "we could
            get out immediately"; ``0`` disables the check.
        MIN_EDGE_BPS: how much better (in basis points of APY) the target must
            be before a market-to-market rotation pays two legs of cost.
            Applies to every such rotation; leaving cash is always allowed.
        MAX_HOLD_BARS: force a re-selection once a position has been held this
            many bars (``0`` disables). It never forces an exit to cash.
        SIGNAL_LAG_BARS: bars of observation lag in the signal (``>= 1``).
        SIGNAL_WINDOW_BARS: bars averaged to form the signal level.
        RANK_ON_PROJECTED_APY: rank on the APY the portfolio's own size would
            earn in each market (self-dilution included) instead of the
            headline rate. No effect when ``OWN_IMPACT`` is off.
        CASH_ENTITY_NAME: registry name of the entity holding idle capital.
        OWN_IMPACT / ABSORPTION / COST_BPS_PER_LEG / GAS_USD_PER_LEG / BARS_PER_YEAR:
            passed straight to each :class:`MorphoSupplyEntity` — whether the
            position's own size moves the rate, the share of it borrowers
            take up, the per-leg cost in basis points, the fixed gas per leg in
            USD, and the bar length convention.
        USE_REWARDS / REWARD_WEIGHT: credit market reward campaigns, diluted by
            our own supply (see :class:`MorphoSupplyEntity`).
        GAS_USD_BY_CHAIN: ``{chain_id: USD gas per leg}``; overrides
            ``GAS_USD_PER_LEG`` for the markets on that chain.
    """
    TOP_K: int = 1
    MAX_LLTV: float = 0.915
    ORACLE_TYPES: tuple[str, ...] = ("ChainlinkOracle", "ChainlinkOracleV2")
    LOAN_SYMBOLS: tuple[str, ...] = ("USDC", "USDT")
    MIN_POOL_SIZE_USD: float = 1_000_000.0
    MIN_LIQUIDITY_MULTIPLE: float = 1.0
    MIN_EDGE_BPS: float = 10.0
    MAX_HOLD_BARS: int = 0
    SIGNAL_LAG_BARS: int = 1
    SIGNAL_WINDOW_BARS: int = 24
    RANK_ON_PROJECTED_APY: bool = True
    CASH_ENTITY_NAME: str = "CASH"
    OWN_IMPACT: bool = True
    ABSORPTION: float = 0.0
    USE_REWARDS: bool = False
    REWARD_WEIGHT: float = 1.0
    COST_BPS_PER_LEG: float = 0.0
    GAS_USD_PER_LEG: float = 0.0
    GAS_USD_BY_CHAIN: dict[int, float] = field(default_factory=dict)
    BARS_PER_YEAR: float = 8760.0


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
        self._history: dict[str, deque[float]] = {self._name(listing): deque() for listing in self._markets}
        self._metadata: dict[str, MorphoMarketListing] = {}
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
                entity=MorphoSupplyEntity(
                    own_impact=self._params.OWN_IMPACT,
                    absorption=self._params.ABSORPTION,
                    use_rewards=self._params.USE_REWARDS,
                    reward_weight=self._params.REWARD_WEIGHT,
                    cost_bps_per_leg=self._params.COST_BPS_PER_LEG,
                    gas_usd_per_leg=float(
                        self._params.GAS_USD_BY_CHAIN.get(listing.chain_id, self._params.GAS_USD_PER_LEG)),
                    bars_per_year=self._params.BARS_PER_YEAR,
                ),
            ))
        self.register_entity(NamedEntity(entity_name=self._params.CASH_ENTITY_NAME, entity=self._cash_entity))
        self._metadata = {self._name(listing): listing for listing in self._markets}
        self._holder = self._params.CASH_ENTITY_NAME

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
        if not candidates:
            return []
        ranking = sorted(candidates, key=lambda name: candidates[name], reverse=True)
        self._bars_held += 1
        source = self._holder or params.CASH_ENTITY_NAME
        desired = self._first_enterable(ranking, source)
        if desired is None:
            return []  # every candidate is too illiquid for this position: stay put, refusals recorded

        if not self._should_rotate(desired, ranking, signals):
            self._clear_deferrals()
            return []
        if source == desired:
            return []
        return self._rotate(source, desired)

    def _first_enterable(self, ranking: list[str], source: str) -> str | None:
        """Best-ranked market whose withdrawable liquidity can take the position.

        Walks down the ranking instead of giving up on the top market: a
        position too large for the best market is not a reason to sit in cash
        when the second-best could hold it. Every market skipped on the way is
        recorded as an entry refusal. The current holder always
        passes (we are already in it).
        """
        amount = self._leg_amount(source)
        multiple = self._params.MIN_LIQUIDITY_MULTIPLE
        for name in ranking:
            if name == source:
                return name
            if self.get_entity(name).withdrawable >= multiple * amount:
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
            amount = self._transfer_amount(name, holder)
            self._defer_exit(name, self._blocked_notional(name))
            self._clear_deferrals()
            if amount <= 0.0:
                return []
            return self.transfer(
                from_entity=name, to_entity=holder,
                amount_in_notional=lambda strategy, source=name, dest=holder: self._transfer_amount(source, dest),
            )
        return []

    def _rotate(self, source: str, desired: str) -> list[ActionToTake]:
        """Emit the two-leg transfer, sizing it to what the market can pay out."""
        params = self._params
        amount = self._transfer_amount(source, desired)
        # Record the blocked part before any early return: a fully gated exit is
        # a deferral even when no transfer can execute.
        self._defer_exit(source, self._blocked_notional(source))
        if amount <= 0.0:
            return []
        target_entity = self.get_entity(desired)
        # Entry gate (structural re-check; ``_first_enterable`` already applied it).
        if target_entity.withdrawable < params.MIN_LIQUIDITY_MULTIPLE * amount:
            self._defer_entry(desired, amount)
            return []
        if source != params.CASH_ENTITY_NAME and self._blocked_notional(source) <= 1e-9:
            self._clear_deferrals()
        self._holder = desired
        self._bars_held = 0
        return self.transfer(
            from_entity=source,
            to_entity=desired,
            amount_in_notional=lambda strategy: self._transfer_amount(source, desired),
        )

    # ------------------------------------------------------------- helpers
    def _transfer_amount(self, source: str, dest: str) -> float:
        """:meth:`_leg_amount`, or ``0`` when a market destination's fixed gas would eat the leg."""
        amount = self._leg_amount(source)
        if dest != self._params.CASH_ENTITY_NAME:
            gas, _ = self.get_entity(dest).leg_cost_model
            if amount <= 2.0 * gas:
                return 0.0
        return amount

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
        return {
            name: signal for name, signal in signals.items()
            if self._selectable(name)
        }

    def _selectable(self, name: str) -> bool:
        """Static (LLTV, oracle, loan asset) and current-bar (pool size) eligibility of a market."""
        params, listing = self._params, self._metadata[name]
        return (
            listing.lltv <= params.MAX_LLTV
            and listing.oracle_type in params.ORACLE_TYPES
            and listing.loan_symbol in params.LOAN_SYMBOLS
            and self.get_entity(name).global_state.supply_assets_usd >= params.MIN_POOL_SIZE_USD
        )

    def _should_rotate(self, desired: str, ranking: list[str], signals: dict[str, float]) -> bool:
        params = self._params
        source = self._holder or params.CASH_ENTITY_NAME
        if source == params.CASH_ENTITY_NAME:
            return True  # idle capital should be earning
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
