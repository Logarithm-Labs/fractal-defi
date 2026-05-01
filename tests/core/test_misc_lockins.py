"""Cross-cutting lock-ins for ``Observation`` equality, ``BaseStrategy``
construction, ``SQLiteObservationsStorage`` lifecycle and a few small
strategy/entity invariants."""
from datetime import datetime, timezone
from dataclasses import dataclass

import pytest

from fractal.core.base import (Action, BaseStrategy, BaseStrategyParams,
                               EntityException, GlobalState,
                               NamedEntity, Observation)
from fractal.core.base.observations import SQLiteObservationsStorage
from fractal.core.entities import (SimpleLendingEntity, SimplePerpEntity,
                                   SimpleSpotExchange, SimpleSpotExchangeGlobalState)
from fractal.core.entities.simple.lending import SimpleLendingException

UTC = timezone.utc


@dataclass
class _S(GlobalState):
    price: float = 0.0


@pytest.mark.core
def test_observation_eq_returns_not_implemented_for_non_observation():
    obs = Observation(timestamp=datetime(2024, 1, 1), states={"X": _S(price=100)})
    # Comparing to non-Observation must NOT raise — Python falls back to identity.
    assert obs != 42
    assert obs != "hi"
    assert obs != {"X": _S(price=100)}


@pytest.mark.core
def test_observation_eq_compares_timestamp_and_states():
    """Same states + same timestamp → equal."""
    a = Observation(timestamp=datetime(2024, 1, 1), states={"X": _S(price=100)})
    b = Observation(timestamp=datetime(2024, 1, 1), states={"X": _S(price=100)})
    assert a == b
    assert hash(a) == hash(b)


@pytest.mark.core
def test_observation_eq_distinguishes_by_timestamp():
    """Same states, different timestamps → not equal (hash diverges too)."""
    a = Observation(timestamp=datetime(2024, 1, 1), states={"X": _S(price=100)})
    b = Observation(timestamp=datetime(2024, 1, 2), states={"X": _S(price=100)})
    assert a != b
    assert hash(a) != hash(b)


@pytest.mark.core
def test_observation_eq_distinguishes_by_states():
    """Same timestamp, different states → not equal."""
    a = Observation(timestamp=datetime(2024, 1, 1), states={"X": _S(price=100)})
    b = Observation(timestamp=datetime(2024, 1, 1), states={"X": _S(price=200)})
    assert a != b


@pytest.mark.core
def test_observation_set_does_not_collapse_distinct_timestamps():
    """Two snapshots with same states but different timestamps must coexist
    in a set — earlier behavior collapsed them."""
    a = Observation(timestamp=datetime(2024, 1, 1), states={"X": _S(price=100)})
    b = Observation(timestamp=datetime(2024, 1, 2), states={"X": _S(price=100)})
    assert len({a, b}) == 2


@pytest.mark.core
def test_execute_unavailable_action_error_message_clean():
    e = SimpleSpotExchange(trading_fee=0.0)
    e.update_state(SimpleSpotExchangeGlobalState(close=100))
    with pytest.raises(EntityException) as excinfo:
        e.execute(Action("explode_pool"))
    msg = str(excinfo.value)
    # No backslash-continuation garbage of repeated indentation spaces.
    assert "  " * 4 not in msg
    assert "'explode_pool' is not available" in msg
    assert "SimpleSpotExchange" in msg


class _ParamsTestStrategy(BaseStrategy):
    def set_up(self):
        pass

    def predict(self):
        return []


@pytest.mark.core
def test_params_property_returns_a_copy():
    s = _ParamsTestStrategy(params={"FOO": 1})
    p1 = s.params
    p1["FOO"] = 999  # mutate the returned dict
    p2 = s.params
    assert p2["FOO"] == 1, "params property should return an isolated copy"


class _CustomValidatingStrategy(BaseStrategy):
    STRICT_OBSERVATIONS = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.calls = 0

    def set_up(self):
        self.register_entity(NamedEntity("X", SimpleSpotExchange(trading_fee=0.0)))

    def predict(self):
        return []

    def _validate_observation(self, observation):  # type: ignore[override]
        self.calls += 1
        super()._validate_observation(observation)


@pytest.mark.core
def test_validate_observation_can_be_overridden_in_subclass():
    """Single-underscore (vs name-mangled double) lets subclasses hook in."""
    s = _CustomValidatingStrategy()
    s.step(Observation(
        timestamp=datetime(2024, 1, 1),
        states={"X": SimpleSpotExchangeGlobalState(close=100)},
    ))
    assert s.calls == 1


class _SnapshotStrategy(BaseStrategy):
    STRICT_OBSERVATIONS = False

    def set_up(self):
        self.register_entity(NamedEntity("X", SimpleSpotExchange(trading_fee=0.0)))

    def predict(self):
        return []


@pytest.mark.core
def test_run_snapshots_global_state_independently_of_later_mutation():
    """Mutating an entity's global_state after run() must NOT change the snapshot."""
    s = _SnapshotStrategy()
    obs = [
        Observation(timestamp=datetime(2024, 1, 1),
                    states={"X": SimpleSpotExchangeGlobalState(close=100)}),
        Observation(timestamp=datetime(2024, 1, 2),
                    states={"X": SimpleSpotExchangeGlobalState(close=200)}),
    ]
    result = s.run(obs)
    snap_close = result.global_states[0]["X"].close  # taken right after first step
    # Mutate the state of the entity post-run; the snapshot must not move.
    s.get_entity("X")._global_state.close = 999  # type: ignore[attr-defined]
    assert result.global_states[0]["X"].close == snap_close


class _SetUpUsesDebugStrategy(BaseStrategy):
    STRICT_OBSERVATIONS = False

    def set_up(self):
        # If the logger was not yet initialized this would silently no-op;
        # we only care that it does not raise here.
        self._debug("inside set_up")
        self.register_entity(NamedEntity("X", SimpleSpotExchange(trading_fee=0.0)))

    def predict(self):
        return []


@pytest.mark.core
def test_set_up_can_use_debug_when_logger_initialized_first(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # keep runs/ artifacts under tmp_path
    s = _SetUpUsesDebugStrategy(debug=True)
    assert s.logger is not None
    assert "X" in s.get_all_available_entities()


@pytest.mark.core
def test_base_entity_init_does_not_set_none_states():
    """``_initialize_states`` populates states; the parent init no longer
    pre-sets None to mask that responsibility."""
    e = SimpleSpotExchange(trading_fee=0.0)
    assert e._internal_state is not None
    assert e._global_state is not None


@pytest.mark.core
def test_abstract_set_up_and_predict_take_no_args_only_self():
    import inspect
    assert list(inspect.signature(BaseStrategy.set_up).parameters) == ["self"]
    assert list(inspect.signature(BaseStrategy.predict).parameters) == ["self"]


@pytest.mark.core
def test_create_logger_signature_only_self():
    import inspect
    params = list(inspect.signature(BaseStrategy._create_logger).parameters)
    assert params == ["self"]


class _DebugTrackerStrategy(BaseStrategy):
    STRICT_OBSERVATIONS = False

    def set_up(self):
        self.register_entity(NamedEntity("X", SimpleSpotExchange(trading_fee=0.0)))

    def predict(self):
        return []


@pytest.mark.core
def test_debug_when_logger_disabled_is_noop():
    s = _DebugTrackerStrategy(debug=False)
    assert s.logger is None
    s._debug("anything")  # must not raise


@pytest.mark.core
def test_simple_perp_initialize_states_no_local_annotations():
    """``_initialize_states`` body should not redeclare the field type."""
    import inspect
    src = inspect.getsource(SimplePerpEntity._initialize_states)
    assert "self._internal_state: SimplePerpInternalState" not in src
    assert "self._global_state: SimplePerpGlobalState" not in src


@pytest.mark.core
def test_simple_perp_config_attrs_set_before_initialize_states():
    """Override ``_initialize_states`` in a subclass and rely on trading_fee.
    If the parent init still ran ``_initialize_states`` before our config
    setters, the assertion below would fail."""
    seen: dict = {}

    class _Probe(SimplePerpEntity):
        def _initialize_states(self):
            seen["trading_fee_at_init_states"] = self.trading_fee
            super()._initialize_states()

    _Probe(trading_fee=0.0007)
    assert seen["trading_fee_at_init_states"] == pytest.approx(0.0007)


@pytest.mark.core
def test_base_strategy_params_is_not_a_dataclass():
    from dataclasses import is_dataclass
    assert not is_dataclass(BaseStrategyParams), (
        "BaseStrategyParams should be a plain class — dataclass decorator was misleading."
    )


@pytest.mark.core
def test_base_strategy_params_dict_form_still_works():
    p = BaseStrategyParams(data={"FOO": 1, "BAR": "x"})
    assert p.FOO == 1 and p.BAR == "x"


@pytest.mark.core
def test_sqlite_storage_default_db_path_under_tempfile():
    import tempfile
    storage = SQLiteObservationsStorage()
    try:
        assert storage.db_path.startswith(tempfile.gettempdir())
        assert storage.db_path.endswith(".db")
    finally:
        storage.close()


@pytest.mark.core
def test_sqlite_storage_context_manager_closes_connection(tmp_path):
    db = str(tmp_path / "obs.db")
    with SQLiteObservationsStorage(db_path=db) as storage:
        assert storage.connection is not None
    assert storage.connection is None  # closed on __exit__


@pytest.mark.core
def test_sqlite_storage_close_is_idempotent(tmp_path):
    db = str(tmp_path / "obs.db")
    storage = SQLiteObservationsStorage(db_path=db)
    storage.close()
    storage.close()  # must not raise


@pytest.mark.core
def test_calculate_repay_raises_when_ltv_is_inf():
    e = SimpleLendingEntity()
    e.update_state(__import__(
        "fractal.core.entities.simple.lending", fromlist=["SimpleLendingGlobalState"],
    ).SimpleLendingGlobalState(collateral_price=1, debt_price=1))
    e._internal_state.collateral = 0.0
    e._internal_state.borrowed = 100.0  # debt with no collateral → LTV = inf
    with pytest.raises(SimpleLendingException, match="non-finite"):
        e.calculate_repay(0.5)
