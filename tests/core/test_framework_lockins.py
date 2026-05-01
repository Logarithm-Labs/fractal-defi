"""Cross-cutting framework lock-ins.

* ``DefaultLogger`` does not wipe the global ``loguru`` sink list and
  does not duplicate strategy debug output to stderr.
* ``BaseStrategy.step`` validates the observation BEFORE writing it to
  ``observations_storage``; rejected observations stay out.
* Neither the loader cache root nor the logger output root reads
  ``PYTHONPATH`` (it's a colon-separated import list, not a directory).
* ``get_all_available_entities`` returns a read-only mapping view.
* GraphQL loaders validate EVM-shape addresses up front.
* ``UniswapV3LPEntity.update_state`` rejects degenerate snapshots.
* ``UniswapV3Loader`` lifecycle methods raise ``NotImplementedError``.
"""
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pytest

from fractal.core.base import BaseStrategy, BaseStrategyParams, GlobalState, NamedEntity, Observation
from fractal.core.base.observations import SQLiteObservationsStorage
from fractal.core.entities import SimpleSpotExchange


@dataclass
class _S(GlobalState):
    price: float = 0.0


class _Strat(BaseStrategy[BaseStrategyParams]):
    STRICT_OBSERVATIONS = True

    def set_up(self):
        self.register_entity(NamedEntity("X", SimpleSpotExchange(trading_fee=0.0)))

    def predict(self):
        return []


@pytest.mark.core
def test_invalid_observation_does_not_reach_storage(tmp_path):
    """An observation that fails validation must NOT be persisted —
    storage write happens only after ``_validate_observation`` succeeds."""
    db = tmp_path / "obs.db"
    storage = SQLiteObservationsStorage(str(db))
    s = _Strat(observations_storage=storage)
    # ``UNKNOWN`` is not a registered entity → validation rejects.
    bad = Observation(timestamp=datetime(2024, 1, 1),
                      states={"UNKNOWN": _S(price=1.0)})
    with pytest.raises(ValueError):
        s.step(bad)
    assert not list(storage.read())
    storage.close()


@pytest.mark.core
def test_valid_observation_is_persisted_once(tmp_path):
    """A valid observation is written to storage exactly once per step."""
    db = tmp_path / "obs.db"
    storage = SQLiteObservationsStorage(str(db))
    s = _Strat(observations_storage=storage)
    obs = Observation(timestamp=datetime(2024, 1, 1, 12, 0),
                      states={"X": _S(price=100.0)})
    s.step(obs)
    persisted = list(storage.read())
    assert len(persisted) == 1
    storage.close()


@pytest.mark.core
def test_default_logger_does_not_wipe_external_sink(tmp_path, monkeypatch):
    """Creating a debug-mode strategy must NOT remove sinks added by
    other code in the same process. An earlier implementation called
    the global ``logger.remove()`` which silently deleted every sink.
    """
    from loguru import logger

    monkeypatch.setenv("FRACTAL_RUNS_PATH", str(tmp_path))

    external_sink = tmp_path / "external.log"
    external_id = logger.add(str(external_sink), level="DEBUG")
    try:
        # Instantiating the strategy in debug mode would have wiped sinks.
        _Strat(debug=True)
        # External sink must still be alive.
        logger.bind().debug("hello-from-external-test")
    finally:
        logger.remove(external_id)

    assert external_sink.exists()
    contents = external_sink.read_text(encoding="utf-8")
    assert "hello-from-external-test" in contents


@pytest.mark.core
def test_default_logger_does_not_print_strategy_debug_to_stderr(
        tmp_path, monkeypatch, capsys):
    """``debug=True`` must keep the strategy's own debug output OUT of
    stderr — only the per-instance file sink should receive it.

    Loguru installs a default stderr sink at handler id 0 on import.
    An earlier fix stopped wiping ALL sinks but left that default in
    place, which fanned every ``self._debug(...)`` call out to the
    console. ``DefaultLogger`` now strips handler 0 once per process.
    """
    monkeypatch.setenv("FRACTAL_RUNS_PATH", str(tmp_path))
    s = _Strat(debug=True)
    s._debug("strategy-private-debug-line")
    captured = capsys.readouterr()
    assert "strategy-private-debug-line" not in captured.err
    assert "strategy-private-debug-line" not in captured.out

    log_files = list(Path(s.logger.logs_path).glob("*.log"))
    assert log_files, f"no log file under {s.logger.logs_path}"
    contents = log_files[0].read_text(encoding="utf-8")
    assert "strategy-private-debug-line" in contents


@pytest.mark.core
def test_loader_base_path_ignores_pythonpath(monkeypatch, tmp_path):
    """``PYTHONPATH`` (colon-separated import list) must NOT be used
    as a single filesystem root. With no ``DATA_PATH`` set, the cache
    must fall back to the current working directory."""
    from fractal.loaders.base_loader import Loader, LoaderType

    monkeypatch.delenv("DATA_PATH", raising=False)
    monkeypatch.setenv("PYTHONPATH", "/some:/colon:/separated:/list")
    monkeypatch.chdir(tmp_path)

    class _DummyLoader(Loader):
        def extract(self):
            return None

        def transform(self):
            return None

        def read(self, with_run=False):
            return None

    loader = _DummyLoader(loader_type=LoaderType.CSV)
    assert ":" not in loader._base_path
    assert os.path.commonpath([str(tmp_path), loader._base_path]) == str(tmp_path)


@pytest.mark.core
def test_logger_runs_path_ignores_pythonpath(monkeypatch, tmp_path):
    """Same for the logger's runs/ directory — uses
    ``FRACTAL_RUNS_PATH`` or cwd, never ``PYTHONPATH``."""
    from fractal.core.base.strategy.logger import DefaultLogger

    monkeypatch.delenv("FRACTAL_RUNS_PATH", raising=False)
    monkeypatch.setenv("PYTHONPATH", "/a:/b:/c")
    monkeypatch.chdir(tmp_path)

    lg = DefaultLogger(class_name="DummyStrat")
    try:
        assert ":" not in lg.base_artifacts_path
        assert str(tmp_path) in lg.base_artifacts_path
    finally:
        lg.close()


@pytest.mark.core
def test_logger_runs_path_honors_fractal_runs_path(monkeypatch, tmp_path):
    """``FRACTAL_RUNS_PATH`` is the explicit knob and wins over cwd."""
    from fractal.core.base.strategy.logger import DefaultLogger

    explicit = tmp_path / "explicit_runs"
    monkeypatch.setenv("FRACTAL_RUNS_PATH", str(explicit))

    lg = DefaultLogger(class_name="DummyStrat")
    try:
        assert str(explicit) in lg.base_artifacts_path
    finally:
        lg.close()


@pytest.mark.core
def test_get_all_available_entities_returns_read_only_view():
    """External callers must not be able to bypass ``register_entity``
    by mutating the registry through this getter."""
    s = _Strat()
    view = s.get_all_available_entities()
    assert "X" in view
    with pytest.raises(TypeError):
        view["Y"] = SimpleSpotExchange(trading_fee=0.0)  # type: ignore[index]


@pytest.mark.core
def test_validate_evm_address_accepts_canonical_address():
    from fractal.loaders.thegraph.base_graph_loader import validate_evm_address
    out = validate_evm_address("0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2")
    assert out == "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2"


@pytest.mark.core
def test_validate_evm_address_rejects_short_or_garbage():
    from fractal.loaders.thegraph.base_graph_loader import GraphLoaderException, validate_evm_address
    with pytest.raises(GraphLoaderException):
        validate_evm_address("0x123")  # too short
    with pytest.raises(GraphLoaderException):
        validate_evm_address("0x" + "z" * 40)  # non-hex
    with pytest.raises(GraphLoaderException):
        validate_evm_address('0xC0"; DROP TABLE pools; --')  # injection-shaped


@pytest.mark.core
def test_aave_loader_rejects_non_address_asset():
    """Lock-in via Aave loader: a bogus asset_address raises at __init__."""
    from fractal.loaders.aave import AaveV3ArbitrumLoader
    from fractal.loaders.thegraph.base_graph_loader import GraphLoaderException
    with pytest.raises(GraphLoaderException, match="asset_address"):
        AaveV3ArbitrumLoader(asset_address="not-an-address")


@pytest.mark.core
def test_uniswap_v3_lp_update_state_rejects_non_positive_price():
    """``update_state`` with an open position must reject ``price <= 0``
    rather than silently producing nonsense token amounts."""
    from fractal.core.base.entity import EntityException
    from fractal.core.entities.protocols.uniswap_v3_lp import (
        UniswapV3LPConfig,
        UniswapV3LPEntity,
        UniswapV3LPGlobalState,
    )

    e = UniswapV3LPEntity(UniswapV3LPConfig())
    e.action_deposit(1000)
    e.update_state(UniswapV3LPGlobalState(price=2000.0, tvl=1e9, volume=0,
                                          fees=0, liquidity=1e15))
    e.action_open_position(amount_in_notional=500.0, price_lower=1500, price_upper=2500)
    # Now try to update with a non-positive price.
    with pytest.raises(EntityException, match="price"):
        e.update_state(UniswapV3LPGlobalState(price=0.0, tvl=1e9, volume=0,
                                              fees=0, liquidity=1e15))


@pytest.mark.core
def test_uniswap_v3_lp_update_state_rejects_negative_liquidity():
    from fractal.core.base.entity import EntityException
    from fractal.core.entities.protocols.uniswap_v3_lp import (
        UniswapV3LPConfig,
        UniswapV3LPEntity,
        UniswapV3LPGlobalState,
    )

    e = UniswapV3LPEntity(UniswapV3LPConfig())
    e.action_deposit(1000)
    e.update_state(UniswapV3LPGlobalState(price=2000.0, tvl=1e9, volume=0,
                                          fees=0, liquidity=1e15))
    e.action_open_position(amount_in_notional=500.0, price_lower=1500, price_upper=2500)
    with pytest.raises(EntityException, match="liquidity"):
        e.update_state(UniswapV3LPGlobalState(price=2100.0, tvl=1e9, volume=0,
                                              fees=0, liquidity=-1.0))


@pytest.mark.core
def test_uniswap_v3_loader_lifecycle_methods_raise_not_implemented():
    """``UniswapV3Loader`` (the abstract layer) must not silently
    ``pass`` — subclasses that forget to override should fail loudly."""
    from fractal.loaders.thegraph.uniswap_v3.uniswap_loader import UniswapV3Loader

    class _NoOverride(UniswapV3Loader):
        # Note: ``get_pool_decimals`` is abstract elsewhere; for this test
        # we only need to verify that lifecycle methods raise.
        def get_pool_decimals(self, address):
            return 18, 18

    loader = _NoOverride(api_key="key", subgraph_id="id")
    with pytest.raises(NotImplementedError):
        loader.extract()
    with pytest.raises(NotImplementedError):
        loader.transform()
    with pytest.raises(NotImplementedError):
        loader.load()
    with pytest.raises(NotImplementedError):
        loader.read()
