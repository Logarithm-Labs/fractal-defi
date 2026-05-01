"""Tests for ``Pipeline`` / ``DefaultPipeline`` — MLflow side effects stubbed."""
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, List
from unittest.mock import MagicMock

import pytest

from fractal.core import pipeline as pipeline_module
from fractal.core.base import BaseStrategy, BaseStrategyParams, NamedEntity, Observation
from fractal.core.entities.simple.spot import SimpleSpotExchange, SimpleSpotExchangeGlobalState
from fractal.core.pipeline import DefaultPipeline, ExperimentConfig, MLflowConfig, Pipeline, _params_to_dict


@dataclass
class _Params(BaseStrategyParams):
    A: int = 1
    B: float = 2.0


class _DummyStrategy(BaseStrategy[_Params]):
    STRICT_OBSERVATIONS = False

    def set_up(self):
        self.register_entity(NamedEntity('SPOT', SimpleSpotExchange(trading_fee=0.0)))

    def predict(self):
        return []


@pytest.fixture
def mlflow_calls(monkeypatch):
    calls: List[tuple] = []

    def _make_recorder(name):
        def _record(*args, **kwargs):
            calls.append((name, args, kwargs))
            return None
        return _record

    fake_run = MagicMock()
    fake_run.__enter__ = MagicMock(return_value=fake_run)
    fake_run.__exit__ = MagicMock(return_value=False)

    def _start_run(*args, **kwargs):
        calls.append(('start_run', args, kwargs))
        return fake_run

    monkeypatch.setattr(pipeline_module.mlflow, 'set_tracking_uri',
                        _make_recorder('set_tracking_uri'))
    monkeypatch.setattr(pipeline_module.mlflow, 'get_experiment_by_name',
                        _make_recorder('get_experiment_by_name'))
    monkeypatch.setattr(pipeline_module.mlflow, 'create_experiment',
                        _make_recorder('create_experiment'))
    monkeypatch.setattr(pipeline_module.mlflow, 'set_experiment',
                        _make_recorder('set_experiment'))
    monkeypatch.setattr(pipeline_module.mlflow, 'start_run', _start_run)
    monkeypatch.setattr(pipeline_module.mlflow, 'end_run', _make_recorder('end_run'))
    monkeypatch.setattr(pipeline_module.mlflow, 'log_params', _make_recorder('log_params'))
    monkeypatch.setattr(pipeline_module.mlflow, 'log_metrics', _make_recorder('log_metrics'))
    monkeypatch.setattr(pipeline_module.mlflow, 'log_text', _make_recorder('log_text'))
    monkeypatch.setattr(pipeline_module.mlflow, 'log_artifact', _make_recorder('log_artifact'))
    return calls


def _make_obs(price, ts=datetime(2024, 1, 1)):
    return Observation(timestamp=ts, states={
        'SPOT': SimpleSpotExchangeGlobalState(close=price),
    })


def _default_obs():
    # Three points spanning a year; otherwise ``StrategyResult.get_metrics``
    # divides by ``total_years == 0``.
    return [
        _make_obs(100.0, datetime(2024, 1, 1)),
        _make_obs(101.0, datetime(2024, 6, 1)),
        _make_obs(102.0, datetime(2024, 12, 31)),
    ]


def _make_pipeline(mlflow_cfg=None, exp_cfg=None):
    if mlflow_cfg is None:
        mlflow_cfg = MLflowConfig(experiment_name='test', mlflow_uri='http://stub')
    if exp_cfg is None:
        exp_cfg = ExperimentConfig(
            strategy_type=_DummyStrategy,
            params_grid=[_Params(A=1, B=1.0)],
            backtest_observations=_default_obs(),
        )
    return DefaultPipeline(mlflow_cfg, exp_cfg)


@pytest.mark.core
def test_params_to_dict_accepts_plain_dict():
    assert _params_to_dict({'A': 1, 'B': 2}) == {'A': 1, 'B': 2}


@pytest.mark.core
def test_params_to_dict_returns_a_copy_for_dict():
    src = {'A': 1}
    out = _params_to_dict(src)
    out['A'] = 999
    assert src['A'] == 1


@pytest.mark.core
def test_params_to_dict_accepts_dataclass_subclass_of_base_strategy_params():
    p = _Params(A=42, B=3.14)
    assert _params_to_dict(p) == {'A': 42, 'B': 3.14}


@pytest.mark.core
def test_params_to_dict_accepts_plain_object_with_dict():
    p = BaseStrategyParams(data={'X': 'y', 'N': 1})
    assert _params_to_dict(p) == {'X': 'y', 'N': 1}


@pytest.mark.core
def test_params_to_dict_rejects_non_convertible():
    with pytest.raises(TypeError, match="cannot coerce"):
        _params_to_dict(42)


@pytest.mark.core
def test_aws_env_vars_unchanged_when_config_is_none(monkeypatch):
    monkeypatch.setenv('AWS_ACCESS_KEY_ID', 'host_key')
    monkeypatch.setenv('AWS_SECRET_ACCESS_KEY', 'host_secret')
    cfg = MLflowConfig(experiment_name='e', mlflow_uri='http://s')
    p = _make_pipeline(mlflow_cfg=cfg)
    p._set_aws_env()
    assert os.environ['AWS_ACCESS_KEY_ID'] == 'host_key'
    assert os.environ['AWS_SECRET_ACCESS_KEY'] == 'host_secret'


@pytest.mark.core
def test_aws_env_vars_unchanged_when_config_is_empty_string(monkeypatch):
    monkeypatch.setenv('AWS_ACCESS_KEY_ID', 'host_key')
    cfg = MLflowConfig(
        experiment_name='e', mlflow_uri='http://s',
        aws_access_key_id='', aws_secret_access_key='',
    )
    p = _make_pipeline(mlflow_cfg=cfg)
    p._set_aws_env()
    assert os.environ['AWS_ACCESS_KEY_ID'] == 'host_key'


@pytest.mark.core
def test_aws_env_vars_set_when_config_provides_them(monkeypatch):
    monkeypatch.delenv('AWS_ACCESS_KEY_ID', raising=False)
    monkeypatch.delenv('AWS_SECRET_ACCESS_KEY', raising=False)
    cfg = MLflowConfig(
        experiment_name='e', mlflow_uri='http://s',
        aws_access_key_id='cfg_key', aws_secret_access_key='cfg_secret',
    )
    p = _make_pipeline(mlflow_cfg=cfg)
    p._set_aws_env()
    assert os.environ['AWS_ACCESS_KEY_ID'] == 'cfg_key'
    assert os.environ['AWS_SECRET_ACCESS_KEY'] == 'cfg_secret'


@pytest.mark.core
def test_pipeline_does_not_connect_mlflow_on_init(monkeypatch):
    called: List[str] = []
    monkeypatch.setattr(pipeline_module.mlflow, 'set_tracking_uri',
                        lambda uri: called.append('set_tracking_uri'))
    monkeypatch.setattr(pipeline_module.mlflow, 'get_experiment_by_name',
                        lambda name: called.append('get_experiment_by_name'))
    monkeypatch.setattr(pipeline_module.mlflow, 'create_experiment',
                        lambda **kw: called.append('create_experiment'))
    monkeypatch.setattr(pipeline_module.mlflow, 'set_experiment',
                        lambda name: called.append('set_experiment'))
    _make_pipeline()
    assert not called


@pytest.mark.core
def test_ensure_connected_runs_connect_only_once(mlflow_calls):
    p = _make_pipeline()
    p._ensure_connected()
    p._ensure_connected()
    p._ensure_connected()
    assert sum(1 for c in mlflow_calls if c[0] == 'set_tracking_uri') == 1


@pytest.mark.core
def test_run_triggers_lazy_connect(mlflow_calls):
    p = _make_pipeline()
    p.run()
    names = [c[0] for c in mlflow_calls]
    assert 'set_tracking_uri' in names
    assert names.index('set_tracking_uri') < names.index('start_run')


@pytest.mark.core
def test_grid_step_triggers_lazy_connect_when_called_directly(mlflow_calls):
    p = _make_pipeline()
    p.grid_step(_Params(A=1, B=1.0))
    assert 'set_tracking_uri' in [c[0] for c in mlflow_calls]


@pytest.mark.core
def test_step_size_default_is_24():
    cfg = ExperimentConfig(strategy_type=_DummyStrategy, params_grid=[_Params()])
    assert cfg.step_size == 24


@pytest.mark.core
def test_scenario_uses_configured_step_size(mlflow_calls, monkeypatch):
    captured: dict = {}
    real = pipeline_module.Launcher.run_scenario

    def _spy(self, observations, window_size, step_size=24, debug=False):
        captured['window_size'] = window_size
        captured['step_size'] = step_size
        return real(self, observations, window_size=window_size,
                    step_size=step_size, debug=debug)

    monkeypatch.setattr(pipeline_module.Launcher, 'run_scenario', _spy)
    obs = [_make_obs(100.0 + i, datetime(2024, 1, 1, i % 24)) for i in range(100)]
    cfg = ExperimentConfig(
        strategy_type=_DummyStrategy,
        params_grid=[_Params()],
        backtest_observations=obs,
        window_size=24,
        step_size=12,
    )
    p = _make_pipeline(exp_cfg=cfg)
    p.grid_step(_Params())
    assert captured == {'window_size': 24, 'step_size': 12}


@pytest.mark.core
def test_grid_step_logs_params_for_dataclass_subclass(mlflow_calls):
    p = _make_pipeline()
    p.grid_step(_Params(A=7, B=0.5))
    log_calls = [c for c in mlflow_calls if c[0] == 'log_params']
    assert log_calls
    assert log_calls[0][1][0] == {'A': 7, 'B': 0.5}


@pytest.mark.core
def test_grid_step_logs_params_for_plain_dict(mlflow_calls):
    p = _make_pipeline()
    p.grid_step({'A': 3, 'B': 9.9})
    log_calls = [c for c in mlflow_calls if c[0] == 'log_params']
    assert log_calls[0][1][0] == {'A': 3, 'B': 9.9}


@pytest.mark.core
def test_grid_step_invokes_run_name_formatter(mlflow_calls):
    cfg = MLflowConfig(
        experiment_name='e', mlflow_uri='http://s',
        run_name_formatter=lambda params: f"A={_params_to_dict(params)['A']}",
    )
    p = _make_pipeline(mlflow_cfg=cfg)
    p.grid_step(_Params(A=11, B=0.1))
    start_calls = [c for c in mlflow_calls if c[0] == 'start_run']
    assert start_calls[0][2].get('run_name') == 'A=11'


@pytest.mark.core
def test_run_iterates_grid_calling_grid_step_per_params(monkeypatch):
    seen: List[Any] = []

    class _Recorder(DefaultPipeline):
        def grid_step(self, params):
            seen.append(params)

    monkeypatch.setattr(pipeline_module, 'mlflow', MagicMock())
    cfg = MLflowConfig(experiment_name='e', mlflow_uri='http://s')
    grid = [_Params(A=1), _Params(A=2), _Params(A=3)]
    p = _Recorder(cfg, ExperimentConfig(
        strategy_type=_DummyStrategy, params_grid=grid,
    ))
    p.run()
    assert [s.A for s in seen] == [1, 2, 3]


@pytest.mark.core
def test_default_pipeline_subclasses_pipeline():
    assert issubclass(DefaultPipeline, Pipeline)


@pytest.mark.core
def test_pipeline_is_abstract():
    with pytest.raises(TypeError):
        Pipeline(MLflowConfig(experiment_name='e', mlflow_uri='http://s'),
                 ExperimentConfig(strategy_type=_DummyStrategy, params_grid=[]))
