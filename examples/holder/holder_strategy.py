from typing import List
from dataclasses import dataclass
import os
from dotenv import load_dotenv
import mlflow
from io import StringIO

from fractal.core.base import (
    BaseStrategy, Action, BaseStrategyParams,
    ActionToTake, NamedEntity, Observation)
from fractal.core.entities import BaseSpotEntity
from fractal.loaders import BinanceDayPriceLoader, LoaderType
from fractal.core.pipeline import MLFlowConfig

from binance_entity import BinanceSpot, BinanceGlobalState

load_dotenv()

@dataclass
class HolderStrategyParams(BaseStrategyParams):
    BUY_PRICE: float
    SELL_PRICE: float
    TRADE_SHARE: float = 0.01
    INITIAL_BALANCE: float = 10_000


class HodlerStrategy(BaseStrategy):

    def __init__(self, debug: bool = False, params: HolderStrategyParams | None = None, 
                 mlflow_config: MLFlowConfig | None = None, *args, **kwargs):
        super().__init__(params=params, debug=debug, *args, **kwargs)
        self.mlflow_config = mlflow_config
        if self.mlflow_config:
            mlflow.set_tracking_uri(self.mlflow_config.mlflow_uri)
            if self.mlflow_config.aws_access_key_id and self.mlflow_config.aws_secret_access_key:
                os.environ['AWS_ACCESS_KEY_ID'] = self.mlflow_config.aws_access_key_id
                os.environ['AWS_SECRET_ACCESS_KEY'] = self.mlflow_config.aws_secret_access_key
            mlflow.set_experiment(self.mlflow_config.experiment_name)

    def set_up(self):
        # check that the entity 'exchange' is registered
        assert 'exchange' in self.get_all_available_entities()
        # deposit initial balance into the exchange
        if self._params is not None:
            self.__deposit_into_exchange()

    def predict(self) -> ActionToTake:
        exchange: BaseSpotEntity = self.get_entity('exchange')
        if exchange.global_state.price < self._params.BUY_PRICE:
            # Emit a buy action to apply to the entity registered as 'exchange'
            # We buy a fraction of the total cash available
            amount_to_buy = self._params.TRADE_SHARE * exchange.internal_state.cash / exchange.global_state.price
            if amount_to_buy < 1e-6:
                return []
            return [ActionToTake(
                entity_name='exchange',
                action=Action(action='buy', args={'amount': amount_to_buy})
            )]
        elif exchange.global_state.price > self._params.SELL_PRICE:
            # Emit a sell action to apply to the entity registered as 'exchange'
            # We sell a fraction of the total BTC available
            amount_to_sell = self._params.TRADE_SHARE * exchange.internal_state.amount
            if amount_to_sell < 1e-6:
                return []
            return [ActionToTake(
                entity_name='exchange',
                action=Action(action='sell', args={'amount': amount_to_sell})
            )]
        else:
            # HODL
            return []

    def __deposit_into_exchange(self):
        exchange: BaseSpotEntity = self.get_entity('exchange')
        action: Action = Action(action='deposit', args={'amount_in_notional': self._params.INITIAL_BALANCE})
        exchange.execute(action)
        
    def log_to_mlflow(self, result):
        """Сохраняет результаты стратегии в MLflow"""
        if not self.mlflow_config:
            return
            
        run_name = f"strategy_run_{self._params.BUY_PRICE}_{self._params.SELL_PRICE}_{self._params.TRADE_SHARE}"
        with mlflow.start_run(run_name=run_name):
            # Логирование параметров
            params_dict = {k: v for k, v in self._params.__dict__.items() if not k.startswith('_')}
            mlflow.log_params(params_dict)
            
            # Логирование результатов
            result_df = result.to_dataframe()
            metrics = result.get_default_metrics()
            
            # Сохранение CSV с результатами
            csv_buffer = StringIO()
            result_df.to_csv(csv_buffer, index=False)
            mlflow.log_text(csv_buffer.getvalue(), "strategy_result.csv")
            
            # Логирование метрик - преобразуем объект StrategyMetrics в словарь
            metrics_dict = metrics.__dict__ if hasattr(metrics, '__dict__') else {}
            mlflow.log_metrics(metrics_dict)
            
            print(result.logger.logs_path)
            try:
                if hasattr(result, 'logger') and hasattr(result.logger, 'logs_path'):
                    mlflow.log_artifact(result.logger.logs_path)
            except Exception as e:
                print(f"Не удалось сохранить логи: {e}")


class BinanceHodlerStrategy(HodlerStrategy):
    def set_up(self):
        self.register_entity(NamedEntity(entity_name='exchange', entity=BinanceSpot()))
        super().set_up()


if __name__ == '__main__':
    # Load prices from Binance
    binance_prices = BinanceDayPriceLoader('BTCUSDT', loader_type=LoaderType.CSV).read(with_run=True)

    # Build observations list
    observations: List[Observation] = [
        Observation(timestamp=timestamp, states={'exchange': BinanceGlobalState(price=price)})
        for timestamp, price in zip(binance_prices.index, binance_prices['price'])
    ]

    # Настройка MLflow
    mlflow_config = MLFlowConfig(
        mlflow_uri='https://mlflow.devcryptoservices.xyz/',
        experiment_name='binance_hodler_btc_0',
        aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
    )

    # Run the strategy
    params: HolderStrategyParams = HolderStrategyParams(
        BUY_PRICE=50_000, SELL_PRICE=60_000,
        TRADE_SHARE=0.01, INITIAL_BALANCE=100_000
    )
    strategy = BinanceHodlerStrategy(debug=True, params=params, mlflow_config=mlflow_config)
    result = strategy.run(observations)

    result.to_dataframe().to_csv('resutl.csv')  # save results of strategy states
    
    # Сохранение результатов в MLflow
    strategy.log_to_mlflow(result)
