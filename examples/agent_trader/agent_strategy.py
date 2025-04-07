import time

from typing import List
from dataclasses import dataclass
from datetime import datetime, UTC

from fractal.core.base import (
    BaseStrategy, Action, BaseStrategyParams,
    ActionToTake, NamedEntity, Observation)

from fractal.core.entities.single_spot_exchange import (
    SingleSpotExchange, SingleSpotExchangeGlobalState, SingleSpotExchangeInternalState
)
from fractal.core.base.observations import ObservationsStorage, SQLiteObservationsStorage
from fractal.loaders import LoaderType
from fractal.loaders.binance import BinanceKlinesLoader

from openai_agent import create_agent, AgentAction
from agents import function_tool, Runner, Agent
from prompts import NEUTRAL_PROMPT

@dataclass
class AgentTradingStrategyParams(BaseStrategyParams):
    INITIAL_BALANCE: float = 100_000
    WINDOW_SIZE: int = 30
    MODEL: str = 'o3-mini'
    PROMPT: str = NEUTRAL_PROMPT


class AgentTradingStrategy(BaseStrategy):

    def __init__(self, debug: bool = False, params: AgentTradingStrategyParams | None = None,
                 observations_storage: ObservationsStorage | None = None):
        super().__init__(params=params, debug=debug, observations_storage=observations_storage)
        self._agent = self.__create_agent()
        self._window_size = self._params.WINDOW_SIZE

    def __create_agent(self) -> Agent:
        @function_tool
        def get_klines() -> List:
            """
            This function is used as a tool to get the klines data.
            Returns:
                List: list of klines (open, high, low, close)
            """
            return self.observations_storage.read()
        return create_agent(model=self._params.MODEL, tools=[get_klines], prompt=self._params.PROMPT)

    def set_up(self):
        self.register_entity(NamedEntity(entity_name='exchange', entity=SingleSpotExchange()))
        exchange = self.get_entity('exchange')
        exchange.action_deposit(self._params.INITIAL_BALANCE)

    def predict(self) -> ActionToTake:

        if self._window_size == 0:
            exchange: SingleSpotExchange = self.get_entity('exchange')
            internal_state: SingleSpotExchangeInternalState = exchange.internal_state
            global_state: SingleSpotExchangeGlobalState = exchange.global_state
            res = Runner.run_sync(
                self._agent,
                (
                    f"You have {internal_state.cash} in USD and you have {internal_state.amount} of tokens."
                    f"Make a prediction of actions to take."
                    f"You can not buy more than you can afford = {internal_state.cash / global_state.close} or sell more than you have {internal_state.amount}."
                )
            )
            prediction: AgentAction = res.final_output
            self._debug(prediction)
            # sleep to avoid rps limit
            time.sleep(1)
            self._window_size = self._params.WINDOW_SIZE
            if prediction.action.lower() == 'hold':
                return []
            else:
                return [ActionToTake(
                    entity_name='exchange',
                    action=Action(action=prediction.action.lower(), args={'amount': prediction.amount})
                )]
        else:
            self._window_size -= 1
            return []


if __name__ == '__main__':
    # Load prices from Binance
    binance_klines = BinanceKlinesLoader('BTCUSDT', interval='1d',
                                         start_time=datetime(2024, 1, 1, tzinfo=UTC), end_time=datetime(2025, 4, 1, tzinfo=UTC),
                                         loader_type=LoaderType.CSV).read(with_run=True)

    # Build observations list
    observations: List[Observation] = [
        Observation(timestamp=timestamp, states={'exchange': SingleSpotExchangeGlobalState(open=o, high=h, low=l, close=c)})
        for timestamp, o, h, l, c in zip(binance_klines.index, binance_klines['open'], binance_klines['high'],
                                         binance_klines['low'], binance_klines['close'])
    ]

    # Run the strategy with an Agent
    params: AgentTradingStrategyParams = AgentTradingStrategyParams()
    strategy = AgentTradingStrategy(debug=True, params=params,
                                    observations_storage=SQLiteObservationsStorage())
    result = strategy.run(observations)
    print(result.get_default_metrics())  # show metrics
    result.to_dataframe().to_csv('result.csv')  # save result to csv
