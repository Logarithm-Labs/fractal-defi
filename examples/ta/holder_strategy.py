from typing import List
from dataclasses import dataclass
import logging
import os
from datetime import datetime

from fractal.core.base import (
    BaseStrategy, Action, BaseStrategyParams,
    ActionToTake, NamedEntity, Observation)
from fractal.core.entities import BaseSpotEntity
from fractal.loaders import BinanceDayPriceLoader, LoaderType

from binance_entity import BinanceSpot, BinanceGlobalState

# Создаем директорию для логов, если она не существует
log_dir = "logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Создаем имя файла с временной меткой
log_filename = os.path.join(log_dir, f"strategy_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

# Настройка логирования - в консоль и файл
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(log_filename, mode='w', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
logger.info(f"Логи будут сохранены в файл: {log_filename}")


@dataclass
class HolderStrategyParams(BaseStrategyParams):
    BUY_PRICE: float
    SELL_PRICE: float
    TRADE_SHARE: float = 0.01
    INITIAL_BALANCE: float = 10_000
    MA_PERIOD: int = 20


class HodlerStrategy(BaseStrategy):

    def __init__(self, debug: bool = False, params: HolderStrategyParams | None = None):
        logger.info(f"Инициализация HodlerStrategy с параметрами: debug={debug}, params={params}")
        super().__init__(params=params, debug=debug)

    def set_up(self):
        logger.info("Запуск метода set_up")
        # check that the entity 'exchange' is registered
        entities = self.get_all_available_entities()
        logger.info(f"Доступные сущности: {entities}")
        assert 'exchange' in entities
        # deposit initial balance into the exchange
        if self._params is not None:
            logger.info(f"Внесение депозита в размере {self._params.INITIAL_BALANCE}")
            self.__deposit_into_exchange()

    def predict(self) -> ActionToTake:
        exchange: BaseSpotEntity = self.get_entity('exchange')
        current_price = exchange.global_state.price
        current_cash = exchange.internal_state.cash
        current_amount = exchange.internal_state.amount
        current_sma = exchange.calculate_sma(self._params.MA_PERIOD)

        logger.info(f"predict: текущая цена = {current_price}, наличные = {current_cash}, количество BTC = {current_amount}, SMA = {current_sma}")
        logger.info(f"Параметры: BUY_PRICE = {self._params.BUY_PRICE}, SELL_PRICE = {self._params.SELL_PRICE}, TRADE_SHARE = {self._params.TRADE_SHARE}")
        
        
        if current_price < self._params.BUY_PRICE and current_sma < self._params.BUY_PRICE:
            # Emit a buy action to apply to the entity registered as 'exchange'
            # We buy a fraction of the total cash available
            amount_to_buy = self._params.TRADE_SHARE * current_cash / current_price
            logger.info(f"💰 Цена ниже порога покупки. Рассчитанное количество для покупки: {amount_to_buy}")
            if amount_to_buy < 1e-6:
                logger.info("⚠️ Количество слишком мало для покупки, пропускаем")
                return []
            logger.info(f"✅ Запланирована покупка: {amount_to_buy} BTC по цене {current_price}")
            return [ActionToTake(
                entity_name='exchange',
                action=Action(action='buy', args={'amount': amount_to_buy})
            )]
        elif current_price > self._params.SELL_PRICE:
            # Emit a sell action to apply to the entity registered as 'exchange'
            # We sell a fraction of the total BTC available
            amount_to_sell = self._params.TRADE_SHARE * current_amount
            
            # Определяем, продажа в плюс или минус (для эмодзи)
            # Так как у нас нет прямого доступа к цене покупки, будем условно считать продажу выгодной,
            # если текущая цена выше средней между BUY_PRICE и SELL_PRICE
            avg_price = (self._params.BUY_PRICE + self._params.SELL_PRICE) / 2
            emoji = "🚀" if current_price > avg_price else "📉"
            
            logger.info(f"{emoji} Цена выше порога продажи. Рассчитанное количество для продажи: {amount_to_sell}")
            if amount_to_sell < 1e-6:
                logger.info("⚠️ Количество слишком мало для продажи, пропускаем")
                return []
            logger.info(f"💸 Запланирована продажа: {amount_to_sell} BTC по цене {current_price}")
            return [ActionToTake(
                entity_name='exchange',
                action=Action(action='sell', args={'amount': amount_to_sell})
            )]
        else:
            logger.info("🔄 Цена в промежуточном диапазоне. Стратегия HODL - ничего не делаем")
            # HODL
            return []

    def __deposit_into_exchange(self):
        exchange: BaseSpotEntity = self.get_entity('exchange')
        action: Action = Action(action='deposit', args={'amount_in_notional': self._params.INITIAL_BALANCE})
        logger.info(f"💼 Выполняется депозит: {self._params.INITIAL_BALANCE}")
        exchange.execute(action)
        logger.info(f"✅ Депозит завершен. Текущее состояние: cash={exchange.internal_state.cash}, amount={exchange.internal_state.amount}")


class BinanceHodlerStrategy(HodlerStrategy):
    def set_up(self):
        logger.info("Запуск set_up в BinanceHodlerStrategy")
        self.register_entity(NamedEntity(entity_name='exchange', entity=BinanceSpot()))
        logger.info("Зарегистрирована сущность 'exchange'")
        super().set_up()


if __name__ == '__main__':
    logger.info("🚀 ------ Запуск стратегии ------")
    # Load prices from Binance
    logger.info("📊 Загрузка исторических цен из Binance")
    binance_prices = BinanceDayPriceLoader('BTCUSDT', loader_type=LoaderType.CSV).read(with_run=True)
    logger.info(f"📈 Загружено {len(binance_prices)} значений цен")

    # Build observations list
    logger.info("🔍 Формирование списка наблюдений")

    # тут состояние глобально передается через observation 
    observations: List[Observation] = [
        Observation(timestamp=timestamp, states={'exchange': BinanceGlobalState(price=price)})
        for timestamp, price in zip(binance_prices.index, binance_prices['price'])
    ]
    logger.info(f"👁️ Сформировано {len(observations)} наблюдений")

    # Run the strategy
    params: HolderStrategyParams = HolderStrategyParams(
        BUY_PRICE=50_000, SELL_PRICE=60_000,
        TRADE_SHARE=0.01, INITIAL_BALANCE=100_000,
        MA_PERIOD=20
    )
    logger.info(f"⚙️ Создание стратегии с параметрами: {params}")
    strategy = BinanceHodlerStrategy(debug=True, params=params)
    
    logger.info("▶️ Запуск стратегии на исторических данных")
    result = strategy.run(observations)
    
    logger.info("🏁 Стратегия завершена. Вывод метрик:")
    metrics = result.get_default_metrics()
    logger.info(f"📊 Метрики: {metrics}")
    
    logger.info("💾 Сохранение результатов в CSV файл")
    result.to_dataframe().to_csv('resutl.csv')  # save results of strategy states
    logger.info("✅ Результаты сохранены в файл resutl.csv")
