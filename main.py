from fractal.loaders.binance import BinancePriceLoader, BinanceFundingLoader, BinanceSpotPriceLoader
from fractal.loaders.base_loader import LoaderType
from datetime import datetime, timedelta, timezone
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import requests
from typing import Optional
from tqdm import tqdm

td = delta = timedelta(days=180, 
                           seconds=0, 
                           microseconds=0, 
                           milliseconds=0, 
                           minutes=0, 
                           hours=0, 
                           weeks=0)

current = datetime.now(timezone.utc)


def get_futures_trading_tickers() -> list:
    url = "https://fapi.binance.com/fapi/v1/ticker/price"
    try:
        response = requests.get(url, timeout=10).json()
    except requests.exceptions.RequestException as e:
        raise SystemExit(e)
    return response


def get_spot_volumes() -> list:
    url = "https://api.binance.com/api/v3/ticker/24hr"
    url = (
        f"{url}?"
        f"&type=MINI"
    )
    try:
        response = requests.get(url, timeout=10).json()
    except requests.exceptions.RequestException as e:
        raise SystemExit(e)
    return response


futures = pd.DataFrame(get_futures_trading_tickers())
spot = pd.DataFrame(get_spot_volumes())

joint = pd.merge(futures, spot, how='inner', on='symbol')[['symbol', 'count', 'volume']]
popular_first = joint[joint['count'] > 0].sort_values(by="count", ascending=False) # Если делать по volume то получится странно

futures_spot_tickers = popular_first['symbol'].to_list()
for i in tqdm(range(len(futures_spot_tickers))):
    ticker = futures_spot_tickers[i]
    
    # Скачиваем сспотовые цены
    interval = '30m'
    binance_spot_kandle_loader = BinanceSpotPriceLoader(
        ticker=ticker,
        start_time=current - td,
        end_time=current,
        interval=interval,
        loader_type=LoaderType.CSV)
    binance_spot_kandle_loader.read(with_run=True)

    # Скачиаем цены фьюч контрактов
    binance_future_kandle_loader = BinancePriceLoader(
    ticker=ticker,
    start_time=current - td,
    end_time=current,
    interval=interval,
    loader_type=LoaderType.CSV)
    binance_future_kandle_loader.read(with_run=True)

    # скачиваем funding rate
    binance_funding_rate_loader = BinanceFundingLoader(
    ticker="BTCUSDT",
    start_time=current - td,
    end_time=current,
    loader_type=LoaderType.CSV)
    binance_funding_rate_loader.read(with_run=True)
