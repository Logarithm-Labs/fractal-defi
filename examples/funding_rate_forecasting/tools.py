import pandas as pd
import requests
from tqdm import tqdm

from fractal.loaders.base_loader import LoaderType
from fractal.loaders.binance import BinanceFundingLoader, BinancePriceLoader


def get_futures_trading_tickers() -> list[dict]:
    url = "https://fapi.binance.com/fapi/v1/ticker/price"
    try:
        response = requests.get(url, timeout=10).json()
    except requests.exceptions.RequestException as e:
        raise SystemExit(e)
    return response


def get_spot_volumes() -> list[dict]:
    url = "https://api.binance.com/api/v3/ticker/24hr"
    url = f"{url}?" f"&type=MINI"
    try:
        response = requests.get(url, timeout=10).json()
    except requests.exceptions.RequestException as e:
        raise SystemExit(e)
    return response


def get_list_top_n_tickers(n_top_tickers) -> list:

    list_futures = get_futures_trading_tickers()
    futures = pd.DataFrame(list_futures)

    list_spots = get_spot_volumes()
    spot = pd.DataFrame(list_spots)

    df_spot_future = futures.merge(spot, on="symbol", how="inner")[
        ["symbol", "count", "volume"]
    ]

    df_spot_future = df_spot_future[df_spot_future["count"] > 0].sort_values(
        by="count", ascending=False
    )

    futures_spot_tickers = df_spot_future["symbol"].to_list()[:n_top_tickers]

    return futures_spot_tickers


def download_spot_future_fr_data(
    futures_spot_tickers, start_timestamp, end_timstamp, interval="1h"
) -> None:

    for ticker in tqdm(futures_spot_tickers):
        # Download spot data
        binance_spot_kandle_loader = BinanceSpotPriceLoader(
            ticker=ticker,
            start_time=start_timestamp,
            end_time=end_timstamp,
            interval=interval,
            loader_type=LoaderType.CSV,
        )
        binance_spot_kandle_loader.read(with_run=True)

        # Download future data
        binance_future_kandle_loader = BinancePriceLoader(
            ticker=ticker,
            start_time=start_timestamp,
            end_time=end_timstamp,
            interval=interval,
            loader_type=LoaderType.CSV,
        )
        binance_future_kandle_loader.read(with_run=True)

        # Download funding rate
        binance_funding_rate_loader = BinanceFundingLoader(
            ticker=ticker,
            start_time=start_timestamp,
            end_time=end_timstamp,
            loader_type=LoaderType.CSV,
        )
        binance_funding_rate_loader.read(with_run=True)
