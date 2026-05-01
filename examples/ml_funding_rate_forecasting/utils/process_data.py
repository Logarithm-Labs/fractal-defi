import glob

import pandas as pd


def _read_latest_cache(loader_name: str, ticker: str) -> pd.DataFrame:
    """Read the most-recent cached CSV for ``ticker`` under ``loader_name``.

    fractal-defi >= 1.3.0 stores cache files as
    ``{ticker}-{interval}-{start_ms}-{end_ms}.csv``; older versions used
    just ``{ticker}.csv``. Glob-match handles both — sorted picks the
    most recent slice if multiple windows were cached.
    """
    pattern = f"./fractal_data/{loader_name}/{ticker}-*.csv"
    matches = sorted(glob.glob(pattern))
    if not matches:
        # Pre-1.3.0 cache filename (no time range suffix).
        legacy = f"./fractal_data/{loader_name}/{ticker}.csv"
        if glob.glob(legacy):
            matches = [legacy]
    if not matches:
        raise FileNotFoundError(
            f"no cached CSV for {ticker} under fractal_data/{loader_name}/. "
            f"Run the data-download cell first."
        )
    return pd.read_csv(matches[-1], sep=",", index_col=0)


def process_ticker(ticker_name):

    df_futr = _read_latest_cache("binancepriceloader", ticker_name)
    df_spot = _read_latest_cache("binancespotpriceloader", ticker_name)
    df_rate = _read_latest_cache("binancefundingloader", ticker_name)

    df_futr.rename(
        columns={
            "openTime": "open_time",
            "open": "open_futr",
            "high": "high_futr",
            "low": "low_futr",
            "close": "close_futr",
            "volume": "volume_futr",
        },
        inplace=True,
    )
    df_spot.rename(
        columns={
            "openTime": "open_time",
            "open": "open_spot",
            "high": "high_spot",
            "low": "low_spot",
            "close": "close_spot",
            "volume": "volume_spot",
        },
        inplace=True,
    )
    df_rate.rename(
        columns={
            "fundingTime": "open_time",
        },
        inplace=True,
    )

    df = df_spot.merge(df_futr, on="open_time", how="left")
    df = df.merge(df_rate, on="open_time", how="left")

    df["open_time"] = df["open_time"].str[:-6]
    df.sort_values(by="open_time", ascending=False, inplace=True)
    df.drop(columns=["ticker"], inplace=True)

    df["fundingRate"] = df["fundingRate"].ffill()
    df.dropna(inplace=True)
    df.sort_values(by="open_time", ascending=True, inplace=True)

    return df
