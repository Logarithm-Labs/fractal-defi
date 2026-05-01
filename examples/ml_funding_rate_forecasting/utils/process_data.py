import pandas as pd


def process_ticker(ticker_name):

    df_futr = pd.read_csv(
        f"./fractal_data/binancepriceloader/{ticker_name}.csv", sep=",", index_col=0
    )
    df_spot = pd.read_csv(
        f"./fractal_data/binancespotpriceloader/{ticker_name}.csv", sep=",", index_col=0
    )
    df_rate = pd.read_csv(
        f"./fractal_data/binancefundingloader/{ticker_name}.csv", sep=",", index_col=0
    )

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

    df["fundingRate"].ffill(inplace=True)
    df.dropna(inplace=True)
    df.sort_values(by="open_time", ascending=True, inplace=True)

    return df
