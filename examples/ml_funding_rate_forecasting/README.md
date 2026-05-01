# ML Funding Rate Forecasting

This directory contains an example project for forecasting funding rates in DeFi markets using ML & fractal-defi

## Project Structure

- `utils/`: Contains data downloading and processing tools
- `utils/get_data.py`: Contains wrappers over fractal-defi for easy data extraction
- `utils/process_data.py`: Brings the spot, futures, and financing rates data into a single format
- `utils/create_features.py`: Extracts basic features for clusterization and modeling
- `utils/metics.py`: SMAPE for time series data
- `notebooks/`: Jupyter notebook for exploratory data analysis and model development

## Using Pipeline

1. **Install Dependencies**
    Requires Python 3.10–3.13 (matches `fractal-defi`'s `python_requires`).
    Install from this directory:
    ```bash
    pip install -r requirements.txt
    ```
    The file pulls in `fractal-defi>=1.3.1` and a small set of ML
    extras the framework doesn't ship (scipy, statsmodels, catboost,
    hdbscan, tqdm). Runtime libraries (numpy / pandas / mlflow / etc.)
    come transitively through `fractal-defi` at the floors it pins.

2. **Download and Prepare Data**
    Just run notebook `notebooks/simple_research.ipynb` and get data in the first section with functions `list_top_n_tickers()` and `download_spot_future_fr_data()`. Processing (join) by `process_ticker()`

3. **Train Model**
    In the section "research, modeling" build a model with our features

## Metrics

SMAPE on a 1-year window of hourly ETHUSDT (Apr 2024 – Apr 2025), held-out
20% test split:

| Ticker | Model | Baseline rolling mean | Baseline constant |
|--------|-------|-----------------------|-------------------|
| ETH    | 1.094 | 1.163                 | 1.199             |

Numbers regenerated after a fix to the forecasting target — earlier
versions of this notebook used `df['basis'].shift(2)` (predicting
basis from 2 hours **earlier**), inflating apparent skill. The current
notebook uses `shift(-2)` (predicting 2 hours **ahead**, as originally
intended).


## Requirements

- Python 3.10–3.13
- Libraries: see `requirements.txt`. Notebook needs `BinanceSpotPriceLoader`,
  which is part of `fractal-defi>=1.3.1`.
