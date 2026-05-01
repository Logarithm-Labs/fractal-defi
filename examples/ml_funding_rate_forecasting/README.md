# ML Funding Rate Forecasting

This directory contains an example project for forecasting funding rates in DeFi markets using ML & fractal-defi

## Project Structure

- `utils/`: Contains data downloading and processing tools
- `utils/get_data.py`: Contains wrappers over fractal-defi for easy data extraction
- `utils/process_data.py`: Brings the spot, futures, and financing rates data into a single format
- `utils/create_features.py`: Extracts basic features for clusterization and modeling
- `utils/metrics.py`: SMAPE for time series data
- `notebooks/`: Jupyter notebook for exploratory data analysis and model development

## Using Pipeline

1. **Install dependencies** (Python 3.10–3.13):
    ```bash
    pip install -r requirements.txt
    ```

2. **Run the notebook** `notebooks/baseline_research.ipynb` — first
   section downloads data, then processing, clustering, and modelling.

## Metrics

SMAPE on hourly ETHUSDT, held-out 20% test split:

| Ticker | Model | Baseline rolling mean | Baseline constant |
|--------|-------|-----------------------|-------------------|
| ETH    | 1.093 | 1.170                 | 1.197             |
