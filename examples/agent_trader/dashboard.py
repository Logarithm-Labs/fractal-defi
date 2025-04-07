import re
import textwrap
from datetime import datetime

import dash
from dash import dcc, html
import plotly.graph_objects as go
import pandas as pd


# TradingView-like style template
TRADINGVIEW_TEMPLATE = {
    "paper_bgcolor": "#131722",
    "plot_bgcolor": "#131722",
    "font": {"color": "#D5D5D5"},
    "xaxis": {"gridcolor": "#363c4e", "title_font": {"color": "#D5D5D5"}},
    "yaxis": {"gridcolor": "#363c4e", "title_font": {"color": "#D5D5D5"}},
}


def load_candle_data(filepath: str) -> pd.DataFrame:
    """
    Loads candle data from a CSV file, creates a Date column from the timestamp,
    sorts the DataFrame, and maps OHLC and total balance columns.
    """
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        raise Exception(f"Error reading {filepath}: {e}")

    df['Date'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df.sort_values('Date', inplace=True)

    df['Close'] = df['exchange_close']
    df['Open'] = df['exchange_open']
    df['High'] = df['exchange_high']
    df['Low'] = df['exchange_low']
    df['total_balance'] = df['net_balance']
    return df


def wrap_text(text: str, width: int = 50) -> str:
    """
    Wraps a string into multiple lines using <br> for HTML breaks.
    """
    return textwrap.fill(text, width=width).replace("\n", "<br>")


def parse_log_file(log_file_path: str) -> pd.DataFrame:
    """
    Parses a log file to extract agent actions. For each action, the timestamp of
    the last observation is used. Returns a DataFrame of actions.
    """
    actions = []
    last_observation = None

    try:
        with open(log_file_path, 'r') as f:
            for line in f:
                # Update last_observation if the line contains an Observation timestamp.
                if "Observation:" in line:
                    obs_match = re.search(
                        r'Observation:\s*(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', line
                    )
                    if obs_match:
                        try:
                            last_observation = datetime.strptime(
                                obs_match.group(1), '%Y-%m-%d %H:%M:%S'
                            )
                        except ValueError:
                            last_observation = None
                # If the line contains an action, use the last observation timestamp.
                elif "action=" in line:
                    action_match = re.search(
                        r"action='(BUY|SELL|HOLD)'\s+amount=([\d\.]+)\s+reasoning='(.*?)'",
                        line,
                        re.IGNORECASE  # This makes it match any case variation
                    )
                    if action_match and last_observation is not None:
                        actions.append({
                            "Date": last_observation,
                            "Action": action_match.group(1),
                            "Amount": float(action_match.group(2)),
                            "Reasoning": action_match.group(3),
                        })
    except Exception as e:
        print(f"Error reading {log_file_path}: {e}")

    actions_df = pd.DataFrame(actions)
    if not actions_df.empty:
        actions_df["Date"] = pd.to_datetime(actions_df["Date"])
        actions_df["WrappedReasoning"] = actions_df["Reasoning"].apply(lambda x: wrap_text(x, width=50))
    return actions_df


def get_marker_y(candle_df: pd.DataFrame, date: datetime, action_type: str) -> float:
    """
    Determines the y-coordinate for a marker based on the candle data.
    For BUY actions, returns a value slightly below the candle's low;
    for SELL actions, slightly above the candle's high.
    """
    row = candle_df[candle_df['Date'].dt.date == date.date()]
    if not row.empty:
        if action_type == 'BUY':
            return row.iloc[0]['Low'] * 0.9
        elif action_type == 'SELL':
            return row.iloc[0]['High'] * 1.1
    return None


def create_candlestick_chart(candle_df: pd.DataFrame, actions_df: pd.DataFrame, template: dict) -> go.Figure:
    """
    Creates a candlestick chart with overlaid markers for BUY and SELL actions.
    """
    fig = go.Figure(data=[go.Candlestick(
        x=candle_df['Date'],
        open=candle_df['Open'],
        high=candle_df['High'],
        low=candle_df['Low'],
        close=candle_df['Close'],
        name='Price'
    )])

    if not actions_df.empty:
        buy_actions = actions_df[actions_df['Action'] == 'BUY']
        sell_actions = actions_df[actions_df['Action'] == 'SELL']

        if not buy_actions.empty:
            buy_y = [get_marker_y(candle_df, row['Date'], 'BUY') for _, row in buy_actions.iterrows()]
            fig.add_trace(go.Scatter(
                x=buy_actions['Date'],
                y=buy_y,
                mode='markers',
                marker=dict(symbol='triangle-up', color='green', size=12),
                name='BUY',
                text=buy_actions['WrappedReasoning'],
                hovertemplate='<b>BUY</b><br>%{text}<extra></extra>'
            ))

        if not sell_actions.empty:
            sell_y = [get_marker_y(candle_df, row['Date'], 'SELL') for _, row in sell_actions.iterrows()]
            fig.add_trace(go.Scatter(
                x=sell_actions['Date'],
                y=sell_y,
                mode='markers',
                marker=dict(symbol='triangle-down', color='red', size=12),
                name='SELL',
                text=sell_actions['WrappedReasoning'],
                hovertemplate='<b>SELL</b><br>%{text}<extra></extra>'
            ))

    fig.update_layout(
        title='Candle History with Agent Actions',
        xaxis_title='Date',
        yaxis_title='Price',
        **template
    )
    return fig


def create_performance_chart(candle_df: pd.DataFrame, template: dict) -> go.Figure:
    """
    Creates a performance chart comparing the strategy's total balance with a benchmark.
    """
    strategy_returns = (candle_df['total_balance'] / candle_df['total_balance'].iloc[0] - 1) * 100
    benchmark_returns = (candle_df['Close'] / candle_df['Close'].iloc[0] - 1) * 100

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=candle_df['Date'],
        y=strategy_returns,
        mode='lines',
        name='Strategy Perfomance'
    ))
    fig.add_trace(go.Scatter(
        x=candle_df['Date'],
        y=benchmark_returns,
        mode='lines',
        name='Benchmark (HODL)'
    ))
    fig.update_layout(
        title='Strategy Performance vs Benchmark',
        xaxis_title='Date',
        yaxis_title='Return (%)',
        **template
    )
    return fig


def main():
    # Load candle data and log actions
    candle_df = load_candle_data('result.csv')
    log_file_path = 'runs/AgentTradingStrategy/722dd0b2-d068-4584-9427-21a6075f3b0f/logs/logs.log'

    actions_df = parse_log_file(log_file_path)

    # Create charts
    fig_candle = create_candlestick_chart(candle_df, actions_df, TRADINGVIEW_TEMPLATE)
    fig_perf = create_performance_chart(candle_df, TRADINGVIEW_TEMPLATE)

    # Build and run the Dash application
    app = dash.Dash(__name__)
    app.layout = html.Div([
        html.H1("Agent Backtest", style={'textAlign': 'center', 'color': '#131722'}),
        dcc.Graph(id='candlestick-chart', figure=fig_candle),
        dcc.Graph(id='performance-chart', figure=fig_perf)
    ])

    app.run(debug=True)


if __name__ == '__main__':
    main()
