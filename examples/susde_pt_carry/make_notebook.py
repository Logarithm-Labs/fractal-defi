"""Generate ``analysis.ipynb`` (run once; the notebook is then executed and committed)."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text))


def code(src):
    cells.append(nbf.v4.new_code_cell(src))


md("# sUSDe PT carry: loop vs loop + Boros vs loop + basis leg\n\nEvery section reads `results/` written by `run.py` and `grid.py`.")
code('''import glob, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({"figure.dpi": 72, "savefig.dpi": 72, "figure.max_open_warning": 0})
RESULTS = "results"
YEAR = 365 * 86400
validation = pd.read_csv(os.path.join(RESULTS, "validation.csv"))
runs = {}
for path in sorted(glob.glob(os.path.join(RESULTS, "*_*.csv"))):
    name = os.path.basename(path)[:-4]
    if name.startswith(("grid_", "validation")):
        continue
    df = pd.read_csv(path, parse_dates=["timestamp"])
    runs[name] = df
markets = sorted({n.rsplit("_", 1)[0] for n in runs})
variants = ["none", "boros", "perp"]
print(markets, {k: len(v) for k, v in runs.items()})''')

md("## 1. Equity curves")
code('''def market_axes(n, height=3.6):
    cols = 2
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, height * rows), squeeze=False)
    flat = axes.flatten()
    for ax in flat[n:]:
        ax.axis("off")
    return fig, flat[:n]

fig, axes = market_axes(len(markets))
for ax, market in zip(axes, markets):
    for variant in variants:
        df = runs[f"{market}_{variant}"]
        ax.plot(df["timestamp"], df["net_balance"] / df["net_balance"].iloc[0] - 1, label=variant)
    ax.set_title(market); ax.set_ylabel("equity return"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    ax.tick_params(axis="x", rotation=30)
plt.tight_layout()''')

md("## 2. PnL decomposition per bar\n\n`pt_carry` = PT units × Δprice (accretion to par and implied-APY marks), `interest` = debt × per-bar rate, `hedge` = Δ balance of the floating leg, `costs` = residual (swap fee + impact, hedge fees).")
code('''def decompose(df):
    price = (1.0 + df["PT_implied_apy"]) ** (-df["PT_seconds_to_expiry"] / YEAR)
    units = (df["PT_amount"] + df["LENDING_collateral"]).shift(1)
    out = pd.DataFrame(index=df["timestamp"])
    out["pt_carry"] = (units * price.diff()).values
    out["interest"] = -(df["LENDING_borrowed"].shift(1) * np.expm1(df["LENDING_borrowing_rate"])).values
    hedge = pd.Series(0.0, index=df.index)
    if "BOROS_balance" in df:
        hedge = df["BOROS_balance"].ffill().diff()
    elif "PERP_balance" in df:
        hedge = (df["SPOT_balance"] + df["PERP_balance"]).diff()
    out["hedge"] = hedge.values
    out["costs"] = (df["net_balance"].diff() - out["pt_carry"].values - out["interest"].values - out["hedge"].values).values
    out.iloc[0] = 0.0
    return out.fillna(0.0)

decomp = {name: decompose(df) for name, df in runs.items()}
fig, axes = plt.subplots(len(markets), len(variants), figsize=(5 * len(variants), 2.6 * len(markets)), squeeze=False)
for i, market in enumerate(markets):
    for j, variant in enumerate(variants):
        d = decomp[f"{market}_{variant}"].cumsum()
        ax = axes[i][j]
        for col in d.columns:
            ax.plot(d.index, d[col], label=col)
        ax.plot(d.index, d.sum(axis=1), color="black", lw=1.5, label="total")
        ax.set_title(f"{market} / {variant}"); ax.grid(alpha=0.3); ax.tick_params(axis="x", rotation=30)
        if i == 0 and j == 0:
            ax.legend(fontsize=8)
plt.tight_layout()''')
code('''totals = pd.DataFrame({name: d.sum() for name, d in decomp.items()}).T
totals["equity_change"] = [runs[n]["net_balance"].iloc[-1] - runs[n]["net_balance"].iloc[0] for n in totals.index]
totals.round(1)''')

md("## 3. APY and drawdown")
code('''cols = ["market", "variant", "bars", "leverage_at_entry", "pt_apy_at_entry", "borrow_apy_mean", "funding_apr_mean",
        "boros_mark_apr_mean", "realised_apy", "max_drawdown", "hedge_coverage_mean", "hedge_pnl", "min_health_factor",
        "liquidations"]
validation[cols].round(4)''')
code('''fig, axes = market_axes(len(markets), height=3.2)
for ax, market in zip(axes, markets):
    for variant in variants:
        df = runs[f"{market}_{variant}"]
        ax.plot(df["timestamp"], df["net_balance"] / df["net_balance"].cummax() - 1, label=variant)
    ax.set_title(f"{market}: drawdown"); ax.grid(alpha=0.3); ax.legend(fontsize=8); ax.tick_params(axis="x", rotation=30)
plt.tight_layout()''')

md("## 4. Costs\n\n`entry_cost` = equity after the entry loops minus the 100k deposit (swap fee + impact); `costs_total` = residual of the decomposition after entry (re-lever / de-lever swaps, hedge fees); turnover = bars where the debt moved by more than 1 %.")
code('''rows = []
for name, d in decomp.items():
    df = runs[name]
    start_equity = df["net_balance"].iloc[0]
    years = (df["timestamp"].iloc[-1] - df["timestamp"].iloc[0]).total_seconds() / YEAR
    initial = float(validation.loc[(validation["market"] + "_" + validation["variant"]) == name, "final_equity"].size
                    and 100_000.0)
    rows.append({
        "run": name,
        "entry_cost": df["net_balance"].iloc[0] - initial,  # swap fee + impact of the entry loops
        "costs_total": d["costs"].sum(),
        "costs_pct_of_equity": d["costs"].sum() / start_equity,
        "costs_apy_drag": d["costs"].sum() / start_equity / years,
        "interest_total": d["interest"].sum(),
        "pt_carry_total": d["pt_carry"].sum(),
        "hedge_total": d["hedge"].sum(),
        "debt_changes": int((df["LENDING_borrowed"].pct_change().abs() > 0.01).sum()),
    })
pd.DataFrame(rows).set_index("run").round(4)''')

md("## 5. Parameter grid\n\nRealised APY by target LTV and hedge variant (`grid.py`: LTV × hedge × margin share × ratio).")
code('''for market in markets:
    path = os.path.join(RESULTS, f"grid_{market}.csv")
    if not os.path.exists(path):
        continue
    g = pd.read_csv(path)
    g["cell"] = g.apply(lambda r: r["variant"] if r["variant"] == "none"
                        else f"{r['variant']} s={r['hedge_margin_share']} r={r['hedge_ratio']}", axis=1)
    table = g.pivot_table(index="cell", columns="target_ltv", values="realised_apy")
    fig, ax = plt.subplots(figsize=(8, 0.4 * len(table) + 1.5))
    im = ax.imshow(table.values, aspect="auto", cmap="RdYlGn", vmin=-0.1, vmax=0.3)
    ax.set_xticks(range(len(table.columns))); ax.set_xticklabels(table.columns)
    ax.set_yticks(range(len(table.index))); ax.set_yticklabels(table.index)
    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            ax.text(j, i, f"{table.values[i, j]:+.1%}", ha="center", va="center", fontsize=8)
    ax.set_title(f"{market}: realised APY"); ax.set_xlabel("target LTV")
    plt.colorbar(im, ax=ax); plt.tight_layout()
    display(g.pivot_table(index="cell", columns="target_ltv", values="max_drawdown").round(4))''')

md("## 6. Hedge leg: coverage, settlements and mark-to-maturity")
code('''fig, axes = market_axes(len(markets), height=3.2)
for ax, market in zip(axes, markets):
    df = runs[f"{market}_boros"]
    if "BOROS_size" in df:
        cov = (df["BOROS_size"].fillna(0) * df["BOROS_underlying_price"].fillna(0)
               / df["LENDING_borrowed"].replace(0, np.nan))
        ax.plot(df["timestamp"], cov, label="boros coverage")
        ax2 = ax.twinx()
        ax2.plot(df["timestamp"], df["BOROS_realized_settlements"].fillna(0), color="tab:green", label="settlements")
        ax2.plot(df["timestamp"], (df["BOROS_balance"].ffill() - df["BOROS_collateral"].ffill()).fillna(0),
                 color="tab:red", label="unrealised MtM")
        ax2.legend(loc="upper left", fontsize=8)
    ax.set_title(f"{market}: Boros leg"); ax.grid(alpha=0.3); ax.legend(loc="upper right", fontsize=8)
    ax.tick_params(axis="x", rotation=30)
plt.tight_layout()''')

nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, "analysis.ipynb")
print("written analysis.ipynb with", len(cells), "cells")
