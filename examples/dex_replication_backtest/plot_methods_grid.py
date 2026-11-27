"""Plot the three-method comparison: fee error per position and wall time.

Reads ``results/methods_grid_results.csv`` (written by
``methods_grid_backtests.py``) and plots

(a) the absolute relative error of backtested against on-chain fee accrual
    (USD) per position for ``fee_growth``, ``aggregate`` and the
    ``per_event`` swap-level reference over the six-day window, plus
    ``fee_growth`` over the five-month re-run on release 1.4.0
    (``results/rerun_2026-11-27/summary.csv``);
(b) the full wall time per position (data pull + preparation + run) against
    the number of in-range swaps, for ``fee_growth`` and ``per_event``.

    python plot_methods_grid.py [CSV] [OUT] [RERUN_CSV]

Defaults: ``results/methods_grid_results.csv`` and
``results/rerun_2026-11-27/summary.csv`` -> ``results/methods_grid.pdf``.
Needs matplotlib. The figure appears in the Fractal JORS software paper.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parent
DEFAULT_CSV = HERE / "results" / "methods_grid_results.csv"
DEFAULT_OUT = HERE / "results" / "methods_grid.pdf"
DEFAULT_RERUN = HERE / "results" / "rerun_2026-11-27" / "summary.csv"

# Categorical slots 1-3 of the validated default palette (blue, orange, aqua),
# each paired with its own marker so identity never rests on colour alone.
METHODS = [
    ("growth", "fee_growth model", "#2a78d6", "o"),
    ("aggregate", "aggregate model", "#eb6834", "s"),
    ("event", "swap-level reference", "#1baf7a", "^"),
]
RERUN = ("fee_growth model, 5 months (release 1.4.0)", "#1c4f8f", "D")
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3de"


def main(csv_path: Path = DEFAULT_CSV, out: Path = DEFAULT_OUT, rerun_csv: Path = DEFAULT_RERUN) -> None:
    df = pd.read_csv(csv_path).sort_values(["pair", "fee_tier"]).reset_index(drop=True)
    rerun = pd.read_csv(rerun_csv)[["replicated_position_id", "usd_ratio"]]
    df = df.merge(rerun.rename(columns={"usd_ratio": "rerun_usd_ratio"}), on="replicated_position_id",
                  how="left", validate="one_to_one")
    labels = [f"{p} {t / 1e4:g}%" for p, t in zip(df["pair"], df["fee_tier"], strict=True)]

    plt.rcParams.update({"font.family": "sans-serif", "font.size": 8, "axes.edgecolor": MUTED,
                         "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED})
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(7.2, 2.75), gridspec_kw={"width_ratios": [1.55, 1]})

    x = range(len(df))
    series = [(f"{key}_usd_ratio", label + ", 6 days" if key == "growth" else label, color, marker)
              for key, label, color, marker in METHODS]
    series.insert(1, ("rerun_usd_ratio", *RERUN))
    for offset, (column, label, color, marker) in zip((-0.3, -0.1, 0.1, 0.3), series, strict=True):
        err = (df[column] - 1.0).abs().clip(lower=1e-10)
        ax_a.scatter([i + offset for i in x], err, s=20, color=color, marker=marker,
                     edgecolor="white", linewidth=0.6, zorder=3, label=label)
    ax_a.set_yscale("log")
    ax_a.set_ylim(1e-10, 1e4)
    ax_a.set_xticks(list(x), labels, rotation=60, ha="right", fontsize=6.5)
    ax_a.set_ylabel("|backtest / on-chain - 1|  (USD fees)")
    ax_a.set_title("(a) Fee error per position", loc="left", fontsize=8.5, color=INK)
    ax_a.legend(frameon=False, fontsize=6.2, loc="upper center", ncol=2, handletextpad=0.2, columnspacing=1.0)

    for key, label, color, marker in (METHODS[0], METHODS[2]):
        ax_b.scatter(df["swaps_in_range"], df[f"{key}_total_seconds"], s=22, color=color,
                     marker=marker, edgecolor="white", linewidth=0.6, zorder=3, label=label)
    ax_b.set_xscale("log")
    ax_b.set_xlabel("in-range swaps in the window")
    ax_b.set_ylabel("wall time per position (s)")
    ax_b.set_title("(b) Cost: snapshots vs. events", loc="left", fontsize=8.5, color=INK)
    ax_b.legend(frameon=False, fontsize=6.5, loc="upper left")
    ax_b.set_ylim(0, None)

    for ax in (ax_a, ax_b):
        ax.grid(axis="y", color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    worst = {column: float((df[column] - 1).abs().max()) for column, *_ in series}
    print(f"wrote {out}; worst |ratio-1|: {worst}")
    print(df[["swaps_in_range", "growth_total_seconds", "event_total_seconds"]].describe().loc[["min", "50%", "max"]])


if __name__ == "__main__":
    main(*(Path(a) for a in sys.argv[1:4]))
