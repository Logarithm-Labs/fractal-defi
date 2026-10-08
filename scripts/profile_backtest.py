"""Profile one deterministic year-long strategy run without external services."""
import argparse
import cProfile
import pstats
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fractal.core.base import BaseStrategy, BaseStrategyParams, NamedEntity, Observation
from fractal.core.entities.simple.spot import SimpleSpotExchange, SimpleSpotExchangeGlobalState


@dataclass
class ProfileParams(BaseStrategyParams):
    """No-op parameters keep the representative strategy typed."""


class ProfileStrategy(BaseStrategy[ProfileParams]):
    """Minimal hold strategy used to expose framework overhead."""

    def set_up(self) -> None:
        self.register_entity(NamedEntity("SPOT", SimpleSpotExchange(trading_fee=0.0)))

    def predict(self):
        return []


def build_observations(count: int = 365 * 24):
    """Build a fixed hourly trajectory; no loader cache or network is involved."""
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    return [
        Observation(
            timestamp=start + timedelta(hours=index),
            states={"SPOT": SimpleSpotExchangeGlobalState(close=2_000.0 + index % 100)},
        )
        for index in range(count)
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("profile-artifacts/fractal-backtest.prof"),
        help="cProfile artifact path",
    )
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    observations = build_observations()
    profiler = cProfile.Profile()
    profiler.enable()
    ProfileStrategy(params=ProfileParams()).run(observations)
    profiler.disable()
    profiler.dump_stats(args.output)

    text_path = args.output.with_suffix(".txt")
    with text_path.open("w", encoding="utf-8") as stream:
        stats = pstats.Stats(profiler, stream=stream).strip_dirs().sort_stats("cumulative")
        stats.print_stats(80)
    print(f"wrote {args.output} and {text_path}")
    print(f"inspect interactively with: snakeviz {args.output}")


if __name__ == "__main__":
    main()
