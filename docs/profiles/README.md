# Profiling and benchmarks

Performance checks are deliberately separate from the default offline test suite.
They use fixed, in-memory observations and never call loaders, MLflow servers, or
other network services.

## Benchmarks

```bash
make benchmark
```

This runs the `benchmark` pytest marker and reports three representative paths:

- one year (8,760 hourly bars) through `BaseStrategy.run`;
- one year of `UniswapV3LPEntity.update_state` calls;
- a 16-cell `DefaultPipeline` grid with external MLflow I/O stubbed.

Wall-clock values depend on the host. Compare results from the same machine and
Python version; do not treat a result produced on one CI runner as a universal
performance guarantee.

## cProfile

```bash
make profile
```

The command writes `profile-artifacts/fractal-backtest.prof` plus a text report.
Generated profiles are ignored because they contain machine- and interpreter-
specific timings. Inspect the call graph interactively with:

```bash
make profile-view
# or: snakeviz profile-artifacts/fractal-backtest.prof
```
