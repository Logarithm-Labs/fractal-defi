"""Deprecated shim — use :mod:`fractal.loaders.simulations.monte_carlo_gbm`.

The original implementation here was an arithmetic-returns multiplicative
walk that could produce negative prices and was not vectorized. It has
been replaced with proper log-normal GBM in
:class:`MonteCarloPriceLoader`. ``MonteCarloHourPriceLoader`` is kept as
a thin alias so existing call sites and tests continue to work.
"""
import warnings

from fractal.loaders.simulations.monte_carlo_gbm import MonteCarloPriceLoader


class MonteCarloHourPriceLoader(MonteCarloPriceLoader):
    """Deprecated. Use :class:`MonteCarloPriceLoader` directly.

    Behavior change: the simulation is now exact log-normal GBM with the
    same calibrated σ. Trajectories are strictly positive and the first
    value of every path equals the historical first price (``S_0``).
    """

    def __init__(self, *args, **kwargs):
        warnings.warn(
            "MonteCarloHourPriceLoader is deprecated; use "
            "MonteCarloPriceLoader from fractal.loaders.simulations.monte_carlo_gbm. "
            "The simulation is now true log-normal GBM (was an arithmetic-returns walk).",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)
