"""Deprecated shim — use :mod:`fractal.core.entities.simple_spot`.

The class was renamed to :class:`SimpleSpotExchange` for naming consistency
with :class:`SimplePerpEntity`, and the trading semantics were corrected
to match :class:`BaseSpotEntity` (``buy`` takes ``amount_in_notional``,
``sell`` takes ``amount_in_product`` — previously both took bare ``amount``
in product, silently violating the base contract).
"""
import warnings

from fractal.core.entities.simple.spot import (SimpleSpotExchange,
                                               SimpleSpotExchangeException,
                                               SimpleSpotExchangeGlobalState,
                                               SimpleSpotExchangeInternalState)


class SingleSpotExchange(SimpleSpotExchange):
    """Deprecated alias for :class:`SimpleSpotExchange`.

    .. note::
        Trading semantics changed alongside the rename:
        ``action_buy`` now takes ``amount_in_notional`` (was bare ``amount``
        in product), ``action_sell`` now takes ``amount_in_product`` (was
        bare ``amount``). Update existing ``Action(...)`` payloads.
    """

    def __init__(self, *args, **kwargs):
        warnings.warn(
            "SingleSpotExchange is deprecated; use SimpleSpotExchange from "
            "fractal.core.entities.simple_spot. Trading args also changed: "
            "buy now takes amount_in_notional, sell takes amount_in_product.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)


# Re-export the state aliases too so old `from single_spot_exchange import ...` still works.
SingleSpotExchangeGlobalState = SimpleSpotExchangeGlobalState
SingleSpotExchangeInternalState = SimpleSpotExchangeInternalState

__all__ = [
    "SingleSpotExchange",
    "SingleSpotExchangeGlobalState",
    "SingleSpotExchangeInternalState",
    "SimpleSpotExchangeException",
]
