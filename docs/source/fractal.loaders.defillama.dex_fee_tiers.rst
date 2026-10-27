fractal.loaders.defillama.dex\_fee\_tiers module
==================================================

``DefiLlamaDexFeeTierLoader`` lists the DefiLlama yield pools on one chain
whose underlying tokens are exactly a requested pair and ranks them by the
swap fee parsed from each pool's ``poolMeta`` label. ``poolMeta`` is free
text, so ``parse_pool_fee_pct`` accepts only one percentage, a percentage
labelled as the fee, or one basis-point value; every other label yields
``NaN`` rather than an error. The fee tier excludes price impact, dynamic or
hook fees and gas, so it is a lower bound on the swap cost.

.. automodule:: fractal.loaders.defillama.dex_fee_tiers
   :members:
   :show-inheritance:
   :undoc-members:
