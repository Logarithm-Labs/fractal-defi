fractal.loaders.bridges.fees module
===================================

Pure helpers for realised bridge fees. ``fee_bps`` converts one transfer's
input and output amounts into a fee in basis points, rescaling 18-decimal and
6-decimal legs of the same asset and rejecting swaps and outliers;
``fee_stats`` summarises many fees (median, quartiles, volume-weighted mean).

``fit_fee_schedule`` fits the affine per-route schedule
``fee = fixed + bps * size / 1e4`` by least squares with both parts clipped at
zero, and returns a ``FeeSchedule`` that evaluates the fee, or the all-in fee
in basis points, at any size.

.. automodule:: fractal.loaders.bridges.fees
   :members:
   :show-inheritance:
   :undoc-members:
