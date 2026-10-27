fractal.loaders.bridges.fees module
===================================

Pure helpers for realised bridge fees. ``fee_bps`` converts one transfer's
input and output amounts into a fee in basis points, rescaling 18-decimal and
6-decimal legs of the same asset and rejecting swaps and outliers;
``fee_stats`` summarises many fees (median, quartiles, volume-weighted mean).
The affine per-route fit ``fee = fixed + bps * size`` is in core, in
``fractal.core.entities.models.route_costs``.

.. automodule:: fractal.loaders.bridges.fees
   :members:
   :show-inheritance:
   :undoc-members:
