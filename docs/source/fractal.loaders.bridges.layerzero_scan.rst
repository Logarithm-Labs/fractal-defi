fractal.loaders.bridges.layerzero_scan module
==============================================

``LayerZeroScanLoader`` queries the keyless LayerZero Scan v1 API for OFT
messages on one source-EID, destination-EID and application-address route.
The window is UTC and half-open. ``source_timestamp`` and
``destination_timestamp`` are Unix seconds; ``latency_seconds`` is the
destination timestamp minus the source timestamp and is missing until a
destination timestamp exists.

Pagination follows ``nextToken``. LayerZero Scan's documented HTTP 404 on an
empty continuation page ends the scan. ``max_messages`` applies a deterministic,
even time subsample after all matching messages are collected and before any
per-message follow-up work.

.. automodule:: fractal.loaders.bridges.layerzero_scan
   :members:
   :show-inheritance:
   :undoc-members:
