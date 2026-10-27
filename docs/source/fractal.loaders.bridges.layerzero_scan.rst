fractal.loaders.bridges.layerzero_scan module
==============================================

``LayerZeroScanLoader`` queries the keyless LayerZero Scan v1 API for OFT
messages on one source-EID, destination-EID and application-address route.
The window is UTC and half-open. ``src_ts`` and ``dst_ts`` are Unix seconds;
``dst_ts`` is missing until the message executes on the destination.
``latency_s`` is ``dst_ts - src_ts`` for ``DELIVERED`` messages and NaN for
any other status (a stored, failed payload has a destination transaction but
is not a completed transfer). ``amount`` is the token amount sent, decoded
from the OFT payload's ``amountSD`` with ``shared_decimals`` (default 6).

Pagination follows ``nextToken``. LayerZero Scan's documented HTTP 404 on an
empty continuation page ends the scan; a repeated token raises. Messages are
deduplicated by GUID and ordered by source timestamp. ``max_messages`` keeps a
deterministic, even time subsample of the matching messages, so per-message
follow-up work (for example fee lookups from source receipts) stays bounded.

.. automodule:: fractal.loaders.bridges.layerzero_scan
   :members:
   :show-inheritance:
   :undoc-members:
