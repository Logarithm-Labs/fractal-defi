fractal.loaders.bridges.decoders module
=======================================

Pure decoders for bridge delivery events. Each takes a log's ``topics`` and
``data`` (``0x`` hex) and returns plain Python values: integer amounts in the
token's base units and lower-case ``0x`` addresses. They read no chain; the
caller attaches the transaction hash and log index.

Covered events: Across V3 ``FilledRelay`` (with sped-up output and recipient),
deBridge DLN ``FulfilledOrder`` (the documented signature and the deployed one
with an extra filled amount), LayerZero OFT ``OFTReceived``, CCTP V1
``MintAndWithdraw`` and Circle Gateway ``AttestationUsed``. A log whose
``topics[0]`` does not match the decoder's event raises ``ValueError``.

``abi_decode`` is the small ABI decoder behind them (``uintN``, ``address``,
``bytes32``, ``bytes`` and nested tuples); it needs no extra dependency.

.. automodule:: fractal.loaders.bridges.decoders
   :members:
   :show-inheritance:
   :undoc-members:
