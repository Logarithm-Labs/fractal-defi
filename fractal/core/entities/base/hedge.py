"""Deprecated shim — use :mod:`fractal.core.entities.perp`.

The class was renamed from ``BaseHedgeEntity`` to ``BasePerpEntity`` to
match the actual semantics (single-position perpetual-futures entity).
The old name is kept as a transparent alias so existing imports and
``isinstance`` checks keep working.
"""
from fractal.core.entities.base.perp import BasePerpEntity

# Transparent alias — no DeprecationWarning on use because it appears
# both in subclass definitions and in `isinstance(...)` checks scattered
# across strategy code; warning on each one would be noise.
BaseHedgeEntity = BasePerpEntity

__all__ = ["BaseHedgeEntity"]
