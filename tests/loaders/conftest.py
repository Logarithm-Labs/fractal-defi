"""Loader-suite conftest.

GitHub-hosted runners live in datacenter IP ranges that several
exchanges geo-block (notably Binance: ``fapi.binance.com`` returns
HTTP 451 "Service unavailable from a restricted location" to AWS
US-East / Azure egress, etc.). These tests are still useful when run
from a developer machine or an unblocked CI runner, so deleting them
is wrong; failing the whole integration job because of a runner-only
network policy is also wrong.

The hook below converts a ``LoaderHttpError`` carrying ``HTTP 451``
into a ``pytest.skip`` so the suite reports the geo-block honestly
without masking real loader regressions.
"""
import pytest

from fractal.loaders._http import LoaderHttpError


@pytest.hookimpl(hookwrapper=True, tryfirst=True)
def pytest_runtest_call(item):  # noqa: D401  (pytest hook signature)
    outcome = yield
    if outcome.excinfo is None:
        return
    exc = outcome.excinfo[1]
    if isinstance(exc, LoaderHttpError) and "HTTP 451" in str(exc):
        from _pytest.outcomes import Skipped
        outcome.force_exception(Skipped(
            f"endpoint geo-blocked from this runner: {str(exc)[:200]}"
        ))
