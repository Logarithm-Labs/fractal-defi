"""Unit tests for one Aave V3 ``liquidationCall`` (``aave_math.liquidation_call``).

Expected values are worked by hand from ``LiquidationLogic.sol``
(aave-v3-origin @ 8305565): close factor 50%, raised to 100% when the
health factor is at or below 0.95 or when the position's collateral or
debt is under $2000; a partial call may not leave under $1000 of
collateral or debt; the borrower loses ``debt repaid × bonus`` of collateral.
"""
import pytest

from fractal.core.entities.models.aave_math import liquidation_call

pytestmark = pytest.mark.core


def test_no_liquidation_at_health_factor_one():
    """HF == 1 exactly is healthy on Aave (``HF < 1`` is required to liquidate)."""
    # HF = 0.8 · 10000 / 8000 = 1.0
    assert liquidation_call(10_000, 8_000, 1.0, 1.0, liq_thr=0.8, liquidation_bonus=1.05) == (0.0, 0.0)


def test_no_liquidation_without_debt():
    assert liquidation_call(10_000, 0.0, 1.0, 1.0, liq_thr=0.85, liquidation_bonus=1.05) == (0.0, 0.0)


def test_half_close_factor_above_095():
    """HF 0.984 and both legs ≥ $2000 ⇒ only 50% of the debt is repaid."""
    # HF = 0.85 · 10000 / (8000 · 1.08) = 0.98380; repay 4000, seize 4000 · 1.08 · 1.05 = 4536
    repaid, seized = liquidation_call(10_000, 8_000, 1.0, 1.08, liq_thr=0.85, liquidation_bonus=1.05)
    assert repaid == pytest.approx(4_000)
    assert seized == pytest.approx(4_536)


def test_full_close_factor_at_or_below_095():
    """HF 0.924 ≤ 0.95 ⇒ the whole debt is repaid."""
    # HF = 0.85 · 10000 / (8000 · 1.15) = 0.92391; seize 8000 · 1.15 · 1.05 = 9660
    repaid, seized = liquidation_call(10_000, 8_000, 1.0, 1.15, liq_thr=0.85, liquidation_bonus=1.05)
    assert repaid == pytest.approx(8_000)
    assert seized == pytest.approx(9_660)


def test_full_close_factor_at_exactly_095():
    """The 50% cap needs HF *strictly* above 0.95; HF == 0.95 repays everything."""
    # HF = 0.76 · 10000 / 8000 = 0.95 exactly; seize 8000 · 1.05 = 8400
    repaid, seized = liquidation_call(10_000, 8_000, 1.0, 1.0, liq_thr=0.76, liquidation_bonus=1.05)
    assert repaid == pytest.approx(8_000)
    assert seized == pytest.approx(8_400)


def test_full_close_factor_when_collateral_under_2000():
    """HF 0.984 but collateral $1500 < $2000 ⇒ the whole debt is repaid."""
    # seize 1200 · 1.08 · 1.05 = 1360.8
    repaid, seized = liquidation_call(1_500, 1_200, 1.0, 1.08, liq_thr=0.85, liquidation_bonus=1.05)
    assert repaid == pytest.approx(1_200)
    assert seized == pytest.approx(1_360.8)


def test_full_close_factor_when_debt_under_2000():
    """Collateral $2100 ≥ $2000 but debt $1814.4 < $2000 ⇒ the whole debt is repaid."""
    # HF = 0.85 · 2100 / 1814.4 = 0.98380; seize 1680 · 1.08 · 1.05 = 1905.12
    repaid, seized = liquidation_call(2_100, 1_680, 1.0, 1.08, liq_thr=0.85, liquidation_bonus=1.05)
    assert repaid == pytest.approx(1_680)
    assert seized == pytest.approx(1_905.12)


def test_collateral_shortfall_seizes_everything_and_clears_debt():
    """Debt · bonus above collateral ⇒ all collateral goes, the rest of the debt becomes reserve deficit."""
    # HF = 0.85 · 10000 / 9600 = 0.885; 9600 · 1.05 = 10080 > 10000
    assert liquidation_call(10_000, 9_600, 1.0, 1.0, liq_thr=0.85, liquidation_bonus=1.05) == (9_600, 10_000)


def test_worthless_collateral_clears_debt():
    """Collateral priced at 0 with debt outstanding: nothing to seize, the debt is written off."""
    assert liquidation_call(5.0, 1_000, 0.0, 1.0, liq_thr=0.85, liquidation_bonus=1.05) == (1_000, 5.0)


def test_leftover_rule_shrinks_a_dusty_half_liquidation():
    """A 50% call that would leave $990 of collateral is cut so exactly $1000 remains."""
    # 1 ETH at $2040, debt 2000 USDC: HF = 0.95 · 2040 / 2000 = 0.969 > 0.95 ⇒ cap 1000 USDC.
    # Repaying 1000 seizes $1050 and leaves $990 < $1000, so the liquidator may repay at most
    # (2040 − 1000) / 1.05 = 990.476 USDC, seizing $1040 = 1040 / 2040 ETH.
    repaid, seized = liquidation_call(1.0, 2_000, 2_040.0, 1.0, liq_thr=0.95, liquidation_bonus=1.05)
    assert repaid == pytest.approx(1_040 / 1.05)
    assert seized == pytest.approx(1_040 / 2_040)
    assert (1.0 - seized) * 2_040 == pytest.approx(1_000)


def test_leftover_of_exactly_1000_is_allowed():
    """The leftover floor is inclusive: $1000 of collateral left is fine."""
    # 1 ETH at $2050, debt 2000: HF 0.974, repay 1000, seize $1050, leave $1000
    repaid, seized = liquidation_call(1.0, 2_000, 2_050.0, 1.0, liq_thr=0.95, liquidation_bonus=1.05)
    assert repaid == pytest.approx(1_000)
    assert seized == pytest.approx(1_050 / 2_050)
