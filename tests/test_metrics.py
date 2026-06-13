import pandas as pd
import pytest

from metrics import annualised_sharpe, max_drawdown, buy_and_hold_metrics


def test_sharpe_zero_when_no_volatility():
    # Constant returns -> zero standard deviation -> Sharpe defined as 0.
    assert annualised_sharpe([0.01, 0.01, 0.01]) == 0.0


def test_sharpe_positive_for_upward_drift():
    assert annualised_sharpe([0.01, 0.02, -0.005, 0.015, 0.008]) > 0


def test_max_drawdown_basic():
    # Largest drop is from the peak of 120 down to 90 -> -25%.
    assert max_drawdown([100, 120, 90, 130]) == pytest.approx(-0.25)


def test_max_drawdown_monotonic_increase_is_zero():
    assert max_drawdown([100, 110, 120, 130]) == 0.0


def test_buy_and_hold_return():
    close = pd.Series([100.0, 105.0, 110.0])
    result = buy_and_hold_metrics(close, initial_capital=100000.0)
    assert result['return'] == pytest.approx(0.10)
    assert result['max_dd'] == 0.0  # price only rises
