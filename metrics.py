"""Reusable performance metrics shared by the portfolio, the benchmark and the tests."""

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


def annualised_sharpe(daily_returns, periods=TRADING_DAYS_PER_YEAR):
    """Annualised Sharpe ratio of a daily return series (0% risk-free rate).

    Returns 0.0 when volatility is undefined or zero.
    """
    returns = pd.Series(daily_returns).dropna()
    std = returns.std()
    if std and std > 0 and not np.isnan(std):
        return (returns.mean() / std) * np.sqrt(periods)
    return 0.0


def max_drawdown(equity):
    """Maximum peak-to-trough drawdown of an equity curve, as a negative fraction."""
    equity = pd.Series(equity)
    if equity.empty:
        return 0.0
    peak = equity.cummax()
    drawdown = (equity - peak) / peak
    return float(drawdown.min())


def buy_and_hold_metrics(close, initial_capital=100000.0):
    """Return/Sharpe/max-drawdown for buying the asset on day one and holding.

    Provides an apples-to-apples benchmark for any active strategy on the same data.
    """
    close = pd.Series(close).dropna()
    shares = initial_capital / close.iloc[0]
    equity = shares * close

    total_return = (close.iloc[-1] - close.iloc[0]) / close.iloc[0]
    return {
        'return': float(total_return),
        'sharpe': annualised_sharpe(close.pct_change()),
        'max_dd': max_drawdown(equity),
    }
