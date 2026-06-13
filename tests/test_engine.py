import queue

import pandas as pd
import pytest

from data import HistoricCSVDataHandler
from execution import SimulatedExecutionHandler
from events import OrderEvent


def make_data(rows):
    """Build a small OHLCV DataFrame indexed by date."""
    idx = pd.to_datetime([f"2023-01-0{i + 1}" for i in range(len(rows))])
    return pd.DataFrame(rows, index=idx)


def test_data_handler_reveals_bars_in_order_without_lookahead():
    df = make_data([
        {'Open': 10, 'Close': 11, 'Volume': 100},
        {'Open': 11, 'Close': 12, 'Volume': 100},
        {'Open': 12, 'Close': 13, 'Volume': 100},
    ])
    handler = HistoricCSVDataHandler(queue.Queue(), symbol='TEST', data=df)

    # Before any update, nothing has been seen.
    assert handler.get_latest_bars(5) == []

    handler.update_bars()
    assert len(handler.get_latest_bars(99)) == 1  # only the first bar is visible

    handler.update_bars()
    assert handler.get_latest_bars(1)[0][1]['Close'] == 12

    assert handler.update_bars() is True
    assert handler.update_bars() is False  # data exhausted


def test_order_fills_at_next_bar_open_not_current():
    """An order decided on bar N must fill at bar N+1's open price (no look-ahead)."""
    df = make_data([
        {'Open': 10, 'Close': 11, 'Volume': 100},
        {'Open': 20, 'Close': 21, 'Volume': 100},
    ])
    events = queue.Queue()
    handler = HistoricCSVDataHandler(events, symbol='TEST', data=df)
    execution = SimulatedExecutionHandler(events, handler)

    # Bar 1 arrives; the strategy "decides" to buy based on its close.
    handler.update_bars()
    execution.execute_order(OrderEvent('TEST', 'MKT', 100, 'BUY'))
    assert execution.pending_orders, "order should be queued, not filled on the same bar"

    # Bar 2 arrives; the pending order fills at bar 2's OPEN (20), never bar 1's price.
    handler.update_bars()
    execution.process_pending_orders()

    fills = [events.get() for _ in range(events.qsize())]
    fill_events = [e for e in fills if e.type == 'FILL']
    assert len(fill_events) == 1
    assert fill_events[0].fill_cost == 20
    assert execution.pending_orders == []
