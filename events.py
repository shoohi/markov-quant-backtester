class Event(object):
    """Base class for every event flowing through the system."""
    pass


class MarketEvent(Event):
    """Fired whenever new market data arrives (a new row in the CSV)."""

    def __init__(self):
        self.type = 'MARKET'


class SignalEvent(Event):
    """Emitted by the strategy when it identifies a trading opportunity."""

    def __init__(self, symbol, datetime, signal_type):
        self.type = 'SIGNAL'
        self.symbol = symbol            # e.g. 'AAPL'
        self.datetime = datetime        # when the signal was generated
        self.signal_type = signal_type  # 'LONG' (enter) or 'EXIT' (close)


class OrderEvent(Event):
    """Sent by the portfolio to the broker to place an actual order."""

    def __init__(self, symbol, order_type, quantity, direction):
        self.type = 'ORDER'
        self.symbol = symbol
        self.order_type = order_type  # 'MKT' (market) or 'LMT' (limit)
        self.quantity = quantity      # number of shares
        self.direction = direction    # 'BUY' or 'SELL'

    def print_order(self):
        print(f"Order: {self.direction} {self.quantity} {self.symbol} ({self.order_type})")


class FillEvent(Event):
    """Returned by the broker once an order is executed, so the portfolio can update its cash."""

    def __init__(self, timeindex, symbol, exchange, quantity, direction, fill_cost, commission=None):
        self.type = 'FILL'
        self.timeindex = timeindex
        self.symbol = symbol
        self.exchange = exchange
        self.quantity = quantity
        self.direction = direction
        self.fill_cost = fill_cost  # price at which the trade was executed

        # Default to a simple Interactive Brokers-style commission if none supplied.
        if commission is None:
            self.commission = max(1.3, 0.005 * quantity)
        else:
            self.commission = commission
