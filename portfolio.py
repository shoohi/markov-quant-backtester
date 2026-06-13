import pandas as pd

from events import OrderEvent
from metrics import annualised_sharpe, max_drawdown


class Portfolio:
    def __init__(self, bars, events_queue, initial_capital=100000.0):
        self.bars = bars
        self.events = events_queue
        self.initial_capital = initial_capital
        self.current_cash = initial_capital
        self.positions = {}

        # Portfolio equity history, used to compute metrics at the end of the run.
        self.equity_curve = []

    def update_signal(self, event):
        if event.type == 'SIGNAL':
            symbol = event.symbol
            direction = event.signal_type
            quantity = 100

            if direction == 'LONG':
                order = OrderEvent(symbol, 'MKT', quantity, 'BUY')
                self.events.put(order)
            elif direction == 'EXIT':
                if symbol in self.positions and self.positions[symbol] > 0:
                    order = OrderEvent(symbol, 'MKT', self.positions[symbol], 'SELL')
                    self.events.put(order)

    def update_fill(self, event):
        if event.type == 'FILL':
            symbol = event.symbol
            if symbol not in self.positions:
                self.positions[symbol] = 0

            transaction_cost = (event.quantity * event.fill_cost)

            if event.direction == 'BUY':
                self.positions[symbol] += event.quantity
                self.current_cash -= (transaction_cost + event.commission)
            elif event.direction == 'SELL':
                self.positions[symbol] -= event.quantity
                self.current_cash += (transaction_cost - event.commission)

    def record_current_equity(self, dt):
        """Compute and record total equity (cash + market value of holdings) for the current day."""
        latest_bar = self.bars.get_latest_bars(1)
        close_price = latest_bar[0][1]['Close'] if latest_bar else 0

        symbol = self.bars.symbol
        share_quantity = self.positions.get(symbol, 0)

        # Total equity = free cash + (shares held * current close price)
        total_equity = self.current_cash + (share_quantity * close_price)
        self.equity_curve.append({'Date': dt, 'Equity': total_equity})

    def calculate_performance_metrics(self):
        """Compute ROI, annualised Sharpe ratio and maximum drawdown for the equity curve."""
        df = pd.DataFrame(self.equity_curve).set_index('Date')

        total_return = (df['Equity'].iloc[-1] - self.initial_capital) / self.initial_capital
        sharpe_ratio = annualised_sharpe(df['Equity'].pct_change())
        max_dd = max_drawdown(df['Equity'])

        return total_return, sharpe_ratio, max_dd
