import pandas as pd
from events import MarketEvent


class HistoricCSVDataHandler:
    """Reads market data from a CSV file and pushes a MarketEvent onto the queue on demand."""

    def __init__(self, events_queue, csv_path=None, symbol=None, data=None):
        self.events = events_queue
        self.symbol = symbol

        # Accept either a CSV path or an in-memory DataFrame (handy for train/test splits).
        if data is not None:
            self.data = data.copy()
        elif csv_path is not None:
            self.data = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        else:
            raise ValueError("Provide either csv_path or data.")
        self.data.sort_index(inplace=True)

        # Iterator that lets us pull one row at a time.
        self.data_iterator = self.data.iterrows()

        # Rows already seen, kept to avoid look-ahead bias.
        self.latest_symbol_data = []

    def get_latest_bars(self, N=1):
        """Return the last N bars the system has already seen.

        Useful for computing moving averages or other rolling statistics.
        """
        return self.latest_symbol_data[-N:]

    def update_bars(self):
        """Pull the next historical bar and push a MarketEvent.

        Returns False once the data is exhausted.
        """
        try:
            index, row = next(self.data_iterator)
            # Expose the new row to the rest of the system.
            self.latest_symbol_data.append((index, row))
            # Notify the system that new data is available.
            self.events.put(MarketEvent())
            return True
        except StopIteration:
            # Reached the end of the file.
            return False
