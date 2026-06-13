from events import FillEvent


class SimulatedExecutionHandler:
    """A simple broker simulation.

    To avoid look-ahead bias, an order generated from a bar's close is **not** filled on that same
    bar. Instead it is held as pending and filled at the *next* bar's open price -- the earliest
    point at which the order could realistically be executed.
    """

    def __init__(self, events_queue, data_handler):
        self.events = events_queue
        self.data_handler = data_handler
        self.pending_orders = []

    def execute_order(self, event):
        """Queue an order to be filled on the next bar (no same-bar fills)."""
        if event.type == 'ORDER':
            self.pending_orders.append(event)

    def process_pending_orders(self):
        """Fill any orders queued on previous bars at the current bar's open price.

        Call this at the start of each new bar, before the strategy runs.
        """
        if not self.pending_orders:
            return

        latest_bar = self.data_handler.get_latest_bars(1)[0]
        timeindex = latest_bar[0]
        fill_price = latest_bar[1]['Open']  # fill at the open of the new bar

        for order in self.pending_orders:
            fill_event = FillEvent(
                timeindex, order.symbol, 'SIMULATED_EXCHANGE',
                order.quantity, order.direction, fill_price
            )
            self.events.put(fill_event)

        self.pending_orders = []
