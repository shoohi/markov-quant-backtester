from events import FillEvent


class SimulatedExecutionHandler:
    """A simple broker simulation that fills orders at the current close price with no slippage or delay."""

    def __init__(self, events_queue, data_handler):
        self.events = events_queue
        self.data_handler = data_handler

    def execute_order(self, event):
        if event.type == 'ORDER':
            # Grab the latest (current) bar to determine the fill price.
            latest_bar = self.data_handler.get_latest_bars(1)[0]

            timeindex = latest_bar[0]               # current timestamp
            fill_cost = latest_bar[1]['Close']      # current close price

            # Create a FILL event to tell the portfolio the trade went through.
            fill_event = FillEvent(
                timeindex, event.symbol, 'SIMULATED_EXCHANGE',
                event.quantity, event.direction, fill_cost
            )

            # Push the execution confirmation back onto the queue.
            self.events.put(fill_event)
