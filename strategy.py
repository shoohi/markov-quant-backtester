import numpy as np
import pandas as pd
from events import SignalEvent


class MarkovChainStrategy:
    """Pure statistical strategy: a Markov chain over volatility-adjusted (Z-score) states,
    with no lagging trend filters.

    Daily returns are converted into discrete states (Bearish / Neutral / Bullish) based on a
    rolling Z-score. A transition matrix estimated over a lookback window then predicts tomorrow's
    most likely state; the strategy goes long when a high-confidence bullish transition is expected
    and volume confirms it, and exits when a bearish transition becomes most likely.
    """

    def __init__(self, data_handler, events_queue, lookback_window=60, vol_window=20,
                 markov_threshold=0.40, z_threshold=0.25):
        self.bars = data_handler
        self.events = events_queue
        self.lookback_window = lookback_window
        self.vol_window = vol_window
        self.markov_threshold = markov_threshold
        self.z_threshold = z_threshold

        self.bought = {}
        self.state_history = []

    def _get_dynamic_states(self, returns, z_window):
        """Map each daily return to a state using a rolling Z-score.

        Returns a list where 0 = Bearish, 1 = Neutral, 2 = Bullish.
        """
        returns_series = pd.Series(returns)
        rolling_mean = returns_series.rolling(window=z_window).mean()
        rolling_std = returns_series.rolling(window=z_window).std().replace(0, 1e-8)

        z_scores = (returns_series - rolling_mean) / rolling_std

        states = []
        for z in z_scores:
            if pd.isna(z):
                states.append(1)
            elif z > self.z_threshold:
                states.append(2)
            elif z < -self.z_threshold:
                states.append(0)
            else:
                states.append(1)

        return states

    def _build_transition_matrix(self, states):
        """Estimate a 3x3 row-normalised Markov transition matrix from a state sequence."""
        matrix = np.zeros((3, 3))
        if len(states) < 2:
            return np.full((3, 3), 1 / 3)

        for i in range(len(states) - 1):
            current_state = states[i]
            next_state = states[i + 1]
            matrix[current_state][next_state] += 1

        for i in range(3):
            row_sum = matrix[i].sum()
            if row_sum > 0:
                matrix[i] /= row_sum
            else:
                matrix[i] = np.array([1 / 3, 1 / 3, 1 / 3])
        return matrix

    def calculate_signals(self, event):
        if event.type == 'MARKET':
            symbol = self.bars.symbol

            # We only need enough bars for the Markov lookback plus the volatility window
            # (no need to wait for long 200/300-day SMAs).
            required_bars = self.lookback_window + self.vol_window + 1
            bars = self.bars.get_latest_bars(required_bars)

            if len(bars) >= required_bars:
                closes = [b[1]['Close'] for b in bars]

                # --- Volume filter ---
                volumes = [b[1]['Volume'] for b in bars]
                current_volume = volumes[-1]
                sma_vol = sum(volumes[-self.vol_window:]) / self.vol_window
                is_high_volume = current_volume > sma_vol

                # --- Pure Markov logic ---
                markov_closes = closes[-(self.lookback_window + self.vol_window + 1):]
                daily_returns = [(markov_closes[i] - markov_closes[i - 1]) / markov_closes[i - 1]
                                 for i in range(1, len(markov_closes))]

                all_states = self._get_dynamic_states(daily_returns, self.vol_window)
                relevant_states = all_states[-self.lookback_window:]
                self.state_history = relevant_states

                transition_matrix = self._build_transition_matrix(relevant_states[:-1])
                current_state = relevant_states[-1]
                probabilities_for_tomorrow = transition_matrix[current_state]
                predicted_next_state = np.argmax(probabilities_for_tomorrow)

                if symbol not in self.bought:
                    self.bought[symbol] = "OUT"

                # Entry: bullish prediction confirmed by volume (no trend filter).
                if predicted_next_state == 2 and probabilities_for_tomorrow[2] > self.markov_threshold:
                    if is_high_volume and self.bought[symbol] == "OUT":
                        dt = bars[-1][0]
                        self.events.put(SignalEvent(symbol, dt, 'LONG'))
                        self.bought[symbol] = "LONG"

                # Exit: bearish prediction while holding a position.
                elif predicted_next_state == 0 and self.bought[symbol] == "LONG":
                    dt = bars[-1][0]
                    self.events.put(SignalEvent(symbol, dt, 'EXIT'))
                    self.bought[symbol] = "OUT"
