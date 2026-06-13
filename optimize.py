import queue
import itertools
import pandas as pd

from data import HistoricCSVDataHandler
from strategy import MarkovChainStrategy
from portfolio import Portfolio
from execution import SimulatedExecutionHandler


def run_headless_backtest(csv_path, symbol, lookback, markov_threshold, z_threshold, vol_window=20):
    events_queue = queue.Queue()
    data_handler = HistoricCSVDataHandler(events_queue, csv_path, symbol)

    strategy = MarkovChainStrategy(
        data_handler, events_queue,
        lookback_window=lookback,
        vol_window=vol_window,
        markov_threshold=markov_threshold,
        z_threshold=z_threshold,
    )

    portfolio = Portfolio(data_handler, events_queue, initial_capital=100000.0)
    execution = SimulatedExecutionHandler(events_queue, data_handler)

    while True:
        if not data_handler.update_bars():
            break

        while True:
            try:
                event = events_queue.get(False)
            except queue.Empty:
                break

            if event is not None:
                if event.type == 'MARKET':
                    current_dt = data_handler.get_latest_bars(1)[0][0]
                    strategy.calculate_signals(event)
                    portfolio.record_current_equity(current_dt)

                elif event.type == 'SIGNAL':
                    portfolio.update_signal(event)

                elif event.type == 'ORDER':
                    execution.execute_order(event)

                elif event.type == 'FILL':
                    portfolio.update_fill(event)

    total_return, sharpe_ratio, max_dd = portfolio.calculate_performance_metrics()
    return total_return, sharpe_ratio, max_dd


def grid_search_optimization():
    csv_path = 'sample_data.csv'
    symbol = 'AAPL'

    # Search space across the three strategy parameters.
    lookback_windows = [30, 60, 90, 120, 150]
    markov_thresholds = [0.20, 0.40, 0.45, 0.50]
    z_thresholds = [0.25, 0.5, 0.75, 1.0]

    param_combinations = list(itertools.product(lookback_windows, markov_thresholds, z_thresholds))

    print(f"Starting grid search over {len(param_combinations)} combinations...")

    results = []

    for i, (lookback, m_thresh, z_thresh) in enumerate(param_combinations):
        ret, sharpe, max_dd = run_headless_backtest(csv_path, symbol, lookback, m_thresh, z_thresh)

        results.append({
            'Lookback': lookback,
            'M_Thresh': m_thresh,
            'Z_Thresh': z_thresh,
            'Return': ret,
            'Sharpe': sharpe,
            'Max_DD': max_dd,
        })

        if (i + 1) % 10 == 0:
            print(f"Completed {i + 1}/{len(param_combinations)} runs.")

    results_df = pd.DataFrame(results)

    # Sort by Sharpe ratio, descending.
    top_results = results_df.sort_values(by='Sharpe', ascending=False).head(5)

    print("\n" + "=" * 65)
    print("      Top 5 Parameter Sets (ranked by Sharpe Ratio)")
    print("=" * 65)
    print(top_results.to_string(index=False))
    print("=" * 65)


if __name__ == "__main__":
    grid_search_optimization()
