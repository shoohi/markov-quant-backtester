import queue
import itertools
import pandas as pd

from data import HistoricCSVDataHandler
from strategy import MarkovChainStrategy
from portfolio import Portfolio
from execution import SimulatedExecutionHandler

CSV_PATH = 'sample_data.csv'
SYMBOL = 'AAPL'
TRAIN_FRACTION = 0.7  # first 70% of bars used to optimise, last 30% held out for validation


def run_headless_backtest(data, symbol, lookback, markov_threshold, z_threshold, vol_window=20):
    """Run a single backtest on an in-memory DataFrame and return (return, sharpe, max_dd)."""
    events_queue = queue.Queue()
    data_handler = HistoricCSVDataHandler(events_queue, symbol=symbol, data=data)

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

        current_dt = data_handler.get_latest_bars(1)[0][0]
        execution.process_pending_orders()

        while True:
            try:
                event = events_queue.get(False)
            except queue.Empty:
                break

            if event is not None:
                if event.type == 'MARKET':
                    strategy.calculate_signals(event)
                elif event.type == 'SIGNAL':
                    portfolio.update_signal(event)
                elif event.type == 'ORDER':
                    execution.execute_order(event)
                elif event.type == 'FILL':
                    portfolio.update_fill(event)

        portfolio.record_current_equity(current_dt)

    return portfolio.calculate_performance_metrics()


def grid_search_optimization():
    full = pd.read_csv(CSV_PATH, index_col=0, parse_dates=True).sort_index()

    split = int(len(full) * TRAIN_FRACTION)
    train, test = full.iloc[:split], full.iloc[split:]
    print(f"Loaded {len(full)} bars -> {len(train)} train / {len(test)} test (held out).\n")

    # Search space across the three strategy parameters.
    lookback_windows = [30, 60, 90, 120, 150]
    markov_thresholds = [0.20, 0.40, 0.45, 0.50]
    z_thresholds = [0.25, 0.5, 0.75, 1.0]

    param_combinations = list(itertools.product(lookback_windows, markov_thresholds, z_thresholds))
    print(f"Starting grid search over {len(param_combinations)} combinations (in-sample / train set)...")

    results = []
    for i, (lookback, m_thresh, z_thresh) in enumerate(param_combinations):
        ret, sharpe, max_dd = run_headless_backtest(train, SYMBOL, lookback, m_thresh, z_thresh)
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

    results_df = pd.DataFrame(results).sort_values(by='Sharpe', ascending=False)

    print("\n" + "=" * 65)
    print("      Top 5 Parameter Sets (in-sample, ranked by Sharpe Ratio)")
    print("=" * 65)
    print(results_df.head(5).to_string(index=False))
    print("=" * 65)

    # --- Out-of-sample validation: evaluate the single best in-sample set on the held-out test data ---
    best = results_df.iloc[0]
    oos_ret, oos_sharpe, oos_dd = run_headless_backtest(
        test, SYMBOL, int(best['Lookback']), float(best['M_Thresh']), float(best['Z_Thresh'])
    )

    print("\n" + "=" * 65)
    print("      Out-of-Sample Validation (best in-sample params on test set)")
    print("=" * 65)
    print(f"Params: Lookback={int(best['Lookback'])}, "
          f"M_Thresh={best['M_Thresh']}, Z_Thresh={best['Z_Thresh']}")
    print("-" * 65)
    print(f"{'Metric':<20}{'In-Sample':>20}{'Out-of-Sample':>20}")
    print(f"{'Return':<20}{best['Return'] * 100:>19.2f}%{oos_ret * 100:>19.2f}%")
    print(f"{'Sharpe':<20}{best['Sharpe']:>20.2f}{oos_sharpe:>20.2f}")
    print(f"{'Max Drawdown':<20}{best['Max_DD'] * 100:>19.2f}%{oos_dd * 100:>19.2f}%")
    print("=" * 65)
    print("\nNote: a large gap between in-sample and out-of-sample performance is a sign of overfitting.")


if __name__ == "__main__":
    grid_search_optimization()
