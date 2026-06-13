import queue
import matplotlib.pyplot as plt
import pandas as pd

from data import HistoricCSVDataHandler
from strategy import MarkovChainStrategy
from portfolio import Portfolio
from execution import SimulatedExecutionHandler

# --- Configuration ---
CSV_PATH = 'sample_data.csv'
SYMBOL = 'AAPL'
LOOKBACK_WINDOW = 60
VOL_WINDOW = 20
MARKOV_THRESHOLD = 0.50
Z_THRESHOLD = 0.25


def run_backtest():
    events_queue = queue.Queue()

    print("Initialising backtest with the optimised parameters...")
    data_handler = HistoricCSVDataHandler(events_queue, CSV_PATH, SYMBOL)

    # --- Best parameters found via grid search ---
    strategy = MarkovChainStrategy(
        data_handler,
        events_queue,
        lookback_window=LOOKBACK_WINDOW,
        vol_window=VOL_WINDOW,
        markov_threshold=MARKOV_THRESHOLD,
        z_threshold=Z_THRESHOLD,
    )

    portfolio = Portfolio(data_handler, events_queue, initial_capital=100000.0)
    execution = SimulatedExecutionHandler(events_queue, data_handler)

    buy_signals = []
    sell_signals = []

    print("Running simulation and computing dynamic transition matrices...\n")

    while True:
        if not data_handler.update_bars():
            print("\nAnalytical simulation finished.")
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
                    if event.signal_type == 'LONG':
                        buy_signals.append(event.datetime)
                    elif event.signal_type == 'EXIT':
                        sell_signals.append(event.datetime)

                elif event.type == 'ORDER':
                    execution.execute_order(event)

                elif event.type == 'FILL':
                    portfolio.update_fill(event)

    # --- Compute and print quantitative performance metrics ---
    total_return, sharpe_ratio, max_dd = portfolio.calculate_performance_metrics()

    print("\n" + "=" * 40)
    print("         Strategy Performance Report")
    print("=" * 40)
    print(f"Total Return (ROI):       {total_return * 100:.2f}%")
    print(f"Annualised Sharpe Ratio:  {sharpe_ratio:.2f}")
    print(f"Maximum Drawdown:         {max_dd * 100:.2f}%")
    print("=" * 40)

    # --- Print the transition matrix ---
    final_matrix = strategy._build_transition_matrix(strategy.state_history)
    states_labels = ['Bearish (0)', 'Neutral (1)', 'Bullish (2)']
    matrix_df = pd.DataFrame(final_matrix, index=states_labels, columns=states_labels)

    print("\n" + "=" * 45)
    print("      Transition Matrix (probabilities)")
    print("=" * 45)
    print(matrix_df.map(lambda x: f"{x * 100:.1f}%"))
    print("=" * 45)

    # --- Plot the price chart with signals ---
    df = pd.read_csv(CSV_PATH, index_col=0, parse_dates=True)
    df.sort_index(inplace=True)

    plt.figure(figsize=(14, 6))
    plt.plot(df.index, df['Close'], label=f'{SYMBOL} Price', color='gray', alpha=0.5)

    # Overlay a 100-day SMA so the trend is visible at a glance.
    df['SMA_100'] = df['Close'].rolling(window=100).mean()
    plt.plot(df.index, df['SMA_100'], label='SMA 100', color='blue', alpha=0.3, linestyle='--')

    if buy_signals:
        plt.scatter(df.loc[buy_signals].index, df.loc[buy_signals]['Close'], marker='^',
                    color='green', s=120, label='Buy (Markov)')
    if sell_signals:
        plt.scatter(df.loc[sell_signals].index, df.loc[sell_signals]['Close'], marker='v',
                    color='red', s=120, label='Exit (Markov)')

    plt.title(
        f'Markov Chain Backtest '
        f'(Lookback: {LOOKBACK_WINDOW}, Markov Thresh: {MARKOV_THRESHOLD}, Z Thresh: {Z_THRESHOLD})',
        fontsize=14,
    )
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


if __name__ == "__main__":
    run_backtest()
