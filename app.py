import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import queue
import yfinance as yf
import datetime
import itertools
import time

from data import HistoricCSVDataHandler
from strategy import MarkovChainStrategy
from portfolio import Portfolio
from execution import SimulatedExecutionHandler
from metrics import buy_and_hold_metrics

SAMPLE_DATA_PATH = "sample_data.csv"
SAMPLE_SYMBOL = "AAPL"


def load_price_data(symbol, start_date, end_date):
    """Fetch live data from Yahoo Finance, falling back to the bundled sample dataset.

    Yahoo Finance often rate-limits shared cloud IPs, so this keeps the hosted demo working
    even when the live download fails. Returns (csv_path, effective_symbol, source) where
    ``source`` is 'live' or 'sample'.
    """
    try:
        raw_df = yf.Ticker(symbol).history(start=start_date, end=end_date)
    except Exception:
        raw_df = pd.DataFrame()

    if not raw_df.empty:
        csv_path = f"temp_{symbol}_data.csv"
        raw_df.to_csv(csv_path)
        return csv_path, symbol, "live"

    # Live data unavailable -> fall back to the bundled AAPL sample.
    return SAMPLE_DATA_PATH, SAMPLE_SYMBOL, "sample"


def run_simulation(csv_path, symbol, lookback, m_thresh, z_thresh):
    events_queue = queue.Queue()
    data_handler = HistoricCSVDataHandler(events_queue, csv_path, symbol)
    strategy = MarkovChainStrategy(
        data_handler, events_queue,
        lookback_window=lookback,
        markov_threshold=m_thresh,
        z_threshold=z_thresh
    )
    portfolio = Portfolio(data_handler, events_queue, initial_capital=100000.0)
    execution = SimulatedExecutionHandler(events_queue, data_handler)

    buy_signals, sell_signals = [], []

    while True:
        if not data_handler.update_bars():
            break

        current_dt = data_handler.get_latest_bars(1)[0][0]
        # Fill orders queued on the previous bar at this bar's open (no look-ahead).
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
                    if event.signal_type == 'LONG':
                        buy_signals.append(event.datetime)
                    elif event.signal_type == 'EXIT':
                        sell_signals.append(event.datetime)
                elif event.type == 'ORDER':
                    execution.execute_order(event)
                elif event.type == 'FILL':
                    portfolio.update_fill(event)

        # Record equity after this bar's fills have been applied.
        portfolio.record_current_equity(current_dt)

    total_return, sharpe, max_dd = portfolio.calculate_performance_metrics()
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True).sort_index()
    matrix = strategy._build_transition_matrix(strategy.state_history)
    benchmark = buy_and_hold_metrics(df['Close'])

    return df, buy_signals, sell_signals, total_return, sharpe, max_dd, matrix, benchmark


# ================= UI STREAMLIT =================

st.set_page_config(page_title="Quant Backtester", layout="wide", page_icon="📈")
st.title("📈 Pure Markov Chain Backtester")

# --- Sidebar: Data Selection & Strategy Explanation ---
st.sidebar.header("🔍 Data Selection")
symbol = st.sidebar.text_input("Stock Symbol (e.g., AAPL, INTC, TSLA)", "AAPL").upper()
col1, col2 = st.sidebar.columns(2)
start_date = col1.date_input("Start Date", datetime.date(2020, 1, 1))
end_date = col2.date_input("End Date", datetime.date.today())

st.sidebar.markdown("---")
st.sidebar.header("📖 Strategy Logic")
st.sidebar.info("""
**This is a pure statistical model devoid of lagging indicators. It operates in two layers:**

1. **Volatility-Adjusted States (Z-Score):** The strategy converts daily returns into a Z-score. An upward anomaly (above threshold) is 'Bullish' (2), a downward anomaly is 'Bearish' (0), and the rest is 'Neutral' (1).
2. **Probabilistic Prediction (Markov Chain):** The algorithm analyzes the recent sequence (Lookback) to generate a dynamic transition matrix. If the statistical probability of transitioning to 'Bullish' tomorrow crosses the confidence threshold, the system buys (provided volume supports it). It exits when the probability leans towards 'Bearish'.
""")

# --- Main Content: Tabs ---
tab1, tab2 = st.tabs(["🎯 Manual Backtest", "🔬 Grid Search Optimizer"])

# ================= Tab 1: Manual Backtest =================
with tab1:
    st.header("⚙️ Manual Parameter Tuning")

    col_p1, col_p2 = st.columns(2)
    lookback = col_p1.slider("Markov Lookback Window", 30, 120, 60, step=10)
    m_thresh = col_p2.slider("Markov Confidence Threshold", 0.20, 0.80, 0.40, step=0.05)
    z_thresh = col_p2.slider("Z-Score Volatility Threshold", 0.1, 1.5, 0.25, step=0.05)

    if st.button("🚀 Run Pure Strategy"):
        with st.spinner(f"Loading data for {symbol}..."):
            csv_filename, run_symbol, source = load_price_data(symbol, start_date, end_date)

        if source == "sample":
            st.warning(
                f"Live data for **{symbol}** is unavailable right now "
                f"(Yahoo Finance may be rate-limiting this host). "
                f"Showing the bundled **{SAMPLE_SYMBOL}** sample dataset instead."
            )

        with st.spinner("Running quantitative simulation..."):
            df, buys, sells, ret, sharpe, max_dd, matrix, bench = run_simulation(
                csv_filename, run_symbol, lookback, m_thresh, z_thresh
            )

            c1, c2, c3 = st.columns(3)
            c1.metric("Sharpe Ratio", f"{sharpe:.2f}",
                      delta=f"{sharpe - bench['sharpe']:.2f} vs Buy & Hold",
                      help="Excess return per unit of risk. The delta compares against simply buying and holding the asset.")
            c2.metric("Total Return (ROI)", f"{ret * 100:.2f}%",
                      delta=f"{(ret - bench['return']) * 100:.2f}% vs Buy & Hold",
                      help="Total return over the selected period, compared against a buy-and-hold benchmark.")
            c3.metric("Max Drawdown", f"{max_dd * 100:.2f}%",
                      delta=f"{(max_dd - bench['max_dd']) * 100:.2f}% vs Buy & Hold", delta_color="inverse",
                      help="Maximum peak-to-trough loss. A smaller drawdown than buy-and-hold means lower risk.")

            st.caption(
                f"📌 Benchmark — Buy & Hold {run_symbol}: "
                f"Return {bench['return'] * 100:.2f}% · "
                f"Sharpe {bench['sharpe']:.2f} · "
                f"Max DD {bench['max_dd'] * 100:.2f}%"
            )

            st.subheader("📊 Equity & Signals Chart")
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(df.index, df['Close'], label='Price', color='gray', alpha=0.5)
            if buys: ax.scatter(df.loc[buys].index, df.loc[buys]['Close'], marker='^', color='green', s=100,
                                label='Buy Signal')
            if sells: ax.scatter(df.loc[sells].index, df.loc[sells]['Close'], marker='v', color='red', s=100,
                                 label='Sell Signal')
            ax.legend()
            ax.grid(alpha=0.3)
            st.pyplot(fig)

            st.subheader("🎲 Transition Percentage Heatmap")
            matrix_df = pd.DataFrame(matrix,
                                     index=['Bearish (0)', 'Neutral (1)', 'Bullish (2)'],
                                     columns=['Bearish (0)', 'Neutral (1)', 'Bullish (2)'])

            styled_matrix = matrix_df.style.background_gradient(cmap='Blues', axis=None).format("{:.1%}")
            st.dataframe(styled_matrix, use_container_width=True)

# ================= Tab 2: Dynamic Optimization =================
with tab2:
    st.header("🔬 Automatic Optimization (Grid Search)")
    st.markdown("Enter comma-separated values to generate a custom grid search.")

    opt_c1, opt_c2 = st.columns(2)
    opt_lookbacks_str = opt_c1.text_input("Lookback Windows", "30, 60, 90")
    opt_m_threshs_str = opt_c2.text_input("Markov Thresholds", "0.4, 0.45, 0.5")
    opt_z_threshs_str = opt_c2.text_input("Z-Score Thresholds", "0.2, 0.25, 0.3")

    if st.button("🔍 Run Optimization Scan"):
        try:
            opt_lookbacks = [int(x.strip()) for x in opt_lookbacks_str.split(',')]
            opt_m_threshs = [float(x.strip()) for x in opt_m_threshs_str.split(',')]
            opt_z_threshs = [float(x.strip()) for x in opt_z_threshs_str.split(',')]
        except ValueError:
            st.error("❌ Input error: Please ensure you only entered comma-separated numbers.")
        else:
            with st.spinner(f"Loading data for {symbol}..."):
                csv_filename, run_symbol, source = load_price_data(symbol, start_date, end_date)

            if source == "sample":
                st.warning(
                    f"Live data for **{symbol}** is unavailable right now "
                    f"(Yahoo Finance may be rate-limiting this host). "
                    f"Optimising on the bundled **{SAMPLE_SYMBOL}** sample dataset instead."
                )

            combinations = list(itertools.product(opt_lookbacks, opt_m_threshs, opt_z_threshs))
            total_runs = len(combinations)
            st.info(f"Starting scan of {total_runs} combinations (Pure Model)...")

            progress_bar = st.progress(0)
            status_text = st.empty()
            results = []

            start_time = time.time()

            for i, (l, m, z) in enumerate(combinations):
                if i == 0:
                    status_text.text(f"Running combination {i + 1}/{total_runs} | Calculating ETA...")
                else:
                    elapsed_time = time.time() - start_time
                    avg_time_per_run = elapsed_time / i
                    remaining_time = avg_time_per_run * (total_runs - i)
                    mins, secs = divmod(int(remaining_time), 60)
                    status_text.text(f"Running combination {i + 1}/{total_runs} | ETA: {mins:02d}:{secs:02d}")

                _, _, _, ret, sharpe, max_dd, _, _ = run_simulation(csv_filename, run_symbol, l, m, z)

                results.append({
                    'Lookback': l,
                    'Markov Thresh': m,
                    'Z-Score Thresh': z,
                    'Return': ret,
                    'Sharpe': sharpe,
                    'Max DD': max_dd
                })
                progress_bar.progress((i + 1) / total_runs)

            status_text.empty()
            progress_bar.empty()
            st.success("Statistical optimization completed successfully!")

            res_df = pd.DataFrame(results).sort_values(by='Sharpe', ascending=False)
            res_df['Return'] = res_df['Return'].apply(lambda x: f"{x * 100:.2f}%")
            res_df['Max DD'] = res_df['Max DD'].apply(lambda x: f"{x * 100:.2f}%")
            res_df['Sharpe'] = res_df['Sharpe'].apply(lambda x: f"{x:.3f}")

            st.subheader("🏆 Top 10 Configurations (Sorted by Sharpe)")
            st.dataframe(res_df.head(10), use_container_width=True)