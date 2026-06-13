import yfinance as yf

OUTPUT_FILE = "sample_data.csv"


def download_sample_data(symbol="AAPL", start="2022-01-01", end="2024-01-01"):
    """Download daily OHLCV data from Yahoo Finance and save it as the bundled sample dataset."""
    print(f"Downloading {symbol} data from Yahoo Finance ({start} to {end})...")

    ticker = yf.Ticker(symbol)
    df = ticker.history(start=start, end=end)
    df.to_csv(OUTPUT_FILE)

    print(f"Data saved successfully to {OUTPUT_FILE}")


if __name__ == "__main__":
    download_sample_data()
