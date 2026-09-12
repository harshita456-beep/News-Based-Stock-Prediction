"""
Downloads historical daily OHLCV stock/index price data from Yahoo Finance
and saves it to stock_price1.csv.
"""

import pandas as pd
import yfinance as yf

from config import TICKER, START_DATE, END_DATE, STOCK_PRICE_CSV


def download_stock_data(ticker, start, end, out_path=STOCK_PRICE_CSV):
 
    stock_data = yf.download(ticker, start=start, end=end, auto_adjust=False)
    df = pd.DataFrame(stock_data)
    df.to_csv(out_path)
    print(f"Downloaded {len(df)} rows for {ticker} ({start} to {end}) -> {out_path}")
    return df


if __name__ == "__main__":
    download_stock_data(TICKER, START_DATE, END_DATE)
