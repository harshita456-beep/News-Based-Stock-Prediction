"""
Aligns the raw news headlines with the trading calendar: keeps only news
rows whose date matches a date the market was actually open, and reformats
dates from YYYY-MM-DD to DD-MM-YYYY so they can be matched against the
stock price file. Saves the result to news_data1.csv.
"""

import pandas as pd

from config import NEWS_RAW_CSV, STOCK_PRICE_CSV, NEWS_CLEANED_CSV


def clean_news_data(news_path=NEWS_RAW_CSV, stock_path=STOCK_PRICE_CSV, out_path=NEWS_CLEANED_CSV):
    news_df = pd.read_csv(news_path)
    stock_df = pd.read_csv(stock_path)

    # Keep only the date portion (strip any time component)
    stock_df["Date"] = stock_df["Date"].str[:10]

    # Reformat news dates from YYYY-MM-DD to DD-MM-YYYY
    def to_ddmmyyyy(date_str):
        year, month, day = date_str.split("-")
        return f"{day}-{month}-{year}"

    news_df["Date"] = news_df["Date"].apply(to_ddmmyyyy)

    trading_dates = set(stock_df["Date"].tolist())
    news_df = news_df[news_df["Date"].isin(trading_dates)]

    news_df.to_csv(out_path, index=False)
    print(f"Kept {len(news_df)} news rows matching trading days -> {out_path}")
    return news_df


if __name__ == "__main__":
    clean_news_data()
