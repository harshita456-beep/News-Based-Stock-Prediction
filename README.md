# FinBERT-LSTM Stock Market Prediction

A stock market prediction project that combines historical price data with financial news sentiment.

The project uses **FinBERT** to analyse the sentiment of financial news and **LSTM/MLP models** to predict the next day's closing price of the NASDAQ-100.

## Overview

The idea is to see whether adding information from financial news can improve stock price prediction compared to using price history alone.

The project follows this pipeline:

**News → Sentiment Analysis → Combine with Stock Data → Model Training → Price Prediction**

Three models are used:

- **MLP** – baseline model using previous prices
- **LSTM** – uses historical price sequences
- **FinBERT-LSTM** – combines historical prices with daily news sentiment

## Features

- Collects financial news from the New York Times API
- Downloads stock data using Yahoo Finance
- Cleans and aligns news with trading dates
- Uses FinBERT for financial sentiment analysis
- Trains MLP and LSTM models using PyTorch
- Compares the models on a held-out test set
- Can generate predictions for upcoming days

## Project Structure

```text
├── main.py
├── config.py
├── analysis.py
│
├── 1_news_collection.py
├── 2_stock_data_collection.py
├── 3_news_data_cleaning.py
├── 4_news_sentiment_analysis.py
├── 5_MLP_model.py
├── 6_LSTM_model.py
├── 7_lstm_model_bert.py
│
├── Lstm + Finbert.ipynb
│
├── news1.csv
├── news_data1.csv
├── sentiment1.csv
├── stock_price1.csv
│
├── requirements.txt
├── .env.example
├── .gitignore
