
import os
from dotenv import load_dotenv

load_dotenv()

# New York Times Article Search API key.
NYT_API_KEY = os.environ.get("NYT_API_KEY", "")

# Default ticker / date range used by the data-collection scripts.
TICKER = os.environ.get("STOCK_TICKER", "NDX")
START_DATE = os.environ.get("START_DATE", "2025-08-27")
END_DATE = os.environ.get("END_DATE", "2026-08-27")

# Filenames 
NEWS_RAW_CSV = "news1.csv"
NEWS_CLEANED_CSV = "news_data1.csv"
SENTIMENT_CSV = "sentiment1.csv"
STOCK_PRICE_CSV = "stock_price1.csv"

LSTM_MODEL_PATH = "lstm_model.h5"
BERT_LSTM_MODEL_PATH = "bertmodel.keras"

# Shared modeling hyperparameters.
TRAIN_SPLIT = 0.85
SEQUENCE_LENGTH = 10
