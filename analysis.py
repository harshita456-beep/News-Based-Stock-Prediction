
import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yfinance as yf

from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error
)
from sklearn.preprocessing import MinMaxScaler

from config import (
    TICKER,
    STOCK_PRICE_CSV,
    SENTIMENT_CSV,
    LSTM_MODEL_PATH,
    BERT_LSTM_MODEL_PATH,
    TRAIN_SPLIT,
    SEQUENCE_LENGTH,
)

from importlib import import_module

LSTMModel = import_module("6_LSTM_model").LSTMModel
BertLSTMModel = import_module("7_lstm_model_bert").BertLSTMModel


_finbert_pipeline = None


 
def get_finbert_sentiment(headlines):
  
    global _finbert_pipeline

    headlines = [
        h for h in headlines
        if h and str(h) not in {"No News", "0", "nan"}
    ]

    if not headlines:
        return 0.0

    if _finbert_pipeline is None:

        from transformers import (
            AutoTokenizer,
            AutoModelForSequenceClassification,
            pipeline
        )

        tokenizer = AutoTokenizer.from_pretrained(
            "ProsusAI/finbert"
        )

        model = AutoModelForSequenceClassification.from_pretrained(
            "ProsusAI/finbert"
        )

        # PyTorch
        _finbert_pipeline = pipeline(
            "sentiment-analysis",
            model=model,
            tokenizer=tokenizer,
            framework="pt"
        )

    results = _finbert_pipeline(
        headlines,
        truncation=True
    )

    scores = []

    for r in results:

        if r["label"] == "positive":
            scores.append(r["score"])

        elif r["label"] == "negative":
            scores.append(-r["score"])

        else:
            scores.append(0.0)

    return sum(scores) / len(scores)


# Today's News
def get_today_news():
   
    from importlib import import_module

    news_collection = import_module(
        "1_news_collection"
    )

    today = datetime.datetime.now()

    return news_collection.get_news(
        today.year,
        today.month,
        today.day
    )


#stock data
def get_recent_stock_data(
    ticker=TICKER,
    days=30
): 

    end_date = datetime.datetime.now()

    start_date = (
        end_date -
        datetime.timedelta(days=days)
    )

    return yf.download(
        ticker,
        start=start_date.strftime("%Y-%m-%d"),
        end=end_date.strftime("%Y-%m-%d"),
        auto_adjust=False,
    )


#load models
def load_pytorch_model(
    model_path,
    model_type
):
    

    if model_type == "lstm":
        model = LSTMModel()

    elif model_type == "bert_lstm":
        model = BertLSTMModel()

    else:

        raise ValueError(
            f"Unknown model type: {model_type}"
        )

    # Load saved weights
    state_dict = torch.load(
        model_path,
        map_location=torch.device("cpu"),
        weights_only=True
    )

    model.load_state_dict(state_dict)

    model.eval()

    return model


#predict future prices

def predict_future_prices(
    model_path,
    stock_data,
    sentiment_score=None,
    sequence_length=SEQUENCE_LENGTH,
    days_ahead=5
):
    """

    If sentiment_score is given, it is appended as an extra timestep
    for the FinBERT-LSTM model.
    If sentiment_score is None, the plain LSTM model is used.
    """

    try:


#which model to use?

        if sentiment_score is None:

            model = load_pytorch_model(
                model_path,
                "lstm"
            )

        else:

            model = load_pytorch_model(
                model_path,
                "bert_lstm"
            )

        
        close_prices = stock_data["Close"].values

        close_prices = np.asarray(
            close_prices
        ).reshape(-1, 1)

        scaler = MinMaxScaler()

        scaled_data = scaler.fit_transform(
            close_prices
        )

        # Initial sequence
        window = scaled_data[
            -sequence_length:
        ].tolist()

        predictions = []

          # Predict one day at a time
        for _ in range(days_ahead):

            if sentiment_score is not None:

                # Same structure as training:
                # sequence_length prices + sentiment
                model_input = np.array(
                    window + [[sentiment_score]],
                    dtype=np.float32
                ).reshape(
                    1,
                    sequence_length + 1,
                    1
                )

            else:

                model_input = np.array(
                    window,
                    dtype=np.float32
                ).reshape(
                    1,
                    sequence_length,
                    1
                )

            # Convert to PyTorch tensor
            model_input = torch.tensor(
                model_input,
                dtype=torch.float32
            )

            # Prediction
            model.eval()

            with torch.no_grad():

                pred = model(
                    model_input
                ).numpy()

            predicted_value = float(
                pred[0][0]
            )

            predictions.append(
                predicted_value
            )

            # Add prediction to rolling window
            window.append(
                [predicted_value]
            )

            window = window[1:]

       # Convert predictions back to original scale
  
        predictions = scaler.inverse_transform(
            np.array(predictions).reshape(-1, 1)
        ).flatten()

        return predictions

    except Exception as e:

        print(
            f"Error in prediction: {e}"
        )

        return None




def analyze_market_future(
    days_ahead=5,
    ticker=TICKER,
    plot=True
):
   

    stock_data = get_recent_stock_data(
        ticker,
        days=30
    )

    news_list = get_today_news()

    sentiment_score = get_finbert_sentiment(
        news_list
    )

    print(
        f"Today's aggregate FinBERT sentiment score: "
        f"{sentiment_score:.4f}"
    )

    # Plain LSTM
    lstm_preds = predict_future_prices(
        LSTM_MODEL_PATH,
        stock_data,
        None,
        days_ahead=days_ahead
    )

    # FinBERT + LSTM
    bert_preds = predict_future_prices(
        BERT_LSTM_MODEL_PATH,
        stock_data,
        sentiment_score,
        days_ahead=days_ahead
    )

    
    # Plot predictions
 

    if plot:

        future_dates = [
            datetime.datetime.now()
            + datetime.timedelta(days=i)
            for i in range(1, days_ahead + 1)
        ]

        plt.figure(
            figsize=(12, 6)
        )

        if lstm_preds is not None:

            plt.plot(
                future_dates,
                lstm_preds,
                "go-",
                label="LSTM Prediction"
            )

        if bert_preds is not None:

            plt.plot(
                future_dates,
                bert_preds,
                "ro-",
                label="FinBERT-LSTM Prediction"
            )

        plt.legend()

        plt.title(
            f"Predicted {ticker} Prices "
            f"for Next {days_ahead} Days"
        )

        plt.xlabel("Date")

        plt.ylabel("Price (USD)")

        plt.grid(True)

        plt.tight_layout()

        plt.show()

    return lstm_preds, bert_preds





def compare_models_on_test_set(
    stock_path=STOCK_PRICE_CSV,
    sentiment_path=SENTIMENT_CSV,
    split=TRAIN_SPLIT,
    sequence_length=SEQUENCE_LENGTH,
    plot=True,
):
    """
    Load the saved LSTM and FinBERT-LSTM PyTorch models, run them on
    the held-out test split, print MAE / MAPE / accuracy for each
    """

    stock_data = pd.read_csv(
        stock_path
    )

    news_data = pd.read_csv(
        sentiment_path
    )

    stock_price = stock_data[
        "Close"
    ].values

    split_idx = int(
        len(stock_price) * split
    )

    train_data = stock_price[
        :split_idx
    ]

    test_data = stock_price[
        split_idx:
    ]

   
    scaler = MinMaxScaler()

    scaler.fit(
        train_data.reshape(-1, 1)
    )

    scaled_test_data = scaler.transform(
        test_data.reshape(-1, 1)
    )

    #only lstm
    X_test_lstm = np.array(
        [
            scaled_test_data[
                i: i + sequence_length
            ]
            for i in range(
                len(scaled_test_data)
                - sequence_length
            )
        ],
        dtype=np.float32
    )

    #finbert + lstm
    test_sentiment = news_data[
        "FinBERT score"
    ].values[
        split_idx:
    ].reshape(-1, 1)

    X_test_bert = []

    for i in range(
        len(scaled_test_data)
        - sequence_length
    ):

        seq = scaled_test_data[
            i: i + sequence_length
        ].reshape(-1).tolist()

        # Append same-day sentiment
        seq.append(
            test_sentiment[
                sequence_length + i
            ][0]
        )

        X_test_bert.append(
            seq
        )

    X_test_bert = np.array(
        X_test_bert,
        dtype=np.float32
    ).reshape(
        len(X_test_bert),
        len(X_test_bert[0]),
        1
    )


    lstm_model = load_pytorch_model(
        LSTM_MODEL_PATH,
        "lstm"
    )

    bert_model = load_pytorch_model(
        BERT_LSTM_MODEL_PATH,
        "bert_lstm"
    )

    #lstm predictions
    X_test_lstm_tensor = torch.tensor(
        X_test_lstm,
        dtype=torch.float32
    )

    with torch.no_grad():

        lstm_predictions = (
            lstm_model(
                X_test_lstm_tensor
            )
            .numpy()
        )

    lstm_predictions = scaler.inverse_transform(
        lstm_predictions
    )
#finbert + lstm predictions
    X_test_bert_tensor = torch.tensor(
        X_test_bert,
        dtype=torch.float32
    )

    with torch.no_grad():

        bert_predictions = (
            bert_model(
                X_test_bert_tensor
            )
            .numpy()
        )

    bert_predictions = scaler.inverse_transform(
        bert_predictions
    )

    
    actual_values = test_data[
        sequence_length:
    ]

    #EVALUATION 
    results = {}

    for name, preds in [
        ("LSTM", lstm_predictions),
        ("FinBERT-LSTM", bert_predictions)
    ]:

        mae = mean_absolute_error(
            actual_values,
            preds
        )

        mape = mean_absolute_percentage_error(
            actual_values,
            preds
        )

        acc = 1 - mape

        results[name] = {
            "MAE": mae,
            "MAPE": mape,
            "Accuracy": acc
        }

        print(
            f"\n{name} Model Metrics:"
        )

        print(
            f"  MAE:      {mae:.2f}"
        )

        print(
            f"  MAPE:     {mape:.4f}"
        )

        print(
            f"  Accuracy: {acc:.4f}"
        )




  #plotting actual vs predicted prices
    if plot:

        plt.figure(
            figsize=(12, 6)
        )

        plt.plot(
            actual_values,
            linewidth=2.6,
            color="black",
            label="Actual price"
        )

        plt.plot(
            lstm_predictions,
            color="orange",
            label="LSTM model"
        )

        plt.plot(
            bert_predictions,
            color="red",
            label="FinBERT-LSTM model"
        )

        plt.xlabel(
            "Timestep",
            fontsize=10,
            labelpad=10
        )

        plt.ylabel(
            "Closing price (USD)",
            fontsize=10,
            labelpad=10
        )

        plt.title(
            f"{TICKER} Closing Price: "
            f"Actual vs Predicted",
            fontsize=16,
            pad=15
        )

        plt.legend(
            loc="upper right"
        )

        plt.grid(
            alpha=0.3
        )

        plt.tight_layout()

        plt.show()

    return results


if __name__ == "__main__":

    compare_models_on_test_set()
