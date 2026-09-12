"""
Trains an LSTM model that combines historical closing prices with the
FinBERT news-sentiment score for each day to predict the next day's
closing price, then evaluates it on a held-out test set.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

from config import (
    STOCK_PRICE_CSV,
    SENTIMENT_CSV,
    TRAIN_SPLIT,
    SEQUENCE_LENGTH,
    BERT_LSTM_MODEL_PATH,
)


# Hyperparameters
split = TRAIN_SPLIT
sequence_length = SEQUENCE_LENGTH
epochs = 100
learning_rate = 0.02
batch_size = 32


# ---------------------------------------------------------
# Load and prepare data
# ---------------------------------------------------------

def load_data(
    stock_path=STOCK_PRICE_CSV,
    sentiment_path=SENTIMENT_CSV
):

    stock_data = pd.read_csv(stock_path)
    news_data = pd.read_csv(sentiment_path)

    stock_column = ["Close"]
    news_column = ["FinBERT score"]

    len_stock_data = stock_data.shape[0]
    train_examples = int(len_stock_data * split)

    # Split stock prices
    train = stock_data.get(stock_column).values[:train_examples]
    test = stock_data.get(stock_column).values[train_examples:]

    # Split sentiment
    train_sentiment = news_data.get(news_column).values[:train_examples]
    test_sentiment = news_data.get(news_column).values[train_examples:]

    # Scale stock prices
    # Fit ONLY on training data
    scaler = MinMaxScaler()

    train = scaler.fit_transform(train)
    test = scaler.transform(test)

    len_train = train.shape[0]
    len_test = test.shape[0]

    # -----------------------------------------------------
    # Create price sequences
    # -----------------------------------------------------

    X_train = [
        train[i: i + sequence_length]
        for i in range(len_train - sequence_length)
    ]

    y_train = np.array(
        train[sequence_length:]
    ).astype(np.float32)

    X_test = [
        test[i: i + sequence_length]
        for i in range(len_test - sequence_length)
    ]

    y_test = np.array(
        test[sequence_length:]
    ).astype(np.float32)

    # -----------------------------------------------------
    # Add FinBERT sentiment as an extra timestep
    # -----------------------------------------------------

    for i in range(len(X_train)):

        X_train[i] = X_train[i].tolist()

        X_train[i].append(
            train_sentiment[sequence_length + i].tolist()
        )

    X_train = np.array(
        X_train
    ).astype(np.float32)

    for i in range(len(X_test)):

        X_test[i] = X_test[i].tolist()

        X_test[i].append(
            test_sentiment[sequence_length + i].tolist()
        )

    X_test = np.array(
        X_test
    ).astype(np.float32)

    return (
        X_train,
        y_train,
        X_test,
        y_test,
        scaler
    )


# ---------------------------------------------------------
# LSTM Model
# ---------------------------------------------------------

class BertLSTMModel(nn.Module):

    def __init__(self):

        super().__init__()

        self.lstm1 = nn.LSTM(
            input_size=1,
            hidden_size=70,
            batch_first=True
        )

        self.lstm2 = nn.LSTM(
            input_size=70,
            hidden_size=30,
            batch_first=True
        )

        self.lstm3 = nn.LSTM(
            input_size=30,
            hidden_size=10,
            batch_first=True
        )

        self.fc = nn.Linear(
            10,
            1
        )

    def forward(self, x):

        x, _ = self.lstm1(x)

        x, _ = self.lstm2(x)

        x, _ = self.lstm3(x)

        # Equivalent to return_sequences=False
        x = x[:, -1, :]

        x = self.fc(x)

        return x


# ---------------------------------------------------------
# Create and train model
# ---------------------------------------------------------

def model_create(X_train, y_train):

    # Reproducibility
    torch.manual_seed(1234)
    np.random.seed(1234)

    model = BertLSTMModel()

    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate
    )

    # Convert NumPy arrays to tensors
    X_train_tensor = torch.tensor(
        X_train,
        dtype=torch.float32
    )

    y_train_tensor = torch.tensor(
        y_train,
        dtype=torch.float32
    )

    dataset = TensorDataset(
        X_train_tensor,
        y_train_tensor
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True
    )

    # -----------------------------------------------------
    # Training loop
    # -----------------------------------------------------

    model.train()

    for epoch in range(epochs):

        total_loss = 0

        for X_batch, y_batch in dataloader:

            optimizer.zero_grad()

            predictions = model(X_batch)

            loss = criterion(
                predictions,
                y_batch
            )

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:

            avg_loss = total_loss / len(dataloader)

            print(
                f"Epoch [{epoch + 1}/{epochs}] "
                f"Loss: {avg_loss:.6f}"
            )

    return model


# ---------------------------------------------------------
# Prediction
# ---------------------------------------------------------

def predict(model, X_test, scaler):

    model.eval()

    X_test_tensor = torch.tensor(
        X_test,
        dtype=torch.float32
    )

    with torch.no_grad():

        predictions = model(
            X_test_tensor
        ).numpy()

    # Convert predictions back to original price scale
    predictions = scaler.inverse_transform(
        predictions.reshape(-1, 1)
    ).reshape(-1, 1)

    return predictions


# ---------------------------------------------------------
# Evaluation
# ---------------------------------------------------------

def evaluate(predictions, y_test):

    mae = mean_absolute_error(
        predictions,
        y_test
    )

    mape = mean_absolute_percentage_error(
        predictions,
        y_test
    )

    return mae, mape, (1 - mape)


# ---------------------------------------------------------
# Run model
# ---------------------------------------------------------

def run_model(n=1, save_model=True):

    (
        X_train,
        y_train,
        X_test,
        y_test,
        scaler
    ) = load_data()

    # Convert actual test prices back to original scale
    y_test = scaler.inverse_transform(y_test)

    total_mae = 0
    total_mape = 0
    total_acc = 0

    model = None

    for _ in range(n):

        model = model_create(
            X_train,
            y_train
        )

        predictions = predict(
            model,
            X_test,
            scaler
        )

        mae, mape, acc = evaluate(
            predictions,
            y_test
        )

        total_mae += mae
        total_mape += mape
        total_acc += acc

    # -----------------------------------------------------
    # Save PyTorch model
    # -----------------------------------------------------

    if save_model and model is not None:

        torch.save(
            model.state_dict(),
            BERT_LSTM_MODEL_PATH
        )

        print(
            f"Saved model -> {BERT_LSTM_MODEL_PATH}"
        )

    return (
        total_mae / n,
        total_mape / n,
        total_acc / n
    )


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

if __name__ == "__main__":

    mae, mape, acc = run_model(1)

    print(
        f"Mean Absolute Error = {mae}"
    )

    print(
        f"Mean Absolute Percentage Error = {mape}%"
    )

    print(
        f"Accuracy = {acc}"
    )