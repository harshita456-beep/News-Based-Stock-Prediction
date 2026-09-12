"""
Train a Multi-Layer Perceptron baseline model for stock price prediction
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

from config import STOCK_PRICE_CSV, TRAIN_SPLIT, SEQUENCE_LENGTH


# Hyperparameters
split = TRAIN_SPLIT
sequence_length = SEQUENCE_LENGTH
epochs = 100
learning_rate = 0.01
batch_size = 32


# Set random seed
torch.manual_seed(1234)
np.random.seed(1234)


def load_data(path=STOCK_PRICE_CSV):

    stock_data = pd.read_csv(path)

    column = ["Close"]

    len_stock_data = stock_data.shape[0]
    train_examples = int(len_stock_data * split)

    train = stock_data[column].values[:train_examples]
    test = stock_data[column].values[train_examples:]

    # Fit scaler ONLY on training data
    scaler = MinMaxScaler()

    train = scaler.fit_transform(train)
    test = scaler.transform(test)

    len_train = train.shape[0]
    len_test = test.shape[0]

    # Create training sequences
    X_train = [
        train[i:i + sequence_length]
        for i in range(len_train - sequence_length)
    ]

    X_train = np.array(X_train).astype(np.float32)

    y_train = np.array(
        train[sequence_length:]
    ).astype(np.float32)

    # Create test sequences
    X_test = [
        test[i:i + sequence_length]
        for i in range(len_test - sequence_length)
    ]

    X_test = np.array(X_test).astype(np.float32)

    y_test = np.array(
        test[sequence_length:]
    ).astype(np.float32)

    return X_train, y_train, X_test, y_test, scaler


class MLPModel(nn.Module):

    def __init__(self, input_size):

        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_size, 50),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(50, 30),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(30, 20),
            nn.ReLU(),
            nn.Dropout(0.01),
            nn.Linear(20, 1)
        )

    def forward(self, x):
        return self.network(x)


def model_create(X_train, y_train):

    X_train_flat = X_train.reshape(
        X_train.shape[0],
        X_train.shape[1]
    )

    X_tensor = torch.tensor(
        X_train_flat,
        dtype=torch.float32
    )

    y_tensor = torch.tensor(
        y_train,
        dtype=torch.float32
    )

    dataset = TensorDataset(X_tensor, y_tensor)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True
    )

    model = MLPModel(
        input_size=X_train_flat.shape[1]
    )

    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate
    )

    model.train()

    for epoch in range(epochs):

        total_loss = 0.0
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
            average_loss = total_loss / len(dataloader)
            print(
                f"Epoch [{epoch + 1}/{epochs}], "
                f"Loss: {average_loss:.6f}"
            )

    return model


def predict(model, X_test, scaler):

    X_test_flat = X_test.reshape(
        X_test.shape[0],
        X_test.shape[1]
    )

    X_tensor = torch.tensor(
        X_test_flat,
        dtype=torch.float32
    )

    model.eval()

    with torch.no_grad():
        predictions = model(X_tensor)

    predictions = predictions.numpy()

    # Convert scaled predictions back to original prices
    predictions = scaler.inverse_transform(
        predictions
    )
    return predictions


def evaluate(predictions, y_test):

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    mape = mean_absolute_percentage_error(
        y_test,
        predictions
    )

    accuracy = 1 - mape

    return mae, mape, accuracy


def run_model(n=1):

    X_train, y_train, X_test, y_test, scaler = load_data()

    # Train multiple times if n > 1
    total_mae = 0
    total_mape = 0
    total_acc = 0

    predictions = None

    # Convert y_test back to original price scale
    y_test_original = scaler.inverse_transform(
        y_test
    )

    for i in range(n):

        print(f"\nTraining MLP model {i + 1}/{n}")

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
            y_test_original
        )

        total_mae += mae
        total_mape += mape
        total_acc += acc

    return (
        total_mae / n,
        total_mape / n,
        total_acc / n,
        predictions.tolist()
    )


if __name__ == "__main__":

    mae, mape, acc, preds = run_model(1)

    print(f"\nMean Absolute Error = {mae}")
    print(f"Mean Absolute Percentage Error = {mape}")
    print(f"Accuracy = {acc}")