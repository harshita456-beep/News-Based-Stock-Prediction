"""
Train a plain LSTM model
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

from config import STOCK_PRICE_CSV, TRAIN_SPLIT, SEQUENCE_LENGTH, LSTM_MODEL_PATH


# Hyperparameters
split = TRAIN_SPLIT
sequence_length = SEQUENCE_LENGTH
epochs = 100
learning_rate = 0.02
batch_size = 32


def load_data(path=STOCK_PRICE_CSV):
    stock_data = pd.read_csv(path)
    column = ["Close"]

    len_stock_data = stock_data.shape[0]
    train_examples = int(len_stock_data * split)

    train = stock_data.get(column).values[:train_examples]
    test = stock_data.get(column).values[train_examples:]

    scaler = MinMaxScaler()

    train = scaler.fit_transform(train)
    test = scaler.transform(test)

    len_train = train.shape[0]
    len_test = test.shape[0]

    # Create training sequences
    X_train = [
        train[i: i + sequence_length]
        for i in range(len_train - sequence_length)
    ]

    X_train = np.array(X_train).astype(np.float32)

    y_train = np.array(
        train[sequence_length:]
    ).astype(np.float32)

    # Create testing sequences
    X_test = [
        test[i: i + sequence_length]
        for i in range(len_test - sequence_length)
    ]

    X_test = np.array(X_test).astype(np.float32)

    y_test = np.array(
        test[sequence_length:]
    ).astype(np.float32)

    return X_train, y_train, X_test, y_test, scaler


class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.lstm1 = nn.LSTM(
            input_size=1,
            hidden_size=50,
            batch_first=True
        )
        self.dropout1 = nn.Dropout(0.15)
        self.lstm2 = nn.LSTM(
            input_size=50,
            hidden_size=30,
            batch_first=True
        )

        self.dropout2 = nn.Dropout(0.05)
        self.lstm3 = nn.LSTM(
            input_size=30,
            hidden_size=20,
            batch_first=True
        )

        self.dropout3 = nn.Dropout(0.01)
        self.fc = nn.Linear(20, 1)


    def forward(self, x):

        x, _ = self.lstm1(x)
        x = self.dropout1(x)

        x, _ = self.lstm2(x)
        x = self.dropout2(x)

        x, _ = self.lstm3(x)

        # Take output from final time step
        x = x[:, -1, :]

        x = self.dropout3(x)

        x = self.fc(x)

        return x



def model_create(X_train, y_train):

    torch.manual_seed(1234)
    np.random.seed(1234)
    model = LSTMModel()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate
    )

    # Convert NumPy arrays to PyTorch tensors
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

    # Training
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

    # Convert predictions back to original stock-price scale
    predictions = scaler.inverse_transform(
        predictions.reshape(-1, 1)
    ).reshape(-1, 1)

    return predictions


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



def run_model(n=1, save_model=True):

    X_train, y_train, X_test, y_test, scaler = load_data()

    # Convert actual test values back to original scale
    y_test = scaler.inverse_transform(y_test)

    total_mae = 0
    total_mape = 0
    total_acc = 0

    model = None
    predictions = None

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

    # Save PyTorch model
    if save_model and model is not None:

        torch.save(
            model.state_dict(),
            LSTM_MODEL_PATH
        )
        print(
            f"Saved model -> {LSTM_MODEL_PATH}"
        )

    return (
        total_mae / n,
        total_mape / n,
        total_acc / n,
        predictions.tolist()
    )




if __name__ == "__main__":

    mae, mape, acc, preds = run_model(1)

    print(
        f"Mean Absolute Error = {mae}"
    )

    print(
        f"Mean Absolute Percentage Error = {mape}%"
    )

    print(
        f"Accuracy = {acc}"
    )