import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout
from tensorflow.keras.models import Sequential

# Use Optimized GPU is possible
GPUinUse = False
physical_devices = tf.config.list_physical_devices("GPU")
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print(f"Using GPU: {physical_devices[0].name}")
    GPUinUse = True
else:
    print("No GPU available, using CPU.")


# Fetch historical stock data
def fetch_data(ticker):
    data = yf.download(ticker, start="2010-01-01", end="2024-01-01")
    data["SMA_50"] = data["Close"].rolling(window=50).mean()
    data["SMA_200"] = data["Close"].rolling(window=200).mean()
    return data.dropna()


# Fetch multiple stocks data
def fetch_multiple_stocks_data(tickers, start_date, end_date):
    all_data = []
    for ticker in tickers:
        data = yf.download(ticker, start=start_date, end=end_date)
        data["SMA_50"] = data["Close"].rolling(window=50).mean()
        data["SMA_200"] = data["Close"].rolling(window=200).mean()
        data = data.dropna()
        data["Ticker"] = ticker
        all_data.append(data)
    combined_data = pd.concat(all_data)
    return combined_data


# Prepare the data for LSTM
def prepare_data(data, look_back=60):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data[["Close", "Volume", "SMA_50", "SMA_200"]])
    X, y = [], []
    for i in range(len(scaled_data) - look_back - 1):
        X.append(scaled_data[i : (i + look_back)])
        y.append(scaled_data[i + look_back, 0])
    return np.array(X), np.array(y), scaler


# Create LSTM model
def create_model(input_shape):
    model = Sequential()
    model.add(
        Bidirectional(LSTM(units=64, return_sequences=True), input_shape=input_shape)
    )
    model.add(Dropout(0.3))
    model.add(Bidirectional(LSTM(units=64, return_sequences=True)))
    model.add(Dropout(0.3))
    model.add(Bidirectional(LSTM(units=64, return_sequences=False)))
    model.add(Dropout(0.3))
    model.add(Dense(units=1, activation="linear"))
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model


def train_model(ticker):
    data = fetch_data(ticker)
    X, y, scaler = prepare_data(data)
    train_size = int(len(X) * 0.8)
    X_train, y_train = X[:train_size], y[:train_size]
    X_test, y_test = X[train_size:], y[train_size:]

    model = create_model((X_train.shape[1], X_train.shape[2]))
    if GPUinUse:
        with tf.device("/GPU:0"):  # Ensure the model is trained on the GPU
            history = model.fit(
                X_train,
                y_train,
                epochs=50,
                batch_size=32,
                validation_data=(X_test, y_test),
            )
    else:
        # Kepps track of histoy in order to debug.
        history = model.fit(
            X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test)
        )

    # Plot training & validation loss values Used for debugging
    # plt.figure(figsize=(10, 6))
    # plt.plot(history.history["loss"], label="Train Loss")
    # plt.plot(history.history["val_loss"], label="Validation Loss")
    # plt.title("Model loss")
    # plt.ylabel("Loss")
    # plt.xlabel("Epoch")
    # plt.legend(loc="upper right")
    # plt.show()
    modelName = f"SavedModels/stock_price_{ticker}_model.h5"
    model.save(modelName)
    return model, scaler


# Train a general model for multiple stocks
def train_general_model(tickers, start_date="2010-01-01", end_date="2024-01-01"):
    data = fetch_multiple_stocks_data(tickers, start_date, end_date)
    X, y, scaler = prepare_data(data)

    # Shuffle the data to mix different stocks
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]

    train_size = int(len(X) * 0.8)
    X_train, y_train = X[:train_size], y[:train_size]
    X_test, y_test = X[train_size:], y[train_size:]

    model = create_model((X_train.shape[1], X_train.shape[2]))

    # Callbacks
    early_stopping = EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True
    )
    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6
    )

    if GPUinUse:
        with tf.device("/GPU:0"):  # Ensure the model is trained on the GPU
            history = model.fit(
                X_train,
                y_train,
                epochs=100,
                batch_size=32,
                validation_data=(X_test, y_test),
                callbacks=[early_stopping, reduce_lr],
            )
    else:
        # Kepps track of histoy in order to debug.
        history = model.fit(
            X_train,
            y_train,
            epochs=100,
            batch_size=32,
            validation_data=(X_test, y_test),
            callbacks=[early_stopping, reduce_lr],
        )

    # Plot training & validation loss values for debugging
    plt.figure(figsize=(10, 6))
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title("Model loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend(loc="upper right")
    plt.show()

    model.save("savedModels/general_stock_price_model.h5")
    return model, scaler


if __name__ == "__main__":
    # Option to train for an individual stock although will resort in over fitting in the one stock
    # train_model("AAPL")

    # Train a larger general model tohandle most cases
    tickers = [
        "AAPL",
        "GOOGL",
        "MSFT",
        "AMZN",
        "TSLA",
        "FB",
        "NFLX",
        "NVDA",
        "BABA",
        "INTC",
        "AMD",
        "QCOM",
        "CSCO",
        "ORCL",
        "IBM",
        "SAP",
        "ADBE",
        "CRM",
        "PYPL",
        "AVGO",
    ]
    train_general_model(tickers)
    print("finished training")
