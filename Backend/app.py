import numpy as np
import pandas as pd
import tensorflow as tf
import yfinance as yf
from flask import Flask, jsonify, request
from flask_cors import CORS
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

app = Flask(__name__)
CORS(app)

# Ensure TensorFlow uses the GPU if available
physical_devices = tf.config.list_physical_devices("GPU")
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print(f"Using GPU: {physical_devices[0].name}")
else:
    print("No GPU available, using CPU.")

# Load pre-trained models
models = {
    "AAPL": load_model("SavedModels/stock_price_AAPL_model.h5"),
    "general": load_model("SavedModels/general_stock_price_model.h5"),
}

# Placeholder for scalers (to be loaded or created)
scalers = {"AAPL": MinMaxScaler(), "general": MinMaxScaler()}


# Function to fetch stock data
def fetch_data(ticker):
    data = yf.download(ticker, start="2010-01-01", end="2024-01-01")
    data["SMA_50"] = data["Close"].rolling(window=50).mean()
    data["SMA_200"] = data["Close"].rolling(window=200).mean()
    return data.dropna()


# Prepare the data for LSTM
def prepare_data(data, look_back=60):
    scaler = scalers["general"]
    scaled_data = scaler.fit_transform(data[["Close", "Volume", "SMA_50", "SMA_200"]])
    X = []
    for i in range(len(scaled_data) - look_back):
        X.append(scaled_data[i : (i + look_back)])
    return np.array(X)


@app.route("/get_stock_data", methods=["GET"])
def get_stock_data():
    ticker = request.args.get("ticker")
    data = fetch_data(ticker)
    data.reset_index(inplace=True)
    data_dict = data.to_dict(orient="records")
    return jsonify(data_dict)


@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    ticker = data["ticker"]
    look_back = data.get("look_back", 60)

    if ticker in models:
        model = models[ticker]
    else:
        model = models["general"]

    stock_data = fetch_data(ticker)
    X = prepare_data(stock_data, look_back)

    if len(X) == 0:
        return jsonify({"error": "Not enough data to make predictions"}), 400

    with tf.device("/GPU:0" if len(physical_devices) > 0 else "/CPU:0"):
        predictions = model.predict(X)

    predictions = scalers["general"].inverse_transform(
        np.concatenate([predictions, np.zeros((predictions.shape[0], 3))], axis=1)
    )[:, 0]

    result = {
        "ticker": ticker,
        "predictions": predictions.tolist(),
        "dates": stock_data.index[-len(predictions) :].strftime("%Y-%m-%d").tolist(),
    }
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True)
