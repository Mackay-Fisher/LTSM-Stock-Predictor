import React from "react";
import { Link } from "react-router-dom";

const LandingPage = () => {
  return (
    <div style={{ textAlign: "center", padding: "50px" }}>
      <h1>Welcome to the AI Stock Predictor</h1>
      <p>
        Our AI Stock Predictor uses advanced machine learning models to predict
        the future prices of stocks based on historical data. By leveraging
        techniques such as LSTM (Long Short-Term Memory) networks and
        Bidirectional LSTMs, we aim to provide accurate and real-time stock
        price predictions.
      </p>
      <p>
        Simply enter the stock ticker of your choice, and our system will fetch
        the historical data, process it through our trained models, and provide
        you with the predicted stock prices. You can also train new models for
        specific stocks or use our general model trained on multiple stocks.
      </p>
      <Link to="/predictor">
        <button
          style={{ padding: "10px 20px", fontSize: "16px", cursor: "pointer" }}
        >
          Go to Stock Predictor
        </button>
      </Link>
    </div>
  );
};

export default LandingPage;
