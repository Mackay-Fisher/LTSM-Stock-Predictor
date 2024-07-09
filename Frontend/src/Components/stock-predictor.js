import React, { useState, useEffect } from "react";
import axios from "axios";
import { Line } from "react-chartjs-2";
import "chart.js/auto";

const StockPredictor = () => {
  const [ticker, setTicker] = useState("AAPL");
  const [data, setData] = useState([]);
  const [predictions, setPredictions] = useState([]);
  const [dates, setDates] = useState([]);

  const fetchData = async () => {
    const response = await axios.get(
      `http://localhost:5000/get_data?ticker=${ticker}`,
    );
    setData(response.data);
  };

  const fetchPredictions = async () => {
    const response = await axios.post("http://localhost:5000/predict", {
      data,
    });
    setDates(response.data.dates);
    setPredictions(response.data.predictions);
  };

  const trainModel = async () => {
    await axios.post("http://localhost:5000/train", { ticker });
    fetchPredictions();
  };

  useEffect(() => {
    fetchData();
  }, [ticker]);

  const handlePredict = () => {
    fetchPredictions();
  };

  const chartData = {
    labels: dates,
    datasets: [
      {
        label: "Actual Price",
        data: data.map((d) => d.Close),
        borderColor: "blue",
        fill: false,
      },
      {
        label: "Predicted Price",
        data: predictions,
        borderColor: "red",
        fill: false,
      },
    ],
  };

  return (
    <div style={{ textAlign: "center", padding: "20px" }}>
      <h1>Stock Price Predictor</h1>
      <input
        type="text"
        value={ticker}
        onChange={(e) => setTicker(e.target.value.toUpperCase())}
        placeholder="Enter Stock Ticker"
        style={{ padding: "10px", fontSize: "16px", margin: "10px" }}
      />
      <br />
      <button
        onClick={trainModel}
        style={{ padding: "10px 20px", fontSize: "16px", cursor: "pointer" }}
      >
        Train Model
      </button>
      <button
        onClick={handlePredict}
        style={{
          padding: "10px 20px",
          fontSize: "16px",
          cursor: "pointer",
          marginLeft: "10px",
        }}
      >
        Predict
      </button>
      <div style={{ marginTop: "20px" }}>
        <Line data={chartData} />
      </div>
    </div>
  );
};

export default StockPredictor;
