import React, { useState } from "react";
import axios from "axios";
import { Line } from "react-chartjs-2";

const App = () => {
  const [ticker, setTicker] = useState("");
  const [data, setData] = useState([]);
  const [predictions, setPredictions] = useState([]);

  const fetchData = async () => {
    const response = await axios.get(
      `http://localhost:5000/get_stock_data?ticker=${ticker}`,
    );
    setData(response.data);
  };

  const predictData = async () => {
    const response = await axios.post("http://localhost:5000/predict", {
      ticker,
    });
    setPredictions(response.data.predictions);
  };

  const chartData = {
    labels: data.map((d) => d.Date),
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
    <div>
      <h1>Stock Price Predictor</h1>
      <input
        type="text"
        value={ticker}
        onChange={(e) => setTicker(e.target.value.toUpperCase())}
        placeholder="Enter Stock Ticker"
      />
      <button onClick={fetchData}>Fetch Data</button>
      <button onClick={predictData}>Predict</button>
      <Line data={chartData} />
    </div>
  );
};

export default App;
