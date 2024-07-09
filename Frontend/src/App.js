import React from "react";
import { BrowserRouter as Router, Route, Routes } from "react-router-dom";
import LandingPage from "./Components/landing-page.js";
import StockPredictor from "./Components/stock-predictor.js";

const App = () => {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<LandingPage />} />
        <Route path="/predictor" element={<StockPredictor />} />
      </Routes>
    </Router>
  );
};

export default App;
