# Real-Time Stock Price Predictor using Deep Learning and CUDA
**Description:**

This project involves developing a real-time stock price prediction system leveraging deep learning and CUDA for GPU acceleration. The system uses an LSTM network to predict stock prices based on historical data and integrates a user-friendly web interface for real-time predictions. The backend is built with Flask and the frontend with React.js, providing an interactive platform for users to select datasets, visualize historical data, and see future price predictions.

**Features:**
- **Real-Time Data Collection**: Fetches real-time stock data from financial APIs such as Yahoo Finance.
- **Deep Learning Model**: Utilizes LSTM networks for accurate time series prediction of stock prices.
- **CUDA Acceleration**: Implements CUDA for efficient training and inference, reducing computation time significantly.
- **User Interface**: Interactive web application built with React.js allowing users to select different datasets and view predictions.
- **Backend API**: Flask-based API handling data requests and model predictions, ensuring seamless integration with the frontend.

**Technologies Used:**
- Python, TensorFlow, CUDA, Flask, React.js, Axios, yFinance, MinMaxScaler, LSTM, Chart.js

**Setup Instructions:**
1. Clone the repository.
2. Set up the Python environment and install dependencies.
3. Run the Flask backend server.
4. Set up the React frontend and run the application.

**Usage:**
1. Enter the stock ticker in the frontend interface.
2. Click "Predict" to fetch data and generate predictions.
3. View historical data and future predictions on the interactive chart.

**Future Enhancements:**
- Incorporate additional financial indicators and news sentiment analysis.
- Optimize the model and data pipeline for larger datasets and multiple stocks.
- Enhance UI/UX with more interactive features and visualizations.

