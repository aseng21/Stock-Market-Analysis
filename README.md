# Stock Market Analysis

## Features
- **Stock Price Predictions**: Uses models like Linear Regression, Random Forest, Gradient Boosting, XGBoost, Support Vector Regressor, Multi-Layer Perceptron, and LSTM to forecast stock prices.
- **Technical Indicators**: Analyzes popular indicators such as Moving Averages, RSI, MACD, etc.
- **Interactive GUI**: Visualizes stock performance and prediction results 
  using Matplotlib and Plotly.
- **Feature Engineering**: Automated feature selection using SelectKBest for improved accuracy.
- **Hyperparameter Tuning**: Optimizes models with GridSearchCV.

## Technology Stack
- **Frontend**: PyQt5
- **Backend**: Python, Machine Learning models (Scikit-learn, TensorFlow, XGBoost)
- **Data Sources**: `yfinance` for stock data
- **Libraries Used**: 
  - `pandas` for data manipulation
  - `matplotlib` & `plotly` for visualization
  - `seaborn` for statistical plots
  - `scikit-learn` for machine learning algorithms and utilities
  - `tensorflow` & `keras` for deep learning (LSTM)
  - `xgboost` for gradient boosting algorithms

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/aseng21/Stock_Market_Analysis
   cd Stock_Market_Analysis
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   python app.py
   ```

## Usage
1. **Select a Stock**: Enter a stock ticker (e.g., AAPL, TSLA) to retrieve historical data.
2. **Choose Model & Indicator**: Select a prediction model and any technical indicators you'd like to analyze.
3. **Train & Predict**: The app will train the selected model and display predictions along with relevant charts.
4. **View Results**: Analyze the predicted stock price, model performance metrics, and visual trends.

## Models Supported
- Linear Regression
- Random Forest
- Gradient Boosting
- XGBoost
- Support Vector Regressor
- Multi-Layer Perceptron (MLP)
- Long Short-Term Memory (LSTM)

## Contributing
Feel free to contribute by submitting a pull request. Please ensure all changes are tested and documented.

## Future Enhancements
- Add more machine learning models and techniques.
- Improve the GUI with additional customization options.
- Implement advanced portfolio management tools.
- Add support for cryptocurrency predictions.
