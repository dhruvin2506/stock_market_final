# Advanced Stock Market Prediction with LSTM and Technical Analysis

## Overview
An advanced machine learning application that combines LSTM neural networks with technical analysis to predict stock prices and generate trading signals. The system provides comprehensive analysis including technical indicators, automated trading signals, and future price predictions.

## Technologies Used

### Core Technologies
- Python 3.10
- TensorFlow 2.12.0 (Deep Learning Framework)
- Keras (Neural Network API)
- NumPy 1.24.3 (Numerical Computing)
- Pandas 2.0.2 (Data Manipulation)
- Plotly 5.18.0 (Interactive Visualizations)
- yfinance 0.2.18 (Stock Data Retrieval)
- scikit-learn 1.2.2 (Data Preprocessing)

### Machine Learning Components
- LSTM (Long Short-Term Memory) Neural Networks
- MinMax Scaling for Data Normalization
- Train-Test Data Splitting
- Sequence Prediction
- Time Series Analysis

### Technical Analysis Features
- MACD (Moving Average Convergence Divergence)
- RSI (Relative Strength Index)
- Moving Averages (20 and 50-day)
- Bollinger Bands
- Buy/Sell Signal Generation
- Performance Metrics

## Project Structure
```
stock_market_lstm/
│
├── data/                  # Data storage
├── models/               # Saved models
├── src/                 # Source code
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── model.py
│   ├── technical_analysis.py
│   ├── trading_strategy.py
│   └── utils.py
├── requirements.txt
└── main.py
```

## Key Features
1. Real-time Data Analysis
   - Live stock data fetching
   - Historical data processing
   - Technical indicator calculations

2. Price Prediction
   - LSTM-based prediction model
   - 30-day future forecasting
   - Historical prediction analysis

3. Technical Analysis
   - Multiple technical indicators
   - Trend analysis
   - Support/Resistance levels

4. Trading Signals
   - Automated buy/sell signals
   - Risk assessment
   - Performance metrics

5. Interactive Visualization
   - Real-time charting
   - Multiple timeframe analysis
   - Customizable indicators

## Results and Performance
- Accurate price predictions with LSTM
- Technical indicator integration
- Automated trading signals
- Performance metrics including:
  - Total Return
  - Maximum Drawdown
  - Win Rate
  - Risk-Reward Ratio
  - Sharpe Ratio

## Installation

1. Clone the repository
```bash
git clone https://github.com/yourusername/stock_market_prediction.git
cd stock_market_prediction
```

2. Create virtual environment
```bash
python -m venv venv_tf
# Windows
venv_tf\Scripts\activate
# macOS/Linux
source venv_tf/bin/activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

## Usage
1. Run the main script:
```bash
python main.py
```

2. Enter a stock symbol when prompted (e.g., AAPL, MSFT, GOOGL)

3. The program will:
   - Fetch historical data
   - Process and analyze the data
   - Generate predictions
   - Display interactive visualizations
   - Provide trading signals and metrics

## Technical Implementation Details

### LSTM Model Architecture
- Input Layer: 60-day sequence length
- LSTM Layers with dropout
- Dense output layer
- Adam optimizer
- Mean Squared Error loss function

### Data Processing Pipeline
- Sequence generation (60-day windows)
- MinMax scaling (0-1 range)
- 80/20 train-test split
- Real-time data updates

### Technical Analysis
- Multiple indicator calculations
- Signal generation algorithms
- Risk assessment metrics
- Performance tracking

## Future Improvements
- Enhanced ML model architecture
- Additional technical indicators
- Portfolio optimization
- Risk management tools
- Backtesting capabilities

## Contributing
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
- yfinance for providing stock market data
- TensorFlow team for the deep learning framework
- Plotly for interactive visualizations
