# Stock Market Prediction with LSTM and Technical Analysis

A machine learning application that predicts stock prices using LSTM neural networks and provides technical analysis with trading signals.

## Features

- Real-time stock data fetching for any US-listed stock
- Price prediction using LSTM (Long Short-Term Memory) neural networks
- Technical analysis indicators:
  - MACD (Moving Average Convergence Divergence)
  - RSI (Relative Strength Index)
  - Moving Averages (20 and 50-day)
  - Bollinger Bands
- Trading signals:
  - Buy/Sell indicators
  - Performance metrics
  - Risk assessment
- Interactive visualization with Plotly
- Future price predictions (30 days forecast)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd stock_market_lstm
```

2. Create and activate a virtual environment:
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

```
stock_market_lstm/
│
├── data/                  # Directory for storing data
├── models/               # Directory for saving trained models
├── src/                 # Source code directory
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── model.py
│   ├── technical_analysis.py
│   ├── trading_strategy.py
│   └── utils.py
├── requirements.txt     # Project dependencies
└── main.py            # Main script to run the program
```

## Usage

1. Run the main script:
```bash
python main.py
```

2. Enter a stock symbol when prompted (e.g., AAPL, MSFT, GOOGL)

3. The program will:
   - Fetch historical data
   - Train the LSTM model
   - Generate predictions
   - Display technical indicators
   - Show trading signals and metrics

## Technical Indicators

- **MACD**: Trend-following momentum indicator
- **RSI**: Momentum oscillator measuring speed and magnitude of price changes
- **Moving Averages**: 20-day and 50-day simple moving averages
- **Bollinger Bands**: Volatility indicator

## Trading Metrics

The application provides various trading metrics:
- Total Return
- Maximum Drawdown
- Win Rate
- Risk-Reward Ratio
- Sharpe Ratio
- Current Position Recommendations

## Dependencies

- numpy
- pandas
- tensorflow
- plotly
- yfinance
- scikit-learn
- python-dotenv

## Model Architecture

The LSTM model includes:
- Input layer with sequence length of 60 days
- LSTM layers with dropout for regularization
- Dense layers for output prediction
- Adam optimizer with MSE loss function

## Data Processing

- Sequences of 60 days used for prediction
- MinMax scaling for data normalization
- 80/20 train-test split
- Real-time data fetching using yfinance

## Visualization

Interactive plots include:
- Stock price with predictions
- Technical indicators
- Trading signals
- Performance metrics
- Date range selector
- Zoom capabilities

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details

## Acknowledgments

- yfinance for providing stock market data
- Plotly for interactive visualizations
- TensorFlow team for the deep learning framework#   s t o c k _ m a r k e t  
 