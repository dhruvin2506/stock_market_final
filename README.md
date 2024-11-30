# Stock Market Prediction with LSTM and Technical Analysis

## Project Overview
An advanced machine learning application that combines LSTM neural networks with technical analysis to predict stock prices and generate trading signals. The system provides real-time stock data analysis, future price predictions, and comprehensive trading metrics.

## 🚀 Features
- Real-time stock data fetching for any US-listed stock
- Deep learning-based price predictions using LSTM networks
- Technical analysis with multiple indicators
- Interactive data visualization
- Automated trading signals
- Performance metrics and risk assessment
- 30-day future price forecasting

## 🛠️ Technologies Used

### Core Technologies
- **Python 3.10** - Primary programming language
- **TensorFlow 2.12.0** - Deep learning framework
- **Keras** - Neural network API
- **NumPy 1.24.3** - Numerical computations
- **Pandas 2.0.2** - Data manipulation
- **Plotly 5.18.0** - Interactive visualizations
- **yfinance 0.2.18** - Stock data retrieval
- **scikit-learn 1.2.2** - Data preprocessing

### Technical Components

#### Machine Learning
- LSTM (Long Short-Term Memory) neural networks
- Sequential model architecture
- MinMax scaling for data normalization
- Train-test data splitting
- Time series prediction

#### Technical Indicators
- MACD (Moving Average Convergence Divergence)
- RSI (Relative Strength Index)
- Moving Averages (20 and 50-day)
- Bollinger Bands

#### Trading Analysis
- Buy/Sell signal generation
- Risk-reward calculations
- Performance metrics computation
- Sharpe ratio analysis

## 📊 Project Structure
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

## 🚀 Installation and Setup

### Prerequisites
- Python 3.10 or higher
- Git
- pip (Python package installer)

### Installation Steps

1. Clone the repository
```bash
git clone https://github.com/dhruvin2506/stock_market_final.git
cd stock_market_final
```

2. Create and activate virtual environment
```bash
# Create environment
python -m venv venv_tf

# Activate environment
# Windows:
venv_tf\Scripts\activate
# macOS/Linux:
source venv_tf/bin/activate
```

3. Install required packages
```bash
pip install -r requirements.txt
```

## 💻 Usage

1. Run the main script:
```bash
python main.py
```

2. When prompted, enter a stock symbol (e.g., AAPL, MSFT, GOOGL)

3. The program will:
   - Fetch historical stock data
   - Process and prepare the data
   - Train the LSTM model
   - Generate predictions
   - Display interactive visualizations
   - Show trading metrics and signals

## 📈 Technical Features

### LSTM Model Architecture
- Input Layer: 60-day sequence length
- LSTM Layers with dropout regularization
- Dense output layer
- Adam optimizer
- Mean Squared Error loss function

### Data Processing Pipeline
- Sequence generation (60-day windows)
- MinMax scaling (0-1 range)
- 80/20 train-test split
- Real-time data updates

### Technical Analysis
- MACD calculation with standard periods (12, 26, 9)
- RSI computation with 14-period window
- Moving averages: 20-day and 50-day
- Bollinger Bands with 20-day window

### Trading Metrics
- Total Return
- Maximum Drawdown
- Win Rate
- Risk-Reward Ratio
- Sharpe Ratio
- Current Position Analysis

## 📊 Visualization Features
- Interactive price charts
- Technical indicator overlays
- Buy/Sell signal markers
- Performance metric dashboard
- Future prediction visualization
- Adjustable time ranges
- Zoom and pan capabilities

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch
```bash
git checkout -b feature/AmazingFeature
```
3. Commit your changes
```bash
git commit -m 'Add some AmazingFeature'
```
4. Push to the branch
```bash
git push origin feature/AmazingFeature
```
5. Open a Pull Request

## 📝 License
This project is licensed under the MIT License - see the LICENSE file for details.

## 🎯 Future Improvements
- Additional technical indicators
- Enhanced prediction accuracy
- Portfolio management features
- Risk management tools
- Backtesting capabilities
- API integration
- Web interface

## 👥 Contact
Dhruvin Patel - dhruvin2506@gmail.com
Project Link: [https://github.com/dhruvin2506/stock_market_final](https://github.com/dhruvin2506/stock_market_final)
