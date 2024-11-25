import numpy as np
import pandas as pd

class TechnicalAnalysis:
    @staticmethod
    def calculate_macd(data, fast=12, slow=26, signal=9):
        """Calculate MACD (Moving Average Convergence Divergence)"""
        # Ensure data is 1-dimensional
        if isinstance(data, np.ndarray):
            data = data.ravel()
        elif isinstance(data, (pd.Series, pd.DataFrame)):
            data = data.values.ravel()

        # Calculate MACD using pandas Series
        series = pd.Series(data)
        exp1 = series.ewm(span=fast, adjust=False).mean()
        exp2 = series.ewm(span=slow, adjust=False).mean()
        macd_line = exp1 - exp2
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        macd_histogram = macd_line - signal_line
        
        return macd_line.values, signal_line.values, macd_histogram.values

    @staticmethod
    def calculate_rsi(data, periods=14):
        """Calculate RSI (Relative Strength Index)"""
        # Ensure data is 1-dimensional
        if isinstance(data, np.ndarray):
            data = data.ravel()
        elif isinstance(data, (pd.Series, pd.DataFrame)):
            data = data.values.ravel()

        series = pd.Series(data)
        delta = series.diff()
        
        gain = (delta.where(delta > 0, 0))
        loss = (-delta.where(delta < 0, 0))
        
        avg_gain = gain.rolling(window=periods).mean()
        avg_loss = loss.rolling(window=periods).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi.values

    @staticmethod
    def calculate_moving_averages(data):
        """Calculate various moving averages"""
        # Ensure data is 1-dimensional
        if isinstance(data, np.ndarray):
            data = data.ravel()
        elif isinstance(data, (pd.Series, pd.DataFrame)):
            data = data.values.ravel()

        series = pd.Series(data)
        sma_20 = series.rolling(window=20).mean()
        sma_50 = series.rolling(window=50).mean()
        sma_200 = series.rolling(window=200).mean()
        
        return sma_20.values, sma_50.values, sma_200.values

    @staticmethod
    def calculate_bollinger_bands(data, window=20):
        """Calculate Bollinger Bands"""
        # Ensure data is 1-dimensional
        if isinstance(data, np.ndarray):
            data = data.ravel()
        elif isinstance(data, (pd.Series, pd.DataFrame)):
            data = data.values.ravel()

        series = pd.Series(data)
        sma = series.rolling(window=window).mean()
        std = series.rolling(window=window).std()
        upper_band = sma + (std * 2)
        lower_band = sma - (std * 2)
        
        return upper_band.values, sma.values, lower_band.values

    @staticmethod
    def get_signals(close, sma_20, sma_50, rsi, macd_hist):
        """Generate trading signals based on technical indicators"""
        signals = []
        
        # Ensure all inputs are 1-dimensional
        close = np.asarray(close).ravel()
        sma_20 = np.asarray(sma_20).ravel()
        sma_50 = np.asarray(sma_50).ravel()
        rsi = np.asarray(rsi).ravel()
        macd_hist = np.asarray(macd_hist).ravel()

        if len(close) < 20:
            return signals

        try:
            # SMA Crossover
            if sma_20[-1] > sma_50[-1] and sma_20[-2] <= sma_50[-2]:
                signals.append("Golden Cross: Bullish signal - SMA 20 crossed above SMA 50")
            elif sma_20[-1] < sma_50[-1] and sma_20[-2] >= sma_50[-2]:
                signals.append("Death Cross: Bearish signal - SMA 20 crossed below SMA 50")

            # RSI signals
            current_rsi = rsi[-1]
            if current_rsi > 70:
                signals.append(f"Overbought: RSI is {current_rsi:.2f}")
            elif current_rsi < 30:
                signals.append(f"Oversold: RSI is {current_rsi:.2f}")

            # MACD signals
            if len(macd_hist) >= 2:
                if macd_hist[-1] > 0 and macd_hist[-2] <= 0:
                    signals.append("MACD Bullish Crossover")
                elif macd_hist[-1] < 0 and macd_hist[-2] >= 0:
                    signals.append("MACD Bearish Crossover")

            # Price trend
            price_change = ((close[-1] - close[-20]) / close[-20]) * 100
            signals.append(f"20-day price change: {price_change:.2f}%")
            
        except Exception as e:
            print(f"Error generating signals: {e}")
            
        return signals