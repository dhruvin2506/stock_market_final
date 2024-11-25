import numpy as np
import pandas as pd
from datetime import datetime

class TradingStrategy:
    def __init__(self, close_prices, rsi, macd_hist, sma_20, sma_50):
        """
        Initialize trading strategy with technical indicators.
        """
        self.close_prices = close_prices
        self.rsi = rsi
        self.macd_hist = macd_hist
        self.sma_20 = sma_20
        self.sma_50 = sma_50
        
    def generate_signals(self, initial_capital=10000, risk_per_trade=0.02):
        """
        Generate trading signals with position sizing and risk assessment.
        
        Args:
            initial_capital (float): Initial trading capital
            risk_per_trade (float): Maximum risk per trade as percentage of capital
        """
        signals = pd.DataFrame(index=range(len(self.close_prices)))
        signals['price'] = self.close_prices
        signals['signal'] = 0  # 1 for buy, -1 for sell, 0 for hold
        signals['position'] = 0
        signals['stop_loss'] = 0
        signals['take_profit'] = 0
        signals['position_size'] = 0
        signals['capital'] = initial_capital
        signals['risk_amount'] = 0
        
        # Calculate signals based on multiple indicators
        for i in range(1, len(signals)):
            # RSI signals
            rsi_buy = self.rsi[i-1] < 30 and self.rsi[i] >= 30
            rsi_sell = self.rsi[i-1] > 70 and self.rsi[i] <= 70
            
            # MACD signals
            macd_buy = self.macd_hist[i-1] < 0 and self.macd_hist[i] > 0
            macd_sell = self.macd_hist[i-1] > 0 and self.macd_hist[i] < 0
            
            # Moving average signals
            ma_buy = self.sma_20[i] > self.sma_50[i] and self.sma_20[i-1] <= self.sma_50[i-1]
            ma_sell = self.sma_20[i] < self.sma_50[i] and self.sma_20[i-1] >= self.sma_50[i-1]
            
            # Combined signals
            if (rsi_buy or macd_buy or ma_buy) and signals['position'][i-1] <= 0:
                signals.loc[i, 'signal'] = 1
            elif (rsi_sell or macd_sell or ma_sell) and signals['position'][i-1] >= 0:
                signals.loc[i, 'signal'] = -1
            
            # Position tracking
            if signals['signal'][i] == 1:  # Buy signal
                signals.loc[i, 'position'] = 1
                # Calculate stop loss (2% below entry)
                signals.loc[i, 'stop_loss'] = signals['price'][i] * 0.98
                # Calculate take profit (1.5x risk)
                risk = signals['price'][i] - signals.loc[i, 'stop_loss']
                signals.loc[i, 'take_profit'] = signals['price'][i] + (risk * 1.5)
                # Calculate position size based on risk
                risk_amount = signals['capital'][i-1] * risk_per_trade
                signals.loc[i, 'risk_amount'] = risk_amount
                signals.loc[i, 'position_size'] = risk_amount / (signals['price'][i] - signals.loc[i, 'stop_loss'])
                signals.loc[i, 'capital'] = signals['capital'][i-1]
            
            elif signals['signal'][i] == -1:  # Sell signal
                signals.loc[i, 'position'] = -1
                # Update capital based on previous position if exists
                if signals['position'][i-1] == 1:
                    profit = (signals['price'][i] - signals['price'][i-1]) * signals['position_size'][i-1]
                    signals.loc[i, 'capital'] = signals['capital'][i-1] + profit
                else:
                    signals.loc[i, 'capital'] = signals['capital'][i-1]
            
            else:  # Hold
                signals.loc[i, 'position'] = signals['position'][i-1]
                signals.loc[i, 'stop_loss'] = signals['stop_loss'][i-1]
                signals.loc[i, 'take_profit'] = signals['take_profit'][i-1]
                signals.loc[i, 'position_size'] = signals['position_size'][i-1]
                
                # Check if stop loss or take profit hit
                if signals['position'][i-1] == 1:
                    if signals['price'][i] <= signals['stop_loss'][i-1]:
                        # Stop loss hit
                        loss = (signals['stop_loss'][i-1] - signals['price'][i-1]) * signals['position_size'][i-1]
                        signals.loc[i, 'capital'] = signals['capital'][i-1] + loss
                        signals.loc[i, 'position'] = 0
                    elif signals['price'][i] >= signals['take_profit'][i-1]:
                        # Take profit hit
                        profit = (signals['take_profit'][i-1] - signals['price'][i-1]) * signals['position_size'][i-1]
                        signals.loc[i, 'capital'] = signals['capital'][i-1] + profit
                        signals.loc[i, 'position'] = 0
                    else:
                        signals.loc[i, 'capital'] = signals['capital'][i-1]
                else:
                    signals.loc[i, 'capital'] = signals['capital'][i-1]
        
        return signals

    def calculate_performance_metrics(self, signals):
        """
        Calculate trading performance metrics.
        """
        metrics = {}
        
        # Calculate returns
        signals['returns'] = signals['capital'].pct_change()
        
        # Total return
        metrics['total_return'] = (signals['capital'].iloc[-1] - signals['capital'].iloc[0]) / signals['capital'].iloc[0] * 100
        
        # Maximum drawdown
        cummax = signals['capital'].cummax()
        drawdown = (signals['capital'] - cummax) / cummax
        metrics['max_drawdown'] = drawdown.min() * 100
        
        # Win rate
        trades = signals[signals['signal'] != 0]
        winning_trades = trades[trades['capital'] > trades['capital'].shift(1)]
        metrics['win_rate'] = len(winning_trades) / len(trades) * 100 if len(trades) > 0 else 0
        
        # Risk-reward ratio
        avg_win = winning_trades['capital'].diff().mean()
        losing_trades = trades[trades['capital'] < trades['capital'].shift(1)]
        avg_loss = abs(losing_trades['capital'].diff().mean())
        metrics['risk_reward_ratio'] = abs(avg_win / avg_loss) if avg_loss != 0 else 0
        
        # Sharpe ratio (assuming risk-free rate of 2%)
        rf_rate = 0.02
        excess_returns = signals['returns'] - rf_rate/252  # Daily risk-free rate
        sharpe_ratio = np.sqrt(252) * excess_returns.mean() / excess_returns.std()
        metrics['sharpe_ratio'] = sharpe_ratio
        
        return metrics

    def get_current_position_recommendation(self, current_price):
        """
        Get recommendation for current market position.
        """
        recommendation = {
            'action': 'HOLD',
            'confidence': 0,
            'stop_loss': 0,
            'take_profit': 0,
            'position_size': 0,
            'risk_level': 'LOW'
        }
        
        # Calculate confidence based on indicators
        rsi_signal = 1 if self.rsi[-1] < 30 else -1 if self.rsi[-1] > 70 else 0
        macd_signal = 1 if self.macd_hist[-1] > 0 else -1
        ma_signal = 1 if self.sma_20[-1] > self.sma_50[-1] else -1
        
        # Combine signals with weights
        confidence = (rsi_signal * 0.35 + macd_signal * 0.35 + ma_signal * 0.3) * 100
        
        # Determine action and risk level
        if confidence > 30:
            recommendation['action'] = 'BUY'
            stop_loss = current_price * 0.98
            risk = current_price - stop_loss
            recommendation['stop_loss'] = stop_loss
            recommendation['take_profit'] = current_price + (risk * 1.5)
            recommendation['risk_level'] = 'HIGH' if abs(confidence) > 70 else 'MEDIUM'
        elif confidence < -30:
            recommendation['action'] = 'SELL'
            recommendation['risk_level'] = 'HIGH' if abs(confidence) > 70 else 'MEDIUM'
        
        recommendation['confidence'] = abs(confidence)
        
        return recommendation