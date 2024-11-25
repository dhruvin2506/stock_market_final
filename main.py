import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.utils import fetch_complete_stock_data, save_data, search_ticker
from src.data_preprocessing import DataPreprocessor
from src.model import StockPredictor
from src.technical_analysis import TechnicalAnalysis
from src.trading_strategy import TradingStrategy 

def create_interactive_plot(stock_data, actual, predicted, future_predictions, title, dates, future_dates, sequence_length):
    """Create an interactive plot using Plotly with technical indicators."""
    # Ensure arrays are properly shaped
    actual = np.array(actual).ravel()
    predicted = np.array(predicted).ravel()
    future_predictions = np.array(future_predictions).ravel()

    # Calculate technical indicators
    ta = TechnicalAnalysis()
    
    # Ensure close prices are 1-dimensional
    close_prices = stock_data['Close'].values.ravel()
    
    try:
        # Calculate indicators
        print("Calculating MACD...")
        macd_line, signal_line, macd_hist = ta.calculate_macd(close_prices)
        
        print("Calculating RSI...")
        rsi = ta.calculate_rsi(close_prices)
        
        print("Calculating Moving Averages...")
        sma_20, sma_50, sma_200 = ta.calculate_moving_averages(close_prices)
        
        print("Calculating Bollinger Bands...")
        bb_upper, bb_middle, bb_lower = ta.calculate_bollinger_bands(close_prices)

        # Calculate trading signals first
        trading_strategy = TradingStrategy(
            close_prices=close_prices,
            rsi=rsi,
            macd_hist=macd_hist,
            sma_20=sma_20,
            sma_50=sma_50
        )
        
        # Generate trading signals and metrics
        signals_df = trading_strategy.generate_signals()
        metrics = trading_strategy.calculate_performance_metrics(signals_df)
        current_recommendation = trading_strategy.get_current_position_recommendation(close_prices[-1])
        
        # Get buy/sell signals
        buy_signals = signals_df[signals_df['signal'] == 1]
        sell_signals = signals_df[signals_df['signal'] == -1]

        # Create subplots
        fig = make_subplots(rows=4, cols=1,
                           shared_xaxes=True,
                           vertical_spacing=0.20,
                           row_heights=[0.45, 0.2, 0.2, 0.15],
                           subplot_titles=(title, 'MACD', 'RSI', 'Price Difference'))

        # Main price plot
        fig.add_trace(
            go.Scatter(x=dates, y=actual,
                      name="Actual",
                      line=dict(color='blue'),
                      hovertemplate="Date: %{x}<br>" +
                                   "Actual Price: $%{y:.2f}<br>" +
                                   "<extra></extra>"),
            row=1, col=1
        )

        fig.add_trace(
            go.Scatter(x=dates, y=predicted,
                      name="Historical Predictions",
                      line=dict(color='red'),
                      hovertemplate="Date: %{x}<br>" +
                                   "Predicted Price: $%{y:.2f}<br>" +
                                   "<extra></extra>"),
            row=1, col=1
        )

        fig.add_trace(
            go.Scatter(x=future_dates, y=future_predictions,
                      name="Future Predictions",
                      line=dict(color='green', dash='dash'),
                      hovertemplate="Date: %{x}<br>" +
                                   "Predicted Price: $%{y:.2f}<br>" +
                                   "<extra></extra>"),
            row=1, col=1
        )
        
        # Calculate aligned dates and valid length
        indicator_dates = stock_data.index[sequence_length:]
        valid_length = len(indicator_dates)

        # Add Moving Averages
        if len(sma_20) >= valid_length:
            fig.add_trace(
                go.Scatter(x=indicator_dates,
                          y=sma_20[-valid_length:],
                          name="SMA 20",
                          line=dict(color='orange', width=1)),
                row=1, col=1
            )
        
        if len(sma_50) >= valid_length:
            fig.add_trace(
                go.Scatter(x=indicator_dates,
                          y=sma_50[-valid_length:],
                          name="SMA 50",
                          line=dict(color='purple', width=1)),
                row=1, col=1
            )

        # Add buy/sell markers with proper index checking
        if len(buy_signals) > 0:
            valid_buy_indices = [idx for idx in buy_signals.index if idx < valid_length]
            if valid_buy_indices:
                buy_dates = [indicator_dates[idx] for idx in valid_buy_indices]
                buy_prices = [close_prices[sequence_length + idx] for idx in valid_buy_indices]
                fig.add_trace(
                    go.Scatter(
                        x=buy_dates,
                        y=buy_prices,
                        mode='markers',
                        name='Buy Signal',
                        marker=dict(
                            symbol='triangle-up',
                            size=15,
                            color='green',
                            line=dict(width=2)
                        )
                    ),
                    row=1, col=1
                )

        if len(sell_signals) > 0:
            valid_sell_indices = [idx for idx in sell_signals.index if idx < valid_length]
            if valid_sell_indices:
                sell_dates = [indicator_dates[idx] for idx in valid_sell_indices]
                sell_prices = [close_prices[sequence_length + idx] for idx in valid_sell_indices]
                fig.add_trace(
                    go.Scatter(
                        x=sell_dates,
                        y=sell_prices,
                        mode='markers',
                        name='Sell Signal',
                        marker=dict(
                            symbol='triangle-down',
                            size=15,
                            color='red',
                            line=dict(width=2)
                        )
                    ),
                    row=1, col=1
                )

        # Add MACD
        if len(macd_line) >= valid_length:
            fig.add_trace(
                go.Scatter(x=indicator_dates,
                          y=macd_line[-valid_length:],
                          name="MACD",
                          line=dict(color='blue', width=1)),
                row=2, col=1
            )
            
            fig.add_trace(
                go.Scatter(x=indicator_dates,
                          y=signal_line[-valid_length:],
                          name="Signal",
                          line=dict(color='orange', width=1)),
                row=2, col=1
            )
            
            fig.add_trace(
                go.Bar(x=indicator_dates,
                      y=macd_hist[-valid_length:],
                      name="MACD Histogram",
                      marker_color='gray'),
                row=2, col=1
            )

        # Add RSI
        if len(rsi) >= valid_length:
            fig.add_trace(
                go.Scatter(x=indicator_dates,
                          y=rsi[-valid_length:],
                          name="RSI",
                          line=dict(color='purple', width=1)),
                row=3, col=1
            )
            
            # Add RSI levels
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)

        # Price Difference
        price_diff = actual - predicted
        fig.add_trace(
            go.Scatter(x=dates, y=price_diff,
                      name="Price Difference",
                      fill='tozeroy',
                      line=dict(color='gray'),
                      hovertemplate="Date: %{x}<br>" +
                                   "Difference: $%{y:.2f}<br>" +
                                   "<extra></extra>"),
            row=4, col=1
        )

        # Add metrics text
        metrics_text = (
            f"Trading Metrics:<br>" +
            f"Total Return: {metrics['total_return']:.2f}%<br>" +
            f"Max Drawdown: {metrics['max_drawdown']:.2f}%<br>" +
            f"Win Rate: {metrics['win_rate']:.2f}%<br>" +
            f"Risk-Reward: {metrics['risk_reward_ratio']:.2f}<br>" +
            f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}<br><br>" +
            f"Current Position:<br>" +
            f"Action: {current_recommendation['action']}<br>" +
            f"Confidence: {current_recommendation['confidence']:.2f}%<br>" +
            f"Risk Level: {current_recommendation['risk_level']}"
        )

        # Update layout
        fig.update_layout(
            height=2000,
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            ),
            hovermode='x unified',
            template='plotly_white',
            margin=dict(r=150)  # Add right margin for metrics
        )

        # Add metrics annotation
        fig.add_annotation(
            xref="paper",
            yref="paper",
            x=1.15,
            y=0.98,
            text=metrics_text,
            showarrow=False,
            font=dict(size=10),
            align="left",
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="gray",
            borderwidth=1
        )

        # Update axes labels
        fig.update_yaxes(title_text="Price ($)", row=1, col=1)
        fig.update_yaxes(title_text="MACD", row=2, col=1)
        fig.update_yaxes(title_text="RSI", row=3, col=1)
        fig.update_yaxes(title_text="Difference ($)", row=4, col=1)

        # Add range selector and slider
        fig.update_xaxes(
            rangeselector=dict(
                buttons=list([
                    dict(count=7, label="1w", step="day", stepmode="backward"),
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=3, label="3m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(step="all", label="All")
                ])
            ),
            rangeslider=dict(visible=True),
            type="date"
        )

        # Show the plot
        fig.show()
        
    except Exception as e:
        print(f"Error in create_interactive_plot: {e}")
        raise

# Rest of your code (generate_future_dates, predict_future, and main functions) remains the same

def generate_future_dates(last_date, num_days):
    """Generate future dates for predictions."""
    future_dates = []
    current_date = last_date
    for _ in range(num_days):
        current_date = current_date + timedelta(days=1)
        # Skip weekends
        while current_date.weekday() > 4:  # 5 is Saturday, 6 is Sunday
            current_date = current_date + timedelta(days=1)
        future_dates.append(current_date)
    return future_dates

def predict_future(model, preprocessor, last_sequence, num_days=30):
    """Generate predictions for future dates."""
    predictions = []
    current_sequence = last_sequence.copy()
    
    for _ in range(num_days):
        # Get the prediction for the next day
        current_sequence_reshaped = current_sequence.reshape((1, current_sequence.shape[0], 1))
        next_pred = model.predict(current_sequence_reshaped)
        predictions.append(next_pred[0, 0])
        
        # Update the sequence with the new prediction
        current_sequence = np.roll(current_sequence, -1)
        current_sequence[-1] = next_pred
        
    # Convert predictions back to original scale
    scaled_predictions = np.array(predictions).reshape(-1, 1)
    return preprocessor.inverse_transform(scaled_predictions)

def main():
    while True:
        print("\nStock Market Predictor")
        print("1. Enter a stock symbol")
        print("2. Exit")
        
        choice = input("\nEnter your choice (1-2): ")
        
        if choice == "2":
            break
            
        if choice == "1":
            search_query = input("\nEnter stock symbol (e.g., AAPL): ").strip().upper()
            if not search_query:
                print("Please enter a valid stock symbol.")
                continue
                
            print("\nSearching...")
            stock_info = search_ticker(search_query)
            
            if not stock_info:
                print("No stock found. Please verify the symbol and try again.")
                continue
            
            selected_stock = stock_info[0]
            
            proceed = input("\nWould you like to predict this stock's price? (y/n): ").lower()
            if proceed != 'y':
                continue
            
            # Configuration
            SYMBOL = selected_stock['symbol']
            SEQUENCE_LENGTH = 60
            TRAIN_SIZE = 0.8
            FUTURE_DAYS = 30  # Number of days to predict into the future
            
            # Fetch data
            print(f"\nFetching complete historical and recent data for {SYMBOL}...")
            stock_data = fetch_complete_stock_data(SYMBOL)
            
            if stock_data is None or stock_data.empty:
                print("Failed to fetch data. Please try another stock.")
                continue
            
            # Print data range
            print(f"\nData range: {stock_data.index[0].strftime('%Y-%m-%d')} to {stock_data.index[-1].strftime('%Y-%m-%d')}")
            future_start_date = stock_data.index[-1] + timedelta(days=1)
            print(f"Future predictions will start from: {future_start_date.strftime('%Y-%m-%d')}")
            
            # Save raw data
            save_data(stock_data, f'{SYMBOL}_raw_data.csv')
            
            # Preprocess data
            print("\nPreprocessing data...")
            preprocessor = DataPreprocessor()
            X, y = preprocessor.prepare_sequences(stock_data.values, SEQUENCE_LENGTH)
            X_train, X_test, y_train, y_test = preprocessor.split_data(X, y, TRAIN_SIZE)
            
            # Create and train model
            print("\nInitializing model...")
            model = StockPredictor(SEQUENCE_LENGTH)
            model.build_model()
            
            # Split training data into training and validation sets
            val_split = int(len(X_train) * 0.8)
            X_train_final = X_train[:val_split]
            y_train_final = y_train[:val_split]
            X_val = X_train[val_split:]
            y_val = y_train[val_split:]
            
            print("\nTraining model... This may take a few minutes.")
            history = model.train(X_train_final, y_train_final, X_val, y_val)
            
            # Make predictions
            print("\nGenerating predictions...")
            train_predictions = model.predict(X_train)
            test_predictions = model.predict(X_test)
            
            # Generate future predictions
            print("\nGenerating future predictions...")
            last_sequence = X_test[-1]  # Use the last sequence from test data
            future_pred = predict_future(model, preprocessor, last_sequence, FUTURE_DAYS)
            
            # Inverse transform predictions
            train_predictions = preprocessor.inverse_transform(train_predictions)
            test_predictions = preprocessor.inverse_transform(test_predictions)
            actual_values = preprocessor.inverse_transform(y.reshape(-1, 1))
            
            # Get dates for plotting
            dates = stock_data.index[SEQUENCE_LENGTH:]
            train_dates = dates[:len(train_predictions)]
            test_dates = dates[len(train_predictions):]
            future_dates = generate_future_dates(test_dates[-1], FUTURE_DAYS)
            
            print("\nGenerating interactive plots...")
            # Plot results with future predictions
            create_interactive_plot(
                stock_data,
                actual_values[SEQUENCE_LENGTH:len(train_predictions)+SEQUENCE_LENGTH],
                train_predictions,
                future_pred,
                f'{SYMBOL} Stock Price Predictions with Future Forecast',
                train_dates,
                future_dates,
                SEQUENCE_LENGTH
            )
            
            input("\nPress Enter to continue...")

if __name__ == "__main__":
    main()