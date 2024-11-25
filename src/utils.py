import os
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

def fetch_complete_stock_data(symbol):
    """
    Fetch complete stock data including recent data up to current date.
    
    Args:
        symbol (str): Stock ticker symbol (e.g., 'AAPL')
    
    Returns:
        pd.DataFrame: DataFrame containing stock data
    """
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=1000)  # Historical data
        
        print(f"Fetching historical and recent data for {symbol}...")
        stock_data = yf.download(symbol, 
                               start=start_date.strftime('%Y-%m-%d'),
                               end=end_date.strftime('%Y-%m-%d'),
                               progress=False)
        
        if stock_data.empty:
            print("No data found for this symbol")
            return None
            
        # Get the most recent data point date
        last_date = stock_data.index[-1]
        print(f"Data fetched from {stock_data.index[0].strftime('%Y-%m-%d')} to {last_date.strftime('%Y-%m-%d')}")
        
        # If there's a gap between last date and current date
        if last_date < end_date:
            print(f"Fetching additional recent data...")
            recent_data = yf.download(symbol,
                                    start=(last_date + timedelta(days=1)).strftime('%Y-%m-%d'),
                                    end=end_date.strftime('%Y-%m-%d'),
                                    progress=False)
            if not recent_data.empty:
                stock_data = pd.concat([stock_data, recent_data])
                print(f"Complete data range: {stock_data.index[0].strftime('%Y-%m-%d')} to {stock_data.index[-1].strftime('%Y-%m-%d')}")
        
        return stock_data[['Close']]
        
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None

def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def search_ticker(query):
    try:
        query = query.strip().upper()
        ticker = yf.Ticker(query)
        info = ticker.info
        
        if info and 'longName' in info:
            return [{
                'symbol': query,
                'name': info.get('longName', 'N/A'),
                'sector': info.get('sector', 'N/A'),
                'industry': info.get('industry', 'N/A'),
                'current_price': info.get('regularMarketPrice', 
                               info.get('currentPrice', 'N/A'))
            }]
    except Exception as e:
        print(f"Error in search: {e}")
    return []

def save_data(data, filename, directory='data'):
    create_directory(directory)
    filepath = os.path.join(directory, filename)
    data.to_csv(filepath)
    print(f"Data saved to {filepath}")