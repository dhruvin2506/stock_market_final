import numpy as np
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup
import time

class NewsAnalyzer:
    def __init__(self):
        nltk.download('vader_lexicon', quiet=True)
        self.sia = SentimentIntensityAnalyzer()
        
    def get_stock_news(self, symbol, days=30):
        """Fetch news articles for a stock"""
        stock = yf.Ticker(symbol)
        news = stock.news
        
        news_data = []
        for article in news:
            news_data.append({
                'date': datetime.fromtimestamp(article['providerPublishTime']),
                'title': article['title'],
                'summary': article.get('summary', ''),
            })
            
        return pd.DataFrame(news_data)
    
    def analyze_sentiment(self, text):
        """Analyze sentiment of text"""
        scores = self.sia.polarity_scores(text)
        return scores['compound']
    
    def get_sentiment_signals(self, symbol, days=30):
        """Get sentiment-based trading signals"""
        news_df = self.get_stock_news(symbol, days)
        
        # Analyze sentiment of titles and summaries
        news_df['title_sentiment'] = news_df['title'].apply(self.analyze_sentiment)
        news_df['summary_sentiment'] = news_df['summary'].apply(self.analyze_sentiment)
        
        # Combined sentiment score
        news_df['sentiment_score'] = (news_df['title_sentiment'] + news_df['summary_sentiment']) / 2
        
        # Generate daily sentiment
        daily_sentiment = news_df.groupby(news_df['date'].dt.date)['sentiment_score'].mean()
        
        return daily_sentiment

class EnsembleLearning:
    def __init__(self, models):
        """Initialize with list of models"""
        self.models = models
        
    def train(self, X_train, y_train, X_val, y_val):
        """Train all models"""
        for model in self.models:
            model.train(X_train, y_train, X_val, y_val)
    
    def predict(self, X):
        """Get ensemble predictions"""
        predictions = []
        for model in self.models:
            pred = model.predict(X)
            predictions.append(pred)
            
        # Average predictions using numpy
        ensemble_pred = np.mean(predictions, axis=0)
        return ensemble_pred