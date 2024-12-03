import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, GRU, Bidirectional
from tensorflow.keras.optimizers import Adam
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from src.sentiment_analysis import EnsembleLearning

class LSTMModel:
    def __init__(self, sequence_length):
        self.model = Sequential([
            LSTM(100, return_sequences=True, input_shape=(sequence_length, 1)),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        self.model.compile(optimizer='adam', loss='mean_squared_error')
    
    def train(self, X_train, y_train, X_val, y_val):
        self.model.fit(X_train, y_train, validation_data=(X_val, y_val),
                      epochs=50, batch_size=32, verbose=1)
    
    def predict(self, X):
        return self.model.predict(X)

class BiLSTMModel:
    def __init__(self, sequence_length):
        self.model = Sequential([
            Bidirectional(LSTM(100, return_sequences=True), 
                         input_shape=(sequence_length, 1)),
            Dropout(0.2),
            Bidirectional(LSTM(50, return_sequences=False)),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        self.model.compile(optimizer='adam', loss='mean_squared_error')
    
    def train(self, X_train, y_train, X_val, y_val):
        self.model.fit(X_train, y_train, validation_data=(X_val, y_val),
                      epochs=50, batch_size=32, verbose=1)
    
    def predict(self, X):
        return self.model.predict(X)

class GRUModel:
    def __init__(self, sequence_length):
        self.model = Sequential([
            GRU(100, return_sequences=True, input_shape=(sequence_length, 1)),
            Dropout(0.2),
            GRU(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        self.model.compile(optimizer='adam', loss='mean_squared_error')
    
    def train(self, X_train, y_train, X_val, y_val):
        self.model.fit(X_train, y_train, validation_data=(X_val, y_val),
                      epochs=50, batch_size=32, verbose=1)
    
    def predict(self, X):
        return self.model.predict(X)

class EnhancedPredictor:
    def __init__(self, sequence_length, sentiment_analyzer):
        self.sequence_length = sequence_length
        self.sentiment_analyzer = sentiment_analyzer
        self.models = self._build_models()
        self.ensemble = EnsembleLearning(self.models)
        
    def _build_models(self):
        """Build all models for ensemble"""
        models = []
        models.append(LSTMModel(self.sequence_length))
        models.append(BiLSTMModel(self.sequence_length))
        models.append(GRUModel(self.sequence_length))
        return models
    
    def prepare_features(self, stock_data, symbol):
        """Prepare enhanced feature set including sentiment"""
        # Convert to numpy array if it's a pandas Series/DataFrame
        if isinstance(stock_data, (pd.Series, pd.DataFrame)):
            price_data = stock_data['Close'].values
        else:
            price_data = stock_data
            
        try:
            # Get sentiment data
            sentiment_scores = self.sentiment_analyzer.get_sentiment_signals(symbol)
            
            # Create a date index that matches the stock data
            dates = pd.date_range(start=stock_data.index[0], 
                                end=stock_data.index[-1], 
                                freq='B')
            
            # Align sentiment scores with stock data dates
            aligned_sentiment = pd.Series(index=dates, data=np.nan)
            aligned_sentiment.loc[sentiment_scores.index] = sentiment_scores
            aligned_sentiment.fillna(method='ffill', inplace=True)
            
            # Combine features
            features = np.column_stack([
                price_data,
                aligned_sentiment.values[:len(price_data)]
            ])
            
            return features
            
        except Exception as e:
            print(f"Error preparing features: {e}")
            # Fallback to using just price data if sentiment fails
            return price_data.reshape(-1, 1)
    
    def train(self, X_train, y_train, X_val, y_val):
        """Train the ensemble"""
        print("Training ensemble models...")
        try:
            self.ensemble.train(X_train, y_train, X_val, y_val)
            print("Ensemble training completed successfully")
        except Exception as e:
            print(f"Error during training: {e}")
            raise
    
    def predict(self, X):
        """Get ensemble predictions"""
        try:
            predictions = self.ensemble.predict(X)
            return predictions
        except Exception as e:
            print(f"Error during prediction: {e}")
            raise

    def evaluate_models(self, X_test, y_test):
        """Evaluate individual models and ensemble"""
        results = {}
        
        # Evaluate each model
        for i, model in enumerate(self.models):
            pred = model.predict(X_test)
            mse = np.mean((y_test - pred.flatten()) ** 2)
            results[f'Model_{i+1}_MSE'] = mse
        
        # Evaluate ensemble
        ensemble_pred = self.predict(X_test)
        ensemble_mse = np.mean((y_test - ensemble_pred.flatten()) ** 2)
        results['Ensemble_MSE'] = ensemble_mse
        
        return results