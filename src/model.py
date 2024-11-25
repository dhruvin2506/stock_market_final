import os
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

class StockPredictor:
    def __init__(self, sequence_length):
        self.sequence_length = sequence_length
        self.model = None
        
    def build_model(self, units=50, dropout_rate=0.2):
        """
        Build LSTM model architecture.
        
        Args:
            units (int): Number of LSTM units
            dropout_rate (float): Dropout rate for regularization
        """
        self.model = Sequential([
            LSTM(units=units, return_sequences=True, 
                 input_shape=(self.sequence_length, 1)),
            Dropout(dropout_rate),
            LSTM(units=units, return_sequences=False),
            Dropout(dropout_rate),
            Dense(units=25),
            Dense(units=1)
        ])
        
        self.model.compile(optimizer='adam', loss='mean_squared_error')
        print(self.model.summary())
        
    def train(self, X_train, y_train, X_val, y_val, 
              epochs=50, batch_size=32, save_dir='models'):
        """
        Train the LSTM model.
        
        Args:
            X_train (np.array): Training sequences
            y_train (np.array): Training targets
            X_val (np.array): Validation sequences
            y_val (np.array): Validation targets
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            save_dir (str): Directory to save model checkpoints
        """
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        # Define callbacks
        checkpoint = ModelCheckpoint(
            filepath=os.path.join(save_dir, 'best_model.h5'),
            monitor='val_loss',
            save_best_only=True,
            mode='min'
        )
        
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        # Train the model
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            callbacks=[checkpoint, early_stopping],
            verbose=1
        )
        
        return history
    
    def predict(self, X):
        """Make predictions using the trained model."""
        return self.model.predict(X)
    
    def save_model(self, filepath):
        """Save the trained model."""
        self.model.save(filepath)
        
    def load_model(self, filepath):
        """Load a trained model."""
        self.model = load_model(filepath)