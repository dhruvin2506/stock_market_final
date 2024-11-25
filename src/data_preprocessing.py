import numpy as np
from sklearn.preprocessing import MinMaxScaler

class DataPreprocessor:
    def __init__(self):
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        
    def prepare_sequences(self, data, sequence_length=60):
        """
        Prepare sequences for LSTM model.
        
        Args:
            data (np.array): Input data
            sequence_length (int): Number of time steps in each sequence
            
        Returns:
            tuple: (X, y) where X is the input sequences and y is the target values
        """
        # Scale the data
        scaled_data = self.scaler.fit_transform(data)
        
        X = []
        y = []
        
        for i in range(sequence_length, len(scaled_data)):
            X.append(scaled_data[i-sequence_length:i, 0])
            y.append(scaled_data[i, 0])
            
        X = np.array(X)
        y = np.array(y)
        
        # Reshape X to be [samples, time steps, features]
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        
        return X, y
    
    def split_data(self, X, y, train_size=0.8):
        """
        Split data into training and testing sets.
        
        Args:
            X (np.array): Input sequences
            y (np.array): Target values
            train_size (float): Proportion of data to use for training
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        split_idx = int(len(X) * train_size)
        
        X_train = X[:split_idx]
        X_test = X[split_idx:]
        y_train = y[:split_idx]
        y_test = y[split_idx:]
        
        return X_train, X_test, y_train, y_test
    
    def inverse_transform(self, data):
        """Convert scaled values back to original scale."""
        if len(data.shape) == 1:
            data = data.reshape(-1, 1)
        return self.scaler.inverse_transform(data)