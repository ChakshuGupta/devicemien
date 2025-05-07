import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler


class StackedLSTMAutoencoder:
    def __init__(self, timesteps, features, encoding_dim=32):
        self.timesteps = timesteps
        self.features = features
        self.encoding_dim = encoding_dim
        self.model = self._build_model()
        self.scaler = MinMaxScaler()

    def _build_model(self):
        input_seq = Input(shape=(self.timesteps, self.features))

        # Encoder: 3 stacked LSTM layers
        x = LSTM(128, activation='relu', return_sequences=True)(input_seq)
        x = LSTM(64, activation='relu', return_sequences=True)(x)
        encoded = LSTM(self.encoding_dim, activation='relu', return_sequences=False)(x)

        # Decoder: Dense layers to reconstruct original features from encoded vector
        x = Dense(64, activation='relu')(encoded)
        x = Dense(128, activation='relu')(x)
        decoded = Dense(self.timesteps * self.features, activation='sigmoid')(x)

        model = Model(inputs=input_seq, outputs=decoded)
        model.compile(optimizer=Adam(), loss='mse')
        return model

    def fit(self, X, epochs=20, batch_size=64, validation_split=0.2):
        X_scaled = self._scale(X)
        y = X_scaled.reshape(X_scaled.shape[0], -1)  # Flatten for comparison
        history = self.model.fit(
            X_scaled, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split
        )
        # self._plot_loss(history)
        return history

    def reconstruct(self, X):
        X_scaled = self._scale(X)
        y_pred = self.model.predict(X_scaled)
        return y_pred.reshape(X.shape)

    def _scale(self, X):
        original_shape = X.shape
        X_reshaped = X.reshape(-1, self.features)
        X_scaled = self.scaler.fit_transform(X_reshaped)
        return X_scaled.reshape(original_shape)

    def _plot_loss(self, history):
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Val Loss')
        plt.title('Training Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

    def plot_reconstruction(self, X):
        reconstructed = self.reconstruct(X)
        plt.figure(figsize=(10, 6))
        plt.plot(X[0].flatten(), label='Original')
        plt.plot(reconstructed[0].flatten(), label='Reconstructed', linestyle='dashed')
        plt.title('Original vs Reconstructed (Flattened Sequence)')
        plt.legend()
        plt.show()
