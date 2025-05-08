import yfinance as yf
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

# Scaricare i dati storici dell'S&P 500
ticker = '^GSPC'
start_date = '2015-01-01'
end_date = '2025-01-01'
data = yf.download(ticker, start=start_date, end=end_date)

# Creazione di feature ingegnerizzate
def calculate_rsi(data, window=14):
    delta = data['Adj Close'].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(data, short_window=12, long_window=26, signal_window=9):
    short_ema = data['Adj Close'].ewm(span=short_window, adjust=False).mean()
    long_ema = data['Adj Close'].ewm(span=long_window, adjust=False).mean()
    macd = short_ema - long_ema
    signal = macd.ewm(span=signal_window, adjust=False).mean()
    return macd, signal

# Aggiunta delle feature
data['RSI'] = calculate_rsi(data)
data['MACD'], data['Signal Line'] = calculate_macd(data)
data['MA50'] = data['Adj Close'].rolling(window=50).mean()
data['MA200'] = data['Adj Close'].rolling(window=200).mean()

# Pulizia dei dati
data.dropna(inplace=True)

# Analisi esplorativa dei dati
plt.figure(figsize=(12,6))
plt.plot(data['Adj Close'], label='Prezzo di Chiusura')
plt.title('Andamento Storico S&P 500')
plt.legend()
plt.show()

# Istogramma delle feature
fig, axes = plt.subplots(2, 2, figsize=(12,8))
sns.histplot(data['RSI'], bins=50, kde=True, ax=axes[0,0]).set_title('RSI Distribution')
sns.histplot(data['MACD'], bins=50, kde=True, ax=axes[0,1]).set_title('MACD Distribution')
sns.histplot(data['MA50'], bins=50, kde=True, ax=axes[1,0]).set_title('MA50 Distribution')
sns.histplot(data['MA200'], bins=50, kde=True, ax=axes[1,1]).set_title('MA200 Distribution')
plt.tight_layout()
plt.show()

# Matrice di correlazione
plt.figure(figsize=(10,6))
sns.heatmap(data[['Adj Close', 'RSI', 'MACD', 'Signal Line', 'MA50', 'MA200']].corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Matrice di Correlazione')
plt.show()

# Normalizzazione dei dati
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data[['Adj Close', 'RSI', 'MACD', 'Signal Line', 'MA50', 'MA200']])

# Creazione del dataset per PyTorch
def create_sequences(data, seq_length=50):
    sequences = []
    labels = []
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i+seq_length])
        labels.append(data[i+seq_length, 0])  # Prezzo di chiusura come target
    return np.array(sequences), np.array(labels)

seq_length = 50
X, y = create_sequences(data_scaled, seq_length)

# Conversione in tensori PyTorch
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

print("Dataset preparato! Shape X:", X_tensor.shape, "Shape y:", y_tensor.shape)
