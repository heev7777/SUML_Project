import matplotlib
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
import tkinter
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, LSTM
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import streamlit as st


# Load the data
df = pd.read_csv('BTC-Hourly.csv')

# Convert the date column to datetime
df['date'] = pd.to_datetime(df['date'])

# Sort the data by date
df.sort_values('date', inplace=True)

# Reset the index
df.reset_index(drop=True, inplace=True)

# Exploratory Data Analysis (EDA)
# Visualize the closing price over time
plt.figure(figsize=(14,7))
plt.plot(df['date'], df['close'], label='BTC Close Price')
plt.title('BTC Close Price Over Time')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.show()

# Feature Engineering
# Create new features like lagged returns, moving averages etc.
df['return'] = df['close'].pct_change() # Simple return
df['return_lag1'] = df['return'].shift(1) # Lagged return
df['ma7'] = df['close'].rolling(window=7).mean() # 7-period moving average
df['ma30'] = df['close'].rolling(window=30).mean() # 30-period moving average
df['std30'] = df['close'].rolling(window=30).std() # 30-period standard deviation

# Drop NA values generated by lag/rolling features
df.dropna(inplace=True)

# Features and target
X = df[['open', 'high', 'low', 'Volume_BTC', 'Volume_USD', 'return', 'return_lag1', 'ma7', 'ma30', 'std30']]
y = df['close']

# Scaling features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/test split
n_estimators = 100
random_state = 42
test_size = 0.2
# Note: For time series, it's important not to shuffle the data
model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, shuffle=False)
model.fit(X_train, y_train)

# Model prediction
y_pred = model.predict(X_test)

# Model Evaluation
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Prediction for the next hour
# We need to take the last row of features, scale it, and reshape it for the model input
last_row = scaler.transform(X.iloc[-1].values.reshape(1, -1))
next_hour_prediction = model.predict(last_row)

print(f'Predicted BTC Close Price for the next hour: {next_hour_prediction[0]}')
