import matplotlib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import streamlit as st

def load_data(file_path):
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    df.sort_values('date', inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def visualize_data(date, close_price, figsize=(14, 7)):
    plt.figure(figsize=figsize)
    plt.plot(date, close_price, label='BTC Close Price')
    plt.title('BTC Close Price Over Time')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.legend()
    plt.show()

def feature_engineering(df):
    df['return'] = df['close'].pct_change()
    df['return_lag1'] = df['return'].shift(1)
    df['ma7'] = df['close'].rolling(window=7).mean()
    df['ma30'] = df['close'].rolling(window=30).mean()
    df['std30'] = df['close'].rolling(window=30).std()
    df.dropna(inplace=True)
    return df

def preprocess_data(df):
    X = df[['open', 'high', 'low', 'Volume_BTC', 'Volume_USD', 'return', 'return_lag1', 'ma7', 'ma30', 'std30']]
    y = df['close']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y, scaler

def split_data(X, y, test_size=0.2):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train, model=None, n_estimators=100, random_state=42):
    if model is None:
        model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f'Mean Squared Error: {mse}')

def evaluate_model_cv(model, X, y, cv=5):
    scores = cross_val_score(model, X, y, cv=cv, scoring='neg_mean_squared_error')
    mse_scores = -scores
    avg_mse = np.mean(mse_scores)
    print(f'Average Mean Squared Error (CV): {avg_mse}')

def predict_next_hour(model, scaler, X_last_row):
    last_row_scaled = scaler.transform(X_last_row.reshape(1, -1))
    next_hour_prediction = model.predict(last_row_scaled)
    return next_hour_prediction[0]

if __name__ == "__main__":

    st.write("AAAAAAAAAAAAAAAAAAAAAAAAAAA")
    st.image("xdxd.png")
    st.checkbox('yes')
    # Load data
    file_path = 'BTC-Hourly.csv'
    df = load_data(file_path)

    # Visualize data
    visualize_data(df['date'], df['close'])

    # Feature engineering
    df = feature_engineering(df)

    # Preprocess data
    X, y, scaler = preprocess_data(df)

    # Train/test split
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Train model
    model = train_model(X_train, y_train)

    # Evaluate model
    evaluate_model(model, X_test, y_test)
    evaluate_model_cv(model, X, y)

    # Prediction for the next hour
    next_hour_prediction = predict_next_hour(model, scaler, df.iloc[-1][['open', 'high', 'low', 'Volume_BTC', 'Volume_USD', 'return', 'return_lag1', 'ma7', 'ma30', 'std30']].values)
    print(f'Predicted BTC Close Price for the next hour: {next_hour_prediction}')
