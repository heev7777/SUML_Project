import requests
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from datetime import datetime, timedelta

API_KEY = "C06C0BB9-CEE4-4328-92B6-BE96DA8155E2"
url_template = "https://rest.coinapi.io/v1/exchangerate/BTC/USD/history?period_id=1DAY&time_start={}&time_end={}"

def get_date_range():
    end_date = datetime.now()
    start_date = end_date - timedelta(days=100)
    return start_date.strftime('%Y-%m-%dT%H:%M:%S'), end_date.strftime('%Y-%m-%dT%H:%M:%S')

def fetch_btc_data(api_key, start_date, end_date):
    headers = {
        'X-CoinAPI-Key': api_key
    }
    url = url_template.format(start_date, end_date)
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    
    return response.json()

def preprocess_data(btc_data):
    df = pd.DataFrame(btc_data)

    # a. Cleaning the Data:

    # 1. Handle Missing Values:
    df.ffill(inplace=True) # df.fillna(method='ffill', inplace=True)

    # 2. Detect and Handle Outliers:
    Q1 = df['rate_close'].quantile(0.25)
    Q3 = df['rate_close'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df = df[(df['rate_close'] >= lower_bound) & (df['rate_close'] <= upper_bound)]

    # 3. Standardize 'rate_close':
    scaler = StandardScaler()
    df['rate_close_standardized'] = scaler.fit_transform(df[['rate_close']])

    # b. Feature Engineering:

    # 1. Moving Average:
    df['MA5'] = df['rate_close'].rolling(window=5).mean()

    # 2. RSI:
    delta = df['rate_close'].diff()
    gain = (delta.where(delta > 0, 0))
    loss = (-delta.where(delta < 0, 0))
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # 3. MACD:
    df['EMA12'] = df['rate_close'].ewm(span=12, adjust=False).mean()
    df['EMA26'] = df['rate_close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    return df

def plot_btc_data(df):
    # Set the index
    df['time_period_start'] = pd.to_datetime(df['time_period_start'])
    df.set_index('time_period_start', inplace=True)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(df.index, df['rate_close'], marker='o', linestyle='-', color='b')
    plt.title("BTC Daily Price")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

start_date, end_date = get_date_range()
print(f"Start date: {start_date}, End date: {end_date}")

btc_data = fetch_btc_data(API_KEY, start_date, end_date)
df = preprocess_data(btc_data)
plot_btc_data(df)