import requests
import pandas as pd

api_key = 'C06C0BB9-CEE4-4328-92B6-BE96DA8155E2'

def fetch_historical_data(api_key):
    """
    Fetch historical Bitcoin price data from CoinAPI.

    Parameters:
    - api_key (str): Your CoinAPI API key.

    Returns:
    pd.DataFrame: A DataFrame containing historical Bitcoin prices.
    """

    # Define the endpoint URL
    url = "https://rest.coinapi.io/v1/ohlcv/BITSTAMP_SPOT_BTC_USD/history"
    
    # Specify the parameters (You may modify according to your requirement)
    params = {
        'period_id': '1DAY',  # Change period as needed (1DAY, 1HRS, etc.)
        "time_exchange": "2023-10-13T09:36:10.058Z",
        "time_coinapi": "2023-10-13T09:36:10.058Z",
        'limit': 1000,  # Limit the number of returned results
    }
    
    # Include the API key in headers
    headers = {
        'X-CoinAPI-Key': api_key
    }
    
    # Make the API request and check the response
    response = requests.get(url, headers=headers, params=params)
    response.raise_for_status()  # Check if the request was successful
    
    # Convert the data to DataFrame
    data = response.json()
    df = pd.DataFrame(data)
    
    # Convert time_start and time_end to datetime format
    df['time_period_start'] = pd.to_datetime(df['time_period_start'])
    df['time_period_end'] = pd.to_datetime(df['time_period_end'])
    
    # Set the start time as index (optional)
    df.set_index('time_period_start', inplace=True)
    
    return df

def save_data(df, filename):
    """
    Save DataFrame to a CSV file.

    Parameters:
    - df (pd.DataFrame): DataFrame to be saved.
    - filename (str): The name of the output file.
    """
    df.to_csv(filename)