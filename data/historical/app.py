import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
from pandas.plotting import register_matplotlib_converters
from statsmodels.tsa.vector_ar.var_model import VAR

register_matplotlib_converters()
import warnings
warnings.filterwarnings("ignore")

from bayesian import run_bayesian as bayesian_function
from maybe import run_maybe as maybe_function
from sarimax import run_sarimax as sarimax_function
from polyreg import poly_reg as polyreg_function
from auto_arima import auto_arima as auto_arima_function

def run_var():
    # Fetching data from the server
    url = "https://web-api.coinmarketcap.com/v1/cryptocurrency/ohlcv/historical"
    param = {"convert":"USD","slug":"bitcoin","time_end":"1601510400","time_start":"1367107200"}
    content = requests.get(url=url, params=param).json()
    df = pd.json_normalize(content['data']['quotes'])

    # Extracting and renaming the important variables
    df['Date']=pd.to_datetime(df['quote.USD.timestamp']).dt.tz_localize(None)
    df['Low'] = df['quote.USD.low']
    df['High'] = df['quote.USD.high']
    df['Open'] = df['quote.USD.open']
    df['Close'] = df['quote.USD.close']
    df['Volume'] = df['quote.USD.volume']

    # Drop original and redundant columns
    df=df.drop(columns=['time_open','time_close','time_high','time_low', 'quote.USD.low', 'quote.USD.high', 'quote.USD.open', 'quote.USD.close', 'quote.USD.volume', 'quote.USD.market_cap', 'quote.USD.timestamp'])

    # Creating a new feature for better representing day-wise values
    df['Mean'] = (df['Low'] + df['High'])/2

    # Cleaning the data for any NaN or Null fields
    df = df.dropna()

    # Creating a copy for making small changes
    dataset_for_prediction = df.copy()
    dataset_for_prediction['Actual']=dataset_for_prediction['Mean'].shift()
    dataset_for_prediction=dataset_for_prediction.dropna()

    # date time typecast
    dataset_for_prediction['Date'] =pd.to_datetime(dataset_for_prediction['Date'])
    dataset_for_prediction.index= dataset_for_prediction['Date']

    #predictiion
    data=df[['Mean','Close']]
    data=np.array(data,dtype='float32')
    data=data[:2500]

    #Exogeous variables
    exo=df[['Open']]
    exo=np.array(exo,dtype='float32')
    exo=exo[:2500,:]
    model=VAR(data,exog=exo)
    x=np.array(df['Date'])
    model.index=x[:2500]
    result=model.fit()
    arr=np.array(df['Mean'])

    #test data
    N=200
    ap=arr[-N:]
    z=exo[-N:,:]
    a2=result.forecast(model.endog,N,z);
    act=a2[:,1:]

    #VAR model call
    st.write("VAR")
    fig, ax = plt.subplots()
    ax.plot(act,color='cyan',label='predicted')
    ax.plot(ap,label='actual')
    c=0
    for i in range(N):
       c+=(act[i]-ap[i])**2
    c/=N

    #print RMSE
    st.write(c**0.5)
    plt.xlabel('Days')
    plt.ylabel('Value')
    plt.legend()
    st.pyplot(fig)

    
def run_bayesian():
    bayesian_function()
    pass

def run_maybe():
    maybe_function()
    pass

def run_sarimax():
    sarimax_function()
    pass


def run_polyreg():
    polyreg_function()
    pass

def run_auto_arima():
    auto_arima_function()
    pass

# Create a sidebar for navigation
st.sidebar.title('Navigation')
options = st.sidebar.radio('Go to', ['Home', 'VAR', 'Maybe', 'SARIMAX', 'Bayesian', 'PolyReg', 'Auto-Arima'])

if options == 'Home':
    st.title('Home')
    st.write('Welcome to our project!')

elif options == 'VAR':
    st.title('VAR')
    if st.button('Start VAR'):
        run_var()
    if st.button('Stop VAR'):
        st.write('VAR stopped')

elif options == 'Maybe':
    st.title('Maybe')
    if st.button('Run Maybe'):
        maybe_function()

elif options == 'SARIMAX':
    st.title('SARIMAX')
    if st.button('Run SARIMAX'):
        fig = sarimax_function()
        st.pyplot(fig)

elif options == 'Bayesian':
    st.title('Bayesian')
    if st.button('Run Bayesian'):
        from bayesian import run_bayesian
        fig = run_bayesian()
        st.pyplot(fig)

elif options == 'PolyReg':
    st.title('PolyReg')
    if st.button('Run PolyReg'):
        fig = polyreg_function()
        st.pyplot(fig)

elif options == 'Auto-Arima':
    st.title('Auto-Arima')
    if st.button('Run Auto-Arima'):
        fig = auto_arima_function()
        st.pyplot(fig)