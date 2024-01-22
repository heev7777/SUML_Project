# Title: Bayesian Regression

## Introduction
We aim to predict Bitcoin prices using Bayesian Regression. We use historical data from the CoinMarketCap API, which includes features such as opening price, closing price, high, low, and volume.

## Data Collection
We fetch the data from the CoinMarketCap API. The data includes the following features:
- Date
- Low
- High
- Open
- Close
- Volume

We also create a new feature, 'Mean', which is the average of 'Low' and 'High' prices.

## Data Preprocessing
We clean the data by dropping any rows with NaN or Null fields. We also drop redundant columns that are not needed for our prediction model.

## Model Training
We use the Bayesian Regression model from the sklearn library. We train the model on the 'Open' price and predict the 'Mean' price. We use the first 2441 data points for training and the last 272 data points for testing.

## Model Evaluation
We evaluate the model by calculating the Root Mean Square Error (RMSE) between the predicted and actual 'Mean' prices.

![Bayesian Regression Image](https://i.imgur.com/NpZmGsx.png)

<div style="page-break-after: always;"></div>

# Title: Polynomial Regression

## Introduction
In this section, we aim to predict Bitcoin prices using Polynomial Regression. We use historical data from the CoinMarketCap API, which includes features such as opening price, closing price, high, low, and volume.

## Data Collection
We fetch the data from the CoinMarketCap API. The data includes the following features:
- Date
- Low
- High
- Open
- Close
- Volume

We also create a new feature, 'Mean', which is the average of 'Low' and 'High' prices.

## Data Preprocessing
We clean the data by dropping any rows with NaN or Null fields. We also drop redundant columns that are not needed for our prediction model.

## Model Training
We use the Polynomial Regression model from the sklearn library. We perform a grid search to find the optimal degree for the polynomial. We train the model on the 'Open' price and predict the 'Mean' price. We use the first 2411 data points for training and the last 272 data points for testing.

## Model Evaluation
We evaluate the model by calculating the Root Mean Square Error (RMSE) between the predicted and actual 'Mean' prices. We perform this evaluation for different degrees of the polynomial to find the optimal degree.

![Polynomial Regression Image](https://i.imgur.com/gUX0kJF.png)

<div style="page-break-after: always;"></div>

# Title: Vector Autoregression (VAR)

## Introduction
In this section, we aim to predict Bitcoin prices using Vector Autoregression (VAR). We use historical data from the CoinMarketCap API, which includes features such as opening price, closing price, high, low, and volume.

## Data Collection
We fetch the data from the CoinMarketCap API. The data includes the following features:
- Date
- Low
- High
- Open
- Close
- Volume

We also create a new feature, 'Mean', which is the average of 'Low' and 'High' prices.

## Data Preprocessing
We clean the data by dropping any rows with NaN or Null fields. We also drop redundant columns that are not needed for our prediction model.

## Model Training
We use the Vector Autoregression (VAR) model from the statsmodels library. We train the model on the 'Mean' and 'Close' prices, with 'Open' price as the exogenous variable. We use the first 2500 data points for training.

## Model Evaluation
We evaluate the model by calculating the Root Mean Square Error (RMSE) between the predicted and actual 'Mean' prices for the last 200 data points.

![Vector Autoregression](https://i.imgur.com/MFp54cH.png)

<div style="page-break-after: always;"></div>

# Title: Predicting Bitcoin Prices Using Auto ARIMA

## Introduction
We tried to predict Bitcoin prices using the Auto ARIMA model. We use historical data from the CoinMarketCap API, which includes features such as opening price, closing price, high, low, and volume.

## Data Collection
We fetch the data from the CoinMarketCap API. The data includes the following features:
- Date
- Low
- High
- Open
- Close
- Volume

We also create a new feature, 'Mean', which is the average of 'Low' and 'High' prices.

## Data Preprocessing
We clean the data by dropping any rows with NaN or Null fields. We also drop redundant columns that are not needed for our prediction model.

## Model Training
We use the Auto ARIMA model from the pmdarima library. We perform a grid search to find the optimal parameters for the ARIMA model. We train the model on the 'Mean' price, with 'Low', 'High', 'Open', 'Close', and 'Volume' as exogenous variables. We use the first 90% of data points for training.

## Model Evaluation
We evaluate the model by calculating the Root Mean Square Error (RMSE) between the predicted and actual 'Mean' prices for the last 10% of data points.

![Auto ARIMA](https://i.imgur.com/YLPrBPb.png)

<div style="page-break-after: always;"></div>

# Title: Predicting Bitcoin Prices Using SARIMAX

## Introduction
In this section, we aim to predict Bitcoin prices using the SARIMAX model. We use historical data from the CoinMarketCap API, which includes features such as opening price, closing price, high, low, and volume.

## Data Collection
We fetch the data from the CoinMarketCap API. The data includes the following features:
- Date
- Low
- High
- Open
- Close
- Volume

We also create a new feature, 'Mean', which is the average of 'Low' and 'High' prices.

## Data Preprocessing
We clean the data by dropping any rows with NaN or Null fields. We also drop redundant columns that are not needed for our prediction model. We then normalize the data using the MinMaxScaler from the sklearn library.

## Model Training
We use the SARIMAX model from the statsmodels library. We train the model on the 'Mean' price, with 'Low', 'High', 'Open', 'Close', and 'Volume' as exogenous variables. We use the first 90% of data points for training.

## Model Evaluation
We evaluate the model by calculating the Root Mean Square Error (RMSE) between the predicted and actual 'Mean' prices for the last 10% of data points.

![SARIMAX](https://i.imgur.com/mNcPbw8.png)

<div style="page-break-after: always;"></div>

# Title: Predicting Using Random Forest Regression

## Introduction
In this project, we aim to predict Bitcoin prices using the Random Forest Regression model. We use historical data from the CoinMarketCap API, which includes features such as opening price, closing price, high, low, and volume.

## Data Collection
We fetch the data from a CSV file named 'BTC-Hourly.csv'. The data includes the following features:
- Date
- Open
- High
- Low
- Close
- Volume_BTC
- Volume_USD

## Data Preprocessing
We clean the data by converting the 'date' column to datetime and sorting the data by date. We also create new features such as 'return', 'return_lag1', 'ma7', 'ma30', and 'std30'. We then drop any rows with NaN values.

## Model Training
We use the Random Forest Regressor model from the sklearn library. We scale the features using the StandardScaler from the sklearn library. We split the data into training and testing sets, with 80% of the data for training and 20% for testing. We train the model on the features and predict the 'Close' price.

## Model Evaluation
We evaluate the model by calculating the Mean Squared Error (MSE) between the predicted and actual 'Close' prices.

![Random Forest Regression](https://i.imgur.com/Bd0SVaJ.png)