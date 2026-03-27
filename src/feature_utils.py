import numpy as np
import pandas as pd
import datetime
import yfinance as yf
import pandas_datareader.data as web
import requests

# Define these once, above the function
ccy_tickers = ['DEXUSUK', 'DEXUSEU']
idx_tickers = ['SP500', 'DCOILWTICO']


def extract_features_pair():
    return_period = 5

    START_DATE = (datetime.date.today() - datetime.timedelta(days=365)).strftime("%Y-%m-%d")
    END_DATE = datetime.date.today().strftime("%Y-%m-%d")
    stk_tickers = ['ADBE', 'DPZ']

    stk_data = yf.download(stk_tickers, start=START_DATE, end=END_DATE, auto_adjust=False)

    ccy_data = web.DataReader(ccy_tickers, 'fred', start=START_DATE, end=END_DATE)
    idx_data = web.DataReader(idx_tickers, 'fred', start=START_DATE, end=END_DATE)

    Y = np.log(stk_data.loc[:, ('Adj Close', 'ADBE')]).diff(return_period).shift(-return_period)
    Y.name = 'ADBE_Future'

    X1 = np.log(stk_data.loc[:, ('Adj Close', 'DPZ')]).diff(return_period)
    X1.name = 'DPZ'

    X2 = np.log(ccy_data).diff(return_period)
    X3 = np.log(idx_data).diff(return_period)

    X = pd.concat([X1, X2, X3], axis=1)

    dataset = pd.concat([Y, X], axis=1).dropna().iloc[::return_period, :]
    dataset.index.name = 'Date'

    features = dataset.sort_index().reset_index(drop=True)
    features = features.iloc[:, 1:]

    return features


def get_bitcoin_historical_prices(days=60):
    BASE_URL = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"

    params = {
        'vs_currency': 'usd',
        'days': days,
        'interval': 'daily'
    }

    response = requests.get(BASE_URL, params=params)
    data = response.json()
    prices = data['prices']

    df = pd.DataFrame(prices, columns=['Timestamp', 'Close Price (USD)'])
    df['Date'] = pd.to_datetime(df['Timestamp'], unit='ms').dt.normalize()
    df = df[['Date', 'Close Price (USD)']].set_index('Date')

    return df

