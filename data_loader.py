import pandas as pd
import numpy as np
import yfinance as yf

def fetch_portfolio_data(tickers, start, end):
    data = yf.download(tickers, start=start, end=end)['Adj close']
    log_returns = np.log(data/data.shift(1)).dropna()
    return log_returns

def get_portfolio_stats(log_returns):
    mu = log_returns.mean()
    cov_matrix = log_returns.cov()
    return mu, cov_matrix

