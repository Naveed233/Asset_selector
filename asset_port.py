import streamlit as st
from alpha_vantage.timeseries import TimeSeries
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models, expected_returns
import requests
from datetime import datetime
import plotly.graph_objects as go

# Initialize Alpha Vantage API
API_KEY = 'GVRDOJPM18JD9YDK'
ts = TimeSeries(key=API_KEY, output_format='pandas')

# Authentication Setup (Placeholder for streamlit-authenticator or other library)
def authenticate_user():
    users = {
        "user1": {"name": "Alice", "password": "password1"},
        "user2": {"name": "Bob", "password": "password2"}
    }
    user = st.sidebar.text_input("Username")
    pwd = st.sidebar.text_input("Password", type="password")
    if st.sidebar.button("Login"):
        if user in users and users[user]["password"] == pwd:
            return True
        else:
            st.sidebar.error("Incorrect username or password")
    return False

@st.cache
def get_stock_data(symbol):
    data, _ = ts.get_daily_adjusted(symbol, outputsize='full')
    return data['5. adjusted close'].dropna()

def optimize_portfolio(assets, weight_bounds=(0, 1)):
    prices = pd.DataFrame({asset: get_stock_data(asset) for asset in assets})
    mu = expected_returns.mean_historical_return(prices)
    S = risk_models.sample_cov(prices)
    ef = EfficientFrontier(mu, S, weight_bounds=weight_bounds)
    weights = ef.max_sharpe()
    cleaned_weights = ef.clean_weights()
    performance = ef.portfolio_performance(verbose=True)
    return cleaned_weights, performance

def fetch_news(symbol):
    url = f"https://newsapi.org/v2/everything?q={symbol}&apiKey=c1b710a8638d4e55ab8ec4415e97388a"
    response = requests.get(url)
    return response.json()['articles'][:3]

# Main UI
if authenticate_user():
    st.title("Advanced Portfolio Optimizer")
    st.sidebar.header("Portfolio Settings")
    assets = st.sidebar.multiselect("Select Assets:", ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'FB'])
    weight_bounds = st.sidebar.slider("Weight Bounds", 0.0, 1.0, (0.01, 0.99))

    if st.sidebar.button("Optimize"):
        weights, performance = optimize_portfolio(assets, weight_bounds=(weight_bounds[0], weight_bounds[1]))
        st.write("Optimized Weights:", weights)
        st.write("Performance Metrics:", performance)

    for asset in assets:
        st.subheader(f"{asset} Historical Data")
        data = get_stock_data(asset)
        st.line_chart(data)

        st.subheader(f"Latest News for {asset}")
        articles = fetch_news(asset)
        for article in articles:
            st.markdown(f"[{article['title']}]({article['url']})")
