import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import requests

# Portfolio Optimizer Class
class PortfolioOptimizer:
    def __init__(self, tickers, start_date, end_date, risk_free_rate=0.02):
        self.tickers = tickers.split(',')
        self.start_date = start_date
        self.end_date = end_date
        self.risk_free_rate = risk_free_rate / 100  # Convert percentage to decimal
        self.returns = None

    def fetch_data(self):
        data = yf.download(self.tickers, start=self.start_date, end=self.end_date, progress=False)["Adj Close"]
        self.returns = data.pct_change().dropna()

    def fetch_news(self):
        news_data = {}
        API_KEY = 'c1b710a8638d4e55ab8ec4415e97388a'  # Your NewsAPI key
        for ticker in self.tickers:
            url = f"https://newsapi.org/v2/everything?q={ticker}&apiKey={API_KEY}"
            response = requests.get(url)
            if response.status_code == 200:
                articles = response.json()['articles'][:3]  # Get the top 3 news articles
                news_data[ticker] = articles
            else:
                news_data[ticker] = []  # No news found or error in fetching
        return news_data

    def denoise_returns(self):
        pca = PCA(n_components=min(10, len(self.returns.columns)))  # Ensure PCA component count doesn't exceed column count
        pca_returns = pca.fit_transform(self.returns)
        explained_variance = pca.explained_variance_ratio_.cumsum()
        num_components = np.argmax(explained_variance >= 0.95) + 1
        denoised_returns = pca.inverse_transform(pca_returns[:, :num_components])
        self.returns = pd.DataFrame(denoised_returns, index=self.returns.index, columns=self.returns.columns)

    def cluster_assets(self, n_clusters=3):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(self.returns.T)
        return clusters

    def portfolio_stats(self, weights):
        annual_return = np.dot(weights, self.returns.mean()) * 252
        annual_volatility = np.sqrt(np.dot(weights.T, np.dot(self.returns.cov() * 252, weights)))
        sharpe_ratio = (annual_return - self.risk_free_rate) / annual_volatility
        return annual_return, annual_volatility, sharpe_ratio

# Streamlit UI
st.title("Portfolio Optimization with News Feature")

# User inputs for the optimization process
tickers = st.text_input("Enter stock tickers separated by commas (e.g., AAPL, MSFT, TSLA):")
start_date = st.date_input("Start date", value=pd.to_datetime("2023-01-01"))
end_date = st.date_input("End date", value=pd.to_datetime("2023-12-31"))
risk_free_rate = st.number_input("Enter the risk-free rate (in %)", value=2.0, step=0.1)

if st.button("Optimize Portfolio"):
    if not tickers:
        st.error("Please enter at least one ticker.")
    else:
        optimizer = PortfolioOptimizer(tickers, start_date, end_date, risk_free_rate)
        optimizer.fetch_data()
        optimizer.denoise_returns()
        news = optimizer.fetch_news()  # Fetch news for the tickers

        # Display news for each ticker
        for ticker, articles in news.items():
            if articles:
                st.subheader(f"Top News for {ticker}")
                for article in articles:
                    st.markdown(f"[{article['title']}]({article['url']}) - {article['source']['name']}")

        # Additional optimization and clustering as needed
        clusters = optimizer.cluster_assets()
        st.write("Clusters:", clusters)
