import streamlit as st
import yfinance as yf
import pandas as pd
from scipy.optimize import minimize
import numpy as np
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import io

# Define the Portfolio Optimizer class
class PortfolioOptimizer:
    def __init__(self, tickers, start_date, end_date, risk_free_rate=0.02):
        self.tickers = tickers.split(',')
        self.start_date = start_date
        self.end_date = end_date
        self.risk_free_rate = risk_free_rate
        self.returns = None

    def fetch_data(self):
        data = yf.download(self.tickers, start=self.start_date, end=self.end_date, progress=False)["Adj Close"]
        self.returns = data.pct_change().dropna()

    def denoise_returns(self):
        pca = PCA(n_components=len(self.returns.columns))
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

    def min_volatility(self, target_return):
        num_assets = len(self.returns.columns)
        constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1},
                       {'type': 'eq', 'fun': lambda weights: self.portfolio_stats(weights)[0] - target_return})
        bounds = tuple((0, 1) for _ in num_assets)
        init_guess = num_assets * [1. / num_assets]
        result = minimize(lambda weights: self.portfolio_stats(weights)[1],
                          init_guess, method='SLSQP', bounds=bounds, constraints=constraints)
        return result.x

    def generate_efficient_frontier(self, target_returns):
        efficient_portfolios = []
        for ret in target_returns:
            weights = self.min_volatility(ret)
            _, portfolio_return, portfolio_volatility = self.portfolio_stats(weights)
            efficient_portfolios.append((portfolio_volatility, portfolio_return))
        return np.array(efficient_portfolios)

    def monte_carlo_simulation(self, num_simulations=10000):
        num_assets = len(self.returns.columns)
        results = np.zeros((3, num_simulations))
        weights_record = []

        for i in range(num_simulations):
            weights = np.random.dirichlet(np.ones(num_assets), size=1).flatten()
            results[:, i] = self.portfolio_stats(weights)
            weights_record.append(weights)

        return results, weights_record

    def backtest_portfolio(self, weights):
        weighted_returns = (self.returns * weights).sum(axis=1)
        cumulative_returns = (1 + weighted_returns).cumprod()
        return cumulative_returns

# Streamlit UI setup
if __name__ == "__main__":
    st.title("Portfolio Optimization with Advanced Features")

    # User Inputs
    tickers = st.text_input("Enter stock tickers separated by commas (e.g., AAPL, MSFT, TSLA):")
    start_date = st.date_input("Start date", value=pd.to_datetime("2023-01-01"))
    end_date = st.date_input("End date", value=pd.to_datetime("2023-12-31"))
    risk_free_rate = st.number_input("Enter the risk-free rate (in %)", value=2.0, step=0.1) / 100
    specific_target_return = st.slider("Select a specific target return (in %)", min_value=-5.0, max_value=30.0, value=15.0, step=0.1) / 100

    # Allow uploading custom datasets
    uploaded_file = st.file_uploader("Upload your dataset (CSV with columns as asset returns)", type="csv")

    if st.button("Optimize Portfolio"):
        try:
            optimizer = PortfolioOptimizer(tickers, start_date, end_date, risk_free_rate)
            if uploaded_file:
                user_data = pd.read_csv(uploaded_file, index_col=0)
                optimizer.returns = user_data
                st.success("Custom data loaded successfully!")
            else:
                optimizer.fetch_data()

            optimizer.denoise_returns()
            clusters = optimizer.cluster_assets()
            st.write("Clusters:", clusters)

            target_returns = np.linspace(0, specific_target_return, 50)
            frontier = optimizer.generate_efficient_frontier(target_returns)
            st.line_chart(frontier)

            results, _ = optimizer.monte_carlo_simulation()
            st.line_chart(results)

            weights = optimizer.min_volatility(specific_target_return)
            st.write("Optimal weights:", weights)

            cumulative_returns = optimizer.backtest_portfolio(weights)
            st.line_chart(cumulative_returns)

        except Exception as e:
            st.error(f"An error occurred: {e}")
