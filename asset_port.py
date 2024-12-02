import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import cvxpy as cp
import matplotlib.pyplot as plt
import io
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Portfolio Optimizer with Asset Selection
class PortfolioOptimizer:
    def __init__(self, tickers, start_date, end_date, risk_free_rate=0.02):
        # Initialize with user-specified parameters
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.risk_free_rate = risk_free_rate
        self.returns = None

    def fetch_data(self):
        # Fetch historical price data and calculate daily returns
        data = yf.download(
            self.tickers, start=self.start_date, end=self.end_date, progress=False
        )["Adj Close"]
        if data.empty:
            raise ValueError("No data fetched. Please check the tickers and date range.")
        self.returns = data.pct_change().dropna()

    def denoise_returns(self):
        # Reduce noise in returns data using Principal Component Analysis (PCA)
        num_assets = len(self.returns.columns)
        n_components = min(num_assets, len(self.returns))
        pca = PCA(n_components=n_components)
        pca_returns = pca.fit_transform(self.returns)
        denoised_returns = pca.inverse_transform(pca_returns)
        self.returns = pd.DataFrame(
            denoised_returns, index=self.returns.index, columns=self.returns.columns
        )

    def cluster_assets(self, n_clusters=3):
        # Group similar assets into clusters using KMeans clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(self.returns.T)
        return clusters

    def portfolio_stats(self, weights):
        # Calculate portfolio return, volatility, and Sharpe ratio
        weights = np.array(weights).flatten()
        num_assets = len(self.returns.columns)
        if weights.shape[0] != num_assets:
            raise ValueError(
                f"Weights dimension {weights.shape[0]} does not match number of assets {num_assets}"
            )

        mean_returns = self.returns.mean().values
        cov_matrix = self.returns.cov().values * 252  # Annualize covariance
        portfolio_return = np.dot(weights, mean_returns) * 252
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_volatility
        return portfolio_return, portfolio_volatility, sharpe_ratio

    def min_volatility_with_selection(self, target_return, num_assets_to_select):
        # Optimize portfolio to achieve minimum volatility with asset selection
        num_assets = len(self.returns.columns)
        mean_returns = self.returns.mean().values * 252  # Annualized returns
        cov_matrix = self.returns.cov().values * 252  # Annualized covariance

        # Define variables
        w = cp.Variable(num_assets)
        y = cp.Variable(num_assets, boolean=True)

        # Objective: Minimize portfolio variance
        portfolio_variance = cp.quad_form(w, cov_matrix)
        objective = cp.Minimize(portfolio_variance)

        # Constraints
        constraints = [
            cp.sum(w) == 1,  # Weights sum to 1
            w >= 0,          # No short selling
            w <= y,          # If y_i is 0, w_i must be 0
            cp.sum(y) == num_assets_to_select,  # Select exactly x assets
            w @ mean_returns >= target_return,  # Target return constraint
        ]

        problem = cp.Problem(objective, constraints)

        # Solve the problem
        try:
            problem.solve(solver=cp.ECOS_BB)
        except cp.error.SolverError:
            raise ValueError("Optimization did not converge. Try adjusting the target return or number of assets.")

        if problem.status != cp.OPTIMAL:
            raise ValueError("Optimization did not find an optimal solution.")

        optimal_weights = w.value
        return optimal_weights

    def backtest_portfolio(self, weights):
        # Evaluate portfolio performance over historical data
        weighted_returns = (self.returns * weights).sum(axis=1)
        cumulative_returns = (1 + weighted_returns).cumprod()
        return cumulative_returns

# Streamlit App to interact with the user
if __name__ == "__main__":
    st.title(
        "Portfolio Optimization with Asset Selection"
    )

    # User inputs
    # Define a larger asset universe (e.g., S&P 100 tickers)
    universe_options = {
        'Tech Giants': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'ADBE', 'INTC', 'CSCO'],
        'Finance Leaders': ['JPM', 'BAC', 'WFC', 'C', 'GS', 'MS', 'AXP', 'BLK', 'SCHW', 'USB'],
        'Healthcare Majors': ['JNJ', 'PFE', 'UNH', 'MRK', 'ABBV', 'ABT', 'TMO', 'MDT', 'DHR', 'BMY'],
        'Custom': []
    }

    universe_choice = st.selectbox(
        "Select an asset universe:",
        options=list(universe_options.keys()),
        index=0
    )

    if universe_choice == 'Custom':
        custom_tickers = st.text_input(
            "Enter stock tickers separated by commas (e.g., AAPL, MSFT, TSLA):"
        )
        ticker_list = [ticker.strip() for ticker in custom_tickers.split(",") if ticker.strip()]
        if not ticker_list:
            st.error("Please enter at least one ticker.")
            st.stop()
    else:
        ticker_list = universe_options[universe_choice]

    start_date = st.date_input("Start date", value=pd.to_datetime("2018-01-01"))
    end_date = st.date_input("End date", value=pd.to_datetime("2023-12-31"))
    risk_free_rate = (
        st.number_input(
            "Enter the risk-free rate (in %):",
            value=2.0,
            step=0.1,
        )
        / 100
    )

    num_assets_to_select = st.number_input(
        "Number of assets to include in the portfolio:",
        min_value=1,
        max_value=len(ticker_list),
        value=3,
        step=1
    )

    optimize_button = st.button("Optimize Portfolio")

    if optimize_button:
        try:
            # Validate dates
            if start_date >= end_date:
                st.error("Start date must be earlier than end date.")
                st.stop()

            optimizer = PortfolioOptimizer(
                ticker_list, start_date, end_date, risk_free_rate
            )
            optimizer.fetch_data()

            # Apply denoising to the returns data
            optimizer.denoise_returns()

            # Calculate annualized returns using geometric mean
            cumulative_returns = (1 + optimizer.returns).prod() - 1
            num_years = (end_date - start_date).days / 365.25
            annualized_returns = (1 + cumulative_returns) ** (1 / num_years) - 1

            min_return = annualized_returns.min() * 100  # Convert to percentage
            max_return = annualized_returns.max() * 100  # Convert to percentage

            # Adjust min and max if they are equal
            if min_return == max_return:
                min_return -= 5
                max_return += 5

            # Define the target return slider dynamically
            specific_target_return = (
                st.slider(
                    "Select a specific target return (in %)",
                    min_value=round(min_return, 2),
                    max_value=round(max_return, 2),
                    value=round(min_return, 2),
                    step=0.1,
                )
                / 100
            )

            # Adjust the target return validation
            tolerance = 1e-6
            if (
                specific_target_return < (min_return / 100 - tolerance)
                or specific_target_return > (max_return / 100 + tolerance)
            ):
                st.error(
                    f"The target return must be between {min_return:.2f}% and {max_return:.2f}%."
                )
                st.stop()

            # Optimize the portfolio for the user's specific target return and asset count
            optimal_weights = optimizer.min_volatility_with_selection(
                specific_target_return, num_assets_to_select
            )

            # Get portfolio stats
            portfolio_return, portfolio_volatility, sharpe_ratio = optimizer.portfolio_stats(
                optimal_weights
            )

            # Display the optimal portfolio allocation
            allocation = pd.DataFrame(
                {
                    "Asset": optimizer.returns.columns,
                    "Weight": optimal_weights.round(4),
                }
            )
            allocation = allocation[allocation['Weight'] > 0]

            st.subheader(
                f"Optimal Portfolio Allocation (Target Return: {specific_target_return*100:.2f}%)"
            )
            st.write(allocation)

            # Show portfolio performance metrics
            st.write("Portfolio Performance Metrics:")
            st.write(f"Expected Annual Return: {portfolio_return * 100:.2f}%")
            st.write(f"Annual Volatility (Risk): {portfolio_volatility * 100:.2f}%")
            st.write(f"Sharpe Ratio: {sharpe_ratio:.2f}")

            # Backtest the portfolio and display cumulative returns
            st.subheader("Backtest Portfolio Performance")
            cumulative_returns = optimizer.backtest_portfolio(optimal_weights)
            fig, ax = plt.subplots(figsize=(10, 6))
            plt.plot(
                cumulative_returns.index,
                cumulative_returns.values,
                label="Portfolio Cumulative Returns",
            )
            plt.xlabel("Date")
            plt.ylabel("Cumulative Return")
            plt.title("Portfolio Backtesting Performance")
            plt.legend()
            st.pyplot(fig)

            # Show a bar chart of the portfolio allocation
            st.subheader("Portfolio Allocation")
            fig, ax = plt.subplots()
            allocation.set_index("Asset").plot(kind="bar", y="Weight", legend=False, ax=ax)
            plt.ylabel("Weight")
            st.pyplot(fig)

            # Provide an option to download the portfolio allocation
            st.subheader("Download Portfolio Allocation and Metrics")
            buffer = io.StringIO()
            allocation.to_csv(buffer, index=False)
            st.download_button(
                label="Download Portfolio Allocation (CSV)",
                data=buffer.getvalue(),
                file_name="portfolio_allocation.csv",
                mime="text/csv",
            )

        except Exception as e:
            st.error(f"An error occurred: {e}")
