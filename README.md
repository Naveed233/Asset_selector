Portfolio Optimization App

Overview

This project is a Streamlit-based Portfolio Optimization Web App that enables users to:

Fetch historical stock data from Yahoo Finance.

Perform portfolio optimization using Sharpe Ratio Maximization and Risk Parity.

Analyze portfolio risk metrics including VaR, CVaR, Maximum Drawdown, and HHI.

Cluster stocks using KMeans Clustering and visualize them with PCA.

Fetch latest news related to selected stocks from NewsAPI.

Display optimized portfolio weights, risk metrics, and asset clusters with interactive Plotly charts.

Features

1. Data Fetching & Processing

Retrieves adjusted closing prices for user-specified stocks.

Computes daily returns.

Applies Principal Component Analysis (PCA) to denoise returns.

2. Portfolio Optimization

Sharpe Ratio Maximization: Finds the optimal allocation that maximizes risk-adjusted returns.

Risk Parity: Allocates weights so that all assets contribute equally to portfolio risk.

3. Risk Metrics Calculation

Annual Return & Volatility

Sharpe Ratio

Value at Risk (VaR)

Conditional Value at Risk (CVaR)

Maximum Drawdown (MDD)

Herfindahl-Hirschman Index (HHI) for Diversification

4. Stock Clustering & Visualization

KMeans clustering groups stocks based on return profiles.

PCA visualization shows asset groupings.

5. News Fetching

Fetches top 3 recent news articles for each stock from NewsAPI.

6. Streamlit UI

User Inputs: Stock tickers, date range, risk-free rate, optimization method.

Interactive Visuals:

Portfolio weights (Pie Chart)

Risk Metrics (Bar Chart)

Asset Clusters (Scatter Plot)

Error Handling: Checks for missing data and invalid tickers.

Installation

Prerequisites

Ensure you have Python installed along with the required dependencies.

Step 1: Clone the Repository

git clone https://github.com/yourusername/portfolio-optimization-app.git
cd portfolio-optimization-app

Step 2: Install Dependencies

pip install -r requirements.txt

Step 3: Set Up NewsAPI Key

Replace YOUR_NEWSAPI_KEY in the PortfolioOptimizer class with your NewsAPI key.

Step 4: Run the App

streamlit run app.py

Dependencies

streamlit
yfinance
pandas
numpy
scipy
plotly
scikit-learn
requests

Usage

Enter Stock Symbols (comma-separated, e.g., AAPL, MSFT, TSLA).

Select Start & End Date for historical data.

Input Risk-Free Rate (default: 2%).

Choose Optimization Method: Maximize Sharpe Ratio or Risk Parity.

Click 'Optimize Portfolio' to generate results.

View results: Portfolio weights, risk metrics, stock clusters, and news.

Future Enhancements

Integrate Monte Carlo simulations for stress testing.

Add customizable risk constraints.

Enhance news sentiment analysis.

License

This project is open-source under the MIT License.

Author
Maqbool Naveed Ahmed
[GitHub Profile](https://github.com/Naveed233/)
