# Stochastic Portfolio Manager: Multi-stage Optimization under Uncertainty

This project implements a comprehensive framework for multi-stage stochastic portfolio optimization. It employs a hybrid computational approach, utilizing **Python** for financial data analysis and Monte Carlo scenario generation, and **Julia (JuMP)** for solving large-scale linear and non-linear optimization problems using the Gurobi solver.

The objective is to determine the optimal asset allocation strategy over a multi-period horizon (T=4 to T=6 semesters) to maximize expected wealth or meet specific financial targets while strictly managing risk through various mathematical formulations.

## Project Overview

The system operates in two distinct phases:

1.  **Market Simulation (Python):** Analysis of historical data and generation of correlated future market scenarios using Geometric Brownian Motion (GBM) and Cholesky decomposition.
2.  **Stochastic Optimization (Julia):** Construction of a scenario tree and execution of optimization solvers to suggest rebalancing decisions at each node of the tree.

## Methodological Approach

### 1. Scenario Generation
The Python module (`potfolio.py`) handles the stochastic simulation of asset returns.
* **Data Source:** Historical data is fetched via Yahoo Finance for a diverse basket of assets, including Equities (S&P 500 EW, Nasdaq 100), Fixed Income (Corp Bonds, Treasuries), Commodities (Gold, Uranium), and Cryptocurrencies (BTC, ETH).
* **Correlation Structure:** Asset correlations are preserved using Cholesky decomposition applied to the covariance matrix of logarithmic returns.
* **Discretization:** Continuous Monte Carlo paths are discretized into a scenario tree structure:
    * **Root Node:** 5 initial branches representing market regimes (Very Bearish to Very Bullish).
    * **Future Nodes:** Binary branching (High/Low) for subsequent periods.

### 2. Optimization Models
The project implements three distinct mathematical strategies using Julia:

* **Standard Stochastic Model (`Optimizacion_de_cartera.jl`):**
    Balances the trade-off between expected return and risk. It implements constraints for transaction costs, diversification limits, and specific exposure caps (e.g., maximum allocation to cryptocurrencies). Calculates Conditional Value at Risk (CVaR) for ex-post analysis.

* **Robust Optimization - Min-Max (`minmax.jl`):**
    Adopts a conservative approach by maximizing the wealth of the worst-case scenario across the entire tree. This model is designed to guarantee a minimum performance floor regardless of market realization.

* **Risk-Constrained Model - VaR (`var.jl`):**
    Focuses on the tail risk distribution. This formulation optimizes the portfolio while explicitly monitoring the Value at Risk (VaR) and ensuring that the probability of falling below a certain wealth threshold is minimized.

## Project Structure

* **potfolio.py**: Main Python script. Calculates descriptive statistics, runs Monte Carlo simulations, visualizes density plots, and exports the scenario trees to CSV format (`escenarios_semestrales.csv`, `escenarios_futuros_binarios.csv`).
* **Optimizacion_de_cartera.jl**: Primary optimization script implementing the hybrid scenario tree and general business constraints.
* **minmax.jl**: Implementation of the Robust (Min-Max) optimization strategy.
* **var.jl**: Implementation of the risk-focused strategy with emphasis on VaR and tail-risk metrics.

## Requirements

To replicate the results, the following software and libraries are required:

### Python
* numpy
* pandas
* yfinance
* seaborn
* matplotlib

### Julia
* JuMP
* Gurobi (Requires a valid license)
* DataFrames
* CSV
* Plots

## Usage Instructions

1.  **Generate Scenarios:**
    Run the Python script first to update market data and generate the scenario files.
    ```bash
    python potfolio.py
    ```
    This will create `escenarios_semestrales.csv` and `escenarios_futuros_binarios.csv`.

2.  **Run Optimization:**
    Execute the desired Julia model. Ensure the CSV files generated in the previous step are in the same directory.
    ```bash
    julia Optimizacion_de_cartera.jl
    ```
    Or for specific strategies:
    ```bash
    julia minmax.jl
    ```
