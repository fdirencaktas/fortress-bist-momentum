# BIST-100 Momentum & Mean Reversion Backtesting Framework

A comprehensive backtesting framework for evaluating momentum and mean reversion trading strategies on BIST-100 stocks (2018-2025). Developed as part of academic thesis research, this framework provides dual-currency analysis (TRY/USD), extensive performance metrics, and rigorous robustness checks.

## 📊 Features

### Core Capabilities
- **5 Trading Strategies**: DMA, Donchian Breakout, Combo Filtered, RSI, Bollinger Bands
- **Dual Currency Analysis**: Local (TRY) and International (USD) investor perspectives
- **Realistic Implementation**: Slippage (0.1%), interest on cash (25%), position limits (15 max)
- **Comprehensive Metrics**: Sharpe, Sortino, Calmar ratios, drawdown/drawup analysis
- **Trade-Level Statistics**: Win rate, profit factor, expectancy, turnover analysis

### Robustness Framework
- ✅ Bootstrap significance testing (10,000 iterations)
- ✅ T-test comparison with benchmarks
- ✅ Subperiod analysis (4 market regimes)
- ✅ Rolling 3-year Sharpe ratios
- ✅ Transaction cost sensitivity (0.05%, 0.10%, 0.20%)
- ✅ Parameter sensitivity for all strategies
- ✅ Macro correlations (currency, inflation, interest rates)

### Visualization
- 📈 Equity curves (TRY and USD)
- 📉 Drawdown analysis
- 📊 Correlation matrices
- 📋 Sector performance breakdown
- 📐 Parameter sensitivity plots

## 🚀 Quick Start

### Prerequisites
```bash
Python 3.8+
pip install pandas numpy yfinance matplotlib seaborn scipy
