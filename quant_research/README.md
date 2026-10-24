# Quant Research Platform

Implementation of the three-phase quantitative research framework from the paper:

> **"Alpha Trading Strategies Analysis, Maximizing Alpha for Hedge Funds, and High Frequency Econometrics for Analyzing Price Impact of Trades, Liquidity, and Market Microstructure"**
> 
> *Dr. Yogesh Malhotra, Global Risk Management Network, LLC (SSRN-id3306817)*

## Overview

This platform implements all three phases of the research framework:

### Phase 1: Alpha Trading Strategies Analysis
- Evaluation framework for 400+ trading strategies using 7 criteria
- Criteria: Alpha, Plausibility, Technical Profile, Operational Framework, Portfolio Risk, Market Crowding, Sector Capacity
- Strategy database with JSON persistence and ranking

### Phase 2: Post-Financial Crisis Liquidity Risk Model
- Position liquidity (market microstructure) and fund liquidity (redemption terms)
- Illiquidity premium estimation
- Liquidation cost modeling with urgency levels
- Stress testing with redemption shocks
- Macro liquidity indicators (credit spreads, bid-ask spreads, volume, yield curves, VIX)

### Phase 3: High-Frequency Econometrics & Market Microstructure
- Hasbrouck (1991) VAR model for price impact of trades
- Orthogonalized impulse response functions
- Spread decomposition: order processing, inventory, adverse selection
- VPIN (Volume-synchronized Probability of Informed Trading)
- Kyle's Lambda estimation
- HFT activity detection

## Installation

```bash
cd quant_research
pip install -r requirements.txt
```

## Usage

### CLI Commands

```bash
# Market microstructure analysis
python -m quant_research microstructure --symbol AAPL --date 2024-01-15

# Liquidity risk analysis
python -m quant_research liquidity --symbol AAPL --fund-aum 1e9 --redemption-days 30

# Alpha strategy evaluation
python -m quant_research strategy-eval --generate 200 --top 20

# Backtesting
python -m quant_research backtest --strategy momentum --symbol SPY --start 2020-01-01

# Hasbrouck VAR price impact
python -m quant_research hasbrouck --symbol AAPL --date 2024-01-15

# Full integrated analysis
python -m quant_research full-analysis --symbol AAPL --date 2024-01-15 --fund-aum 1e9

# Fetch historical data
python -m quant_research fetch-data --symbol AAPL --start 2023-01-01 --end 2023-12-31
```

### Python API

```python
from quant_research.microstructure.analyzer import MarketMicrostructureAnalyzer
from quant_research.risk.liquidity_model import LiquidityRiskModel, PositionLiquidity, FundLiquidityProfile
from quant_research.models.hasbrouck_var import PriceImpactAnalyzer, HasbrouckVARModel
from quant_research.strategies.evaluator import StrategyDatabase, create_sample_strategies
from quant_research.backtest.engine import BacktestEngine, MomentumStrategy, MarketImpactModel

# Market microstructure
analyzer = MarketMicrostructureAnalyzer()
results = analyzer.analyze(trades_df, quotes_df)

# Liquidity risk
model = LiquidityRiskModel()
premium = model.estimate_illiquidity_premium(position, fund)
stress = model.stress_test_liquidity(positions, fund, redemption_shock=0.2)

# Price impact (Hasbrouck VAR)
var_model = HasbrouckVARModel()
var_model.fit(var_data)
price_impact = var_model.get_price_impact()

# Strategy evaluation
db = StrategyDatabase()
for strat in create_sample_strategies(400):
    db.add_strategy(strat)
top_strategies = db.rank_strategies(50)

# Backtesting
engine = BacktestEngine(execution_model=MarketImpactModel())
result = engine.run(price_data, MomentumStrategy())
```

## Project Structure

```
quant_research/
├── config/           # Configuration management
├── data/             # TAQ data loading & preprocessing
├── models/           # Hasbrouck VAR model
├── strategies/       # Alpha strategy evaluation framework
├── risk/             # Liquidity risk models
├── microstructure/   # Market microstructure analysis
├── backtest/         # Backtesting engine
├── api/              # Data provider integrations
└── cli/              # Command-line interface
```

## Data Sources

Configured via `.env`:
- **Alpaca** - Paper trading execution
- **Polygon.io** - TAQ trades/quotes, aggregates
- **Finnhub** - Real-time quotes, news
- **FMP** - Fundamentals, SEC filings
- **Alpha Vantage** - Technical indicators

## Key Features

| Feature | Implementation |
|---------|---------------|
| Price Impact (Hasbrouck) | VAR model with OIRF, variance decomposition, Granger causality |
| Spread Decomposition | 3 components: order processing, inventory, adverse selection |
| VPIN | Volume-synchronized PIN for toxicity measurement |
| Kyle's Lambda | Rolling regression of price changes on signed volume |
| Illiquidity Premium | Combined position + fund liquidity scoring |
| Liquidation Cost | Market impact + spread cost with urgency multipliers |
| Strategy Scoring | 7-criteria weighted composite with configurable weights |
| Walk-Forward Optimization | Train/test splits with parameter grid search |

## References

- Hasbrouck, J. (1991). "Measuring the Information Content of Stock Trades." *Journal of Finance*
- Dufour, A. & Engle, R.F. (2000). "Time and the Price Impact of a Trade." *Journal of Finance*
- Kyle, A.S. (1985). "Continuous Auctions and Insider Trading." *Econometrica*
- Easley, D., et al. (2012). "Volume Synchronized Probability of Informed Trading." *Journal of Finance*

## License

Proprietary - Global Risk Management Network, LLC