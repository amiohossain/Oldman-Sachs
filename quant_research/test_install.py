from quant_research.config.settings import settings
from quant_research.data.taq_loader import create_sample_taq_data, TAQPreprocessor
from quant_research.microstructure.analyzer import MarketMicrostructureAnalyzer
from quant_research.risk.liquidity_model import run_liquidity_analysis
from quant_research.strategies.evaluator import run_strategy_analysis
from quant_research.backtest.engine import run_backtest_example

print('All imports successful!')

# Quick test
trades, quotes = create_sample_taq_data()
print(f'Sample data: {len(trades)} trades, {len(quotes)} quotes')

# Test microstructure
analyzer = MarketMicrostructureAnalyzer()
micro_results = analyzer.analyze(trades, quotes)
print(f'Microstructure: spread={micro_results["spread_components"]["total_spread"]:.4f}')

# Test liquidity
liq_results = run_liquidity_analysis()
print(f'Liquidity: premium={liq_results["illiquidity_premium"]["premium_bps"]:.1f}bps')

# Test strategies
strat_results = run_strategy_analysis()
print(f'Strategies: {strat_results["total_strategies"]} total, top score={strat_results["top_20"][0]["scores"]["composite"]:.3f}')

# Test backtest
bt_results = run_backtest_example()
print(f'Backtest: Sharpe={bt_results["sharpe_ratio"]:.3f}, Return={bt_results["annual_return"]:.2%}')

print('All tests passed!')