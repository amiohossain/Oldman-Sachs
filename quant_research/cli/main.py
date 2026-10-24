import argparse
import sys
import json
from pathlib import Path
from typing import Optional
import pandas as pd
import numpy as np


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Quant Research Platform - Alpha Strategies, Liquidity Risk, Market Microstructure",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s microstructure --symbol AAPL --date 2024-01-15
  %(prog)s liquidity --symbol AAPL --fund-aum 1e9 --redemption-days 30
  %(prog)s strategy-eval --input strategies.json --top 20
  %(prog)s backtest --strategy momentum --symbol SPY --start 2020-01-01
  %(prog)s hasbrouck --symbol AAPL --date 2024-01-15
  %(prog)s full-analysis --symbol AAPL --date 2024-01-15
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Microstructure analysis
    micro = subparsers.add_parser("microstructure", help="Market microstructure analysis")
    micro.add_argument("--symbol", required=True, help="Stock symbol")
    micro.add_argument("--date", required=True, help="Date (YYYY-MM-DD)")
    micro.add_argument("--source", choices=["local", "polygon"], default="polygon")
    micro.add_argument("--output", help="Output JSON file")
    
    # Liquidity risk
    liq = subparsers.add_parser("liquidity", help="Liquidity risk analysis")
    liq.add_argument("--symbol", required=True, help="Stock symbol")
    liq.add_argument("--fund-aum", type=float, default=1e9, help="Fund AUM")
    liq.add_argument("--redemption-days", type=int, default=30, help="Redemption notice days")
    liq.add_argument("--lockup-months", type=int, default=0, help="Lockup period in months")
    liq.add_argument("--fund-age-months", type=int, default=36, help="Fund age in months")
    liq.add_argument("--redemption-frequency", choices=["daily", "weekly", "monthly", "quarterly"], default="monthly")
    liq.add_argument("--output", help="Output JSON file")
    
    # Strategy evaluation
    strat = subparsers.add_parser("strategy-eval", help="Alpha strategy evaluation")
    strat.add_argument("--input", help="Input JSON file with strategies")
    strat.add_argument("--generate", type=int, default=200, help="Generate N sample strategies")
    strat.add_argument("--top", type=int, default=20, help="Show top N strategies")
    strat.add_argument("--category", help="Filter by category")
    strat.add_argument("--min-score", type=float, help="Minimum composite score")
    strat.add_argument("--output", help="Output JSON file")
    
    # Backtesting
    bt = subparsers.add_parser("backtest", help="Run backtest")
    bt.add_argument("--strategy", choices=["momentum", "mean_reversion"], default="momentum")
    bt.add_argument("--symbol", default="SPY", help="Symbol to trade")
    bt.add_argument("--start", default="2020-01-01", help="Start date")
    bt.add_argument("--end", help="End date")
    bt.add_argument("--capital", type=float, default=1e6, help="Initial capital")
    bt.add_argument("--execution", choices=["simple", "market_impact"], default="market_impact")
    bt.add_argument("--output", help="Output JSON file")
    
    # Hasbrouck VAR
    hasb = subparsers.add_parser("hasbrouck", help="Hasbrouck VAR price impact analysis")
    hasb.add_argument("--symbol", required=True, help="Stock symbol")
    hasb.add_argument("--date", required=True, help="Date (YYYY-MM-DD)")
    hasb.add_argument("--max-lags", type=int, default=5, help="Maximum VAR lags")
    hasb.add_argument("--horizon", type=int, default=12, help="Impulse response horizon")
    hasb.add_argument("--output", help="Output JSON file")
    
    # Full integrated analysis
    full = subparsers.add_parser("full-analysis", help="Run complete integrated analysis")
    full.add_argument("--symbol", required=True, help="Stock symbol")
    full.add_argument("--date", required=True, help="Date (YYYY-MM-DD)")
    full.add_argument("--fund-aum", type=float, default=1e9, help="Fund AUM")
    full.add_argument("--output-dir", default="./results", help="Output directory")
    
    # Data fetching
    data = subparsers.add_parser("fetch-data", help="Fetch market data")
    data.add_argument("--symbol", required=True, help="Stock symbol")
    data.add_argument("--start", required=True, help="Start date")
    data.add_argument("--end", required=True, help="End date")
    data.add_argument("--source", choices=["polygon", "alpaca", "alphavantage"], default="polygon")
    data.add_argument("--output", help="Output CSV file")
    
    return parser


def run_microstructure(args):
    from quant_research.microstructure.analyzer import run_microstructure_analysis
    results = run_microstructure_analysis(args.symbol, args.date)
    output_results(results, args.output)
    print(f"Microstructure analysis complete for {args.symbol} on {args.date}")
    print(f"  Effective Spread: {results['microstructure_metrics']['effective_spread']:.4f}")
    print(f"  Price Impact: {results['microstructure_metrics']['price_impact']:.6f}")
    print(f"  VPIN: {results['microstructure_metrics']['vpin']:.4f}")


def run_liquidity(args):
    from quant_research.risk.liquidity_model import (
        PositionLiquidity, FundLiquidityProfile, LiquidityRiskModel, run_liquidity_analysis
    )
    
    position = PositionLiquidity(
        symbol=args.symbol,
        avg_daily_volume=50_000_000,
        bid_ask_spread_bps=2,
        market_depth=10_000_000,
        days_to_liquidate=0.5,
        price_impact_1pct=0.001,
        price_impact_5pct=0.008
    )
    
    fund = FundLiquidityProfile(
        redemption_notice_days=args.redemption_days,
        redemption_frequency=args.redemption_frequency,
        lockup_months=args.lockup_months,
        fund_age_months=args.fund_age_months,
        aum=args.fund_aum,
        strategy_type="equity_long_short"
    )
    
    model = LiquidityRiskModel()
    premium = model.estimate_illiquidity_premium(position, fund)
    liquidation = model.estimate_liquidation_cost(position, 100_000, "normal")
    stress = model.stress_test_liquidity([position], fund, 0.2)
    
    results = {
        "illiquidity_premium": premium,
        "liquidation_cost": liquidation,
        "stress_test": stress
    }
    
    output_results(results, args.output)
    print(f"Liquidity analysis complete for {args.symbol}")
    print(f"  Illiquidity Premium: {premium['premium_bps']:.1f} bps")
    print(f"  Liquidation Cost: {liquidation['total_cost_bps']:.1f} bps")
    print(f"  Stress Test Cost: {stress['total_liquidation_cost_bps']:.1f} bps")


def run_strategy_eval(args):
    from quant_research.strategies.evaluator import StrategyDatabase, create_sample_strategies
    
    db = StrategyDatabase()
    
    if args.input:
        db.load_from_json(args.input)
    else:
        for strat in create_sample_strategies(args.generate):
            db.add_strategy(strat)
    
    if args.category:
        from quant_research.strategies.evaluator import StrategyCategory
        try:
            cat = StrategyCategory(args.category)
            strategies = db.filter_by_category(cat)
        except ValueError:
            print(f"Unknown category: {args.category}")
            return
    else:
        strategies = db.strategies
    
    if args.min_score:
        strategies = db.filter_by_score(args.min_score)
    
    ranked = sorted(strategies, key=lambda s: s.scores.composite_score(), reverse=True)
    top_strategies = ranked[:args.top]
    
    results = {
        "total_strategies": len(db.strategies),
        "filtered_strategies": len(strategies),
        "top_strategies": [s.to_dict() for s in top_strategies],
        "category_stats": db.get_category_stats()
    }
    
    output_results(results, args.output)
    print(f"Strategy evaluation complete. Top {args.top}:")
    for i, s in enumerate(top_strategies, 1):
        print(f"  {i}. {s.metadata.name} ({s.metadata.category.value}) - Score: {s.scores.composite_score():.3f}")


def run_backtest(args):
    from quant_research.backtest.engine import (
        BacktestEngine, MomentumStrategy, MeanReversionStrategy,
        SimpleExecutionModel, MarketImpactModel
    )
    
    from quant_research.api.data_providers import UnifiedDataClient
    
    client = UnifiedDataClient()
    data = client.get_historical_data(args.symbol, args.start, args.end or "2024-12-31")
    
    if data.empty:
        print(f"No data found for {args.symbol}")
        return
    
    execution_model = SimpleExecutionModel() if args.execution == "simple" else MarketImpactModel()
    engine = BacktestEngine(execution_model=execution_model, initial_capital=args.capital)
    
    if args.strategy == "momentum":
        strategy = MomentumStrategy(lookback=20, threshold=0.01)
    else:
        strategy = MeanReversionStrategy(window=20, num_std=2.0)
    
    result = engine.run(data, strategy, args.start, args.end)
    
    output_results(result["metrics"], args.output)
    print(f"Backtest complete for {args.strategy} on {args.symbol}")
    for k, v in result["metrics"].items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")


def run_hasbrouck(args):
    from quant_research.models.hasbrouck_var import run_hasbrouck_analysis
    results = run_hasbrouck_analysis(args.symbol, args.date)
    output_results(results, args.output)
    print(f"Hasbrouck VAR analysis complete for {args.symbol} on {args.date}")
    if "price_impact" in results:
        pi = results["price_impact"]
        print(f"  Permanent Price Impact: {pi.get('permanent_price_impact', 0):.6f}")
        print(f"  Half-life (lags): {pi.get('half_life_lags', 'N/A')}")
        print(f"  Information Content: {pi.get('information_content', 0):.6f}")


def run_full_analysis(args):
    from quant_research.microstructure.analyzer import run_microstructure_analysis
    from quant_research.risk.liquidity_model import run_liquidity_analysis
    from quant_research.models.hasbrouck_var import run_hasbrouck_analysis
    import os
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Running full integrated analysis for {args.symbol} on {args.date}...")
    
    micro_results = run_microstructure_analysis(args.symbol, args.date)
    with open(f"{args.output_dir}/{args.symbol}_microstructure.json", "w") as f:
        json.dump(micro_results, f, indent=2, default=str)
    print("  [OK] Microstructure analysis complete")
    
    liq_results = run_liquidity_analysis()
    with open(f"{args.output_dir}/{args.symbol}_liquidity.json", "w") as f:
        json.dump(liq_results, f, indent=2, default=str)
    print("  [OK] Liquidity risk analysis complete")
    
    hasb_results = run_hasbrouck_analysis(args.symbol, args.date)
    with open(f"{args.output_dir}/{args.symbol}_hasbrouck.json", "w") as f:
        json.dump(hasb_results, f, indent=2, default=str)
    print("  [OK] Hasbrouck VAR analysis complete")
    
    summary = {
        "symbol": args.symbol,
        "date": args.date,
        "microstructure_summary": {
            "effective_spread": micro_results["microstructure_metrics"]["effective_spread"],
            "price_impact": micro_results["microstructure_metrics"]["price_impact"],
            "vpin": micro_results["microstructure_metrics"]["vpin"]
        },
        "liquidity_summary": {
            "illiquidity_premium_bps": liq_results["illiquidity_premium"]["premium_bps"],
            "liquidation_cost_bps": liq_results["liquidation_cost"]["total_cost_bps"]
        },
        "price_impact_summary": {
            "permanent_impact": hasb_results.get("price_impact", {}).get("permanent_price_impact", 0),
            "half_life": hasb_results.get("price_impact", {}).get("half_life_lags", 0)
        }
    }
    
    with open(f"{args.output_dir}/{args.symbol}_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"\nFull analysis complete. Results saved to {args.output_dir}/")
    print(json.dumps(summary, indent=2, default=str))


def run_fetch_data(args):
    from quant_research.api.data_providers import UnifiedDataClient
    
    client = UnifiedDataClient()
    data = client.get_historical_data(args.symbol, args.start, args.end, args.source)
    
    if data.empty:
        print(f"No data found for {args.symbol}")
        return
    
    if args.output:
        data.to_csv(args.output)
        print(f"Data saved to {args.output}")
    else:
        print(data.head())
        print(f"\nShape: {data.shape}")
        print(f"Date range: {data.index[0]} to {data.index[-1]}")


def output_results(results: dict, output_file: Optional[str]):
    if output_file:
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Results saved to {output_file}")


def main():
    parser = create_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    command_map = {
        "microstructure": run_microstructure,
        "liquidity": run_liquidity,
        "strategy-eval": run_strategy_eval,
        "backtest": run_backtest,
        "hasbrouck": run_hasbrouck,
        "full-analysis": run_full_analysis,
        "fetch-data": run_fetch_data
    }
    
    if args.command in command_map:
        try:
            command_map[args.command](args)
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        print(f"Unknown command: {args.command}")
        parser.print_help()


if __name__ == "__main__":
    main()