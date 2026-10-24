import argparse
import sys
import json
import pandas as pd
from typing import Optional

from prediction.pipeline import PredictionPipeline, PredictionPipelineConfig, quick_predict


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stock Prediction Pipeline using Quant Research Features",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s train --symbols SPY QQQ --horizon 1 --model ridge
  %(prog)s predict --symbol SPY --model-path models.pkl
  %(prog)s quick --symbol AAPL --horizon 5 --model rf
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    train = subparsers.add_parser("train", help="Train prediction models")
    train.add_argument("--symbols", nargs="+", default=["SPY"], help="Symbols to train")
    train.add_argument("--start", default="2020-01-01", help="Start date")
    train.add_argument("--end", help="End date")
    train.add_argument("--horizon", type=int, default=1, help="Prediction horizon (days)")
    train.add_argument("--target", choices=["return", "direction", "volatility"], default="return")
    train.add_argument("--model", choices=["ridge", "rf", "gb", "mlp", "xgb", "lgb", "logistic", "rf_cls", "mlp_cls", "xgb_cls", "lgb_cls"], default="ridge")
    train.add_argument("--splits", type=int, default=5, help="Walk-forward splits")
    train.add_argument("--cost", type=float, default=0.001, help="Transaction cost")
    train.add_argument("--save", help="Save models to path")
    
    predict = subparsers.add_parser("predict", help="Make predictions with saved model")
    predict.add_argument("--symbol", required=True, help="Symbol to predict")
    predict.add_argument("--model-path", required=True, help="Path to saved models")
    predict.add_argument("--days", type=int, default=30, help="Days of recent data to use")
    
    quick = subparsers.add_parser("quick", help="Quick train and predict")
    quick.add_argument("--symbol", default="SPY", help="Symbol")
    quick.add_argument("--horizon", type=int, default=1, help="Horizon")
    quick.add_argument("--model", default="ridge", help="Model type")
    quick.add_argument("--target", choices=["return", "direction", "volatility"], default="return")
    
    return parser


def run_train(args):
    config = PredictionPipelineConfig(
        symbols=args.symbols,
        start_date=args.start,
        end_date=args.end,
        target_horizon=args.horizon,
        target_type=args.target,
        model_type=args.model,
        walk_forward_splits=args.splits,
        transaction_cost=args.cost
    )
    
    pipeline = PredictionPipeline(config)
    results = pipeline.run()
    
    if args.save:
        pipeline.save_models(args.save)
        print(f"Models saved to {args.save}")
    
    return results


def run_predict(args):
    pipeline = PredictionPipeline.load_models(args.model_path)
    
    import yfinance as yf
    from quant_research.prediction.data.pipeline import DataPipeline
    from quant_research.prediction.features.engineer import FeatureEngineer
    
    ticker = yf.Ticker(args.symbol)
    df = ticker.history(period=f"{args.days}d", interval="1d")
    df.columns = [c.lower() for c in df.columns]
    
    if pipeline.feature_engineer and pipeline.feature_engineer.config.macro_features:
        dp = DataPipeline()
        macro = dp.fetch_macro_data(
            (df.index[0] - pd.Timedelta(days=60)).strftime("%Y-%m-%d"),
            df.index[-1].strftime("%Y-%m-%d")
        )
        if not macro.empty:
            macro_daily = macro.resample("D").last()
            df_idx = df.index
            if df_idx.tz is not None:
                macro_daily = macro_daily.tz_convert(df_idx.tz)
            else:
                macro_daily = macro_daily.tz_localize(None)
            macro_aligned = macro_daily.reindex(df.index, method="ffill")
            macro_features = pipeline.feature_engineer._macro_features(macro_aligned)
            tech_features = pipeline.feature_engineer._technical_features(df)
            features = pd.concat([tech_features, macro_features], axis=1)
        else:
            features = pipeline.feature_engineer._technical_features(df)
    else:
        features = pipeline.feature_engineer._technical_features(df)
    
    features = features.iloc[-1:].fillna(method="ffill").fillna(0)
    
    pred = pipeline.models[args.symbol].predict(features)
    print(f"Prediction for {args.symbol}: {pred[0]:.6f}")
    
    return pred


def run_quick(args):
    from quant_research.prediction.pipeline import PredictionPipeline, PredictionPipelineConfig
    
    config = PredictionPipelineConfig(
        symbols=[args.symbol],
        start_date="2020-01-01",
        target_horizon=args.horizon,
        target_type=args.target,
        model_type=args.model,
        walk_forward_splits=5
    )
    pipeline = PredictionPipeline(config)
    result = pipeline.run()
    result = result[args.symbol]
    
    print(f"\nResults for {args.symbol}:")
    print(f"  Metrics: {json.dumps(result['metrics'], indent=4, default=str)}")
    print(f"  Backtest Sharpe: {result['backtest']['sharpe']:.3f}")
    print(f"  Backtest Return: {result['backtest']['annual_return']:.2%}")


def main():
    parser = create_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        if args.command == "train":
            run_train(args)
        elif args.command == "predict":
            run_predict(args)
        elif args.command == "quick":
            run_quick(args)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()