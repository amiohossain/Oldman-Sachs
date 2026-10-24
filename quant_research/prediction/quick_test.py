import numpy as np
import pandas as pd
import json
from typing import Dict, List

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prediction.pipeline import PredictionPipeline, PredictionPipelineConfig


def run_quick_test(name: str, config: PredictionPipelineConfig) -> Dict:
    print(f"\n{'='*60}")
    print(f"TEST: {name}")
    print(f"{'='*60}")
    
    pipeline = PredictionPipeline(config)
    results = pipeline.run()
    
    summary = {}
    for symbol, result in results.items():
        summary[symbol] = {
            "mse": result["metrics"].get("mse", 0),
            "directional_accuracy": result["metrics"].get("directional_accuracy", 0),
            "correlation": result["metrics"].get("correlation", 0),
            "sharpe": result["backtest"].get("sharpe", 0),
            "annual_return": result["backtest"].get("annual_return", 0),
            "max_drawdown": result["backtest"].get("max_drawdown", 0),
            "win_rate": result["backtest"].get("win_rate", 0),
            "profit_factor": result["backtest"].get("profit_factor", 0),
            "num_trades": result["backtest"].get("num_trades", 0)
        }
        
        print(f"\n{symbol}:")
        for k, v in summary[symbol].items():
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    
    return summary


# Quick tests with fewer trials
tests = [
    # Best so far: XGB 10d with tuning
    {
        "name": "XGB_10d_fast",
        "config": PredictionPipelineConfig(
            symbols=["SPY"],
            start_date="2020-01-01",
            target_horizon=10,
            target_type="return",
            model_type="xgb",
            tune_hyperparams=True,
            n_tuning_trials=20,
            walk_forward_splits=5
        )
    },
    # XGB 5d
    {
        "name": "XGB_5d_fast",
        "config": PredictionPipelineConfig(
            symbols=["SPY"],
            start_date="2020-01-01",
            target_horizon=5,
            target_type="return",
            model_type="xgb",
            tune_hyperparams=True,
            n_tuning_trials=20,
            walk_forward_splits=5
        )
    },
    # XGB 20d
    {
        "name": "XGB_20d_fast",
        "config": PredictionPipelineConfig(
            symbols=["SPY"],
            start_date="2020-01-01",
            target_horizon=20,
            target_type="return",
            model_type="xgb",
            tune_hyperparams=True,
            n_tuning_trials=20,
            walk_forward_splits=5
        )
    },
    # Ridge (fast baseline)
    {
        "name": "Ridge_10d",
        "config": PredictionPipelineConfig(
            symbols=["SPY"],
            start_date="2020-01-01",
            target_horizon=10,
            target_type="return",
            model_type="ridge",
            tune_hyperparams=False,
            walk_forward_splits=5
        )
    },
    # RF (fast)
    {
        "name": "RF_10d_fast",
        "config": PredictionPipelineConfig(
            symbols=["SPY"],
            start_date="2020-01-01",
            target_horizon=10,
            target_type="return",
            model_type="rf",
            tune_hyperparams=True,
            n_tuning_trials=15,
            walk_forward_splits=5
        )
    },
]

if __name__ == "__main__":
    all_results = {}
    
    for test in tests:
        try:
            result = run_quick_test(test["name"], test["config"])
            all_results[test["name"]] = result
        except Exception as e:
            print(f"Error in {test['name']}: {e}")
            all_results[test["name"]] = {"error": str(e)}
    
    # Save results
    with open("quick_test_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    for test_name, results in all_results.items():
        if "error" in results:
            print(f"\n{test_name}: ERROR - {results['error']}")
            continue
        
        print(f"\n{test_name}:")
        for symbol, metrics in results.items():
            print(f"  {symbol}: DA={metrics['directional_accuracy']:.2%}, "
                  f"Sharpe={metrics['sharpe']:.3f}, "
                  f"Return={metrics['annual_return']:.2%}, "
                  f"MaxDD={metrics['max_drawdown']:.2%}, "
                  f"WinRate={metrics['win_rate']:.2%}, "
                  f"Corr={metrics['correlation']:.3f}")