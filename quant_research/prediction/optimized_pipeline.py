import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import warnings
warnings.filterwarnings("ignore")

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prediction.pipeline import PredictionPipeline, PredictionPipelineConfig
from prediction.models.ensemble import EnsembleModel, EnsembleConfig, RegimeAwareModel, RegimeDetector, create_ensemble_config
from prediction.models.registry import ModelConfig, create_model, walk_forward_validation, evaluate_predictions


class OptimizedPipeline:
    def __init__(self, config: PredictionPipelineConfig):
        self.config = config
        self.pipeline = PredictionPipeline(config)
        self.ensemble_models = {}
        self.regime_detectors = {}

    def run(self) -> Dict:
        print(f"Fetching data for {self.config.symbols}...")
        price_data, macro_data = self.pipeline._fetch_data()
        
        print("Engineering features...")
        ml_data = self.pipeline._engineer_features(price_data, macro_data)
        
        print("Training optimized models...")
        results = {}
        for symbol in self.config.symbols:
            print(f"  {symbol}...")
            results[symbol] = self._train_optimized(symbol, ml_data[symbol])
        
        print("Generating report...")
        self._generate_report(results)
        
        return results

    def _train_optimized(self, symbol: str, data: Dict) -> Dict:
        X, y = data["X"], data["y"]
        
        # 1. Train ensemble
        print(f"    Training ensemble...")
        ensemble_config = EnsembleConfig(
            models=["xgb", "lgb", "ridge"],
            weights=[0.5, 0.3, 0.2],
            use_stacking=False
        )
        
        model_configs = create_ensemble_config("xgb")
        for k, v in model_configs.items():
            v.random_state = self.config.random_state
            v.target_type = self.config.target_type
            v.horizon = self.config.target_horizon
        
        ensemble = EnsembleModel(ensemble_config, model_configs)
        ensemble.fit(X, y)
        
        # 2. Train regime-aware model
        print(f"    Training regime-aware model...")
        regime_detector = RegimeDetector(n_regimes=3)
        
        # Create base model configs for each regime
        regime_models = {}
        for regime_id in range(3):
            model_config = ModelConfig(
                model_type="xgb",
                target_type=self.config.target_type,
                horizon=self.config.target_horizon,
                hyperparams={
                    "n_estimators": 200,
                    "max_depth": 5,
                    "learning_rate": 0.05,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                    "reg_alpha": 1.0,
                    "reg_lambda": 0.01
                },
                random_state=self.config.random_state
            )
            model = create_model(model_config)
            regime_models[regime_id] = model
        
        regime_aware = RegimeAwareModel(
            lambda: create_model(ModelConfig(
                model_type="xgb",
                target_type=self.config.target_type,
                horizon=self.config.target_horizon,
                hyperparams={
                    "n_estimators": 200,
                    "max_depth": 5,
                    "learning_rate": 0.05,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                    "reg_alpha": 1.0,
                    "reg_lambda": 0.01
                },
                random_state=self.config.random_state
            )),
            RegimeDetector(n_regimes=3)
        )
        regime_aware.fit(X, y)
        
        # 3. Walk-forward validation for ensemble
        from prediction.models.registry import walk_forward_validation, evaluate_predictions, BaseModel
        from prediction.eval.metrics import backtest_strategy
        
        # Create a wrapper for ensemble to work with walk_forward_validation
        class EnsembleWrapper(BaseModel):
            def __init__(self, ensemble):
                config = ModelConfig(target_type="return", horizon=10)
                super().__init__(config)
                self.ensemble = ensemble
            
            def _create_model(self):
                return self.ensemble
            
            def fit(self, X, y):
                self.ensemble.fit(X, y)
                self.is_fitted = True
                self.feature_names = X.columns.tolist()
                return self
            
            def predict(self, X):
                return self.ensemble.predict(X)
        
        ensemble_wrapper = EnsembleWrapper(ensemble)
        wf_result = walk_forward_validation(ensemble_wrapper, X, y, n_splits=self.config.walk_forward_splits)
        
        # Fit final ensemble
        ensemble.fit(X, y)
        
        # Evaluate
        metrics = evaluate_predictions(wf_result.actuals, wf_result.predictions, "return")
        
        # Backtest
        from prediction.eval.metrics import backtest_strategy
        backtest = backtest_strategy(
            wf_result.actuals, wf_result.predictions,
            transaction_cost=0.001,
            horizon=10
        )
        
        # Feature importance from XGBoost
        feature_importance = None
        if "xgb" in ensemble.base_models:
            xgb_model = ensemble.base_models["xgb"].model
            if hasattr(xgb_model, "feature_importances_"):
                feature_importance = pd.DataFrame({
                    "feature": ensemble.feature_names,
                    "importance": xgb_model.feature_importances_
                }).sort_values("importance", ascending=False)
        
        return {
            "ensemble": ensemble,
            "regime_aware": regime_aware,
            "walk_forward": wf_result,
            "metrics": metrics,
            "backtest": backtest,
            "feature_importance": feature_importance,
            "X": X, "y": y
        }

    def _generate_report(self, results: Dict):
        from prediction.eval.metrics import generate_report
        report = generate_report(results)
        print(report)


if __name__ == "__main__":
    config = PredictionPipelineConfig(
        symbols=["SPY"],
        start_date="2020-01-01",
        target_horizon=10,
        target_type="return",
        model_type="xgb",
        tune_hyperparams=False,
        walk_forward_splits=5,
        transaction_cost=0.001
    )
    
    opt = OptimizedPipeline(config)
    results = opt.run()