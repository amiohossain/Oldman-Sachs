import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import warnings
warnings.filterwarnings("ignore")

from prediction.features.engineer import FeatureEngineer, FeatureConfig, create_target, align_features_targets
from prediction.models.registry import ModelConfig, create_model, walk_forward_validation, evaluate_predictions
from prediction.models.tuner import ModelTuner
from prediction.data.pipeline import DataPipeline, create_sample_data
from prediction.eval.metrics import compute_metrics, plot_predictions, plot_feature_importance, backtest_strategy, generate_report

try:
    from quant_research.microstructure.analyzer import MarketMicrostructureAnalyzer
    from quant_research.risk.liquidity_model import LiquidityRiskModel, PositionLiquidity, FundLiquidityProfile
    HAS_QUANT_RESEARCH = True
except ImportError:
    HAS_QUANT_RESEARCH = False


@dataclass
class PredictionPipelineConfig:
    symbols: List[str] = field(default_factory=lambda: ["SPY"])
    start_date: str = "2020-01-01"
    end_date: str = None
    target_horizon: int = 1
    target_type: str = "return"
    model_type: str = "xgb"
    model_hyperparams: Dict = field(default_factory=dict)
    use_microstructure: bool = False
    use_liquidity: bool = False
    tune_hyperparams: bool = True
    n_tuning_trials: int = 50
    walk_forward_splits: int = 5
    test_size: float = 0.2
    transaction_cost: float = 0.001
    random_state: int = 42


class PredictionPipeline:
    def __init__(self, config: PredictionPipelineConfig = None):
        self.config = config or PredictionPipelineConfig()
        self.data_pipeline = DataPipeline()
        self.models = {}
        self.results = {}
        self.feature_engineer = None

    def run(self) -> Dict:
        print(f"Fetching data for {self.config.symbols}...")
        price_data, macro_data = self._fetch_data()
        
        print("Engineering features...")
        ml_data = self._engineer_features(price_data, macro_data)
        
        print("Training models...")
        for symbol in self.config.symbols:
            print(f"  {symbol}...")
            self.results[symbol] = self._train_symbol(symbol, ml_data[symbol])
        
        print("Generating report...")
        self._generate_report()
        
        return self.results

    def _fetch_data(self) -> Tuple[Dict, pd.DataFrame]:
        try:
            price_data = self.data_pipeline.fetch_price_data(
                self.config.symbols, self.config.start_date, self.config.end_date
            )
            macro_data = self.data_pipeline.fetch_macro_data(self.config.start_date, self.config.end_date)
        except:
            print("Using sample data (API unavailable)")
            price_data, macro_data = create_sample_data(self.config.symbols, days=1000)
        
        return price_data, macro_data

    def _engineer_features(self, price_data: Dict, macro_data: pd.DataFrame) -> Dict:
        from prediction.features.engineer import FeatureEngineer, FeatureConfig
        
        fe_config = FeatureConfig(
            technical_indicators=True,
            microstructure_features=self.config.use_microstructure,
            liquidity_features=self.config.use_liquidity,
            macro_features=macro_data is not None and not macro_data.empty
        )
        self.feature_engineer = FeatureEngineer(fe_config)
        
        # Compute microstructure and liquidity features if enabled
        microstructure_data = {}
        liquidity_data = {}
        
        if self.config.use_microstructure and HAS_QUANT_RESEARCH:
            print("    Computing microstructure features...")
            microstructure_data = self._compute_microstructure_features(price_data)
        
        if self.config.use_liquidity and HAS_QUANT_RESEARCH:
            print("    Computing liquidity features...")
            liquidity_data = self._compute_liquidity_features(price_data)
        
        ml_data = {}
        for symbol, df in price_data.items():
            features = self.feature_engineer._technical_features(df)
            
            if macro_data is not None and not macro_data.empty:
                macro_daily = macro_data.resample("D").last()
                
                df_idx = df.index
                if df_idx.tz is not None:
                    macro_daily = macro_daily.tz_convert(df_idx.tz)
                else:
                    macro_daily = macro_daily.tz_localize(None)
                
                macro_aligned = macro_daily.reindex(df.index, method="ffill")
                macro_features = self.feature_engineer._macro_features(macro_aligned)
                features = pd.concat([features, macro_features], axis=1)
            
            # Add microstructure features
            if self.config.use_microstructure and symbol in microstructure_data:
                micro_features = self.feature_engineer._microstructure_features(microstructure_data[symbol])
                features = pd.concat([features, micro_features], axis=1)
            
            # Add liquidity features
            if self.config.use_liquidity and symbol in liquidity_data:
                liq_features = self.feature_engineer._liquidity_features(liquidity_data[symbol])
                features = pd.concat([features, liq_features], axis=1)
            
            target = create_target(df, horizon=self.config.target_horizon, target_type=self.config.target_type)
            X, y = align_features_targets(features, target)
            
            ml_data[symbol] = {"X": X, "y": y, "raw_data": df}
        
        return ml_data
    
    def _compute_microstructure_features(self, price_data: Dict) -> Dict:
        """Compute microstructure features using the main platform's analyzer."""
        microstructure_data = {}
        
        for symbol, df in price_data.items():
            # Use synthetic TAQ data for now (in production, use real TAQ data)
            from quant_research.data.taq_loader import create_sample_taq_data
            from quant_research.microstructure.analyzer import MarketMicrostructureAnalyzer
            
            # Generate sample TAQ data for each day
            dates = df.index
            vpins = []
            spreads = []
            price_impacts = []
            kyle_lambdas = []
            hft_ratios = []
            timestamps = []
            
            analyzer = MarketMicrostructureAnalyzer()
            
            # Sample a subset of days for speed
            sample_dates = dates[::max(1, len(dates)//50)]
            
            for date in sample_dates:
                date_str = date.strftime("%Y-%m-%d")
                trades, quotes = create_sample_taq_data(symbol, date_str, n_trades=1000, n_quotes=5000)
                result = analyzer.analyze(trades, quotes)
                
                if "microstructure_metrics" in result:
                    m = result["microstructure_metrics"]
                    vpins.append(m.get("vpin", 0))
                    spreads.append(m.get("effective_spread", 0))
                    price_impacts.append(m.get("price_impact", 0))
                    hft_ratios.append(m.get("hft_probability", 0))
                    timestamps.append(date)
            
            # Kyle's Lambda (simplified)
            returns = df["close"].pct_change()
            volume = df["volume"]
            signed_vol = volume * np.sign(returns)
            kyle_lambda = (returns * signed_vol).rolling(20).apply(
                lambda x: np.polyfit(x[:-1], x[1:], 1)[0] if len(x) > 10 else 0
            )
            
            microstructure_data[symbol] = {
                "vpins": vpins,
                "spreads": spreads,
                "price_impacts": price_impacts,
                "kyle_lambdas": kyle_lambda.fillna(0).tolist(),
                "hft_ratios": hft_ratios,
                "timestamps": timestamps
            }
        
        return microstructure_data
    
    def _compute_liquidity_features(self, price_data: Dict) -> Dict:
        """Compute liquidity risk features using the main platform's model."""
        liquidity_data = {}
        
        for symbol, df in price_data.items():
            from quant_research.risk.liquidity_model import (
                LiquidityRiskModel, PositionLiquidity, FundLiquidityProfile
            )
            
            model = LiquidityRiskModel()
            
            # Estimate position liquidity from price data
            avg_volume = df["volume"].rolling(60).mean()
            spread_bps = (df["high"] - df["low"]) / df["close"] * 10000
            volume_z = (df["volume"] - df["volume"].rolling(60).mean()) / df["volume"].rolling(60).std()
            
            # Simplified position liquidity
            position = PositionLiquidity(
                symbol=symbol,
                avg_daily_volume=float(avg_volume.iloc[-1]) if not avg_volume.empty else 1e6,
                bid_ask_spread_bps=float(spread_bps.iloc[-1]) if not spread_bps.empty else 10,
                market_depth=float(avg_volume.iloc[-1] * df["close"].iloc[-1]) if not avg_volume.empty else 1e7,
                days_to_liquidate=1.0,
                price_impact_1pct=0.001,
                price_impact_5pct=0.005
            )
            
            # Simplified fund liquidity
            fund = FundLiquidityProfile(
                redemption_notice_days=30,
                redemption_frequency="monthly",
                lockup_months=0,
                fund_age_months=36,
                aum=1e9,
                strategy_type="equity_long_short"
            )
            
            illiq_premiums = []
            liq_costs = []
            fund_liq_scores = []
            
            for i in range(0, len(df), max(1, len(df)//50)):
                premium = model.estimate_illiquidity_premium(position, fund)
                cost = model.estimate_liquidation_cost(position, 10000, "normal")
                illiq_premiums.append(premium["premium_bps"])
                liq_costs.append(cost["total_cost_bps"])
                fund_liq_scores.append(fund.liquidity_score())
            
            liquidity_data[symbol] = {
                "illiquidity_premiums": illiq_premiums,
                "liquidation_costs": liq_costs,
                "fund_liquidity_scores": fund_liq_scores,
                "timestamps": df.index[::max(1, len(df)//50)].tolist()
            }
        
        return liquidity_data

    def _train_symbol(self, symbol: str, data: Dict) -> Dict:
        X, y = data["X"], data["y"]
        
        if self.config.tune_hyperparams and self.config.model_type in ["xgb", "lgb", "rf", "xgb_cls", "lgb_cls", "rf_cls"]:
            print(f"    Tuning hyperparameters...")
            tuner = ModelTuner(n_trials=self.config.n_tuning_trials, n_splits=self.config.walk_forward_splits)
            best_params = tuner.tune(X, y, self.config.model_type, self.config.target_type)
            model_config = ModelConfig(
                model_type=self.config.model_type,
                target_type=self.config.target_type,
                horizon=self.config.target_horizon,
                hyperparams=best_params,
                random_state=self.config.random_state
            )
        else:
            model_config = ModelConfig(
                model_type=self.config.model_type,
                target_type=self.config.target_type,
                horizon=self.config.target_horizon,
                hyperparams=self.config.model_hyperparams,
                random_state=self.config.random_state
            )
        
        model = create_model(model_config)
        wf_result = walk_forward_validation(model, X, y, n_splits=self.config.walk_forward_splits)
        
        model.fit(X, y)
        self.models[symbol] = model
        
        metrics = evaluate_predictions(wf_result.actuals, wf_result.predictions, self.config.target_type)
        
        backtest = backtest_strategy(
            wf_result.actuals, wf_result.predictions, 
            transaction_cost=self.config.transaction_cost,
            horizon=self.config.target_horizon
        )
        
        return {
            "model": model,
            "walk_forward": wf_result,
            "metrics": metrics,
            "backtest": backtest,
            "feature_importance": wf_result.feature_importance,
            "X": X, "y": y
        }

    def _generate_report(self):
        report = generate_report(self.results)
        print(report)
        return report

    def predict(self, symbol: str, current_data: pd.DataFrame) -> np.ndarray:
        if symbol not in self.models:
            raise ValueError(f"Model for {symbol} not trained")
        
        features = self.feature_engineer._technical_features(current_data)
        features = features.iloc[-1:].fillna(method="ffill").fillna(0)
        
        return self.models[symbol].predict(features)

    def save_models(self, path: str):
        import joblib
        data = {
            "models": self.models,
            "config": self.config,
            "feature_engineer": self.feature_engineer
        }
        joblib.dump(data, path)

    @classmethod
    def load_models(cls, path: str) -> "PredictionPipeline":
        import joblib
        data = joblib.load(path)
        pipeline = cls(data["config"])
        pipeline.models = data["models"]
        pipeline.feature_engineer = data["feature_engineer"]
        return pipeline


def quick_predict(symbol: str = "SPY", horizon: int = 1, model_type: str = "ridge") -> Dict:
    config = PredictionPipelineConfig(
        symbols=[symbol],
        target_horizon=horizon,
        model_type=model_type,
        start_date="2020-01-01"
    )
    pipeline = PredictionPipeline(config)
    results = pipeline.run()
    return results[symbol]


if __name__ == "__main__":
    config = PredictionPipelineConfig(
        symbols=["SPY", "QQQ"],
        start_date="2020-01-01",
        target_horizon=1,
        target_type="return",
        model_type="ridge",
        walk_forward_splits=5
    )
    pipeline = PredictionPipeline(config)
    results = pipeline.run()