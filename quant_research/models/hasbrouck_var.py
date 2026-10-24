import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
from typing import Tuple, Dict, Optional, List
import warnings
warnings.filterwarnings("ignore")
from quant_research.config.settings import settings


class HasbrouckVARModel:
    def __init__(self, max_lags: int = None, trend: str = None):
        self.max_lags = max_lags or settings.model.var_max_lags
        self.trend = trend or settings.model.var_trend
        self.model = None
        self.results = None
        self.lag_order = None
        self.variables = ["quote_change", "net_order_flow"]

    def prepare_data(self, merged_data: pd.DataFrame) -> pd.DataFrame:
        df = merged_data.copy()
        df = df.sort_values("time").reset_index(drop=True)
        
        df["quote_mid"] = (df["bid_price"] + df["ask_price"]) / 2
        df["quote_change"] = df["quote_mid"].diff()
        df["trade_direction"] = self._infer_trade_direction(df)
        df["signed_volume"] = df["size"] * df["trade_direction"]
        df["net_order_flow"] = df["signed_volume"]
        
        df = df.dropna(subset=["quote_change", "net_order_flow"])
        return df[["quote_change", "net_order_flow"]]

    def _infer_trade_direction(self, df: pd.DataFrame) -> pd.Series:
        direction = pd.Series(0, index=df.index)
        
        if "price" in df.columns and "quote_mid" in df.columns:
            tick_test = (df["price"] > df["price"].shift(1)).astype(int) * 2 - 1
            quote_test = (df["price"] > df["quote_mid"]).astype(int) * 2 - 1
            
            direction = np.where(
                (tick_test == quote_test) & (tick_test != 0),
                tick_test,
                0
            )
        
        return pd.Series(direction, index=df.index)

    def select_lag_order(self, data: pd.DataFrame, max_lags: int = None) -> int:
        max_lags = max_lags or self.max_lags
        model = VAR(data)
        lag_results = model.select_order(maxlags=max_lags)
        self.lag_order = lag_results.aic
        return self.lag_order

    def fit(self, data: pd.DataFrame, lag_order: int = None) -> "HasbrouckVARModel":
        lag_order = lag_order or self.select_lag_order(data)
        self.model = VAR(data)
        self.results = self.model.fit(lag_order, trend=self.trend)
        return self

    def impulse_response(self, horizon: int = None) -> np.ndarray:
        if self.results is None:
            raise ValueError("Model must be fitted first")
        horizon = horizon or settings.model.impulse_response_horizon
        irf = self.results.irf(horizon)
        return irf.irfs

    def orthogonalized_impulse_response(self, horizon: int = None) -> np.ndarray:
        if self.results is None:
            raise ValueError("Model must be fitted first")
        horizon = horizon or settings.model.impulse_response_horizon
        irf = self.results.irf(horizon)
        return irf.orth_irfs

    def get_price_impact(self, horizon: int = None) -> Dict:
        oirf = self.orthogonalized_impulse_response(horizon)
        n_vars = len(self.variables)
        
        impact_on_quotes = oirf[:, 0, 1]
        impact_on_flow = oirf[:, 1, 1]
        
        permanent_impact = np.sum(impact_on_quotes)
        half_life = self._calculate_half_life(impact_on_quotes)
        
        return {
            "impulse_response_quotes": impact_on_quotes.tolist(),
            "impulse_response_flow": impact_on_flow.tolist(),
            "permanent_price_impact": float(permanent_impact),
            "half_life_lags": half_life,
            "information_content": float(impact_on_quotes[0]),
            "long_term_effect": float(np.sum(impact_on_quotes[1:]))
        }

    def _calculate_half_life(self, irf: np.ndarray) -> Optional[int]:
        peak = np.max(np.abs(irf))
        target = peak / 2
        for i, val in enumerate(irf):
            if abs(val) <= target:
                return i
        return len(irf)

    def variance_decomposition(self, horizon: int = 10) -> pd.DataFrame:
        if self.results is None:
            raise ValueError("Model must be fitted first")
        fevd = self.results.fevd(horizon)
        return pd.DataFrame(
            fevd.decomp,
            columns=self.variables,
            index=[f"h={i+1}" for i in range(horizon)]
        )

    def granger_causality(self) -> Dict:
        if self.results is None:
            raise ValueError("Model must be fitted first")
        gc = self.results.test_causality("quote_change", "net_order_flow", kind="f")
        return {
            "f_statistic": gc.statistic,
            "p_value": gc.pvalue,
            "significant": gc.pvalue < 0.05
        }

    def summary(self) -> str:
        if self.results is None:
            return "Model not fitted"
        return self.results.summary()


class PriceImpactAnalyzer:
    def __init__(self):
        self.var_model = HasbrouckVARModel()

    def analyze(self, trades: pd.DataFrame, quotes: pd.DataFrame) -> Dict:
        from quant_research.data.taq_loader import TAQPreprocessor
        
        merged = TAQPreprocessor.merge_trades_quotes(trades, quotes)
        if merged.empty:
            return {"error": "No merged data available"}
        
        var_data = self.var_model.prepare_data(merged)
        if len(var_data) < 20:
            return {"error": "Insufficient data for VAR estimation"}
        
        self.var_model.fit(var_data)
        
        price_impact = self.var_model.get_price_impact()
        var_decomp = self.var_model.variance_decomposition()
        granger = self.var_model.granger_causality()
        
        spread_analysis = self._analyze_spreads(merged)
        order_flow_analysis = self._analyze_order_flow(merged)
        
        return {
            "price_impact": price_impact,
            "variance_decomposition": var_decomp.to_dict(),
            "granger_causality": granger,
            "spread_analysis": spread_analysis,
            "order_flow_analysis": order_flow_analysis,
            "model_summary": self.var_model.summary()
        }

    def _analyze_spreads(self, merged: pd.DataFrame) -> Dict:
        if "spread" not in merged.columns:
            return {}
        return {
            "mean_spread": float(merged["spread"].mean()),
            "median_spread": float(merged["spread"].median()),
            "mean_relative_spread_bps": float(merged["relative_spread"].mean() * 10000),
            "spread_volatility": float(merged["spread"].std()),
            "spread_components": self._decompose_spread(merged)
        }

    def _decompose_spread(self, merged: pd.DataFrame) -> Dict:
        if "spread" not in merged.columns or "size" not in merged.columns:
            return {}
        
        order_processing = merged["spread"].quantile(0.1)
        adverse_selection = merged["spread"].corr(merged["size"]) * merged["spread"].std()
        inventory = merged["spread"].mean() - order_processing - max(0, adverse_selection)
        
        return {
            "order_processing_cost": float(max(0, order_processing)),
            "adverse_selection_cost": float(max(0, adverse_selection)),
            "inventory_cost": float(max(0, inventory))
        }

    def _analyze_order_flow(self, merged: pd.DataFrame) -> Dict:
        if "net_order_flow" not in merged.columns:
            return {}
        flow = merged["net_order_flow"]
        return {
            "mean_order_flow": float(flow.mean()),
            "order_flow_volatility": float(flow.std()),
            "order_flow_autocorr": float(flow.autocorr(lag=1)),
            "buy_volume": float(merged[merged["trade_direction"] > 0]["size"].sum()),
            "sell_volume": float(merged[merged["trade_direction"] < 0]["size"].sum()),
            "vp_in": float((flow > 0).sum() / len(flow)),
            "trade_intensity": float(len(flow) / (merged["time"].max() - merged["time"].min()).total_seconds())
        }


def run_hasbrouck_analysis(symbol: str = "AAPL", date: str = "2024-01-15") -> Dict:
    from quant_research.data.taq_loader import TAQDataLoader, create_sample_taq_data
    
    loader = TAQDataLoader()
    trades, quotes = create_sample_taq_data(symbol, date)
    
    analyzer = PriceImpactAnalyzer()
    results = analyzer.analyze(trades, quotes)
    
    return results