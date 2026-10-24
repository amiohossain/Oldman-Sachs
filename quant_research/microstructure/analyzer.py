import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from quant_research.config.settings import settings


@dataclass
class SpreadComponents:
    order_processing_cost: float
    inventory_cost: float
    adverse_selection_cost: float
    total_spread: float
    
    def to_dict(self) -> Dict:
        return {
            "order_processing_cost": self.order_processing_cost,
            "inventory_cost": self.inventory_cost,
            "adverse_selection_cost": self.adverse_selection_cost,
            "total_spread": self.total_spread,
            "op_share": self.order_processing_cost / self.total_spread if self.total_spread > 0 else 0,
            "inv_share": self.inventory_cost / self.total_spread if self.total_spread > 0 else 0,
            "as_share": self.adverse_selection_cost / self.total_spread if self.total_spread > 0 else 0
        }


@dataclass
class MicrostructureMetrics:
    effective_spread: float
    realized_spread: float
    price_impact: float
    quoted_spread: float
    depth: float
    volume: float
    volatility: float
    trade_intensity: float
    order_imbalance: float
    vpin: float


class MarketMicrostructureAnalyzer:
    def __init__(self, window_seconds: int = 15):
        self.window_seconds = window_seconds

    def analyze(self, trades: pd.DataFrame, quotes: pd.DataFrame) -> Dict:
        from quant_research.data.taq_loader import TAQPreprocessor
        
        merged = TAQPreprocessor.merge_trades_quotes(trades, quotes, self.window_seconds)
        if merged.empty:
            return {"error": "No data to analyze"}
        
        merged = TAQPreprocessor.standardize_trades(merged)
        merged = TAQPreprocessor.standardize_quotes(merged)
        
        spread_components = self._decompose_spread(merged)
        microstructure_metrics = self._compute_metrics(merged)
        toxicity_metrics = self._compute_toxicity(merged)
        hft_metrics = self._detect_hft_activity(merged)
        
        return {
            "spread_components": spread_components.to_dict(),
            "microstructure_metrics": microstructure_metrics.__dict__,
            "toxicity_metrics": toxicity_metrics,
            "hft_metrics": hft_metrics,
            "summary_stats": self._summary_stats(merged)
        }

    def _decompose_spread(self, merged: pd.DataFrame) -> SpreadComponents:
        if "spread" not in merged.columns:
            return SpreadComponents(0, 0, 0, 0)
        
        total_spread = merged["spread"].mean()
        
        order_processing = merged["spread"].quantile(0.05)
        
        if "net_order_flow" in merged.columns:
            adverse_selection = abs(merged["spread"].corr(merged["net_order_flow"])) * merged["spread"].std()
        else:
            adverse_selection = merged["spread"].std() * 0.3
        
        inventory = max(0, total_spread - order_processing - adverse_selection)
        order_processing = max(0, order_processing)
        adverse_selection = max(0, adverse_selection)
        
        return SpreadComponents(
            order_processing_cost=order_processing,
            inventory_cost=inventory,
            adverse_selection_cost=adverse_selection,
            total_spread=total_spread
        )

    def _compute_metrics(self, merged: pd.DataFrame) -> MicrostructureMetrics:
        if "price" not in merged.columns or "mid_price" not in merged.columns:
            return MicrostructureMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        
        merged = merged.copy()
        merged["trade_direction"] = self._infer_direction(merged)
        
        effective_spread = 2 * merged["trade_direction"] * (merged["price"] - merged["mid_price"])
        effective_spread = effective_spread.mean()
        
        price_impact = 2 * merged["trade_direction"] * (merged["mid_price"].shift(-1) - merged["mid_price"])
        price_impact = price_impact.mean()
        
        realized_spread = effective_spread - price_impact
        
        quoted_spread = merged["spread"].mean()
        depth = (merged["bid_size"] + merged["ask_size"]).mean()
        volume = merged["size"].sum()
        volatility = merged["mid_price"].pct_change().std() * np.sqrt(252 * 390 * 60)
        trade_intensity = len(merged) / max(1, (merged["time"].max() - merged["time"].min()).total_seconds() / 60)
        
        buy_vol = merged[merged["trade_direction"] > 0]["size"].sum()
        sell_vol = merged[merged["trade_direction"] < 0]["size"].sum()
        order_imbalance = (buy_vol - sell_vol) / max(1, buy_vol + sell_vol)
        
        vpin = self._compute_vpin(merged)
        
        return MicrostructureMetrics(
            effective_spread=float(effective_spread),
            realized_spread=float(realized_spread),
            price_impact=float(price_impact),
            quoted_spread=float(quoted_spread),
            depth=float(depth),
            volume=float(volume),
            volatility=float(volatility),
            trade_intensity=float(trade_intensity),
            order_imbalance=float(order_imbalance),
            vpin=float(vpin)
        )

    def _compute_vpin(self, merged: pd.DataFrame, bucket_size: int = 50) -> float:
        if "net_order_flow" not in merged.columns:
            return 0.0
        
        merged = merged.copy()
        merged["bucket"] = (merged.index // bucket_size).astype(int)
        bucket_flow = merged.groupby("bucket")["net_order_flow"].sum()
        
        if len(bucket_flow) < 2:
            return 0.0
        
        vpin = np.mean(np.abs(bucket_flow)) / np.mean(np.abs(bucket_flow).rolling(10).mean().dropna())
        return float(np.clip(vpin, 0, 2))

    def _compute_toxicity(self, merged: pd.DataFrame) -> Dict:
        if "price" not in merged.columns or "mid_price" not in merged.columns:
            return {}
        
        merged = merged.copy()
        merged["trade_direction"] = self._infer_direction(merged)
        
        price_changes = merged["mid_price"].diff()
        signed_price_changes = price_changes * merged["trade_direction"]
        
        toxicity = signed_price_changes.mean()
        adverse_selection = signed_price_changes[signed_price_changes > 0].mean()
        
        return {
            "toxicity": float(toxicity),
            "adverse_selection": float(adverse_selection) if not np.isnan(adverse_selection) else 0.0,
            "pin": self._compute_pin(merged)
        }

    def _compute_pin(self, merged: pd.DataFrame) -> float:
        if "trade_direction" not in merged.columns:
            return 0.0
        buy_prob = (merged["trade_direction"] > 0).mean()
        return float(2 * buy_prob * (1 - buy_prob))

    def _detect_hft_activity(self, merged: pd.DataFrame) -> Dict:
        if "time" not in merged.columns:
            return {}
        
        merged = merged.copy()
        merged["time_diff"] = merged["time"].diff().dt.total_seconds()
        
        sub_second = (merged["time_diff"] < 1).sum()
        sub_millisecond = (merged["time_diff"] < 0.001).sum()
        
        trade_sizes = merged["size"]
        small_trades = (trade_sizes < 100).sum()
        odd_lot_trades = (trade_sizes % 100 != 0).sum()
        
        return {
            "sub_second_trades": int(sub_second),
            "sub_millisecond_trades": int(sub_millisecond),
            "small_trade_ratio": float(small_trades / len(merged)),
            "odd_lot_ratio": float(odd_lot_trades / len(merged)),
            "hft_probability": float((sub_second + odd_lot_trades) / (2 * len(merged)))
        }

    def _infer_direction(self, merged: pd.DataFrame) -> pd.Series:
        direction = pd.Series(0, index=merged.index)
        if "price" in merged.columns and "mid_price" in merged.columns:
            tick_test = np.sign(merged["price"].diff())
            quote_test = np.sign(merged["price"] - merged["mid_price"])
            direction = np.where((tick_test == quote_test) & (tick_test != 0), tick_test, 0)
        return pd.Series(direction, index=merged.index)

    def _summary_stats(self, merged: pd.DataFrame) -> Dict:
        return {
            "n_trades": len(merged),
            "n_quotes": len(merged) if "bid_price" in merged.columns else 0,
            "date_range": {
                "start": merged["time"].min().isoformat() if "time" in merged.columns else None,
                "end": merged["time"].max().isoformat() if "time" in merged.columns else None
            },
            "symbols": merged["symbol"].unique().tolist() if "symbol" in merged.columns else []
        }


class KyleLambdaEstimator:
    @staticmethod
    def estimate(trades: pd.DataFrame, quotes: pd.DataFrame, window: str = "1min") -> pd.Series:
        from quant_research.data.taq_loader import TAQPreprocessor
        
        merged = TAQPreprocessor.merge_trades_quotes(trades, quotes)
        if merged.empty:
            return pd.Series()
        
        merged = merged.copy()
        merged["trade_direction"] = np.sign(merged["price"].diff())
        merged["signed_volume"] = merged["size"] * merged["trade_direction"]
        merged["mid_price"] = (merged["bid_price"] + merged["ask_price"]) / 2
        merged["price_change"] = merged["mid_price"].diff()
        
        merged.set_index("time", inplace=True)
        
        lambda_estimates = merged.groupby(pd.Grouper(freq=window)).apply(
            lambda x: KyleLambdaEstimator._kyle_lambda(x) if len(x) > 10 else np.nan
        )
        
        return lambda_estimates.dropna()

    @staticmethod
    def _kyle_lambda(group: pd.DataFrame) -> float:
        if len(group) < 10:
            return np.nan
        x = group["signed_volume"].values.reshape(-1, 1)
        y = group["price_change"].values
        x = np.hstack([x, np.ones((len(x), 1))])
        try:
            beta = np.linalg.lstsq(x, y, rcond=None)[0]
            return float(beta[0])
        except:
            return np.nan


def run_microstructure_analysis(symbol: str = "AAPL", date: str = "2024-01-15") -> Dict:
    from quant_research.data.taq_loader import create_sample_taq_data
    
    trades, quotes = create_sample_taq_data(symbol, date)
    analyzer = MarketMicrostructureAnalyzer()
    return analyzer.analyze(trades, quotes)