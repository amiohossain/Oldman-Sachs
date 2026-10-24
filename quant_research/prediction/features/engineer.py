import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")


@dataclass
class FeatureConfig:
    lookback_windows: List[int] = None
    microstructure_windows: List[int] = None
    liquidity_windows: List[int] = None
    technical_indicators: bool = True
    microstructure_features: bool = True
    liquidity_features: bool = True
    macro_features: bool = True

    def __post_init__(self):
        if self.lookback_windows is None:
            self.lookback_windows = [5, 10, 20, 60]
        if self.microstructure_windows is None:
            self.microstructure_windows = [10, 20, 60]
        if self.liquidity_windows is None:
            self.liquidity_windows = [20, 60]


class FeatureEngineer:
    def __init__(self, config: FeatureConfig = None):
        self.config = config or FeatureConfig()

    def build_features(self, 
                       price_data: pd.DataFrame,
                       microstructure_data: Dict = None,
                       liquidity_data: Dict = None,
                       macro_data: pd.DataFrame = None) -> pd.DataFrame:
        features = pd.DataFrame(index=price_data.index)
        
        if self.config.technical_indicators:
            features = pd.concat([features, self._technical_features(price_data)], axis=1)
        
        if self.config.microstructure_features and microstructure_data:
            features = pd.concat([features, self._microstructure_features(microstructure_data)], axis=1)
        
        if self.config.liquidity_features and liquidity_data:
            features = pd.concat([features, self._liquidity_features(liquidity_data)], axis=1)
        
        if self.config.macro_features and macro_data is not None:
            features = pd.concat([features, self._macro_features(macro_data)], axis=1)
        
        features = features.replace([np.inf, -np.inf], np.nan).ffill().bfill()
        return features

    def _technical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        df = pd.DataFrame(index=data.index)
        close = data["close"]
        high = data["high"]
        low = data["low"]
        open_ = data["open"]
        volume = data.get("volume", pd.Series(1, index=data.index))
        
        for window in self.config.lookback_windows:
            df[f"return_{window}"] = close.pct_change(window)
            df[f"log_return_{window}"] = np.log(close / close.shift(window))
            df[f"volatility_{window}"] = close.pct_change().rolling(window).std() * np.sqrt(252)
            df[f"sma_{window}"] = close.rolling(window).mean() / close - 1
            df[f"rsi_{window}"] = self._rsi(close, window)
            df[f"volume_ratio_{window}"] = volume / volume.rolling(window).mean()
            
            # Additional features
            df[f"max_dd_{window}"] = (close / close.rolling(window).max() - 1).min()
            df[f"skew_{window}"] = close.pct_change().rolling(window).skew()
            df[f"kurt_{window}"] = close.pct_change().rolling(window).kurt()
        
        df["hl_spread"] = (high - low) / close
        df["oc_spread"] = (close - open_) / close
        
        # Bollinger Bands
        for window in [20, 50]:
            ma = close.rolling(window).mean()
            std = close.rolling(window).std()
            df[f"bb_upper_{window}"] = (ma + 2*std) / close - 1
            df[f"bb_lower_{window}"] = (ma - 2*std) / close - 1
            df[f"bb_width_{window}"] = (df[f"bb_upper_{window}"] - df[f"bb_lower_{window}"])
            df[f"bb_pos_{window}"] = (close - (ma - 2*std)) / (4*std)
        
        # MACD
        ema12 = close.ewm(span=12).mean()
        ema26 = close.ewm(span=26).mean()
        df["macd"] = (ema12 - ema26) / close
        df["macd_signal"] = df["macd"].ewm(span=9).mean()
        df["macd_hist"] = df["macd"] - df["macd_signal"]
        
        # ATR
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs()
        ], axis=1).max(axis=1)
        df["atr_14"] = tr.rolling(14).mean() / close
        df["atr_30"] = tr.rolling(30).mean() / close
        
        # Momentum
        for w in [5, 10, 20, 60]:
            df[f"mom_{w}"] = close / close.shift(w) - 1
        
        # Volume features
        df["volume_trend"] = volume.rolling(20).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 20 else 0)
        df["volume_zscore"] = (volume - volume.rolling(60).mean()) / volume.rolling(60).std()
        
        # Moving average crosses
        for w1, w2 in [(5, 20), (10, 60), (20, 120), (50, 200)]:
            if w2 <= max(self.config.lookback_windows + [200]):
                df[f"ma_cross_{w1}_{w2}"] = (close.rolling(w1).mean() / close.rolling(w2).mean()) - 1
        
        # Regime features
        df["trend_strength"] = abs(close.rolling(60).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 60 else 0))
        df["regime_bull"] = (close > close.rolling(200).mean()).astype(int)
        df["regime_bear"] = (close < close.rolling(200).mean()).astype(int)
        
        return df

    def _rsi(self, prices: pd.Series, window: int) -> pd.Series:
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(window).mean()
        loss = -delta.where(delta < 0, 0).rolling(window).mean()
        rs = gain / loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    def _microstructure_features(self, micro_data: Dict) -> pd.DataFrame:
        if not micro_data:
            return pd.DataFrame()
        
        df = pd.DataFrame()
        
        if "vpins" in micro_data:
            vpins = pd.Series(micro_data["vpins"])
            for w in self.config.microstructure_windows:
                df[f"vpin_mean_{w}"] = vpins.rolling(w).mean()
                df[f"vpin_std_{w}"] = vpins.rolling(w).std()
                df[f"vpin_max_{w}"] = vpins.rolling(w).max()
        
        if "spreads" in micro_data:
            spreads = pd.Series(micro_data["spreads"])
            for w in self.config.microstructure_windows:
                df[f"spread_mean_{w}"] = spreads.rolling(w).mean()
                df[f"spread_vol_{w}"] = spreads.rolling(w).std()
        
        if "price_impacts" in micro_data:
            impacts = pd.Series(micro_data["price_impacts"])
            for w in self.config.microstructure_windows:
                df[f"impact_mean_{w}"] = impacts.rolling(w).mean()
                df[f"impact_std_{w}"] = impacts.rolling(w).std()
        
        if "kyle_lambdas" in micro_data:
            lambdas = pd.Series(micro_data["kyle_lambdas"])
            for w in self.config.microstructure_windows:
                df[f"lambda_mean_{w}"] = lambdas.rolling(w).mean()
        
        if "hft_ratios" in micro_data:
            hft = pd.Series(micro_data["hft_ratios"])
            for w in self.config.microstructure_windows:
                df[f"hft_ratio_{w}"] = hft.rolling(w).mean()
        
        df.index = micro_data.get("timestamps", pd.date_range("2020-01-01", periods=len(df)))
        return df

    def _liquidity_features(self, liq_data: Dict) -> pd.DataFrame:
        if not liq_data:
            return pd.DataFrame()
        
        df = pd.DataFrame()
        
        if "illiquidity_premiums" in liq_data:
            prem = pd.Series(liq_data["illiquidity_premiums"])
            for w in self.config.liquidity_windows:
                df[f"illiq_prem_{w}"] = prem.rolling(w).mean()
        
        if "liquidation_costs" in liq_data:
            costs = pd.Series(liq_data["liquidation_costs"])
            for w in self.config.liquidity_windows:
                df[f"liq_cost_{w}"] = costs.rolling(w).mean()
        
        if "fund_liquidity_scores" in liq_data:
            scores = pd.Series(liq_data["fund_liquidity_scores"])
            for w in self.config.liquidity_windows:
                df[f"fund_liq_{w}"] = scores.rolling(w).mean()
        
        df.index = liq_data.get("timestamps", pd.date_range("2020-01-01", periods=len(df)))
        return df

    def _macro_features(self, macro: pd.DataFrame) -> pd.DataFrame:
        df = pd.DataFrame(index=macro.index)
        
        if "vix" in macro.columns:
            df["vix"] = macro["vix"]
            df["vix_change"] = macro["vix"].pct_change()
            df["vix_ma20"] = macro["vix"].rolling(20).mean()
        
        if "credit_spread" in macro.columns:
            df["credit_spread"] = macro["credit_spread"]
            df["credit_spread_change"] = macro["credit_spread"].pct_change()
        
        if "term_spread" in macro.columns:
            df["term_spread"] = macro["term_spread"]
        
        if "dxy" in macro.columns:
            df["dxy_change"] = macro["dxy"].pct_change()
        
        return df


def create_target(data: pd.DataFrame, horizon: int = 1, target_type: str = "return") -> pd.Series:
    close = data["close"]
    
    if target_type == "return":
        target = close.shift(-horizon) / close - 1
    elif target_type == "log_return":
        target = np.log(close.shift(-horizon) / close)
    elif target_type == "direction":
        target = (close.shift(-horizon) > close).astype(int)
    elif target_type == "volatility":
        target = close.pct_change().rolling(horizon).std().shift(-horizon) * np.sqrt(252)
    else:
        raise ValueError(f"Unknown target_type: {target_type}")
    
    return target


def align_features_targets(features: pd.DataFrame, target: pd.Series, 
                           dropna: bool = True) -> Tuple[pd.DataFrame, pd.Series]:
    common_idx = features.index.intersection(target.index)
    X = features.loc[common_idx]
    y = target.loc[common_idx]
    
    if dropna:
        valid = X.notna().all(axis=1) & y.notna()
        X = X[valid]
        y = y[valid]
    
    return X, y