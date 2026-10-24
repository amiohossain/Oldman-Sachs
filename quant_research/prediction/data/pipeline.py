import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import yfinance as yf
import warnings
warnings.filterwarnings("ignore")


class DataPipeline:
    def __init__(self, cache_dir: str = "./data/cache"):
        self.cache_dir = cache_dir
        import os
        os.makedirs(cache_dir, exist_ok=True)

    def fetch_price_data(self, symbols: List[str], start: str, end: str = None) -> Dict[str, pd.DataFrame]:
        data = {}
        for symbol in symbols:
            cache_file = f"{self.cache_dir}/{symbol}_{start}_{end or 'now'}.pkl"
            
            try:
                df = pd.read_pickle(cache_file)
            except:
                ticker = yf.Ticker(symbol)
                df = ticker.history(start=start, end=end, interval="1d")
                df.columns = [c.lower() for c in df.columns]
                df.to_pickle(cache_file)
            
            data[symbol] = df
        return data

    def fetch_macro_data(self, start: str, end: str = None) -> pd.DataFrame:
        macro_symbols = {
            "vix": "^VIX",
            "dxy": "DX-Y.NYB",
            "tnx": "^TNX",
            "credit_spread": "BAMLH0A0HYM2",
        }
        
        macro_data = {}
        for name, symbol in macro_symbols.items():
            try:
                ticker = yf.Ticker(symbol)
                df = ticker.history(start=start, end=end, interval="1d")
                macro_data[name] = df["Close"]
            except:
                pass
        
        if macro_data:
            df = pd.DataFrame(macro_data)
            df.columns = macro_data.keys()
            return df
        return pd.DataFrame()

    def prepare_ml_data(self, 
                        price_data: Dict[str, pd.DataFrame],
                        macro_data: pd.DataFrame = None,
                        target_horizon: int = 1,
                        target_type: str = "return") -> Tuple[Dict, Dict]:
        from prediction.features.engineer import FeatureEngineer, FeatureConfig, create_target, align_features_targets
        
        results = {}
        for symbol, df in price_data.items():
            config = FeatureConfig()
            engineer = FeatureEngineer(config)
            
            features = engineer._technical_features(df)
            target = create_target(df, horizon=target_horizon, target_type=target_type)
            
            X, y = align_features_targets(features, target)
            
            if macro_data is not None:
                macro_aligned = macro_data.reindex(X.index, method="ffill")
                macro_features = engineer._macro_features(macro_aligned)
                X = pd.concat([X, macro_features], axis=1)
                X, y = align_features_targets(X, y)
            
            results[symbol] = {"X": X, "y": y, "raw_data": df}
        
        return results


def create_sample_data(symbols: List[str] = None, days: int = 500) -> Tuple[Dict, pd.DataFrame]:
    if symbols is None:
        symbols = ["SPY", "QQQ", "IWM"]
    
    np.random.seed(42)
    dates = pd.date_range(end=datetime.now(), periods=days, freq="D")
    
    price_data = {}
    for sym in symbols:
        base = 100 + np.random.normal(0, 10)
        returns = np.random.normal(0.0005, 0.015, days)
        prices = base * np.exp(np.cumsum(returns))
        
        df = pd.DataFrame({
            "open": prices * np.random.uniform(0.995, 1.005, days),
            "high": prices * np.random.uniform(1.0, 1.02, days),
            "low": prices * np.random.uniform(0.98, 1.0, days),
            "close": prices,
            "volume": np.random.lognormal(15, 0.5, days).astype(int)
        }, index=dates)
        
        price_data[sym] = df
    
    macro_data = pd.DataFrame({
        "vix": np.random.uniform(12, 35, days),
        "credit_spread": np.random.uniform(0.02, 0.08, days),
        "term_spread": np.random.uniform(0.5, 2.5, days),
        "dxy": np.random.uniform(90, 110, days)
    }, index=dates)
    
    return price_data, macro_data