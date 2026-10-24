import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Tuple, List
import requests
import gzip
import io
from quant_research.config.settings import settings


class TAQDataLoader:
    def __init__(self, data_dir: str = None):
        self.data_dir = data_dir or settings.data_dir
        self.cache_dir = settings.cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

    def load_taq_trades(self, symbol: str, date: str, source: str = "local") -> pd.DataFrame:
        if source == "local":
            return self._load_local_trades(symbol, date)
        elif source == "polygon":
            return self._fetch_polygon_trades(symbol, date)
        else:
            raise ValueError(f"Unknown source: {source}")

    def load_taq_quotes(self, symbol: str, date: str, source: str = "local") -> pd.DataFrame:
        if source == "local":
            return self._load_local_quotes(symbol, date)
        elif source == "polygon":
            return self._fetch_polygon_quotes(symbol, date)
        else:
            raise ValueError(f"Unknown source: {source}")

    def _load_local_trades(self, symbol: str, date: str) -> pd.DataFrame:
        filepath = os.path.join(self.data_dir, "taq", f"{symbol}_trades_{date}.csv.gz")
        if not os.path.exists(filepath):
            filepath = os.path.join(self.data_dir, "taq", f"{symbol}_trades_{date}.csv")
        if os.path.exists(filepath):
            if filepath.endswith(".gz"):
                return pd.read_csv(filepath, compression="gzip")
            return pd.read_csv(filepath)
        return pd.DataFrame()

    def _load_local_quotes(self, symbol: str, date: str) -> pd.DataFrame:
        filepath = os.path.join(self.data_dir, "taq", f"{symbol}_quotes_{date}.csv.gz")
        if not os.path.exists(filepath):
            filepath = os.path.join(self.data_dir, "taq", f"{symbol}_quotes_{date}.csv")
        if os.path.exists(filepath):
            if filepath.endswith(".gz"):
                return pd.read_csv(filepath, compression="gzip")
            return pd.read_csv(filepath)
        return pd.DataFrame()

    def _fetch_polygon_trades(self, symbol: str, date: str) -> pd.DataFrame:
        url = f"https://api.polygon.io/v3/trades/{symbol}"
        params = {
            "timestamp.gte": f"{date}T00:00:00Z",
            "timestamp.lt": f"{date}T23:59:59Z",
            "limit": 50000,
            "apiKey": settings.api.polygon_api_key
        }
        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            return pd.DataFrame(data.get("results", []))
        return pd.DataFrame()

    def _fetch_polygon_quotes(self, symbol: str, date: str) -> pd.DataFrame:
        url = f"https://api.polygon.io/v3/quotes/{symbol}"
        params = {
            "timestamp.gte": f"{date}T00:00:00Z",
            "timestamp.lt": f"{date}T23:59:59Z",
            "limit": 50000,
            "apiKey": settings.api.polygon_api_key
        }
        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            return pd.DataFrame(data.get("results", []))
        return pd.DataFrame()


class TAQPreprocessor:
    @staticmethod
    def standardize_trades(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        df = df.copy()
        df.columns = df.columns.str.lower()
        required = ["price", "size", "time"]
        for col in required:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        df["time"] = pd.to_datetime(df["time"])
        df = df.sort_values("time").reset_index(drop=True)
        df["dollar_volume"] = df["price"] * df["size"]
        return df

    @staticmethod
    def standardize_quotes(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        df = df.copy()
        df.columns = df.columns.str.lower()
        required = ["bid_price", "ask_price", "bid_size", "ask_size", "time"]
        for col in required:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        df["time"] = pd.to_datetime(df["time"])
        df = df.sort_values("time").reset_index(drop=True)
        df["mid_price"] = (df["bid_price"] + df["ask_price"]) / 2
        df["spread"] = df["ask_price"] - df["bid_price"]
        df["relative_spread"] = df["spread"] / df["mid_price"]
        return df

    @staticmethod
    def merge_trades_quotes(trades: pd.DataFrame, quotes: pd.DataFrame, window_seconds: int = 1) -> pd.DataFrame:
        if trades.empty or quotes.empty:
            return pd.DataFrame()
        trades = trades.copy()
        quotes = quotes.copy()
        trades["time_sec"] = trades["time"].dt.floor(f"{window_seconds}s")
        quotes["time_sec"] = quotes["time"].dt.floor(f"{window_seconds}s")
        merged = pd.merge_asof(
            trades.sort_values("time_sec"),
            quotes.sort_values("time_sec"),
            on="time_sec",
            direction="backward",
            suffixes=("_trade", "_quote")
        )
        if "time_trade" in merged.columns:
            merged["time"] = merged["time_trade"]
        elif "time_quote" in merged.columns:
            merged["time"] = merged["time_quote"]
        return merged
        return merged

    @staticmethod
    def infer_trade_direction(tick_test: bool = True, quote_test: bool = True) -> pd.DataFrame:
        pass

    @staticmethod
    def aggregate_same_timestamp(trades: pd.DataFrame) -> pd.DataFrame:
        if trades.empty:
            return trades
        grouped = trades.groupby(["price", trades["time"].dt.floor("1s")]).agg({
            "size": "sum",
            "dollar_volume": "sum"
        }).reset_index()
        grouped.columns = ["price", "time", "size", "dollar_volume"]
        return grouped


def create_sample_taq_data(symbol: str = "AAPL", date: str = "2024-01-15", n_trades: int = 10000, n_quotes: int = 50000) -> Tuple[pd.DataFrame, pd.DataFrame]:
    np.random.seed(42)
    base_price = 150.0
    start_time = pd.Timestamp(f"{date} 09:30:00")
    end_time = pd.Timestamp(f"{date} 16:00:00")
    
    trade_times = pd.date_range(start_time, end_time, periods=n_trades)
    price_changes = np.random.normal(0, 0.01, n_trades).cumsum()
    trade_prices = base_price + price_changes
    trade_sizes = np.random.randint(100, 10000, n_trades)
    
    trades = pd.DataFrame({
        "time": trade_times,
        "price": trade_prices,
        "size": trade_sizes,
        "symbol": symbol
    })
    
    quote_times = pd.date_range(start_time, end_time, periods=n_quotes)
    mid_prices = base_price + np.random.normal(0, 0.005, n_quotes).cumsum()
    spreads = np.abs(np.random.normal(0.01, 0.005, n_quotes))
    
    quotes = pd.DataFrame({
        "time": quote_times,
        "bid_price": mid_prices - spreads/2,
        "ask_price": mid_prices + spreads/2,
        "bid_size": np.random.randint(100, 5000, n_quotes),
        "ask_size": np.random.randint(100, 5000, n_quotes),
        "symbol": symbol
    })
    
    return trades, quotes