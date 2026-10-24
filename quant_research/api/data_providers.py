import os
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from quant_research.config.settings import settings


class BaseDataProvider:
    def __init__(self, api_key: str = None):
        self.api_key = api_key
        self.session = requests.Session()
    
    def _get(self, url: str, params: Dict = None) -> Dict:
        if params is None:
            params = {}
        if self.api_key:
            params["apiKey"] = self.api_key
        response = self.session.get(url, params=params, timeout=30)
        response.raise_for_status()
        return response.json()


class AlpacaProvider(BaseDataProvider):
    BASE_URL = "https://paper-api.alpaca.markets/v2"
    
    def __init__(self):
        super().__init__()
        self.api_key = settings.api.alpaca_api_key
        self.secret_key = settings.api.alpaca_secret_key
        self.session.headers.update({
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.secret_key
        })
    
    def get_account(self) -> Dict:
        return self._get(f"{self.BASE_URL}/account")
    
    def get_positions(self) -> List[Dict]:
        return self._get(f"{self.BASE_URL}/positions")
    
    def get_orders(self, status: str = "all") -> List[Dict]:
        return self._get(f"{self.BASE_URL}/orders", {"status": status})
    
    def place_order(self, symbol: str, qty: float, side: str, 
                    order_type: str = "market", time_in_force: str = "day") -> Dict:
        data = {
            "symbol": symbol,
            "qty": qty,
            "side": side,
            "type": order_type,
            "time_in_force": time_in_force
        }
        response = self.session.post(f"{self.BASE_URL}/orders", json=data)
        response.raise_for_status()
        return response.json()
    
    def get_bars(self, symbols: List[str], timeframe: str = "1Day",
                 start: str = None, end: str = None) -> Dict:
        params = {"symbols": ",".join(symbols), "timeframe": timeframe}
        if start:
            params["start"] = start
        if end:
            params["end"] = end
        return self._get(f"{self.BASE_URL}/stocks/bars", params)


class PolygonProvider(BaseDataProvider):
    BASE_URL = "https://api.polygon.io"
    
    def __init__(self):
        super().__init__(settings.api.polygon_api_key)
    
    def get_aggregates(self, symbol: str, multiplier: int = 1, 
                       timespan: str = "day", from_date: str = None, 
                       to_date: str = None) -> pd.DataFrame:
        url = f"{self.BASE_URL}/v2/aggs/ticker/{symbol}/range/{multiplier}/{timespan}/{from_date}/{to_date}"
        params = {"adjusted": "true", "sort": "asc", "limit": 50000}
        data = self._get(url, params)
        if "results" in data:
            df = pd.DataFrame(data["results"])
            df["timestamp"] = pd.to_datetime(df["t"], unit="ms")
            return df
        return pd.DataFrame()
    
    def get_trades(self, symbol: str, date: str) -> pd.DataFrame:
        url = f"{self.BASE_URL}/v3/trades/{symbol}"
        params = {
            "timestamp.gte": f"{date}T00:00:00Z",
            "timestamp.lt": f"{date}T23:59:59Z",
            "limit": 50000
        }
        data = self._get(url, params)
        if "results" in data:
            df = pd.DataFrame(data["results"])
            df["timestamp"] = pd.to_datetime(df["sip_timestamp"])
            return df
        return pd.DataFrame()
    
    def get_quotes(self, symbol: str, date: str) -> pd.DataFrame:
        url = f"{self.BASE_URL}/v3/quotes/{symbol}"
        params = {
            "timestamp.gte": f"{date}T00:00:00Z",
            "timestamp.lt": f"{date}T23:59:59Z",
            "limit": 50000
        }
        data = self._get(url, params)
        if "results" in data:
            df = pd.DataFrame(data["results"])
            df["timestamp"] = pd.to_datetime(df["sip_timestamp"])
            return df
        return pd.DataFrame()


class FinnhubProvider(BaseDataProvider):
    BASE_URL = "https://finnhub.io/api/v1"
    
    def __init__(self):
        super().__init__(settings.api.finnhub_api_key)
    
    def get_quote(self, symbol: str) -> Dict:
        return self._get(f"{self.BASE_URL}/quote", {"symbol": symbol})
    
    def get_candles(self, symbol: str, resolution: str, 
                    from_ts: int, to_ts: int) -> pd.DataFrame:
        data = self._get(f"{self.BASE_URL}/stock/candle", {
            "symbol": symbol, "resolution": resolution,
            "from": from_ts, "to": to_ts
        })
        if data.get("s") == "ok":
            df = pd.DataFrame({
                "timestamp": pd.to_datetime(data["t"], unit="s"),
                "open": data["o"], "high": data["h"],
                "low": data["l"], "close": data["c"], "volume": data["v"]
            })
            return df
        return pd.DataFrame()
    
    def get_company_news(self, symbol: str, from_date: str, to_date: str) -> List[Dict]:
        return self._get(f"{self.BASE_URL}/company-news", {
            "symbol": symbol, "from": from_date, "to": to_date
        })


class FMPProvider(BaseDataProvider):
    BASE_URL = "https://financialmodelingprep.com/api/v3"
    
    def __init__(self):
        super().__init__(settings.api.fmp_api_key)
    
    def get_income_statement(self, symbol: str, period: str = "annual") -> List[Dict]:
        return self._get(f"{self.BASE_URL}/income-statement/{symbol}", {"period": period})
    
    def get_balance_sheet(self, symbol: str, period: str = "annual") -> List[Dict]:
        return self._get(f"{self.BASE_URL}/balance-sheet-statement/{symbol}", {"period": period})
    
    def get_cash_flow(self, symbol: str, period: str = "annual") -> List[Dict]:
        return self._get(f"{self.BASE_URL}/cash-flow-statement/{symbol}", {"period": period})
    
    def get_key_metrics(self, symbol: str) -> List[Dict]:
        return self._get(f"{self.BASE_URL}/key-metrics/{symbol}")


class AlphaVantageProvider(BaseDataProvider):
    BASE_URL = "https://www.alphavantage.co/query"
    
    def __init__(self):
        super().__init__(settings.api.alpha_vantage_api_key)
    
    def get_daily_adjusted(self, symbol: str, outputsize: str = "full") -> pd.DataFrame:
        data = self._get(self.BASE_URL, {
            "function": "TIME_SERIES_DAILY_ADJUSTED",
            "symbol": symbol,
            "outputsize": outputsize
        })
        key = "Time Series (Daily)"
        if key in data:
            df = pd.DataFrame.from_dict(data[key], orient="index")
            df.index = pd.to_datetime(df.index)
            df.columns = [c.split(". ")[1] for c in df.columns]
            df = df.astype(float)
            return df.sort_index()
        return pd.DataFrame()
    
    def get_technical_indicator(self, symbol: str, indicator: str, 
                                 interval: str = "daily", **kwargs) -> pd.DataFrame:
        params = {
            "function": indicator.upper(),
            "symbol": symbol,
            "interval": interval,
            **kwargs
        }
        data = self._get(self.BASE_URL, params)
        for k, v in data.items():
            if "Technical Analysis" in k:
                df = pd.DataFrame.from_dict(v, orient="index")
                df.index = pd.to_datetime(df.index)
                return df.sort_index()
        return pd.DataFrame()


class DataProviderFactory:
    _providers = {}
    
    @classmethod
    def get_provider(cls, name: str) -> BaseDataProvider:
        if name not in cls._providers:
            providers = {
                "alpaca": AlpacaProvider,
                "polygon": PolygonProvider,
                "finnhub": FinnhubProvider,
                "fmp": FMPProvider,
                "alphavantage": AlphaVantageProvider
            }
            if name in providers:
                cls._providers[name] = providers[name]()
            else:
                raise ValueError(f"Unknown provider: {name}")
        return cls._providers[name]
    
    @classmethod
    def get_all_providers(cls) -> Dict[str, BaseDataProvider]:
        return {name: cls.get_provider(name) for name in 
                ["alpaca", "polygon", "finnhub", "fmp", "alphavantage"]}


class UnifiedDataClient:
    def __init__(self):
        self.providers = DataProviderFactory.get_all_providers()
    
    def get_historical_data(self, symbol: str, start: str, end: str, 
                           source: str = "polygon") -> pd.DataFrame:
        provider = self.providers.get(source)
        if not provider:
            raise ValueError(f"Provider {source} not available")
        
        if hasattr(provider, "get_aggregates"):
            return provider.get_aggregates(symbol, 1, "day", start, end)
        elif hasattr(provider, "get_daily_adjusted"):
            return provider.get_daily_adjusted(symbol)
        return pd.DataFrame()
    
    def get_intraday_data(self, symbol: str, date: str, 
                          source: str = "polygon") -> Dict[str, pd.DataFrame]:
        provider = self.providers.get(source)
        if not provider or not hasattr(provider, "get_trades"):
            return {"trades": pd.DataFrame(), "quotes": pd.DataFrame()}
        
        trades = provider.get_trades(symbol, date)
        quotes = provider.get_quotes(symbol, date)
        return {"trades": trades, "quotes": quotes}
    
    def get_fundamentals(self, symbol: str) -> Dict[str, Any]:
        fmp = self.providers.get("fmp")
        if not fmp:
            return {}
        return {
            "income_statement": fmp.get_income_statement(symbol),
            "balance_sheet": fmp.get_balance_sheet(symbol),
            "cash_flow": fmp.get_cash_flow(symbol),
            "key_metrics": fmp.get_key_metrics(symbol)
        }
    
    def get_news(self, symbol: str, from_date: str, to_date: str) -> List[Dict]:
        finnhub = self.providers.get("finnhub")
        if not finnhub:
            return []
        return finnhub.get_company_news(symbol, from_date, to_date)