import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import warnings
warnings.filterwarnings("ignore")


class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


@dataclass
class Order:
    symbol: str
    quantity: float
    side: OrderSide
    order_type: OrderType = OrderType.MARKET
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    timestamp: Optional[pd.Timestamp] = None
    order_id: Optional[str] = None
    filled: bool = False
    filled_price: Optional[float] = None
    filled_qty: float = 0.0


@dataclass
class Position:
    symbol: str
    quantity: float = 0.0
    avg_price: float = 0.0
    market_value: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0


@dataclass
class Portfolio:
    cash: float = 1_000_000.0
    positions: Dict[str, Position] = field(default_factory=dict)
    initial_capital: float = 1_000_000.0
    
    @property
    def total_value(self) -> float:
        return self.cash + sum(p.market_value for p in self.positions.values())
    
    @property
    def total_pnl(self) -> float:
        return self.total_value - self.initial_capital
    
    def get_position(self, symbol: str) -> Position:
        if symbol not in self.positions:
            self.positions[symbol] = Position(symbol=symbol)
        return self.positions[symbol]
    
    def update_market_value(self, prices: Dict[str, float]):
        for symbol, pos in self.positions.items():
            if symbol in prices:
                pos.market_value = pos.quantity * prices[symbol]
                pos.unrealized_pnl = pos.quantity * (prices[symbol] - pos.avg_price)


class ExecutionModel(ABC):
    @abstractmethod
    def execute(self, order: Order, market_data: pd.DataFrame) -> Order:
        pass


class SimpleExecutionModel(ExecutionModel):
    def __init__(self, slippage_bps: float = 5, commission_per_share: float = 0.005):
        self.slippage_bps = slippage_bps
        self.commission_per_share = commission_per_share
    
    def execute(self, order: Order, market_data: pd.DataFrame) -> Order:
        if market_data.empty:
            return order
        
        current_price = market_data["close"].iloc[-1]
        
        if order.side == OrderSide.BUY:
            fill_price = current_price * (1 + self.slippage_bps / 10000)
        else:
            fill_price = current_price * (1 - self.slippage_bps / 10000)
        
        if order.order_type == OrderType.LIMIT and order.limit_price:
            if order.side == OrderSide.BUY and fill_price > order.limit_price:
                return order
            if order.side == OrderSide.SELL and fill_price < order.limit_price:
                return order
        
        commission = order.quantity * self.commission_per_share
        
        order.filled = True
        order.filled_price = fill_price
        order.filled_qty = order.quantity
        order.commission = commission
        
        return order


class MarketImpactModel(ExecutionModel):
    def __init__(self, base_impact_bps: float = 10, participation_rate: float = 0.1):
        self.base_impact_bps = base_impact_bps
        self.participation_rate = participation_rate
    
    def execute(self, order: Order, market_data: pd.DataFrame) -> Order:
        if market_data.empty or "volume" not in market_data.columns:
            return SimpleExecutionModel().execute(order, market_data)
        
        current_price = market_data["close"].iloc[-1]
        avg_volume = market_data["volume"].rolling(20).mean().iloc[-1]
        
        participation = order.quantity / max(1, avg_volume)
        impact_bps = self.base_impact_bps * np.sqrt(participation / self.participation_rate)
        
        if order.side == OrderSide.BUY:
            fill_price = current_price * (1 + impact_bps / 10000)
        else:
            fill_price = current_price * (1 - impact_bps / 10000)
        
        commission = order.quantity * 0.005
        
        order.filled = True
        order.filled_price = fill_price
        order.filled_qty = order.quantity
        order.commission = commission
        order.market_impact_bps = impact_bps
        
        return order


class Strategy(ABC):
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame, portfolio: Portfolio) -> List[Order]:
        pass
    
    @abstractmethod
    def get_required_data(self) -> List[str]:
        pass


class MomentumStrategy(Strategy):
    def __init__(self, lookback: int = 20, threshold: float = 0.02):
        self.lookback = lookback
        self.threshold = threshold
    
    def generate_signals(self, data: pd.DataFrame, portfolio: Portfolio) -> List[Order]:
        if len(data) < self.lookback:
            return []
        
        returns = data["close"].pct_change(self.lookback).iloc[-1]
        signals = []
        
        if returns > self.threshold:
            position = portfolio.get_position(data.index[-1] if hasattr(data.index, 'name') else "SPY")
            if position.quantity <= 0:
                signals.append(Order(
                    symbol="SPY",
                    quantity=100,
                    side=OrderSide.BUY
                ))
        elif returns < -self.threshold:
            position = portfolio.get_position("SPY")
            if position.quantity > 0:
                signals.append(Order(
                    symbol="SPY",
                    quantity=position.quantity,
                    side=OrderSide.SELL
                ))
        
        return signals
    
    def get_required_data(self) -> List[str]:
        return ["close"]


class MeanReversionStrategy(Strategy):
    def __init__(self, window: int = 20, num_std: float = 2.0):
        self.window = window
        self.num_std = num_std
    
    def generate_signals(self, data: pd.DataFrame, portfolio: Portfolio) -> List[Order]:
        if len(data) < self.window:
            return []
        
        close = data["close"]
        ma = close.rolling(self.window).mean().iloc[-1]
        std = close.rolling(self.window).std().iloc[-1]
        current = close.iloc[-1]
        
        signals = []
        position = portfolio.get_position("SPY")
        
        if current < ma - self.num_std * std and position.quantity <= 0:
            signals.append(Order(symbol="SPY", quantity=100, side=OrderSide.BUY))
        elif current > ma + self.num_std * std and position.quantity > 0:
            signals.append(Order(symbol="SPY", quantity=position.quantity, side=OrderSide.SELL))
        
        return signals
    
    def get_required_data(self) -> List[str]:
        return ["close"]


class BacktestEngine:
    def __init__(self, 
                 execution_model: ExecutionModel = None,
                 initial_capital: float = 1_000_000.0):
        self.execution_model = execution_model or SimpleExecutionModel()
        self.portfolio = Portfolio(initial_capital=initial_capital)
        self.trades: List[Dict] = []
        self.equity_curve: List[Dict] = []
    
    def run(self, 
            data: pd.DataFrame, 
            strategy: Strategy,
            start_date: str = None,
            end_date: str = None) -> Dict:
        
        if start_date:
            data = data[data.index >= start_date]
        if end_date:
            data = data[data.index <= end_date]
        
        self.portfolio = Portfolio(initial_capital=self.portfolio.initial_capital)
        self.trades = []
        self.equity_curve = []
        
        for i in range(len(data)):
            current_data = data.iloc[:i+1]
            current_prices = {"SPY": current_data["close"].iloc[-1]}
            
            self.portfolio.update_market_value(current_prices)
            
            signals = strategy.generate_signals(current_data, self.portfolio)
            
            for order in signals:
                filled_order = self.execution_model.execute(order, current_data)
                
                if filled_order.filled:
                    self._process_fill(filled_order, current_data.index[-1])
            
            self.equity_curve.append({
                "timestamp": current_data.index[-1],
                "total_value": self.portfolio.total_value,
                "cash": self.portfolio.cash,
                "positions_value": sum(p.market_value for p in self.portfolio.positions.values())
            })
        
        return self._generate_results()
    
    def _process_fill(self, order: Order, timestamp: pd.Timestamp):
        pos = self.portfolio.get_position(order.symbol)
        
        if order.side == OrderSide.BUY:
            new_qty = pos.quantity + order.filled_qty
            pos.avg_price = ((pos.quantity * pos.avg_price) + 
                           (order.filled_qty * order.filled_price)) / new_qty
            pos.quantity = new_qty
            self.portfolio.cash -= order.filled_qty * order.filled_price
        else:
            realized = order.filled_qty * (order.filled_price - pos.avg_price)
            pos.realized_pnl += realized
            pos.quantity -= order.filled_qty
            self.portfolio.cash += order.filled_qty * order.filled_price
            
            if pos.quantity == 0:
                pos.avg_price = 0
        
        commission = getattr(order, "commission", 0)
        self.portfolio.cash -= commission
        
        self.trades.append({
            "timestamp": timestamp,
            "symbol": order.symbol,
            "side": order.side.value,
            "quantity": order.filled_qty,
            "price": order.filled_price,
            "commission": commission,
            "market_impact_bps": getattr(order, "market_impact_bps", 0)
        })
    
    def _generate_results(self) -> Dict:
        equity_df = pd.DataFrame(self.equity_curve).set_index("timestamp")
        trades_df = pd.DataFrame(self.trades) if self.trades else pd.DataFrame()
        
        returns = equity_df["total_value"].pct_change().dropna()
        
        total_return = (equity_df["total_value"].iloc[-1] / equity_df["total_value"].iloc[0]) - 1
        annual_return = (1 + total_return) ** (252 / len(equity_df)) - 1
        volatility = returns.std() * np.sqrt(252)
        sharpe = annual_return / volatility if volatility > 0 else 0
        
        max_dd = self._max_drawdown(equity_df["total_value"])
        
        win_rate = 0
        profit_factor = 0
        if not trades_df.empty:
            trade_pnl = trades_df.groupby("symbol").apply(
                lambda x: (x[x["side"] == "sell"]["price"] * x[x["side"] == "sell"]["quantity"]).sum() -
                          (x[x["side"] == "buy"]["price"] * x[x["side"] == "buy"]["quantity"]).sum()
            )
            winners = (trade_pnl > 0).sum()
            win_rate = winners / len(trade_pnl) if len(trade_pnl) > 0 else 0
            
            gross_profit = trade_pnl[trade_pnl > 0].sum()
            gross_loss = abs(trade_pnl[trade_pnl < 0].sum())
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf
        
        return {
            "equity_curve": equity_df,
            "trades": trades_df,
            "metrics": {
                "total_return": total_return,
                "annual_return": annual_return,
                "volatility": volatility,
                "sharpe_ratio": sharpe,
                "max_drawdown": max_dd,
                "win_rate": win_rate,
                "profit_factor": profit_factor,
                "num_trades": len(trades_df),
                "final_value": self.portfolio.total_value
            }
        }
    
    def _max_drawdown(self, equity: pd.Series) -> float:
        running_max = equity.expanding().max()
        drawdown = (equity - running_max) / running_max
        return abs(drawdown.min())


class WalkForwardOptimizer:
    def __init__(self, train_window: int = 252, test_window: int = 63, step: int = 21):
        self.train_window = train_window
        self.test_window = test_window
        self.step = step
    
    def optimize(self, data: pd.DataFrame, strategy_class: type, 
                 param_grid: Dict[str, List]) -> Dict:
        best_params = None
        best_sharpe = -np.inf
        results = []
        
        for params in self._param_combinations(param_grid):
            strategy = strategy_class(**params)
            engine = BacktestEngine()
            result = engine.run(data, strategy)
            sharpe = result["metrics"]["sharpe_ratio"]
            
            results.append({"params": params, "sharpe": sharpe, "metrics": result["metrics"]})
            
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_params = params
        
        return {
            "best_params": best_params,
            "best_sharpe": best_sharpe,
            "all_results": results
        }
    
    def _param_combinations(self, param_grid: Dict[str, List]) -> List[Dict]:
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        
        from itertools import product
        return [dict(zip(keys, combo)) for combo in product(*values)]


def run_backtest_example() -> Dict:
    dates = pd.date_range("2020-01-01", "2023-12-31", freq="D")
    np.random.seed(42)
    prices = 100 * np.exp(np.random.normal(0.0005, 0.015, len(dates)).cumsum())
    
    data = pd.DataFrame({
        "open": prices * np.random.uniform(0.99, 1.01, len(dates)),
        "high": prices * np.random.uniform(1.0, 1.02, len(dates)),
        "low": prices * np.random.uniform(0.98, 1.0, len(dates)),
        "close": prices,
        "volume": np.random.randint(1_000_000, 10_000_000, len(dates))
    }, index=dates)
    
    engine = BacktestEngine(execution_model=MarketImpactModel())
    strategy = MomentumStrategy(lookback=20, threshold=0.01)
    
    result = engine.run(data, strategy)
    
    return result["metrics"]