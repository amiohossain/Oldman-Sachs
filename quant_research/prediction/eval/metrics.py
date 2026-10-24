import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


def compute_metrics(y_true: pd.Series, y_pred: np.ndarray, 
                    target_type: str = "return") -> Dict:
    results = {}
    
    if target_type in ["return", "log_return"]:
        mse = np.mean((y_true - y_pred) ** 2)
        results["mse"] = mse
        results["rmse"] = np.sqrt(mse)
        results["mae"] = np.mean(np.abs(y_true - y_pred))
        results["directional_accuracy"] = ((y_true > 0) == (y_pred > 0)).mean()
        results["correlation"] = np.corrcoef(y_true, y_pred)[0, 1] if len(y_true) > 1 else 0
        results["r2"] = 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - y_true.mean()) ** 2)
    elif target_type == "direction":
        from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score
        y_pred_binary = (y_pred > 0.5).astype(int)
        results["accuracy"] = accuracy_score(y_true, y_pred_binary)
        results["precision"] = precision_score(y_true, y_pred_binary, zero_division=0)
        results["recall"] = recall_score(y_true, y_pred_binary, zero_division=0)
        if len(np.unique(y_pred)) > 1:
            results["auc"] = roc_auc_score(y_true, y_pred)
    elif target_type == "volatility":
        results["mse"] = np.mean((y_true - y_pred) ** 2)
        results["mae"] = np.mean(np.abs(y_true - y_pred))
    
    return results


def plot_predictions(y_true: pd.Series, y_pred: np.ndarray, 
                     dates: pd.DatetimeIndex = None, title: str = "Predictions vs Actuals"):
    if dates is None:
        dates = y_true.index
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    axes[0].plot(dates, y_true, label="Actual", alpha=0.7)
    axes[0].plot(dates, y_pred, label="Predicted", alpha=0.7)
    axes[0].set_title(f"{title} - Time Series")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].scatter(y_true, y_pred, alpha=0.5)
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    axes[1].plot(lims, lims, 'r--', alpha=0.5)
    axes[1].set_xlabel("Actual")
    axes[1].set_ylabel("Predicted")
    axes[1].set_title("Actual vs Predicted Scatter")
    axes[1].grid(True, alpha=0.3)
    
    residuals = y_true - y_pred
    axes[2].plot(dates, residuals, alpha=0.7)
    axes[2].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    axes[2].set_title("Residuals")
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_feature_importance(importance_df: pd.DataFrame, top_n: int = 20, title: str = "Feature Importance"):
    if importance_df is None or importance_df.empty:
        return None
    
    top_features = importance_df.head(top_n).sort_values("importance", ascending=True)
    
    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.3)))
    ax.barh(top_features["feature"], top_features["importance"])
    ax.set_title(title)
    ax.set_xlabel("Importance")
    plt.tight_layout()
    return fig


def plot_cumulative_returns(y_true: pd.Series, y_pred: np.ndarray, 
                           dates: pd.DatetimeIndex = None, title: str = "Strategy Returns"):
    if dates is None:
        dates = y_true.index
    
    strategy_returns = np.sign(y_pred) * y_true
    buy_hold = y_true.cumsum()
    strategy = strategy_returns.cumsum()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(dates, buy_hold, label="Buy & Hold", alpha=0.7)
    ax.plot(dates, strategy, label="Strategy", alpha=0.7)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def backtest_strategy(y_true: pd.Series, y_pred: np.ndarray, 
                      dates: pd.DatetimeIndex = None,
                      transaction_cost: float = 0.001,
                      horizon: int = 1) -> Dict:
    if dates is None:
        dates = y_true.index
    
    positions = np.sign(y_pred)
    strategy_returns = positions * y_true
    net_returns = strategy_returns - np.abs(np.diff(np.concatenate([[0], positions]))) * transaction_cost
    
    # For horizon > 1, we need to adjust for overlapping predictions
    # Only take non-overlapping trades (every horizon steps)
    if horizon > 1:
        trade_mask = np.zeros(len(positions), dtype=bool)
        trade_mask[::horizon] = True
        net_returns = net_returns * trade_mask.astype(float)
    
    buy_hold = (1 + y_true).cumprod()
    strategy_eq = (1 + net_returns).cumprod()
    
    total_return = strategy_eq.iloc[-1] - 1
    annual_return = (1 + total_return) ** (252 / len(strategy_eq)) - 1
    volatility = net_returns[net_returns != 0].std() * np.sqrt(252 / max(1, horizon)) if len(net_returns[net_returns != 0]) > 0 else 0
    sharpe = annual_return / volatility if volatility > 0 else 0
    
    running_max = strategy_eq.expanding().max()
    drawdown = (strategy_eq - running_max) / running_max
    max_dd = drawdown.min()
    
    win_rate = (net_returns[net_returns != 0] > 0).mean() if len(net_returns[net_returns != 0]) > 0 else 0
    profit_factor = (net_returns[net_returns > 0].sum() / abs(net_returns[net_returns < 0].sum())) if (net_returns < 0).any() else np.inf
    
    return {
        "total_return": total_return,
        "annual_return": annual_return,
        "volatility": volatility,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "num_trades": int((np.diff(positions) != 0).sum()),
        "equity_curve": strategy_eq,
        "buy_hold_curve": buy_hold
    }


def generate_report(model_results: Dict, save_path: str = None) -> str:
    report = "# Prediction Model Report\n\n"
    
    for symbol, result in model_results.items():
        report += f"## {symbol}\n\n"
        
        if "metrics" in result:
            report += "### Metrics\n"
            for k, v in result["metrics"].items():
                if isinstance(v, float):
                    report += f"- {k}: {v:.4f}\n"
                else:
                    report += f"- {k}: {v}\n"
            report += "\n"
        
        if "backtest" in result:
            report += "### Backtest\n"
            for k, v in result["backtest"].items():
                if isinstance(v, float):
                    report += f"- {k}: {v:.4f}\n"
                elif k not in ["equity_curve", "buy_hold_curve"]:
                    report += f"- {k}: {v}\n"
            report += "\n"
        
        if "feature_importance" in result and result["feature_importance"] is not None:
            report += "### Top Features\n"
            top = result["feature_importance"].head(10)
            for _, row in top.iterrows():
                report += f"- {row['feature']}: {row['importance']:.4f}\n"
            report += "\n"
    
    if save_path:
        with open(save_path, "w") as f:
            f.write(report)
    
    return report