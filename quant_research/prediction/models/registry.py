import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier
from sklearn.linear_model import Ridge, LogisticRegression, LinearRegression
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False


@dataclass
class ModelConfig:
    model_type: str = "ridge"
    target_type: str = "return"
    horizon: int = 1
    scaler: str = "robust"
    n_splits: int = 5
    test_size: float = 0.2
    random_state: int = 42
    hyperparams: Dict = field(default_factory=dict)


class BaseModel(ABC):
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.is_fitted = False

    @abstractmethod
    def _create_model(self):
        pass

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.feature_names = X.columns.tolist()
        
        if self.config.scaler == "standard":
            self.scaler = StandardScaler()
        elif self.config.scaler == "robust":
            self.scaler = RobustScaler()
        else:
            self.scaler = None
        
        if self.scaler:
            X_scaled = self.scaler.fit_transform(X)
            X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        else:
            X_scaled = X
        
        self.model = self._create_model()
        self.model.fit(X_scaled, y)
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        
        X = X[self.feature_names]
        
        if self.scaler:
            X_scaled = self.scaler.transform(X)
            X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        else:
            X_scaled = X
        
        return self.model.predict(X_scaled)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        if not hasattr(self.model, "predict_proba"):
            raise ValueError("Model doesn't support predict_proba")
        
        X = X[self.feature_names]
        if self.scaler:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X
        
        return self.model.predict_proba(X_scaled)

    def save(self, path: str):
        joblib.dump({"model": self.model, "scaler": self.scaler, 
                     "feature_names": self.feature_names, "config": self.config}, path)

    @classmethod
    def load(cls, path: str) -> "BaseModel":
        data = joblib.load(path)
        obj = cls(data["config"])
        obj.model = data["model"]
        obj.scaler = data["scaler"]
        obj.feature_names = data["feature_names"]
        obj.is_fitted = True
        return obj


class RidgeModel(BaseModel):
    def _create_model(self):
        return Ridge(**self.config.hyperparams)


class RandomForestRegModel(BaseModel):
    def _create_model(self):
        return RandomForestRegressor(random_state=self.config.random_state, **self.config.hyperparams)


class GradientBoostingModel(BaseModel):
    def _create_model(self):
        return GradientBoostingRegressor(random_state=self.config.random_state, **self.config.hyperparams)


class MLPRegModel(BaseModel):
    def _create_model(self):
        return MLPRegressor(random_state=self.config.random_state, max_iter=500, **self.config.hyperparams)


class LogisticModel(BaseModel):
    def _create_model(self):
        return LogisticRegression(random_state=self.config.random_state, max_iter=1000, **self.config.hyperparams)


class RandomForestClsModel(BaseModel):
    def _create_model(self):
        return RandomForestClassifier(random_state=self.config.random_state, **self.config.hyperparams)


class MLPClsModel(BaseModel):
    def _create_model(self):
        return MLPClassifier(random_state=self.config.random_state, max_iter=500, **self.config.hyperparams)


class XGBoostRegModel(BaseModel):
    def _create_model(self):
        if not HAS_XGB:
            raise ImportError("xgboost not installed")
        return xgb.XGBRegressor(
            random_state=self.config.random_state,
            n_jobs=-1,
            tree_method="hist",
            **self.config.hyperparams
        )


class XGBoostClsModel(BaseModel):
    def _create_model(self):
        if not HAS_XGB:
            raise ImportError("xgboost not installed")
        return xgb.XGBClassifier(
            random_state=self.config.random_state,
            n_jobs=-1,
            tree_method="hist",
            **self.config.hyperparams
        )


class LightGBMRegModel(BaseModel):
    def _create_model(self):
        if not HAS_LGB:
            raise ImportError("lightgbm not installed")
        return lgb.LGBMRegressor(
            random_state=self.config.random_state,
            n_jobs=-1,
            verbose=-1,
            **self.config.hyperparams
        )


class LightGBMClsModel(BaseModel):
    def _create_model(self):
        if not HAS_LGB:
            raise ImportError("lightgbm not installed")
        return lgb.LGBMClassifier(
            random_state=self.config.random_state,
            n_jobs=-1,
            verbose=-1,
            **self.config.hyperparams
        )


MODEL_REGISTRY = {
    "ridge": RidgeModel,
    "rf": RandomForestRegModel,
    "gb": GradientBoostingModel,
    "mlp": MLPRegModel,
    "xgb": XGBoostRegModel,
    "lgb": LightGBMRegModel,
    "logistic": LogisticModel,
    "rf_cls": RandomForestClsModel,
    "mlp_cls": MLPClsModel,
    "xgb_cls": XGBoostClsModel,
    "lgb_cls": LightGBMClsModel,
}


def create_model(config: ModelConfig) -> BaseModel:
    if config.model_type not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model type: {config.model_type}")
    return MODEL_REGISTRY[config.model_type](config)


@dataclass
class WalkForwardResult:
    train_scores: List[float]
    test_scores: List[float]
    predictions: pd.Series
    actuals: pd.Series
    feature_importance: Optional[pd.DataFrame] = None


def walk_forward_validation(model: BaseModel,
                            X: pd.DataFrame,
                            y: pd.Series,
                            n_splits: int = 5,
                            score_fn: Callable = None) -> WalkForwardResult:
    if score_fn is None:
        if model.config.target_type in ["direction"]:
            score_fn = lambda y_true, y_pred: accuracy_score(y_true, (y_pred > 0.5).astype(int))
        else:
            score_fn = lambda y_true, y_pred: -mean_squared_error(y_true, y_pred)
    
    tscv = TimeSeriesSplit(n_splits=n_splits)
    train_scores = []
    test_scores = []
    all_preds = []
    all_actuals = []
    
    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        model.fit(X_train, y_train)
        
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        
        train_scores.append(score_fn(y_train, train_pred))
        test_scores.append(score_fn(y_test, test_pred))
        
        all_preds.extend(test_pred)
        all_actuals.extend(y_test.values)
    
    predictions = pd.Series(all_preds, index=y.index[-len(all_preds):])
    actuals = pd.Series(all_actuals, index=y.index[-len(all_actuals):])
    
    feature_importance = None
    if hasattr(model.model, "feature_importances_"):
        feature_importance = pd.DataFrame({
            "feature": model.feature_names,
            "importance": model.model.feature_importances_
        }).sort_values("importance", ascending=False)
    
    return WalkForwardResult(
        train_scores=train_scores,
        test_scores=test_scores,
        predictions=predictions,
        actuals=actuals,
        feature_importance=feature_importance
    )


def evaluate_predictions(y_true: pd.Series, y_pred: np.ndarray, 
                         target_type: str = "return") -> Dict:
    results = {}
    
    if target_type in ["return", "log_return"]:
        results["mse"] = mean_squared_error(y_true, y_pred)
        results["rmse"] = np.sqrt(results["mse"])
        results["mae"] = mean_absolute_error(y_true, y_pred)
        results["directional_accuracy"] = ((y_true > 0) == (y_pred > 0)).mean()
        results["correlation"] = np.corrcoef(y_true, y_pred)[0, 1]
    elif target_type == "direction":
        results["accuracy"] = accuracy_score(y_true, (y_pred > 0.5).astype(int))
        if len(np.unique(y_pred)) > 1:
            results["auc"] = roc_auc_score(y_true, y_pred)
    elif target_type == "volatility":
        results["mse"] = mean_squared_error(y_true, y_pred)
        results["mae"] = mean_absolute_error(y_true, y_pred)
    
    return results