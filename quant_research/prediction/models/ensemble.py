import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor

from prediction.models.registry import create_model, ModelConfig, BaseModel


@dataclass
class EnsembleConfig:
    models: List[str] = field(default_factory=lambda: ["xgb", "lgb", "ridge"])
    weights: Optional[List[float]] = None
    use_stacking: bool = False
    meta_model: str = "ridge"
    n_folds: int = 5


class EnsembleModel:
    def __init__(self, config: EnsembleConfig, model_configs: Dict[str, ModelConfig]):
        self.config = config
        self.model_configs = model_configs
        self.base_models = {}
        self.meta_model = None
        self.is_fitted = False
        self.feature_names = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.feature_names = X.columns.tolist()
        
        if self.config.use_stacking:
            self._fit_stacking(X, y)
        else:
            self._fit_ensemble(X, y)
        
        self.is_fitted = True
        return self

    def _fit_ensemble(self, X: pd.DataFrame, y: pd.Series):
        for model_type in self.config.models:
            if model_type in self.model_configs:
                model = create_model(self.model_configs[model_type])
                model.fit(X, y)
                self.base_models[model_type] = model

    def _fit_stacking(self, X: pd.DataFrame, y: pd.Series):
        n_samples = len(X)
        n_models = len(self.config.models)
        oof_preds = np.zeros((n_samples, n_models))
        
        tscv = TimeSeriesSplit(n_splits=self.config.n_folds)
        
        for i, model_type in enumerate(self.config.models):
            if model_type not in self.model_configs:
                continue
            
            model = create_model(self.model_configs[model_type])
            
            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train = y.iloc[train_idx]
                
                model.fit(X_train, y_train)
                oof_preds[val_idx, i] = model.predict(X_val)
            
            # Fit on full data
            model.fit(X, y)
            self.base_models[model_type] = model
        
        # Train meta-model
        if self.config.meta_model == "ridge":
            self.meta_model = Ridge(alpha=1.0)
        elif self.config.meta_model == "rf":
            self.meta_model = RandomForestRegressor(n_estimators=100, random_state=42)
        else:
            self.meta_model = Ridge(alpha=1.0)
        
        self.meta_model.fit(oof_preds, y)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        
        X = X[self.feature_names]
        
        if self.config.use_stacking:
            base_preds = np.column_stack([
                m.predict(X) for m in self.base_models.values()
            ])
            return self.meta_model.predict(base_preds)
        else:
            preds = np.column_stack([
                m.predict(X) for m in self.base_models.values()
            ])
            
            if self.config.weights:
                weights = np.array(self.config.weights)
                weights = weights / weights.sum()
            else:
                weights = np.ones(preds.shape[1]) / preds.shape[1]
            
            return np.average(preds, axis=1, weights=weights)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        
        X = X[self.feature_names]
        
        if self.config.use_stacking:
            base_preds = np.column_stack([
                m.predict_proba(X)[:, 1] if hasattr(m, 'predict_proba') else m.predict(X)
                for m in self.base_models.values()
            ])
            return self.meta_model.predict(base_preds)
        else:
            preds = np.column_stack([
                m.predict_proba(X)[:, 1] if hasattr(m, 'predict_proba') else m.predict(X)
                for m in self.base_models.values()
            ])
            
            if self.config.weights:
                weights = np.array(self.config.weights)
                weights = weights / weights.sum()
            else:
                weights = np.ones(preds.shape[1]) / preds.shape[1]
            
            return np.average(preds, axis=1, weights=weights)


class RegimeDetector:
    def __init__(self, n_regimes: int = 3):
        self.n_regimes = n_regimes
        self.regime_models = {}
        self.regime_labels = None

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
        
        # Use trend and volatility features for regime detection
        regime_features = []
        
        if "trend_strength" in X.columns:
            regime_features.append(X["trend_strength"].fillna(0))
        if "volatility_20" in X.columns:
            regime_features.append(X["volatility_20"].fillna(0))
        if "vix" in X.columns:
            regime_features.append(X["vix"].fillna(X["vix"].median()))
        if "regime_bull" in X.columns:
            regime_features.append(X["regime_bull"])
        
        if not regime_features:
            # Fallback: use returns
            returns = X.get("return_20", X.iloc[:, 0])
            regime_features = [returns.fillna(0), returns.rolling(20).std().fillna(0)]
        
        regime_data = np.column_stack(regime_features)
        regime_data = np.nan_to_num(regime_data)
        
        scaler = StandardScaler()
        regime_data_scaled = scaler.fit_transform(regime_data)
        
        kmeans = KMeans(n_clusters=self.n_regimes, random_state=42, n_init=10)
        self.regime_labels = kmeans.fit_predict(regime_data_scaled)
        
        self.scaler = scaler
        self.kmeans = kmeans
        
        return self

    def predict_regime(self, X: pd.DataFrame) -> np.ndarray:
        regime_features = []
        
        if "trend_strength" in X.columns:
            regime_features.append(X["trend_strength"].fillna(0))
        if "volatility_20" in X.columns:
            regime_features.append(X["volatility_20"].fillna(0))
        if "vix" in X.columns:
            regime_features.append(X["vix"].fillna(X["vix"].median()))
        if "regime_bull" in X.columns:
            regime_features.append(X["regime_bull"])
        
        if not regime_features:
            returns = X.get("return_20", X.iloc[:, 0])
            regime_features = [returns.fillna(0), returns.rolling(20).std().fillna(0)]
        
        regime_data = np.column_stack(regime_features)
        regime_data = np.nan_to_num(regime_data)
        regime_data_scaled = self.scaler.transform(regime_data)
        
        return self.kmeans.predict(regime_data_scaled)

    def get_regime_names(self) -> Dict[int, str]:
        # Map cluster IDs to regime names based on characteristics
        return {
            0: "bull_low_vol",
            1: "bear_high_vol",
            2: "sideways"
        }


class RegimeAwareModel:
    def __init__(self, base_model_class, regime_detector: RegimeDetector = None):
        self.base_model_class = base_model_class
        self.regime_detector = regime_detector or RegimeDetector()
        self.regime_models = {}
        self.is_fitted = False

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.regime_detector.fit(X, y)
        regimes = self.regime_detector.regime_labels
        
        for regime_id in range(self.regime_detector.n_regimes):
            mask = regimes == regime_id
            if mask.sum() > 50:  # Minimum samples
                model = self.base_model_class()
                model.fit(X[mask], y[mask])
                self.regime_models[regime_id] = model
        
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        
        regimes = self.regime_detector.predict_regime(X)
        preds = np.zeros(len(X))
        
        for regime_id, model in self.regime_models.items():
            mask = regimes == regime_id
            if mask.any():
                preds[mask] = model.predict(X[mask])
        
        # Fill any missing with global average
        if (preds == 0).all():
            # Use first model as fallback
            first_model = list(self.regime_models.values())[0]
            preds = first_model.predict(X)
        
        return preds


def create_ensemble_config(model_type: str = "xgb") -> Dict[str, ModelConfig]:
    configs = {}
    
    if model_type == "xgb":
        configs["xgb"] = ModelConfig(model_type="xgb", hyperparams={
            "n_estimators": 300, "max_depth": 6, "learning_rate": 0.05,
            "subsample": 0.8, "colsample_bytree": 0.8, "reg_alpha": 1.0, "reg_lambda": 0.01
        })
        configs["lgb"] = ModelConfig(model_type="lgb", hyperparams={
            "n_estimators": 200, "max_depth": 5, "learning_rate": 0.05,
            "subsample": 0.8, "colsample_bytree": 0.8, "num_leaves": 50
        })
        configs["ridge"] = ModelConfig(model_type="ridge", hyperparams={"alpha": 10.0})
    
    return configs