import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Callable
import warnings
warnings.filterwarnings("ignore")

import optuna
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, accuracy_score

from prediction.models.registry import create_model, ModelConfig, BaseModel


def objective_reg(trial: optuna.Trial, X: pd.DataFrame, y: pd.Series, 
                  model_type: str, n_splits: int = 5, random_state: int = 42) -> float:
    if model_type == "xgb":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }
    elif model_type == "lgb":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 20, 100),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }
    elif model_type == "rf":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 15),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5]),
        }
    elif model_type == "ridge":
        params = {
            "alpha": trial.suggest_float("alpha", 1e-4, 100.0, log=True),
        }
    else:
        params = {}
    
    config = ModelConfig(
        model_type=model_type,
        target_type="return",
        hyperparams=params,
        random_state=random_state
    )
    
    model = create_model(config)
    
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores = []
    
    for train_idx, val_idx in tscv.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model.fit(X_train, y_train)
        pred = model.predict(X_val)
        
        mse = mean_squared_error(y_val, pred)
        scores.append(-mse)  # Negative because Optuna minimizes
    
    return np.mean(scores)


def objective_cls(trial: optuna.Trial, X: pd.DataFrame, y: pd.Series,
                  model_type: str, n_splits: int = 5, random_state: int = 42) -> float:
    if model_type == "xgb_cls":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        }
    elif model_type == "lgb_cls":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 20, 100),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        }
    elif model_type == "rf_cls":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 15),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        }
    elif model_type == "logistic":
        params = {
            "C": trial.suggest_float("C", 1e-4, 100.0, log=True),
            "penalty": trial.suggest_categorical("penalty", ["l1", "l2"]),
            "solver": "liblinear",
        }
    else:
        params = {}
    
    config = ModelConfig(
        model_type=model_type,
        target_type="direction",
        hyperparams=params,
        random_state=random_state
    )
    
    model = create_model(config)
    
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores = []
    
    for train_idx, val_idx in tscv.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model.fit(X_train, y_train)
        pred_proba = model.predict_proba(X_val)[:, 1]
        pred = (pred_proba > 0.5).astype(int)
        
        acc = accuracy_score(y_val, pred)
        scores.append(acc)
    
    return np.mean(scores)


def optimize_hyperparams(X: pd.DataFrame, y: pd.Series,
                         model_type: str,
                         target_type: str = "return",
                         n_trials: int = 50,
                         n_splits: int = 5,
                         random_state: int = 42) -> Dict:
    study = optuna.create_study(direction="maximize")
    
    if target_type == "direction":
        study.optimize(
            lambda trial: objective_cls(trial, X, y, model_type, n_splits, random_state),
            n_trials=n_trials,
            timeout=300
        )
    else:
        study.optimize(
            lambda trial: objective_reg(trial, X, y, model_type, n_splits, random_state),
            n_trials=n_trials,
            timeout=300
        )
    
    return study.best_params


class ModelTuner:
    def __init__(self, n_trials: int = 50, n_splits: int = 5):
        self.n_trials = n_trials
        self.n_splits = n_splits
        self.best_params = {}
    
    def tune(self, X: pd.DataFrame, y: pd.Series, model_type: str, 
             target_type: str = "return") -> Dict:
        print(f"Tuning {model_type} for {target_type}...")
        params = optimize_hyperparams(
            X, y, model_type, target_type, 
            n_trials=self.n_trials, n_splits=self.n_splits
        )
        self.best_params[model_type] = params
        print(f"Best params: {params}")
        return params
    
    def get_best_model(self, model_type: str, target_type: str = "return") -> BaseModel:
        params = self.best_params.get(model_type, {})
        config = ModelConfig(
            model_type=model_type,
            target_type=target_type,
            hyperparams=params
        )
        return create_model(config)


def ensemble_predict(models: List[BaseModel], X: pd.DataFrame, 
                     weights: Optional[List[float]] = None) -> np.ndarray:
    preds = np.column_stack([m.predict(X) for m in models])
    
    if weights is None:
        weights = np.ones(len(models)) / len(models)
    
    return np.average(preds, axis=1, weights=weights)


def ensemble_predict_proba(models: List[BaseModel], X: pd.DataFrame,
                           weights: Optional[List[float]] = None) -> np.ndarray:
    preds = np.column_stack([m.predict_proba(X)[:, 1] for m in models])
    
    if weights is None:
        weights = np.ones(len(models)) / len(models)
    
    return np.average(preds, axis=1, weights=weights)