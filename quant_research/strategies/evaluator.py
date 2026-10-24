import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import json
from quant_research.config.settings import settings


class StrategyCategory(Enum):
    EQUITY_LONG_SHORT = "equity_long_short"
    GLOBAL_MACRO = "global_macro"
    EVENT_DRIVEN = "event_driven"
    RELATIVE_VALUE = "relative_value"
    CTA_MANAGED_FUTURES = "cta_managed_futures"
    FIXED_INCOME_ARB = "fixed_income_arb"
    MULTI_STRATEGY = "multi_strategy"
    CRYPTO = "crypto"
    OTHER = "other"


@dataclass
class StrategyMetadata:
    name: str
    category: StrategyCategory
    description: str
    source_publication: str
    publication_date: str
    author: str = ""
    region: str = "global"
    asset_class: str = "equity"
    frequency: str = "daily"


@dataclass
class StrategyScores:
    alpha: float = 0.0
    plausibility: float = 0.0
    technical_profile: float = 0.0
    operational_framework: float = 0.0
    portfolio_risk: float = 0.0
    market_crowding: float = 0.0
    sector_capacity: float = 0.0
    alternative_reusability: float = 0.0
    
    def composite_score(self, weights: Dict[str, float] = None) -> float:
        if weights is None:
            weights = {
                "alpha": settings.strategy.alpha_weight,
                "plausibility": settings.strategy.plausibility_weight,
                "technical_profile": settings.strategy.technical_profile_weight,
                "operational_framework": settings.strategy.operational_framework_weight,
                "portfolio_risk": settings.strategy.portfolio_risk_weight,
                "market_crowding": settings.strategy.market_crowding_weight,
                "sector_capacity": settings.strategy.sector_capacity_weight
            }
        
        scores = [
            self.alpha, self.plausibility, self.technical_profile,
            self.operational_framework, self.portfolio_risk,
            self.market_crowding, self.sector_capacity
        ]
        weight_vals = [weights.get(k, 0) for k in [
            "alpha", "plausibility", "technical_profile", "operational_framework",
            "portfolio_risk", "market_crowding", "sector_capacity"
        ]]
        
        total_weight = sum(weight_vals)
        if total_weight == 0:
            return 0.0
        
        return float(np.average(scores, weights=weight_vals))

    def to_dict(self) -> Dict:
        return {
            "alpha": self.alpha,
            "plausibility": self.plausibility,
            "technical_profile": self.technical_profile,
            "operational_framework": self.operational_framework,
            "portfolio_risk": self.portfolio_risk,
            "market_crowding": self.market_crowding,
            "sector_capacity": self.sector_capacity,
            "alternative_reusability": self.alternative_reusability,
            "composite": self.composite_score()
        }


@dataclass
class TradingStrategy:
    metadata: StrategyMetadata
    scores: StrategyScores
    backtest_results: Dict = field(default_factory=dict)
    risk_metrics: Dict = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            "metadata": {
                "name": self.metadata.name,
                "category": self.metadata.category.value,
                "description": self.metadata.description,
                "source_publication": self.metadata.source_publication,
                "publication_date": self.metadata.publication_date,
                "author": self.metadata.author,
                "region": self.metadata.region,
                "asset_class": self.metadata.asset_class,
                "frequency": self.metadata.frequency
            },
            "scores": self.scores.to_dict(),
            "backtest_results": self.backtest_results,
            "risk_metrics": self.risk_metrics,
            "tags": self.tags
        }


class StrategyEvaluator:
    def __init__(self, criteria_weights: Dict[str, float] = None):
        self.criteria_weights = criteria_weights or {
            "alpha": settings.strategy.alpha_weight,
            "plausibility": settings.strategy.plausibility_weight,
            "technical_profile": settings.strategy.technical_profile_weight,
            "operational_framework": settings.strategy.operational_framework_weight,
            "portfolio_risk": settings.strategy.portfolio_risk_weight,
            "market_crowding": settings.strategy.market_crowding_weight,
            "sector_capacity": settings.strategy.sector_capacity_weight
        }
    
    def evaluate(self, strategy: TradingStrategy) -> StrategyScores:
        scores = StrategyScores()
        
        scores.alpha = self._score_alpha(strategy)
        scores.plausibility = self._score_plausibility(strategy)
        scores.technical_profile = self._score_technical_profile(strategy)
        scores.operational_framework = self._score_operational_framework(strategy)
        scores.portfolio_risk = self._score_portfolio_risk(strategy)
        scores.market_crowding = self._score_market_crowding(strategy)
        scores.sector_capacity = self._score_sector_capacity(strategy)
        scores.alternative_reusability = self._score_alternative_reusability(strategy)
        
        return scores
    
    def _score_alpha(self, strategy: TradingStrategy) -> float:
        if "sharpe_ratio" in strategy.backtest_results:
            sharpe = strategy.backtest_results["sharpe_ratio"]
            return float(np.clip(sharpe / 3.0, 0, 1))
        if "annual_return" in strategy.backtest_results:
            ret = strategy.backtest_results["annual_return"]
            return float(np.clip(ret / 0.20, 0, 1))
        return 0.5
    
    def _score_plausibility(self, strategy: TradingStrategy) -> float:
        score = 0.5
        if strategy.metadata.source_publication:
            top_journals = ["JF", "RFS", "JFE", "QF", "JBF", "MF"]
            if any(j in strategy.metadata.source_publication for j in top_journals):
                score += 0.2
        if strategy.risk_metrics.get("economic_rationale", False):
            score += 0.2
        if strategy.risk_metrics.get("out_of_sample_test", False):
            score += 0.1
        return float(np.clip(score, 0, 1))
    
    def _score_technical_profile(self, strategy: TradingStrategy) -> float:
        score = 0.5
        if strategy.backtest_results.get("max_drawdown", 1) < 0.15:
            score += 0.2
        if strategy.backtest_results.get("win_rate", 0) > 0.55:
            score += 0.15
        if strategy.backtest_results.get("profit_factor", 0) > 1.5:
            score += 0.15
        return float(np.clip(score, 0, 1))
    
    def _score_operational_framework(self, strategy: TradingStrategy) -> float:
        score = 0.5
        freq = strategy.metadata.frequency.lower()
        if freq in ["daily", "weekly"]:
            score += 0.2
        elif freq == "monthly":
            score += 0.1
        if strategy.risk_metrics.get("capacity_estimate", 0) > 1e9:
            score += 0.2
        if strategy.risk_metrics.get("turnover", 1) < 12:
            score += 0.1
        return float(np.clip(score, 0, 1))
    
    def _score_portfolio_risk(self, strategy: TradingStrategy) -> float:
        score = 0.5
        if "var_95" in strategy.risk_metrics:
            var = strategy.risk_metrics["var_95"]
            if var < 0.02:
                score += 0.2
            elif var < 0.05:
                score += 0.1
        if strategy.risk_metrics.get("leverage", 1) < 2:
            score += 0.15
        if strategy.risk_metrics.get("liquidity_score", 0) > 0.7:
            score += 0.15
        return float(np.clip(score, 0, 1))
    
    def _score_market_crowding(self, strategy: TradingStrategy) -> float:
        score = 0.5
        if strategy.risk_metrics.get("crowding_score", 0.5) < 0.3:
            score += 0.3
        elif strategy.risk_metrics.get("crowding_score", 0.5) < 0.6:
            score += 0.1
        if strategy.metadata.category in [StrategyCategory.CTA_MANAGED_FUTURES, StrategyCategory.GLOBAL_MACRO]:
            score += 0.1
        return float(np.clip(score, 0, 1))
    
    def _score_sector_capacity(self, strategy: TradingStrategy) -> float:
        score = 0.5
        capacity = strategy.risk_metrics.get("capacity_estimate", 1e8)
        if capacity > 1e10:
            score += 0.3
        elif capacity > 1e9:
            score += 0.2
        elif capacity > 1e8:
            score += 0.1
        return float(np.clip(score, 0, 1))
    
    def _score_alternative_reusability(self, strategy: TradingStrategy) -> float:
        score = 0.5
        if len(strategy.tags) > 3:
            score += 0.2
        if "alpha" in strategy.tags and "hedge" in strategy.tags:
            score += 0.15
        if strategy.risk_metrics.get("modular_components", False):
            score += 0.15
        return float(np.clip(score, 0, 1))


class StrategyDatabase:
    def __init__(self):
        self.strategies: List[TradingStrategy] = []
        self.evaluator = StrategyEvaluator()
    
    def add_strategy(self, strategy: TradingStrategy):
        self.strategies.append(strategy)
    
    def load_from_json(self, filepath: str):
        with open(filepath, 'r') as f:
            data = json.load(f)
        for item in data:
            metadata = StrategyMetadata(**item["metadata"])
            scores = StrategyScores(**item["scores"])
            strategy = TradingStrategy(
                metadata=metadata,
                scores=scores,
                backtest_results=item.get("backtest_results", {}),
                risk_metrics=item.get("risk_metrics", {}),
                tags=item.get("tags", [])
            )
            self.add_strategy(strategy)
    
    def save_to_json(self, filepath: str):
        with open(filepath, 'w') as f:
            json.dump([s.to_dict() for s in self.strategies], f, indent=2)
    
    def rank_strategies(self, top_n: int = 50) -> List[TradingStrategy]:
        ranked = sorted(
            self.strategies,
            key=lambda s: s.scores.composite_score(self.evaluator.criteria_weights),
            reverse=True
        )
        return ranked[:top_n]
    
    def filter_by_category(self, category: StrategyCategory) -> List[TradingStrategy]:
        return [s for s in self.strategies if s.metadata.category == category]
    
    def filter_by_score(self, min_composite: float) -> List[TradingStrategy]:
        return [s for s in self.strategies if s.scores.composite_score() >= min_composite]
    
    def get_category_stats(self) -> Dict:
        stats = {}
        for cat in StrategyCategory:
            cat_strats = self.filter_by_category(cat)
            if cat_strats:
                stats[cat.value] = {
                    "count": len(cat_strats),
                    "avg_composite": np.mean([s.scores.composite_score() for s in cat_strats]),
                    "avg_alpha": np.mean([s.scores.alpha for s in cat_strats]),
                    "best": max(cat_strats, key=lambda s: s.scores.composite_score()).metadata.name
                }
        return stats


def create_sample_strategies(n: int = 100) -> List[TradingStrategy]:
    np.random.seed(42)
    categories = list(StrategyCategory)
    strategies = []
    
    for i in range(n):
        cat = np.random.choice(categories)
        metadata = StrategyMetadata(
            name=f"{cat.value.replace('_', ' ').title()} Strategy {i+1}",
            category=cat,
            description=f"Sample {cat.value} strategy with synthetic backtest",
            source_publication=np.random.choice(["JF", "RFS", "JFE", "QF", "SSA Quarterly", "Working Paper"]),
            publication_date=f"202{np.random.randint(0,5)}-{np.random.randint(1,13):02d}",
            frequency=np.random.choice(["daily", "weekly", "monthly"])
        )
        
        scores = StrategyScores(
            alpha=np.random.beta(2, 2),
            plausibility=np.random.beta(3, 2),
            technical_profile=np.random.beta(2, 3),
            operational_framework=np.random.beta(2, 2),
            portfolio_risk=np.random.beta(3, 3),
            market_crowding=np.random.beta(2, 3),
            sector_capacity=np.random.beta(2, 2),
            alternative_reusability=np.random.beta(2, 2)
        )
        
        backtest = {
            "sharpe_ratio": np.random.normal(1.5, 0.5),
            "annual_return": np.random.normal(0.15, 0.08),
            "max_drawdown": np.random.beta(2, 5) * 0.3,
            "win_rate": np.random.beta(5, 4),
            "profit_factor": np.random.gamma(2, 1)
        }
        
        risk = {
            "var_95": np.random.beta(2, 5) * 0.1,
            "leverage": np.random.uniform(1, 3),
            "liquidity_score": np.random.beta(3, 2),
            "crowding_score": np.random.beta(2, 3),
            "capacity_estimate": 10**np.random.uniform(7, 11),
            "turnover": np.random.uniform(2, 50),
            "economic_rationale": np.random.choice([True, False]),
            "out_of_sample_test": np.random.choice([True, False]),
            "modular_components": np.random.choice([True, False])
        }
        
        tags = np.random.choice(
            ["alpha", "hedge", "momentum", "mean_reversion", "arbitrage", "macro", "event_driven", "systematic"],
            size=np.random.randint(1, 5), replace=False
        ).tolist()
        
        strategies.append(TradingStrategy(
            metadata=metadata,
            scores=scores,
            backtest_results=backtest,
            risk_metrics=risk,
            tags=tags
        ))
    
    return strategies


def run_strategy_analysis() -> Dict:
    db = StrategyDatabase()
    
    for strat in create_sample_strategies(200):
        db.add_strategy(strat)
    
    top_20 = db.rank_strategies(20)
    category_stats = db.get_category_stats()
    
    return {
        "total_strategies": len(db.strategies),
        "top_20": [s.to_dict() for s in top_20],
        "category_stats": category_stats,
        "weight_config": db.evaluator.criteria_weights
    }