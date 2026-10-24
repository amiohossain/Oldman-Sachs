import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from quant_research.config.settings import settings


class LiquidityTier(Enum):
    HIGHLY_LIQUID = "highly_liquid"
    MODERATELY_LIQUID = "moderately_liquid"
    LOW_LIQUIDITY = "low_liquidity"
    ILLIQUID = "illiquid"


@dataclass
class FundLiquidityProfile:
    redemption_notice_days: int
    redemption_frequency: str
    lockup_months: int
    fund_age_months: int
    aum: float
    strategy_type: str
    
    def liquidity_score(self) -> float:
        notice_score = min(self.redemption_notice_days / 90, 1.0)
        freq_map = {"daily": 1.0, "weekly": 0.75, "monthly": 0.5, "quarterly": 0.25}
        freq_score = freq_map.get(self.redemption_frequency.lower(), 0.5)
        lockup_score = min(self.lockup_months / 12, 1.0)
        age_score = min(self.fund_age_months / 60, 1.0)
        aum_score = min(np.log10(self.aum) / 12, 1.0) if self.aum > 0 else 0
        
        weights = [0.25, 0.20, 0.20, 0.15, 0.20]
        scores = [notice_score, freq_score, lockup_score, age_score, aum_score]
        return float(np.average(scores, weights=weights))

    def tier(self) -> LiquidityTier:
        score = self.liquidity_score()
        if score >= 0.75:
            return LiquidityTier.HIGHLY_LIQUID
        elif score >= 0.5:
            return LiquidityTier.MODERATELY_LIQUID
        elif score >= 0.25:
            return LiquidityTier.LOW_LIQUIDITY
        return LiquidityTier.ILLIQUID


@dataclass
class PositionLiquidity:
    symbol: str
    avg_daily_volume: float
    bid_ask_spread_bps: float
    market_depth: float
    days_to_liquidate: float
    price_impact_1pct: float
    price_impact_5pct: float
    
    def liquidity_score(self) -> float:
        volume_score = min(self.avg_daily_volume / 1e6, 1.0)
        spread_score = max(0, 1 - self.bid_ask_spread_bps / 100)
        depth_score = min(self.market_depth / 1e6, 1.0)
        impact_score = max(0, 1 - self.price_impact_1pct / 0.05)
        
        return float(np.mean([volume_score, spread_score, depth_score, impact_score]))

    def tier(self) -> LiquidityTier:
        score = self.liquidity_score()
        if score >= 0.75:
            return LiquidityTier.HIGHLY_LIQUID
        elif score >= 0.5:
            return LiquidityTier.MODERATELY_LIQUID
        elif score >= 0.25:
            return LiquidityTier.LOW_LIQUIDITY
        return LiquidityTier.ILLIQUID


class LiquidityRiskModel:
    def __init__(self):
        self.illiquidity_premium_estimate = 0.03
    
    def estimate_illiquidity_premium(self, position: PositionLiquidity, 
                                      fund: FundLiquidityProfile) -> Dict:
        position_illiquidity = 1 - position.liquidity_score()
        fund_illiquidity = 1 - fund.liquidity_score()
        
        combined_illiquidity = np.sqrt(position_illiquidity**2 + fund_illiquidity**2)
        
        premium = self.illiquidity_premium_estimate * combined_illiquidity
        
        return {
            "position_illiquidity": position_illiquidity,
            "fund_illiquidity": fund_illiquidity,
            "combined_illiquidity": combined_illiquidity,
            "estimated_annual_premium": premium,
            "premium_bps": premium * 10000,
            "position_tier": position.tier().value,
            "fund_tier": fund.tier().value
        }

    def estimate_liquidation_cost(self, position: PositionLiquidity, 
                                   shares: float, urgency: str = "normal") -> Dict:
        urgency_multiplier = {"low": 0.5, "normal": 1.0, "high": 2.0, "fire_sale": 5.0}
        mult = urgency_multiplier.get(urgency, 1.0)
        
        pct_of_adv = shares / max(1, position.avg_daily_volume)
        
        if pct_of_adv <= 0.01:
            impact = position.price_impact_1pct * pct_of_adv * 100
        elif pct_of_adv <= 0.05:
            impact = position.price_impact_1pct + position.price_impact_5pct * (pct_of_adv - 0.01) / 0.04
        else:
            impact = position.price_impact_5pct * (1 + (pct_of_adv - 0.05) * 10)
        
        total_cost = impact * mult
        spread_cost = position.bid_ask_spread_bps / 10000 / 2
        
        return {
            "market_impact_bps": impact * 10000,
            "spread_cost_bps": spread_cost * 10000,
            "total_cost_bps": (total_cost + spread_cost) * 10000,
            "days_to_liquidate": position.days_to_liquidate * max(1, pct_of_adv / 0.1),
            "urgency_multiplier": mult
        }

    def stress_test_liquidity(self, positions: List[PositionLiquidity], 
                               fund: FundLiquidityProfile,
                               redemption_shock: float = 0.2) -> Dict:
        total_aum = fund.aum
        redemption_amount = total_aum * redemption_shock
        
        position_values = [p.avg_daily_volume * 100 for p in positions]
        total_portfolio_value = sum(position_values)
        
        weights = np.array(position_values) / total_portfolio_value
        shares_to_sell = weights * redemption_amount / np.array([p.avg_daily_volume * 100 for p in positions])
        
        liquidation_costs = []
        for pos, shares in zip(positions, shares_to_sell):
            cost = self.estimate_liquidation_cost(pos, shares, "high")
            liquidation_costs.append(cost)
        
        total_cost = sum(c["total_cost_bps"] for c in liquidation_costs) / 10000
        
        return {
            "redemption_shock": redemption_shock,
            "redemption_amount": redemption_amount,
            "total_liquidation_cost_bps": total_cost * 10000,
            "position_costs": liquidation_costs,
            "survival_probability": self._survival_probability(total_cost, fund),
            "gate_recommendation": total_cost > 0.05
        }

    def _survival_probability(self, liquidation_cost: float, fund: FundLiquidityProfile) -> float:
        base_survival = 1 - liquidation_cost * 10
        liquidity_buffer = fund.liquidity_score() * 0.3
        return float(np.clip(base_survival + liquidity_buffer, 0, 1))


class MacroLiquidityIndicators:
    @staticmethod
    def compute_indicators(market_data: pd.DataFrame) -> Dict:
        indicators = {}
        
        if "credit_spread" in market_data.columns:
            indicators["credit_spread"] = market_data["credit_spread"].iloc[-1]
            indicators["credit_spread_change"] = market_data["credit_spread"].pct_change().iloc[-1]
        
        if "bid_ask_spread" in market_data.columns:
            indicators["avg_bid_ask_spread"] = market_data["bid_ask_spread"].mean()
            indicators["spread_volatility"] = market_data["bid_ask_spread"].std()
        
        if "volume" in market_data.columns:
            indicators["volume_trend"] = market_data["volume"].rolling(20).mean().iloc[-1]
            indicators["volume_ratio"] = market_data["volume"].iloc[-1] / indicators["volume_trend"]
        
        if "yield_curve" in market_data.columns:
            indicators["term_spread"] = market_data["yield_curve"].iloc[-1]
        
        if "vix" in market_data.columns:
            indicators["vix"] = market_data["vix"].iloc[-1]
            indicators["vix_percentile"] = MacroLiquidityIndicators._percentile_rank(
                market_data["vix"], market_data["vix"].iloc[-1]
            )
        
        indicators["composite_liquidity_index"] = MacroLiquidityIndicators._composite_index(indicators)
        
        return indicators

    @staticmethod
    def _percentile_rank(series: pd.Series, value: float) -> float:
        return (series <= value).mean()

    @staticmethod
    def _composite_index(indicators: Dict) -> float:
        weights = {
            "credit_spread": -0.25,
            "avg_bid_ask_spread": -0.20,
            "spread_volatility": -0.15,
            "volume_ratio": 0.20,
            "term_spread": 0.10,
            "vix_percentile": -0.10
        }
        
        score = 0
        for key, weight in weights.items():
            if key in indicators:
                score += weight * indicators[key]
        
        return float(np.clip(score, -1, 1))


class LiquiditySpectrumPortfolio:
    def __init__(self):
        self.buckets = {
            LiquidityTier.HIGHLY_LIQUID: [],
            LiquidityTier.MODERATELY_LIQUID: [],
            LiquidityTier.LOW_LIQUIDITY: [],
            LiquidityTier.ILLIQUID: []
        }
    
    def add_position(self, position: PositionLiquidity, weight: float):
        tier = position.tier()
        self.buckets[tier].append({"position": position, "weight": weight})
    
    def get_allocation(self) -> Dict:
        total_weight = sum(
            sum(b["weight"] for b in bucket) 
            for bucket in self.buckets.values()
        )
        
        if total_weight == 0:
            return {tier.value: 0 for tier in LiquidityTier}
        
        return {
            tier.value: sum(b["weight"] for b in bucket) / total_weight
            for tier, bucket in self.buckets.items()
        }
    
    def rebalance_for_liquidity(self, target_liquidity: float) -> Dict:
        current = self.get_allocation()
        
        if current[LiquidityTier.HIGHLY_LIQUID.value] < target_liquidity:
            return {
                "action": "increase_liquid_allocation",
                "current_liquid": current[LiquidityTier.HIGHLY_LIQUID.value],
                "target_liquid": target_liquidity,
                "recommendation": "Shift from illiquid to liquid strategies"
            }
        return {"action": "hold", "current_liquid": current[LiquidityTier.HIGHLY_LIQUID.value]}


def run_liquidity_analysis() -> Dict:
    position = PositionLiquidity(
        symbol="AAPL",
        avg_daily_volume=50_000_000,
        bid_ask_spread_bps=2,
        market_depth=10_000_000,
        days_to_liquidate=0.5,
        price_impact_1pct=0.001,
        price_impact_5pct=0.008
    )
    
    fund = FundLiquidityProfile(
        redemption_notice_days=30,
        redemption_frequency="monthly",
        lockup_months=0,
        fund_age_months=36,
        aum=1_000_000_000,
        strategy_type="equity_long_short"
    )
    
    model = LiquidityRiskModel()
    
    premium = model.estimate_illiquidity_premium(position, fund)
    liquidation = model.estimate_liquidation_cost(position, 100_000, "normal")
    stress = model.stress_test_liquidity([position], fund, 0.2)
    
    return {
        "illiquidity_premium": premium,
        "liquidation_cost": liquidation,
        "stress_test": stress
    }