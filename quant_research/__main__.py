#!/usr/bin/env python
"""
Quant Research Platform - Integrated Alpha Strategies, Liquidity Risk, and Market Microstructure Analysis

Based on the paper: "Alpha Trading Strategies Analysis, Maximizing Alpha for Hedge Funds, 
and High Frequency Econometrics for Analyzing Price Impact of Trades, Liquidity, and Market Microstructure"
by Dr. Yogesh Malhotra (SSRN-id3306817)

This platform implements the three phases from the paper:
1. Alpha Trading Strategies Analysis (400 strategies, 7 criteria)
2. Post-Financial Crisis Strategies for Maximizing Alpha (Liquidity risk, illiquidity premium)
3. High Frequency Econometrics (Hasbrouck VAR, Market Microstructure, Price Impact)
"""

from quant_research.cli.main import main

if __name__ == "__main__":
    main()