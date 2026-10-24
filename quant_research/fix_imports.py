import os
import re

# Files to fix
files_to_fix = [
    "quant_research/api/data_providers.py",
    "quant_research/data/taq_loader.py",
    "quant_research/microstructure/analyzer.py",
    "quant_research/models/hasbrouck_var.py",
    "quant_research/risk/liquidity_model.py",
    "quant_research/strategies/evaluator.py",
    "quant_research/test_install.py",
]

# Patterns to replace
replacements = [
    (r"from config\.settings import settings", "from quant_research.config.settings import settings"),
    (r"from data\.taq_loader import", "from quant_research.data.taq_loader import"),
    (r"from microstructure\.analyzer import", "from quant_research.microstructure.analyzer import"),
    (r"from risk\.liquidity_model import", "from quant_research.risk.liquidity_model import"),
    (r"from strategies\.evaluator import", "from quant_research.strategies.evaluator import"),
    (r"from backtest\.engine import", "from quant_research.backtest.engine import"),
    (r"from models\.hasbrouck_var import", "from quant_research.models.hasbrouck_var import"),
    (r"from api\.data_providers import", "from quant_research.api.data_providers import"),
]

for filepath in files_to_fix:
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            content = f.read()
        
        original = content
        for pattern, replacement in replacements:
            content = re.sub(pattern, replacement, content)
        
        if content != original:
            with open(filepath, "w") as f:
                f.write(content)
            print(f"Fixed: {filepath}")
        else:
            print(f"No changes: {filepath}")

print("Done!")