"""Test your strategy.py on the training data:   python run_backtest.py"""
import time
from pathlib import Path

import engine
import strategy

HERE = Path(__file__).resolve().parent

data = engine.load(HERE / "data" / "train.csv")
print(f"Team: {strategy.TEAM_NAME}")

t0 = time.time()
result = engine.backtest(strategy.decide, data)
print(f"Run time: {time.time() - t0:.1f}s")

print("Benchmark (equal-weight buy & hold):")
bench = engine.backtest(engine.equal_weight, data)

engine.plot(result, title=strategy.TEAM_NAME, benchmark=bench)
