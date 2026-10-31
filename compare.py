"""
Compare every strategy you tried:   python compare.py

Runs every  decide_*  function in strategy.py (and the equal-weight benchmark).
The training data is split in two:
  * "fit"        - the first 10 years: build and tune your ideas ONLY here
  * "validation" - all the years after that: the honest check, like a mini secret period
The table shows validation numbers, plus how much the Sharpe ratio dropped
from fit to validation (that is what "Generalization" is scored on).

Saves comparison.csv and comparison.png - put them in your slides.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

import engine
import strategy

HERE = Path(__file__).resolve().parent
FIT_DAYS = 10 * 252    # 10 years of trading days
SHOW = ["total_return_%", "sharpe", "sortino", "calmar", "max_drawdown_%", "trading_accuracy_%",
        "costs_paid_%", "avg_exposure_%", "fit_sharpe", "generalization"]


def main():
    data = engine.load(HERE / "data" / "train.csv")
    days = data["close"].index
    split = days[min(FIT_DAYS, len(days) - 1)]

    candidates = {n: f for n, f in vars(strategy).items() if n.startswith("decide_") and callable(f)}
    if strategy.decide not in candidates.values():
        candidates["decide"] = strategy.decide
    candidates["BENCHMARK equal weight"] = engine.equal_weight

    rows, curves = {}, {}
    for name, fn in candidates.items():
        print(f"running {name} ...")
        res = engine.backtest(fn, data, verbose=False)
        fit, val = res["log"][res["log"].index < split], res["log"][res["log"].index >= split]
        m = engine.metrics(val)
        m["fit_sharpe"] = engine.metrics(fit)["sharpe"]
        m["generalization"] = m["sharpe"] - m["fit_sharpe"]
        label = name + ("  <- submitted" if fn is strategy.decide else "")
        rows[label], curves[label] = m, res["equity"]

    table = pd.DataFrame(rows).T[SHOW].sort_values("sharpe", ascending=False)
    pd.set_option("display.width", 200)
    print(f"\nVALIDATION results (days {split}..{days[-1]}), best Sharpe first:\n")
    print(table.round(2).to_string())
    table.round(3).to_csv(HERE / "comparison.csv")

    fig, ax = plt.subplots(figsize=(11, 5))
    for label, eq in curves.items():
        ax.plot(eq.index, eq.values / 1e5, label=label,
                **({"color": "black", "linestyle": "--"} if label.startswith("BENCHMARK") else {}))
    ax.axvline(split, color="grey", linestyle=":")
    ax.text(split, ax.get_ylim()[1], " validation ->", va="top", color="grey")
    ax.set_title(f"{strategy.TEAM_NAME}: strategies compared")
    ax.set_xlabel("day")
    ax.set_ylabel("portfolio value (Rs lakh)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(HERE / "comparison.png", dpi=120)
    print("\nSaved comparison.csv and comparison.png")
    plt.show()


if __name__ == "__main__":
    main()
