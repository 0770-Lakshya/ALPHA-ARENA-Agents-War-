"""
AGENTS WAR - your trading agent.  This is the ONLY code file you submit.

How to work:
  1. Write every idea you try as its own function named  decide_<something>(history).
  2. Run  python compare.py  to test all of them side by side (table + chart for your slides).
  3. Point  decide  (bottom of this file) at the one you want us to run on the secret period.

Rules for this file:
  * `decide` must exist - that is the function we run.
  * Allowed imports: python standard library, numpy, pandas, scikit-learn.
  * No internet, no reading other files, no hard-coded prices or day numbers
    (your agent must not behave differently just because of which day number it is).
  * On average your agent must keep at least 30% of the money invested.
  * A full run (training + secret days) must finish within 3 minutes.
"""
import numpy as np
import pandas as pd

TEAM_NAME = "<Your team name here>"

# history: dict with keys "open", "high", "low", "close", "volume".
#          Each one is a pandas DataFrame: rows = days so far (last row = today),
#          columns = stocks (STOCK_01 ... STOCK_12).
# return:  dict {stock: weight}
#          weight =  0.25 -> put 25% of your money in that stock (buy)
#          weight = -0.10 -> bet 10% against that stock (short sell)
#          stocks you leave out get 0; whatever is unused stays in cash.
#          The sum of |weights| must be <= 1 (scaled down for you if it is more).


def decide_momentum(history):
    """Buy the 3 stocks that went up the most in the last 20 days."""
    close = history["close"]
    if len(close) < 21:
        return {}                                   # not enough data yet -> stay in cash
    past_return = close.iloc[-1] / close.iloc[-21] - 1
    winners = past_return.nlargest(3).index
    return {stock: 1 / 3 for stock in winners}


def decide_trend_filter(history):
    """Hold every stock that is above its 50-day average, equally weighted."""
    close = history["close"]
    if len(close) < 50:
        return {}
    above = close.iloc[-1] > close.iloc[-50:].mean()
    picks = above[above].index
    return {stock: 1 / len(picks) for stock in picks} if len(picks) else {}


# Ideas to try next: mean reversion, RSI, volatility filters, stop-losses,
# position sizing by risk, ML models (train once, cache in a global), ...


decide = decide_trend_filter      # <- the agent we will run on the secret period
