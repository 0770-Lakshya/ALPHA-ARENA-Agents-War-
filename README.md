# Agents War — Alpha Arena (Meraz 7.0)

You get 15 years of daily prices for 12 Indian stocks. Build a trading agent that decides,
every day, how to split your Rs 10 lakh (virtual) between those stocks. After the event we let
your agent trade a **secret market period it has never seen**, right after the training data ends.
We don't reveal which years the data or the secret period come from, so your agent has to cope
with whatever the market does. The best agent wins.

**Getting one agent to run takes 10 minutes. Winning takes thinking:** try several ideas,
test them honestly, understand why they work because you also have to present your idea and why it is best, and control your risk.

## What's in this kit
| File | What it is |
|---|---|
| `data/train.csv` | Prices: `day, stock, open, high, low, close, volume`. Stock names are hidden (STOCK_01..12) and every stock starts at 100. |
| `strategy.py` | **Your agents. The only code file you submit.** Write each idea as its own `decide_xxx` function. |
| `compare.py` | `python compare.py` tests all your `decide_xxx` functions side by side and saves a table and chart for your slides. |
| `run_backtest.py` | `python run_backtest.py` tests the agent you picked (`decide`) and plots it against the benchmark. |
| `starter.ipynb` | The same steps as a notebook (works in Google Colab / Jupyter). |
| `engine.py` | The simulator. Don't edit it; we score you with the same file. |

Setup, either way works:
* **Google Colab (nothing to install):** File → Open notebook → GitHub tab → paste this repo's link → open `starter.ipynb`,
  then run the cells. The first cell downloads the rest of the kit.
* **Laptop:** `git clone <this repo>` (or Code → Download ZIP), then `pip install pandas numpy matplotlib scikit-learn`.

## The function you write
```python
def decide_my_idea(history):
    close = history["close"]         # DataFrame: rows = days so far, columns = stocks
    ...
    return {"STOCK_04": 0.5, "STOCK_09": -0.2}   # 50% long STOCK_04, 20% short STOCK_09, 30% cash

decide = decide_my_idea              # the one we run on the secret period
```
* It is called once per day after the close. `history` only goes up to today, so you can't peek at the future.
* Weights go from -1 to 1. The total of |weights| must be ≤ 1; if it's more, it gets scaled down.
* Changing a weight costs **0.1%** of the amount traded, so trading every day gets expensive.

## How to work
1. Write 3 or more different ideas as `decide_xxx` functions.
2. `python compare.py` splits the data into **fit** (the first 10 years) and **validation** (all the years after).
   Build and tune **only on the first 10 years**; treat the rest as your mini secret period and don't tune on it.
3. Pick your final agent: `decide = decide_xxx`. Be wary of an agent that is great on fit but weak on validation.

## Rules
1. Submit `strategy.py` using only python, numpy, pandas and scikit-learn. No internet or file reading inside it.
2. No hard-coded prices or day numbers: your agent must not act differently just because of which day number it is.
3. On average your agent must keep **at least 30%** of the money invested. An agent that sits in cash is disqualified.
4. We run the agent over the training days plus the secret period. That must finish within **3 minutes**.
5. If the agent crashes on more than **5%** of days, it is disqualified. On any day it crashes, it keeps yesterday's weights.
6. No sharing code between teams. AI assistants are allowed; say so in your explanation.

## What to submit
* `strategy.py`, with all the `decide_xxx` ideas you tried kept inside it and `decide` pointing at your final one.
  Working in the notebook / Colab? Its last cell downloads `strategy.py` for you. Submit that file, not the notebook.
* `comparison.csv` from `compare.py`.
* A 3–5 line explanation of your final strategy.
* **If the organizers announce presentations:** 3–5 slides:
  1. Your final agent in one sentence.
  2. The strategies you tried (your `compare.py` table and chart).
  3. Why you picked the final one: the evidence from validation.
  4. Risk: how your agent limits losses when the market crashes.
  5. What you learnt / what you would do with more time.

## Scoring — 100 backtest marks (measured on the secret period)
| Metric | Marks | What it means |
|---|---|---|
| Net Profit / Return | 20 | total return |
| Sharpe Ratio | 15 | return per unit of risk (ups and downs) |
| Calmar Ratio | 15 | yearly return ÷ worst fall from a peak |
| Maximum Drawdown | 10 | worst fall from a peak (smaller is better) |
| Generalization | 15 | secret-period Sharpe minus training Sharpe (a small drop is better) |
| Trading Accuracy | 10 | % of held positions that moved the right way the next day |
| Transaction Cost Efficiency | 10 | total trading costs you paid (less is better) |
| Technical Reliability | 5 | full marks if the agent never crashes |

Every metric except reliability is **ranked**: your percentile among all teams, so the best team on a
metric gets full marks. The tie-break is a higher Sharpe ratio. If presentations are held, the final score
is 70% backtest + 30% slides. `compare.py` and `run_backtest.py` print every one of these numbers
(plus the Sortino ratio, which is like Sharpe but only counts the falls).
