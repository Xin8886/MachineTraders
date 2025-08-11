# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timezone
from pathlib import Path
from scipy.stats import zscore
from dotenv import load_dotenv
import os
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from pandas_datareader import data as pdr
from scipy.stats import zscore

plt.style.use("seaborn-v0_8")

# Path to your CSV file
file_path = "/Users/xinhuang/Documents/GitHub/Machine-Trader/tlt_yield_prediction_strategy/price_bars_20250807.csv"

# Read CSV into DataFrame
df_tlt = pd.read_csv(file_path)

df_tlt["timestamp"] = pd.to_datetime(
    df_tlt["timestamp"]).dt.tz_convert("America/New_York")
df_tlt.set_index("timestamp", inplace=True)
df_tlt.index = df_tlt.index.tz_localize(None)
df_tlt = df_tlt.rename(columns={"close": "TLT_close"})
df_tlt = df_tlt[["TLT_close"]]
# %%
# print(df_tlt.head())

### 📊 Fetch 10Y yield daily from FRED
df_yield = pdr.DataReader("DGS10", "fred",
                          df_tlt.index.min().tz_localize(None),
                          df_tlt.index.max().tz_localize(None))
df_yield.rename(columns={"DGS10": "10Y_yield"}, inplace=True)

# %%
### 📊 Merge & align data
df = df_tlt.join(df_yield, how="inner").dropna()
#%%
# --- Feature Engineering ---
df["yield_momentum"] = df["10Y_yield"].diff(5)  # 5-day yield change
df["yield_zscore"] = zscore(df["10Y_yield"].dropna())  # z-score
df["tlt_ret"] = df["TLT_close"].pct_change()
df.dropna(inplace=True)

# --- Strategy 1: Yield Momentum ---
df["signal_momentum"] = np.where(df["yield_momentum"] < 0, 1, -1)
df["signal_momentum"] = df["signal_momentum"].shift(1)

# --- Strategy 2: Yield Z-Score ---
df["signal_zscore"] = np.where(df["yield_zscore"] < 0, 1, -1)
df["signal_zscore"] = df["signal_zscore"].shift(1)

# Drop any resulting NaNs
df.dropna(inplace=True)

# --- Backtest ---
df["strategy_ret_momentum"] = df["signal_momentum"] * df["tlt_ret"]
df["strategy_ret_zscore"] = df["signal_zscore"] * df["tlt_ret"]
df["cum_tlt"] = (1 + df["tlt_ret"]).cumprod()
df["cum_momentum"] = (1 + df["strategy_ret_momentum"]).cumprod()
df["cum_zscore"] = (1 + df["strategy_ret_zscore"]).cumprod()

# --- Plot ---
plt.figure(figsize=(12, 6))
plt.plot(df.index, df["cum_tlt"], label="Buy & Hold", linestyle="--")
plt.plot(df.index, df["cum_momentum"], label="Momentum Strategy")
plt.plot(df.index, df["cum_zscore"], label="Z-Score Strategy")
plt.title("Backtest: Yield Momentum vs Z-Score Strategy")
plt.ylabel("Cumulative Return")
plt.legend()
plt.tight_layout()
plt.show()

# %%


#%%
### 📊 Evaluation metrics
def sharpe(returns, rf=0.0):
    excess = returns - rf / 252
    return np.sqrt(252) * excess.mean() / excess.std()


def max_drawdown(cum_returns):
    peak = cum_returns.cummax()
    dd = cum_returns / peak - 1
    return dd.min()


def annualized_return(cum_returns, periods_per_year=252):
    n_days = len(cum_returns)
    total_return = cum_returns.iloc[-1]
    return total_return**(periods_per_year / n_days) - 1


print("Momentum Strategy:")
print(f"  Sharpe: {sharpe(df['strategy_ret_momentum']):.2f}")
print(f"  Max Drawdown: {max_drawdown(df['cum_momentum']):.2%}")
print(f"  Annual Return: {annualized_return(df['cum_momentum']):.2%}\n")

print("Z-Score Strategy:")
print(f"  Sharpe: {sharpe(df['strategy_ret_zscore']):.2f}")
print(f"  Max Drawdown: {max_drawdown(df['cum_zscore']):.2%}")
print(f"  Annual Return: {annualized_return(df['cum_zscore']):.2%}")
#%%
### 📊 Save merged dataset
df.to_csv(f"tlt_yield_strategy_{datetime.now():%Y%m%d}.csv",
          index_label="timestamp")

# %%
