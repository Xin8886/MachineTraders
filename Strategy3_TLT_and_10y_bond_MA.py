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

plt.style.use("seaborn-v0_8")

# Path to your CSV file
file_path = "/Users/xinhuang/Documents/GitHub/Machine-Trader/tlt_yield_prediction_strategy/price_bars_20250721.csv"

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
#%% Step 2: Feature engineering
### 📊 Trend-Following Strategy (SMA on 10Y yield)
# Short-term and long-term moving averages
df["tlt_ret"] = df["TLT_close"].pct_change()
df["yield_sma_short"] = df["10Y_yield"].rolling(
    10).mean()  # mean of the previous 10 days’ yields
df["yield_sma_long"] = df["10Y_yield"].rolling(30).mean()
df["tlt_sma"] = df["TLT_close"].rolling(5).mean()
df["volatility"] = df["tlt_ret"].rolling(10).std()
df["tlt_mom_5d"] = df["TLT_close"].pct_change(5)

#%% Step 3: Generate signal with confirmation & filter
# Signal: if yield is trending down → bond price up → long TLT
# if yield is trending up → bond price down → short TLT
df["signal"] = 0
# Long condition: falling yield + TLT above its trend
# long_cond = (df["yield_sma_short"] < df["yield_sma_long"]) & (df["TLT_close"] > df["tlt_sma"])
# short_cond = (df["yield_sma_short"] > df["yield_sma_long"]) & (df["TLT_close"] < df["tlt_sma"])
long_cond = (df["yield_sma_short"] < df["yield_sma_long"])
short_cond = (df["yield_sma_short"] > df["yield_sma_long"])

df.loc[long_cond, "signal"] = 1
df.loc[short_cond, "signal"] = -1

# df.loc[ df["tlt_mom_5d"] < 0, "signal"] = 0

# Volatility filter: skip top 10% vol
# vol_threshold = df["volatility"].quantile(0.9)
# df.loc[df["volatility"] > vol_threshold, "signal"] = 0

#df["signal"] = df["signal"].shift(1)  # avoid lookahead
df.dropna(inplace=True)
# %% Step 4: Backtest strategy
### 📊 Strategy returns
df["strategy_ret"] = df["signal"] * df["tlt_ret"]
df["cum_tlt"] = (1 + df["tlt_ret"]).cumprod()
df["cum_strategy"] = (1 + df["strategy_ret"]).cumprod()

#%% Step 5: Plot results
### 📊 Plot results
plt.figure(figsize=(12, 6))
plt.plot(df.index, df["cum_tlt"], label="TLT Buy & Hold")
plt.plot(df.index, df["cum_strategy"], label="Improved Yield Strategy")
plt.title("TLT vs Improved Yield Strategy")
plt.ylabel("Cumulative Return")
plt.legend()
plt.tight_layout()
plt.show()


#%% Step 6: Evaluation metrics
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


print(f"Buy & Hold Sharpe: {sharpe(df['tlt_ret']):.2f}")
print(f"Strategy Sharpe:   {sharpe(df['strategy_ret']):.2f}")
print(f"Buy & Hold Max Drawdown: {max_drawdown(df['cum_tlt']):.2%}")
print(f"Strategy Max Drawdown: {max_drawdown(df['cum_strategy']):.2%}")
print(f"Buy & Hold Annual Return:    {annualized_return(df['cum_tlt']):.2%}")
print(
    f"Strategy Annual Return:      {annualized_return(df['cum_strategy']):.2%}"
)
#%%
### 📊 Save merged dataset
df.to_csv(f"tlt_yield_strategy_{datetime.now():%Y%m%d}.csv",
          index_label="timestamp")

# %%
