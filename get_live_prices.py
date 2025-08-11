### Get live stock prices using the Alpaca API.

import requests
from dotenv import load_dotenv
import os
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from pandas_datareader import data as pdr

# === Step 1: Load existing CSV ===
file_path = "/Users/xinhuang/Documents/GitHub/Machine-Trader/tlt_yield_strategy/tlt_yield_strategy_20250708.csv"
# Read CSV into DataFrame
df = pd.read_csv(file_path, parse_dates=["timestamp"])
df.set_index("timestamp", inplace=True)

# === Step 2: Load Alpaca API keys ===
# Load environment variables from .env file
# load_dotenv("/Users/jerzy/Develop/Python/.env")
load_dotenv("/Users/xinhuang/Documents/GitHub/Machine-Trader/.env")

# Get API keys from environment
# Data keys
DATA_KEY = os.getenv("DATA_KEY")
DATA_SECRET = os.getenv("DATA_SECRET")

# === Step 3: Fetch latest TLT close price ===
headers = {"APCA-API-KEY-ID": DATA_KEY, "APCA-API-SECRET-KEY": DATA_SECRET}
symbol = "TLT"
params = {"symbols": symbol}

### Get the latest OHLCV bar of prices
response = requests.get("https://data.alpaca.markets/v2/stocks/bars/latest",
                        headers=headers,
                        params=params)
bar_data = response.json()
bar = bar_data["bars"][symbol]
tlt_close = bar["c"]

# === Step 4: Fetch latest 10Y yield ===
today = datetime.now().date()
start = today - timedelta(days=5)  # to ensure coverage if weekends/holidays
df_yield = pdr.DataReader("DGS10", "fred", start, today)
df_yield.dropna(inplace=True)
latest_yield = df_yield.iloc[-1, 0]  # today's yield

# === Step 5: Append new row to DataFrame ===
new_row = pd.DataFrame(
    {
        "TLT_close": [tlt_close],
        "10Y_yield": [latest_yield],
        "yield_z": [None],  # unused in this strategy
        "signal": [None],
        "tlt_ret": [None],
        "strategy_ret": [None],
        "cum_tlt": [None],
        "cum_strategy": [None],
    },
    index=[pd.to_datetime(today)])
df = pd.concat([df, new_row])
df.sort_index(inplace=True)

# Extend df with today's data temporarily
temp_df = df.copy()
temp_df = temp_df[["10Y_yield"]].copy()

# === Step 6: Compute short & long yield moving averages ===
# Use yesterday and earlier data to compute rolling values
df["yield_sma_short"] = df["10Y_yield"].shift(1).rolling(window=10).mean()
df["yield_sma_long"] = df["10Y_yield"].shift(1).rolling(window=60).mean()

# === Step 7: Generate signal based on trend-following logic ===
df["signal"] = 0
df.loc[df["yield_sma_short"] < df["yield_sma_long"],
       "signal"] = 1  # falling yields → long TLT
df.loc[df["yield_sma_short"] > df["yield_sma_long"],
       "signal"] = -1  # rising yields → short TLT

# === Step 8: Compute returns and cumulative returns ===
df["tlt_ret"] = df["TLT_close"].pct_change()
df["strategy_ret"] = df["signal"] * df["tlt_ret"]
df["cum_tlt"] = (1 + df["tlt_ret"].fillna(0)).cumprod()
df["cum_strategy"] = (1 + df["strategy_ret"].fillna(0)).cumprod()

df.to_csv(file_path, index_label="timestamp")
