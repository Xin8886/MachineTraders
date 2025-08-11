#%%
import os
import numpy as np
import pandas as pd
import torch
from datetime import datetime, timedelta
from dotenv import load_dotenv
from sklearn.preprocessing import StandardScaler

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from pandas_datareader import data as pdr

# --- your utils ---
import utils
from utils import *

# --- load and prepossessing data ---
tlt_csv_path = "/Users/xinhuang/Documents/GitHub/Machine-Trader/tlt_yield_prediction_strategy/price_bars_20250807.csv"


def load_and_merge_tlt_yield(tlt_csv_path: str,
                             start_date: str = None,
                             end_date: str = None) -> pd.DataFrame:
    """
    Load TLT price data from CSV and merge with 10Y yield from FRED.
    
    Parameters:
        tlt_csv_path (str): Path to the TLT CSV file.
        start_date (str): Optional start date in 'YYYY-MM-DD' format.
        end_date (str): Optional end date in 'YYYY-MM-DD' format.
    
    Returns:
        pd.DataFrame: Merged DataFrame with columns [TLT_close, 10Y_yield].
    """
    # ---1. Load TLT data ---
    df_tlt = pd.read_csv(tlt_csv_path)
    df_tlt["timestamp"] = pd.to_datetime(df_tlt["timestamp"])
    df_tlt["timestamp"] = df_tlt["timestamp"].dt.tz_convert("America/New_York")
    df_tlt.index = df_tlt["timestamp"].dt.tz_localize(None)
    df_tlt = df_tlt.rename(columns={"close": "TLT_close"})
    df_tlt = df_tlt[["TLT_close"]]

    # If no start/end provided, use TLT range
    if start_date is None:
        start_date = df_tlt.index.min()
    if end_date is None:
        end_date = df_tlt.index.max()

    # --- Fetch 10Y yield from FRED ---
    df_yield = pdr.DataReader("DGS10", "fred", start_date, end_date)
    df_yield.rename(columns={"DGS10": "10Y_yield"}, inplace=True)

    # --- Merge & drop NA ---
    df_merged = df_tlt.join(df_yield, how="inner").dropna()

    return df_merged


df = load_and_merge_tlt_yield(tlt_csv_path, start_date=None, end_date=None)
#%%
# 2. Prepare features and target
df["10Y_yield_t+1"] = df["10Y_yield"].shift(-1)
df = df.dropna()
X = df[["10Y_yield"]]  # feature
y = df[["10Y_yield_t+1"]]  # target

# 3. Define scaler and split index
scaler = MinMaxScaler(feature_range=(0, 1))
split_index = int(len(df) * 0.8)  # 80% train, 20% test
prediction_index = df.iloc[split_index:].index
#%%
# 4. Train + rolling forecast with utils
predictions_original, y_test_original = LSTM_train_and_rolling_predict(
    X,
    y,
    hidden_size=30,
    num_layers=2,
    scaler=scaler,
    epoch=5000,
    split_index=split_index,
    model_name="LSTM",
    summary_table=summary_table)
#%%
# 5. Store predictions with dates
predictions_df = pd.DataFrame(
    {
        "y_test_original": y_test_original,
        "predictions": predictions_original
    },
    index=prediction_index)
#%%
