#%%
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Strategy:
# - Before streaming starts (once per run/day), load TLT + 10Y yield, run LSTM rolling forecast (utils),
#   predict tomorrow's yield, compute a daily trading signal (BUY/SELL/NO_TRADE).
# - Start Alpaca websocket bar stream.
# - On the first bar received, place ONE order according to the daily signal (market or limit),
#   then just log subsequent bars (no repeated orders).
#
# NOTE: This is an illustration using live streaming data to trigger execution once per day,
#       based on a yield forecast model you already have in utils.

import os
import sys
import signal
import time
from datetime import datetime
from zoneinfo import ZoneInfo
import pandas as pd

from dotenv import load_dotenv

# Alpaca trading + streaming
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.live.stock import StockDataStream
from alpaca.data.enums import DataFeed

# Your utils
import utils
from utils import *
import utils_MachineTrader
from utils_MachineTrader import *
# Optional: sklearn for scaler
from sklearn.preprocessing import MinMaxScaler
from pandas_datareader import data as pdr

# ---------- Config you likely want to edit ----------
# Local TLT CSV (daily or intraday; needs 'timestamp' and 'close' columns)
TLT_CSV_PATH = "/Users/xinhuang/Documents/GitHub/Machine-Trader/tlt_yield_prediction_strategy/price_bars_20251103.csv"

# Forecast hyperparams
HIDDEN_SIZE = 30
NUM_LAYERS = 2
EPOCHS = 1500
TRAIN_SPLIT = 0.8
YIELD_THRESHOLD = 0.0005  # ~5 bps; only act if predicted change > threshold

# Data merge window (None → auto from TLT file)
START_DATE = None
END_DATE = None
# ---------------------------------------------------

# --------- Create the SDK clients --------
load_dotenv(".env")
DATA_KEY = os.getenv("DATA_KEY")
DATA_SECRET = os.getenv("DATA_SECRET")
TRADE_KEY = os.getenv("TRADE_KEY")
TRADE_SECRET = os.getenv("TRADE_SECRET")

# Create streaming and trading clients
stream_client = StockDataStream(DATA_KEY, DATA_SECRET, feed=DataFeed.IEX)
trading_client = TradingClient(
    TRADE_KEY, TRADE_SECRET)  # paper/live depends on your key type

# --------- CLI params (match your reference style) --------
if len(sys.argv) > 5:
    symbol = sys.argv[1].strip().upper()
    order_type = sys.argv[2].strip().lower()  # "market" or "limit"
    _side_hint = sys.argv[3].strip().lower(
    )  # ignored (strategy decides); keep for compatibility
    shares_per_trade = float(sys.argv[4])
    delta = float(sys.argv[5])  # limit offset
else:
    symbol = input("Enter symbol (default TLT): ").strip().upper() or "TLT"
    order_type = input("Enter order type (market/limit) [market]: ").strip(
    ).lower() or "market"
    _ = input("Enter side (ignored—strategy decides) [buy/sell]: ").strip(
    ).lower()  # ignored
    shares_str = input("Enter number of shares (default 1): ").strip()
    shares_per_trade = float(shares_str) if shares_str else 1.0
    delta_str = input("Enter limit delta (default 0.05): ").strip()
    delta = float(delta_str) if delta_str else 0.05

# --------- Filenames / logging --------
tzone = ZoneInfo("America/New_York")
now_ny = datetime.now(tzone)
date_short = now_ny.strftime("%Y%m%d")
dir_name = os.getenv(
    "data_dir_name"
) or "/Users/xinhuang/Documents/GitHub/Machine-Trader/tlt_yield_prediction_strategy/"
os.makedirs(dir_name, exist_ok=True)

from pathlib import Path
# Ensure directory exists
Path(dir_name).mkdir(parents=True, exist_ok=True)

print("Data will be saved in:", Path(dir_name).resolve())

strategy_name = "StratYieldLSTM001"
state_file = f"{dir_name}state_{strategy_name}_{symbol}_{date_short}.csv"
submits_file = f"{dir_name}submits_{strategy_name}_{symbol}_{date_short}.csv"
canceled_file = f"{dir_name}canceled_{strategy_name}_{symbol}_{date_short}.csv"
error_file = f"{dir_name}error_{strategy_name}_{symbol}_{date_short}.csv"


#%%
# --------- Data loading / merge --------
def load_and_merge_tlt_yield(tlt_csv_path: str,
                             start_date=None,
                             end_date=None) -> pd.DataFrame:
    """
    Load TLT price data from CSV and merge with 10Y yield (FRED).
    Expects columns: 'timestamp' and 'close' in CSV.
    Returns: DataFrame with index = NY tz-naive timestamp, columns: ['TLT_close', '10Y_yield']
    """
    df_tlt = pd.read_csv(tlt_csv_path)
    df_tlt["timestamp"] = pd.to_datetime(df_tlt["timestamp"])
    # normalize to NY tz-naive
    df_tlt["timestamp"] = df_tlt["timestamp"].dt.tz_convert("America/New_York")
    df_tlt.index = df_tlt["timestamp"].dt.tz_localize(None)
    df_tlt = df_tlt.rename(columns={"close": "TLT_close"})[["TLT_close"]]

    if start_date is None:
        start_date = df_tlt.index.min()
    if end_date is None:
        end_date = df_tlt.index.max()

    df_yield = pdr.DataReader("DGS10", "fred", start_date, end_date)
    df_yield = df_yield.rename(columns={"DGS10": "10Y_yield"})

    df = df_tlt.join(df_yield, how="inner").dropna()
    return df


# --------- Forecast next-day yield & decide signal --------
def forecast_daily_signal(df: pd.DataFrame):
    """
    Runs your utils.LSTM_train_and_rolling_predict to forecast t+1 yield.
    Returns (signal, predicted_tomorrow, yesterday, delta, reason)
      signal: OrderSide.BUY / OrderSide.SELL / None
    """
    d = df.copy()
    d["10Y_yield_t+1"] = d["10Y_yield"].shift(-1)
    d = d.dropna()

    if len(d) < 50:
        return None, None, None, None, "Not enough data"

    X = d[["10Y_yield"]]
    y = d[["10Y_yield_t+1"]]

    split_index = int(len(d) * TRAIN_SPLIT)
    prediction_index = d.iloc[split_index:].index

    scaler = MinMaxScaler(feature_range=(0, 1))
    preds, y_true = LSTM_train_and_rolling_predict(X=X,
                                                   y=y,
                                                   hidden_size=HIDDEN_SIZE,
                                                   num_layers=NUM_LAYERS,
                                                   scaler=scaler,
                                                   epoch=EPOCHS,
                                                   split_index=split_index,
                                                   model_name="LSTM",
                                                   summary_table=summary_table)

    out = pd.DataFrame({
        "y_true": y_true,
        "y_pred": preds
    },
                       index=prediction_index)
    predicted_tomorrow = float(out["y_pred"].iloc[-1])

    # "yesterday" = last realized yield in df
    if len(df) < 2:
        return None, None, None, None, "Too few rows in df to compute yesterday"
    yesterday = float(df["10Y_yield"].iloc[-2])

    delta = predicted_tomorrow - yesterday

    if delta <= -YIELD_THRESHOLD:
        return OrderSide.BUY, predicted_tomorrow, yesterday, delta, "Yield ↓ ⇒ BUY TLT"
    elif delta >= YIELD_THRESHOLD:
        return OrderSide.SELL, predicted_tomorrow, yesterday, delta, "Yield ↑ ⇒ SELL TLT"
    else:
        return None, predicted_tomorrow, yesterday, delta, "Change < threshold ⇒ NO TRADE"


# --------- Pre-compute today's 1-shot signal --------
try:
    merged_df = load_and_merge_tlt_yield(TLT_CSV_PATH, START_DATE, END_DATE)
    daily_signal, pred_next, y_yday, y_delta, rationale = forecast_daily_signal(
        merged_df)
except Exception as e:
    daily_signal, pred_next, y_yday, y_delta, rationale = None, None, None, None, f"Forecast error: {e}"
    with open(error_file, "a") as f:
        f.write(f"{datetime.now(tzone)}, Forecast error: {e}\n")

print(f"Daily signal calc → {rationale}")
if pred_next is not None and y_yday is not None:
    print(
        f"Predicted(t+1): {pred_next:.6f} | Yesterday: {y_yday:.6f} | Δ: {y_delta:.6f}"
    )

# Use a simple “execute once” flag for the session/day
_already_traded_today = False


#%%
# --------- Callback: executes exactly once on first bar --------
async def trade_bars(bar):
    global _already_traded_today

    # Log bar
    close_price = bar.close
    vwap_price = bar.vwap
    ts = bar.timestamp.astimezone(
        ZoneInfo("America/New_York")).strftime("%Y-%m-%d %H:%M:%S")
    print(f"Time: {ts}, {bar.symbol} Close: {close_price}, VWAP: {vwap_price}")

    # If no signal or already traded, just log and return
    if daily_signal is None:
        print("No signal for today (NO TRADE).")
    if _already_traded_today or (daily_signal is None):
        return

    # Optional: sanity check position
    pos = get_position(trading_client, symbol)
    qty_owned = 0 if pos is None else float(pos.qty)

    # Build order once (market or limit from CLI)
    side = daily_signal
    if order_type == "market":
        limit_price = None
    else:
        # simple limit around last close
        if side == OrderSide.BUY:
            limit_price = round(close_price - delta, 2)
        else:
            limit_price = round(close_price + delta, 2)

    # Submit order using your submit_trade util (logs to submits_file)
    try:
        # od = submit_trade(
        #     trading_client=trading_client,
        #     symbol=symbol,
        #     qty=shares_per_trade,
        #     side=side,
        #     type=order_type,
        #     limit_price=limit_price,
        #     submits_file=submits_file
        # )
        od = submit_trade(
            trading_client=trading_client,
            symbol=symbol,
            qty=shares_per_trade,
            side=side,  # OrderSide.BUY / SELL
            order_type=order_type,  # "market" or "limit"
            limit_price=limit_price,  # None for market
            submits_file=submits_file,
        )
        if od is None:
            print(
                "Order submit failed; cancelling open orders and retrying once..."
            )
            cancel_orders(trading_client, symbol)
            od = submit_trade(trading_client, symbol, shares_per_trade, side,
                              order_type, limit_price, submits_file)

        # Mark traded
        _already_traded_today = True

        # Save state
        state_row = pd.DataFrame([{
            "date_time": ts,
            "symbol": bar.symbol,
            "price": close_price,
            "vwap": vwap_price,
            "pred_tomorrow_yield": pred_next,
            "yesterday_yield": y_yday,
            "yield_delta": y_delta,
            "signal": side.value if side else None,
            "order_type": order_type,
            "limit_price": limit_price,
            "shares": shares_per_trade,
            "qty_owned_before": qty_owned,
        }])
        state_row.to_csv(state_file,
                         mode="a",
                         header=not os.path.exists(state_file),
                         index=False)
        print("Trade executed and state saved.\n")

    except Exception as e:
        err = f"{datetime.now(tzone)} Submit error: {e}\n"
        print(err.strip())
        with open(error_file, "a") as f:
            f.write(err)


#%%
# --------- Run the websocket stream --------
# Subscribe to the symbol bars
# stream_client.subscribe_bars(trade_bars, symbol)

# try:
#     print("Starting stream… (Ctrl-C to stop)\n")
#     stream_client.run()
# except Exception as e:
#     timestamp = datetime.now(tzone).strftime("%Y-%m-%d %H:%M:%S")
#     error_text = f"{timestamp} WebSocket error: {e}. Restarting in 5s...\n"
#     print(error_text.strip())
#     with open(error_file, "a") as f:
#         f.write(error_text)
#     time.sleep(5)

# print("Stream stopped by user.")

import asyncio, nest_asyncio

nest_asyncio.apply()


async def main():
    print("Starting stream… (Ctrl-C to stop)\n")
    stream_client.subscribe_bars(trade_bars, symbol)
    await stream_client.run()


task = asyncio.create_task(main())  # don’t block the notebook


# --------- Handle Ctrl-C interrupt --------
def signal_handler(sig, frame):
    print("\n\nCtrl-C pressed! Exiting gracefully...")
    try:
        stream_client.stop()
    except Exception:
        pass
    sys.exit(0)


print("Press Ctrl-C to stop the stream... \n")
signal.signal(signal.SIGINT, signal_handler)

# %%
