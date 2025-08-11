### Submit trade orders using Alpaca API.

import requests
import alpaca_trade_api as tradeapi
from alpaca.trading.client import TradingClient
from dotenv import load_dotenv
import os
import pandas as pd
import time

# Load environment variables from .env file
load_dotenv("/Users/jerzy/Develop/Python/.env")
# Data keys
DATA_KEY = os.getenv("DATA_KEY")
DATA_SECRET = os.getenv("DATA_SECRET")
# Trade keys
TRADE_KEY = os.getenv("TRADE_KEY")
TRADE_SECRET = os.getenv("TRADE_SECRET")

trading_client = TradingClient(TRADE_KEY, TRADE_SECRET)

# data_client = StockHistoricalDataClient(DATA_KEY, DATA_SECRET)
BASE_URL = "https://paper-api.alpaca.markets"
trade_api = tradeapi.REST(TRADE_KEY, TRADE_SECRET, BASE_URL, api_version="v2")

### Submit market orders

# === Load latest signal from CSV ===
file_path = "/Users/xinhuang/Documents/GitHub/Machine-Trader/tlt_yield_strategy/tlt_yield_strategy_20250708.csv"
# Read CSV into DataFrame
df = pd.read_csv(file_path, parse_dates=["timestamp"])
df.set_index("timestamp", inplace=True)
latest_signal = df["signal"].iloc[-1]

symbol = "TLT"
qty = 1  # adjust size as needed

# === Check existing position ===
try:
    position = trade_api.get_position(symbol)
    current_side = "long" if float(position.qty) > 0 else "short"
    print(f"📌 Current position: {current_side}, qty={position.qty}")
except:
    position = None
    current_side = "none"
    print("📌 No existing position.")

# === Determine desired position ===
if latest_signal == 1:
    desired_side = "long"
elif latest_signal == -1:
    desired_side = "short"
else:
    desired_side = "none"


# === Submit orders accordingly ===
def submit_market_order(side, qty):
    order = trade_api.submit_order(symbol=symbol,
                                   qty=qty,
                                   side=side,
                                   type="market",
                                   time_in_force="day")
    print(f"✅ Submitted {side.upper()} order. Order ID: {order.id}")
    return order


# === Trade logic ===
if desired_side == current_side:
    print("⚠️  Already in desired position. No action needed.")
elif desired_side == "none":
    if current_side != "none":
        print("📤 Closing position...")
        try:
            trade_api.close_position(symbol)
            print("✅ Position closed.")
        except Exception as e:
            print(f"❌ Error closing position: {e}")
    else:
        print("📭 No position and no signal — no action.")

elif desired_side == "long":
    if current_side == "short":
        print("🔁 Switching from short to long...")
        try:
            trade_api.close_position(symbol)
            time.sleep(2)  # Allow time for the close to process
        except Exception as e:
            print(f"❌ Error closing short: {e}")
    try:
        submit_market_order("buy", qty)
    except Exception as e:
        print(f"❌ Error submitting buy order: {e}")

elif desired_side == "short":
    if current_side == "long":
        print("🔁 Switching from long to short...")
        try:
            trade_api.close_position(symbol)
            time.sleep(2)  # Allow time for the close to process
        except Exception as e:
            print(f"❌ Error closing long: {e}")
    try:
        submit_market_order("sell", qty)
    except Exception as e:
        print(f"❌ Error submitting sell order: {e}")

else:
    print(f"❓ Unexpected desired_side: {desired_side}")

# pos = trade_api.get_position("TLT")
