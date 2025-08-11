# Retrieve trade orders using Alpaca API.

import pandas as pd
from datetime import datetime, timezone, timedelta
import pytz
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOrdersRequest
from alpaca.trading.enums import QueryOrderStatus
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv("/Users/jerzy/Develop/Python/.env")
# Trade keys
TRADE_KEY = os.getenv("TRADE_KEY")
TRADE_SECRET = os.getenv("TRADE_SECRET")

trading_client = TradingClient(TRADE_KEY, TRADE_SECRET)

# Get all open orders
orders = trading_client.get_orders(filter=GetOrdersRequest(
    status=QueryOrderStatus.OPEN))
# Get all closed orders
orders = trading_client.get_orders(filter=GetOrdersRequest(
    status=QueryOrderStatus.CLOSED))
# print(orders)
# Convert orders to data frame
orders_frame = pd.DataFrame(orders)
orders_frame.shape
# print(orders_frame)

# Get current NY time
ny_timezone = pytz.timezone("America/New_York")
current_time = datetime.now(ny_timezone)
# file_name = "/Users/jerzy/Develop/MachineTrader/Internship_Summer_2025/data/orders_" + current_time.strftime("%Y%m%d") + ".csv"
file_name = "Data Orders" + current_time.strftime("%Y%m%d") + ".csv"
orders_frame.to_csv(file_name, index=False)
print("Finished getting orders and saved to orders.csv")

# Print the client IDs of the orders
for order in orders:
    print("order_id: " + order.client_order_id)

# Print the client IDs of the orders
for order in orders:
    print("order_id: " + str(order.id))

##########
# Alternative way to fetch orders using requests

# import requests

# headers = {
#     "APCA-API-KEY-ID": TRADE_KEY,
#     "APCA-API-SECRET-KEY": TRADE_SECRET
# }

# BASE_URL = "https://paper-api.alpaca.markets"  # Use "https://api.alpaca.markets" for live trading

# # Fetch orders
# response = requests.get(f"{BASE_URL}/v2/orders?status=all", headers=headers)
# print(f"Fetching orders from {BASE_URL}/v2/orders")

# if response.status_code == 200:
#     orders = response.json()

#     # Convert to DataFrame
#     orders_frame = pd.DataFrame(orders)

#     # Display the first few rows
#     # print(orders_frame)
#     print(orders_frame.head())

#     # Optionally save to CSV
#     orders_frame.to_csv("/Users/jerzy/Develop/Python/alpaca_orders.csv", index=False)
# else:
#     print(f"Error: {response.status_code}, {response.text}")

# # Save DataFrame to CSV
# orders_frame.to_csv("/Users/jerzy/Develop/Python/paper3_orders.csv", index=False)
