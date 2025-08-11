Strategy1_TLT_and_10y_bond.py
Strategy2_TLT_and_10y_bond_momentum_and_Z_score.py 
Strategy3_TLT_and_10y_bond_MA.py
Strategy4_TLT_and_10y_bond_LSTM.py
Strategy5_LSTM_rolling_forecast_backtest.py

These 5 files are all back-testing files and are explained in the html file.


auto_traing_alpaca.py: run this file for final strategy auto-trading on alpaca 
but need to run get_historical_prices.py first and copy the path of the saved csv file as TLT_CSV_PATH at line 41.


# Details for the strategy — Daily Yield-Forecast Trading (Alpaca + Streaming Bars)

This strategy runs a **daily, one-shot trade** on **TLT** based on an **LSTM forecast** of the **U.S. 10Y Treasury yield (FRED: DGS10)**.

It:
1. Loads your **local TLT price CSV** and merges with **10Y yield** data from FRED.
2. Uses your LSTM model (`LSTM_train_and_rolling_predict`) to forecast **tomorrow’s yield**.
3. Converts the forecast into a **BUY/SELL/NO TRADE** signal for TLT.
4. Starts an **Alpaca websocket** and, on the **first bar**, places one order.
5. Logs all actions to CSV files.

⚠️ **Educational example only** — not investment advice.

---

## Requirements

- Python 3.9+
- Install dependencies:
  ```bash
  pip install alpaca-py pandas pandas-datareader python-dotenv scikit-learn pytz
