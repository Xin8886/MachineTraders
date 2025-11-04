# 📈 Auto Trading Script – TLT Yield Prediction Strategy

This script implements a **machine learning–driven bond ETF trading strategy** using **LSTM-based yield forecasting** and executes live trades automatically via the **Alpaca API**.

The core idea:  
- When **10Y U.S. Treasury yields** are expected to **rise**, bond prices fall → **short TLT**.  
- When yields are expected to **fall**, bond prices rise → **buy TLT**.  

The model predicts the next day’s 10Y yield using a **rolling LSTM forecast**, and automatically places trades in **TLT (iShares 20+ Year Treasury Bond ETF)** based on the predicted direction.

---

## ⚙️ Key Features

- **Automatic Data Merge**: Combines historical **TLT prices** and **10Y Treasury yield** (FRED DGS10).
- **LSTM Forecasting**: Predicts next-day yield (`y_{t+1}`) using past yield data.
- **Adaptive Signal Generation**:
  - Predicts next day’s yield change.
  - Generates BUY/SELL/NO-TRADE signals based on yield movement thresholds.
- **Real-Time Trading**:
  - Executes one trade per day (BUY/SELL) via Alpaca API.
  - Supports both **market** and **limit** orders.
- **Logging and State Tracking**:
  - Automatically logs executed trades, signals, and errors to `.csv` files.
- **Graceful Exit**: Handles interrupts (`Ctrl-C`) safely without breaking the trading loop.

---

## 📂 Project Structure

```
tlt_yield_prediction_strategy/
│
├── auto_trading_alpaca.py         # Main trading script
├── utils.py                       # ML training & predictionutilities
├── utils_MachineTrader.py         # Common helper functions
├── price_bars_20250807.csv        # Output of get_historical_prices.py
├── .env                           # API credentials (see below)
│
├── state_StratYieldLSTM001_TLT_*.csv      # Trade state logs
├── submits_StratYieldLSTM001_TLT_*.csv    # Order submissions log
├── canceled_StratYieldLSTM001_TLT_*.csv   # Canceled order log
├── error_StratYieldLSTM001_TLT_*.csv      # Error log
│
└── README.md                       # You are here
```

---

## 🔑 Setup Instructions

### 1. Clone Repository
```bash
git clone https://github.com/Xin8886/MachineTraders.git
cd MachineTraders
```

### 2. Install Dependencies
```bash
pip install pandas numpy torch scikit-learn pandas_datareader alpaca-py python-dotenv nest_asyncio
```

### 3. Set Up `.env` File
```
DATA_KEY=your_alpaca_data_key
DATA_SECRET=your_alpaca_data_secret
TRADE_KEY=your_alpaca_trade_key
TRADE_SECRET=your_alpaca_trade_secret
data_dir_name=/Users/<username>/Documents/GitHub/Machine-Trader/tlt_yield_prediction_strategy/
```

---

## 🧠 Model Logic

### Data Loading
- Loads **TLT close prices** from a local CSV.
- Downloads **10Y Treasury yields** from **FRED (DGS10)**.
- Aligns timestamps and merges the data.

### Yield Forecasting
- LSTM model predicts the **next day’s yield**.
- Parameters:
  ```python
  HIDDEN_SIZE = 30
  NUM_LAYERS = 2
  EPOCHS = 1500
  TRAIN_SPLIT = 0.8
  YIELD_THRESHOLD = 0.0005
  ```

### Signal Rules
- Predicted yield ↓ → **BUY TLT**
- Predicted yield ↑ → **SELL TLT**
- |Δyield| < threshold → **NO TRADE**

### Execution
- One trade per day at market open.
- Logs every order and result.

---

## 🧾 Command-Line Usage

Interactive mode:
```bash
python auto_trading_alpaca.py
```

CLI mode:
```bash
python auto_trading_alpaca.py TLT market buy 10 0.05
```

---

## 📊 Log Files

| File | Description |
|------|--------------|
| state_*.csv | Executed trade details |
| submits_*.csv | Order submissions |
| canceled_*.csv | Canceled orders |
| error_*.csv | Runtime or API errors |

---

## ⚠️ Notes

- Trades **once per day** only.
- Always **test in paper trading** before going live.
- Ensure `.env` keys match **paper or live** environment.
- Data should be synchronized to **New_York timezone**.

---

## 🧑‍💻 Author

**Xin Huang**  
NYU Tandon MFE | Quant Researcher  
GitHub: https://github.com/Xin8886
