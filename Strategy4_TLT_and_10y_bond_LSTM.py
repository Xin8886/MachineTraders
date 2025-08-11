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
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score

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
#%% Parameters
WINDOW = 10
EPOCHS = 80
BATCH_SIZE = 32
LR = 0.001

# 🛠 Feature Engineering
# Target: direction of yield change (1 for up, -1 for down)
df["yield_change"] = df["10Y_yield"].diff().shift(-1)
df["target"] = np.where(df["yield_change"] > 0, 1, -1)

# Drop the last row (because of shift)
df = df.dropna()

# Feature: use past WINDOW days' yield for prediction
features = []
labels = []

for i in range(WINDOW, len(df)):
    features.append(df["10Y_yield"].values[i - WINDOW:i])
    labels.append(df["target"].values[i])

X = np.array(features)
y = np.array(labels)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X.reshape(-1, WINDOW)).reshape(X.shape)

# Train/test split
split = int(len(X) * 0.8)
X_train, X_test = X_scaled[:split], X_scaled[split:]
y_train, y_test = y[:split], y[split:]

# Convert to tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor((y_train + 1) // 2,
                              dtype=torch.long)  # convert -1,1 to 0,1
y_test_tensor = torch.tensor((y_test + 1) // 2, dtype=torch.long)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)


# 🔧 Define LSTM model
class YieldDirectionLSTM(nn.Module):

    def __init__(self, input_size=1, hidden_size=32, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size,
                            hidden_size,
                            num_layers,
                            batch_first=True)
        self.fc = nn.Linear(hidden_size, 2)  # binary classification (up/down)

    def forward(self, x):
        x = x.unsqueeze(-1)  # (batch_size, seq_len, 1)
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # last output
        return self.fc(out)


model = YieldDirectionLSTM()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# 🚆 Train the model
for epoch in range(EPOCHS):
    model.train()
    for xb, yb in train_loader:
        pred = model(xb)
        loss = criterion(pred, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {loss.item():.4f}")

# ✅ Evaluate Accuracy
model.eval()
with torch.no_grad():
    train_preds = model(X_train_tensor).argmax(dim=1).numpy()
    test_preds = model(X_test_tensor).argmax(dim=1).numpy()

train_acc = accuracy_score(y_train_tensor.numpy(), train_preds)
test_acc = accuracy_score(y_test_tensor.numpy(), test_preds)

print(f"\n📈 Training Accuracy: {train_acc:.2%}")
print(f"🧪 Testing Accuracy:  {test_acc:.2%}")

#%%

#%%
### 📊 Save merged dataset
df.to_csv(f"tlt_yield_strategy_{datetime.now():%Y%m%d}.csv",
          index_label="timestamp")

# %%
