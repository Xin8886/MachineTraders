import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from math import sqrt
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.stattools import adfuller
from torch.optim import Adam
from pmdarima import auto_arima
from statsmodels.tsa.arima.model import ARIMA


class LSTMModel(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_directions = 1  # 单向LSTM

        # 定义 LSTM 层
        self.lstm = nn.LSTM(input_size,
                            hidden_size,
                            num_layers,
                            batch_first=True)

        # 定义输出层
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # 初始化隐藏状态和细胞状态
        h0 = torch.randn(self.num_directions * self.num_layers, x.size(0),
                         self.hidden_size).to(x.device)
        c0 = torch.randn(self.num_directions * self.num_layers, x.size(0),
                         self.hidden_size).to(x.device)

        # 前向传播 LSTM
        out, _ = self.lstm(x, (h0, c0))

        # 取最后一步的输出用于预测
        out = self.linear(out[:, -1, :])
        return out


def train(model, criterion, optimizer, epochs, X_train, y_train):
    model.train()
    train_losses = []

    for i in range(epochs):

        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())

        if (epochs + 1) % 10 == 0:
            print(f'Epoch [{epochs+1}/{epochs}], Loss: {loss.item():.4f}')

    return train_losses


def predict(model, X):
    model.eval()  # 将模型设置为评估模式
    with torch.no_grad():  # 在此模式下，所有的计算都不会计算梯度
        predictions = model(X)
    return predictions


def rolling_forecast(X_train, y_train, X_test, y_test, forecast_steps, epoch):
    forecasts = []
    history_X, history_y = X_train, y_train
    input_size = X_train.shape[2]
    hidden_size = 30
    num_layers = 2
    output_size = 1

    for i in range(0, forecast_steps, 10):
        model = LSTMModel(input_size, hidden_size, num_layers, output_size)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        # Train the model on the available history
        train_losses = train(model, criterion, optimizer, epoch, history_X,
                             history_y)

        # Predict the next step
        model.eval()
        with torch.no_grad():
            for j in range(10):
                if i + j < len(X_test):
                    forecast = model(X_test[i + j].unsqueeze(
                        0))  # Predict using the last available input
                    forecasts.append(forecast.item())

        if i + 10 < len(X_test):
            new_X = X_test[i:i + 10]
            new_y = y_test[i:i + 10]
        else:
            new_X = X_test[i:]
            new_y = y_test[i:]

        # Update history with the true value (here using test data to simulate new observations)
        history_X = torch.cat((history_X, new_X), dim=0)
        history_y = torch.cat((history_y, new_y), dim=0)

    return forecasts, train_losses


def reverse_minmax_scaler(train_predictions, predictions, scaler):
    # scaler = MinMaxScaler(feature_range=(0, 1))
    combined_array = np.vstack(
        (train_predictions.reshape(-1, 1), predictions.reshape(-1, 1)))
    combined_array_original = scaler.inverse_transform(combined_array)
    len_train_predictions = len(train_predictions)
    train_predictions_original = combined_array_original[:
                                                         len_train_predictions]
    predictions_original = combined_array_original[len_train_predictions:]

    return train_predictions_original, predictions_original


def calculate_ratio(real_data, predict_data):
    result = pd.DataFrame({
        'real data': real_data,
        'predict data': predict_data
    })
    result['shift'] = result['real data'].shift(1)
    result = result.dropna()
    result['diff_real'] = result['real data'] - result['shift']
    result['diff_prediction'] = result['predict data'] - result['shift']
    # Categorize 'diff_real_prediction' where non-negative values are 1, and negative values are 0
    result['cat_diff_real'] = result['diff_real'].apply(lambda x: 1
                                                        if x >= 0 else 0)
    result['cat_diff_prediction'] = result['diff_prediction'].apply(
        lambda x: 1 if x >= 0 else 0)
    count_matching_categories = ((
        result['cat_diff_real'] == result['cat_diff_prediction'])).sum()
    #calculate ratio
    ratio = count_matching_categories / len(result)
    return ratio


summary_table = []


def LSTM_train_and_rolling_predict(X, y, hidden_size, num_layers, scaler,
                                   epoch, split_index, model_name,
                                   summary_table):
    input_size = X.shape[1]
    X = scaler.fit_transform(X)
    y = scaler.fit_transform(y)

    X = torch.tensor(X).float().reshape(
        -1, 1, input_size)  # Reshaping and converting to tensor
    y = torch.tensor(y).float().reshape(-1, 1)

    X_train = X[:split_index]
    X_test = X[split_index:]
    y_train = y[:split_index]
    y_test = y[split_index:]

    # epoch = 5000
    # hidden_size = 30
    # num_layers = 2
    output_size = 1
    # epoch = 5000
    # input_size = X_train.shape[2]
    model_lstm = LSTMModel(input_size, hidden_size, num_layers, output_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model_lstm.parameters(), lr=0.001)

    # 调用训练函数
    train_losses_lstm = train(model_lstm, criterion, optimizer, epoch, X_train,
                              y_train)
    train_predictions = predict(model_lstm, X_train)

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses_lstm, label='Train Loss')
    plt.title('Training Loss Per Epoch')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    forecast_steps = len(y_test)
    predictions, train_losses = rolling_forecast(X_train, y_train, X_test,
                                                 y_test, forecast_steps, epoch)
    predictions = np.array(predictions)

    train_predictions_original, predictions_original = reverse_minmax_scaler(
        train_predictions, predictions, scaler)
    y_train_original, y_test_original = reverse_minmax_scaler(
        y_train, y_test, scaler)
    train_rmse = sqrt(
        mean_squared_error(y_train_original, train_predictions_original))
    test_rmse = sqrt(mean_squared_error(y_test_original, predictions_original))
    train_ratio = calculate_ratio(y_train_original.flatten(),
                                  train_predictions_original.flatten())
    test_ratio = calculate_ratio(y_test_original.flatten(),
                                 predictions_original.flatten())

    print(f'Train RMSE: {train_rmse:.4f}')
    print(f'Test RMSE: {test_rmse:.4f}')
    print(f"Ratio of train set: {train_ratio:.4f}")
    print(f"Ratio of test set: {test_ratio:4f}")
    summary_table.append(
        [model_name, train_rmse, test_rmse, train_ratio, test_ratio])

    # 设定图形大小
    plt.figure(figsize=(14, 7))

    # 绘制训练集的实际值和预测值
    plt.plot(y_train_original, label='Actual Values (Train)', alpha=0.6)
    plt.plot(train_predictions_original,
             'o-',
             label='Predicted Values (Train)',
             alpha=0.6,
             linestyle='--')

    # 绘制测试集的实际值和预测值
    plt.plot(range(len(y_train_original),
                   len(y_train_original) + len(predictions_original)),
             y_test_original,
             label='Actual Values (Test)',
             color='orange')
    plt.plot(range(len(train_predictions_original),
                   len(train_predictions_original) + len(predictions)),
             predictions_original,
             'o-',
             label='Predicted Values (Test)',
             color='red',
             linestyle='--')

    # 添加标题和图例
    plt.title(f'{model_name} Actual vs Predicted ')
    plt.xlabel('Sample Index')
    plt.ylabel('Values')
    plt.legend()
    plt.show()

    predictions_original = np.array(predictions_original)
    y_test_original = np.array(y_test_original)
    predictions_original = predictions_original.reshape(-1)
    y_test_original = y_test_original.reshape(-1)

    return predictions_original, y_test_original


def ARIMA_rolling_forecast(train, test):
    history = list(train)
    forecasts = []
    forecast_steps = len(test)

    for i in range(forecast_steps):
        model = auto_arima(history,
                           seasonal=True,
                           trace=True,
                           error_action='ignore',
                           suppress_warnings=True,
                           stepwise=True)
        model_fit = model.fit(history)
        forecast = model_fit.predict()[0]
        forecasts.append(forecast)
        #history.append(forecast)
        history.append(test[i])

    return forecasts, model_fit


def ARIMA_evaluate(train, test, train_prediction, test_prediction,
                   summary_table):

    # Calculate MSE and RMSE
    # train = train.iloc[1:]
    # train_prediction = train_prediction.iloc[1:]
    train_mse = mean_squared_error(train, train_prediction)
    test_mse = mean_squared_error(test, test_prediction)
    train_rmse = sqrt(train_mse)
    test_rmse = sqrt(test_mse)
    train_ratio = calculate_ratio(train, train_prediction)
    test_ratio = calculate_ratio(test, test_prediction)
    summary_table.append(
        ['ARIMA Model', train_rmse, test_rmse, train_ratio, test_ratio])

    # Print MSE and RMSE
    print(f"Training MSE: {train_mse}")
    print(f"Training RMSE: {train_rmse}")
    print(f"Test MSE: {test_mse}")
    print(f"Test RMSE: {test_rmse}")
    print(f"Ratio of train set: {train_ratio:.4f}")
    print(f"Ratio of test set: {test_ratio:4f}")

    # Plotting the results
    plt.figure(figsize=(12, 6))
    plt.plot(train.index, train, label='Training Data')
    plt.plot(test.index, test, label='Test Data')
    plt.plot(train.index,
             train_prediction,
             color='blue',
             linestyle='--',
             label='Forecast on Training Data')
    plt.plot(test.index,
             test_prediction,
             color='red',
             linestyle='--',
             label='Forecast on Test Data')
    plt.title('ARIMA Forecast vs Actuals')
    plt.xlabel('Date')
    plt.ylabel('Values')
    plt.legend()
    plt.show()
