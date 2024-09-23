import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import torch.optim as optim
import torch
import os
import torch.nn as nn
from datetime import datetime


def prediction_nn(TICKER, window_future = 15, ta_not_exist = True): # window_future - Сколько дней предсказать
    df = pd.read_csv(f'D:/4/web-app-nn/nnproject/data/{TICKER}.csv', sep =',', parse_dates=['TRADEDATE'], index_col='TRADEDATE')
    df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    # ta_not_exist = True
    if (ta_not_exist): # та не включен
        # Простая скользящая средняя (Simple Moving Averages)
        df['SMA'] = df['Close'].rolling(window=9).mean() # чем больше окно тем гладже будет прямая
        # Экспоненциально взвешенное скользящее среднее (Exponential Moving Average)
        df['EMA'] = df['Close'].ewm(span=9, min_periods=0, adjust=False).mean()

        # Конвергенция-расхождение скользящих средних (Moving Average Convergence Divergence)MACD
        short_ema = df['Close'].ewm(span=12, min_periods=0, adjust=False).mean()
        long_ema = df['Close'].ewm(span=26, min_periods=0, adjust=False).mean()
        df['MACD'] = short_ema - long_ema
        df['signal'] = df['MACD'].ewm(span=9, min_periods=0, adjust=False).mean()
        df['histogram'] = df['MACD'] - df['signal']

        # индикатор относительной силы (RSI)
        delta = df['Close'].diff(1)
        delta.dropna(inplace=True)
        positive = delta.copy()
        negative = delta.copy()
        positive[positive < 0] = 0
        negative[negative > 0] = 0
        days = 14
        avg_gain = positive.rolling(window=days).mean()
        avg_loss = abs(negative.rolling(window=days).mean())
        relative_strength = avg_gain / avg_loss
        df['RSI'] = 100.0 - (100.0 / (1 + relative_strength))

        # Полосы Боллинджера (Bollinger Bands)
        rolling_mean = df['Close'].rolling(window=20).mean() # == SMA
        rolling_std = df['Close'].rolling(window=20).std()
        df['UpperBand'] = rolling_mean + (2 * rolling_std)
        df['LowerBand'] = rolling_mean - (2 * rolling_std)

        # Средний индекс направления (Average Directional Index)
        windows_size = 9
        # Рассчитайте True Range (TR)
        tr = np.maximum((df['High'] - df['Low']),
                         abs(df['High'] - df['Close'].shift(1)),
                         abs(df['Low'] - df['Close'].shift(1)))
        # Directional Movement (DM)
        pdm = df['High'].diff()
        pdm[pdm < 0] = 0
        ndm = df['Low'].diff()
        ndm[ndm > 0] = 0
        # Cкользящее среднее True Range (ATR)
        atr = tr.rolling(window=windows_size).mean()
        # Directional Indicators (DI)
        di_plus = (pdm.rolling(window=windows_size).mean() / atr) * 100
        di_minus = (abs(ndm.rolling(window=windows_size).mean()) / atr) * 100
        # Directional Movement Index (DX)
        dx = (abs(di_plus - di_minus) / (di_plus + di_minus)) * 100
        # Average Directional Index (ADX)
        df['ADX'] = dx.rolling(window=windows_size).mean()

        # Стохастический осциллятор (Stochastic Oscillator)
        windows_size = 9
        # Вычисляем минимальную и максимальную цены за последние windows_size дней
        low_min = df['Low'].rolling(window=windows_size).min()
        high_max = df['High'].rolling(window=windows_size).max()
        # Вычисляем стохастический осциллятор
        df['StochasticOscillator'] = 100 * (df['Close'] - low_min) / (high_max - low_min)
        # Сглаживаем стохастический осциллятор
        df['SmoothedStochastic'] = df['StochasticOscillator'].rolling(window=3).mean()  # Пример сглаживания с помощью скользящего среднего

        df.to_csv(f'D:/4/web-app-nn/nnproject/data/ta/{TICKER}_ta.csv')
    else:
        # Чтение уже готовых датафреймов
        df = pd.read_csv(f'D:/4/web-app-nn/nnproject/data/ta/{TICKER}_ta.csv', parse_dates=['TRADEDATE'],
                             index_col='TRADEDATE')
        # print(df)

    df.dropna(inplace=True)
    # Отбор данных по заданному временному интервалу
    TIME_RANGE2 = datetime(2024,4,1) #df.index[-1]
    mask = (df.index <= TIME_RANGE2)
    df_for_train = df[mask]
    df_for_train.to_csv(f'D:/4/web-app-nn/nnproject/data/ta/train/cleaned_{TICKER}.csv')#, index=None

    # Масштабирование данных
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(df_for_train)

    # SEQ_LEN = 29  # Длина последовательности
    # BATCH_SIZE = 64
    DROPOUT = 0.5
    #
    # # Формирование входных и выходных данных для LSTM
    # num_lines = len(data_scaled)
    # num_col = len(data_scaled[0])
    # x = np.zeros((num_lines - SEQ_LEN, SEQ_LEN, num_col)) # x.shape = (406, SEQ_LEN, 16) будет 406 двумерных массива размерностью SEQ_LEN на 5
    # y = np.zeros((num_lines - SEQ_LEN, 1, num_col))
    #
    # for i in range(num_lines - SEQ_LEN):
    #     x[i] = data_scaled[i:i+SEQ_LEN]
    #     y[i] = data_scaled[i+SEQ_LEN]
    # x = np.array(x)
    # y = np.array(y)
    # print(f'x = {x.shape}')
    # print(f'y = {y.shape}')
    #
    # # Разделение данных на обучающий и тестовый наборы
    # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,random_state=42, shuffle=False) #random_state=42 гарантирует, что каждый раз, когда вы выполняете разделение данных, вы получите одинаковые обучающие и тестовые наборы.
    # print(f"x_train={len(x_train)}, x_test={len(x_test)},\ny_train={len(y_train)},y_test={len(y_test)},")
    #
    # # Преобразование в тензоры PyTorch
    # x_train = torch.tensor(x_train, dtype=torch.float32)
    # y_train = torch.tensor(y_train, dtype=torch.float32)
    # x_test = torch.tensor(x_test, dtype=torch.float32)
    # y_test = torch.tensor(y_test, dtype=torch.float32)
    #
    # train_dataset = TensorDataset(x_train, y_train)
    # test_dataset = TensorDataset(x_test, y_test)
    #
    # train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    # test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    #
    class LSTM_Model(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers, output_size):
            super(LSTM_Model, self).__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            # Инициализация LSTM слоя
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
            self.dropout = nn.Dropout(p=DROPOUT)
            # Полносвязный слой для получения выходных данных
            self.fc = nn.Linear(hidden_size, output_size)

        def forward(self, x):
            # Инициализация скрытого и клеточного состояний LSTM
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

            # Проход через LSTM слой
            out, _ = self.lstm(x, (h0, c0))

            # Выходные данные через полносвязный слой
            out = self.fc(out[:, -1, :])  # Берем последний временной шаг
            return out
        # def load_state_dict(self, state_dict):
        #     own_state = self.state_dict()
        #     for name, param in state_dict.items():
        #         if name in own_state:
        #             own_state[name].copy_(param)
    #
    # # Создание модели LSTM
    input_size = len(df_for_train.columns) # 16
    hidden_size = 150
    num_layers = 1
    output_size = input_size
    model = LSTM_Model(input_size, hidden_size, num_layers, output_size)

    # name_model = f'lstm_model_TA_{TICKER}_{input_size}_{hidden_size}_{num_layers}_{output_size}'
    model_path = r'D:/4/web-app-nn/nnproject/start/100ep_lstm_model_TA_SBER_16_150_1_16_batch64_rdp05.pth'#'D:\\4\\web-app-nn\\nnproject\\start100ep_lstm_model_TA_SBER_16_150_1_16_batch64_rdp05.pth'
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'))) #для восстановления состояний модели, Если вы сохранили контрольную точку модели во время обучения, вы можете использовать этот метод для восстановления состояния модели.
    else:
        print(f'Model file not found at {model_path}')
    # Определение функции потерь и оптимизатора
    # criterion = nn.MSELoss()
    # optimizer = optim.Adam(model.parameters(), lr=0.001) # weight_decay=0.1

    """Прогноз на 15 дней"""

    window_prediction = 30 # Данные для прогнозирования

    # Подготовка данных для прогнозирования на будущее
    future_time_points_15 = pd.bdate_range(start=df_for_train.index[-1] + pd.Timedelta(days=1), periods=window_future)
    future_data_15 = pd.DataFrame(index=future_time_points_15, columns=df_for_train.columns) #(window_future, 5)
    features = data_scaled[-window_prediction:]#берет последние window_prediction тут как раз до TIME_RANGE2
    model.eval()
    # Заполнение прогнозов
    for i, time_point in enumerate(future_time_points_15):
        # Преобразование в тензор PyTorch
        features_tensor = torch.tensor(features, dtype=torch.float32)
        features_tensor = features_tensor.unsqueeze(0)  # Добавление измерения батча
        # Получение прогноза от модели
        with torch.no_grad():
            prediction = model(features_tensor)
        # Запись прогноза в DataFrame для будущего времени
        future_data_15.iloc[i] = prediction.numpy()
        features = np.vstack((features, future_data_15.iloc[i].astype(np.float32)))

    # Получение прогнозов на будущее
    prediction15 = scaler.inverse_transform(future_data_15)
    predicted_df15 = pd.DataFrame(prediction15, index=future_data_15.index, columns=future_data_15.columns)
    # print("Прогнозы на будущее:")
    # print(predicted_df15)

    # mask1 = (pd.to_datetime(df.index) >= predicted_df15.index[0]) & (pd.to_datetime(df.index) <= predicted_df15.index[-1])
    # true_df15 = df[mask1]
    # print("Фактические:")
    # print(true_df15)

    # Построение линейного графика
    # plt.figure(figsize=(20, 8))
    # # days_show = 150
    # plt.plot(true_df15.index, predicted_df15['Close'], label = 'Предсказанные на 15 дней', marker='o', color='b')
    # plt.plot(true_df15.index, true_df15['Close'], label = 'Истинные на 15 дней', marker='x', color='r')
    # plt.xticks(true_df15.index,rotation=45)
    # plt.gca().xaxis.set_major_locator(mdates.DayLocator())
    # plt.grid(True)
    # plt.title(f'График {TICKER} Close price')
    # plt.xlabel('Дата')
    # plt.ylabel('Цена')
    # plt.legend()
    # plt.show()
    #
    # mae_predicred15 = mean_absolute_error(true_df15['Close'], predicted_df15['Close'])
    # mse_predicred15 = mean_squared_error(true_df15['Close'], predicted_df15['Close'])
    # rmse_predicred15 = np.sqrt(mse_predicred15)#mean_squared_error(y_test_2d, test_outputs, squared=False)  # Вычисление RMSE как корень из MSE
    # r2_15 = r2_score(true_df15['Close'], predicted_df15['Close']) #Он обеспечивает показатель того, насколько хорошо наблюдаемые результаты воспроизводятся моделью, на основе доли общего изменения результатов, объясняемых моделью.
    #
    # # Вывод результатов
    # print("Mean Absolute Error (MAE):", mae_predicred15)
    # print("Mean Squared Error (MSE):", mse_predicred15)
    # print("Root Mean Squared Error (RMSE):", rmse_predicred15)
    # print("R2:", r2_15)
    # Рублей................

    # Рассчет направлений движения (1 - рост, 0 - падение)
    # true_directions = np.diff(true_df15) > 0
    # pred_directions = np.diff(predicted_df15) > 0
    # accuracy = np.mean(true_directions == pred_directions)
    # print(f"Accuracy of Directional Prediction: {accuracy * 100}%")

    # Прогнозируемое направление движения
    # pred_directions = np.diff(predicted_df15['Close']) > 0
    # initial_investment = 1000
    # y_true = true_df15['Close']
    # # Доходность инвестиций на основе прогнозов
    # investment_returns = initial_investment
    # for i in range(len(pred_directions)):
    #     if pred_directions[i]:
    #         investment_returns *= (y_true[i + 1] / y_true[i])
    # print(investment_returns)
    # print(f'Доходность на 15 дней, если знаем какие есть цены: {((investment_returns * 100)/initial_investment - 100):.2f}%')

    # На сколько вырастет(>0)/упадет(<0) цена, если купить акцию сейчас
    predicted_change_price = round((predicted_df15['Close'].iloc[-1] / predicted_df15['Close'].iloc[0] * 100) - 100, 2)
    return predicted_change_price, predicted_df15['Close'].iloc[0]

"""Прогноз на 30 дней"""
# window_future = 30 # Сколько дней предсказать 15-01-24 to 15-02-24
#
# # Подготовка данных для прогнозирования на будущее
# future_time_points_30 = pd.bdate_range(start= df_for_train.index[-1] + pd.Timedelta(days=1), periods=window_future) #df_for_train.index[-1] #datetime.strptime('2024-03-10', "%Y-%m-%d")
# future_data_30 = pd.DataFrame(index=future_time_points_30, columns=df_for_train.columns)
# features = data_scaled[-window_prediction:]#берет последние window_prediction
#
# # Заполнение прогнозов
# for i, time_point in enumerate(future_time_points_30):
#     # Преобразование в тензор PyTorch
#     features_tensor = torch.tensor(features, dtype=torch.float32)
#     features_tensor = features_tensor.unsqueeze(0)  # Добавление измерения батча
#     # Получение прогноза от модели
#     with torch.no_grad():
#         prediction = model(features_tensor)
#     # Запись прогноза в DataFrame для будущего времени
#     future_data_30.iloc[i] = prediction.numpy()
#     features = np.vstack((features, future_data_30.iloc[i].astype(np.float32)))
#
# # Получение прогнозов на будущее
# # print("Прогнозы на будущее:")
# print(future_data_30)
# prediction30 = scaler.inverse_transform(future_data_30)
# predicted_df30 = pd.DataFrame(prediction30, index=future_data_30.index, columns=future_data_30.columns)
# # print("Прогнозы на будущее:")
# # print(predicted_df30)
#
# # Фактические
# mask1 = (pd.to_datetime(df.index) >= predicted_df30.index[0]) & (pd.to_datetime(df_ta_main.index) <= predicted_df30.index[-1])
# true_df30 = df[mask1]
# # print("Фактические:")
# # print(true_df30)
#
# # Построение линейного графика
# plt.figure(figsize=(20, 8))
# # days_show = 150
# plt.plot(true_df30.index, predicted_df30['Close'], label = 'Предсказанные на 30 дней', marker='o', color='b')
# plt.plot(true_df30.index, true_df30['Close'], label = 'Истинные', marker='x', color='r')
# plt.xticks(true_df30.index,rotation=45)
# plt.gca().xaxis.set_major_locator(mdates.DayLocator())
# plt.grid(True)
# plt.title(f'График {TICKER}')
# plt.xlabel('Дата')
# plt.ylabel('Цена')
# plt.legend()
# plt.show()
#
# mae_predicred30 = mean_absolute_error(true_df30['Close'], predicted_df30['Close'])
# mse_predicred30 = mean_squared_error(true_df30['Close'], predicted_df30['Close'])
# rmse_predicred30 = np.sqrt(mse_predicred30)#mean_squared_error(y_test_2d, test_outputs, squared=False)  # Вычисление RMSE как корень из MSE
# r2_30 =r2_score(true_df15['Close'], predicted_df15['Close']) #Он обеспечивает показатель того, насколько хорошо наблюдаемые результаты воспроизводятся моделью, на основе доли общего изменения результатов, объясняемых моделью.
#
# # Вывод результатов
# print("Mean Absolute Error (MAE):", mae_predicred30)
# print("Mean Squared Error (MSE):", mse_predicred30)
# print("Root Mean Squared Error (RMSE):", rmse_predicred30)
# print("R2:", r2_30)
# # Рублей................
#
# # Рассчет направлений движения (1 - рост, 0 - падение)
# true_directions = np.diff(true_df30) > 0
# pred_directions = np.diff(predicted_df30) > 0
# accuracy = np.mean(true_directions == pred_directions)
# print(f"Accuracy of Directional Prediction: {accuracy * 100}%")
#
# # Прогнозируемое направление движения
# pred_directions = np.diff(predicted_df30['Close']) > 0
# initial_investment = 1000
# y_true = true_df30['Close']
# # Доходность инвестиций на основе прогнозов
# investment_returns = initial_investment
# for i in range(len(pred_directions)):
#     if pred_directions[i]:
#         investment_returns *= (y_true[i + 1] / y_true[i])
# print(investment_returns)
# print(f'Доходность на 30 дней: {((investment_returns * 100)/initial_investment - 100):.2f}%')

# # Библиотеки, необходимые для работы с данными и их отображения
# import pandas as pd
# import numpy as np
# from datetime import datetime
# import matplotlib.pyplot as plt
# import matplotlib.dates as mdates
# 
# # Библиотека за измерения времени обучения
# import time
# 
# # Библиотеки, необходимые для вычисления метрик
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# 
# # Библиотеки, необходимые для обучения модели
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import MinMaxScaler
# import torch.optim as optim
# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader, TensorDataset
# 
# # Наименование акции
# TICKER = 'SBER'
# 
# # Чтение импортированного файла, переименование столбцов, установка столбца даты индексом
# df = pd.read_csv(f'/content/drive/MyDrive/Colab Notebooks/Data/{TICKER}.csv', sep =',', parse_dates=['TRADEDATE'])
# df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
# df.set_index('Date', inplace=True)
# 
# # Индикаторы технического анализа:
# # Простая скользящая средняя (Simple Moving Averages)
# df['SMA'] = df['Close'].rolling(window=9).mean()
# 
# # Экспоненциально взвешенное скользящее среднее (Exponential Moving Average)
# df['EMA'] = df['Close'].ewm(span=9, min_periods=0, adjust=False).mean()
# 
# # Конвергенция-расхождение скользящих средних (Moving Average Convergence Divergence)MACD
# short_ema = df['Close'].ewm(span=12, min_periods=0, adjust=False).mean()
# long_ema = df['Close'].ewm(span=26, min_periods=0, adjust=False).mean()
# df['MACD'] = short_ema - long_ema
# df['signal'] = df['MACD'].ewm(span=9, min_periods=0, adjust=False).mean()
# df['histogram'] = df['MACD'] - df['signal']
# 
# # Индикатор относительной силы (RSI)
# delta = df['Close'].diff(1)
# delta.dropna(inplace=True)
# positive = delta.copy()
# negative = delta.copy()
# positive[positive < 0] = 0
# negative[negative > 0] = 0
# days = 14
# avg_gain = positive.rolling(window=days).mean()
# avg_loss = abs(negative.rolling(window=days).mean())
# relative_strength = avg_gain / avg_loss
# df['RSI'] = 100.0 - (100.0 / (1 + relative_strength))
# 
# # Полосы Боллинджера (Bollinger Bands)
# rolling_mean = df['Close'].rolling(window=20).mean()
# rolling_std = df['Close'].rolling(window=20).std()
# df['UpperBand'] = rolling_mean + (2 * rolling_std)
# 
# # Средний индекс направления (Average Directional Index)
# windows_size = 9
# # True Range (TR)
# tr = np.maximum((df['High'] - df['Low']),
#                  abs(df['High'] - df['Close'].shift(1)),
#                  abs(df['Low'] - df['Close'].shift(1)))
# # Directional Movement (DM)
# pdm = df['High'].diff()
# pdm[pdm < 0] = 0
# ndm = df['Low'].diff()
# ndm[ndm > 0] = 0
# # скользящее среднее True Range (ATR)
# atr = tr.rolling(window=windows_size).mean()
# # Directional Indicators (DI)
# di_plus = (pdm.rolling(window=windows_size).mean() / atr) * 100
# di_minus = (abs(ndm.rolling(window=windows_size).mean()) / atr) * 100
# # Directional Movement Index (DX)
# dx = (abs(di_plus - di_minus) / (di_plus + di_minus)) * 100
# # Average Directional Index (ADX)
# df['ADX'] = dx.rolling(window=windows_size).mean()
# 
# # Стохастический осциллятор (Stochastic Oscillator)
# windows_size = 9
# # Минимальная и максимальная цены за последние windows_size дней
# low_min = df['Low'].rolling(window=windows_size).min()
# high_max = df['High'].rolling(window=windows_size).max()
# # Вычисляем стохастический осциллятор
# df['StochasticOscillator'] = 100 * (df['Close'] - low_min) / (high_max - low_min)
# # Сглаживаем стохастический осциллятор
# df['SmoothedStochastic'] = df['StochasticOscillator'].rolling(window=3).mean()
# 
# # Удаляем строки с NaN и сохраняем в файл
# df.dropna(inplace=True)
# df.to_csv(f'/content/drive/MyDrive/Colab Notebooks/TA/{TICKER}_TA.csv')
# 
# # Отбор данных по заданному временному интервалу и сохранение в файл
# TIME_RANGE2 = datetime(2024,3,14)
# mask = df.index <= TIME_RANGE2
# df_for_train = df[mask]
# df_for_train.to_csv(f'/content/drive/MyDrive/Colab Notebooks/TA/clean_{TICKER}_TA_library.csv')
# 
# # Отображение временного ряда в графическом виде
# plt.figure(figsize=(20, 8))
# plt.plot(df_for_train.index, df_for_train['Close'])
# plt.xticks(df_for_train.index[::180],rotation=90)
# plt.title(f'График цены на {TICKER} during {df_for_train.index[0]} - {TIME_RANGE2}')
# plt.xlabel('Дата')
# plt.ylabel('Цена')
# plt.show()
# 
# # Нормализация данных
# scaler = MinMaxScaler()
# data_scaled = scaler.fit_transform(df_for_train)
# 
# # Длина последовательности
# SEQ_LEN = 29
# # Размер батча
# BATCH_SIZE = 64
# 
# # Формирование входных и выходных данных для LSTM
# num_lines = len(data_scaled)
# num_col = len(data_scaled[0])
# x = np.zeros((num_lines - SEQ_LEN, SEQ_LEN, num_col))
# y = np.zeros((num_lines - SEQ_LEN, 1, num_col))
# 
# for i in range(num_lines - SEQ_LEN):
#     x[i] = data_scaled[i:i+SEQ_LEN]
#     y[i] = data_scaled[i+SEQ_LEN]
# x = np.array(x)
# y = np.array(y)
# 
# # Разделение данных на обучающий и тестовый наборы
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,random_state=42, shuffle=False)
# 
# # Преобразование в тензоры PyTorch
# x_train = torch.tensor(x_train, dtype=torch.float32)
# y_train = torch.tensor(y_train, dtype=torch.float32)
# x_test = torch.tensor(x_test, dtype=torch.float32)
# y_test = torch.tensor(y_test, dtype=torch.float32)
# 
# # Создание TensorDataset и DataLoader для тестовых и обучающих данных
# train_dataset = TensorDataset(x_train, y_train)
# test_dataset = TensorDataset(x_test, y_test)
# train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
# 
# # Определяем класс LSTM, который наследует класс nn.Moudule PyTorch
# class LSTM_Model(nn.Module):
#     def __init__(self, input_size, hidden_size, num_layers, output_size):
#         super(LSTM_Model, self).__init__()
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
#         # Инициализация LSTM слоя
#         self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
#         # Применение Dropout с вероятностью обнуления равной 0,5
#         self.dropout = nn.Dropout(p=0.5)
#         # Полносвязный слой для получения выходных данных
#         self.fc = nn.Linear(hidden_size, output_size)
# 
#     def forward(self, x):
#         # Инициализация скрытого и клеточного состояний LSTM
#         h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
#         c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
# 
#         # Проход через LSTM слой
#         out, _ = self.lstm(x, (h0, c0))
# 
#         # Выходные данные через полносвязный слой
#         out = self.fc(out[:, -1, :])  # Берем последний временной шаг
#         return out
# 
# # Создание модели LSTM
# input_size = len(df_for_train.columns)
# hidden_size = 150 # Количество нейронов в скрытом слое
# num_layers = 1    # Количество слоев
# output_size = input_size
# model = LSTM_Model(input_size, hidden_size, num_layers, output_size)
# 
# # Восстановление состояния модели
# name_model = f'lstm_model_{TICKER}_{input_size}_{hidden_size}_{num_layers}_{output_size}'
# # model.load_state_dict(torch. load(f'/content/drive/MyDrive/Colab Notebooks/Models/{name_model}.pth'))
# 
# # Определение функции потерь и оптимизатора
# criterion = nn.MSELoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)
# 
# # Создание файла для отслеживания снижения потерь
# f = open(f'/content/drive/MyDrive/Colab Notebooks/Experiments/TA/exp_{num_epochs}epochs_{name_model}.xlsx', 'w')
# f.write('Epoch,Loss\n')
# 
# # Обучение модели
# num_epochs = 100
# start_time = time.time()
# for epoch in range(num_epochs):
#     epoch_loss = 0
#     # Режим обучения
#     model.train()
#     for batch_x, batch_y in train_loader:
#       # Обнуление градиентов в оптимизаторе
#       optimizer.zero_grad()
#       # Прогнозирование
#       outputs = model(x_train)
#       y_train_2d = y_train.reshape(-1, y_train.shape[-1])
#       # Вычисление потерь
#       loss = criterion(outputs, y_train_2d)
#       # Обратное распространение ошибки для вычисления градиентов, выполняя
#       loss.backward()
#       # Обновление параметров модели на основе гравдиентов
#       optimizer.step()
#       epoch_loss += loss.item()
#     epoch_loss /= len(train_loader)
#     if (epoch+1) % 25 == 0:
#         print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.16f}')
#     f.write(f'{epoch+1}/{num_epochs},{epoch_loss:.16f}\n')
# print(f' На обучение ушло {round(time.time() - start_time, 2)} сек')
# f.write(f' На обучение ушло {round(time.time() - start_time, 2)} сек')
# f.close()
# 
# # Оценка модели
# model.eval()
# y_test = y_test.squeeze(1)
# with torch.no_grad():
#     outputs  = model(x_test)
#     y_test_2d = y_test.reshape(-1, y_test.shape[-1])
#     test_loss = criterion(outputs, y_test_2d) #0.063439
# print(f'Train Loss: {loss.item()}')
# print(f'Test Loss: {test_loss.item()}')
# 
# # Вычисление метрик на тестовых данных
# # Обучающие данные
# train_outputs = model(x_train).detach().numpy()
# y_train_2d = y_train.reshape(-1, y_train.shape[-1])
# mae_train = mean_absolute_error(y_train_2d, train_outputs)
# mse_train = mean_squared_error(y_train_2d, train_outputs)
# rmse_train = np.sqrt(mse_train)
# 
# # Тестовые данные
# test_outputs = model(x_test).detach().numpy()
# y_test_2d = y_test.reshape(-1, y_test.shape[-1])
# mae_test = mean_absolute_error(y_test_2d, test_outputs)
# mse_test = mean_squared_error(y_test_2d, test_outputs)
# rmse_test = np.sqrt(mse_test)
# 
# # Вывод результатов
# print("TRAIN: Mean Absolute Error (MAE):", mae_train)
# print("TEST: Mean Absolute Error (MAE):", mae_test)
# print("TRAIN: Mean Squared Error (MSE):", mse_train)
# print("TEST: Mean Squared Error (MSE):", mse_test)
# print("TRAIN: Root Mean Squared Error (RMSE):", rmse_train)
# print("TEST: Root Mean Squared Error (RMSE):", rmse_test)
# 
# # Сохранение состояния модели
# torch.save(model.state_dict(), f'/content/drive/MyDrive/Colab Notebooks/Models/{num_epochs}epoch_{name_model}.pth')
# 
# # СТОИТ ЛИ ВКЛЮЧАТЬ ИНВЕРСТИРОВАНИЕ ТЕСТОВЫХ И ОБУЩАЮЩИХ ДАННЫХ ИХ
# 
# # Прогнозирование
# # Данные для прогнозирования
# data_for_prediction = 200
# # Сколько дней нужно предсказать
# days_for_prediction = 15
# 
# # Подготовка данных для прогнозирования на будущее
# list_future_date = pd.bdate_range(start=df_for_train.index[-1] + pd.Timedelta(days=1), periods=days_for_prediction)
# future_data = pd.DataFrame(index=list_future_date, columns=df_for_train.columns)
# features = data_scaled[-data_for_prediction:]
# 
# # Заполнение прогнозов
# for i, time_point in enumerate(list_future_date):
#     # Преобразование в тензор PyTorch
#     features_tensor = torch.tensor(features, dtype=torch.float32)
#     # Добавление измерения батча
#     features_tensor = features_tensor.unsqueeze(0)
#     # Получение прогноза от модели
#     with torch.no_grad():
#         prediction = model(features_tensor)
#     # Запись прогноза в DataFrame для будущего времени
#     future_data.iloc[i] = prediction.numpy()
#     features = np.vstack((features, future_data.iloc[i].astype(np.float32)))
# 
# # Получение прогнозов на будущее
# prediction = scaler.inverse_transform(future_data)
# predicted_df = pd.DataFrame(prediction, index=future_data.index, columns=future_data.columns)
# print("Предсказанные:")
# print(predicted_df)
# 
# # Отбор фактических данных для графика
# mask1 = (pd.to_datetime(df.index) >= predicted_df.index[0]) & (pd.to_datetime(df.index) <= predicted_df.index[-1])
# true_df = df[mask1]
# print("Фактические:")
# print(true_df)
# 
# # Построение линейного графика
# plt.figure(figsize=(20, 8))
# plt.plot(true_df.index, predicted_df['Close'], label = f'Предсказанные на {days_for_prediction} дней', marker='o', color='b')
# plt.plot(true_df.index, true_df['Close'], label = f'Истинные на {days_for_prediction} дней', marker='x', color='r')
# plt.xticks(true_df.index,rotation=45)
# plt.gca().xaxis.set_major_locator(mdates.DayLocator())
# plt.grid(True)
# plt.title(f'График {TICKER} Close price')
# plt.xlabel('Дата')
# plt.ylabel('Цена')
# plt.legend()
# plt.show()
# 
# # Вычисление метрик спрогнозированных данных
# mae_predicred = mean_absolute_error(true_df['Close'], predicted_df['Close'])
# mse_predicred = mean_squared_error(true_df['Close'], predicted_df['Close'])
# rmse_predicred = np.sqrt(mse_predicred)
# r2 = r2_score(true_df['Close'], predicted_df['Close'])
# 
# # Вывод результатов
# print(f"Mean Absolute Error (MAE): {mae_predicred: .2f}")
# print(f"Mean Squared Error (MSE): {mse_predicred: .2f}")
# print(f"Root Mean Squared Error (RMSE): {rmse_predicred: .2f}")
# print(f"R2:  {r2: .2f}")
