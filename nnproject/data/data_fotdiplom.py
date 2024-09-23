import requests
import apimoex
import pandas as pd
import numpy as np
import os
import openpyxl
from datetime import datetime

"""АКЦИИ"""
# Весь перечень акций, торгующихся в режиме TQBR: акции и дипозитарные расписки
# request_url = ('https://iss.moex.com/iss/engines/stock/'
#                'markets/shares/boards/TQBR/securities.json')
# arguments = {'securities.columns': ('SECID,'
#                                     'SHORTNAME,'
#                                     'LOTSIZE,'
#                                     'LISTLEVEL,')}
# with requests.Session() as session:
#     iss = apimoex.ISSClient(session, request_url, arguments)
#     data = iss.get()
#     all_ticker_in_TQBR = pd.DataFrame(data['securities']) #ТУТ ЕСТЬ ЛОТ КАЖДОЙ АКЦИИ
#     all_ticker_in_TQBR.to_excel('all_ticker_in_TQBR.xlsx', index=False, engine='openpyxl')
#     shares_in_1_2_listlevel = all_ticker_in_TQBR[all_ticker_in_TQBR['LISTLEVEL'].isin([1, 2])]
#     shares_in_1_2_listlevel.to_excel('shares_in_1_2_listlevel.xlsx', index=False, engine='openpyxl')


# чтение тикеров 1 и 2 листинга из файла
tickers = pd.read_excel('shares_in_1_2_listlevel.xlsx', engine='openpyxl') 
tickers.set_index('SECID', inplace=True)

# получаем список тикеров
ticker_list = np.asarray(tickers.index)

# убираем привелегированные акции, так как они похожи на обыкновенные
size = len(ticker_list)
for i in range(size - 1, 0, -1):
    if (ticker_list[i - 1] in ticker_list[i]):  # 'sber' in 'sberp' => True
        ticker_list = np.delete(ticker_list, i)
# for ticker in ticker_list: # ДОКАЧКА НОВЫХ ДАННЫХ # НУЖНО ВСЕ ПРОДУМАТЬ
#     file_path = f'.\\shares\\{ticker}.csv'
#     if os.path.exists(file_path): #Есть загруженный файл с этим тикером
#         data = pd.read_csv(f'.\\shares\\{ticker}.csv', sep=',', parse_dates=['TRADEDATE'], index_col=0)
#         from_date = data['TRADEDATE'][data.shape[0] - 1]
#         # if (from_date < today): # надо ли это условие
#         with requests.Session() as session:
#             # получение исторических данных с определенными столбцами по конкретной акции
#             data = apimoex.get_board_history(session, ticker, from_date,
#                                              columns=('TRADEDATE', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME'))
#             df = pd.DataFrame(data)
#         # df.set_index('TRADEDATE', inplace=True)
#         df.to_csv(f'.\\shares\\{ticker}.csv', header=False, index=False, mode='a')
#     else:
#         with requests.Session() as session:
#             # получение исторических данных с определенными столбцами по конкретной акции
#             data = apimoex.get_board_history(session, ticker,
#                                              columns=('TRADEDATE', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME'))
#             df = pd.DataFrame(data)
# Получить историю торгов для указанной бумаги в указанном режиме торгов за указанный интервал дат.
# костыльное получение начальной даты котировок, кол-во данных и фильтр на кол-во данных
# with open('volume_of_quotes2.txt', 'w') as file:

# for ticker in ticker_list:
#     with requests.Session() as session:
#         # получение исторических данных с определенными столбцами по конкретной акции
#         data = apimoex.get_board_history(session, ticker, columns=('TRADEDATE','OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME'))
#         df = pd.DataFrame(data)
#     df.set_index('TRADEDATE', inplace=True)
#     #запишем в файл тикер обыкновенных акций и их начало данных с размером
#     size = df.shape[0]  #кол-во данных
#     start_date = df.index[0]  # начала истории
#     # возьмем в обучение нейросети только те акции, которые торгуются более 9 лет
#     if (size > 2000):
#        #почистим от пустых строк
#        df_cleaned = df.dropna(subset=['OPEN'])
#        df_cleaned.to_csv(f'{ticker}.csv')

# иак всего 54 компаний :)
headers = ['ticker', 'name', 'lot']
ticker_list_lot = pd.DataFrame(columns = headers)
for ticker in ticker_list:
    if os.path.exists(f'{ticker}.csv'):
        row = [ticker, tickers.loc[ticker, 'SHORTNAME'], tickers.loc[ticker, 'LOTSIZE']]
        temp_df = pd.DataFrame([row], columns=headers)
        ticker_list_lot = pd.concat([ticker_list_lot, temp_df], ignore_index=True)
        # ticker_list_lot['ticker'] = ticker
        # ticker_list_lot['lot'] = tickers.iloc[ticker,2]#tickers['LOTSIZE']
print(ticker_list_lot)
ticker_list_lot.to_csv('ticker_list_for_site.csv')