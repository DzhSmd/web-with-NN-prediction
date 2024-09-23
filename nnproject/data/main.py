import requests
import apimoex
import pandas as pd
import numpy as np
from datetime import datetime

"""АКЦИИ"""
# Весь перечень акций, торгующихся в режиме TQBR: акции и дипозитарные расписки
# Это делается в самом начале, один раз, так как не так часто бывает IPO (или не один.....)
request_url = ('https://iss.moex.com/iss/engines/stock/'
               'markets/shares/boards/TQBR/securities.json')
arguments = {'securities.columns': ('SECID,'
                                    'SHORTNAME,'
                                    'SECTORID,'
                                    #'LOTSIZE,'
                                    #'LISTLEVEL,'
                                    )}
with requests.Session() as session:
    iss = apimoex.ISSClient(session, request_url, arguments)
    data = iss.get()
    all_ticker_in_TQBR = pd.DataFrame(data['securities']) #ТУТ ЕСТЬ ЛОТ КАЖДОЙ АКЦИИ
    print(all_ticker_in_TQBR)
#     all_ticker_in_TQBR.to_excel('all_ticker_in_TQBR.xlsx', index=False, engine='openpyxl')
#     shares_in_1_2_listlevel = all_ticker_in_TQBR[all_ticker_in_TQBR['LISTLEVEL'].isin([1, 2])]
#     shares_in_1_2_listlevel.to_excel('shares_in_1_2_listlevel.xlsx', index=False, engine='openpyxl')
# all_ticker_lot_in_TQBR.set_index('SECID', inplace=True) # так как при чтении все равно придется заменять тикер на индекс, лучше убрать эту строку

# # чтение тикеров 1 и 2 листинга из файла
# tickers = pd.read_excel('shares_in_1_2_listlevel.xlsx', engine='openpyxl') #ПОСЛЕ ОБУЧЕНИЯ НЕЙРОНКИ ОТСЮДА ПО ТИКЕРУ ВЗЯТЬ ЛОТНОСТЬ
# tickers.set_index('SECID', inplace=True)
#
# # получаем список тикеров
# ticker_list = np.asarray(tickers.index)
#
# # убираем привелегированные акции, так как они похожи на обыкновенные
# size = len(ticker_list)
# # print(size)
# for i in range(size - 1, 0, -1):
#     if (ticker_list[i - 1] in ticker_list[i]):  # 'sber' in 'sberp' => True
#         ticker_list = np.delete(ticker_list, i)
# # print(len(ticker_list))
# # print(ticker_list)
#
# # Получить историю торгов для указанной бумаги в указанном режиме торгов за указанный интервал дат.
# # костыльное получение начальной даты котировок, кол-во данных и фильтр на кол-во данных
# with open('volume_of_quotes2.txt', 'w') as file:
#     for ticker in ticker_list:
#         with requests.Session() as session:
#             # получение исторических данных с определенными столбцами по конкретной акции
#             data = apimoex.get_board_history(session, ticker, columns=('TRADEDATE','OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME'))
#             df = pd.DataFrame(data)
#         df.set_index('TRADEDATE', inplace=True)
#
#         #запишем в файл тикер обыкновенных акций и их начало данных с размером
#         size = df.shape[0]  #кол-во данных
#         start_date = df.index[0]  # начала истории
#         file.write(f"{ticker}  {start_date}  {str(size)}\n")
#
#         # возьмем в обучение нейросети только те акции, которые торгуются более 9 лет
#         if (size > 2000):
#            #почистим от пустых строк
#            df_cleaned = df.dropna(subset=['OPEN'])
#            df_cleaned.to_csv(f'.\\data\\shares\\{ticker}.csv')
# итак всего 54 компаний :)

"""1h"""
# ticker_list = ['SBER', 'MOEX', 'VTBR',  # 'TCSG',   # финансы
#                'ROSN', 'LKOH', 'NVTK', 'GAZP', 'SIBN', 'SNGS', 'TATN',  # энергетика
#                'PLZL', 'CHMF', 'NLMK', 'PHOR', 'ALRS',  # материалы и сырье
#                # 'POSI', #'VKCO',  # инф технологии
#                'FIXP', 'MVID',  # 'DSKY',  # товары 2ой необ
#                'MGNT', 'GCHE', 'BELU',  # 'LENT', # товары 1ой необ
#                'HYDR', 'IRAO', 'LSNG',  # ком услуги
#                'PIKK', 'AFLT',  # промышленность
#                'YNDX', 'MTSS', 'RTKM',  # коммуникация
#                # 'MDMG'  # мало данных здароохранение депозитарная расписка?
#                ]
# Чтобы поулчить больше данных для обучения возьмем таймфрем поменьше дневного, н-р, 1 час

# Получить таблицу интервалов доступных дат для свечей различного размера в указанном режиме торгов.
# with open('history_candle_ticker.txt', 'a') as f: # как можно было вывести на экран или в файл: 1 столбец-тикер, 2столбец- end-begin будничные дни, 3столбец - таймфрейм
#     for ticker in ticker_list:
#         f.write(f"{ticker}\n")
#         with requests.Session() as session:
#             data = apimoex.get_board_candle_borders(session, ticker, 'TQBR', 'shares', 'stock')
#             df = pd.DataFrame(data)
#         df.to_csv(f, sep=',', index=False)
#        # f.write('\n')
#     print(df)

# нужно ли обновить данные?
# update = False
# interval = 60  # изменить таймфрейм: 60-1ч, 7-неделя, 4-квартал, 10-10 минут
# today = datetime.now()

# Получить свечи в формате HLOCV указанного инструмента в указанном режиме торгов за интервал дат.
# for interval in [60, 10]: # 60 - 1ч, 7 - неделя, 4 - квартал, 10 - минут
# for ticker in ['MDMG']:#:ticker_list:  # ticker_list:
#     if (update):
#         data = pd.read_csv(f'.\\data\\shares\\{interval}\\{ticker}_{interval}.csv', sep=',', parse_dates=['begin'],
#                            index_col=0)  # f'/data/shares/{interval}/{ticker}_{interval}.csv'
#         from_date = data['begin'][data.shape[0] - 1]
#         # if (from_date < today): # надо ли это условие
#         with requests.Session() as session:
#             data_new = apimoex.get_board_candles(session, ticker, interval, from_date, columns=("begin", "open", "close","high","low","volume"))
#             df = pd.DataFrame(data_new)
#         df.to_csv(f'.\\data\\shares\\{interval}\\{ticker}_{interval}.csv', header=False, index=False, mode='a')
#     else: #f'.\\data\\shares\\{ticker}.csv'
#         with requests.Session() as session:
#             data = apimoex.get_board_candles (session, ticker, interval)
#             df = pd.DataFrame(data)
#             df.to_csv(f'.\\data\\shares\\{interval}\\{ticker}_{interval}.csv', index=False)

"""ОБЛИГАЦИИ"""
# Получить свечи в формате HLOCV указанного инструмента в указанном режиме торгов за интервал дат.
# interval = 1  # изменить таймфрейм: 60-1ч, 7-неделя, 4-квартал, 10-10 минут
# isin = 'RU000A1008D7'
# with requests.Session() as session:
#     data_bonds = apimoex.get_board_candles(session, isin, interval, board='TQCB', market='bonds')
#     df = pd.DataFrame(data_bonds)
# df.to_csv(f'.\\data\\bonds\\{isin}_{interval}.csv', index=False)

# Это делается в самом начале, один раз, так как не так часто бывает первичное размещение ХОТЯ! (или не один.....)
# Весь перечень облиг, торгующихся в режиме TQCB: облиги
# request_url = ('https://iss.moex.com/iss/engines/stock/'
#                'markets/bonds/boards/TQCB/securities.json')
# arguments = {'securities.columns': ('SECID,'  # Код инструмента == ISIN
#                                     # 'YIELDATPREVWAPRICE,' #Доходность по оценке пред. дня
#                                     # 'COUPONVALUE,' #Сумма купона
#                                     # 'ACCRUEDINT,'#НКД
#                                     'MATDATE,' #Дата погашения
#                                     # 'ISSUESIZE,' #Объем выпуска
#                                     # 'STATUS,' #статус
#                                     # 'PREVPRICE,' #Последняя цена пред.дня
#                                     'LISTLEVEL,'  # Уровень листинга = 1
#                                     # 'OFFERDATE,'  # Дата оферты ==null(date)
#                                     'LOTVALUE,'  # Номинал лота = 1000.0
#                                     'BUYBACKPRICE,'  # Цена оферты !=100??? вложенный is??
#                                     'PREVLEGALCLOSEPRICE,'  # Официальная цена закрытия предыдущего дня !=null
#                                     'FACEUNIT,'  # валюта номинала == 'SUR'
#                                     # 'BUYBACKDATE,'  # Дата, к кот.рассч.доходность = '0000-00-00'(date)
#                                     'COUPONPERCENT,'  # Ставка купона, % != null
#                                     )}
# with requests.Session() as session:
#     iss = apimoex.ISSClient(session, request_url, arguments)
#     data = iss.get()
#     df_tqcb = pd.DataFrame(data['securities'])
# df_tqcb.to_csv('.\\data\\all_bonds_in_TQCB.csv') #(2153, 12) # все облиги на фр
# # Фильтр: возьмем только облиги в 1 листинге без оферт, с купонной ставкой, с номиналом 1000 руб, и в пред день торговалась
# # mask = ((df_tqcb['OFFERDATE'] == null)
# mask = (df_tqcb['LISTLEVEL'] == 1) & (df_tqcb['LOTVALUE'] == 1000.0) & \
#        (df_tqcb['FACEUNIT'] == 'SUR') & (df_tqcb['BUYBACKPRICE'] != 100.0)
# clean_all_bonds_in_TQCB = df_tqcb[mask]
# clean_all_bonds_in_TQCB = clean_all_bonds_in_TQCB.dropna(subset=['PREVLEGALCLOSEPRICE', 'COUPONPERCENT'])
# clean_all_bonds_in_TQCB.set_index('SECID', inplace=True)
# clean_all_bonds_in_TQCB.to_csv('.\\data\\clean_all_bonds.csv') # (112,6) облиги для анализа и вывода на сайт
# #print(clean_all_bonds_in_TQCB.shape) #112,6


# Получить интервал дат, доступных в истории для рынка  по заданному режиму торгов TQCB
# with requests.Session() as session:
#     data = apimoex.get_board_dates(session, 'TQCB',  'bonds',  'stock')
#     print(data)#{'from': '2019-10-14', 'till': '2024-05-03'}

# берем исторические данные по некоторым облигам для ОБУЧЕНИЯ нейронки просто так)))
# bonds_list = clean_all_bonds_in_TQCB.index
# bonds_isin.set_index('SECID', inplace=True)
# print(bonds_list)
# bonds_list = pd.read_csv('.\\data\\clean_all_bonds.csv')
# with open('volume_of_bonds1.txt', 'w') as file:
#     for oblig in bonds_list['SECID']:
#         with requests.Session() as session:
#             data = apimoex.get_board_history(session, oblig,
#                 columns = ('TRADEDATE', 'VOLUME', 'OPEN','LOW','HIGH','LEGALCLOSEPRICE', 'YIELDATWAP', 'MARKETPRICE','COUPONVALUE'),
#                                              board="TQCB", market="bonds", engine = 'stock')
#             df = pd.DataFrame(data)
#         df = df[df['VOLUME'] != 0]
#         size = df.shape[0]
#         if (size > 1000):
#             df.to_csv(f'.\\data\\bonds\\{oblig}.csv')
#             # print(oblig, " ",size)
#             # file.write(f"{oblig}  {str(size)}\n")


""" ШЛАК """
# Мой список акций по возможным секторам (их тут 9), более предпочтительнее, не брала привелигированные
# ticker_list = ['SBER', 'MOEX', 'VTBR', #'TCSG',   # финансы
#                'ROSN', 'LKOH', 'NVTK', 'GAZP', 'SIBN', 'SNGS', 'TATN',  # энергетика
#                'PLZL', 'CHMF', 'NLMK', 'PHOR', 'ALRS',  # материалы и сырье
#                #'POSI', #'VKCO',  # инф технологии
#                'FIXP', 'MVID', #'DSKY',  # товары 2ой необ
#                'MGNT', 'GCHE',  'BELU', #'LENT', # товары 1ой необ
#                'HYDR', 'IRAO', 'LSNG',  # ком услуги
#                'PIKK', 'AFLT',  # промышленность
#                'YNDX', 'MTSS', 'RTKM',  # коммуникация
#                #'MDMG'  # мало данных здароохранение депозитарная расписка?
#                ]
# костыльное получение начальной даты котировок
# with open('start_data_quotes.txt', 'w') as file:
#     for ticker in ticker_list:
#         with requests.Session() as session:
#             data = apimoex.get_board_history(session, ticker, columns=('TRADEDATE', 'CLOSE'))
#             start_date = data[0]['TRADEDATE']
#             size = len(data)
#             # print(ticker, '  ', data[0]['TRADEDATE'], data.shape[0])
#             file.write(f"{ticker}  {start_date}  {str(size)}\n")

#         with open('start_data_quotes.txt', 'a') as file:
#             for ticker in ticker_list:
#                 with requests.Session() as session:
#                     data = apimoex.get_board_history(session, ticker, columns=('TRADEDATE', 'CLOSE'))
#                     df = pd.DataFrame(data)
#                     start_date = df['TRADEDATE'][0]
#                     size = df.shape[0]
#                     # print(ticker, '  ', data[0]['TRADEDATE'], data.shape[0])
#                     file.write(f"{ticker}  {start_date}  {str(size)}\n")

# with requests.Session() as session:
# data = apimoex.get_board_history(session, 'MDMG', columns=('TRADEDATE', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME'))
# df = pd.DataFrame(data)
# df.set_index('TRADEDATE', inplace=True)
# # print(df.head(), '\n')
# # print(df.tail(), '\n')
# df.info()

# получение исторических данных котировок по указанному тикеру
# for ticker in ticker_list:
#     with requests.Session() as session:
#         data = apimoex.get_board_history(session, ticker, columns=('TRADEDATE', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME','LOTSIZE'))
#         df = pd.DataFrame(data)
#         df.set_index('TRADEDATE', inplace=True)
#         df.to_csv(f'.\\data\\{ticker}.csv')
#         print(ticker)
#         # print(df.head(), '\n')
#         # print(df.tail(), '\n')
#         # df.info()

# # Перечень акций, торгующихся в режиме TQBR
# request_url = ('https://iss.moex.com/iss/engines/stock/'
#                'markets/shares/boards/TQBR/securities.json')
# arguments = {'securities.columns': ('SECID,'
#                                     'LOTSIZE,')}
# with requests.Session() as session:
#     iss = apimoex.ISSClient(session, request_url, arguments)
#     data = iss.get()
#     df = pd.DataFrame(data['securities']) #ТУТ ЕСТЬ ЛОТ КАЖДОЙ АКЦИИ
#     df.set_index('SECID', inplace=True)
#     df.to_excel('all_shares_in_TQBR.xlsx', engine='openpyxl')
# df.to_csv('.\\all_shares_in_TQBR_all_columns.csv',sep=',', index=False)
# print(df.index) # тут список всех тикеров, которые торгуются на ФР
# print(df)
# print(df.head(), '\n')
# print(df.tail(), '\n')
# df.info()

# Такая инфа выходит после apimoex.ISSClient(requests.Session(), ('https://iss.moex.com/iss/engines/stock/markets/shares/boards/TQBR/securities.json'), {'securities.columns': ('SECID,''REGNUMBER,''LOTSIZE,''SHORTNAME')}).get()
# {'SECID': 'ZVEZ', 'BOARDID': 'TQBR', 'BID': None, 'BIDDEPTH': None, 'OFFER': None, 'OFFERDEPTH': None, 'SPREAD': 0, 'BIDDEPTHT': 0, 'OFFERDEPTHT': 0, 'OPEN': 12.7, 'LOW': 12.64, 'HIGH': 12.79, 'LAST': 12.67, 'LASTCHANGE': 0, 'LASTCHANGEPRCNT': 0, 'QTY': 2, 'VALUE': 25340.0, 'VALUE_USD': 276.36, 'WAPRICE': 12.69, 'LASTCNGTOLASTWAPRICE': 0, 'WAPTOPREVWAPRICEPRCNT': 0, 'WAPTOPREVWAPRICE': 0, 'CLOSEPRICE': None, 'MARKETPRICETODAY': 12.69, 'MARKETPRICE': 12.69, 'LASTTOPREVPRICE': -1.02, 'NUMTRADES': 51, 'VOLTODAY': 108000, 'VALTODAY': 1370230, 'VALTODAY_USD': 14944, 'ETFSETTLEPRICE': None, 'TRADINGSTATUS': 'N', 'UPDATETIME': '23:50:05', 'LASTBID': None, 'LASTOFFER': None, 'LCLOSEPRICE': 12.67, 'LCURRENTPRICE': 12.67, 'MARKETPRICE2': 12.69, 'NUMBIDS': None, 'NUMOFFERS': None, 'CHANGE': -0.13, 'TIME': '18:48:00', 'HIGHBID': None, 'LOWOFFER': None, 'PRICEMINUSPREVWAPRICE': -0.02, 'OPENPERIODPRICE': None, 'SEQNUM': 20240505000500, 'SYSTIME': '2024-05-05 00:05:00', 'CLOSINGAUCTIONPRICE': 12.67, 'CLOSINGAUCTIONVOLUME': 10000, 'ISSUECAPITALIZATION': 7120799481.6, 'ISSUECAPITALIZATION_UPDATETIME': '18:37:55', 'ETFSETTLECURRENCY': None, 'VALTODAY_RUR': 1370230, 'TRADINGSESSION': None, 'TRENDISSUECAPITALIZATION': -73062662.4}], 'dataversion': [{'data_version': 8095, 'seqnum': 20240503235958}], 'marketdata_yields': []}

# request_url = ('https://iss.moex.com/iss/engines/stock/'
#                'markets/shares/boards/TQBR/securities/PLZL/dates.json')
