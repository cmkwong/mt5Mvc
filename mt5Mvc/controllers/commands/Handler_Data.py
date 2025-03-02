from mt5Mvc.controllers.myNodeJs.NodeJsApiController import NodeJsApiController
from mt5Mvc.controllers.myMT5.MT5Controller import MT5Controller
from mt5Mvc.controllers.myStock.StockPriceLoader import StockPriceLoader
from mt5Mvc.controllers.DfController import DfController

from mt5Mvc.models.myUtils import paramModel, fileModel, timeModel, inputModel
import config

import os
import pandas as pd

class Handler_Data:
    def __init__(self):
        self.nodeJsApiController = NodeJsApiController()
        self.mt5Controller = MT5Controller()
        self.stockPriceLoader = StockPriceLoader()
        self.dfController = DfController()

    def run(self, command):
        # upload the data into mySql server
        if command == '-upload':
            # setup the source into mt5
            originalSource = self.mt5Controller.pricesLoader.data_source
            self.mt5Controller.pricesLoader.data_source = 'mt5'
            # upload Prices
            paramFormat = {
                'symbols': [config.Default_Forex_Symbols, list, 'field'],
                'start': [(2023, 3, 24, 0, 0), tuple, 'field'],
                'end': [(2023, 4, 24, 23, 59), tuple, 'field'],
                'timeframe': ['1min', str, 'field'],
                'count': [0, int, 'field'],
                'ohlcvs': ['111111', str, 'field']
            }
            param = paramModel.ask_param(paramFormat)
            Prices = self.mt5Controller.pricesLoader.getPrices(**param)
            # upload Prices
            self.nodeJsApiController.uploadOneMinuteForexData(Prices)
            # resume to original source
            self.mt5Controller.pricesLoader.data_source = originalSource

        # download the stock data
        elif command == '-download_test':
            obj, param = paramModel.ask_param_fn(self.stockPriceLoader.getPrices)
            Prices = obj(**param)
            print()

        # all symbol info upload
        elif command == '-symbol':
            # upload all_symbol_info
            all_symbol_info = self.mt5Controller.symbolController.get_all_symbols_info()
            paramFormat = {'broker': config.Broker}
            obj, param = paramModel.ask_param_fn(self.nodeJsApiController.uploadAllSymbolInfo, **paramFormat)
            # append param
            param['all_symbol_info'] = all_symbol_info
            obj(**param)

        # read the stock price csv and upload into server
        elif command == '-upload_stock':
            params = paramModel.ask_param({
                'path': ['C:/Users/Chris/projects/221227_mt5Mvc/docs/datas/US Stock', str, 'field']
            })
            filename, df = self.dfController.read_as_df_with_selection(**params)
            # ask table name
            params = paramModel.ask_param({
                'tableName': [filename.lower().split('.', -1)[0], str, 'field']
            })
            # change the datetime
            df['datetime'] = pd.to_datetime(df['datetime'])

            # change the index into datetime
            if (self.nodeJsApiController.postDataframe(self.nodeJsApiController.uploadTableUrl, df, {'schemaName': 'stock', **params})):
                print(f"{params}: {len(df)} data being uploaded. ")

        # read the tick data csv and upload into server (because it is normally larger, it added chunksize)
        elif command == '-upload_tick':
            params = paramModel.ask_param({
                'path': [r'E:\forex_data\USDJPY', str, 'field'],
                'chunksize': [500000, any, 'field'], # because the CSV size is too large
                'colnames': [['datetime', 'bid', 'ask', 'volume', 'spread'], list, 'field'], # assign column names
                'header': [None, any, 'field'] # this CSV has no header
            })
            filename, df_iter = self.dfController.read_as_df_with_selection(**params)
            for data in df_iter:
                # set the datetime format and set it as index (just for resample)
                data['datetime'] = pd.to_datetime(data['datetime'], format='%d.%m.%Y %H:%M:%S.%f')
                data.set_index('datetime', inplace=True)
                resample_ohlc = data['bid'].resample('1Min').ohlc()
                resample_ohlc['volume'] = data['volume'].resample('1Min').sum().fillna(0)
                resample_ohlc['spread'] = data['spread'].resample('1Min').max().fillna(0)

                # reset the index
                resample_ohlc.reset_index(names=['datetime'], inplace=True)
                if (self.nodeJsApiController.postDataframe(self.nodeJsApiController.uploadTableUrl, resample_ohlc, {'schemaName': 'forex_v2', 'tableName': 'usdjpy_1m'})):
                    print(f"{len(resample_ohlc)} data being uploaded. ")

        # execute the query and get the dataframe
        elif command == '-sql':
            # get the query name
            fileList = fileModel.getFileList(config.SQLQUERY_FOREX_DIR)
            userInput = inputModel.askSelection(fileList)
            queryName = fileList[userInput].rsplit('.', 1)[0]
            df = self.nodeJsApiController.executeMySqlQuery(queryName)
            # out and open the excel
            fullPath = os.path.join('./docs/excel', f"{queryName}_{timeModel.getTimeS(outputFormat='%Y%m%d%H%M%S')}.xlsx")
            df.to_excel(fullPath)
            os.system(os.path.abspath(fullPath))

        else:
            return True