from models.myUtils import paramModel, fileModel, timeModel, inputModel
import config

import os
import pandas as pd

class Handler_Data:
    def __init__(self, nodeJsApiController, mt5Controller, stockPriceLoader, dfController):
        self.nodeJsApiController = nodeJsApiController
        self.mt5Controller = mt5Controller
        self.stockPriceLoader = stockPriceLoader
        self.dfController = dfController

    def run(self, command):
        # upload the data into mySql server
        if command == '-upload':
            # setup the source into mt5
            originalSource = self.mt5Controller.pricesLoader.source
            self.mt5Controller.pricesLoader.source = 'mt5'
            # upload Prices
            paramFormat = {
                'symbols': [config.Default_Forex_Symbols, list],
                'start': [(2023, 3, 24, 0, 0), tuple],
                'end': [(2023, 4, 24, 23, 59), tuple],
                'timeframe': ['1min', str],
                'count': [0, int],
                'ohlcvs': ['111111', str]
            }
            param = paramModel.ask_param(paramFormat)
            Prices = self.mt5Controller.pricesLoader.getPrices(**param)
            # upload Prices
            self.nodeJsApiController.uploadOneMinuteForexData(Prices)
            # resume to original source
            self.mt5Controller.pricesLoader.source = originalSource

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

        # read the price csv and upload into server
        elif command == '-upload_stock':
            params = paramModel.ask_param({
                'path': ['C:/Users/Chris/projects/221227_mt5Mvc/docs/datas/US Stock', str]
            })
            filename, df = self.dfController.readAsDf(**params)
            # ask table name
            params = paramModel.ask_param({
                'tableName': [filename.lower().split('.', -1)[0], str]
            })
            # change the datetime
            df['datetime'] = pd.to_datetime(df['datetime'])

            # change the index into datetime
            if (self.nodeJsApiController.postDataframe(self.nodeJsApiController.uploadTableUrl, df, {'schemaName': 'stock', **params})):
                print(f"{params}: {len(df)} data being uploaded. ")

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