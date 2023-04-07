from controllers.myNodeJs.HttpController import HttpController
from models.myUtils import timeModel
import config

import pandas as pd

class NodeJsApiController(HttpController):
    def __init__(self):
        self.mainUrl = None
        self.switchEnv()

    def switchEnv(self):
        if self.mainUrl == "http://localhost:3002/":
            self.mainUrl = "http://192.168.1.165:3002/"
        else:
            self.mainUrl = "http://localhost:3002/"
        print(f"Connecting to {self.mainUrl} ... ")
        # define the url
        self.uploadForexDataUrl = self.mainUrl + "api/v1/query/forexTable/upload?tableName={}"
        self.downloadForexDataUrl = self.mainUrl + "api/v1/query/forexTable/download?tableName={}"
        self.createTableUrl = self.mainUrl + "api/v1/query/forexTable/create?tableName={}"
        self.allSymbolInfoUrl = self.mainUrl + "api/v1/query/forexTable/symbolInfo"

    def createForex1MinTable(self, tableName):
        schemaObj = {
            "columns": {
                "datetime": ["DATETIME", "NOT NULL"],
                "open": ["FLOAT"],
                "high": ["FLOAT"],
                "low": ["FLOAT"],
                "close": ["FLOAT"],
                "volume": ["FLOAT"],
                "spread": ["FLOAT"],
                "base_exchg": ["FLOAT"],
                "quote_exchg": ["FLOAT"]
            },
            "keys": ["datetime"]

        }
        created = self.createTable(self.createTableUrl.format(tableName), schemaObj)
        if created:
            print(f"The table is created.")

    # upload data
    def uploadOneMinuteForexData(self, Prices):
        """
        :param symbols: list for the symbols
        :param startTime: tuple for time
        :param endTime: tuple for time
        :return:
        """
        # process into dataframe
        dfs = Prices.getOhlcvsFromPrices()
        # get to upload the data to database
        for symbol, df in dfs.items():
            df['datetime'] = df.index
            # change to UTC + 0
            df['datetime'] = (df['datetime'] + pd.Timedelta(hours=-config.Broker_Time_Between_UTC)).dt.strftime('%Y-%m-%d %H:%M:%S')
            if (self.postDataframe(self.uploadForexDataUrl.format(symbol.lower() + '_1m'), df)):
                print(f"{symbol}: {len(dfs[symbol])} data being uploaded. ")

    # get data
    def downloadForexData(self, symbol: str, timeframe: str, startTime: tuple, endTime: tuple):
        """
        :param symbol: str
        :param startTime: tuple for time
        :param endTime: tuple for time
        :param timeframe: str
        :return:
        """
        forexDataDf = pd.DataFrame()
        # define the table name
        tableName = symbol.lower() + '_1m'
        url = self.downloadForexDataUrl.format(tableName)
        body = {
            'from': timeModel.getTimeS(startTime, outputFormat='%Y-%m-%d %H:%M:%S'),
            'to': timeModel.getTimeS(endTime, outputFormat='%Y-%m-%d %H:%M:%S')
        }
        forexDataDf_raw = self.getDataframe(url, body)
        if (len(forexDataDf_raw) == 0):
            print('No data being fetched. ')
            return False
        # change to dataframe
        forexDataDf_raw = forexDataDf_raw.set_index('datetime')
        # change index into datetimeIndex
        forexDataDf_raw.index = pd.to_datetime(forexDataDf_raw.index)

        # resample the dataframe
        for col in forexDataDf_raw:
            if col == 'open':
                forexDataDf[col] = forexDataDf_raw[col].resample(timeframe).first()
            elif col == 'high':
                forexDataDf[col] = forexDataDf_raw[col].resample(timeframe).max()
            elif col == 'low':
                forexDataDf[col] = forexDataDf_raw[col].resample(timeframe).min()
            elif col == 'close':
                forexDataDf[col] = forexDataDf_raw[col].resample(timeframe).last()
            elif col == 'volume':
                forexDataDf[col] = forexDataDf_raw[col].resample(timeframe).sum()
            elif col == 'spread':
                forexDataDf[col] = forexDataDf_raw[col].resample(timeframe).last()
            elif col == 'base_exchg':
                forexDataDf[col] = forexDataDf_raw[col].resample(timeframe).last()
            elif col == 'quote_exchg':
                forexDataDf[col] = forexDataDf_raw[col].resample(timeframe).last()
            elif col == 'ptDv':
                forexDataDf[col] = forexDataDf_raw[col].resample(timeframe).sum()
        # drop the nan rows that is holiday
        return forexDataDf.dropna()

    # upload all symbol info
    def uploadAllSymbolInfo(self, *, all_symbol_info: dict, broker: str):
        all_symbol_info_df = pd.DataFrame.from_dict(all_symbol_info).transpose()
        all_symbol_info_df['symbol'] = all_symbol_info_df.index
        all_symbol_info_df.reset_index(inplace=True, drop=True) # drop the index
        all_symbol_info_df['broker'] = broker
        self.postDataframe(self.allSymbolInfoUrl, all_symbol_info_df)

    def get_all_symbols_info(self):
        df = self.getDataframe(self.allSymbolInfoUrl, {})
        rawDict = df.transpose().to_dict()
        all_symbols_info = {}
        for info in rawDict.values():
            symbol_name = info['symbol']
            all_symbols_info[symbol_name] = {}
            all_symbols_info[symbol_name]['digits'] = info["digits"]
            all_symbols_info[symbol_name]['base'] = info["base"]
            all_symbols_info[symbol_name]['quote'] = info["quote"]
            all_symbols_info[symbol_name]['swap_long'] = info["swap_long"]
            all_symbols_info[symbol_name]['swap_short'] = info["swap_short"]
            if symbol_name[3:] == 'JPY':
                all_symbols_info[symbol_name]['pt_value'] = 100  # 100 dollar for quote per each point    (See note Stock Market - Knowledge - note 3)
            else:
                all_symbols_info[symbol_name]['pt_value'] = 1  # 1 dollar for quote per each point  (See note Stock Market - Knowledge - note 3)
        print(f"Local Symbol Info is fetched from database. {len(all_symbols_info)}")
        return all_symbols_info

