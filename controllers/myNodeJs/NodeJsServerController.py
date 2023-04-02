from controllers.myNodeJs.ApiController import ApiController
from models.myUtils import timeModel

import pandas as pd

class NodeJsServerController(ApiController):
    def __init__(self):
        self.mainUrl = None
        self.switchEnv()

    def switchEnv(self):
        if self.mainUrl == "http://192.168.1.165:3002/":
            self.mainUrl = "http://localhost:3002/"
        else:
            self.mainUrl = "http://192.168.1.165:3002/"
        print(f"Connecting to {self.mainUrl} ... ")
        # define the url
        self.uploadForexDataUrl = self.mainUrl + "api/v1/query/forexTable/upload?tableName={}"
        self.downloadForexDataUrl = self.mainUrl + "api/v1/query/forexTable/download?tableName={}"
        self.createTableUrl = self.mainUrl + "api/v1/query/forexTable/create?tableName={}"
        self.uploadAllSymbolInfoUrl = self.mainUrl + "api/v1/query/forexTable/symbolInfo"

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
            df['datetime'] = df['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')
            self.postDataframe(self.uploadForexDataUrl.format(symbol.lower() + '_1m'), df)

    # get data
    def downloadForexData(self, symbol: str, startTime: tuple, endTime: tuple, timeframe: str):
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
        self.postDataframe(self.uploadAllSymbolInfoUrl, all_symbol_info_df)



