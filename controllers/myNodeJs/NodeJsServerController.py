from controllers.myNodeJs.ApiController import ApiController
from models.myUtils import timeModel

import pandas as pd


class NodeJsServerController(ApiController):
    def __init__(self, mt5Controller):
        super(NodeJsServerController, self).__init__()
        self.mt5Controller = mt5Controller

        # define the url
        self.mainUrl = "http://192.168.1.165:3002/"
        self.uploadForexDataUrl = self.mainUrl + "api/v1/query/forexTable/upload?tableName={}"
        self.downloadForexDataUrl = self.mainUrl + "api/v1/query/forexTable/download?tableName={}"
        self.createTableUrl = self.mainUrl + "api/v1/query/forexTable/create?tableName={}"
        self.uploadSymbolInfoUrl = self.mainUrl + "api/v1/query/forexTable/symbolInfo"

    def createForexTable(self, tableName):
        schemaObj = {
            "datetime": "DATETIME NOT NULL PRIMARY KEY",
            "open": "FLOAT",
            "high": "FLOAT",
            "low": "FLOAT",
            "close": "FLOAT",
            "volume": "FLOAT",
            "spread": "FLOAT",
            "base_exchg": "FLOAT",
            "quote_exchg": "FLOAT"
        }
        created = self.createTable(self.createTableUrl.format(tableName), schemaObj)
        if created:
            print(f"The table is created.")

    # upload data
    def uploadSymbolData(self, Prices):
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
    def downloadSymbolData(self, symbol: str, startTime: tuple, endTime: tuple, timeframe: str):
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

    def uploadSymbolInfo(self, broker: str, all_symbol_info: dict):
        pass

# mt5Controller = MT5Controller()
# dataFeeder = DataFeeder(mt5Controller)
#
# dataFeeder.downloadData('AUDJPY', (2022, 8, 31, 0, 0), (2022, 10, 27, 0, 0))
