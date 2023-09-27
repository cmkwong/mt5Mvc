from controllers.myNodeJs.HttpController import HttpController
from models.myUtils import timeModel
import config

import pandas as pd


class NodeJsApiController(HttpController):
    def __init__(self):
        self.mainUrl = None
        self.switchEnv('prod')

    # create the forex 1min table
    def createForex1MinTable(self, tableName, schemaType):
        created = self.restRequest(self.createTableUrl, {'tableName': tableName, 'schemaType': schemaType})
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
            if (self.postDataframe(self.uploadForexDataUrl, df, {'tableName': f'{symbol.lower()}_1m'})):
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
        body = {
            'from': timeModel.getTimeS(startTime, outputFormat='%Y-%m-%d %H:%M:%S'),
            'to': timeModel.getTimeS(endTime, outputFormat='%Y-%m-%d %H:%M:%S')
        }
        forexDataDf_raw = self.getDataframe(self.downloadForexDataUrl, {'tableName': tableName}, body)
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
            # elif col == 'base_exchg':
            #     forexDataDf[col] = forexDataDf_raw[col].resample(timeframe).last()
            # elif col == 'quote_exchg':
            #     forexDataDf[col] = forexDataDf_raw[col].resample(timeframe).last()
            # elif col == 'ptDv':
            #     forexDataDf[col] = forexDataDf_raw[col].resample(timeframe).sum()
        # drop the nan rows that is holiday
        return forexDataDf.rename(columns={"volume": "tick_volume"}).dropna()

    # upload all symbol info
    def uploadAllSymbolInfo(self, *, all_symbol_info: dict, broker: str):
        all_symbol_info_df = pd.DataFrame.from_dict(all_symbol_info).transpose()
        all_symbol_info_df['symbol'] = all_symbol_info_df.index
        all_symbol_info_df.reset_index(inplace=True, drop=True)  # drop the index
        all_symbol_info_df['broker'] = broker
        self.postDataframe(self.allSymbolInfoUrl, all_symbol_info_df)

    def get_all_symbols_info(self):
        df = self.getDataframe(self.allSymbolInfoUrl)
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

    # get the live strategy parameter
    def getStrategyParam(self, *, strategy_name: str = 'ma', live: int = 1, backtest: int = 1):
        """
        :param strategy_name: str
        :param live: int, index of live
        :return:
        """
        # argStrs = [f'name={strategyName}']
        param = {}
        param['strategy_name'] = strategy_name
        if live:
            # argStrs.append(f'live={live}')
            param['live'] = live
        if backtest:
            # argStrs.append(f'backtest={backtest}')
            param['backtest'] = backtest
        # url = self.strategyParamUrl.format("&".join(argStrs))
        df = self.getDataframe(self.strategyParamUrl, param)
        return df

    # get the backtest strategy parameter
    # def getBacktestStrategyParam(self, *, strategyName: str = 'ma', backtest: int = 1):
    #     url = self.strategyParamUrl.format(strategyName, backtest)
    #     df = self.getDataframe(url)
    #     return df
