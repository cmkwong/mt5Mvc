from models.myUtils.paramModel import SymbolList, DatetimeTuple
from models.myBacktest import techModel
from models.myUtils import dfModel, timeModel

import pandas as pd
import os

class Train:
    def __init__(self, mainController):
        self.mt5Controller = mainController.mt5Controller
        self.nodeJsServerController = mainController.nodeJsApiController
        self.plotController = mainController.plotController

    def get_ma_data(self, Prices, fast_param, slow_param):
        """
        :param Prices: Prices object
        :param fast_param: int
        :param slow_param: int
        :return: pd.Dataframe
        """
        columnIndex = dfModel.getLevelColumnIndex(level_1=list(Prices.close.columns), level_2=['close', 'valueDiff', 'fast', 'slow', 'long', 'short'])
        closePrices = Prices.close
        valueDiff = Prices.ptDv.values * Prices.quote_exchg.values
        ma_data = pd.DataFrame(columns=columnIndex, index=closePrices.index)
        for i, (symbol, closeSeries) in enumerate(Prices.close.items()):
            ma_data[symbol, 'close'] = closeSeries
            ma_data[symbol, 'valueDiff'] = valueDiff[:, i]
            ma_data[symbol, 'fast'] = techModel.get_MA(closeSeries, fast_param, False)
            ma_data[symbol, 'slow'] = techModel.get_MA(closeSeries, slow_param, False)
            # long signal
            ma_data[symbol, 'long'] = ma_data[symbol, 'fast'] > ma_data[symbol, 'slow']
            ma_data[symbol, 'long'] = ma_data[symbol, 'long'].shift(1).fillna(False)    # signal should delay 1 timeframe
            # short signal
            ma_data[symbol, 'short'] = ma_data[symbol, 'fast'] < ma_data[symbol, 'slow']
            ma_data[symbol, 'short'] = ma_data[symbol, 'short'].shift(1).fillna(False)  # signal should delay 1 timeframe
        return ma_data

    def run(self, *, symbols: SymbolList = 'USDJPY', timeframe: str = '15min', start: DatetimeTuple = (2023, 6, 1, 0, 0), end: DatetimeTuple = (2023, 6, 11, 23, 59)):
        """
        - loop for each combination (fast vs slow)
        - return the excel: fast, slow, mode, count, meanDuration, pointEarn
        """
        summaryDf = pd.DataFrame(columns=['symbol', 'fast', 'slow', 'longEarn', 'shortEarn'])
        maIndex = 250
        Prices = self.mt5Controller.pricesLoader.getPrices(symbols=symbols, start=start, end=end, timeframe=timeframe)
        for f in range(1, maIndex - 1):
            for s in range(f + 1, maIndex):
                ma_data = self.get_ma_data(Prices, f, s)
                for symbol in symbols:
                    longEarn = (ma_data[symbol, 'long'] * ma_data[symbol, 'valueDiff']).sum()
                    shortEarn = (ma_data[symbol, 'short'] * ma_data[symbol, 'valueDiff'] * -1).sum()
                    summaryDf.loc[len(summaryDf)] = [symbol, f, s, longEarn, shortEarn]
                    print(f"{symbol}: fast: {f}; slow: {s}; Long Earn: {longEarn:.2f}; Short Earn: {shortEarn:.2f}")
        timeStr = timeModel.getTimeS(outputFormat="%Y%m%d%H%M%S")
        summaryDf.to_excel(os.path.join("./docs/ma", f"{timeStr}_summary.xlsx"), index=False)
