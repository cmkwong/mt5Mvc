from models.myUtils.paramModel import SymbolList, DatetimeTuple
from models.myBacktest import techModel
from models.myUtils import dfModel

import pandas as pd


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
        columnIndex = dfModel.getLevelColumnIndex(level_1=list(Prices.close.columns), level_2=['close', 'ptDv', 'fast', 'slow', 'long', 'short'])
        closePrices = Prices.close
        ptDvs = Prices.ptDv
        ma_data = pd.DataFrame(columns=columnIndex, index=closePrices.index)
        for symbol, closeSeries in Prices.close.items():
            ma_data[symbol, 'close'] = closeSeries
            ma_data[symbol, 'ptDv'] = ptDvs[symbol]
            ma_data[symbol, 'fast'] = techModel.get_MA(closeSeries, fast_param, False)
            ma_data[symbol, 'slow'] = techModel.get_MA(closeSeries, slow_param, False)
            # long signal
            ma_data[symbol, 'long'] = ma_data[symbol, 'fast'] > ma_data[symbol, 'slow']
            ma_data[symbol, 'long'] = ma_data[symbol, 'long'].shift(1).fillna(False)
            # short signal
            ma_data[symbol, 'short'] = ma_data[symbol, 'fast'] < ma_data[symbol, 'slow']
            ma_data[symbol, 'short'] = ma_data[symbol, 'short'].shift(1).fillna(False)
        return ma_data

    def run(self, *, symbols: SymbolList = 'USDJPY', start: DatetimeTuple = (2023, 6, 1, 0, 0), end: DatetimeTuple = (2023, 6, 30, 23, 59)):
        """
        - loop for each combination (fast vs slow)
        - return the excel: fast, slow, mode, count, meanDuration, pointEarn
        """
        slowMax = 100
        fastMax = 99
        Prices = self.mt5Controller.pricesLoader.getPrices(symbols=symbols, start=start, end=end)
        for f in range(1, fastMax):
            for s in range(f + 1, slowMax):
                ma_data = self.get_ma_data(Prices, f, s)
                print()
