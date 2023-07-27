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
        self.mainPath = "./docs/ma"
        self.maSummaryDf_cols = ['symbol', 'fast', 'slow', 'operation', 'total', 'count', 'start', 'end']

    def get_ma_data(self, Prices, fast_param, slow_param):
        """
        :param Prices: Prices object
        :param fast_param: int
        :param slow_param: int
        :return: pd.Dataframe
        """
        columnIndex = dfModel.getLevelColumnIndex(level_1=list(Prices.close.columns), level_2=['close', 'valueDiff', 'fast', 'slow', 'long', 'short'])
        # get the close price
        closePrices = Prices.close
        # get the point value (Based on Deposit Dollar)
        valueDiff = Prices.ptDv.values * Prices.quote_exchg.values
        ma_data = pd.DataFrame(columns=columnIndex, index=closePrices.index)
        for i, (symbol, closeSeries) in enumerate(Prices.close.items()):
            ma_data[symbol, 'close'] = closeSeries
            ma_data[symbol, 'valueDiff'] = valueDiff[:, i]
            ma_data[symbol, 'fast'] = techModel.get_MA(closeSeries, fast_param, False)
            ma_data[symbol, 'slow'] = techModel.get_MA(closeSeries, slow_param, False)
            # long signal
            ma_data[symbol, 'long'] = ma_data[symbol, 'fast'] > ma_data[symbol, 'slow']
            ma_data[symbol, 'long'] = ma_data[symbol, 'long'].shift(1).fillna(False)  # signal should delay 1 timeframe
            # short signal
            ma_data[symbol, 'short'] = ma_data[symbol, 'fast'] < ma_data[symbol, 'slow']
            ma_data[symbol, 'short'] = ma_data[symbol, 'short'].shift(1).fillna(False)  # signal should delay 1 timeframe
        return ma_data

    def getMaSummaryDf(self, *, symbols: SymbolList = 'USDJPY EURUSD', timeframe: str = '15min', start: DatetimeTuple = (2023, 6, 1, 0, 0), end: DatetimeTuple = (2023, 6, 30, 23, 59)):
        """
        - loop for each combination (fast vs slow)
        - return the excel: fast, slow, mode, count, meanDuration, pointEarn
        """
        periods = timeModel.splitTimePeriod(start, end, {'days': 7}, True)
        MaSummaryDf = pd.DataFrame(columns=self.maSummaryDf_cols)
        maIndex = 250
        for (periodStart, periodEnd) in periods:
            periodStartT = timeModel.getTimeT(periodStart, "%Y-%m-%d %H:%M")
            periodEndT = timeModel.getTimeT(periodEnd, "%Y-%m-%d %H:%M")
            Prices = self.mt5Controller.pricesLoader.getPrices(symbols=symbols, start=periodStartT, end=periodEndT, timeframe=timeframe)
            for f in range(1, maIndex - 1):
                for s in range(f + 1, maIndex):
                    ma_data = self.get_ma_data(Prices, f, s)
                    # calculate for each symbol
                    for symbol in symbols:
                        # calculate the earning
                        total_long = (ma_data[symbol, 'long'] * ma_data[symbol, 'valueDiff']).sum()
                        total_short = (ma_data[symbol, 'short'] * ma_data[symbol, 'valueDiff'] * -1).sum()
                        # calculate the deal total
                        count_long = (ma_data[symbol, 'long'] > ma_data[symbol, 'long'].shift(1)).sum()
                        count_short = (ma_data[symbol, 'short'] > ma_data[symbol, 'short'].shift(1)).sum()
                        # append the result
                        MaSummaryDf.loc[len(MaSummaryDf)] = [symbol, f, s, 'long', total_long, count_long, periodStart, periodEnd]
                        MaSummaryDf.loc[len(MaSummaryDf)] = [symbol, f, s, 'short', total_short, count_short, periodStart, periodEnd]
                        # print the results
                        print(f"{symbol}: fast: {f}; slow: {s}; Long Earn: {total_long:.2f}[{count_long}]; Short Earn: {total_short:.2f}[{count_short}]; Period: {periodStartT} - {periodEndT}")
        # getting the current time string
        timeStr = timeModel.getTimeS(outputFormat="%Y%m%d%H%M%S")
        MaSummaryDf.to_excel(os.path.join(self.mainPath, f"{timeStr}_summary.xlsx"), index=False)
