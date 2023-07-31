from models.myUtils.paramModel import SymbolList, DatetimeTuple
from models.myBacktest import techModel
from models.myUtils import dfModel, timeModel

import pandas as pd
import numpy as np
import os


class Train:
    def __init__(self, mainController):
        self.mt5Controller = mainController.mt5Controller
        self.nodeJsServerController = mainController.nodeJsApiController
        self.plotController = mainController.plotController
        self.mainPath = "./docs/ma"
        self.MA_DATA_COLS = ['close', 'cc', 'valueDiff', 'fast', 'slow', 'long', 'short']
        self.MA_SUMMARY_COLS = ['symbol', 'fast', 'slow', 'operation', 'total', 'count', 'start', 'end']

    def getMaDistribution(self, *,
                          symbols: SymbolList = 'USDJPY EURUSD',
                          timeframe: str = '15min', start: DatetimeTuple = (2023, 6, 1, 0, 0), end: DatetimeTuple = (2023, 6, 30, 23, 59),
                          fast_param: int = 21, slow_param: int = 22):

        # set the distribution path
        distPath = os.path.join(self.mainPath, 'dist')

        # getting time string
        startStr = timeModel.getTimeS(start, '%Y%m%d%H%M')
        endStr = timeModel.getTimeS(end, '%Y%m%d%H%M')

        # getting the ma data
        Prices = self.mt5Controller.pricesLoader.getPrices(symbols=symbols, start=start, end=end, timeframe=timeframe)
        ma_data = self.get_ma_data(Prices, fast_param, slow_param)

        Dist = {}
        for symbol in symbols:
            Dist[symbol] = {}
            for operation in ['long', 'short']:
                # create a column for storage required grouping information
                ma_data[symbol, f'{operation}_group'] = ma_data.loc[:, (symbol, operation)]

                # assign the time-index into first timeframe
                mask = (ma_data[symbol, operation] > ma_data[symbol, operation].shift(1)) == True
                ma_data.loc[mask, (symbol, f'{operation}_group')] = ma_data[mask].index

                # assign nan value for ffill
                mask = ma_data[symbol, f'{operation}_group'] == True
                ma_data.loc[mask, (symbol, f'{operation}_group')] = np.nan
                ma_data[symbol, f'{operation}_group'].fillna(method='ffill', inplace=True)

                # getting distribution
                Dist = {}
                mask = ma_data[symbol, f'{operation}_group'] != False  # getting the rows only need to be groupby
                maskedDf = ma_data[mask]
                # values in deposit
                Dist['valueDist'] = maskedDf.loc[:, (symbol, 'valueDiff')]
                # % changes
                Dist['changeDist'] = maskedDf.loc[:, (symbol, 'cc')]
                maskedDf.loc[:, (symbol, 'return')] = maskedDf.loc[:, (symbol, 'cc')] + 1
                # group by deal (change, value and duration)
                Dist['deal_valueDist'] = maskedDf.loc[:, [(symbol, 'valueDiff'), (symbol, f'{operation}_group')]].groupby((symbol, f'{operation}_group')).sum()
                Dist['deal_changeDist'] = (maskedDf.loc[:, [(symbol, 'return'), (symbol, f'{operation}_group')]]).groupby((symbol, f'{operation}_group')).prod() - 1
                Dist['deal_durationDist'] = maskedDf.loc[:, [(symbol, 'valueDiff'), (symbol, f'{operation}_group')]].groupby((symbol, f'{operation}_group')).count()

                # # saved in dictionary
                # Dist[symbol]['valueDist'], Dist[symbol]['dealDist'], Dist[symbol]['durationDist'] = valueDist, dealDist, durationDist

                for name, dist in Dist.items():
                    self.plotController.plotHist(dist, distPath, f'{symbol}-{operation}-{startStr}-{endStr}-{name}.jpg')
        print()

    def get_ma_data(self, Prices, fast_param, slow_param):
        """
        :param Prices: Prices object
        :param fast_param: int
        :param slow_param: int
        :return: pd.Dataframe
        """
        columnIndex = dfModel.getLevelColumnIndex(level_1=list(Prices.close.columns), level_2=self.MA_DATA_COLS)
        # get the close price
        Close = Prices.close
        # get the point value (Based on Deposit Dollar)
        valueDiff = Prices.ptDv.values * Prices.quote_exchg.values
        # get the changes
        Change = Prices.cc
        # create empty ma_data
        ma_data = pd.DataFrame(columns=columnIndex, index=Close.index)
        for i, symbol in enumerate(Prices.symbols):
            ma_data[symbol, 'close'] = Close[symbol]
            ma_data[symbol, 'cc'] = Change[symbol]
            ma_data[symbol, 'valueDiff'] = valueDiff[:, i]
            ma_data[symbol, 'fast'] = techModel.get_MA(Close[symbol], fast_param, False)
            ma_data[symbol, 'slow'] = techModel.get_MA(Close[symbol], slow_param, False)
            # long signal
            ma_data[symbol, 'long'] = ma_data[symbol, 'fast'] > ma_data[symbol, 'slow']
            ma_data[symbol, 'long'] = ma_data[symbol, 'long'].shift(1).fillna(False)  # signal should delay 1 timeframe
            # short signal
            ma_data[symbol, 'short'] = ma_data[symbol, 'fast'] < ma_data[symbol, 'slow']
            ma_data[symbol, 'short'] = ma_data[symbol, 'short'].shift(1).fillna(False)  # signal should delay 1 timeframe
        return ma_data

    def getMaSummaryDf(self, *,
                       symbols: SymbolList = 'USDJPY EURUSD',
                       timeframe: str = '15min', start: DatetimeTuple = (2023, 6, 1, 0, 0), end: DatetimeTuple = (2023, 6, 30, 23, 59)):
        """
        - loop for each combination (fast vs slow)
        - return the excel: fast, slow, mode, count, meanDuration, pointEarn
        """
        periods = timeModel.splitTimePeriod(start, end, {'days': 7}, True)
        MaSummaryDf = pd.DataFrame(columns=self.MA_SUMMARY_COLS)
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
        return MaSummaryDf
