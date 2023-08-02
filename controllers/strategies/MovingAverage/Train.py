from models.myUtils.paramModel import SymbolList, DatetimeTuple
from controllers.strategies.MovingAverage.Base import Base
from models.myUtils import timeModel

import pandas as pd
import numpy as np
import os


class Train(Base):
    def __init__(self, mainController):
        self.mt5Controller = mainController.mt5Controller
        self.nodeJsServerController = mainController.nodeJsApiController
        self.plotController = mainController.plotController
        self.mainPath = "./docs/ma"
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

        for symbol in symbols:
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

                # output image
                for name, dist in Dist.items():
                    self.plotController.plotHist(dist, distPath, f'{symbol}-{operation}-{startStr}-{endStr}-{name}.jpg')

    def getMaSummaryDf(self, *,
                       symbols: SymbolList = 'USDJPY EURUSD',
                       timeframe: str = '15min', start: DatetimeTuple = (2023, 5, 1, 0, 0), end: DatetimeTuple = (2023, 6, 30, 23, 59)):
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
