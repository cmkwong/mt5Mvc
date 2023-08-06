from models.myUtils.paramModel import SymbolList, DatetimeTuple
from controllers.strategies.MovingAverage.Base import Base
from models.myUtils import timeModel, fileModel

import pandas as pd
import os


class Train(Base):
    def __init__(self, mainController):
        self.mt5Controller = mainController.mt5Controller
        self.nodeJsServerController = mainController.nodeJsApiController
        self.plotController = mainController.plotController
        self.mainPath = "./docs/ma"
        self.MA_SUMMARY_COLS = ['symbol', 'fast', 'slow', 'operation', 'total', 'count', 'start', 'end']

    def getMaDistImg(self, *,
                     symbols: SymbolList = 'USDJPY EURUSD',
                     timeframe: str = '15min',
                     start: DatetimeTuple = (2023, 6, 1, 0, 0),
                     end: DatetimeTuple = (2023, 6, 30, 23, 59),
                     fast_param: int = 14,
                     slow_param: int = 22):

        # create folder
        CUR_TIME = timeModel.getTimeS(outputFormat='%Y-%m-%d %H%M%S')
        distPath = fileModel.createDir(os.path.join(self.mainPath, 'dist'), CUR_TIME)

        # getting time string
        startStr = timeModel.getTimeS(start, '%Y%m%d%H%M')
        endStr = timeModel.getTimeS(end, '%Y%m%d%H%M')

        # getting the ma data
        Prices = self.mt5Controller.pricesLoader.getPrices(symbols=symbols, start=start, end=end, timeframe=timeframe)
        MaData = self.getMaData(Prices, fast_param, slow_param)

        Distributions = self.getMaDist(MaData)

        # output image
        for symbol, operations in Distributions.items():
            for operation, dists in operations.items():
                for distName, dist in dists.items():
                    self.plotController.plotHist(dist, distPath, f'{symbol}-{operation}-{startStr}-{endStr}-{distName}.jpg')

    def getMaSummaryDf(self, *,
                       symbols: SymbolList = 'USDJPY',
                       timeframe: str = '15min',
                       start: DatetimeTuple = (2023, 5, 1, 0, 0),
                       end: DatetimeTuple = (2023, 6, 30, 23, 59)):
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
                    ma_data = self.getMaData(Prices, f, s)
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
