from models.myUtils.paramModel import SymbolList, DatetimeTuple
from controllers.strategies.MovingAverage.Base import Base
from models.myUtils import timeModel, fileModel
import config

import pandas as pd
import numpy as np
import os


class Train(Base):
    def __init__(self, mainController):
        self.mt5Controller = mainController.mt5Controller
        self.nodeJsServerController = mainController.nodeJsApiController
        self.plotController = mainController.plotController
        self.mainPath = "./docs/ma"
        self.MA_SUMMARY_COLS = ['symbol', 'fast', 'slow', 'operation', 'total', 'rate', 'count', '25%', '50%', '75%', 'timeframe', 'start', 'end', 'reliable']

    def getMaDistImg(self, *,
                     symbols: SymbolList = 'USDJPY EURUSD',
                     timeframe: str = '15min',
                     start: DatetimeTuple = (2023, 5, 1, 0, 0),
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
                       symbols: list = config.DefaultSymbols,
                       timeframe: str = '15min',
                       start: DatetimeTuple = (2023, 5, 1, 0, 0),
                       end: DatetimeTuple = (2023, 6, 30, 23, 59),
                       subtest: bool = False):
        """
        - loop for each combination (fast vs slow)
        :subtest: if True, will split the whole period into subset for testing
        - return the excel: fast, slow, mode, count, meanDuration, pointEarn
        """
        # create folder
        CUR_TIME = timeModel.getTimeS(outputFormat='%Y-%m-%d %H%M%S')
        distPath = fileModel.createDir(os.path.join(self.mainPath, 'summary'), CUR_TIME)

        # get the required periods
        periods = timeModel.splitTimePeriod(start, end, {'days': 7}, True)
        # only need the whole period
        if not subtest:
            periods = [periods[-1]]
        maIndex = 250
        for (periodStart, periodEnd) in periods:
            periodStartT = timeModel.getTimeT(periodStart, "%Y-%m-%d %H:%M")
            periodEndT = timeModel.getTimeT(periodEnd, "%Y-%m-%d %H:%M")
            Prices = self.mt5Controller.pricesLoader.getPrices(symbols=symbols, start=periodStartT, end=periodEndT, timeframe=timeframe)

            # define the dataframe
            MaSummaryDf = pd.DataFrame(columns=self.MA_SUMMARY_COLS)
            for f in range(1, maIndex - 1):
                for s in range(f + 1, maIndex):
                    # MaData including long and short operation data
                    MaData = self.getMaData(Prices, f, s)
                    # calculate for each symbol
                    for symbol in symbols:
                        for operation in self.OPERATIONS:
                            # calculate the earning
                            earningFactor = 1 if operation == 'long' else -1
                            total_value = (MaData[symbol, operation] * MaData[symbol, 'valueDiff'] * earningFactor).sum()

                            # calculate the deal total
                            counts = (MaData[symbol, operation] > MaData[symbol, operation].shift(1)).sum()

                            # calculate the win rate
                            mask = MaData[symbol, f'{operation}_group'] != False  # getting the rows only need to be groupby
                            masked_MaData = MaData[mask].copy()
                            deal_value = masked_MaData.loc[:, [(symbol, 'valueDiff'), (symbol, f'{operation}_group')]].groupby((symbol, f'{operation}_group')).sum()
                            rate = deal_value[deal_value > 0].count()[0] / counts if counts > 0 else 0

                            # calculate the distribution
                            accum_value = masked_MaData.loc[:, [(symbol, 'valueDiff'), (symbol, f'{operation}_group')]].groupby((symbol, f'{operation}_group')).cumsum()
                            qvs = np.quantile(accum_value.values, (0.25, 0.5, 0.75)) if len(accum_value.values) > 0 else [0, 0, 0]

                            # append the result
                            MaSummaryDf.loc[len(MaSummaryDf)] = [symbol, f, s, operation, total_value, rate, counts, *qvs, timeframe, periodStart, periodEnd, 0]

                            # print the results
                            print(f"{symbol}: fast: {f}; slow: {s}; {operation} Earn: {total_value:.2f}[{counts}]; Period: {periodStartT} - {periodEndT}")

            print("Output the excel ... ")
            for symbol in symbols:
                # save the summary xlsx
                MaSummaryDf[MaSummaryDf['symbol'] == symbol].to_excel(os.path.join(distPath, f"{symbol}_summary.xlsx"), index=False)
        return True
