from models.myUtils.paramModel import DatetimeTuple
from controllers.strategies.MovingAverage.Base import Base
from models.myUtils import timeModel, fileModel
import config

import pandas as pd
import numpy as np
import os


class Train(Base):
    def __init__(self, mainController):
        self.mt5Controller = mainController.mt5Controller
        # self.nodeJsServerController = mainController.nodeJsApiController
        self.MA_SUMMARY_COLS = ['symbol', 'fast', 'slow', 'operation', 'total', 'rate', 'count', 'mean_duration', 'timeframe', 'start', 'end',
                                '0%', '10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%', '100%',
                                'reliable'
                                ]

    def getMaSummaryDf(self, *,
                       symbols: list = config.DefaultSymbols,
                       timeframe: str = '15min',
                       start: DatetimeTuple = (2023, 6, 1, 0, 0),
                       end: DatetimeTuple = (2023, 7, 30, 23, 59),
                       subtest: bool = False):
        """
        - loop for each combination (fast vs slow)
        :subtest: if True, will split the whole period into subset for testing
        - return the excel: fast, slow, mode, count, meanDuration, pointEarn
        """
        # create folder
        CUR_TIME = timeModel.getTimeS(outputFormat='%Y-%m-%d %H%M%S')
        distPath = fileModel.createDir(os.path.join(self.MainPath, 'summary'), CUR_TIME)

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
                            # if not (symbol == 'CADJPY' and operation == 'short' and f == 1 and s == 247): continue
                            # calculate the earning
                            earningFactor = 1 if operation == 'long' else -1
                            total_value = (MaData[symbol, operation] * MaData[symbol, 'valueDiff']).sum() * earningFactor

                            # calculate the deal total
                            counts = (MaData[symbol, operation] > MaData[symbol, operation].shift(1)).sum()

                            # getting the masked MaData
                            mask = MaData[symbol, f'{operation}_group'] != False  # getting the rows only need to be groupby
                            masked_MaData = MaData[mask].copy()

                            # calculate the win rate
                            deal_value = masked_MaData.loc[:, [(symbol, 'valueDiff'), (symbol, f'{operation}_group')]].groupby((symbol, f'{operation}_group')).sum() * earningFactor
                            rate = deal_value[deal_value > 0].count()[0] / counts if counts > 0 else 0

                            # calculate the distribution
                            qvs_range = np.arange(0, 1.1, 0.1)
                            accum_value = masked_MaData.loc[:, [(symbol, 'valueDiff'), (symbol, f'{operation}_group')]].groupby((symbol, f'{operation}_group')).cumsum() * earningFactor
                            qvs = np.quantile(accum_value.values, qvs_range) if len(accum_value.values) > 0 else [0] * len(qvs_range)

                            # calculate the mean duration
                            mean_duration = masked_MaData.loc[:, [(symbol, 'valueDiff'), (symbol, f'{operation}_group')]].groupby((symbol, f'{operation}_group')).count().mean().values[0]

                            # append the result
                            MaSummaryDf.loc[len(MaSummaryDf)] = [symbol, f, s, operation, total_value, rate, counts, mean_duration, timeframe, periodStart, periodEnd, *qvs, 0]

                            # print the results
                            print(f"{symbol}: fast: {f}; slow: {s}; {operation} Earn: {total_value:.2f}[{counts}]; Period: {periodStartT} - {periodEndT}")

            print("Output the excel ... ")
            for symbol in symbols:
                # save the summary xlsx
                MaSummaryDf[MaSummaryDf['symbol'] == symbol].to_excel(os.path.join(distPath, f"{symbol}_summary.xlsx"), index=False)
        return True
