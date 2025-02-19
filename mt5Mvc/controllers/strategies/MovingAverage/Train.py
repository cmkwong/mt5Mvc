from mt5Mvc.models.myUtils.paramModel import DatetimeTuple
from mt5Mvc.controllers.strategies.MovingAverage.Base import Base
from mt5Mvc.controllers.myMT5.MT5Controller import MT5Controller

from mt5Mvc.models.myUtils import timeModel, fileModel
import config

import pandas as pd
import numpy as np
import os

class Train(Base):
    def __init__(self):
        self.mt5Controller = MT5Controller()
        # self.nodeJsServerController = mainController.nodeJsApiController
        self.MA_SUMMARY_COLS = ['symbol', 'fast', 'slow', 'operation', 'total', 'ret', 'rate', 'count', 'mean_duration', 'timeframe', 'start', 'end',
                                '0%r', '10%r', '20%r', '30%r', '40%r', '50%r', '60%r', '70%r', '80%r', '90%r', '100%r',
                                '0%v', '10%v', '20%v', '30%v', '40%v', '50%v', '60%v', '70%v', '80%v', '90%v', '100%v',
                                'baseRet', 'reliable'
                                ]

    # def genData(self):
    #     Prices = self.mt5Controller.pricesLoader.getPrices()

    def analysis(self, Prices, maIndex):

        periodStart = timeModel.getTimeS(Prices.start_index, '%Y-%m-%d %H:%M')
        periodEnd = timeModel.getTimeS(Prices.end_index, '%Y-%m-%d %H:%M')

        # calculate the base return
        baseRets = np.exp((Prices.logRet).sum(axis=0))

        # define the dataframe
        MaSummaryDf = pd.DataFrame(columns=self.MA_SUMMARY_COLS)
        for f in range(1, maIndex - 1):
            for s in range(f + 1, maIndex):
                # MaData including long and short operation data
                MaData = self.getMaData(Prices, f, s)
                # calculate for each symbol
                for symbol in Prices.symbols:
                    for operation in self.OPERATIONS:
                        # if not (symbol == 'CADJPY' and operation == 'short' and f == 1 and s == 247): continue
                        # calculate the earning
                        earningFactor = 1 if operation == 'long' else -1
                        total_value = (MaData[symbol, operation] * MaData[symbol, 'valueDiff']).sum(axis=0) * earningFactor

                        # calculate the return
                        ret_percent = np.exp((MaData[symbol, operation] * MaData[symbol, 'logRet']).sum(axis=0) * earningFactor) - 1

                        # calculate the deal total
                        counts = (MaData[symbol, operation] > MaData[symbol, operation].shift(1)).sum(axis=0)

                        # getting the masked MaData
                        mask = MaData[symbol, f'{operation}_group'] != False  # getting the rows only need to be groupby
                        masked_MaData = MaData[mask].copy()

                        # calculate the win rate
                        deal_value = masked_MaData.loc[:, [(symbol, 'valueDiff'), (symbol, f'{operation}_group')]].groupby((symbol, f'{operation}_group')).sum() * earningFactor
                        rate = deal_value.where(deal_value > 0).count().iloc[0] / counts if counts > 0 else 0

                        # calculate the distribution (log return)
                        qvs_range = np.arange(0, 1.1, 0.1)
                        accum_logRet = np.exp(masked_MaData.loc[:, [(symbol, 'logRet'), (symbol, f'{operation}_group')]].groupby((symbol, f'{operation}_group')).cumsum() * earningFactor) - 1
                        qrs = np.quantile(accum_logRet.values, qvs_range) if len(accum_logRet.values) > 0 else [0] * len(qvs_range)

                        # calculate the distribution (value difference)
                        accum_value = masked_MaData.loc[:, [(symbol, 'valueDiff'), (symbol, f'{operation}_group')]].groupby((symbol, f'{operation}_group')).cumsum() * earningFactor
                        qvs = np.quantile(accum_value.values, qvs_range) if len(accum_value.values) > 0 else [0] * len(qvs_range)

                        # calculate the mean duration
                        mean_duration = masked_MaData.loc[:, [(symbol, 'valueDiff'), (symbol, f'{operation}_group')]].groupby((symbol, f'{operation}_group')).count().mean().values[0]

                        # append the result
                        MaSummaryDf.loc[len(MaSummaryDf)] = [symbol, f, s, operation, total_value, ret_percent, rate, counts, mean_duration, Prices.timeframe, periodStart, periodEnd, *qrs, *qvs, baseRets.loc[symbol], 0]

                        # print the results
                        print(f"{symbol}: fast: {f}; slow: {s}; {operation} Earn: {total_value:.2f}[{counts}]; Period: {periodStart} - {periodEnd}")
        print("Output the excel ... ")
        # create folder
        cur_time = timeModel.getTimeS(outputFormat='%Y-%m-%d %H%M%S')
        distPath = fileModel.createDir(os.path.join(self.MainPath, 'summary'), cur_time)
        for symbol in Prices.symbols:
            # save the summary xlsx
            MaSummaryDf[MaSummaryDf['symbol'] == symbol].to_excel(os.path.join(distPath, f"{symbol}_summary.xlsx"), index=False)
        return MaSummaryDf

    # get summary dataframe
    def getMaSummaryDf(self, *,
                       symbols: list = config.Default_Forex_Symbols,
                       timeframe: str = '15min',
                       start: DatetimeTuple = (2023, 6, 1, 0, 0),
                       end: DatetimeTuple = (2023, 7, 30, 23, 59),
                       subtest: bool = False):
        """
        - loop for each combination (fast vs slow)
        :subtest: if True, will split the whole period into subset for testing
        - return the excel: fast, slow, mode, count, meanDuration, pointEarn
        """

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

            # analysis data
            self.analysis(Prices, maIndex)

        return True
