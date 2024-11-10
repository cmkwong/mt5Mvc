from mt5Mvc.models.myBacktest import techModel
from mt5Mvc.models.myUtils import dfModel
from mt5Mvc.models.myUtils import timeModel

import pandas as pd
import numpy as np
import os

pd.set_option("future.no_silent_downcasting", True)
pd.options.mode.copy_on_write = True

class Base:
    MainPath = "./docs/ma"
    SummaryPath = os.path.join(MainPath, 'summary')
    DistributionPath = os.path.join(MainPath, 'dist')
    MA_DATA_COLS = ['close', 'cc', 'valueDiff', 'fast', 'slow', 'long', 'short']
    OPERATIONS = ['long', 'short']

    def getMaData(self, Prices, fast_param, slow_param):
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
        valueDiff = Prices.ptDv
        # get the point difference
        ptDiff = Prices.ptD
        # get the changes
        Change = Prices.cc
        # get the ret
        logRet = Prices.logRet
        # create empty ma_data
        MaData = pd.DataFrame(columns=columnIndex, index=Close.index)
        for i, symbol in enumerate(Prices.symbols):
            MaData[symbol, 'close'] = Close[symbol]
            MaData[symbol, 'cc'] = Change[symbol]
            MaData[symbol, 'valueDiff'] = valueDiff.iloc[:, i]
            MaData[symbol, 'ptDiff'] = ptDiff.iloc[:, i]
            MaData[symbol, 'logRet'] = logRet.iloc[:, i]
            MaData[symbol, 'fast'] = techModel.get_MA(Close[symbol], fast_param, False)
            MaData[symbol, 'slow'] = techModel.get_MA(Close[symbol], slow_param, False)
            # long signal
            MaData[symbol, 'long'] = MaData[symbol, 'fast'] > MaData[symbol, 'slow']
            MaData[symbol, 'long'] = MaData[symbol, 'long'].shift(1).fillna(False)  # signal should delay 1 timeframe
            # short signal
            MaData[symbol, 'short'] = MaData[symbol, 'fast'] < MaData[symbol, 'slow']
            MaData[symbol, 'short'] = MaData[symbol, 'short'].shift(1).fillna(False)  # signal should delay 1 timeframe

        # for grouping usage, if Signal=True, ffill the signal start date
        MaData = self.getOperationGroup_MaData(MaData)

        return MaData

    # assigning the first start date to signal==True
    def getOperationGroup_MaData(self, MaData):

        # get the symbol list
        symbols = list(MaData.columns.levels[0])

        for symbol in symbols:
            for operation in self.OPERATIONS:
                # create a column for storage required grouping information (for temporary usage)
                MaData[symbol, f'{operation}_group'] = MaData.loc[:, (symbol, operation)]

                # assign the time-index into first timeframe
                mask = (MaData[symbol, operation] > MaData[symbol, operation].shift(1)) == True
                MaData.loc[mask, (symbol, f'{operation}_group')] = MaData[mask].index

                # assign nan value for ffill
                mask = MaData[symbol, f'{operation}_group'] == True
                MaData.loc[mask, (symbol, f'{operation}_group')] = np.nan
                MaData[symbol, f'{operation}_group'] = MaData[symbol, f'{operation}_group'].ffill()
        return MaData

    def getMaDist(self, Prices, fast, slow):

        # get the Ma Data
        MaData = self.getMaData(Prices, fast, slow)

        # get the symbols
        symbols = list(MaData.columns.levels[0])

        # create the distribution
        Distributions = {}
        for symbol in symbols:
            Distributions[symbol] = {}
            for operation in self.OPERATIONS:
                # with timeModel.TimeCounter("cumprod testing. ") as timeCounter:
                earningFactor = 1 if operation == 'long' else -1
                # getting distribution
                dist = {}
                mask = MaData[symbol, f'{operation}_group'] != False  # getting the rows only need to be groupby
                masked_MaData = MaData[mask]
                # deposit change
                dist['valueDist'] = masked_MaData.loc[:, (symbol, 'valueDiff')] * earningFactor
                # % changes
                dist['changeDist'] = masked_MaData.loc[:, (symbol, 'cc')] * earningFactor
                # point change
                dist['pointDist'] = masked_MaData.loc[:, (symbol, 'ptDiff')] * earningFactor
                # return
                # masked_MaData.loc[:, (symbol, f'{operation}_return')] = dist['changeDist'] + 1
                # accumulated value
                dist['accumValue'] = masked_MaData.loc[:, [(symbol, 'valueDiff'), (symbol, f'{operation}_group')]].groupby((symbol, f'{operation}_group')).cumsum() * earningFactor
                # accumulated return
                # masked_MaData.loc[:, (symbol, f'{operation}_accumReturn')] = masked_MaData.loc[:, [(symbol, f'{operation}_return'), (symbol, f'{operation}_group')]].groupby((symbol, f'{operation}_group')).cumprod() - 1
                masked_MaData.loc[:, (symbol, f'{operation}_accumReturn')] = np.exp(masked_MaData.loc[:, symbol].loc[:, ('logRet', f'{operation}_group')].groupby(f'{operation}_group').cumsum()) - 1   # same as above
                dist['accumReturn'] = masked_MaData.loc[:, (symbol, f'{operation}_accumReturn')]
                # accumulated points
                dist['accumPoint'] = masked_MaData.loc[:, [(symbol, 'ptDiff'), (symbol, f'{operation}_group')]].groupby((symbol, f'{operation}_group')).cumsum() * earningFactor
                # group by deal (change, value and duration)
                dist['deal_valueDist'] = masked_MaData.loc[:, [(symbol, 'valueDiff'), (symbol, f'{operation}_group')]].groupby((symbol, f'{operation}_group')).sum() * earningFactor
                # dist['deal_changeDist'] = (masked_MaData.loc[:, [(symbol, f'{operation}_return'), (symbol, f'{operation}_group')]]).groupby((symbol, f'{operation}_group')).prod() - 1
                dist['deal_changeDist'] = np.exp(masked_MaData.loc[:, symbol].loc[:, ('logRet', f'{operation}_group')].groupby(f'{operation}_group').sum()) - 1
                dist['deal_durationDist'] = masked_MaData.loc[:, [(symbol, 'valueDiff'), (symbol, f'{operation}_group')]].groupby((symbol, f'{operation}_group')).count()
                # save the distribution
                Distributions[symbol][operation] = dist
        return Distributions
