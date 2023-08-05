from models.myBacktest import techModel
from models.myUtils import dfModel

import pandas as pd
import numpy as np


class Base:
    MA_DATA_COLS = ['close', 'cc', 'valueDiff', 'fast', 'slow', 'long', 'short']

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
        # create empty ma_data
        MaData = pd.DataFrame(columns=columnIndex, index=Close.index)
        for i, symbol in enumerate(Prices.symbols):
            MaData[symbol, 'close'] = Close[symbol]
            MaData[symbol, 'cc'] = Change[symbol]
            MaData[symbol, 'valueDiff'] = valueDiff.iloc[:, i]
            MaData[symbol, 'ptDiff'] = ptDiff.iloc[:, i]
            MaData[symbol, 'fast'] = techModel.get_MA(Close[symbol], fast_param, False)
            MaData[symbol, 'slow'] = techModel.get_MA(Close[symbol], slow_param, False)
            # long signal
            MaData[symbol, 'long'] = MaData[symbol, 'fast'] > MaData[symbol, 'slow']
            MaData[symbol, 'long'] = MaData[symbol, 'long'].shift(1).fillna(False)  # signal should delay 1 timeframe
            # short signal
            MaData[symbol, 'short'] = MaData[symbol, 'fast'] < MaData[symbol, 'slow']
            MaData[symbol, 'short'] = MaData[symbol, 'short'].shift(1).fillna(False)  # signal should delay 1 timeframe
        return MaData

    def getMaDist(self, MaData):
        # get the symbols
        symbols = list(MaData.columns.levels[0])

        # create the distribution
        Distributions = {}
        for symbol in symbols:
            Distributions[symbol] = {}
            for operation in ['long', 'short']:
                # create a column for storage required grouping information (for temporary usage)
                MaData[symbol, f'{operation}_group'] = MaData.loc[:, (symbol, operation)]

                # assign the time-index into first timeframe
                mask = (MaData[symbol, operation] > MaData[symbol, operation].shift(1)) == True
                MaData.loc[mask, (symbol, f'{operation}_group')] = MaData[mask].index

                # assign nan value for ffill
                mask = MaData[symbol, f'{operation}_group'] == True
                MaData.loc[mask, (symbol, f'{operation}_group')] = np.nan
                MaData[symbol, f'{operation}_group'].fillna(method='ffill', inplace=True)

                # getting distribution
                dist = {}
                mask = MaData[symbol, f'{operation}_group'] != False  # getting the rows only need to be groupby
                masked_MaData = MaData[mask]
                # deposit change
                dist['valueDist'] = masked_MaData.loc[:, (symbol, 'valueDiff')]
                # % changes
                dist['changeDist'] = masked_MaData.loc[:, (symbol, 'cc')]
                # point change
                dist['pointDist'] = masked_MaData.loc[:, (symbol, 'ptDiff')]
                # return
                masked_MaData.loc[:, (symbol, f'{operation}_return')] = masked_MaData.loc[:, (symbol, 'cc')] + 1
                # accumulated return
                masked_MaData.loc[:, (symbol, f'{operation}_accumReturn')] = masked_MaData.loc[:, [(symbol, f'{operation}_return'), (symbol, f'{operation}_group')]].groupby((symbol, f'{operation}_group')).cumprod()
                dist['accumReturn'] = masked_MaData.loc[:, (symbol, f'{operation}_accumReturn')]
                # accumulated value
                dist['accumValue'] = masked_MaData.loc[:, [(symbol, 'valueDiff'), (symbol, f'{operation}_group')]].groupby((symbol, f'{operation}_group')).cumsum()
                # group by deal (change, value and duration)
                dist['deal_valueDist'] = masked_MaData.loc[:, [(symbol, 'valueDiff'), (symbol, f'{operation}_group')]].groupby((symbol, f'{operation}_group')).sum()
                dist['deal_changeDist'] = (masked_MaData.loc[:, [(symbol, f'{operation}_return'), (symbol, f'{operation}_group')]]).groupby((symbol, f'{operation}_group')).prod() - 1
                dist['deal_durationDist'] = masked_MaData.loc[:, [(symbol, 'valueDiff'), (symbol, f'{operation}_group')]].groupby((symbol, f'{operation}_group')).count()
                # save the distribution
                Distributions[symbol][operation] = dist
        return Distributions
