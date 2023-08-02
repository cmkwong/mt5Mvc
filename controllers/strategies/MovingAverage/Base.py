from models.myBacktest import techModel
from models.myUtils import dfModel

import pandas as pd

class Base:

    MA_DATA_COLS = ['close', 'cc', 'valueDiff', 'fast', 'slow', 'long', 'short']

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