from controllers.myMT5.InitPrices import InitPrices
from models.myBacktest import exchgModel, pointsModel
from models.myUtils import dfModel
from models.myUtils.paramModel import SymbolList, DatetimeTuple

import collections
from datetime import datetime

import MetaTrader5 as mt5
import pandas as pd


class DataController:
    def __init__(self, all_symbol_info, deposit_currency='USD'):
        self.all_symbol_info = all_symbol_info
        self.deposit_currency = deposit_currency
        self.type_names = ['open', 'high', 'low', 'close', 'tick_volume', 'spread']

    def _get_ohlc_rule(self, df):
        """
        note 85e
        Only for usage on change_timeframe()
        :param check_code: list
        :return: raise exception
        """
        check_code = [0, 0, 0, 0]
        ohlc_rule = {}
        for key in df.columns:
            if key == 'open':
                check_code[0] = 1
                ohlc_rule['open'] = 'first'
            elif key == 'high':
                check_code[1] = 1
                ohlc_rule['high'] = 'max'
            elif key == 'low':
                check_code[2] = 1
                ohlc_rule['low'] = 'min'
            elif key == 'close':
                check_code[3] = 1
                ohlc_rule['close'] = 'last'
        # first exception
        if check_code[1] == 1 or check_code[2] == 1:
            if check_code[0] == 0 or check_code[3] == 0:
                raise Exception("When high/low needed, there must be open/close loader included. \nThere is not open/close loader.")
        # Second exception
        if len(df.columns) > 4:
            raise Exception("The DataFrame columns is exceeding 4")
        return ohlc_rule

    def change_timeframe(self, df, timeframe='1H'):
        """
        note 84f
        :param df: pd.DataFrame, having header: open high low close
        :param rule: can '2H', https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#resampling
        :return:
        """
        ohlc_rule = self._get_ohlc_rule(df)
        df = df.resample(timeframe).apply(ohlc_rule)
        df.dropna(inplace=True)
        return df

    def _price_type_from_code(self, ohlcvs):
        """
        :param ohlcvs: str of code, eg: '100100'
        :return: list, eg: ['open', 'close']
        """
        required_types = []
        for i, c in enumerate(ohlcvs):
            if c == '1':
                required_types.append(self.type_names[i])
        return required_types

    def _prices_df2dict(self, prices_raw_df, symbols, ohlcvs):

        # rename columns of the prices_df
        col_names = self._price_type_from_code(ohlcvs)
        prices_raw_df.columns = col_names * len(symbols)

        prices = {}
        max_length = len(prices_raw_df.columns)
        step = len(col_names)
        for i in range(0, max_length, step):
            symbol = symbols[int(i / step)]
            prices[symbol] = prices_raw_df.iloc[:, i:i + step]
        return prices

    def _get_specific_from_prices(self, prices: pd.DataFrame, required_symbols, ohlcvs):
        """
        :param prices: {symbol: pd.DataFrame}
        :param required_symbols: [str]
        :param ohlcvs: str, '1000'
        :return: pd.DataFrame
        """
        types = self._price_type_from_code(ohlcvs)
        required_prices = pd.DataFrame()
        for i, symbol in enumerate(required_symbols):
            if i == 0:
                required_prices = prices[symbol].loc[:, types].copy()
            else:
                required_prices = pd.concat([required_prices, prices[symbol].loc[:, types]], axis=1)
        required_prices.columns = required_symbols
        return required_prices

    def get_latest_Prices_format(self, symbols, prices, q2d_exchg_symbols, count):

        close_prices = self._get_specific_from_prices(prices, symbols, ohlcvs='000100')
        if len(close_prices) != count:  # note 63a
            print("prices_df length of Data is not equal to count")
            return False

        # calculate the change of close price (with latest close prices)
        change_close_prices = ((close_prices - close_prices.shift(1)) / close_prices.shift(1)).fillna(0.0)

        # get point diff values with latest value
        points_dff_values_df = pointsModel.get_points_dff_values_df(symbols, close_prices, close_prices.shift(periods=1), self.all_symbol_info)

        # get quote exchange with values
        exchg_close_prices = self._get_specific_from_prices(prices, q2d_exchg_symbols, ohlcvs='000100')
        q2d_exchange_rate_df = exchgModel.get_exchange_df(symbols, q2d_exchg_symbols, exchg_close_prices, self.deposit_currency, "q2d")
        # if len(q2d_exchange_rate_df_o) or len(q2d_exchange_rate_df_c) == 39, return false and run again
        if len(q2d_exchange_rate_df) != count:  # note 63a
            print("q2d_exchange_rate_df_o or q2d_exchange_rate_df_c length of Data is not equal to count")
            return False

        Prices = InitPrices(symbols=symbols,
                            close=close_prices,
                            cc=change_close_prices,
                            ptDv=points_dff_values_df,
                            quote_exchg=q2d_exchange_rate_df
                            )

        return Prices

    def get_Prices_format(self, symbols, prices, q2d_exchg_symbols, b2d_exchg_symbols, ohlcvs):

        # init to None
        open_prices, high_prices, low_prices, close_prices, changes, volume, spread = None, None, None, None, None, None, None

        # get the change of close price
        close_prices = self._get_specific_from_prices(prices, symbols, ohlcvs='000100')
        changes = ((close_prices - close_prices.shift(1)) / close_prices.shift(1)).fillna(0.0)

        # get point diff values
        # open_prices = _get_specific_from_prices(prices, symbols, ohlcvs='1000')
        points_dff_values_df = pointsModel.get_points_dff_values_df(symbols, close_prices, close_prices.shift(periods=1), self.all_symbol_info)

        # get the quote to deposit exchange rate
        exchg_close_prices = self._get_specific_from_prices(prices, q2d_exchg_symbols, ohlcvs='000100')
        q2d_exchange_rate_df = exchgModel.get_exchange_df(symbols, q2d_exchg_symbols, exchg_close_prices, self.deposit_currency, "q2d")

        # get the base to deposit exchange rate
        exchg_close_prices = self._get_specific_from_prices(prices, b2d_exchg_symbols, ohlcvs='000100')
        b2d_exchange_rate_df = exchgModel.get_exchange_df(symbols, q2d_exchg_symbols, exchg_close_prices, self.deposit_currency, "b2d")

        # assign the column into each collection tuple
        Prices = InitPrices(symbols=symbols,
                            close=close_prices,
                            cc=changes,
                            ptDv=points_dff_values_df,
                            quote_exchg=q2d_exchange_rate_df,
                            base_exchg=b2d_exchange_rate_df
                            )
        # get open prices
        if ohlcvs[0] == '1':
            Prices.open = self._get_specific_from_prices(prices, symbols, ohlcvs='100000')

        # get the change of high price
        if ohlcvs[1] == '1':
            Prices.high = self._get_specific_from_prices(prices, symbols, ohlcvs='010000')

        # get the change of low price
        if ohlcvs[2] == '1':
            Prices.low = self._get_specific_from_prices(prices, symbols, ohlcvs='001000')

        # get the tick volume
        if ohlcvs[4] == '1':
            Prices.volume = self._get_specific_from_prices(prices, symbols, ohlcvs='000010')

        if ohlcvs[5] == '1':
            Prices.spread = self._get_specific_from_prices(prices, symbols, ohlcvs='000001')

        return Prices
