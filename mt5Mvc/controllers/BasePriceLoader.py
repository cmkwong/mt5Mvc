from mt5Mvc.models.myBacktest import exchgModel
from mt5Mvc.controllers.myMT5.InitPrices import InitPrices

import config
import pandas as pd

class BasePriceLoader:
    def __init__(self):
        self.data_source = None
        self.DATA_SOURCES = ['mt5', 'sql', 'local']

    def switch_source(self, switch_command='mt5'):
        assert switch_command in self.DATA_SOURCES, f'The command of switch source is not correct - {self.DATA_SOURCES}'
        self.data_source = switch_command
        print(f"The price loader has switched to {self.data_source}")

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
        # define the column
        type_names = ['open', 'high', 'low', 'close', 'tick_volume', 'spread']
        # getting required columns
        required_types = []
        for i, c in enumerate(ohlcvs):
            if c == '1':
                required_types.append(type_names[i])
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

    def get_Prices_format(self, symbols, prices, ohlcvs, q2d_exchg_symbols=None, b2d_exchg_symbols=None, all_symbols_info=None):

        # get the change of close price
        close_prices = self._get_specific_from_prices(prices, symbols, ohlcvs='000100')
        changes = ((close_prices - close_prices.shift(1)) / close_prices.shift(1)).fillna(0.0)

        # get the quote to deposit exchange rate
        quote_exchg = pd.DataFrame()
        if q2d_exchg_symbols:
            exchg_close_prices = self._get_specific_from_prices(prices, q2d_exchg_symbols, ohlcvs='000100')
            q2d_exchange_rate_df = exchgModel.get_exchange_df(symbols, q2d_exchg_symbols, exchg_close_prices, config.DepositCurrency, "q2d")
            quote_exchg = q2d_exchange_rate_df

        # get the base to deposit exchange rate
        base_exchg = pd.DataFrame()
        if b2d_exchg_symbols:
            exchg_close_prices = self._get_specific_from_prices(prices, b2d_exchg_symbols, ohlcvs='000100')
            b2d_exchange_rate_df = exchgModel.get_exchange_df(symbols, b2d_exchg_symbols, exchg_close_prices, config.DepositCurrency, "b2d")
            base_exchg = b2d_exchange_rate_df

        # get open prices
        open = pd.DataFrame()
        if ohlcvs[0] == '1':
            open = self._get_specific_from_prices(prices, symbols, ohlcvs='100000')

        # get the change of high price
        high = pd.DataFrame()
        if ohlcvs[1] == '1':
            high = self._get_specific_from_prices(prices, symbols, ohlcvs='010000')

        # get the change of low price
        low = pd.DataFrame()
        if ohlcvs[2] == '1':
            low = self._get_specific_from_prices(prices, symbols, ohlcvs='001000')

        # get the tick volume
        volume = pd.DataFrame()
        if ohlcvs[4] == '1':
            volume = self._get_specific_from_prices(prices, symbols, ohlcvs='000010')

        # get the spread
        spread = pd.DataFrame()
        if ohlcvs[5] == '1':
            spread = self._get_specific_from_prices(prices, symbols, ohlcvs='000001')

        # assign the column into each collection tuple
        Prices = InitPrices(symbols=symbols,
                            close=close_prices,
                            cc=changes,
                            all_symbols_info=all_symbols_info,
                            quote_exchg=quote_exchg,
                            base_exchg=base_exchg,
                            open=open,
                            high=high,
                            low=low,
                            volume=volume,
                            spread=spread
                            )
        return Prices