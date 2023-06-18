from controllers.myMT5.InitPrices import InitPrices
from models.myBacktest import exchgModel, pointsModel
from models.myUtils import timeModel
from models.myUtils.paramModel import SymbolList, DatetimeTuple
import config

from datetime import datetime

import MetaTrader5 as mt5
import pandas as pd


# Mt5f loader price loader
class MT5PricesLoader:  # created note 86a
    def __init__(self, mt5TimeController, mt5SymbolController, nodeJsApiController, timezone='Hongkong', deposit_currency='USD'):
        self.nodeJsApiController = nodeJsApiController
        self.mt5SymbolController = mt5SymbolController
        self.mt5TimeController = mt5TimeController

        # for Mt5f
        self.all_symbol_info = self.mt5SymbolController.get_all_symbols_info()
        self.timezone = timezone  # Check: set(pytz.all_timezones_set) - (Etc/UTC)
        self.deposit_currency = deposit_currency

        # prepare
        self._symbols_available = False  # only for usage of _check_if_symbols_available()

        self.source = 'mt5'

    def switchSource(self):
        if self.source == 'mt5':
            self.source = 'local'
        else:
            self.source = 'mt5'
        print(f"The price loader has switched to {self.source}")

    def _get_mt5_historical_data(self, symbol, timeframe, start, end=None):
        """
        :param symbol: str
        :param timeframe: str, '1H'
        :param start (local time): tuple (year, month, day, hour, mins) eg: (2010, 10, 30, 0, 0)
        :param end (local time): tuple (year, month, day, hour, mins), if None, then take loader until present
        :return: dataframe
        """
        if self.source == 'local':
            utc_from_tuple = timeModel.get_utc_time_with_timezone(start, self.timezone, 1)
            utc_to_tuple = timeModel.get_utc_time_with_timezone(end, self.timezone, 1)
            rates_frame = self.nodeJsApiController.downloadForexData(symbol, timeframe, utc_from_tuple, utc_to_tuple)
        else:
            utc_from = self.mt5TimeController.get_utc_time_from_broker(start, self.timezone)
            if end == None:  # if end is None, get the loader at current time
                now = datetime.today()
                now_tuple = (now.year, now.month, now.day, now.hour, now.minute)
                utc_to = self.mt5TimeController.get_utc_time_from_broker(now_tuple, self.timezone)
            else:
                utc_to = self.mt5TimeController.get_utc_time_from_broker(end, self.timezone)
            mt5Timeframe = self.mt5TimeController.get_txt2timeframe(timeframe)
            rates = mt5.copy_rates_range(symbol, mt5Timeframe, utc_from, utc_to)
            rates_frame = pd.DataFrame(rates, dtype=float)  # create DataFrame out of the obtained loader
            rates_frame['time'] = pd.to_datetime(rates_frame['time'], unit='s')  # convert time in seconds into the datetime format
            rates_frame = rates_frame.set_index('time')
        return rates_frame

    def _get_mt5_current_bars(self, symbol, timeframe, count):
        """
        :param symbols: str
        :param timeframe: str, '1H'
        :param count: int
        :return: df
        """
        timeframe = self.mt5TimeController.get_txt2timeframe(timeframe)
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)  # 0 means the current bar
        rates_frame = pd.DataFrame(rates, dtype=float)
        rates_frame['time'] = pd.to_datetime(rates_frame['time'], unit='s')
        rates_frame = rates_frame.set_index('time')
        return rates_frame

    # checking if symbol available
    def check_if_symbols_available(self, required_symbols):
        """
        check if symbols exist, note 83h
        :param required_symbols: [str]
        :return: None
        """
        if not self._symbols_available:
            for symbol in required_symbols:
                try:
                    _ = self.all_symbol_info[symbol]
                except KeyError:
                    raise Exception("The {} is not provided.".format(symbol))
            self._symbols_available = True

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
        if self.source == 'local':
            type_names = ['open', 'high', 'low', 'close', 'volume', 'spread']
        else:
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

    def get_exchange_symbols(self, symbols, exchg_type='q2d'):
        """
        Find all the currency pair related to and required currency and deposit symbol
        :param symbols: [str] : ["AUDJPY", "AUDUSD", "CADJPY", "EURUSD", "NZDUSD", "USDCAD"]
        :param exchg_type: str, 'q2d' = quote to deposit OR 'b2d' = base to deposit
        :return: [str], get required exchange symbol in list: ['USDJPY', 'AUDUSD', 'USDJPY', 'EURUSD', 'NZDUSD', 'USDCAD']
        """
        symbol_names = list(self.all_symbol_info.keys())
        exchange_symbols = []
        target_symbol = None
        for symbol in symbols:
            if exchg_type == 'b2d':
                target_symbol = symbol[:3]
            elif exchg_type == 'q2d':
                target_symbol = symbol[3:]
            if target_symbol != self.deposit_currency:  # if the symbol not relative to required deposit currency
                test_symbol_1 = target_symbol + self.deposit_currency
                test_symbol_2 = self.deposit_currency + target_symbol
                if test_symbol_1 in symbol_names:
                    exchange_symbols.append(test_symbol_1)
                    continue
                elif test_symbol_2 in symbol_names:
                    exchange_symbols.append(test_symbol_2)
                    continue
                else:  # if not found the relative pair with respect to deposit currency, raise the error
                    raise Exception("{} has no relative currency with respect to deposit {}.".format(target_symbol, self.deposit_currency))
            else:  # if the symbol already relative to deposit currency
                exchange_symbols.append(symbol)
        return exchange_symbols

    def _get_prices_df(self, symbols, timeframe, start=None, end=None, ohlcvs='111100', count: int = 10):
        """
        :param symbols: [str]
        :param timeframe: str, '1H'
        :param start: (2010,1,1,0,0), if both start and end is None, use function get_current_bars()
        :param end: (2020,1,1,0,0), if just end is None, get the historical loader from date to current
        :param ohlcvs: str, eg: '111100' => open, high, low, close, volume, spread
        :param count: int, for get_current_bar_function()
        :return: pd.DataFrame
        """
        join = 'outer'
        required_types = self._price_type_from_code(ohlcvs)
        prices_df = None
        for i, symbol in enumerate(symbols):
            if count > 0:  # get the latest units of loader
                price = self._get_mt5_current_bars(symbol, timeframe, count).loc[:, required_types]
                join = 'inner'  # if getting count, need to join=inner to check if loader getting completed
            elif count == 0:  # get loader from start to end
                price = self._get_mt5_historical_data(symbol, timeframe, start, end).loc[:, required_types]
            else:
                raise Exception('start-date must be set when end-date is being set.')
            if i == 0:
                prices_df = price.copy()
            else:
                prices_df = pd.concat([prices_df, price], axis=1, join=join)

        # replace NaN values with preceding values
        prices_df.fillna(method='ffill', inplace=True)
        prices_df.dropna(inplace=True, axis=0)

        # get prices in dict
        prices = self._prices_df2dict(prices_df, symbols, ohlcvs)

        return prices

    # # split the Prices by ratio
    # def split_Prices(self, Prices, percentage):
    #     keys = list(Prices.__dict__.keys())
    #     prices = collections.namedtuple("prices", keys)
    #     train_list, test_list = [], []
    #     for key, df in Prices.__dict__.items():
    #         train, test = dfModel.split_df(df, percentage)
    #         train_list.append(train)
    #         test_list.append(test)
    #     Train_Prices = prices._make(train_list)
    #     Test_Prices = prices._make(test_list)
    #     return Train_Prices, Test_Prices

    # get latest Prices format
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

    def getPrices(self, *,
                  symbols: SymbolList = config.DefaultSymbols,
                  start: DatetimeTuple = (2023, 5, 1, 0, 0),
                  end: DatetimeTuple = (2023, 6, 1, 23, 59),
                  timeframe: str = '15min',
                  count: int = 0,
                  ohlcvs: str = '111111'):
        """
        :param count: 0 if want to get the Data from start to end, otherwise will get the latest bar Data
        :param ohlcvs: 000000 means that get simple version of prices
        """
        q2d_exchg_symbols = self.get_exchange_symbols(symbols, 'q2d')
        b2d_exchg_symbols = self.get_exchange_symbols(symbols, 'b2d')

        # read loader in dictionary format
        required_symbols = list(set(symbols + q2d_exchg_symbols + b2d_exchg_symbols))
        self.check_if_symbols_available(required_symbols)  # if not, raise Exception
        prices = self._get_prices_df(required_symbols, timeframe, start, end, ohlcvs, count)
        Prices = self.get_Prices_format(symbols, prices, q2d_exchg_symbols, b2d_exchg_symbols, ohlcvs)
        return Prices

# @dataclass
# class get_data_TKPARAM(TkWidgetLabel):
#     symbols: dataclass = TkInitWidget(cat='get_data', id='1', type=TkWidgetLabel.DROPDOWN, value=['EURUSD', 'GBPUSD', 'USDJPY'])
#     start: Tuple[int] = field(default_factory=lambda: (2010, 1, 1))
#     end: Tuple[int] = field(default_factory=lambda: (2022, 1, 1))
#
#     def __init__(self):
#         super(get_data_TKPARAM, self).__init__()
#
# d = get_data_TKPARAM()
# print()
