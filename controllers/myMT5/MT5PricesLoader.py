from controllers.DataController import DataController

from models.myBacktest import exchgModel
from models.myUtils import dfModel
from models.myUtils.paramModel import SymbolList, DatetimeTuple

import collections
from datetime import datetime

import MetaTrader5 as mt5
import pandas as pd

# Mt5f loader price loader
class MT5PricesLoader(DataController):  # created note 86a
    def __init__(self, mt5TimeController, mt5SymbolController, timezone='Hongkong', deposit_currency='USD'):
        super(MT5PricesLoader, self).__init__(mt5SymbolController.get_all_symbols_info(), deposit_currency)
        self.mt5TimeController = mt5TimeController

        # for Mt5f
        self.timezone = timezone  # Check: set(pytz.all_timezones_set) - (Etc/UTC)

        # prepare
        self._symbols_available = False  # only for usage of _check_if_symbols_available()

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
                    raise Exception("The {} is not provided in this broker.".format(symbol))
            self._symbols_available = True

    def _get_mt5_historical_data(self, symbol, timeframe, start, end=None):
        """
        :param symbol: str
        :param timeframe: str, '1H'
        :param start (local time): tuple (year, month, day, hour, mins) eg: (2010, 10, 30, 0, 0)
        :param end (local time): tuple (year, month, day, hour, mins), if None, then take loader until present
        :return: dataframe
        """
        timeframe = self.mt5TimeController.get_txt2timeframe(timeframe)
        utc_from = self.mt5TimeController.get_utc_time_from_broker(start, self.timezone)
        if end == None:  # if end is None, get the loader at current time
            now = datetime.today()
            now_tuple = (now.year, now.month, now.day, now.hour, now.minute)
            utc_to = self.mt5TimeController.get_utc_time_from_broker(now_tuple, self.timezone)
        else:
            utc_to = self.mt5TimeController.get_utc_time_from_broker(end, self.timezone)
        rates = mt5.copy_rates_range(symbol, timeframe, utc_from, utc_to)
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

    def _get_mt5_prices(self, symbols, timeframe, start=None, end=None, ohlcvs='111100', count: int = 10):
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

    def split_Prices(self, Prices, percentage):
        keys = list(Prices.__dict__.keys())
        prices = collections.namedtuple("prices", keys)
        train_list, test_list = [], []
        for key, df in Prices.__dict__.items():
            train, test = dfModel.split_df(df, percentage)
            train_list.append(train)
            test_list.append(test)
        Train_Prices = prices._make(train_list)
        Test_Prices = prices._make(test_list)
        return Train_Prices, Test_Prices

    def getPrices(self, *, symbols: SymbolList, start: DatetimeTuple, end: DatetimeTuple, timeframe: str, count: int = 0, ohlcvs: str = '111100'):
        """
        :param count: 0 if want to get the Data from start to end, otherwise will get the latest bar Data
        :param ohlcvs: 000000 means that get simple version of prices
        """
        q2d_exchg_symbols = exchgModel.get_exchange_symbols(symbols, self.all_symbol_info, self.deposit_currency, 'q2d')
        b2d_exchg_symbols = exchgModel.get_exchange_symbols(symbols, self.all_symbol_info, self.deposit_currency, 'b2d')

        # read loader in dictionary format
        required_symbols = list(set(symbols + q2d_exchg_symbols + b2d_exchg_symbols))
        self.check_if_symbols_available(required_symbols)  # if not, raise Exception
        prices = self._get_mt5_prices(required_symbols, timeframe, start, end, ohlcvs, count)
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
