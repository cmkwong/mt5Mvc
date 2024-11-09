from mt5Mvc.controllers.BasePriceLoader import BasePriceLoader
from mt5Mvc.models.myUtils import timeModel
from mt5Mvc.models.myUtils.paramModel import SymbolList, DatetimeTuple
from mt5Mvc.controllers.myNodeJs.NodeJsApiController import NodeJsApiController
from mt5Mvc.controllers.myMT5.MT5SymbolController import MT5SymbolController
from mt5Mvc.controllers.myMT5.MT5TimeController import MT5TimeController

import config

from datetime import datetime

import MetaTrader5 as mt5
import pandas as pd
import numpy as np

# Mt5f loader price loader
class MT5PricesLoader(BasePriceLoader):  # created note 86a
    def __init__(self):
        super().__init__()
        self.nodeJsApiController = NodeJsApiController()
        self.symbolController = MT5SymbolController()
        self.timeController = MT5TimeController()

        # price type
        self.Price_Type = 'forex'

        # prepare
        self._symbols_available = False  # only for usage of _check_if_symbols_available()

        # overwrite the source by mt5
        self.data_source = 'mt5'

    def _get_historical_data(self, symbol, timeframe, start, end=None):
        """
        :param symbol: str
        :param timeframe: str, '1H'
        :param start (local time): tuple (year, month, day, hour, mins) eg: (2010, 10, 30, 0, 0)
        :param end (local time): tuple (year, month, day, hour, mins), if None, then take loader until present
        :return: dataframe
        """
        if self.data_source == 'sql':
            utc_from_tuple = timeModel.get_utc_time_with_timezone(start, config.TimeZone, 1)
            utc_to_tuple = timeModel.get_utc_time_with_timezone(end, config.TimeZone, 1)
            rates_frame = self.nodeJsApiController.downloadSeriesData('forex', symbol, timeframe, utc_from_tuple, utc_to_tuple)
        else:
            utc_from = self.timeController.get_utc_time_from_broker(start, config.TimeZone)
            if end == None:  # if end is None, get the loader at current time
                now = datetime.today()
                now_tuple = (now.year, now.month, now.day, now.hour, now.minute)
                utc_to = self.timeController.get_utc_time_from_broker(now_tuple, config.TimeZone)
            else:
                utc_to = self.timeController.get_utc_time_from_broker(end, config.TimeZone)
            mt5Timeframe = self.timeController.get_txt2timeframe(timeframe)
            rates = mt5.copy_rates_range(symbol, mt5Timeframe, utc_from, utc_to)
            if not isinstance(rates, np.ndarray):
                return False
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
        timeframe = self.timeController.get_txt2timeframe(timeframe)
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
            all_symbols_info = self.symbolController.get_all_symbols_info()
            for symbol in required_symbols:
                try:
                    _ = all_symbols_info[symbol]
                except KeyError:
                    raise Exception("The {} is not provided.".format(symbol))
            self._symbols_available = True

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
                price = self._get_mt5_current_bars(symbol, timeframe, count)
                join = 'inner'  # if getting count, need to join=inner to check if loader getting completed
            elif count == 0:  # get loader from start to end
                price = self._get_historical_data(symbol, timeframe, start, end)
                if not isinstance(price, pd.DataFrame):
                    print("Cannot get data from MT5. Will try to get from SQL ... ")
                    _originalSource = self.data_source
                    self.data_source = 'sql'
                    price = self._get_historical_data(symbol, timeframe, start, end)
                    self.data_source = _originalSource # resume to original data source
            else:
                raise Exception('start-date must be set when end-date is being set.')
            price = price.loc[:, required_types]
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

    def getPrices(self, *,
                  symbols: SymbolList = config.Default_Forex_Symbols,
                  start: DatetimeTuple = (2023, 6, 1, 0, 0),
                  end: DatetimeTuple = (2023, 6, 30, 23, 59),
                  timeframe: str = '15min',
                  count: int = 0,
                  ohlcvs: str = '111111'):
        """
        :param count: 0 if want to get the Data from start to end, otherwise will get the latest bar Data
        :param ohlcvs: 000000 means that get simple version of prices
        """
        q2d_exchg_symbols = self.symbolController.get_exchange_symbols(symbols, 'q2d')
        b2d_exchg_symbols = self.symbolController.get_exchange_symbols(symbols, 'b2d')

        # read loader in dictionary format
        required_symbols = list(set(symbols + q2d_exchg_symbols + b2d_exchg_symbols))
        self.check_if_symbols_available(required_symbols)  # if not, raise Exception
        prices = self._get_prices_df(required_symbols, timeframe, start, end, ohlcvs, count)
        Prices = self.get_Prices_format(symbols, prices, ohlcvs, timeframe, q2d_exchg_symbols, b2d_exchg_symbols, self.symbolController.get_all_symbols_info())
        return Prices

# @dataclass
# class get_data_TKPARAM    (TkWidgetLabel):
#     symbols: dataclass = TkInitWidget(cat='get_data', id='1', type=TkWidgetLabel.DROPDOWN, value=['EURUSD', 'GBPUSD', 'USDJPY'])
#     start: Tuple[int] = field(default_factory=lambda: (2010, 1, 1))
#     end: Tuple[int] = field(default_factory=lambda: (2022, 1, 1))
#
#     def __init__(self):
#         super(get_data_TKPARAM, self).__init__()
#
# d = get_data_TKPARAM()
# print()
