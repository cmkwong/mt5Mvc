from datetime import datetime

import MetaTrader5 as mt5
import pandas as pd

from models.myBacktest import timeModel
from controllers.myMT5.BasePricesLoader import BasePricesLoader

class BaseMT5PricesLoader(BasePricesLoader):
    def __init__(self, broker_time_between_utc=2):
        self.broker_time_between_utc = broker_time_between_utc  # default is 2

    def _get_symbol_info_tick(self, symbol):
        lasttick = mt5.symbol_info_tick(symbol)._asdict()
        return lasttick

    def _get_historical_data(self, symbol, timeframe, timezone, start, end=None):
        """
        :param symbol: str
        :param timeframe: str, '1H'
        :param timezone: Check: set(pytz.all_timezones_set) - (Etc/UTC)
        :param start (local time): tuple (year, month, day, hour, mins) eg: (2010, 10, 30, 0, 0)
        :param end (local time): tuple (year, month, day, hour, mins), if None, then take loader until present
        :return: dataframe
        """
        timeframe = timeModel.get_txt2timeframe(timeframe)
        utc_from = timeModel.get_utc_time_from_broker(start, timezone, self.broker_time_between_utc)
        if end == None:  # if end is None, get the loader at current time
            now = datetime.today()
            now_tuple = (now.year, now.month, now.day, now.hour, now.minute)
            utc_to = timeModel.get_utc_time_from_broker(now_tuple, timezone, self.broker_time_between_utc)
        else:
            utc_to = timeModel.get_utc_time_from_broker(end, timezone, self.broker_time_between_utc)
        rates = mt5.copy_rates_range(symbol, timeframe, utc_from, utc_to)
        rates_frame = pd.DataFrame(rates, dtype=float)  # create DataFrame out of the obtained loader
        rates_frame['time'] = pd.to_datetime(rates_frame['time'], unit='s')  # convert time in seconds into the datetime format
        rates_frame = rates_frame.set_index('time')
        return rates_frame

    def _get_current_bars(self, symbol, timeframe, count):
        """
        :param symbols: str
        :param timeframe: str, '1H'
        :param count: int
        :return: df
        """
        timeframe = timeModel.get_txt2timeframe(timeframe)
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)  # 0 means the current bar
        rates_frame = pd.DataFrame(rates, dtype=float)
        rates_frame['time'] = pd.to_datetime(rates_frame['time'], unit='s')
        rates_frame = rates_frame.set_index('time')
        return rates_frame

    def _get_mt5_prices(self, symbols, timeframe, timezone, start=None, end=None, ohlcvs='111100', count: int = 10):
        """
        :param symbols: [str]
        :param timeframe: str, '1H'
        :param timezone: str "Hongkong"
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
                price = self._get_current_bars(symbol, timeframe, count).loc[:, required_types]
                join = 'inner'  # if getting count, need to join=inner to check if loader getting completed
            elif count == 0:  # get loader from start to end
                price = self._get_historical_data(symbol, timeframe, timezone, start, end).loc[:, required_types]
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
