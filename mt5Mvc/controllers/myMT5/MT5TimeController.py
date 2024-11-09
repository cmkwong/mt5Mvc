import config

from datetime import datetime, timedelta
import pytz

class MT5TimeController:

    def get_txt2timeframe(self, timeframe_txt):
        return config.timeframe_ftext_dicts[timeframe_txt]

    def get_timeframe2txt(self, mt5_timeframe_txt):
        return config.timeframe_ptext_dicts[mt5_timeframe_txt]

    def get_utc_time_from_broker(self, dateTuple, timezone):
        """
        :param dateTuple: tuple (year, month, day, hour, mins) eg: (2010, 10, 30, 0, 0)
        :param timezone: Check: set(pytz.all_timezones_set) - (Etc/UTC)
        :return: datetime format
        """
        dt = datetime(dateTuple[0], dateTuple[1], dateTuple[2], hour=dateTuple[3], minute=dateTuple[4]) + timedelta(hours=config.Broker_Time_Between_UTC, minutes=0)
        utc_time = pytz.timezone(timezone).localize(dt)
        return utc_time

    def get_current_utc_time_from_broker(self, timezone):
        """
        :param time: tuple (year, month, day, hour, mins) eg: (2010, 10, 30, 0, 0)
        :param timezone: Check: set(pytz.all_timezones_set) - (Etc/UTC)
        :return: datetime format
        """
        now = datetime.today()
        dt = datetime(now.year, now.month, now.day, hour=now.hour, minute=now.minute) + timedelta(hours=config.Broker_Time_Between_UTC, minutes=0)
        utc_time = pytz.timezone(timezone).localize(dt)
        return utc_time