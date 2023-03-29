import config

from datetime import datetime, timedelta
import MetaTrader5 as mt5
import pytz

class MT5TimeController:
    # def __init__(self):
        # self.timeframe_dict = {"M1": mt5.TIMEFRAME_M1, "M2": mt5.TIMEFRAME_M2, "M3": mt5.TIMEFRAME_M3, "M4": mt5.TIMEFRAME_M4,
        #                        "M5": mt5.TIMEFRAME_M5, "M6": mt5.TIMEFRAME_M6, "M10": mt5.TIMEFRAME_M10, "M12": mt5.TIMEFRAME_M12,
        #                        "M15": mt5.TIMEFRAME_M15, "M20": mt5.TIMEFRAME_M20, "M30": mt5.TIMEFRAME_M30, "H1": mt5.TIMEFRAME_H1,
        #                        "H2": mt5.TIMEFRAME_H2, "H3": mt5.TIMEFRAME_H3, "H4": mt5.TIMEFRAME_H4, "H6": mt5.TIMEFRAME_H6,
        #                        "H8": mt5.TIMEFRAME_H8, "H12": mt5.TIMEFRAME_H12, "D1": mt5.TIMEFRAME_D1, "W1": mt5.TIMEFRAME_W1,
        #                        "MN1": mt5.TIMEFRAME_MN1}
    timeframe_ftext_dicts = {"1min": mt5.TIMEFRAME_M1, "2min": mt5.TIMEFRAME_M2, "3min": mt5.TIMEFRAME_M3, "4min": mt5.TIMEFRAME_M4,
                             "5min": mt5.TIMEFRAME_M5, "6min": mt5.TIMEFRAME_M6, "10min": mt5.TIMEFRAME_M10,
                             "12min": mt5.TIMEFRAME_M12,
                             "15min": mt5.TIMEFRAME_M15, "20min": mt5.TIMEFRAME_M20, "30min": mt5.TIMEFRAME_M30,
                             "1H": mt5.TIMEFRAME_H1,
                             "2H": mt5.TIMEFRAME_H2, "3H": mt5.TIMEFRAME_H3, "4H": mt5.TIMEFRAME_H4, "6H": mt5.TIMEFRAME_H6,
                             "8H": mt5.TIMEFRAME_H8, "12H": mt5.TIMEFRAME_H12, "1D": mt5.TIMEFRAME_D1, "1W": mt5.TIMEFRAME_W1,
                             "1MN": mt5.TIMEFRAME_MN1}

    timeframe_ptext_dicts = {mt5.TIMEFRAME_M1: "1min", mt5.TIMEFRAME_M2: "2min", mt5.TIMEFRAME_M3: "3min", mt5.TIMEFRAME_M4: "4min",
                             mt5.TIMEFRAME_M5: "5min", mt5.TIMEFRAME_M6: "6min", mt5.TIMEFRAME_M10: "10min",
                             mt5.TIMEFRAME_M12: "12min",
                             mt5.TIMEFRAME_M15: "15min", mt5.TIMEFRAME_M20: "M20", mt5.TIMEFRAME_M30: "30min",
                             mt5.TIMEFRAME_H1: "1H",
                             mt5.TIMEFRAME_H2: "2H", mt5.TIMEFRAME_H3: "3H", mt5.TIMEFRAME_H4: "4H", mt5.TIMEFRAME_H6: "6H",
                             mt5.TIMEFRAME_H8: "8H", mt5.TIMEFRAME_H12: "12H", mt5.TIMEFRAME_D1: "1D", mt5.TIMEFRAME_W1: "1D",
                             mt5.TIMEFRAME_MN1: "1MN"}

    def get_txt2timeframe(self, timeframe_txt):
        return self.timeframe_ftext_dicts[timeframe_txt]

    def get_timeframe2txt(self, mt5_timeframe_txt):
        return self.timeframe_ptext_dicts[mt5_timeframe_txt]

    def get_utc_time_from_broker(self, time, timezone):
        """
        :param time: tuple (year, month, day, hour, mins) eg: (2010, 10, 30, 0, 0)
        :param timezone: Check: set(pytz.all_timezones_set) - (Etc/UTC)
        :return: datetime format
        """
        dt = datetime(time[0], time[1], time[2], hour=time[3], minute=time[4]) + timedelta(hours=config.Broker_Time_Between_UTC, minutes=0)
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