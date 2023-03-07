from datetime import datetime, timedelta
import MetaTrader5 as mt5
import pytz

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


def get_txt2timeframe(timeframe_txt):
    return timeframe_ftext_dicts[timeframe_txt]


def get_timeframe2txt(mt5_timeframe_txt):
    return timeframe_ptext_dicts[mt5_timeframe_txt]


def get_utc_time_from_broker(time, timezone, broker_time_between_utc):
    """
    :param time: tuple (year, month, day, hour, mins) eg: (2010, 10, 30, 0, 0)
    :param timezone: Check: set(pytz.all_timezones_set) - (Etc/UTC)
    :return: datetime format
    """
    dt = datetime(time[0], time[1], time[2], hour=time[3], minute=time[4]) + timedelta(hours=broker_time_between_utc, minutes=0)
    utc_time = pytz.timezone(timezone).localize(dt)
    return utc_time


def get_current_utc_time_from_broker(timezone, broker_time_between_utc):
    """
    :param time: tuple (year, month, day, hour, mins) eg: (2010, 10, 30, 0, 0)
    :param timezone: Check: set(pytz.all_timezones_set) - (Etc/UTC)
    :return: datetime format
    """
    now = datetime.today()
    dt = datetime(now.year, now.month, now.day, hour=now.hour, minute=now.minute) + timedelta(hours=broker_time_between_utc, minutes=0)
    utc_time = pytz.timezone(timezone).localize(dt)
    return utc_time

# def get_action_date(df, signal):
#     """
#     :param signal: Series(Boolean) without time index
#     :return: start_date_list, end_date_list
#     """
#     start_date_list, end_date_list = [], []
#     # int_signal = signal.astype(int).diff(1)
#     start_index, end_index = indexModel.get_start_end_index(signal)
#     # buy date
#     dates = list(df['time'][start_index])
#     start_date_list.extend([str(date) for date in dates])
#
#     # sell date
#     dates = list(df['time'][end_index])
#     end_date_list.extend([str(date) for date in dates])
#
#     return start_date_list, end_date_list
