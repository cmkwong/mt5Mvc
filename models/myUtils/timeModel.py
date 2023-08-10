from datetime import datetime, timedelta

import pytz


def get_time_string(tt, format='yyyy-mm-dd-H-M'):
    """
    :param tt: time_tuple: tuple (yyyy,mm,dd,H,M)
    :return: string
    """
    if format == 'yyyy-mm-dd-H-M':
        time_string = str(tt[0]) + '-' + str(tt[1]).zfill(2) + '-' + str(tt[2]).zfill(2) + '-' + str(tt[3]).zfill(2) + '-' + str(tt[4]).zfill(2)
    else:
        time_string = str(tt[0]) + '-' + str(tt[1]).zfill(2) + '-' + str(tt[2]).zfill(2) + ' ' + str(tt[3]).zfill(2) + ':' + str(tt[4]).zfill(2) + ':' + '00'
    return time_string


# get tuple time to string
def getTimeS(dateInput=None, outputFormat="%Y-%m-%d %H:%M:%S"):
    """
    dateTuple if null/False, get the current time
    """
    if not dateInput:
        now = datetime.today()
        dateInput = (now.year, now.month, now.day, now.hour, now.minute, now.second)
    elif isinstance(dateInput, datetime):
        dateInput = (dateInput.year, dateInput.month, dateInput.day, dateInput.hour, dateInput.minute, dateInput.second)
    else:
        # fill the tuple if it is not full, like (2022, 12, 30) -> (2022, 12, 30, 0, 0, 0)
        lenLeft = 7 - len(dateInput)  # 6 arguments
        for c in range(lenLeft):
            dateInput = dateInput + (0,)
    # replace and get required date string
    requiredDate = outputFormat \
        .replace('%Y', f"{dateInput[0]}".zfill(4)) \
        .replace('%m', f"{dateInput[1]}".zfill(2)) \
        .replace('%d', f"{dateInput[2]}".zfill(2)) \
        .replace('%H', f"{dateInput[3]}".zfill(2)) \
        .replace('%M', f"{dateInput[4]}".zfill(2)) \
        .replace('%S', f"{dateInput[5]}".zfill(2))
    return requiredDate


# get string time to tuple / string (change the expression)
def getTimeT(dateInput=None, inputFormat="%Y-%m-%d %H:%M:%S"):
    """
    :param dateInput: str "2022-01-25 21:52:32" / datetime
    :param inputFormat: str, eg: "%Y-%m-%d %H:%M:%S" === "YYYY-MM-DD HH:mm:ss"
    :return: tuple (2022, 1, 20, 5, 45, 50)
    """
    if not dateInput:
        _now = datetime.today()
    elif isinstance(dateInput, str):
        _now = datetime.strptime(dateInput, inputFormat)
    else:
        _now = dateInput
    requiredDate = []
    if '%Y' in inputFormat: requiredDate.append(_now.year)
    if '%m' in inputFormat: requiredDate.append(_now.month)
    if '%d' in inputFormat: requiredDate.append(_now.day)
    if '%H' in inputFormat: requiredDate.append(_now.hour)
    if '%M' in inputFormat: requiredDate.append(_now.minute)
    if '%S' in inputFormat: requiredDate.append(_now.second)
    requiredDate = tuple(requiredDate)
    return requiredDate

# split the period into different interval
def splitTimePeriod(dateT_start, dateT_end, intervalPeriod: dict, whole: bool = False):
    """
    get time period split by time interval
    :param intervalPeriod: minutes, hours, days
    :param whole: Boolean, if True, then append overall period into list
    :return: []
    """
    dt_start = datetime(*dateT_start)
    dt_end = datetime(*dateT_end)
    interval = timedelta(**intervalPeriod)

    # loop for each period
    periods = []
    period_start = dt_start
    while period_start < dt_end:
        period_end = min(period_start + interval, dt_end)
        periods.append((period_start, period_end))
        period_start = period_end
    # if include the whole period
    if whole:
        periods.append((datetime(*dateT_start), datetime(*dateT_end)))
    return periods


# get utc time with timezone
def get_utc_time_with_timezone(dateTuple: tuple, timezone: str, outputFormat=0):
    """
    :param dateTuple: tuple
    :param timezone: str
    :param outputFormat: 0: datetime, 1: tuple
    :return:
    """
    dt = datetime(dateTuple[0], dateTuple[1], dateTuple[2], hour=dateTuple[3], minute=dateTuple[4])
    utc_time = pytz.timezone(timezone).localize(dt)
    if outputFormat == 0:
        return utc_time
    elif outputFormat == 1:
        # get timezone difference
        timeDiff = int(utc_time.strftime('%z')) / 100
        dt = datetime(dateTuple[0], dateTuple[1], dateTuple[2], hour=dateTuple[3], minute=dateTuple[4]) + timedelta(hours=-timeDiff)
        return (dt.year, dt.month, dt.day, dt.hour, dt.minute)
