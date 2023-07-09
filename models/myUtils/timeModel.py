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


def get_current_time_string():
    """
    :return: return time string
    """
    now = datetime.today()
    end_str = get_time_string((now.year, now.month, now.day, now.hour, now.minute))
    return end_str


# get tuple time to string
def getTimeS(dateTuple=None, outputFormat="%Y-%m-%d %H:%M:%S"):
    """
    dateTuple if null/False, get the current time
    """
    if not dateTuple:
        now = datetime.today()
        dateTuple = (now.year, now.month, now.day, now.hour, now.minute, now.second)
    else:
        # fill the tuple if it is not full, like (2022, 12, 30) -> (2022, 12, 30, 0, 0, 0)
        lenLeft = 7 - len(dateTuple)
        for c in range(lenLeft):
            dateTuple = dateTuple + (0,)
    # replace and get required date string
    requiredDate = outputFormat \
        .replace('%Y', f"{dateTuple[0]}".zfill(4)) \
        .replace('%m', f"{dateTuple[1]}".zfill(2)) \
        .replace('%d', f"{dateTuple[2]}".zfill(2)) \
        .replace('%H', f"{dateTuple[3]}".zfill(2)) \
        .replace('%M', f"{dateTuple[4]}".zfill(2)) \
        .replace('%S', f"{dateTuple[5]}".zfill(2))
    return requiredDate


# get string time to tuple / string (change the expression)
def getTimeT(dateStr, outputFormat="%Y-%m-%d %H:%M:%S", inputFormat="%Y-%m-%d %H:%M:%S", tupleNeed=False):
    """
    :param dateStr: str "2022-01-25 21:52:32"
    :param inputFormat: str, eg: "%Y-%m-%d %H:%M:%S" === "YYYY-MM-DD HH:mm:ss"
    :param outputFormat: str, '%Y-%m-%d %H:%M:%S', if False/empty, then output tuple
    :return: tuple (2022, 1, 20, 5, 45, 50)
    """
    if not dateStr:
        now = datetime.today()
    else:
        now = datetime.strptime(dateStr, inputFormat)
    if not tupleNeed:
        requiredDate = outputFormat \
            .replace('%Y', f"{now.year}".zfill(4)) \
            .replace('%m', f"{now.month}".zfill(2)) \
            .replace('%d', f"{now.day}".zfill(2)) \
            .replace('%H', f"{now.hour}".zfill(2)) \
            .replace('%M', f"{now.minute}".zfill(2)) \
            .replace('%S', f"{now.second}".zfill(2))
    else:
        requiredDate = []
        if '%Y' in outputFormat: requiredDate.append(now.year)
        if '%m' in outputFormat: requiredDate.append(now.month)
        if '%d' in outputFormat: requiredDate.append(now.day)
        if '%H' in outputFormat: requiredDate.append(now.hour)
        if '%M' in outputFormat: requiredDate.append(now.minute)
        if '%S' in outputFormat: requiredDate.append(now.second)
        requiredDate = tuple(requiredDate)
    return requiredDate

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