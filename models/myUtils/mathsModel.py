import numpy as np
from statsmodels.tsa.stattools import adfuller
import collections
import pandas as pd


def z_col(col):
    mean = np.mean(col)
    std = np.std(col)
    normalized_col = (col - mean) / std
    return normalized_col


def z_score(x):
    """
    s: np.array()
    """
    z = (x - np.mean(x)) / np.std(x)
    return z


def z_score_with_rolling_mean(spread, mean_window, std_window):
    """
    :param spread: array, shape = (total_len, )
    :param window: int
    :return: np.array
    """
    spread = spread.reshape(-1, )
    rolling_mean = pd.Series(spread.reshape(-1, )).rolling(mean_window).mean()
    rolling_std = pd.Series(spread.reshape(-1, )).rolling(std_window).std()
    z = (spread - np.array(rolling_mean)) / np.array(rolling_std)
    return z


def perform_ADF_test(x):
    """
    perform the ADF test
    """
    adf_result = collections.namedtuple("adf_result", ["test_statistic", "pvalue", "critical_values"])
    result = adfuller(x)
    adf_result.test_statistic = result[0]
    adf_result.pvalue = result[1]
    adf_result.critical_values = result[4]
    return adf_result


def averageSplit(num, times, decimalMax=2):
    """
    Split the number of part of number
    """
    # into integer
    d_num = num * 10 ** decimalMax
    # taking average of integer
    dividedValues = int(d_num / times)
    # get the residual, as int is round-down
    residual = d_num - (dividedValues * times)
    # changing into list
    splits = [dividedValues] * times
    splits[0] += residual
    return [split / 10 ** (decimalMax) for split in splits]

# values = averageSplit(5, 6, 2)
# print()