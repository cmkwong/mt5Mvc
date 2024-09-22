import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
import collections

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


def generate_ar_process(lags: int, coefs: list, length: int):
    """
    generate the sample of AR(l)
    :param lags: argument of AR(lags)
    :param coefs: [float]
    :param length: the data of length
    :return:
    """
    # cast coefs to np array
    coefs = np.array(coefs)

    # initial values
    series = [np.random.normal() for _ in range(lags)]

    for _ in range(length):
        # get previous values of the series, reversed
        prev_vals = series[-lags:][::-1]

        # get new value of time series
        new_val = np.sum(np.array(prev_vals) * coefs) + np.random.normal()

        series.append(new_val)

    return pd.Series(series)

# -----------------generate_ar_process()---------------------
# for i in range(10):
#     plotController = PlotController()
#     axs = plotController.getAxes(figsize=(15, 9), dpi=200)
#     series = generate_ar_process(2, [0.73, 0.65], 100)
#     plotController.plotSimpleLine(axs[0], series)
#     plotController.saveImg(filename=f'{i}-ar-test.jpg')
# print()
# --------------------------------------