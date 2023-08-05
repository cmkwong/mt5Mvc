import pandas as pd
import numpy as np


# def get_point_diff_value(symbol, new, old, all_symbols_info):
#     digits = all_symbols_info[symbol]['digits']
#     pt_value = (new - old) * (10 ** digits) * all_symbols_info[symbol]['pt_value']
#     return pt_value


# def get_points_dff_df(close_prices, all_symbols_info, col_names=None, ptValue=True):
#     """
#     :param new_prices: pd.Dataframe with open price
#     :param all_symbols_info: tuple, Mt5f.symbols_get(). The info including the digits.
#     :param col_names: list, set None to use the symbols as column names. Otherwise, rename as fake column name
#     :param ptValue: Boolean, if True, return in point value, eg: 1 pt of USDJPY = 100 YEN
#     :return: points_dff_values_df, new pd.Dataframe
#     take the difference from open price
#     """
#     if isinstance(close_prices, pd.Series):
#         close_prices = pd.DataFrame(close_prices, index=close_prices.index)  # avoid the error of "too many index" if len(symbols) = 1
#     # default ptValue is 1
#     pt_value = 1
#     # getting the symbols
#     symbols = close_prices.columns
#     # define the prices
#     new_prices = close_prices
#     old_prices = close_prices.shift(periods=1)
#     points_dff_values_df = pd.DataFrame(index=new_prices.index)
#     for c, symbol in enumerate(symbols):
#         digits = all_symbols_info[symbol]['digits']  # (note 44b)
#         # if need to change the point value
#         if ptValue:
#             pt_value = all_symbols_info[symbol]['pt_value']
#         points_dff_values_df[symbol] = (new_prices.iloc[:, c] - old_prices.iloc[:, c]) * (10 ** digits) * pt_value
#     if col_names != None:
#         points_dff_values_df.columns = col_names
#     elif col_names == None:
#         points_dff_values_df.columns = symbols
#     return points_dff_values_df


def get_points_dff_values_arr(symbols, news, olds, all_symbols_info):
    """
    :param symbols: list
    :param news: np.array
    :param olds: np.array
    :param all_symbols_info: nametuple object
    :return: np.array
    """
    if isinstance(symbols, str): symbols = [symbols]

    pt_values = np.zeros((len(symbols),))
    for i, (symbol, new, old) in enumerate(zip(symbols, news, olds)):
        digits = all_symbols_info[symbol]['digits']
        pt_values[i] = (new - old) * (10 ** digits) * all_symbols_info[symbol]['pt_value']
    return pt_values


# def get_point_diff_from_results(results, requests, expected_prices, all_symbol_info):
#     """
#     :param results: [result]
#     :param requests: [request], request is dict
#     :param expected_prices: np.array
#     :param all_symbol_info: dict for required symbol
#     :return:
#     """
#     pt_diff_arr = []
#     for result, request, expected_price in zip(results, requests, expected_prices):
#         symbol = request['symbol']
#         digits = all_symbol_info[symbol].digits
#         if request['type'] == mt5.ORDER_TYPE_BUY:
#             pt_diff_arr.append((result.price - expected_price) * (10 ** digits))
#         elif request['type'] == mt5.ORDER_TYPE_SELL:
#             pt_diff_arr.append((expected_price - result.price) * (10 ** digits))
#     return np.array(pt_diff_arr)
