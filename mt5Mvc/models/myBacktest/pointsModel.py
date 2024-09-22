import numpy as np

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
