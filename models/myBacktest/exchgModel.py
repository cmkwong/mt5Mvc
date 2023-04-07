from models.myBacktest import signalModel, indexModel
import pandas as pd
from datetime import timedelta

def get_resoluted_exchg(exchg, signal, index):
    """
    :param exchg: pd.DataFrame
    :param signal: pd.Series
    :param index: pd.DateTimeIndex / str in time format
    :return:
    """
    # resume to datetime index
    signal.index = pd.to_datetime(signal.index)
    exchg.index = pd.to_datetime(exchg.index)
    index = pd.to_datetime(index)

    # get int signal and its start_indexes and end_indexes
    int_signal = signalModel.get_int_signal(signal)
    start_indexes = indexModel.find_target_index(int_signal, target=1, step=0)
    end_indexes = indexModel.find_target_index(int_signal, target=-1, step=0)

    # init the empty signal series
    resoluted_exchg = pd.DataFrame(1.0, columns=exchg.columns, index=index)
    for s, e in zip(start_indexes, end_indexes):
        e = e + timedelta(minutes=-1) # note 82e, use the timedelta instead of shift()
        resoluted_exchg.loc[s:e,:] = exchg.loc[s,:].values
    return resoluted_exchg

def get_exchg_by_signal(exchg, signal):
    """
    note 79a
    :param exchg: pd.DataFrame, eg, 1min TimeFrame
    :param signal: pd.Series, original signal, eg: 1H TimeFrame
    :return:
    """
    new_exchg = exchg.copy()
    start_index, end_index = indexModel.get_start_end_index(signal, step=1) # step 1 instead of step 2 because change to use close price as action price (see note 95a)
    for s, e in zip(start_index, end_index):
        new_exchg.loc[s:e + timedelta(minutes=-1), :] = exchg.loc[s:e + timedelta(minutes=-1), :].iloc[0].values    # there is a problem to using shift(), note 89c
        # new_exchg.loc[s:e + timedelta(minutes=-1),:] = exchg.loc[s:s,:].values
    # new_exchg = new_exchg.shift(shift_offset[0], freq=shift_offset[1])
    return new_exchg

def modify_exchange_rate(symbols, exchange_symbols, exchange_rate_df, deposit_currency, exchg_type):
    """
    :param symbols:             ['AUDJPY', 'AUDUSD', 'CADJPY', 'EURUSD', 'NZDUSD', 'USDCAD']
    :param exchange_symbols:    ['USDJPY', 'AUDUSD', 'USDJPY', 'EURUSD', 'NZDUSD', 'USDCAD'] all is related to deposit currency
    :param exchange_rate_df: pd.DataFrame, the price from excahnge_symbols (only open prices)
    :param deposit_currency: "USD" / "GBP" / "EUR"
    :param exchg_type: str, 'q2d' = quote to deposit OR 'b2d' = base to deposit
    :return: pd.DataFrame with cols name: ['JPYUSD', 'USD', 'JPYUSD', 'USD', 'USD', 'CADUSD']
    """
    symbol_new_names = []
    for i, exch_symbol in enumerate(exchange_symbols):
        base, quote = exch_symbol[:3], exch_symbol[3:]
        if exchg_type == 'q2d': # see note 38a
            if quote == deposit_currency:
                if symbols[i] != exch_symbol:
                    symbol_new_names.append("{}to{}".format(exch_symbol[:3], exch_symbol[3:]))
                elif symbols[i] == exch_symbol:
                    symbol_new_names.append(deposit_currency)
                    exchange_rate_df.iloc[:, i] = 1.0
            elif base == deposit_currency:
                symbol_new_names.append("{}to{}".format(exch_symbol[3:], exch_symbol[:3]))
                exchange_rate_df.iloc[:, i] = 1 / exchange_rate_df.iloc[:, i].values
        elif exchg_type == 'b2d':
            if base == deposit_currency:
                if symbols[i] != exch_symbol:
                    symbol_new_names.append("{}to{}".format(exch_symbol[3:], exch_symbol[:3]))
                    exchange_rate_df.iloc[:, i] = 1 / exchange_rate_df.iloc[:, i].values
                elif symbols[i] == exch_symbol:
                    symbol_new_names.append(deposit_currency)
                    exchange_rate_df.iloc[:, i] = 1.0
            elif quote == deposit_currency:
                symbol_new_names.append("{}to{}".format(exch_symbol[:3], exch_symbol[3:]))

    return exchange_rate_df, symbol_new_names

def get_exchange_df(symbols, exchg_symbols, exchange_rate_df, deposit_currency, exchg_type, col_names=None):
    """
    note 86d
    :param symbols: [str]
    :param exchg_symbols: [str]
    :param exchange_rate_df: pd.DataFrame
    :param deposit_currency: str
    :param exchg_type: str, q2d/b2d
    :param col_names: [str], column names going to assign on dataframe
    :return:
    """
    exchange_rate_df, modified_names = modify_exchange_rate(symbols, exchg_symbols, exchange_rate_df, deposit_currency, exchg_type=exchg_type)
    if col_names == None:
        exchange_rate_df.columns = modified_names
    else:
        exchange_rate_df.columns = col_names  # assign temp name
    return exchange_rate_df