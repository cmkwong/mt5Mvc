from models.myBacktest import indexModel
from datetime import timedelta
import pandas as pd
import numpy as np

def get_ret_earning_list(ret_by_signal, earning_by_signal, signal):
    """
    :param new_prices: pd.DataFrame
    :param old_prices: pd.DataFrame
    :param modify_exchg_q2d: pd.DataFrame
    :param points_dff_values_df: pd.DataFrame
    :param coefficient_vector: np.array, raw vector with interception(constant value)
    :param signal: pd.Series
    :param long_mode: Boolean
    :param slsp: tuple(stop loss (negative), stop profit (positive))
    :param lot_times: lot lot_times
    :return: rets (list), earnings (list)
    """
    start_index, end_index = indexModel.get_start_end_index(signal)
    rets, earnings = [], []
    for start, end in zip(start_index, end_index):
        s, e = indexModel.get_step_index_by_index(ret_by_signal, start, step=0), indexModel.get_step_index_by_index(ret_by_signal, end, step=-1)  # why added 1, see notes (6) // Why step=0, note 87b // Why step=0,-1 (see note 96)
        ret_series, earning_series = ret_by_signal.loc[s:e], earning_by_signal.loc[s:e] # attention to use loc, note 87b
        rets.append(ret_series.prod())
        earnings.append(np.sum(earning_series))
    return rets, earnings

def get_ret_earning_by_signal(ret, earning, signal, min_ret=None, min_earning=None, slsp=None, timeframe=None):
    """
    :param ret: pd.Series
    :param earning: earning
    :param signal: pd.Series
    :param slsp: tuple(stop loss (negative), stop profit (positive))
    :return: pd.Series
    """
    ret_by_signal = pd.Series(signal.shift(1).values * ret.values, index=signal.index, name="ret_by_signal").fillna(1.0).replace({0: 1})    # shift 1 (see 95 & 96) instead of shift 2 (see 30e)
    earning_by_signal = pd.Series(signal.shift(1).values * earning.values, index=signal.index, name="earning_by_signal").fillna(0.0)  # shift 1 (see 95 & 96) instead of shift 2 (see 30e)
    if slsp != None:
        start_index, end_index = indexModel.get_start_end_index(signal, step=1, numeric=True) # step=1 (see 95 & 96)
        start_index_cal, end_index_cal = indexModel.get_start_end_index(signal, step=1) # calculate the slsp index
        for s, e, sc, ec in zip(start_index, end_index, start_index_cal, end_index_cal):
            refer_index = earning_by_signal.iloc[s:e].index
            new_ret, new_earning = modify_ret_earning_with_SLSP(min_ret.loc[sc + timedelta(minutes=1):ec], min_earning.loc[sc + timedelta(minutes=1):ec], slsp[0], slsp[1], refer_index, timeframe)
            ret_by_signal.iloc[s:e], earning_by_signal.iloc[s:e] = new_ret.values, new_earning.values
    return ret_by_signal, earning_by_signal

def get_total_ret_earning(ret_list, earning_list):
    """
    :param ret_list: return list
    :param earning_list: earning list
    :return: float, float
    """
    total_ret, total_earning = 1.0, 0.0
    for ret, earning in zip(ret_list, earning_list):
        total_ret *= ret
        total_earning += earning
    return total_ret, total_earning

def get_accum_ret_earning(ret_by_signal, earning_by_signal):
    """
    :param ret: pd.Series
    :param earning: pd.Series
    :param signal: pd.Series
    :param slsp: tuple(stop loss (negative), stop profit (positive))
    :return: accum_ret (pd.Series), accum_earning (pd.Series)
    """
    accum_ret = pd.Series(ret_by_signal.cumprod(), index=ret_by_signal.index, name="accum_ret")                 # Simplify the function note 47a
    accum_earning = pd.Series(earning_by_signal.cumsum(), index=earning_by_signal.index, name="accum_earning")  # Simplify the function note 47a
    return accum_ret, accum_earning

def _packing_datetime(masked_ret, masked_earning, refer_index):
    ret, earning = pd.Series(1.0, index=refer_index), pd.Series(0.0, index=refer_index)
    start = 0
    for ri in refer_index:
        r_buffer, e_buffer = 1.0, 0.0
        for c, fi in enumerate(masked_earning.index[start:]):
            e_buffer = e_buffer + masked_earning.loc[fi]
            r_buffer = r_buffer * masked_ret.loc[fi]
            if fi == ri:
                earning[ri] = e_buffer
                ret[ri] = r_buffer
                start += c + 1
                break
    return ret, earning

def modify_ret_earning_with_SLSP(min_ret_series, min_earning_series, sl, sp, refer_index, timeframe='1H'):
    range_mask = ((min_earning_series.cumsum() >= sl) & (min_earning_series.cumsum() <= sp)).shift(1).fillna(True).cumprod()
    masked_ret = (range_mask * min_ret_series).replace({0.0: 1.0})
    masked_earning = range_mask * min_earning_series
    resampled_masked_ret = masked_ret.resample(timeframe, closed='right', label='right').prod() # note 89a3, what is that mean of right/left
    resampled_masked_earning = masked_earning.resample(timeframe, closed='right', label='right').sum()
    if len(resampled_masked_earning.index) > len(refer_index):
        resampled_masked_ret, resampled_masked_earning = _packing_datetime(resampled_masked_ret, resampled_masked_earning, refer_index)
    return resampled_masked_ret, resampled_masked_earning
