import pandas as pd
import numpy as np

def concatDfs(df_dict):
    """
    :param df_dict: dict
    :return: concated DataFrame
    """
    main_df = pd.DataFrame()
    for key, df in df_dict.items():
        main_df = pd.concat([main_df, df], axis=0, sort=True)
    return main_df


def getLastRow(df, pop=False):
    """
    :param df: machineInOut
    :param colName: -1/1
    :return: int, company name, year, month, day
    """
    last_row = df.tail(1)
    values = last_row.values
    last_index = last_row.index.item()  # get OUT[-1] and its date
    # if pop True
    if pop:
        df.drop(df.tail(1).index, inplace=True)
    if values.size == 1:
        values = values[0]
    return last_index, values


def df2ListDict(df: pd.DataFrame):
    """
    :param df: pd.DataFrame
    :return: [{Data, ...}]
    """
    return df.to_dict('records')


# split the dataframe by rows
def split_df(df, percentage):
    split_index = int(len(df) * percentage)
    upper_df = df.iloc[:split_index, :]
    lower_df = df.iloc[split_index:, :]
    return upper_df, lower_df

def getLevelColumnIndex(**kwargs):
    """
    :param kwargs: level_1, level_2, level_3, ...
    :return:
    """
    # check if not list, should be arise error
    for cols in kwargs.values():
        if not isinstance(cols, list):
            raise Exception(f"The values should be list but gave {type(cols)}")
    totalRows = np.prod([len(v) for v in kwargs.values()])
    # eg: 2 times: a -> a a
    duplicateTimes = []
    # eg: 2 times: a a b b c c -> a a b b c c a a b b c c
    patternTimes = []
    accumTime = 1
    for cols in reversed(kwargs.values()):
        duplicateTimes.append(accumTime)
        patternTimes.append(int(totalRows / (len(cols) * accumTime)))
        accumTime = accumTime * len(cols)
    levelArrs = []
    for i, cols in enumerate(reversed(kwargs.values())):
        l = []
        for col in cols:
            l.extend([col] * duplicateTimes[i])
        levelArrs.append(np.array(l * patternTimes[i]))
    # reverse the level
    levelArrs.reverse()
    return levelArrs