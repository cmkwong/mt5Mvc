import pandas as pd
import numpy as np
from models.myUtils import listModel

#
# # discard the empty field (specify the field name)
# def discardEmptyRows__DISCARD(df, mustFields):
#     """
#     mustFields: [str]
#     """
#     if len(mustFields) != 0:
#         for mustField in mustFields:
#             df = df.loc[df[mustField].notnull()]
#     return df
#
#
# # keep the rows by condition
# def keepRowsByCondition__DISCARD(df, condition: dict):
#     """
#     fields: {key: value}, that the condition being keep
#     """
#     if len(condition) != 0:
#         for field, value in condition.items():
#             df = df.loc[df[field] == value]
#     return df
#
#
# # drop last number of rows
# def dropLastRows__DISCARD(df, n):
#     df.drop(df.tail(n).index, inplace=True)
#
#
# # drop head number of rows
# def dropHeadRows__DISCARD(df, n):
#     df.drop(df.head(n).index, inplace=True)
#

# drop rows by selecting method
def dropRows(df, arg, method):
    """
    arg: str, dict, int
    method: 'last'(int) / 'head'(int) / 'condition'(dict)
    """
    if method not in ['last', 'head', 'condition']:
        raise Exception('Wrong operation.')

    if method == 'head':
        """
        arg: int, that tail rows to be discard
        """
        df = df.drop(df.head(arg).index)
    elif method == 'tail':
        """
        arg: int, that tail rows to be discard
        """
        df = df.drop(df.tail(arg).index)
    elif method == 'condition':
        """
        arg: {key: value} that if matched to be discard
        """
        for field, value in arg.items():
            df = df.loc[df[field] != value]
    return df

# combine the column
def combineCols(df, cols, separator=',', newColName=''):
    colsListType = listModel.checkType(cols)
    if len(newColName) == 0:
        newColName = '-'.join(cols)
    if colsListType == str:
        sub_df = df[cols]
    else:
        sub_df = df[df.iloc[cols]]
    df[newColName] = sub_df.apply(lambda x: separator.join(x.dropna().astype(str)), axis=1)
    return df

# change the dataframe column data type
def changeColsType(df, cols, wantedType):
    """
    :param df: dataframe
    :param cols: col name, not accept index
    :param wantedType: loader type, float, int, etc
    :return:
    """
    for col in cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df = dropRows(df, arg={col: ''}, method='condition')
        df[col] = df[col].astype(wantedType)
    return df


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


def getPreviousIndex(currentIndex, df, limitReplace=None):
    idx = np.searchsorted(df.index, currentIndex)
    if limitReplace and idx == 0:
        return limitReplace
    return df.index[max(0, idx - 1)]


def getNextIndex(currentIndex, df, limitReplace=None):
    idx = np.searchsorted(df.index, currentIndex)
    if limitReplace and idx == len(df) - 1:
        return limitReplace
    return df.index[min(idx + 1, len(df) - 1)]


def df2ListDict(df: pd.DataFrame):
    """
    :param df: pd.DataFrame
    :return: [{Data, ...}]
    """
    return df.to_dict('records')

# split the dataframe by rows
def split_df(df, percentage):
    split_index = int(len(df) * percentage)
    upper_df = df.iloc[:split_index,:]
    lower_df = df.iloc[split_index:, :]
    return upper_df, lower_df

