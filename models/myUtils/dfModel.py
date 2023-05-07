import pandas as pd
import numpy as np
import os

from models.myUtils import listModel

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
    upper_df = df.iloc[:split_index,:]
    lower_df = df.iloc[split_index:, :]
    return upper_df, lower_df

